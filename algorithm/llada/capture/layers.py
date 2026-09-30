"""Read-only observation of deployed Python/CUDA LLaDA layers.

The Python return profiler copies operands during inference. Use a
single-threaded inference process without another Python profiler.
"""

from __future__ import annotations
import inspect
import sys
from pathlib import Path
import torch

LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")
CACHE_NAMES = ("key_codes", "key_scale", "value_codes")


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for (key, item) in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)((clone_tree(item) for item in value))
    return value


def raw_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(
        left.detach().cpu().contiguous().view(torch.uint8),
        right.detach().cpu().contiguous().view(torch.uint8),
    )


class CaptureComplete(Exception):
    """Selected forward has completed; no further generation is required."""


class LayerObserver:
    def __init__(self, block, inputs=None):
        self.block = block
        self.inputs = {} if inputs is None else inputs
        self.expected = {}
        self.handles = []
        self.linear = None
        self.residuals = 0
        self.profiler_started = False
        self.source = str(Path(inspect.getfile(type(block))).resolve().parents[1])
        self.callbacks = []
        self.clipping_configuration = None
        self.observation_status = {}

    def install_callback(self, module, name, callback):
        if getattr(module, name, None) is not None:
            raise RuntimeError(f"another observer already uses {name}")
        previous = getattr(module, name, None)
        existed = hasattr(module, name)
        self.callbacks.append((module, name, callback, existed, previous))
        setattr(module, name, callback)

    def save(self, name, value):
        if name in self.expected:
            raise RuntimeError(f"duplicate numerical checkpoint: {name}")
        self.expected[name] = clone_tree(value)

    def profile(self, frame, event, result):
        if event != "return" or not frame.f_code.co_filename.startswith(
            self.source + "/"
        ):
            return
        (name, local) = (frame.f_code.co_name, frame.f_locals)
        if name in {"accelerated_linear_bf16", "accelerated_mixed_a4a8_linear_bf16"}:
            if self.linear is None:
                raise RuntimeError("Linear return outside selected module")
            rows = local["rows"]
            self.inputs["weight." + self.linear + ".codes"] = clone_tree(
                local["weight_codes"]
            )
            self.inputs["weight." + self.linear + ".scale"] = clone_tree(
                local["weight_scales_bf16"]
            )
            for suffix, value in (
                ("activation_codes", local["codes"][:rows]),
                ("activation_scale", local["activation_scales"]),
                ("accumulator", local["sums"]),
            ):
                self.save(self.linear + "." + suffix, value)
        elif name == "score_override":
            for label, variable in (
                ("query_rope", "q"),
                ("query_codes", "q_codes"),
                ("query_scale", "q_scales"),
                ("qk_key_codes", "k_codes"),
                ("qk_key_scale", "k_scales"),
                ("qk_accumulator", "sums"),
            ):
                self.save(label, local[variable])
            self.save("scores", result)
        elif name == "probability_override":
            self.save("probabilities", result)
        elif name == "context_override":
            for label, variable in (
                ("probability_codes", "p_codes"),
                ("probability_scale", "p_scales"),
                ("pv_value_codes", "v_codes"),
                ("pv_accumulator", "sums"),
            ):
                self.save(label, local[variable])
            self.save("context", result)
        elif name == "encode":
            codec = type(local.get("self")).__name__
            if codec == "SpinQuantK8CacheCodec":
                self.save("key_rope", local["values"])
                self.save("new_key_codes", result[0])
                self.save("new_key_scale", result[1])
            elif codec == "SpinQuantV8CacheCodec":
                self.save("new_value_codes", result)
        elif name == "_silu_multiply":
            self.save("silu", local["activated"])
            self.save("product_before_r4", result)
        elif name == "_residual_add":
            self.save(
                "attention_residual" if self.residuals == 0 else "block_output", result
            )
            self.residuals += 1
        elif name == "softmax_lut_bf16":
            self.inputs["softmax_exp_lut"] = clone_tree(local["lut"])
            self.inputs["softmax_reciprocal_lut"] = clone_tree(local["reciprocal_lut"])
        elif name == "rope_apply_bf16_cuda":
            for label in ("sin", "cos"):
                key = "rope_selected_" + label
                value = clone_tree(local[label])
                if key in self.inputs and (not raw_equal(self.inputs[key], value)):
                    raise RuntimeError("Q and K used different RoPE table rows")
                self.inputs[key] = value

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError("real layer capture requires no other Python profiler")
        try:
            return self.start()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def start(self):
        numerical = sys.modules[type(self.block.q_proj).__module__]
        describe = getattr(numerical, "describe_target_block_a4_clipping", None)
        if describe is None:
            raise ValueError("numerical source does not expose clipping configuration")
        self.clipping_configuration = describe(self.block)
        for name, config in self.clipping_configuration.items():
            module = getattr(self.block, name)
            self.inputs[name + ".clip.installed"] = torch.tensor(
                config["installed"]
            )
            self.observation_status[name] = (
                "incomplete" if config["installed"] else "not_applicable"
            )
            if config["installed"]:
                self.inputs[name + ".clip.ratio"] = clone_tree(config["ratio_bf16"])

                def capture_clip(values, label=name):
                    for field, value in values.items():
                        self.save(label + ".clip." + field, value)
                    self.observation_status[label] = "complete"

                self.install_callback(
                    module, "_target_a4_clip_observer", capture_clip
                )
        if (
            getattr(self.block, "_target_numeric_r4", None)
            != "h12288-bf16-staged/v1"
        ):
            raise ValueError("capture requires R4 format h12288-bf16-staged/v1")
        self.install_callback(
            self.block.ff_out,
            "_target_r4_observer",
            lambda value: self.save("ff_out.r4_output", value),
        )
        for name in LINEARS:
            module = getattr(self.block, name)
            if module.weight_bits != 4 or module.weight_group_size != -1:
                raise ValueError("layer capture requires G-1 W4 weights")

            def before(module, args, label=name):
                self.linear = label
                self.save(label + ".input", args[0])
                clip_output = self.expected.get(label + ".clip.output")
                if clip_output is not None and (not raw_equal(clip_output, args[0])):
                    raise RuntimeError("Linear input changed after clipping")
                if label == "ff_out" and label + ".clip.input" in self.expected:
                    if not raw_equal(
                        self.expected["ff_out.r4_output"],
                        self.expected[label + ".clip.input"],
                    ):
                        raise RuntimeError("Down clipping input differs from R4 output")
                rows = args[0].numel() // module.in_features
                bits = module.row_precision_context.require(
                    rows, args[0].device
                )
                self.inputs[label + ".row_bits"] = clone_tree(bits)

            def after(module, args, result, label=name):
                self.save(label + ".output", result)
                if label + ".accumulator" not in self.expected:
                    raise RuntimeError(
                        "selected execution did not use the supported CUDA G-1 Linear"
                    )
                self.linear = None

            before._target_numeric_read_only = True
            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after))
        for name in ("attn_norm", "ff_norm"):
            norm = getattr(self.block, name)
            self.inputs[name + ".weight"] = clone_tree(norm.weight)
            if norm.bias is not None or not torch.all(norm.weight == 1):
                raise ValueError(
                    "layer capture requires unity RMSNorm weights and no bias"
                )
            self.handles.append(
                norm.register_forward_hook(
                    lambda module, args, output, label=name: self.save(label, output)
                )
            )
        sys.setprofile(self.profile)
        self.profiler_started = True
        return self

    def __exit__(self, *exc):
        if self.profiler_started:
            sys.setprofile(None)
            self.profiler_started = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module, name, callback, existed, previous in reversed(self.callbacks):
            if getattr(module, name, None) is callback:
                if existed:
                    setattr(module, name, previous)
                else:
                    delattr(module, name)
        self.callbacks.clear()

    def numerical_observations(self):
        """Describe actual observations without inferring configuration from missing tensors."""
        return clone_tree(
            {
                "clipping": self.clipping_configuration,
                "clipping_observations": self.observation_status,
                "r4_output_observed": "ff_out.r4_output" in self.expected,
                "encoding": {
                    name: "float32_raw"
                    if name.endswith((".clip.row_max", ".clip.limit_fp32"))
                    else "bf16_raw"
                    for name in self.expected
                    if ".clip." in name or name == "ff_out.r4_output"
                },
            }
        )


def replay_block(
    config, inputs, *, device, layer_id=0
):
    """Instantiate the quantized block from captured weight codes and scales."""
    from model.modeling_llada import BufferCache, LLaDALlamaBlock, ModelConfig
    from quantization.model import (
        SpinQuantW4A8Linear,
        SpinQuantV8CacheCodec,
        _install_target_r4,
        _install_native_attention_numeric,
        _install_target_numeric_rms_norm,
        install_target_block_a4_clipping,
    )
    from quantization.numeric import SpinQuantW4Tensor
    from numerics.linear_kernels import LinearNumericWorkspace
    from numerics.operator_kernels import RMSNormNumericWorkspace
    from numerics.int8_matmul import Int8MatmulWorkspace
    from numerics.precision import RowPrecisionContext
    from torch import nn

    config = dict(config, init_device="cpu")
    block = LLaDALlamaBlock(layer_id, ModelConfig(**config), BufferCache()).eval()
    context = RowPrecisionContext()
    context.activate(inputs["row_bits"].to(device))
    for name in LINEARS:
        bits = inputs[name + ".row_bits"]
        if not raw_equal(bits, inputs["row_bits"]):
            raise ValueError(
                "replay requires matching row precision across the layer Linears"
            )
        linear = SpinQuantW4A8Linear(
            SpinQuantW4Tensor(
                inputs["weight." + name + ".codes"].to(device),
                inputs["weight." + name + ".scale"].to(device),
            ),
            LinearNumericWorkspace(),
            context,
            module_name=name,
        )
        setattr(block, name, linear)
    for name in ("attn_norm", "ff_norm"):
        norm = getattr(block, name)
        norm.weight = nn.Parameter(
            inputs[name + ".weight"].to(device), requires_grad=False
        )
        _install_target_numeric_rms_norm(norm, RMSNormNumericWorkspace())
    _install_target_r4(block)
    if any(name + ".clip.installed" not in inputs for name in LINEARS):
        raise ValueError("replay requires clipping installation state for all seven Linears")
    ratios = {}
    for name in LINEARS:
        installed = inputs[name + ".clip.installed"]
        if installed.dtype != torch.bool or installed.numel() != 1:
            raise ValueError("clipping installed must be a scalar bool")
        ratio = inputs.get(name + ".clip.ratio")
        if bool(installed):
            if ratio is None or ratio.dtype != torch.bfloat16 or ratio.numel() != 1:
                raise ValueError("installed clipping requires a scalar BF16 ratio")
            ratios[name] = float(ratio)
        else:
            if ratio is not None:
                raise ValueError("uninstalled clipping must not carry an active ratio")
            ratios[name] = None
    install_target_block_a4_clipping(block, ratios)
    block._target_numeric_bf16_residual = True
    block._target_numeric_silu_pwl16 = True
    block._target_numeric_silu_coefficients = tuple(
        (
            tuple(inputs["silu_coefficients." + str(i)].float().tolist())
            for i in range(2)
        )
    )
    _install_native_attention_numeric(
        block,
        workspace=Int8MatmulWorkspace(),
        qk_int8=True,
        softmax_lut=True,
        probability_p8=False,
        k8_cache=True,
        v8_codec=SpinQuantV8CacheCodec(inputs["v_scale"].to(device)),
        rope_before_k8=True,
    )
    block.to(device)
    block.rotary_emb._RotaryEmbedding__cache["rope_pos_sin"] = inputs["rope_sine"].to(
        device
    )
    block.rotary_emb._RotaryEmbedding__cache["rope_pos_cos"] = inputs["rope_cosine"].to(
        device
    )
    from numerics.bf16 import _SOFTMAX_LUT_CACHE

    dev = torch.device(device)
    if dev.index is None and dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    _SOFTMAX_LUT_CACHE[dev.type, dev.index] = (
        inputs["softmax_exp_lut"].to(device),
        inputs["softmax_reciprocal_lut"].to(device),
    )
    return block
