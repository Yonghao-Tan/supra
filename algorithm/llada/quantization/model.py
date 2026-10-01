"""Install artifact weights, per-row precision, cache codecs and numeric operators."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
from types import MethodType
from typing import Any
import torch
import torch.nn as nn
from numerics.candidate import _exp_lut as _candidate_exp_lut
from numerics.candidate import _reciprocal_lut as _candidate_reciprocal_lut
from numerics.bf16 import (
    bf16,
    quantize_per_row_bf16,
    rms_norm_bf16,
    rope_apply_bf16,
    softmax_lut_bf16,
)
from numerics.int8_matmul import Int8MatmulWorkspace, int8_batched_matmul
from numerics.linear_kernels import (
    LinearNumericWorkspace,
    accelerated_linear_bf16,
    accelerated_mixed_a4a8_linear_bf16,
    triton_linear_numeric_available,
)
from numerics.operator_kernels import (
    RMSNormNumericWorkspace,
    rope_apply_bf16_cuda,
    rms_norm_bf16_cuda,
    triton_operator_numeric_available,
)
from numerics.precision import (
    RowPrecisionContext,
    clip_a4_rows_bf16,
)
from quantization.artifact import (
    EMBEDDING,
    LM_HEAD,
    ROTATED_VARIANTS,
    SpinQuantArtifactReader,
    expected_w4_ids,
)
from quantization.numeric import SpinQuantW4Tensor
from quantization.rotation import (
    structured_hadamard_12288_bf16,
)
from quantization.numeric import SpinQuantW8Tensor
from numerics.bf16 import (
    quantize_activation_per_row_bf16,
    quantize_activation_per_row_bits_bf16,
)

TARGET_NUMERIC_CAPTURE_GRAPH = (
    "n4-gminus1-bf16-rms-rope-pwl16-swiglu-residual-h12288/v1"
)


def _parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    (parent_name, child_name) = module_name.rsplit(".", 1)
    return (root.get_submodule(parent_name), child_name)


def _module_device(module: nn.Module) -> torch.device:
    """Resolve a module device after either parameter or buffer replacement."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is None:
        raise ValueError(
            f"cannot determine device for parameterless module {type(module).__name__}"
        )
    return buffer.device


class SpinQuantW4A8Linear(nn.Module):
    def __init__(
        self,
        weight: SpinQuantW4Tensor | SpinQuantW8Tensor,
        workspace: LinearNumericWorkspace,
        row_precision_context: RowPrecisionContext | None = None,
        activation_bits: int = 8,
        module_name: str = "",
        weight_bits: int = 4,
    ) -> None:
        super().__init__()
        if weight.codes.ndim != 2 or weight.codes.dtype != torch.int8:
            raise ValueError("invalid SpinQuant W4 code tensor")
        if weight.scale_bf16.ndim != 1:
            raise ValueError("invalid SpinQuant weight scale tensor")
        if weight_bits not in {4, 8}:
            raise ValueError("weight_bits must be 4 or 8")
        (code_min, code_max) = (-8, 7) if weight_bits == 4 else (-127, 127)
        if int(weight.codes.min()) < code_min or int(weight.codes.max()) > code_max:
            raise ValueError(
                f"SpinQuant W{weight_bits} code outside [{code_min},{code_max}]"
            )
        if activation_bits not in {4, 8}:
            raise ValueError("activation_bits must be 4 or 8")
        self.in_features = int(weight.codes.shape[1])
        self.out_features = int(weight.codes.shape[0])
        if weight.scale_bf16.shape != weight.codes.shape[:1]:
            raise ValueError("invalid SpinQuant per-row scale tensor")
        self.weight_group_size = -1
        self.register_buffer("weight_codes", weight.codes.contiguous())
        self.register_buffer("weight_scales_bf16", weight.scale_bf16.contiguous())
        self.workspace = workspace
        self.row_precision_context = row_precision_context
        self.activation_bits = activation_bits
        self.module_name = module_name
        self.weight_bits = weight_bits

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        weight_codes = self.weight_codes
        weight_scales_bf16 = self.weight_scales_bf16
        if values.shape[-1] != self.in_features:
            raise ValueError(
                f"expected K={self.in_features}, got {tuple(values.shape)}"
            )
        if (
            weight_codes.dtype != torch.int8
            or weight_codes.device != values.device
        ):
            raise ValueError("weight codes must use the input device and INT8 format")
        if (
            weight_scales_bf16.device != values.device
            or (not bool(torch.isfinite(weight_scales_bf16).all()))
            or (not bool((weight_scales_bf16 > 0).all()))
            or (
                not torch.equal(
                    weight_scales_bf16,
                    weight_scales_bf16.to(torch.bfloat16).to(weight_scales_bf16.dtype),
                )
            )
        ):
            raise ValueError(
                "weight scales must be finite positive BF16-materialized values on the input device"
            )
        rows = values.reshape(-1, self.in_features)
        row_bits = (
            None
            if self.row_precision_context is None
            else self.row_precision_context.require(
                int(rows.shape[0]), rows.device
            )
        )
        if rows.is_cuda and triton_linear_numeric_available():
            output = (
                accelerated_linear_bf16(
                    rows,
                    weight_codes,
                    weight_scales_bf16,
                    self.workspace,
                    activation_bits=self.activation_bits,
                )
                if row_bits is None
                else accelerated_mixed_a4a8_linear_bf16(
                    rows, row_bits, weight_codes, weight_scales_bf16, self.workspace
                )
            )
        else:
            activation = (
                quantize_activation_per_row_bf16(rows, self.activation_bits)
                if row_bits is None
                else quantize_activation_per_row_bits_bf16(rows, row_bits)
            )
            if rows.is_cuda:
                activation_codes = activation.codes.contiguous()
                if activation_codes.shape[0] <= 16:
                    activation_codes = torch.cat(
                        (
                            activation_codes,
                            torch.zeros(
                                17 - activation_codes.shape[0],
                                activation_codes.shape[1],
                                dtype=torch.int8,
                                device=activation_codes.device,
                            ),
                        ),
                        dim=0,
                    )
                sums = torch._int_mm(
                    activation_codes, weight_codes.transpose(0, 1).contiguous()
                )[: rows.shape[0]]
            else:
                sums = torch.matmul(
                    activation.codes.to(torch.int32),
                    weight_codes.to(torch.int32).transpose(0, 1),
                )
            output = bf16(
                bf16(sums.to(torch.float32) * activation.scale_bf16.unsqueeze(1))
                * weight_scales_bf16.unsqueeze(0)
            ).to(torch.bfloat16)
        return output.reshape(*values.shape[:-1], self.out_features)


class SpinQuantK8CacheCodec:
    """Per-token K8 cache with BF16-materialized scales and BF16 V left untouched."""

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 4:
            raise ValueError(f"K cache must be [B,H,T,D], got {tuple(values.shape)}")
        head_dim = int(values.shape[-1])
        quantized = quantize_per_row_bf16(values.reshape(-1, head_dim))
        return (
            quantized.codes.reshape_as(values).contiguous(),
            quantized.scale_bf16.reshape(*values.shape[:-1], 1).contiguous(),
        )

    def decode(self, codes: torch.Tensor, scale_bf16: torch.Tensor) -> torch.Tensor:
        if codes.ndim != 4 or codes.dtype != torch.int8:
            raise ValueError("K cache codes must be rank-4 INT8")
        if scale_bf16.shape != (*codes.shape[:-1], 1):
            raise ValueError("K cache scale must contain one value per token and head")
        exact = bf16(codes.to(torch.float32) * scale_bf16).to(torch.bfloat16)
        return exact


class SpinQuantV8CacheCodec:
    """Static per-KV-head V8 cache codec for one transformer block."""

    def __init__(self, scale_bf16: torch.Tensor) -> None:
        if scale_bf16.ndim != 1 or scale_bf16.numel() <= 0:
            raise ValueError("V8 scale must contain one value per KV head")
        scale = scale_bf16.detach().to(torch.float32).contiguous()
        if not bool(torch.all(torch.isfinite(scale))) or bool(torch.any(scale <= 0)):
            raise ValueError("V8 scales must be finite and positive")
        self.scale_bf16 = bf16(scale)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != self.scale_bf16.numel():
            raise ValueError("V cache must be [B, num_kv_heads, T, D]")
        scale = self.scale_bf16.to(values.device).view(1, -1, 1, 1)
        return torch.clamp(torch.round(values.float() / scale), -127, 127).to(
            torch.int8
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if (
            codes.ndim != 4
            or codes.dtype != torch.int8
            or codes.shape[1] != self.scale_bf16.numel()
        ):
            raise ValueError(
                "V cache codes must be rank-4 INT8 with the calibrated KV-head axis"
            )
        scale = self.scale_bf16.to(codes.device).view(1, -1, 1, 1)
        return bf16(codes.float() * scale).to(torch.bfloat16)

    def expanded_scales(self, query_heads: int, device: torch.device) -> torch.Tensor:
        kv_heads = int(self.scale_bf16.numel())
        if query_heads % kv_heads:
            raise ValueError(
                "query head count must be divisible by calibrated KV heads"
            )
        return self.scale_bf16.to(device).repeat_interleave(query_heads // kv_heads)


TARGET_BLOCK_LINEARS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_out",
    "ff_proj",
    "up_proj",
    "ff_out",
)
TARGET_A4_CLIPPING_RULE = "a4-row-clip-bf16/v1"


def _block_clipping_arguments(block, ratios_by_linear):
    if set(ratios_by_linear) != set(TARGET_BLOCK_LINEARS):
        raise ValueError(
            "clipping requires exactly the seven Transformer Linear ratios"
        )
    normalized = {}
    for name in TARGET_BLOCK_LINEARS:
        module = getattr(block, name, None)
        if (
            not isinstance(module, SpinQuantW4A8Linear)
            or module.row_precision_context is None
        ):
            raise ValueError(f"clipping requires a quantized {name} with row precision")
        if hasattr(module, "_target_a4_clip_ratio"):
            raise ValueError(f"A4 clipping is already installed on {name}")
        value = ratios_by_linear[name]
        if value is not None:
            value = float(torch.tensor(float(value), dtype=torch.bfloat16))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("A4 clipping ratio must be finite and in (0,1]")
        normalized[name] = value
    if normalized["ff_out"] is not None:
        handle = getattr(block.ff_out, "_target_r4_hook_handle", None)
        if (
            getattr(block, "_target_numeric_r4", None) != "h12288-bf16-staged/v1"
            or handle is None
            or handle.id not in block.ff_out._forward_pre_hooks
        ):
            raise ValueError(
                "Down clipping requires the installed target R4 hook first"
            )
    return normalized


def _target_a4_clip_input(module, inputs):
    if len(inputs) != 1:
        raise ValueError("A4 clipping expects one activation input")
    values = inputs[0]
    bits = module.row_precision_context.require(
        values.numel() // values.shape[-1],
        values.device,
    )
    return (
        clip_a4_rows_bf16(
            values,
            bits,
            module._target_a4_clip_ratio,
            observer=getattr(module, "_target_a4_clip_observer", None),
        ),
    )


@contextmanager
def using_target_block_a4_clip_ratio(block: nn.Module, ratio: float):
    """Override installed scalar clipping without changing the hook order."""
    if ratio == 0:
        yield
        return
    ratio = float(torch.tensor(float(ratio), dtype=torch.bfloat16))
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("temporary A4 clipping ratio must be finite and in (0,1]")
    modules = [getattr(block, name) for name in TARGET_BLOCK_LINEARS]
    if any(not hasattr(module, "_target_a4_clip_ratio") for module in modules):
        raise ValueError("temporary clipping requires all seven installed clipping hooks")
    previous = [module._target_a4_clip_ratio for module in modules]
    try:
        for module in modules:
            module._target_a4_clip_ratio = ratio
        yield
    finally:
        for module, value in zip(modules, previous):
            module._target_a4_clip_ratio = value


def install_target_block_a4_clipping(block: nn.Module, ratios_by_linear: dict) -> None:
    """Install the deployed clipping operation after an existing target R4."""
    ratios = _block_clipping_arguments(block, ratios_by_linear)
    installed = []
    try:
        for name, ratio in ratios.items():
            if ratio is None:
                continue
            module = getattr(block, name)
            handle = module.register_forward_pre_hook(_target_a4_clip_input)
            installed.append((module, handle))
            module._target_a4_clip_ratio = ratio
            module._target_a4_clip_hook_handle = handle
    except BaseException:
        for module, handle in installed:
            handle.remove()
            for attr in ("_target_a4_clip_ratio", "_target_a4_clip_hook_handle"):
                if hasattr(module, attr):
                    delattr(module, attr)
        raise


def describe_target_block_a4_clipping(block: nn.Module) -> dict:
    """Describe actual hooks; reject unrepresented activation transforms."""
    result = {}
    for name in TARGET_BLOCK_LINEARS:
        module = getattr(block, name)
        ratio = getattr(module, "_target_a4_clip_ratio", None)
        clip = getattr(module, "_target_a4_clip_hook_handle", None)
        r4 = getattr(module, "_target_r4_hook_handle", None)
        hooks = module._forward_pre_hooks
        if ratio is not None and (
            clip is None or hooks.get(clip.id) is not _target_a4_clip_input
        ):
            raise ValueError(f"unsupported clipping hook on {name}")
        if ratio is None and clip is not None:
            raise ValueError(f"clipping ratio is missing on {name}")
        known = {h.id for h in (clip, r4) if h is not None}
        if any(
            (
                key not in known
                and (not getattr(hook, "_target_numeric_read_only", False))
                for (key, hook) in hooks.items()
            )
        ):
            raise ValueError(f"unsupported activation pre-hook on {name}")
        if name == "ff_out" and clip is not None:
            keys = list(hooks)
            if (
                r4 is None
                or r4.id not in keys
                or keys.index(r4.id) >= keys.index(clip.id)
                or (
                    getattr(block, "_target_numeric_r4", None)
                    != "h12288-bf16-staged/v1"
                )
            ):
                raise ValueError("Down clipping must follow target R4")
        result[name] = dict(
            installed=clip is not None,
            ratio_bf16=None
            if ratio is None
            else torch.tensor(ratio, dtype=torch.bfloat16),
            rule=TARGET_A4_CLIPPING_RULE,
            observation_supported=True,
        )
    return result


def install_target_a4_clipping(
    model: nn.Module, ratio: float, *, output_ratio: float | None = None
) -> float:
    ratio = float(torch.tensor(float(ratio), dtype=torch.bfloat16))
    output_ratio = (
        ratio
        if output_ratio is None
        else float(torch.tensor(float(output_ratio), dtype=torch.bfloat16))
    )
    if not 0 < ratio <= 1 or not 0 < output_ratio <= 1:
        raise ValueError("A4 clipping ratio must be finite and in (0,1]")
    modules = [
        module
        for (name, module) in model.named_modules()
        if ".blocks." in name and isinstance(module, SpinQuantW4A8Linear)
    ]
    if len(modules) != 224 or any(
        (module.row_precision_context is None for module in modules)
    ):
        raise ValueError(
            "A4 clipping requires all 224 Transformer Linears with row precision"
        )
    if any((hasattr(module, "_target_a4_clip_ratio") for module in modules)):
        raise ValueError("A4 clipping is already installed")
    outputs = [
        module
        for module in modules
        if module.module_name.rsplit(".", 1)[-1] in ("attn_out", "ff_out")
    ]
    if output_ratio != ratio and len(outputs) != 64:
        raise ValueError(
            "output clipping requires 32 Attention output and 32 Down projections"
        )
    blocks = [
        module
        for module in model.modules()
        if all(
            (
                isinstance(getattr(module, name, None), SpinQuantW4A8Linear)
                for name in TARGET_BLOCK_LINEARS
            )
        )
    ]
    if len(blocks) != 32:
        raise ValueError("A4 clipping requires 32 complete Transformer Blocks")
    ratios = {
        name: output_ratio if name in ("attn_out", "ff_out") else ratio
        for name in TARGET_BLOCK_LINEARS
    }
    for block in blocks:
        _block_clipping_arguments(block, ratios)
    installed = []
    try:
        for block in blocks:
            install_target_block_a4_clipping(block, ratios)
            installed.append(block)
    except BaseException:
        for block in installed:
            for name in TARGET_BLOCK_LINEARS:
                module = getattr(block, name)
                module._target_a4_clip_hook_handle.remove()
                del module._target_a4_clip_hook_handle, module._target_a4_clip_ratio
        raise
    return ratio


def _install_target_r4(block: nn.Module) -> None:
    """Install the staged BF16 H12288 immediately before ff_out."""
    if hasattr(block, "_target_numeric_r4") or hasattr(
        block.ff_out, "_target_a4_clip_ratio"
    ):
        raise ValueError("target R4 must be installed once, before clipping")

    def rotate_ff_input(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if len(inputs) != 1:
            raise ValueError("ff_out expects one activation input")
        output = structured_hadamard_12288_bf16(inputs[0])
        observer = getattr(_module, "_target_r4_observer", None)
        if observer is not None:
            observer(output)
        return (output,)

    block.ff_out._target_r4_hook_handle = block.ff_out.register_forward_pre_hook(
        rotate_ff_input
    )
    block._target_numeric_r4 = "h12288-bf16-staged/v1"


def _install_target_numeric_rms_norm(
    norm: nn.Module, workspace: RMSNormNumericWorkspace
) -> None:
    if norm.weight is None or norm.bias is not None:
        raise ValueError("target numeric RMSNorm requires a unity weight and no bias")
    if not bool(torch.all(norm.weight.detach().float() == 1.0)):
        raise ValueError(
            "target numeric RMSNorm gamma must be absorbed before installation"
        )

    def target_rms_forward(
        norm_module: nn.Module,
        values: torch.Tensor,
        *,
        _workspace: RMSNormNumericWorkspace = workspace,
    ) -> torch.Tensor:
        if values.is_cuda and triton_operator_numeric_available():
            result = rms_norm_bf16_cuda(
                values, None, float(norm_module.eps), _workspace
            )
        else:
            result = rms_norm_bf16(values.float(), None, float(norm_module.eps))
        exact = result.to(values.dtype)
        return exact

    norm.forward = MethodType(target_rms_forward, norm)
    norm._target_numeric_rms_norm = "bf16-pairwise-fast-rsqrt-gamma-bypass/v1"


def _install_target_numeric_block_ops(
    model: nn.Module,
    *,
    block_rms_norm: bool = True,
    final_rms_norm: bool = True,
    residual_add: bool = True,
    silu_pwl16: bool = True,
) -> None:
    blocks = model.model.transformer.blocks
    if len(blocks) != 32:
        raise ValueError(
            "target numeric deployment requires exactly 32 Transformer blocks"
        )
    workspace = RMSNormNumericWorkspace()
    norms = (
        [norm for block in blocks for norm in (block.attn_norm, block.ff_norm)]
        if block_rms_norm
        else []
    )
    if final_rms_norm:
        norms.append(model.model.transformer.ln_f)
    for norm in norms:
        _install_target_numeric_rms_norm(norm, workspace)
    for block in blocks:
        if not hasattr(block, "_residual_add") or not hasattr(block, "_silu_multiply"):
            raise TypeError(
                "target numeric deployment requires LLaDALlamaBlock instances"
            )
        if residual_add:
            block._target_numeric_bf16_residual = True
        if silu_pwl16:
            block._target_numeric_silu_pwl16 = True
    if all(
        (
            getattr(block, "_target_numeric_r4", None) == "h12288-bf16-staged/v1"
            for block in blocks
        )
    ):
        model._target_numeric_r4 = "h12288-bf16-staged/v1"
    model._target_numeric_rms_norm_count = len(norms)
    model._target_numeric_bf16_residual_block_count = len(blocks) if residual_add else 0
    model._target_numeric_silu_pwl_segments = 16 if silu_pwl16 else 0
    model._target_numeric_swiglu_multiply_block_count = len(blocks) if silu_pwl16 else 0


def _record_target_rope_table(model: nn.Module) -> None:
    blocks = model.model.transformer.blocks
    if len(blocks) != 32 or any(
        (
            not getattr(block, "_target_numeric_rope_before_k8", False)
            for block in blocks
        )
    ):
        raise ValueError("target RoPE-before-K8 must be installed on all 32 blocks")
    if any(
        (
            not getattr(
                getattr(block, "rotary_emb", None), "_target_numeric_bf16", False
            )
            for block in blocks
        )
    ):
        raise ValueError(
            "target BF16 RoPE arithmetic must be installed on all 32 blocks"
        )
    sequence_length = int(model.config.max_sequence_length)
    rotary = blocks[0].rotary_emb
    device = next(model.parameters()).device
    (position_sin, position_cos) = rotary.get_rotary_embedding(sequence_length, device)
    table = torch.cat(
        (position_sin.to(torch.bfloat16), position_cos.to(torch.bfloat16)), dim=0
    ).contiguous()
    payload = table.view(torch.int16).cpu().numpy().tobytes()
    model._target_numeric_rope = "bf16-mul-add-frozen-table/v1"
    model._target_numeric_k_cache_order = "rope-before-k8/v1"
    model._target_numeric_rope_table_shape = list(table.shape)
    model._target_numeric_rope_table_sha256 = hashlib.sha256(payload).hexdigest()


def build_target_numeric_coverage(
    model: nn.Module,
    *,
    candidate_numeric_mode: str,
    candidate_numeric_schedule: str,
    candidate_suppression: str,
    require_dynamic_a4a8: bool,
    transformer_weight_arithmetic: str = "gminus1_bf16_dual_scale/v1",
    rotation_source: str,
    rotation_sha256: str,
) -> dict[str, Any]:
    weight_arithmetic_group_size = {
        "gminus1_bf16_dual_scale/v1": -1,
    }
    if transformer_weight_arithmetic not in weight_arithmetic_group_size:
        raise ValueError("unsupported Transformer W4 arithmetic")
    head_bits = 8
    head_range = [-127, 127]
    errors: list[str] = []

    def is_sha256(value: str) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all((character in "0123456789abcdef" for character in value))
        )

    def is_bf16_materialized(values: torch.Tensor) -> bool:
        return values.dtype == torch.float32 and torch.equal(
            values, values.to(torch.bfloat16).float()
        )

    def shape_histogram(tensors: list[torch.Tensor]) -> list[dict[str, Any]]:
        counts: dict[tuple[int, ...], int] = {}
        for tensor in tensors:
            shape = tuple((int(extent) for extent in tensor.shape))
            counts[shape] = counts.get(shape, 0) + 1
        return [
            {"shape": list(shape), "count": counts[shape]} for shape in sorted(counts)
        ]

    transformer = model.model.transformer
    blocks = transformer.blocks
    block_linears = [
        getattr(block, name)
        for block in blocks
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_out",
            "ff_proj",
            "up_proj",
            "ff_out",
        )
    ]
    if len(blocks) != 32 or len(block_linears) != 224:
        errors.append("Transformer inventory must contain 32 blocks and 224 Linears")
    expected_weight_group_size = weight_arithmetic_group_size[
        transformer_weight_arithmetic
    ]
    valid_w4_modules = [
        module
        for module in block_linears
        if isinstance(module, SpinQuantW4A8Linear)
        and module.weight_bits == 4
        and (module.weight_group_size == expected_weight_group_size)
    ]
    if len(valid_w4_modules) != len(block_linears):
        errors.append(
            "Transformer Linears must use signed W4 with the declared group arithmetic"
        )
    if any(
        (
            module.weight_codes.dtype != torch.int8
            or int(module.weight_codes.min()) < -8
            or int(module.weight_codes.max()) > 7
            for module in valid_w4_modules
        )
    ):
        errors.append("Transformer W4 runtime codes must be INT8 values in [-8,7]")
    if any(
        (
            module.weight_scales_bf16.shape
            != (
                ((module.out_features,))
            )
            or not is_bf16_materialized(module.weight_scales_bf16)
            for module in valid_w4_modules
        )
    ):
        errors.append(
            "Transformer W4 scale shape must match the declared group arithmetic"
        )
    dynamic_bound = sum(
        (
            isinstance(module, SpinQuantW4A8Linear)
            and module.row_precision_context is not None
            for module in block_linears
        )
    )
    if require_dynamic_a4a8 and dynamic_bound != 224:
        errors.append("dynamic A4/A8 must bind all 224 Transformer Linears")
    lm_head = transformer.ff_out
    if (
        not isinstance(lm_head, SpinQuantW4A8Linear)
        or lm_head.weight_bits != head_bits
        or lm_head.weight_group_size != -1
        or (lm_head.activation_bits != 8)
    ):
        errors.append(f"lm_head must use signed G-1 W{head_bits} with dynamic A8")
    elif (
        lm_head.weight_codes.dtype != torch.int8
        or int(lm_head.weight_codes.min()) < head_range[0]
        or int(lm_head.weight_codes.max()) > head_range[1]
        or (lm_head.weight_scales_bf16.shape != (lm_head.out_features,))
        or (not is_bf16_materialized(lm_head.weight_scales_bf16))
    ):
        errors.append(
            f"lm_head W{head_bits} codes/scales do not match the signed G-1 BF16-scale format"
        )
    if getattr(model.config, "weight_tying", None) is not False:
        errors.append("embedding and lm_head must be untied")
    k_codecs = [getattr(block, "_spinquant_k_cache_codec", None) for block in blocks]
    v_codecs = [getattr(block, "_spinquant_v_cache_codec", None) for block in blocks]
    if any((not isinstance(codec, SpinQuantK8CacheCodec) for codec in k_codecs)):
        errors.append("all 32 blocks must install the K8 cache codec")
    if any(
        (
            not isinstance(codec, SpinQuantV8CacheCodec)
            or int(codec.scale_bf16.numel()) != 32
            for codec in v_codecs
        )
    ):
        errors.append("all 32 blocks must install 32-head static V8 scales")
    if any(
        (
            not callable(getattr(block, "_spinquant_score_override", None))
            or not callable(getattr(block, "_spinquant_probability_override", None))
            or (not callable(getattr(block, "_spinquant_context_override", None)))
            for block in blocks
        )
    ):
        errors.append(
            "all 32 blocks must install Q8/K8, LUT/P8, and P8/V8 overrides"
        )
    if any(
        (
            not is_bf16_materialized(codec.scale_bf16)
            for codec in v_codecs
            if isinstance(codec, SpinQuantV8CacheCodec)
        )
    ):
        errors.append("V8 static scales must be BF16 payload values")
    if getattr(model, "_target_numeric_k_cache_order", None) != "rope-before-k8/v1":
        errors.append("K cache order must be RoPE-before-K8")
    if getattr(model, "_target_numeric_rope", None) != "bf16-mul-add-frozen-table/v1":
        errors.append("Q/K RoPE must use the frozen BF16 target")
    if any(
        (
            not getattr(
                getattr(block, "rotary_emb", None), "_target_numeric_bf16", False
            )
            for block in blocks
        )
    ):
        errors.append("all 32 RotaryEmbedding instances must execute BF16 mul/add")
    if len(str(getattr(model, "_target_numeric_rope_table_sha256", ""))) != 64:
        errors.append("BF16 RoPE table must have a SHA-256 identity")
    if getattr(model, "_target_numeric_rms_norm_count", 0) != 65:
        errors.append("all 65 RMSNorm instances must use target arithmetic")
    if getattr(model, "_target_numeric_bf16_residual_block_count", 0) != 32:
        errors.append("all 32 blocks must use BF16 residual adds")
    if getattr(model, "_target_numeric_silu_pwl_segments", 0) != 16:
        errors.append("all 32 blocks must use the 16-segment SiLU")
    silu_coefficients = getattr(model, "_target_numeric_silu_coefficients", None)
    if any(
        (
            getattr(block, "_target_numeric_silu_coefficients", None)
            != silu_coefficients
            for block in blocks
        )
    ):
        errors.append("all blocks must use the same declared SiLU coefficients")
    if getattr(model, "_target_numeric_swiglu_multiply_block_count", 0) != 32:
        errors.append("all 32 blocks must use BF16 SwiGLU multiply")
    if getattr(model, "_target_numeric_r4", None) != "h12288-bf16-staged/v1":
        errors.append("R4 must use the frozen staged H12288 arithmetic")
    if not rotation_source or not is_sha256(rotation_sha256):
        errors.append("absorbed R1/R2 must record a source and SHA-256 identity")
    from numerics.candidate import CANDIDATE_LANES, CANDIDATE_NUMERIC_SCHEDULE

    if (
        candidate_numeric_mode != "bf16_lut"
        or candidate_numeric_schedule != CANDIDATE_NUMERIC_SCHEDULE
    ):
        errors.append("candidate must declare the active BF16/LUT schedule")
    if candidate_suppression != "raw-confidence-plus-zero-action/v1":
        errors.append(
            "candidate suppression must preserve raw confidence and zero action confidence"
        )
    w4_scales = [
        module.weight_scales_bf16
        for module in block_linears
        if isinstance(module, SpinQuantW4A8Linear) and module.weight_bits == 4
    ]
    query_heads = int(getattr(model.config, "n_heads", 0))
    effective_kv_heads = getattr(model.config, "effective_n_kv_heads", None)
    if effective_kv_heads is not None:
        kv_heads = int(effective_kv_heads)
    else:
        configured_kv_heads = getattr(model.config, "n_kv_heads", None)
        kv_heads = (
            int(configured_kv_heads)
            if configured_kv_heads is not None
            else 1
            if getattr(model.config, "multi_query_attention", None) is True
            else query_heads
        )
    if (
        query_heads != 32 or kv_heads != 32
    ):
        errors.append("target Q8/K8/P8/V8 Attention requires 32 query and KV heads")
    candidate_lut_payload = (
        torch.cat(
            (
                _candidate_exp_lut(torch.device("cpu")),
                _candidate_reciprocal_lut(torch.device("cpu")),
            )
        )
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .numpy()
        .tobytes()
    )
    candidate_lut_sha256 = hashlib.sha256(candidate_lut_payload).hexdigest()
    implementation_paths = sorted(
        {
            Path(__file__).resolve(),
            Path(bf16.__code__.co_filename).resolve(),
            Path(accelerated_linear_bf16.__code__.co_filename).resolve(),
        },
        key=str,
    )
    implementation_hasher = hashlib.sha256()
    implementation_files = []
    for implementation_path in implementation_paths:
        payload = implementation_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        implementation_hasher.update(implementation_path.name.encode("utf-8"))
        implementation_hasher.update(bytes.fromhex(digest))
        implementation_files.append(
            {"file": implementation_path.name, "sha256": digest}
        )
    implementation_sha256 = implementation_hasher.hexdigest()
    deployed_weight_group_size = weight_arithmetic_group_size[transformer_weight_arithmetic]
    group_count_histogram: list[dict[str, int]] = []
    group_counts: dict[int, int] = {}
    for module in block_linears:
        if isinstance(module, SpinQuantW4A8Linear):
            count = 1
            group_counts[count] = group_counts.get(count, 0) + 1
    group_count_histogram = [
        {"group_count": count, "linear_count": group_counts[count]}
        for count in sorted(group_counts)
    ]
    coverage = {
        "schema_version": "llada-target-numeric-coverage/v3",
        "valid": not errors,
        "errors": errors,
        "embedding": {
            "format": "bf16_lookup",
            "weight_tying": getattr(model.config, "weight_tying", None),
        },
        "transformer_weight": {
            "format": "signed_w4[-8,7]",
            "group_size": deployed_weight_group_size,
            "group_count_histogram": group_count_histogram,
            "linear_count": len(block_linears),
            "code_payload": "packed_signed_int4",
            "runtime_code_dtype": "int8",
            "code_range": [-8, 7],
            "zero_point": 0,
            "scale_mode": "per_output_channel",
            "scale_payload_dtype": "bf16",
            "runtime_scale_dtype": "float32_bf16_materialized",
            "scale_shape_rule": "[out_features]",
            "scale_shape_histogram": shape_histogram(w4_scales),
            "arithmetic": ({
                "id": transformer_weight_arithmetic,
                "p_g": "int32_dot_k",
                "first_rescale": "BF16(p_g*A_scale)",
                "weight_rescale": "BF16(first_rescale*W_scale_g)",
                "accumulator": "single_group",
                "group_order": None,
                "output": "bf16",
                "implementation_sha256": implementation_sha256,
                "implementation_files": implementation_files,
            }),
        },
        "transformer_activation": {
            "format": "dynamic_per_token_row_a4a8" if require_dynamic_a4a8 else "bf16",
            "bound_linear_count": dynamic_bound,
            "scale_mode": "dynamic_max_abs_per_token_row"
            if require_dynamic_a4a8
            else None,
            "scale_shape": "[token_rows]" if require_dynamic_a4a8 else None,
            "scale_payload_dtype": "bf16" if require_dynamic_a4a8 else None,
            "runtime_scale_dtype": "float32_bf16_materialized"
            if require_dynamic_a4a8
            else None,
            "allowed_bits": [4, 8] if require_dynamic_a4a8 else None,
            "code_ranges": {"a4": [-7, 7], "a8": [-127, 127]}
            if require_dynamic_a4a8
            else None,
            "zero_point": 0 if require_dynamic_a4a8 else None,
            "grouping": "none" if require_dynamic_a4a8 else None,
        },
        "attention_cache": {
            "qk_pv": "q8_k8_p8_v8_int32",
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "k_cache_order": getattr(model, "_target_numeric_k_cache_order", None),
            "v_scale_shape": [len(v_codecs), 32],
            "rope_table_shape": getattr(
                model, "_target_numeric_rope_table_shape", None
            ),
            "rope_table_sha256": getattr(
                model, "_target_numeric_rope_table_sha256", None
            ),
            "scales": ({
                "q8": {
                    "mode": "dynamic_max_abs_per_token_head",
                    "shape": "[batch,query_heads,query_tokens,1]",
                    "dtype": "bf16",
                },
                "k8": {
                    "mode": "dynamic_max_abs_per_token_head",
                    "shape": "[batch,kv_heads,cache_tokens,1]",
                    "dtype": "bf16",
                },
                "p8": {
                    "mode": "dynamic_max_abs_per_query_row",
                    "shape": "[batch,query_heads,query_tokens,1]",
                    "dtype": "bf16",
                },
                "v8": {
                    "mode": "static_per_layer_per_kv_head",
                    "shape": [len(v_codecs), 32],
                    "dtype": "bf16",
                },
            }),
            "code_range": [-127, 127],
            "zero_point": 0,
            "qk_accumulator": "int32",
            "qk_rescale_order": "BF16(INT32_QK*Q_scale*K_scale*BF16(head_dim^-0.5))",
            "pv_accumulator": "int32",
            "pv_rescale_order": "BF16(INT32_PV*P_scale*V_scale)",
        },
        "norm": {
            "implementation": "bf16_pairwise_fast_rsqrt_gamma_bypass",
            "count": getattr(model, "_target_numeric_rms_norm_count", 0),
            "transformer_count": 64,
            "final_count": 1,
            "gamma": "absorbed_then_bypassed",
        },
        "residual": {
            "attention": "bf16_add",
            "ffn": "bf16_add",
            "block_count": getattr(
                model, "_target_numeric_bf16_residual_block_count", 0
            ),
        },
        "rotation": {
            "r1_r2": "offline_absorbed",
            "source": rotation_source,
            "source_identity_kind": "artifact_manifest_sha256",
            "sha256": rotation_sha256,
            "r1": {
                "source": rotation_source,
                "artifact_manifest_sha256": rotation_sha256,
            },
            "r2": {
                "source": rotation_source,
                "artifact_manifest_sha256": rotation_sha256,
                "layer_count": 32,
            },
            "runtime_transform_count": 0,
            "r4": getattr(model, "_target_numeric_r4", None),
        },
        "ffn": {
            "silu_segments": getattr(model, "_target_numeric_silu_pwl_segments", 0),
            **(
                dict(
                    silu_coefficients=silu_coefficients,
                )
                if silu_coefficients is not None
                else {}
            ),
            "swiglu_multiply": "bf16",
            "r4": getattr(model, "_target_numeric_r4", None),
        },
        "lm_head": {
            "format": (f"signed_w{head_bits}[{head_range[0]},{head_range[1]}]_gminus1_a8"),
            "installed_weight_bits": (getattr(
                model, "_spinquant_lm_head_weight_bits", None
            )),
            "weight_code_payload": "signed_int8",
            "weight_runtime_code_dtype": "int8",
            "weight_code_range": head_range,
            "weight_zero_point": 0,
            "weight_group_size": -1,
            "weight_scale_mode": "per_output_channel",
            "weight_scale_payload_dtype": "bf16",
            "weight_scale_shape": list(lm_head.weight_scales_bf16.shape)
            if isinstance(lm_head, SpinQuantW4A8Linear)
            else None,
            "activation_format": "dynamic_per_token_row_a8",
            "activation_code_range": [-127, 127],
            "activation_scale_mode": "dynamic_max_abs_per_token_row",
            "activation_scale_shape": "[token_rows]",
            "activation_scale_payload_dtype": "bf16",
            "activation_zero_point": 0,
        },
        "candidate": {
            "mode": candidate_numeric_mode,
            "schedule": candidate_numeric_schedule,
            "suppression": candidate_suppression,
            "lanes": CANDIDATE_LANES,
            "logical_state_count": CANDIDATE_LANES,
            "matches_64_lane_reference": CANDIDATE_LANES == 64,
            "top1": "bf16_lane_max_lowest_token_id_tie_break",
            "confidence": "bf16_exp_reciprocal_lut_softmax",
            "lut_dtype": "bf16",
            "exp_lut_entries": 256,
            "reciprocal_lut_entries": 256,
            "lut_sha256": candidate_lut_sha256,
        },
    }
    return coverage


def _apply_rotary_positions(
    rotary_emb: nn.Module, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to local Q/K rows at their exact global token positions."""
    if (
        positions.ndim != 1
        or positions.numel() != q.shape[-2]
        or k.shape[-2] != q.shape[-2]
    ):
        raise ValueError(
            "RoPE positions must contain one global index per local Q/K token"
        )
    if positions.dtype != torch.long:
        positions = positions.to(torch.long)
    if bool(torch.any(positions < 0)):
        raise ValueError("RoPE positions must be nonnegative")
    target_bf16 = bool(getattr(rotary_emb, "_target_numeric_bf16", False))
    q_work = (
        q if target_bf16 else q.float() if rotary_emb.config.rope_full_precision else q
    )
    k_work = (
        k if target_bf16 else k.float() if rotary_emb.config.rope_full_precision else k
    )
    with torch.autocast(q.device.type, enabled=False):
        sequence_length = int(positions.max().item()) + 1
        (position_sin, position_cos) = rotary_emb.get_rotary_embedding(
            sequence_length, q.device
        )
        position_sin = position_sin.to(dtype=q_work.dtype).index_select(2, positions)
        position_cos = position_cos.to(dtype=q_work.dtype).index_select(2, positions)
        if target_bf16:
            position_sin = position_sin.to(torch.bfloat16)
            position_cos = position_cos.to(torch.bfloat16)
            rope = (
                rope_apply_bf16_cuda
                if q.is_cuda and triton_operator_numeric_available()
                else rope_apply_bf16
            )
            q_work = rope(q_work, position_sin, position_cos)
            k_work = rope(k_work, position_sin, position_cos)
        else:
            q_work = rotary_emb.apply_rotary_pos_emb(position_sin, position_cos, q_work)
            k_work = rotary_emb.apply_rotary_pos_emb(position_sin, position_cos, k_work)
    return (q_work.to(q.dtype), k_work.to(k.dtype))


def _install_native_attention_numeric(
    block: nn.Module,
    *,
    workspace: Int8MatmulWorkspace | None,
    qk_int8: bool,
    softmax_lut: bool,
    probability_p8: bool,
    k8_cache: bool = False,
    v8_codec: SpinQuantV8CacheCodec | None = None,
    rope_before_k8: bool = False,
    target_rope_before_cache: bool = False,
) -> None:
    if qk_int8 and workspace is None:
        raise ValueError("Q8/K8 attention requires an INT8 matmul workspace")
    if k8_cache:
        block._spinquant_k_cache_codec = SpinQuantK8CacheCodec()
    if rope_before_k8:
        if not k8_cache:
            raise ValueError("RoPE-before-K8 requires the K8 cache codec")
    if rope_before_k8 or target_rope_before_cache:
        block._target_numeric_rope_before_k8 = True
        block.rotary_emb._target_numeric_bf16 = True
    if v8_codec is not None:
        if workspace is None:
            raise ValueError("P8/V8 attention requires an INT8 matmul workspace")
        block._spinquant_v_cache_codec = v8_codec
    if qk_int8:

        def score_override(
            q: torch.Tensor,
            k: torch.Tensor,
            *,
            key_codes: torch.Tensor | None = None,
            key_scales: torch.Tensor | None = None,
        ) -> torch.Tensor:
            head_dim = int(q.shape[-1])
            q_quantized = quantize_per_row_bf16(q.detach().reshape(-1, head_dim))
            q_codes = q_quantized.codes.reshape_as(q)
            q_scales = q_quantized.scale_bf16.reshape(*q.shape[:-1], 1)
            if (key_codes is None) != (key_scales is None):
                raise ValueError("QK cache codes and scales must be provided together")
            if key_codes is None:
                k_quantized = quantize_per_row_bf16(k.detach().reshape(-1, head_dim))
                k_codes = k_quantized.codes.reshape_as(k)
                k_scales = k_quantized.scale_bf16.reshape(*k.shape[:-1], 1)
            else:
                if key_codes.dtype != torch.int8 or key_codes.shape != k.shape:
                    raise ValueError("QK K8 codes must match the key operand shape")
                if key_scales.shape != (*k.shape[:-1], 1):
                    raise ValueError("QK K8 scales must have one value per head/token")
                (k_codes, k_scales) = (key_codes, key_scales.float())
            runtime_cache = getattr(block, "_LLaDABlock__cache", None)
            if (
                isinstance(runtime_cache, dict)
                and bool(runtime_cache.get("attn_monitor_capture_qk_probe", False))
                and (
                    int(getattr(block, "layer_id", -1))
                    == int(getattr(block.config, "attn_monitor_layer", -2))
                )
            ):
                position_context = runtime_cache.get(
                    "attn_monitor_numeric_position_context"
                )
                if not isinstance(position_context, dict):
                    raise RuntimeError(
                        "Q8/K8 probe capture requires explicit position context"
                    )
                staged_group = int(position_context["staged_group"])
                groups = runtime_cache.setdefault(
                    "attn_monitor_numeric_probe_groups", {}
                )
                group_probe = groups.setdefault(staged_group, {})
                group_probe.update(
                    {
                        "layer_id": int(block.layer_id),
                        "staged_group": staged_group,
                        "query_positions": position_context["query_positions"],
                        "key_positions": position_context["key_positions"],
                        "kv_write_positions": position_context["kv_write_positions"],
                        "q_codes": q_codes.detach().clone(),
                        "q_scale_bf16": q_scales.squeeze(-1).detach().clone(),
                        "k_codes_qk_operand": k_codes.detach().clone(),
                        "k_scale_bf16_qk_operand": k_scales.squeeze(-1)
                        .detach()
                        .clone(),
                    }
                )
            output = workspace.acquire(
                "spinquant_native_qk", (*q.shape[:-1], k.shape[-2]), q.device
            )
            sums = int8_batched_matmul(
                q_codes, k_codes.transpose(-2, -1), output=output
            )
            exact = bf16(
                sums.to(torch.float32)
                * q_scales
                * k_scales.transpose(-2, -1)
                * bf16(head_dim ** (-0.5))
            )
            return exact

        block._spinquant_score_override = score_override
    if softmax_lut or probability_p8:

        def probability_override(scores: torch.Tensor) -> torch.Tensor:
            exact_scores = scores.detach()
            probabilities = (
                softmax_lut_bf16(exact_scores)
                if softmax_lut
                else torch.softmax(exact_scores.to(torch.float32), dim=-1)
            )
            if probability_p8:
                key_length = int(probabilities.shape[-1])
                quantized = quantize_per_row_bf16(probabilities.reshape(-1, key_length))
                exact = bf16(
                    quantized.codes.to(torch.float32)
                    * quantized.scale_bf16.unsqueeze(1)
                ).reshape_as(probabilities)
            else:
                exact = probabilities
            return exact

        block._spinquant_probability_override = probability_override
    if v8_codec is not None:

        def context_override(
            probabilities: torch.Tensor, v_codes: torch.Tensor
        ) -> torch.Tensor:
            if (
                v_codes.dtype != torch.int8
                or probabilities.ndim != 4
                or v_codes.ndim != 4
            ):
                raise ValueError(
                    "P8/V8 context requires rank-4 probabilities and INT8 V codes"
                )
            key_length = int(probabilities.shape[-1])
            if v_codes.shape[-2] != key_length:
                raise ValueError("probability and V cache token axes must match")
            quantized = quantize_per_row_bf16(
                probabilities.detach().reshape(-1, key_length)
            )
            p_codes = quantized.codes.reshape_as(probabilities)
            p_scales = quantized.scale_bf16.reshape(*probabilities.shape[:-1], 1)
            runtime_cache = getattr(block, "_LLaDABlock__cache", None)
            if isinstance(runtime_cache, dict) and (
                bool(runtime_cache.get("attn_monitor_capture_qk_probe", False))
                or bool(
                    runtime_cache.get(
                        "attn_monitor_capture_deployment_p8_relation", False
                    )
                )
            ):
                position_context = runtime_cache.get(
                    "attn_monitor_numeric_position_context"
                )
                if not isinstance(position_context, dict):
                    raise RuntimeError(
                        "deployment P8 probe capture requires explicit position context"
                    )
                staged_group = int(position_context["staged_group"])
                groups = runtime_cache.setdefault(
                    "attn_monitor_numeric_probe_groups", {}
                )
                group_probe = groups.setdefault(staged_group, {})
                group_probe.update(
                    {
                        "layer_id": int(getattr(block, "layer_id", -1)),
                        "staged_group": staged_group,
                        "query_positions": position_context["query_positions"],
                        "key_positions": position_context["key_positions"],
                        "kv_write_positions": position_context["kv_write_positions"],
                    }
                )
                p_bf16 = bf16(p_codes.to(torch.float32) * p_scales).to(torch.bfloat16)
                relation = p_bf16.mean(dim=1).to(torch.bfloat16)
                if int(getattr(block, "layer_id", -1)) == 0:
                    group_probe["l0_p8_relation_bf16"] = relation.detach().clone()
                prior = group_probe.get("all_layer_max_p8_relation_bf16")
                if prior is None:
                    group_probe["all_layer_max_p8_relation_bf16"] = (
                        relation.detach().clone()
                    )
                    group_probe["p8_relation_layer_count"] = 1
                else:
                    if (
                        not isinstance(prior, torch.Tensor)
                        or prior.shape != relation.shape
                    ):
                        raise RuntimeError(
                            "deployment P8 probe changed shape across layers"
                        )
                    group_probe["all_layer_max_p8_relation_bf16"] = torch.maximum(
                        prior, relation
                    )
                    group_probe["p8_relation_layer_count"] = (
                        int(group_probe.get("p8_relation_layer_count", 0)) + 1
                    )
                if int(getattr(block, "layer_id", -1)) == int(
                    getattr(block.config, "attn_monitor_layer", -2)
                ):
                    group_probe.update(
                        {
                            "selected_layer_id": int(getattr(block, "layer_id", -1)),
                            "selected_layer_p8_relation_bf16": relation.detach().clone(),
                        }
                    )
                    if (
                        int(getattr(block, "layer_id", -1))
                        == int(getattr(block.config, "n_layers", 0)) - 1
                    ):
                        group_probe["l31_p8_relation_bf16"] = relation.detach().clone()
                    if bool(runtime_cache.get("attn_monitor_capture_qk_probe", False)):
                        group_probe.update(
                            {
                                "selected_layer_p_codes": p_codes.detach().clone(),
                                "selected_layer_p_scale_bf16": p_scales.squeeze(-1)
                                .detach()
                                .clone(),
                            }
                        )
                        if (
                            int(getattr(block, "layer_id", -1))
                            == int(getattr(block.config, "n_layers", 0)) - 1
                        ):
                            group_probe["l31_p_codes"] = p_codes.detach().clone()
                            group_probe["l31_p_scale_bf16"] = (
                                p_scales.squeeze(-1).detach().clone()
                            )
            output = workspace.acquire(
                "spinquant_native_pv",
                (*probabilities.shape[:-1], v_codes.shape[-1]),
                probabilities.device,
            )
            sums = int8_batched_matmul(p_codes, v_codes, output=output)
            v_scales = v8_codec.expanded_scales(
                int(probabilities.shape[1]), probabilities.device
            ).view(1, -1, 1, 1)
            exact = bf16(sums.float() * p_scales * v_scales).to(torch.bfloat16)
            return exact

        block._spinquant_context_override = context_override


@torch.no_grad()
def override_target_silu_table(model: nn.Module, table: dict) -> None:
    from numerics.bf16 import _SILU_BREAKPOINTS

    if table.get("breakpoints") != list(_SILU_BREAKPOINTS):
        raise ValueError("SiLU table must preserve the sixteen existing intervals")
    values = torch.tensor([table["slopes"], table["intercepts"]], dtype=torch.float64)
    if (
        values.shape != (2, 16)
        or not torch.isfinite(values).all()
        or (not torch.equal(values, values.to(torch.bfloat16).double()))
        or (values[1, 8] != 0)
    ):
        raise ValueError(
            "SiLU coefficients must be finite BF16 [2,16] and preserve SiLU(0)=0"
        )
    blocks = model.model.transformer.blocks
    if len(blocks) != 32 or any(
        (not getattr(block, "_target_numeric_silu_pwl16", False) for block in blocks)
    ):
        raise ValueError("SiLU override requires all 32 deployed PWL blocks")
    coefficients = tuple((tuple(row) for row in values.tolist()))
    for block in blocks:
        block._target_numeric_silu_coefficients = coefficients
    model._target_numeric_silu_coefficients = coefficients


def install_target_numeric_lm_head(
    model: nn.Module, *, artifact_dir: Path, weight_bits: int | None = None
) -> dict[str, int]:
    """Install a G-1 W8 head with dynamic A8 and dual rescale."""
    reader = SpinQuantArtifactReader(artifact_dir, verify_files=False)
    if (
        not reader.head_quantized
        or reader.head_weight_bits != 8
        or (weight_bits is not None and reader.head_weight_bits != weight_bits)
        or (reader.head_group_size != -1)
    ):
        raise ValueError("target-numeric head requires a matching G-1 W8 child")
    head = model.model.transformer.ff_out
    if not isinstance(head, nn.Linear) or head.bias is not None:
        raise TypeError("target-numeric head requires a bias-free BF16 Linear")
    workspace = LinearNumericWorkspace()
    model.model.transformer.ff_out = SpinQuantW4A8Linear(
        reader.read_quantized_head(),
        workspace=workspace,
        activation_bits=8,
        module_name="model.transformer.ff_out",
        weight_bits=reader.head_weight_bits,
    ).to(_module_device(head))
    model._spinquant_lm_head_weight_bits = reader.head_weight_bits
    return {f"lm_head_w{reader.head_weight_bits}a8": 1}


def _install_artifact_weights(
    model: nn.Module,
    artifact_dir: Path,
    *,
    variant: str,
    row_precision_context: RowPrecisionContext | None = None,
    activation_bits: int = 8,
) -> dict[str, Any]:
    """Install G-1 weights and target R4 before binding INT8 Attention."""
    if variant not in ROTATED_VARIANTS:
        raise ValueError(
            "the Linear-A8 path requires a fixed SpinQuant rotation variant"
        )
    reader = SpinQuantArtifactReader(artifact_dir, verify_files=False)
    if reader.variant != variant:
        raise ValueError(
            f"artifact contains variant {reader.variant}, requested {variant}"
        )
    linear_workspace = LinearNumericWorkspace()
    replaced = {
        "linear": 0,
        "bf16": 0,
        "norm_unity": 0,
        "attention_int8": 0,
        "r4_layers": 0,
        "dynamic_a4a8_linear": 0,
        "lm_head_w8a8": 0,
    }
    for logical_id in sorted(expected_w4_ids()):
        module_name = logical_id[: -len(".weight")]
        (parent, child) = _parent(model, module_name)
        original = getattr(parent, child)
        if not isinstance(original, nn.Linear) or original.bias is not None:
            raise TypeError(
                f"{module_name} must be a bias-free Linear before SpinQuant replacement"
            )
        replacement = SpinQuantW4A8Linear(
            reader.read_w4(logical_id),
            workspace=linear_workspace,
            row_precision_context=row_precision_context,
            activation_bits=activation_bits,
            module_name=module_name,
        ).to(_module_device(original))
        setattr(parent, child, replacement)
        replaced["linear"] += 1
        replaced["dynamic_a4a8_linear"] += int(row_precision_context is not None)
    embedding = model.model.transformer.wte
    lm_head = model.model.transformer.ff_out
    with torch.no_grad():
        embedding.weight.copy_(reader.read_bf16(EMBEDDING).to(embedding.weight.device))
        if not reader.head_quantized:
            lm_head.weight.copy_(reader.read_bf16(LM_HEAD).to(lm_head.weight.device))
    replaced["bf16"] = 1
    if reader.head_quantized:
        if not isinstance(lm_head, nn.Linear) or lm_head.bias is not None:
            raise TypeError("model.transformer.ff_out must be a bias-free Linear")
        model.model.transformer.ff_out = SpinQuantW4A8Linear(
            reader.read_quantized_head(),
            workspace=linear_workspace,
            activation_bits=8,
            module_name="model.transformer.ff_out",
            weight_bits=int(reader.head_weight_bits),
        ).to(_module_device(lm_head))
        replaced[f"lm_head_w{reader.head_weight_bits}a8"] = 1
    else:
        replaced["bf16"] += 1
    norms = [
        norm
        for block in model.model.transformer.blocks
        for norm in (block.attn_norm, block.ff_norm)
    ]
    norms.append(model.model.transformer.ln_f)
    for norm in norms:
        if norm.weight is None or norm.bias is not None:
            raise ValueError("SpinQuant expects affine weight-only RMSNorm")
        with torch.no_grad():
            norm.weight.fill_(1.0)
        norm.weight.requires_grad_(False)
        replaced["norm_unity"] += 1
    for block in model.model.transformer.blocks:
        _install_target_r4(block)
        replaced["r4_layers"] += 1
    if replaced != {
        "linear": 224,
        "bf16": 1 if reader.head_quantized else 2,
        "norm_unity": 65,
        "attention_int8": 0,
        "r4_layers": 32,
        "dynamic_a4a8_linear": 224 if row_precision_context is not None else 0,
        "lm_head_w8a8": 1 if reader.head_weight_bits == 8 else 0,
    }:
        raise ValueError(f"incomplete SpinQuant weight installation: {replaced}")
    model._spinquant_lm_head_weight_bits = (
        int(reader.head_weight_bits) if reader.head_quantized else None
    )
    return replaced


def dequantize_spinquant_w4_bf16(weight: SpinQuantW4Tensor) -> torch.Tensor:
    """Reconstruct the persisted GPTQ tensor at its BF16 execution boundary."""
    if weight.codes.ndim != 2 or weight.codes.dtype != torch.int8:
        raise ValueError("invalid SpinQuant W4 code tensor")
    if int(weight.codes.min()) < -8 or int(weight.codes.max()) > 7:
        raise ValueError("SpinQuant W4 code outside [-8,7]")
    if weight.scale_bf16.shape != weight.codes.shape[:1]:
        raise ValueError("invalid SpinQuant W4 scale tensor")
    dequantized = weight.codes.to(torch.float32) * weight.scale_bf16.to(torch.float32).unsqueeze(1)
    return dequantized.to(torch.bfloat16)


def replace_with_spinquant_joint_full_w4a8_v8(
    model: nn.Module,
    artifact_dir: Path,
    *,
    variant: str,
    row_precision_context: RowPrecisionContext | None = None,
    activation_bits: int = 8,
) -> dict[str, Any]:
    """Run jointly calibrated W4/A8 with native Q8/K8, LUT, P8/V8 INT32 PV."""
    if variant not in ROTATED_VARIANTS:
        raise ValueError("joint W4A8/V8 requires a fixed SpinQuant rotation variant")
    reader = SpinQuantArtifactReader(artifact_dir, verify_files=False)
    if not reader.joint_v8:
        raise ValueError("joint W4A8/V8 execution requires a joint W4 artifact")
    v_scales = reader.read_v_cache_scales()
    restored = _install_artifact_weights(
        model,
        artifact_dir,
        variant=variant,
        row_precision_context=row_precision_context,
        activation_bits=activation_bits,
    )
    blocks = model.model.transformer.blocks
    if tuple(v_scales.shape) != (len(blocks), 32) or len(blocks) != 32:
        raise ValueError(
            "deployment V8 calibration must have shape [32 layers, 32 KV heads]"
        )
    attention_workspace = Int8MatmulWorkspace()
    for layer_index, block in enumerate(blocks):
        _install_native_attention_numeric(
            block,
            workspace=attention_workspace,
            qk_int8=True,
            softmax_lut=True,
            probability_p8=False,
            k8_cache=True,
            v8_codec=SpinQuantV8CacheCodec(v_scales[layer_index]),
            rope_before_k8=True,
        )
    restored["attention_int8"] = len(blocks)
    restored["k8_cache"] = len(blocks)
    restored["v8_cache"] = len(blocks)
    _install_target_numeric_block_ops(model)
    _record_target_rope_table(model)
    model._spinquant_v_cache_dtype = "int8-static-per-layer-per-kv-head"
    return restored
