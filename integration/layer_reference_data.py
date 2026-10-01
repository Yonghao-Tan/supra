"""Random standard-layer reference data from the quantized LLaDA forward.

Random weights and inputs produce expected numeric checkpoints.
The exported tensors use logical token order; the hardware adapter must undo its
physical token packing before comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from functools import wraps
from pathlib import Path
from types import MethodType, SimpleNamespace

RELEASE = Path(__file__).resolve().parent.parent
ALGORITHM_ROOT = RELEASE / "algorithm/llada"
sys.path.insert(0, str(RELEASE / "hardware/scripts"))
from artifact_paths import PROJECT_ROOT
import numpy as np


def _load_algorithm():
    global torch, nn, quantize_activation_per_activation_bits_bf16
    global Int8MatmulWorkspace, LinearNumericWorkspace, RMSNormNumericWorkspace
    global ActivationType, BlockType, BufferCache, LayerNormType, LLaDALlamaBlock, ModelConfig
    global RowPrecisionContext, bf16, quantize_per_row_bf16
    global SpinQuantK8CacheCodec, SpinQuantV8CacheCodec, SpinQuantW4A8Linear, SpinQuantW4Tensor
    global _install_native_attention_numeric, _install_target_numeric_rms_norm, _install_target_r4
    if "LLaDALlamaBlock" in globals():
        return
    sys.path.insert(0, str(ALGORITHM_ROOT))
    import torch
    from torch import nn
    from numerics.bf16 import (
        quantize_activation_per_row_bits_bf16
        as quantize_activation_per_activation_bits_bf16,
    )
    from numerics.int8_matmul import Int8MatmulWorkspace
    from numerics.linear_kernels import LinearNumericWorkspace
    from model.modeling_llada import (
        ActivationType, BlockType, BufferCache, LayerNormType, LLaDALlamaBlock, ModelConfig,
    )
    from numerics.operator_kernels import RMSNormNumericWorkspace
    from numerics.precision import RowPrecisionContext
    from numerics.bf16 import bf16, quantize_per_row_bf16
    from quantization.model import (
        SpinQuantK8CacheCodec, SpinQuantV8CacheCodec, SpinQuantW4A8Linear,
        _install_native_attention_numeric, _install_target_numeric_rms_norm, _install_target_r4,
    )
    from quantization.numeric import SpinQuantW4Tensor


def _algorithm_inference(function):
    @wraps(function)
    def call(*args, **kwargs):
        _load_algorithm()
        with torch.inference_mode():
            return function(*args, **kwargs)
    return call


_algorithm_data_root = os.environ.get("SUPRA_ALGORITHM_ROOT")
ALGO_ROOT = Path(_algorithm_data_root).expanduser().resolve() if _algorithm_data_root else None
SCHEMA = "supra-layer-reference-data/v2"
FORWARD_POSTPROCESS_SCHEMA = "supra-forward-postprocess-reference-data/v2"
REAL_LAYER_SCHEMA = "supra-layer-reference-data/v1"
LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")


def supported_layer_index(metadata: dict) -> bool:
    schema = metadata.get("schema")
    return schema in (SCHEMA, FORWARD_POSTPROCESS_SCHEMA) or (
        schema == REAL_LAYER_SCHEMA and
        metadata.get("reference_data_kind") == "real_cuda_consecutive_layers"
    )


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def raw_array(value: torch.Tensor) -> np.ndarray:
    import torch
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view("<u2").copy()
    if tensor.is_floating_point():
        if not torch.equal(tensor.float(), tensor.bfloat16().float()):
            raise ValueError("floating reference_data tensor is not BF16-materialized")
        return tensor.bfloat16().view(torch.int16).numpy().view("<u2").copy()
    return tensor.numpy().astype(tensor.numpy().dtype.newbyteorder("<"), copy=True)


def compare_arrays(expected: np.ndarray, actual: np.ndarray) -> dict:
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {"match": False, "reason": "shape/dtype mismatch",
                "expected_shape": list(expected.shape), "actual_shape": list(actual.shape),
                "expected_dtype": expected.dtype.str, "actual_dtype": actual.dtype.str}
    different = np.flatnonzero(expected.reshape(-1) != actual.reshape(-1))
    if not different.size:
        return {"match": True, "elements": int(expected.size)}
    first = int(different[0])
    return {"match": False, "mismatches": int(different.size),
            "first_index": [int(index) for index in np.unravel_index(first, expected.shape)],
            "expected_raw": int(expected.reshape(-1)[first]),
            "actual_raw": int(actual.reshape(-1)[first])}


@_algorithm_inference
def make_reference_data(*, tokens: int = 3, sequence: int = 17, seed: int = 20260904,
                 hidden: int = 4096, a8_token_count: int = 1,
                 scattered: bool = True,
                 activation_bits: list[int] | None = None,
                 kv_write_token_indices: list[int] | None = None,
                 cache_commit_token_indices: list[int] | None = None,
                 _hidden_input: torch.Tensor | None = None,
                 _positions: torch.Tensor | None = None,
                 _layer_index: int = 0,
                 _layer_count: int = 1,
                 _boundary: dict | None = None,
                 silu_reference_data: Path | None = None,
                 a4_clip_ratios: dict[str, float] | None = None) -> tuple[dict, dict, dict]:
    """Execute one cached block with random nonzero weights and mixed token bits.

    ``hidden=128`` is reserved for fast checker tests. FFN remains 12288 so the
    same staged H12288 path executes. The CLI always uses standard hidden=4096.
    """
    if hidden not in (128, 4096) or not 1 <= tokens <= sequence <= 2048:
        raise ValueError("require hidden=128/4096 and 1<=tokens<=sequence<=2048")
    if activation_bits is not None:
        if len(activation_bits) != tokens or any(bit not in (4, 8) for bit in activation_bits):
            raise ValueError("explicit activation_bits must contain one 4 or 8 per input token")
        a8_token_count = activation_bits.count(8)
    if not 0 <= a8_token_count <= tokens:
        raise ValueError("A8 token count must be between zero and total tokens")
    write_token_indices = list(range(tokens)) if kv_write_token_indices is None else sorted(kv_write_token_indices)
    commit_token_indices = write_token_indices if cache_commit_token_indices is None else sorted(cache_commit_token_indices)
    for name, selected in (("kv_write_token_indices", write_token_indices), ("cache_commit_token_indices", commit_token_indices)):
        if len(selected) != len(set(selected)) or any(token < 0 or token >= tokens for token in selected):
            raise ValueError(f"{name} must contain unique local query token indices")
    if not set(commit_token_indices) <= set(write_token_indices):
        raise ValueError("cache_commit_token_indices must be a subset of kv_write_token_indices")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    heads, ffn = hidden // 128, 12288
    config = ModelConfig(
        d_model=hidden, n_heads=heads, n_kv_heads=heads, n_layers=_layer_count,
        mlp_hidden_size=ffn, activation_type=ActivationType.silu,
        block_type=BlockType.llama, layer_norm_type=LayerNormType.rms,
        include_bias=False, include_qkv_bias=False, init_device="meta",
        rope=True, rope_full_precision=True, max_sequence_length=sequence,
        attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
        use_manual_attention=True, flash_attention=False,
    )
    if not 0 <= _layer_index < _layer_count:
        raise ValueError("layer index must be within the configured layer count")
    block_cache = BufferCache()
    block = LLaDALlamaBlock(_layer_index, config, block_cache).eval()
    context = RowPrecisionContext()
    bits = torch.full((tokens,), 4, dtype=torch.int8)
    if activation_bits is None:
        bits[torch.randperm(tokens, generator=generator)[:a8_token_count]] = 8
    else:
        bits = torch.tensor(activation_bits, dtype=torch.int8)
    context.activate(bits)
    if _positions is None:
        positions = (torch.randperm(sequence, generator=generator)[:tokens] if scattered
                     else torch.arange(sequence - tokens, sequence)).long()
    else:
        positions = _positions.detach().cpu().long().clone()
        if (positions.shape != (tokens,) or torch.unique(positions).numel() != tokens or
                positions.min().item() < 0 or positions.max().item() >= sequence):
            raise ValueError("explicit positions must be unique and within sequence")
    inputs: dict[str, torch.Tensor] = {"activation_bits": bits, "positions": positions}
    write_positions = positions[torch.tensor(write_token_indices, dtype=torch.long)]
    commit_positions = positions[torch.tensor(commit_token_indices, dtype=torch.long)]
    explicit_cache_sets = kv_write_token_indices is not None or cache_commit_token_indices is not None or _boundary is not None
    if explicit_cache_sets:
        inputs.update(kv_write_positions=write_positions, cache_commit_positions=commit_positions)
    expected: dict[str, torch.Tensor] = {}

    def capture(name: str, value: torch.Tensor) -> None:
        if name in expected:
            raise RuntimeError(f"duplicate layer observation: {name}")
        expected[name] = value.detach().clone()

    for name in LINEARS:
        original = getattr(block, name)
        codes = torch.randint(-8, 8, (original.out_features, original.in_features),
                              generator=generator, dtype=torch.int8)
        scales = ((0.5 + torch.rand(original.out_features, generator=generator))
                  / (4 * original.in_features ** 0.5)).bfloat16()
        inputs[f"weight.{name}.codes"] = codes
        inputs[f"weight.{name}.scale"] = scales
        linear = SpinQuantW4A8Linear(SpinQuantW4Tensor(codes, scales),
                                    LinearNumericWorkspace(), context, module_name=name)
        setattr(block, name, linear)

        def observe_linear(module, args, output, *, label=name):
            values = args[0].reshape(-1, module.in_features)
            quantized = quantize_activation_per_activation_bits_bf16(values, bits)
            accum = quantized.codes.int() @ module.weight_codes.int().T
            rescaled = bf16(bf16(accum.float() * quantized.scale_bf16[:, None])
                            * module.weight_scales_bf16[None, :])
            check = compare_arrays(raw_array(rescaled), raw_array(output.reshape_as(rescaled)))
            if not check["match"]:
                raise RuntimeError(f"{label} observed output differs from integer operands: {check}")
            capture(f"{label}.input", args[0])
            capture(f"{label}.activation_codes", quantized.codes)
            capture(f"{label}.activation_scale", quantized.scale_bf16)
            capture(f"{label}.accumulator", accum)
            capture(f"{label}.output", output)

        linear.register_forward_hook(observe_linear)

    for name in ("attn_norm", "ff_norm"):
        norm = getattr(block, name)
        norm.weight = nn.Parameter(torch.ones(hidden, dtype=torch.bfloat16), requires_grad=False)
        _install_target_numeric_rms_norm(norm, RMSNormNumericWorkspace())
        norm.register_forward_hook(lambda module, args, result, label=name: capture(label, result))
    block._target_numeric_bf16_residual = True
    block._target_numeric_silu_pwl16 = True
    from numerics.bf16 import _SILU_BREAKPOINTS
    # Deployed BF16 coefficients are configuration inputs; the layer
    # computes every expected output using these same exported inputs.
    segments = (
        0xbd58bbd6, 0xbe3bbcef, 0xbeb5bd92, 0xbedfbdc9,
        0xbec5bd94, 0xbe883c09, 0xbdf43e22, 0x00003ecc,
        0x00003f17, 0xbde63f56, 0xbe843f7d, 0xbec33f89,
        0xbeda3f8c, 0xbeb33f89, 0xbe293f83, 0xa6fc3f80,
    )
    inputs["silu_breakpoints"] = torch.tensor(_SILU_BREAKPOINTS, dtype=torch.bfloat16)
    for i in range(2):
        inputs[f"silu_coefficients.{i}"] = torch.tensor(
            [(value >> (16 * i)) & 0xffff for value in segments],
            dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    if silu_reference_data is not None:
        source = json.loads(silu_reference_data.read_text())
        if source.get("schema_version") == "shared-silu-pwl-calibration/v1":
            values = torch.tensor([source["slopes"], source["intercepts"]], dtype=torch.float64)
            if (source.get("breakpoints") != list(_SILU_BREAKPOINTS) or values.shape != (2, 16)
                    or not torch.isfinite(values).all()
                    or not torch.equal(values, values.bfloat16().double()) or values[1, 8] != 0):
                raise ValueError("SiLU coefficients must preserve fixed intervals and finite BF16 [2,16] values")
            for i in range(2):
                inputs[f"silu_coefficients.{i}"] = values[i].bfloat16()
        else:
            entries = {e["name"]: e for e in source["tensors"] if e["role"] == "input"}
            for name, count in (("silu_breakpoints", 17), ("silu_coefficients.0", 16),
                                ("silu_coefficients.1", 16)):
                entry = entries.get(name, entries.get(f"layer0.{name}"))
                if (entry is None or entry["encoding"] != "bf16_raw"
                        or entry["dtype"] != "<u2" or entry["shape"] != [count]
                        or entry["byte_count"] != count * 2):
                    raise ValueError(f"SiLU configuration requires {count} BF16 raw values: {name}")
                path = tensor_path(entry, silu_reference_data)
                if path.stat().st_size != count * 2:
                    raise ValueError(f"wrong SiLU payload size: {path}")
                raw = np.fromfile(path, dtype="<u2")
                inputs[name] = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    block._target_numeric_silu_coefficients = tuple(
        tuple(inputs[f"silu_coefficients.{i}"].float().tolist()) for i in range(2))
    _install_target_r4(block)
    if a4_clip_ratios is not None:
        from quantization.model import install_target_block_a4_clipping
        if set(a4_clip_ratios) != set(LINEARS):
            raise ValueError("A4 clipping requires explicit ratios for all seven Linears")
        install_target_block_a4_clipping(block, a4_clip_ratios)
        from quantization.model import describe_target_block_a4_clipping
        clipping = describe_target_block_a4_clipping(block)
        for name, clip_config in clipping.items():
            inputs[name + ".clip.installed"] = torch.tensor(clip_config["installed"])
            inputs[name + ".clip.ratio"] = clip_config["ratio_bf16"].clone()
            def observe_clip(values, label=name):
                for field, value in values.items():
                    capture(label + ".clip." + field, value)
            getattr(block, name)._target_a4_clip_observer = observe_clip
        block.ff_out._target_r4_observer = lambda value: capture("ff_out.r4_output", value)
    v_scale = (0.006 + torch.rand(heads, generator=generator) * 0.014).bfloat16()
    inputs["v_scale"] = v_scale
    _install_native_attention_numeric(
        block, workspace=Int8MatmulWorkspace(), qk_int8=True,
        softmax_lut=True, probability_p8=False, k8_cache=True,
        v8_codec=SpinQuantV8CacheCodec(v_scale), rope_before_k8=True,
    )
    residual_add = block._residual_add
    residual_count = 0

    def observe_residual(self, lhs, rhs):
        nonlocal residual_count
        result = residual_add(lhs, rhs)
        capture("attention_residual" if residual_count == 0 else "block_output", result)
        residual_count += 1
        return result

    block._residual_add = MethodType(observe_residual, block)
    silu_multiply = block._silu_multiply

    def observe_product(self, gate, up):
        result = silu_multiply(gate, up)
        from numerics.bf16 import _silu_pwl_bf16
        activated = _silu_pwl_bf16(gate, *block._target_numeric_silu_coefficients)
        capture("silu", activated)
        capture("product_before_r4", result)
        return result

    block._silu_multiply = MethodType(observe_product, block)
    key_encode = block._spinquant_k_cache_codec.encode

    def observe_key(values):
        result = key_encode(values)
        capture("key_rope", values)
        capture("new_key_codes", result[0])
        capture("new_key_scale", result[1])
        return result

    block._spinquant_k_cache_codec.encode = observe_key
    value_encode = block._spinquant_v_cache_codec.encode

    def observe_value(values):
        result = value_encode(values)
        capture("new_value_codes", result)
        return result

    block._spinquant_v_cache_codec.encode = observe_value
    score_forward = block._spinquant_score_override

    def observe_scores(q, k, **kwargs):
        result = score_forward(q, k, **kwargs)
        capture("query_rope", q)
        q8 = quantize_per_row_bf16(q.reshape(-1, 128))
        qc, kc = q8.codes.reshape_as(q), kwargs["key_codes"]
        capture("query_codes", qc)
        capture("query_scale", q8.scale_bf16.reshape(*q.shape[:-1], 1))
        capture("qk_key_codes", kc)
        capture("qk_key_scale", kwargs["key_scales"])
        capture("qk_accumulator", qc.int() @ kc.int().transpose(-2, -1))
        capture("scores", result)
        return result

    block._spinquant_score_override = observe_scores
    probability_forward = block._spinquant_probability_override

    def observe_probabilities(scores):
        result = probability_forward(scores)
        capture("probabilities", result)
        return result

    block._spinquant_probability_override = observe_probabilities
    context_forward = block._spinquant_context_override

    def observe_context(probability, value):
        result = context_forward(probability, value)
        p8 = quantize_per_row_bf16(probability.reshape(-1, sequence))
        pc = p8.codes.reshape_as(probability)
        capture("probability_codes", pc)
        capture("probability_scale", p8.scale_bf16.reshape(*probability.shape[:-1], 1))
        capture("pv_value_codes", value)
        capture("pv_accumulator", pc.int() @ value.int())
        capture("context", result)
        return result

    block._spinquant_context_override = observe_context
    hidden_input = (torch.randn(1, tokens, hidden, generator=generator).bfloat16()
                    if _hidden_input is None else
                    _hidden_input.detach().cpu().bfloat16().clone())
    if hidden_input.shape != (1, tokens, hidden):
        raise ValueError("explicit hidden input has the wrong shape")
    retained = torch.randn(1, heads, sequence, 128, generator=generator).bfloat16()
    key_codes, key_scale = SpinQuantK8CacheCodec().encode(retained)
    value_codes = SpinQuantV8CacheCodec(v_scale).encode(
        torch.randn(1, heads, sequence, 128, generator=generator).bfloat16())
    inputs.update(hidden=hidden_input, retained_key_codes=key_codes.clone(),
                  retained_key_scale=key_scale.clone(), retained_value_codes=value_codes.clone())
    sine, cosine = block.rotary_emb.get_rotary_embedding(sequence, torch.device("cpu"))
    inputs.update(rope_sine=sine.bfloat16(), rope_cosine=cosine.bfloat16())
    block.rotary_emb._RotaryEmbedding__cache["rope_pos_sin"] = inputs["rope_sine"]
    block.rotary_emb._RotaryEmbedding__cache["rope_pos_cos"] = inputs["rope_cosine"]
    boundary_result = None
    runtime = None
    if _boundary is not None:
        from generation.refresh import begin_boundary_layer0_scout, read_boundary_layer0_scout, end_boundary_layer0_scout
        from generation.engine import _select_boundary_layer0_deep_positions
        if tokens != sequence or _layer_index != 0 or cache_commit_token_indices is not None or not torch.equal(positions, torch.arange(sequence)):
            raise ValueError("boundary scout requires full ordered Layer0 queries and selection-derived commit tokens")
        runtime = SimpleNamespace(_LLaDAModel__cache=block_cache, device=torch.device("cpu"))
        begin_boundary_layer0_scout(runtime,
            current_positions=torch.tensor(_boundary["current_positions"], dtype=torch.long), total_length=sequence,
            transition_positions=(torch.tensor(_boundary["transition_positions"], dtype=torch.long)
                                  if _boundary.get("transition_positions") else None))
    restore_positions = write_positions[~torch.isin(write_positions, commit_positions)]
    if restore_positions.numel():
        from generation.engine import _snapshot_cache_rows, _restore_cache_rows
        snapshot = _snapshot_cache_rows(((key_codes, key_scale, value_codes),), restore_positions)
    try:
        output, cache = block(hidden_input, layer_past=(key_codes, key_scale, value_codes),
                              use_cache=True, query_position_ids=positions[None, :],
                              kv_write_position_ids=write_positions[None, :])
        if runtime is not None:
            score_q8, probability_entries = read_boundary_layer0_scout(runtime)
    finally:
        if runtime is not None:
            end_boundary_layer0_scout(runtime)
    if runtime is not None:
        selection = {name: torch.tensor(_boundary[name], dtype=torch.long)
                     for name in ("current_positions", "eligible_positions", "mandatory_candidate_positions", "required_candidate_positions")}
        deep, optional, ranked = _select_boundary_layer0_deep_positions(score_q8,
            target_rows=_boundary["target_token_count"], required_candidate_count=_boundary.get("required_candidate_count", 0), **selection)
        commit_positions = deep[torch.isin(deep, write_positions)]
        commit_token_indices = torch.isin(positions, commit_positions).nonzero().flatten().tolist()
        inputs["cache_commit_positions"] = commit_positions
        restore_positions = write_positions[~torch.isin(write_positions, commit_positions)]
        from generation.engine import _snapshot_cache_rows, _restore_cache_rows
        snapshot = _snapshot_cache_rows(((inputs["retained_key_codes"], inputs["retained_key_scale"], inputs["retained_value_codes"]),), restore_positions)
        boundary_result = dict(**_boundary, score_q8=score_q8.tolist(), deep_positions=deep.tolist(),
            optional_positions=optional.tolist(), ranked_positions=ranked.tolist(), probability_entries=probability_entries)
    if restore_positions.numel():
        _restore_cache_rows(snapshot, (cache,), restore_positions)
    for name, tensor in zip(("cache_key_codes", "cache_key_scale", "cache_value_codes"), cache):
        capture(name, tensor)
    if not torch.isfinite(output).all():
        raise RuntimeError("random layer produced nonfinite output")
    from numerics.candidate import _exp_lut, _reciprocal_lut
    inputs.update(softmax_exp_lut=_exp_lut(torch.device("cpu")),
                  softmax_reciprocal_lut=_reciprocal_lut(torch.device("cpu")))
    metadata = dict(schema=SCHEMA, seed=seed, tokens=tokens, sequence=sequence,
                    hidden=hidden, ffn=ffn, heads=heads, head_dim=128,
                    a4_token_count=tokens - a8_token_count, a8_token_count=a8_token_count, activation_bits=bits.tolist(),
                    activation_slice_units=tokens + a8_token_count,
                    standard_layer=hidden == 4096, token_order="logical_input_order",
                    weight_layout="output_channel,input_channel; unpacked signed int8 W4 codes",
                    weight_group=-1, gamma_bypass=True, rms_epsilon=config.rms_norm_eps,
                    rotation="h12288-bf16-staged/v1; R1/R2 absorbed",
                    cache_visibility="all current new K/V mutually visible; retained other positions",
                    backend="CPU PyTorch LLaDALlamaBlock",
                    torch_version=torch.__version__)
    if explicit_cache_sets:
        metadata.update(kv_write_token_indices=write_token_indices, cache_commit_token_indices=commit_token_indices,
            cache_visibility="Attention sees new K/V only at kv_write_positions; all other keys retained",
            cache_persistence="cache_* checkpoints follow actual generator cache-token restoration; only cache_commit_positions persist")
    if boundary_result is not None:
        metadata["boundary_scout"] = boundary_result
    metadata["model_layer_index"] = _layer_index
    if silu_reference_data is not None:
        metadata["layer_count"] = 1
    if a4_clip_ratios is not None:
        metadata["a4_clip_ratios"] = {name: float(torch.tensor(ratio).bfloat16())
                                     for name, ratio in a4_clip_ratios.items()}
        rule = next(iter(clipping.values()))["rule"]
        metadata["layer_count"] = 1
        metadata["numeric_capabilities"] = dict(a4_clipping=rule)
        metadata["layers"] = [dict(layer=0, model_layer=_layer_index,
            clipping={name: dict(installed=config["installed"], rule=rule,
                ratio_bf16_raw=int(raw_array(config["ratio_bf16"]).item()),
                observation_status="complete") for name, config in clipping.items()})]
    return inputs, expected, metadata


@_algorithm_inference
def make_multilayer_reference_data(*, layers: int, tokens: int = 49,
                            sequence: int = 2048, seed: int = 20260905,
                            hidden: int = 4096, a8_token_count: int = 0,
                            scattered: bool = True,
                            activation_bits: list[int] | None = None
                            ) -> tuple[dict, dict, dict]:
    if layers < 2:
        raise ValueError("multilayer reference_data requires at least two layers")
    merged_inputs: dict[str, torch.Tensor] = {}
    merged_expected: dict[str, torch.Tensor] = {}
    shared_names = {
        "activation_bits", "positions", "hidden", "rope_sine", "rope_cosine",
        "softmax_exp_lut", "softmax_reciprocal_lut",
        "silu_breakpoints",
    }
    current_hidden = None
    positions = None
    bits = activation_bits
    layer_metadata = []
    base_metadata = None
    for layer in range(layers):
        layer_seed = seed + layer * 1009
        inputs, expected, metadata = make_reference_data(
            tokens=tokens, sequence=sequence, seed=layer_seed, hidden=hidden,
            a8_token_count=a8_token_count, scattered=scattered, activation_bits=bits,
            _hidden_input=current_hidden, _positions=positions,
            _layer_index=layer, _layer_count=layers)
        if layer == 0:
            base_metadata = metadata
            for name in shared_names:
                merged_inputs[name] = inputs[name]
            positions = inputs["positions"]
            bits = inputs["activation_bits"].tolist()
        else:
            for name in ("activation_bits", "positions", "rope_sine", "rope_cosine",
                         "softmax_exp_lut", "softmax_reciprocal_lut"):
                if not torch.equal(inputs[name], merged_inputs[name]):
                    raise RuntimeError(f"layer {layer} changed shared input {name}")
            merged_inputs[f"layer{layer}.hidden"] = inputs["hidden"]
        for name, value in inputs.items():
            if name in shared_names:
                continue
            merged_inputs[f"layer{layer}.{name}"] = value
        for name, value in expected.items():
            merged_expected[f"layer{layer}.{name}"] = value
        current_hidden = expected["block_output"]
        layer_metadata.append({
            "layer": layer, "seed": layer_seed,
            "expected_checkpoints": len(expected),
        })
    if base_metadata is None:
        raise RuntimeError("multilayer reference_data produced no layers")
    metadata = {
        **base_metadata,
        "seed": seed,
        "layers": layers,
        "layer_metadata": layer_metadata,
        "cache_visibility": "each layer sees all current new K/V; retained caches are per-layer",
        "backend": "CPU PyTorch sequential LLaDALlamaBlock",
        "torch_version": torch.__version__,
    }
    return merged_inputs, merged_expected, metadata


@_algorithm_inference
def make_boundary_reference_data(*, sequence: int, current_positions: list[int],
                          target_token_count: int, seed: int = 20260906,
                          hidden: int = 4096, activation_bits: list[int] | None = None,
                          transition_positions: list[int] | None = None,
                          eligible_positions: list[int] | None = None,
                          mandatory_candidate_positions: list[int] | None = None,
                          required_candidate_positions: list[int] | None = None,
                          required_candidate_count: int = 0,
                          shortlist_pending_raw: list[int] | None = None,
                          shortlist_context_positions: list[int] | None = None,
                          deep_activation_bits: int | None = None,
                          silu_reference_data: Path | None = None) -> list[tuple[dict, dict, dict]]:
    """Actual full Layer0 scout, selected cache persistence, and gathered Layer1.

    The two operator_executions have separate checkpoint indices because query counts
    differ. Layer1 hidden is reference data for comparison, never DUT stimulus.
    """
    if deep_activation_bits not in (None, 4, 8):
        raise ValueError("deep_activation_bits must be 4, 8 or absent")
    boundary = dict(current_positions=current_positions, target_token_count=target_token_count,
        eligible_positions=([p for p in range(sequence) if p not in current_positions]
                            if eligible_positions is None else eligible_positions),
        mandatory_candidate_positions=mandatory_candidate_positions or [],
        required_candidate_positions=required_candidate_positions or [],
        required_candidate_count=required_candidate_count,
        transition_positions=transition_positions)
    prefix_pending = None
    if shortlist_pending_raw is not None:
        from generation.refresh import CrossBlockPrefixAttentionState
        if eligible_positions is not None or len(shortlist_pending_raw) != sequence:
            raise ValueError("shortlist requires a full pending vector and selection-derived eligible positions")
        state = CrossBlockPrefixAttentionState(total_length=sequence, block_length=len(current_positions),
            boundary_target_rows=target_token_count, device=torch.device("cpu"))
        prefix_pending = torch.tensor(shortlist_pending_raw, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        if not torch.isfinite(prefix_pending).all() or (prefix_pending < 0).any() or (prefix_pending > 1).any():
            raise ValueError("shortlist pending must be finite BF16 probabilities")
        state.pending.copy_(prefix_pending)
        optional = torch.tensor([p for p in range(sequence) if p not in current_positions], dtype=torch.long)
        count = min(optional.numel(), 2*(target_token_count-len(current_positions)))
        if not 1 <= count <= 96:
            raise ValueError("boundary shortlist must contain 1..96 tokens")
        order = torch.argsort(state.score_positions(optional), descending=True, stable=True)
        context = shortlist_context_positions or []
        merged = torch.cat((optional[order[:count]], torch.tensor(context, dtype=torch.long),
            torch.tensor(mandatory_candidate_positions or [], dtype=torch.long)))
        boundary.update(eligible_positions=torch.sort(torch.unique(merged)).values.tolist(),
            shortlist_token_count=count, shortlist_context_positions=context,
            shortlist_pending_raw=shortlist_pending_raw)
    first = make_reference_data(tokens=sequence, sequence=sequence, seed=seed, hidden=hidden,
        activation_bits=[8] * sequence if activation_bits is None else activation_bits,
        scattered=False, _layer_count=2, _boundary=boundary,
        silu_reference_data=silu_reference_data)
    if prefix_pending is not None:
        first[0]["prefix_pending"] = prefix_pending
    selected = torch.tensor(first[2]["boundary_scout"]["deep_positions"], dtype=torch.long)
    second = make_reference_data(tokens=selected.numel(), sequence=sequence,
        seed=seed + 1009, hidden=hidden,
        activation_bits=(first[0]["activation_bits"].index_select(0, selected).tolist()
                         if deep_activation_bits is None else
                         [deep_activation_bits] * selected.numel()),
        _positions=selected, _hidden_input=first[1]["block_output"].index_select(1, selected),
        _layer_index=1, _layer_count=2, silu_reference_data=silu_reference_data)
    return [first, second]


@_algorithm_inference
def make_forward_postprocess_reference_data(*, tokens: int = 16, vocabulary: int = 64,
                            seed: int = 20260905,
                            _hidden_input: torch.Tensor | None = None,
                            _selected_tokens: torch.Tensor | None = None) -> tuple[dict, dict, dict]:
    from numerics.candidate import (
        _exp_lut, _reciprocal_lut, candidate_action_confidence,
        candidate_token_probability_bf16, streaming_candidate_bf16,
    )
    from model.modeling_llada import RMSLayerNorm
    from quantization.numeric import SpinQuantW8Tensor

    if not 1 <= tokens <= 128 or not 8 <= vocabulary <= 126464 or vocabulary % 8:
        raise ValueError("forward-postprocess requires 1..128 tokens and an M8 vocabulary in 8..126464")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = 4096
    config = ModelConfig(d_model=hidden, init_device="cpu", include_bias=False,
                         rms_norm_eps=1e-6)
    norm = RMSLayerNorm(config).bfloat16().eval()
    _install_target_numeric_rms_norm(norm, RMSNormNumericWorkspace())
    values = (torch.randn(tokens, hidden, generator=generator).bfloat16()
              if _hidden_input is None else _hidden_input.detach().cpu().bfloat16().clone())
    if values.shape != (tokens, hidden):
        raise ValueError("post hidden input must have shape [tokens,4096]")
    codes = torch.randint(-127, 128, (vocabulary, hidden), generator=generator,
                          dtype=torch.int8)
    codes[0, :2] = torch.tensor([-127, 127], dtype=torch.int8)
    scales = ((0.5 + torch.rand(vocabulary, generator=generator)) /
              (64 * hidden**0.5)).bfloat16()
    linear = SpinQuantW4A8Linear(SpinQuantW8Tensor(codes, scales),
                                LinearNumericWorkspace(), activation_bits=8,
                                weight_bits=8, module_name="model.transformer.ff_out")
    normalized = norm(values)
    quantized = quantize_per_row_bf16(normalized)
    accum = quantized.codes.int() @ codes.int().T
    logits = linear(normalized)
    arithmetic = bf16(bf16(accum.float() * quantized.scale_bf16[:, None]) * scales[None, :])
    if not compare_arrays(raw_array(logits), raw_array(arithmetic))["match"]:
        raise RuntimeError("W8 head differs from its recorded integer operands")
    token, top_logit, confidence = streaming_candidate_bf16(logits)
    selected_tokens = (torch.arange(tokens, dtype=torch.long) % vocabulary
                       if _selected_tokens is None else _selected_tokens.detach().cpu().long().clone())
    if selected_tokens.shape != (tokens,) or bool(((selected_tokens < 0) | (selected_tokens >= vocabulary)).any()):
        raise ValueError("post selected tokens must contain one vocabulary index per token")
    selected_probability = candidate_token_probability_bf16(
        logits, selected_tokens, top_logit, confidence)
    suppressed = torch.tensor([int(token[0])], dtype=torch.long)
    action = candidate_action_confidence(token, confidence, suppressed.tolist())
    inputs = dict(hidden=values, weight_codes=codes, weight_scale=scales,
                  selected_tokens=selected_tokens, suppressed_tokens=suppressed,
                  exp_lut=_exp_lut(torch.device("cpu")),
                  reciprocal_lut=_reciprocal_lut(torch.device("cpu")))
    expected = dict(final_norm=normalized, activation_codes=quantized.codes,
                    activation_scale=quantized.scale_bf16, accumulator=accum,
                    logits=logits, top1=token, top_logit=top_logit,
                    confidence=confidence, selected_probability=selected_probability,
                    action_confidence=action)
    metadata = dict(schema=FORWARD_POSTPROCESS_SCHEMA, seed=seed, tokens=tokens,
                    hidden=hidden, vocabulary=vocabulary, weight_bits=8,
                    activation_bits=8, weight_group=-1, rms_epsilon=1e-6,
                    backend="CPU RMSLayerNorm, SpinQuant W8 head and Candidate",
                    torch_version=torch.__version__)
    return inputs, expected, metadata


def make_control_loop_reference_data(token_metadata: bytes, *, seed: int = 20260907,
                              sequence: int = 512, silu_reference_data: Path | None = None,
                              a4_clip_ratios: dict[str, float] | None = None) -> tuple[dict, dict, dict]:
    """Compute numeric checkpoints for the selection in the input descriptor."""
    _load_algorithm()
    import struct
    token_count = token_metadata[0] if token_metadata else 0
    if not 1 <= token_count <= 48 or len(token_metadata) < 32+16*token_count:
        raise ValueError("control loop requires a complete one-round token descriptor")
    tokens, positions, bits = [], [], []
    for token in range(token_count):
        offset = 32+token*16
        tokens.append(struct.unpack_from("<I", token_metadata, offset)[0])
        positions.append(struct.unpack_from("<H", token_metadata, offset+4)[0])
        bits.append(token_metadata[offset+10])
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    if min(tokens) < 0 or max(tokens) >= 64:
        raise ValueError("control loop LM head uses vocabulary64")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    embedding = torch.randn(64,4096,generator=generator).bfloat16()
    inputs, expected, metadata = make_reference_data(tokens=token_count,sequence=sequence,seed=seed,
        activation_bits=bits,_positions=torch.tensor(positions,dtype=torch.long),
        _hidden_input=embedding[token_tensor][None,:], silu_reference_data=silu_reference_data,
        a4_clip_ratios=a4_clip_ratios)
    head_inputs, head_expected, _ = make_forward_postprocess_reference_data(tokens=token_count,seed=seed,
        _hidden_input=expected["block_output"].reshape(token_count,4096),_selected_tokens=token_tensor)
    inputs["embedding"] = embedding
    inputs["tokens"] = token_tensor
    inputs.update({"post."+key:value for key,value in head_inputs.items()})
    expected.update({"post."+key:value for key,value in head_expected.items()})
    metadata["control_loop"] = "Input descriptor supplies token selection; block output feeds LM-head input."
    return inputs,expected,metadata


def export_reference_data(inputs: dict, expected: dict, metadata: dict,
                   output: Path, index: Path) -> None:
    import torch
    output, index = output.resolve(), index.resolve()
    if ALGO_ROOT is None:
        raise ValueError("set SUPRA_ALGORITHM_ROOT before generating reference_data payload")
    if not _within(output, ALGO_ROOT) or _within(output, PROJECT_ROOT):
        raise ValueError(f"reference_data payload must be below {ALGO_ROOT}")
    if index.exists() or output.exists():
        raise FileExistsError("refusing to overwrite an existing reference_data or index")
    output.mkdir(parents=True)
    required_bytes = sum(t.numel() * (4 if t.dtype == torch.float32 and name.endswith((".clip.row_max", ".clip.limit_fp32"))
                                      else 2 if t.is_floating_point() else t.element_size())
                         for values in (inputs, expected) for name, t in values.items())
    if shutil.disk_usage(output).free < required_bytes + 64 * 1024 * 1024:
        raise OSError("insufficient free space for layer reference_data")
    index.parent.mkdir(parents=True, exist_ok=True)
    tensors = []
    for role, values in (("input", inputs), ("expected", expected)):
        for name, tensor in values.items():
            fp32_raw = tensor.dtype == torch.float32 and name.endswith((".clip.row_max", ".clip.limit_fp32"))
            raw = (tensor.detach().cpu().contiguous().view(torch.int32).numpy().view("<u4").copy()
                   if fp32_raw else raw_array(tensor))
            path = output / f"{role}.{name}.bin"
            raw.tofile(path)
            entry = dict(role=role, name=name, path=os.path.relpath(path, index.parent), shape=list(raw.shape),
                         dtype=raw.dtype.str, byte_count=raw.nbytes,
                         encoding="float32_raw" if fp32_raw else "bf16_raw" if tensor.is_floating_point() else "integer")
            if raw.nbytes >= 16 * 1024 * 1024:
                entry["sha256"] = hashlib.sha256(memoryview(raw).cast("B")).hexdigest()
            tensors.append(entry)
    index.write_text(json.dumps({**metadata, "tensors": tensors}, indent=2) + "\n")


def tensor_path(entry: dict, index: Path, payload_root: Path | None = None) -> Path:
    path = Path(entry["path"])
    if path.is_absolute():
        return path
    return (payload_root if payload_root is not None else index.parent) / path


def validate_numeric_capabilities(metadata: dict) -> None:
    entries = {(entry["role"], entry["name"]): entry for entry in metadata["tensors"]}
    capability = metadata.get("numeric_capabilities", {}).get("a4_clipping")
    if capability is None:
        if any(".clip." in name or name.endswith(".r4_output") for _, name in entries):
            raise ValueError("clipping/R4 observations require explicit numeric_capabilities")
        return
    if capability != "a4-row-clip-bf16/v1":
        raise ValueError(f"unsupported clipping numerical rule: {capability}")

    def require(role, name, encoding, dtype, shape):
        entry = entries.get((role, name))
        if entry is None:
            raise ValueError(f"missing required {role} tensor: {name}")
        if (entry["encoding"] != encoding or np.dtype(entry["dtype"]) != np.dtype(dtype)
                or entry["shape"] != shape):
            raise ValueError(f"wrong clipping shape/dtype/encoding: {name}")

    if not metadata.get("layers"):
        raise ValueError("clipping capability requires explicit layer records")
    for layer in metadata["layers"]:
        prefix = ("" if len(metadata["layers"]) == 1 and
                  ("input", "weight.q_proj.codes") in entries else f"layer{layer['layer']}.")
        states = layer.get("clipping", {})
        if set(states) != set(LINEARS):
            raise ValueError("clipping requires explicit state for all seven Linears")
        for linear, state in states.items():
            name = prefix + linear
            require("input", name + ".clip.installed", "integer", "|b1", [])
            if type(state.get("installed")) is not bool or state.get("rule") != capability:
                raise ValueError(f"unknown clipping installation state: {name}")
            installed = state["installed"]
            status = "complete" if installed else "not_applicable"
            if state.get("observation_status") != status:
                raise ValueError(f"incomplete clipping observation: {name}")
            ratio = state.get("ratio_bf16_raw")
            if not installed:
                if ratio is not None or ("input", name + ".clip.ratio") in entries or any(
                        role == "expected" and key.startswith(name + ".clip.") for role, key in entries):
                    raise ValueError(f"uninstalled clipping contains active observations: {name}")
                continue
            if type(ratio) is not int or not 0 < ratio <= 0x3f80:
                raise ValueError(f"invalid BF16 clipping ratio: {name}")
            require("input", name + ".clip.ratio", "bf16_raw", "<u2", [])
            source = entries.get(("expected", name + ".input"))
            if source is None or len(source["shape"]) != 3:
                raise ValueError(f"clipping requires actual Linear input: {name}")
            shape = source["shape"]
            row_shape = [shape[0] * shape[1], 1]
            for field in ("input", "output"):
                require("expected", name + ".clip." + field, "bf16_raw", "<u2", shape)
            require("expected", name + ".clip.limit", "bf16_raw", "<u2", row_shape)
            for field in ("row_max", "limit_fp32"):
                require("expected", name + ".clip." + field, "float32_raw", "<u4", row_shape)
        source = entries.get(("expected", prefix + "ff_out.input"))
        if source is None:
            raise ValueError("R4 observation requires Down input shape")
        require("expected", prefix + "ff_out.r4_output", "bf16_raw", "<u2", source["shape"])


def compare_reference_data(index: Path, actual_root: Path, names: list[str] | None = None,
                    *, payload_root: Path | None = None) -> dict:
    metadata = json.loads(index.read_text())
    if not supported_layer_index(metadata):
        raise ValueError("unsupported numerical reference_data schema")
    validate_numeric_capabilities(metadata)
    available = {entry["name"] for entry in metadata["tensors"] if entry["role"] == "expected"}
    if names is not None and (not names or set(names) - available):
        raise ValueError("requested checkpoint is empty or absent from reference_data")
    results = []
    for entry in metadata["tensors"]:
        if entry["role"] != "expected":
            continue
        if names is not None and entry["name"] not in names:
            continue
        actual_path = actual_root / f"{entry['name']}.bin"
        if not actual_path.is_file() or actual_path.stat().st_size != entry["byte_count"]:
            results.append(dict(name=entry["name"], match=False, reason="missing file or byte count mismatch"))
            continue
        expected_path = tensor_path(entry, index, payload_root)
        if expected_path.stat().st_size != entry["byte_count"]:
            raise ValueError(f"corrupt expected tensor: {expected_path}")
        dtype, shape = np.dtype(entry["dtype"]), tuple(entry["shape"])
        expected = np.fromfile(expected_path, dtype=dtype).reshape(shape)
        actual = np.fromfile(actual_path, dtype=dtype).reshape(shape)
        results.append(dict(name=entry["name"], **compare_arrays(expected, actual)))
    if not results:
        raise ValueError("reference_data has no expected tensors")
    return dict(match=all(item["match"] for item in results),
                scope="selected_checkpoints" if names is not None else
                    "all_forward_postprocess_checkpoints" if metadata["schema"] == FORWARD_POSTPROCESS_SCHEMA else
                    "all_layer_checkpoints",
                tensors=results)


def inspect_reference_data(index: Path, *, payload_root: Path | None = None) -> dict:
    metadata = json.loads(index.read_text())
    if not supported_layer_index(metadata):
        raise ValueError("inspect requires a numerical reference_data index")
    validate_numeric_capabilities(metadata)
    for entry in metadata["tensors"]:
        expected_bytes = int(np.prod(entry["shape"])) * np.dtype(entry["dtype"]).itemsize
        if expected_bytes != entry["byte_count"] or tensor_path(entry, index, payload_root).stat().st_size != expected_bytes:
            raise ValueError(f"reference_data shape/dtype/file size mismatch: {entry['name']}")
    if metadata["schema"] == FORWARD_POSTPROCESS_SCHEMA:
        return dict(status="INPUT_VALIDATED", seed=metadata["seed"],
                    tokens=metadata["tokens"], hidden=metadata["hidden"],
                    vocabulary=metadata["vocabulary"],
                    expected_checkpoints=sum(e["role"] == "expected" for e in metadata["tensors"]),
                    payload_bytes=sum(e["byte_count"] for e in metadata["tensors"]))
    bits_entry = next(e for e in metadata["tensors"] if e["role"] == "input" and e["name"] == "activation_bits")
    bits = np.fromfile(tensor_path(bits_entry, index, payload_root), dtype=bits_entry["dtype"])
    if len(bits) != metadata["tokens"] or not np.isin(bits, [4, 8]).all():
        raise ValueError("reference_data activation_bits do not match the layer shape")
    return dict(status="INPUT_VALIDATED", seed=metadata["seed"],
                tokens=metadata["tokens"], sequence=metadata["sequence"],
                hidden=metadata["hidden"], ffn=metadata["ffn"], heads=metadata["heads"],
                a4_token_count=int((bits == 4).sum()), a8_token_count=int((bits == 8).sum()),
                activation_bits=bits.tolist(), activation_slice_units=int((bits // 4).sum()),
                expected_checkpoints=sum(e["role"] == "expected" for e in metadata["tensors"]),
                payload_bytes=sum(e["byte_count"] for e in metadata["tensors"]))


def make_control_boundaries(device: str = "cpu", *, closeouts_only: bool = False) -> dict:
    """Probe real generation transitions with explicit post-candidate inputs.

    A scripted model supplies cache state and candidate outputs. Selection
    and state transitions run through the production generator and PSME methods.
    """
    _load_algorithm()
    from types import SimpleNamespace
    from unittest.mock import patch
    from numerics import candidate as candidate_numeric
    from generation import engine as generation
    from generation.state import Feature2BlockState as DraftVerifyBlockState
    execution_device = torch.device(device)
    gpu = execution_device.type == "cuda"

    def packed(value):
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            data = tensor.view(torch.int16).to(torch.int32) & 0xffff
        elif tensor.dtype == torch.float32:
            data = tensor.view(torch.int32).to(torch.int64) & 0xffffffff
        else:
            data = tensor
        return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), raw=data.tolist())

    class ScriptedModel:
        def __init__(self, remask):
            self.device = execution_device
            self.config = SimpleNamespace(n_layers=1)
            self._LLaDAModel__cache = {}
            self.calls = 0
            self.remask = remask

        def __call__(self, input_ids, **kwargs):
            logits = torch.zeros(*input_ids.shape, 4, dtype=torch.bfloat16,
                                 device=execution_device)
            logits[..., 0] = 2
            if self.remask and self.calls == 1:
                logits[:, 0, 0], logits[:, 0, 1] = 0, 2
            past = kwargs.get("past_key_values")
            values = input_ids.float()[:, None, :, None]
            if past is None:
                cache = ((values.clone(), values.clone()),)
            else:
                positions = kwargs.get("kv_write_position_ids")
                query = kwargs.get("query_position_ids")
                if positions is None:
                    mask = kwargs.get("replace_position_kv", kwargs.get("replace_position"))
                    positions = mask[0].nonzero().flatten()[None, :]
                if query is None:
                    query = kwargs["replace_position"][0].nonzero().flatten()[None, :]
                local = torch.isin(query[0], positions[0])
                key, val = (item.clone() for item in past[0])
                key[:, :, positions[0]] = values[:, :, local]
                val[:, :, positions[0]] = values[:, :, local]
                cache = ((key, val),)
            region = self._LLaDAModel__cache.get("attn_monitor_query_range")
            if region is not None:
                start, end = (int(x) for x in region.tolist())
                queries = torch.arange(input_ids.shape[1], device=execution_device) if past is None else query[0]
                queries = queries[(queries >= start) & (queries < end)]
                self._LLaDAModel__cache["attn_monitor_current"] = dict(
                    query_positions=queries,
                    dependency_mean=torch.zeros(1, queries.numel(), end - start, dtype=torch.bfloat16,
                                                device=execution_device),
                    layer_reduction="max", layer_count=torch.tensor(1, device=execution_device))
            self.calls += 1
            return SimpleNamespace(logits=logits, past_key_values=cache)

    def probe(kind, budget_scale):
        records, captures, confirmation = [], [], []
        instances = []
        closeout = kind.startswith("closeout_")
        candidate_calls = 0
        config = dict(steps=4, gen_length=32, block_length=32, mask_id=3,
                      tau_high=0.90, tau_low=0.75, confirm_tau=0.75,
                      selective_a4_direct_tau=0.90, stability_bonus=0.05,
                      budget_scale=budget_scale, candidate_numeric_mode="bf16_lut",
                      packed_attention_refresh=True,
                      packed_attention_target_active_rows=32.0,
                      packed_attention_initial_burst_rows=0.0,
                      suppressed_candidate_token_ids=(3,))
        if closeout:
            config.update(steps=1, tail_confirmation_policy="none", tau_high_tail=None,
                          packed_attention_force_full_current=False)

        def initialize(state, *args, **kwargs):
            state_initialize(state, *args, **kwargs)
            instances.append(state)

        def state_snapshot(state):
            result = {name: packed(getattr(state, name)) for name in
                      ("state", "last_top1", "precision_age", "commit_origin")}
            result["row_bits"] = packed(state.row_bits(maturity_age=3, policy="original"))
            return result

        def inject(logits):
            nonlocal candidate_calls
            token = logits.argmax(dim=-1)
            top = logits.gather(-1, token[..., None]).squeeze(-1)
            confidence = torch.ones_like(top).float()
            if candidate_calls == 0:
                if kind == "threshold":
                    confidence[:, :3] = torch.tensor([0x3f65, 0x3f66, 0x3f67],
                                                     dtype=torch.int16, device=execution_device).view(torch.bfloat16).float()
                elif kind == "suppression":
                    token[:, 0] = 3
                else:
                    confidence.fill_(float(torch.tensor(0.8).bfloat16()))
            candidate_calls += 1
            return (token, top.bfloat16(), confidence.bfloat16()) if gpu else (token, top.float(), confidence)

        def select(*args, **kwargs):
            scores = []
            selected = generation_select(*args, score_observer=lambda value: scores.append(packed(value)), **kwargs)
            current = captures[-1]
            records.append(dict(capture_index=current["capture_index"],
                                block_index=current["block_index"], step_index=current["step_index"],
                                prediction_positions=current["prediction_positions"],
                                score=scores[0],
                                admission_mask=packed(args[0]), high=packed(args[1]),
                                stable_low=packed(args[2]), stable=packed(args[3]),
                                action_confidence=packed(args[4]), selected=packed(selected),
                                scheduled_quota=packed(args[5]), **kwargs,
                                confirmation=confirmation.pop() if confirmation else None))
            return selected

        def admit(state, tokens, proposal, selected, high, stable_low):
            before = packed(state.state)
            result = state_admit(state, tokens, proposal, selected, high, stable_low)
            records[-1].update(state_before_admit=before, state_after_admit=packed(state.state),
                               proposal=packed(proposal), direct_condition=packed(high),
                               direct=packed(result.direct_locked), tokens_after_admit=packed(tokens))
            return result

        def confirm(state, tokens, proposal, probability, **kwargs):
            result = state_confirm(state, tokens, proposal, probability, **kwargs)
            confirmation.append(dict(selected_token_probability=packed(probability),
                                     confirmed=packed(result.locked), remasked=packed(result.remasked)))
            if closeout and captures[-1]["forward_kind"] == "local_confirmation":
                captures[-1]["confirmation"] = confirmation[-1]
            return result

        def capture(event):
            captures.append(dict(capture_index=event.capture_index,
                                 block_index=event.block_index, step_index=event.step_index,
                                 forward_kind=event.forward_kind,
                                 state_at_forward_start=packed(event.block_state_before),
                                 tokens_at_forward_start=packed(event.tokens_before),
                                 activation_bits=packed(event.row_bits),
                                 token_positions=packed(event.input_positions),
                                 prediction_positions=packed(event.prediction_positions),
                                 prediction_mask=packed(event.prediction_mask),
                                 refresh_positions=packed(event.refresh_positions),
                                 raw_confidence=packed(event.teacher_top1_confidence),
                                 action_confidence=packed(event.teacher_top1_action_confidence)))
            if closeout:
                if len(instances) != 1:
                    raise RuntimeError("single-block closeout must have exactly one state object")
                captures[-1].update(state_fields_at_forward_start=state_snapshot(instances[0]),
                                    proposal=packed(event.teacher_top1_token_ids))

        def forward_end(event):
            if event["forward_kind"] in ("local_confirmation", "local_forced_finish"):
                return  # Tail transitions are recorded by the closeout observer below.
            matches = [c for c in captures if
                       (c["block_index"], c["step_index"], c["forward_kind"]) ==
                       (event["block_index"], event["step_index"], event["forward_kind"])]
            if len(matches) != 1:
                raise RuntimeError("normal forward must identify exactly one capture")
            current = matches[0]
            trace = event["trace"]
            observation = dict(
                capture_index=current["capture_index"], block_index=event["block_index"],
                step_index=event["step_index"], forward_kind=event["forward_kind"],
                prediction_positions=current["prediction_positions"],
                phase="normal_forward_end_before_block_exit_tail_closure",
                **{name: None if event[name] is None else packed(event[name]) for name in
                   ("decision_positions", "state", "tokens", "last_top1", "precision_age", "commit_origin",
                    "cache_refresh_due")},
                next_current_activation_bits=(
                    None if event["next_current_row_bits"] is None
                    else packed(event["next_current_row_bits"])),
                cache_refresh_due_positions_before=list(trace.cache_refresh_due_positions_before),
                cache_refresh_due_positions_after=list(trace.cache_refresh_due_positions_after))
            matching_decisions = [d for d in records if d["capture_index"] == current["capture_index"]]
            if len(matching_decisions) != 1:
                raise RuntimeError("normal forward must identify exactly one decision")
            matching_decisions[0]["forward_end"] = observation
            matching_decisions[0]["decision_positions"] = observation["decision_positions"]

        generation_select = generation.select_admissions
        state_admit, state_confirm = DraftVerifyBlockState.admit, DraftVerifyBlockState.confirm
        state_initialize = DraftVerifyBlockState.__init__
        reducer = inject if gpu else candidate_numeric.streaming_candidate_bf16
        with patch.object(candidate_numeric, "streaming_candidate_bf16_oracle", side_effect=inject),\
             patch.object(generation, "streaming_candidate_bf16", side_effect=reducer),\
             patch.object(generation, "select_admissions", side_effect=select),\
             patch.object(DraftVerifyBlockState, "__init__", initialize),\
             patch.object(DraftVerifyBlockState, "admit", admit),\
             patch.object(DraftVerifyBlockState, "confirm", confirm):
            output_tokens, model_evaluations, trace, _ = generation.generate(
                ScriptedModel(kind in ("remask", "closeout_force")), torch.tensor([[2]], device=execution_device),
                row_precision_context=RowPrecisionContext(),
                state_capture_callback=capture, forward_end_callback=forward_end, **config)
        transitions = []
        for t in trace:
            matching = [c for c in captures if
                        (c["block_index"], c["step_index"], c["forward_kind"]) ==
                        (t.block_index, t.step_index, t.forward_kind)]
            if len(matching) != 1:
                raise RuntimeError("transition must identify exactly one capture")
            transitions.append(dict(capture_index=matching[0]["capture_index"],
                                    block_index=t.block_index, step_index=t.step_index,
                                    prediction_positions=matching[0]["prediction_positions"],
                                    direct=list(t.direct_locked_positions),
                                    selective_direct=list(t.selective_a4_direct_positions),
                                    confirmed=list(t.confirmed_positions), remasked=list(t.remasked_positions),
                                    tail_bypassed=list(t.tail_bypassed_positions)))
        if closeout:
            closeout_kinds = [c["forward_kind"] for c in captures]
            required = ["local_confirmation"] + (["local_forced_finish"] if kind == "closeout_force" else [])
            if any(closeout_kinds.count(name) != 1 for name in required):
                raise RuntimeError(f"closeout probe did not exercise {required}: {closeout_kinds}")
            return dict(kind=kind, config=config, model_evaluations=model_evaluations,
                        captures=captures, decisions=records,
                        transitions=transitions, final_state=state_snapshot(instances[0]),
                        final_tokens=packed(output_tokens[:, 1:]),
                        final_cache_refresh_due_positions=list(trace[-1].cache_refresh_due_positions_after),
                        final_phase=trace[-1].forward_kind)
        return dict(kind=kind, config=config,
                    model_evaluations=model_evaluations, captures=captures, decisions=records,
                    transitions=transitions)

    if closeouts_only:
        return dict(schema="supra-psme-block-closeouts/v1", device=str(execution_device),
                    source="actual generator and PSME state; synthetic model/cache/candidate inputs",
                    draft_verify=[probe("closeout_confirm", 20), probe("closeout_force", 20)], dynamic_scale=[])

    dynamic_scale = []
    for maximum_raw in (0x0000, 0x0001, 0x0003, 0x007f, 0x0080, 0x0081, 0x3f80, 0x7f7f):
        values = torch.tensor([maximum_raw, maximum_raw | 0x8000, 0],
                              dtype=torch.int32).to(torch.int16).view(torch.bfloat16)[None, :]
        for bits in (4, 8):
            quantized = quantize_activation_per_activation_bits_bf16(values, torch.tensor([bits], dtype=torch.int8))
            dynamic_scale.append(dict(input_bf16_raw=packed(values), activation_bits=bits,
                                      scale_bf16_raw=raw_array(quantized.scale_bf16).tolist(),
                                      codes=quantized.codes.tolist()))
    return dict(schema="supra-psme-state-boundaries/v2",
                source=("actual CUDA generation with injected BF16 candidate values"
                        if gpu else "actual CPU generation with injected candidate values"),
                dynamic_scale=dynamic_scale,
                draft_verify=[probe("threshold", budget) for budget in (20, 26)]
                      + [probe("suppression", 26), probe("confirm", 26), probe("remask", 26)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "control-loop", "forward-postprocess", "compare", "boundaries", "closeouts", "inspect"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--actual-root", type=Path)
    parser.add_argument("--payload-root", type=Path,
                        help="root for relative tensor paths; defaults to the index directory")
    parser.add_argument("--tensor", action="append", help="compare only this named checkpoint; repeatable")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--sequence", type=int, default=17)
    parser.add_argument("--vocabulary", type=int, default=64)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--a8-tokens", type=int,
                           help="A8 token count; remaining tokens are A4, assigned by seed")
    precision.add_argument("--activation-bits", type=int, choices=(4, 8), nargs="+",
                           help="explicit precision for each token in logical input order")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--silu-reference-data", type=Path,
                        help="use the SiLU implementation with an exported coefficient table")
    parser.add_argument("--token-metadata", type=Path,
                        help="control-loop: token metadata for one selection round")
    parser.add_argument("--deployment-fields", type=Path,
                        help="single layer/control-loop: DDR scalar configuration with all seven clipping ratios")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                        help="device for synthetic generation boundaries only")
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--kv-write-tokens", type=int, nargs="*",
                        help="local query tokens whose new K/V is visible during this layer; default all")
    parser.add_argument("--cache-commit-tokens", type=int, nargs="*",
                        help="local query tokens persisted after this layer; default all visible writes")
    parser.add_argument("--boundary-config", type=Path,
                        help="boundary selection arguments; exports full Layer0 and selected Layer1")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    explicit_cache_sets = args.kv_write_tokens is not None or args.cache_commit_tokens is not None
    if explicit_cache_sets and (args.action != "generate" or args.layers != 1):
        parser.error("explicit cache sets currently apply to single-layer generate only")
    if args.action == "inspect":
        print(json.dumps(inspect_reference_data(args.index, payload_root=args.payload_root)))
        return
    if args.action in ("boundaries", "closeouts"):
        if args.index.exists():
            parser.error(f"index already exists: {args.index}; choose a new --index path")
        _load_algorithm()
        torch.set_num_threads(args.threads)
        result = make_control_boundaries(device=args.device, closeouts_only=args.action == "closeouts")
        args.index.parent.mkdir(parents=True, exist_ok=True)
        args.index.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(index=str(args.index), cases=len(result["draft_verify"]),
                              dynamic_scale_cases=len(result["dynamic_scale"]))))
        return
    if args.action == "compare":
        if args.actual_root is None:
            parser.error("compare requires --actual-root")
        result = compare_reference_data(args.index, args.actual_root, args.tensor, payload_root=args.payload_root)
        print(json.dumps(result, default=int))
        raise SystemExit(0 if result["match"] else 1)
    if args.output is None:
        parser.error("generate requires --output below SUPRA_ALGORITHM_ROOT")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if ALGO_ROOT is None:
        parser.error("set SUPRA_ALGORITHM_ROOT before generating reference_data payload")
    if not _within(args.output.resolve(), ALGO_ROOT):
        parser.error(f"--output must be below {ALGO_ROOT}")
    if args.output.exists() or args.index.exists():
        parser.error("reference_data output/index already exists")
    _load_algorithm()
    torch.set_num_threads(args.threads)
    ratios = None
    if args.deployment_fields is not None:
        if args.action not in ("generate", "control-loop") or args.layers != 1 or args.boundary_config:
            parser.error("--deployment-fields requires single-layer generate or control-loop")
        deployment = json.loads(args.deployment_fields.read_text())
        hardware_names = ("query", "key", "value", "attention_output", "ffn_gate", "ffn_up", "ffn_down")
        ratios = {}
        for name, hardware_name in zip(LINEARS, hardware_names):
            raw = deployment["layer"][hardware_name+"_clip_ratio_bf16"]
            if type(raw) is not int or not 0 < raw <= 0x3f80:
                raise ValueError(f"invalid BF16 clipping input: {hardware_name}")
            ratios[name] = float(torch.tensor(raw, dtype=torch.int16).view(torch.bfloat16))
    if args.action == "control-loop":
        if args.token_metadata is None or args.deployment_fields is None or args.silu_reference_data is None:
            parser.error("control-loop requires --token-metadata, --deployment-fields and --silu-reference-data")
        inputs, expected, metadata = make_control_loop_reference_data(
            args.token_metadata.read_bytes(), seed=args.seed, sequence=args.sequence,
            silu_reference_data=args.silu_reference_data, a4_clip_ratios=ratios)
        export_reference_data(inputs, expected, metadata, args.output, args.index)
        print(json.dumps(dict(index=str(args.index), expected_checkpoints=len(expected))))
        return
    if args.boundary_config is not None:
        if args.action != "generate" or args.layers != 1 or explicit_cache_sets:
            parser.error("--boundary-config requires generate without --layers or explicit cache sets")
        deep_index = args.index.with_name(args.index.stem + ".layer1.json")
        if deep_index.exists():
            parser.error("boundary Layer1 index already exists")
        options = json.loads(args.boundary_config.read_text())
        stages = make_boundary_reference_data(sequence=args.sequence, seed=args.seed,
            activation_bits=args.activation_bits, silu_reference_data=args.silu_reference_data, **options)
        stages[0][2]["continuation_index"] = deep_index.name
        stages[1][2]["preceding_index"] = args.index.name
        for layer, (inputs, expected, metadata) in enumerate(stages):
            export_reference_data(inputs, expected, metadata, args.output / f"layer{layer}",
                args.index if layer == 0 else deep_index)
        print(json.dumps(dict(status="REFERENCE_DATA_GENERATED", index=str(args.index),
            continuation_index=str(deep_index), selected_positions=stages[0][2]["boundary_scout"]["deep_positions"])))
        return
    if args.action == "forward-postprocess":
        inputs, expected, metadata = make_forward_postprocess_reference_data(
            tokens=args.tokens if args.tokens is not None else 16,
            vocabulary=args.vocabulary, seed=args.seed)
        export_reference_data(inputs, expected, metadata, args.output, args.index)
        print(json.dumps(dict(index=str(args.index), expected_checkpoints=len(expected))))
        return
    tokens = args.tokens if args.tokens is not None else (len(args.activation_bits) if args.activation_bits else 3)
    reference_data_args = dict(
        tokens=tokens, sequence=args.sequence,
        a8_token_count=args.a8_tokens if args.a8_tokens is not None else 1,
        activation_bits=args.activation_bits, seed=args.seed, scattered=not args.contiguous)
    if args.layers == 1:
        inputs, expected, metadata = make_reference_data(**reference_data_args,
            kv_write_token_indices=args.kv_write_tokens, cache_commit_token_indices=args.cache_commit_tokens,
            silu_reference_data=args.silu_reference_data, a4_clip_ratios=ratios)
    elif args.layers > 1:
        if args.silu_reference_data:
            parser.error("--silu-reference-data currently requires one layer")
        inputs, expected, metadata = make_multilayer_reference_data(
            layers=args.layers, **reference_data_args)
    else:
        parser.error("--layers must be positive")
    export_reference_data(inputs, expected, metadata, args.output, args.index)
    print(json.dumps(dict(status="REFERENCE_DATA_GENERATED", index=str(args.index),
                          observed_tensors=len(expected), **metadata)))


if __name__ == "__main__":
    main()
