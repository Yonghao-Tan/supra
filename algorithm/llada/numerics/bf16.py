"""PyTorch primitives defining the quantized model's numerical boundaries.

Operations include BF16 materialization, per-row activation quantization,
RMSNorm, RoPE, Softmax and SiLU.
There is no zero point or zero-correction path.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

INT8_MIN = -127
INT8_MAX = 127
_SOFTMAX_LUT_CACHE: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class Bf16QuantizedTensor:
    codes: torch.Tensor
    scale_bf16: torch.Tensor
    scale_mode: str


def bf16(value: torch.Tensor | float) -> torch.Tensor:
    """Round a scalar or tensor to BF16, returning FP32 values with BF16 bits."""
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.float32)
    return value.to(torch.float32).to(torch.bfloat16).to(torch.float32)


def bf16_add(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return bf16(bf16(lhs) + bf16(rhs))


def bf16_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return bf16(bf16(lhs) * bf16(rhs))


def _scale_from_maximum(maximum: torch.Tensor) -> torch.Tensor:
    scale = bf16(maximum.to(torch.float32) / float(INT8_MAX))
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def quantize_per_row_bf16(values: torch.Tensor) -> Bf16QuantizedTensor:
    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError(
            f"values must be rank-2 floating tensor, got {tuple(values.shape)} {values.dtype}"
        )
    rounded = bf16(values)
    scale = _scale_from_maximum(rounded.detach().abs().amax(dim=1))
    codes = torch.clamp(
        torch.round(rounded / scale.unsqueeze(1)), INT8_MIN, INT8_MAX
    ).to(torch.int8)
    return Bf16QuantizedTensor(codes=codes, scale_bf16=scale, scale_mode="per_row")


def quantize_mixed_rows_bf16(
    values: torch.Tensor, row_modes: torch.Tensor
) -> Bf16QuantizedTensor:
    """Quantize each row as W4A4 mode 0 or W4A8 mode 1."""
    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError(
            f"values must be rank-2 floating tensor, got {tuple(values.shape)} {values.dtype}"
        )
    if row_modes.shape != values.shape[:1] or row_modes.device != values.device:
        raise ValueError("row_modes must contain one value per row on the same device")
    if not bool(torch.all((row_modes == 0) | (row_modes == 1))):
        raise ValueError("row_modes values must be W4A4(0) or W4A8(1)")
    rounded = bf16(values)
    code_max = torch.where(row_modes == 0, 7, 127).to(torch.float32)
    maximum = rounded.detach().abs().amax(dim=1)
    scale = bf16(maximum / code_max)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(
        torch.round(rounded / scale.unsqueeze(1)),
        -code_max.unsqueeze(1),
        code_max.unsqueeze(1),
    ).to(torch.int8)
    return Bf16QuantizedTensor(codes, scale, "per_row")


def bf16_reduce_sum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Fixed pairwise BF16 reduction tree, including an explicit zero pad."""
    reduced = values.to(torch.float32).movedim(dim, -1)
    size = reduced.shape[-1]
    next_power_of_two = 1 << (size - 1).bit_length()
    if next_power_of_two != size:
        pad = torch.zeros(
            *reduced.shape[:-1],
            next_power_of_two - size,
            device=reduced.device,
            dtype=reduced.dtype,
        )
        reduced = torch.cat((reduced, pad), dim=-1)
    while reduced.shape[-1] > 1:
        reduced = bf16_add(reduced[..., 0::2], reduced[..., 1::2])
    return reduced[..., 0]


def fast_rsqrt_bf16(values: torch.Tensor) -> torch.Tensor:
    if torch.any(values <= 0):
        raise ValueError("fast_rsqrt_bf16 requires positive values")
    x = bf16(values)
    bits = x.contiguous().view(torch.int32)
    estimate = (
        torch.full_like(bits, 1597463007) - torch.bitwise_right_shift(bits, 1)
    ).view(torch.float32)
    y = bf16(estimate)
    correction = bf16_add(
        torch.full_like(y, 1.5), -bf16_mul(bf16(0.5 * x), bf16_mul(y, y))
    )
    return bf16_mul(y, correction)


def rms_norm_bf16(
    values: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    *,
    gemma_style: bool = False,
) -> torch.Tensor:
    squared = bf16_mul(values, values)
    total = bf16_reduce_sum(squared, dim=-1).unsqueeze(-1)
    mean = bf16(total / float(values.shape[-1]))
    normalized = bf16_mul(
        values, fast_rsqrt_bf16(bf16_add(mean, torch.full_like(mean, eps)))
    )
    if weight is None:
        return normalized
    multiplier = bf16(1.0 + weight) if gemma_style else bf16(weight)
    return bf16_mul(normalized, multiplier)


_SILU_BREAKPOINTS = (
    -8.0,
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
)
_SILU_SLOPES = (
    -0.006072998046875,
    -0.028564453125,
    -0.0703125,
    -0.09619140625,
    -0.0703125,
    0.0093994140625,
    0.16015625,
    0.376953125,
    0.62109375,
    0.83984375,
    0.9921875,
    1.0703125,
    1.09375,
    1.0703125,
    1.03125,
    1.0078125,
)
_SILU_INTERCEPTS = (
    -0.05126953125,
    -0.1865234375,
    -0.353515625,
    -0.431640625,
    -0.37890625,
    -0.259765625,
    -0.10888671875,
    -0.000293731689453125,
    0.0,
    -0.10888671875,
    -0.26171875,
    -0.37890625,
    -0.42578125,
    -0.353515625,
    -0.197265625,
    -0.061767578125,
)


def silu_pwl_table_tensors(
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the PWL table with BF16 storage values."""
    return (
        bf16(torch.tensor(_SILU_BREAKPOINTS, dtype=torch.float32, device=device)),
        bf16(torch.tensor(_SILU_SLOPES, dtype=torch.float32, device=device)),
        bf16(torch.tensor(_SILU_INTERCEPTS, dtype=torch.float32, device=device)),
    )


def _silu_pwl_bf16(
    values: torch.Tensor, slopes: tuple[float, ...], intercepts: tuple[float, ...]
) -> torch.Tensor:
    if len(slopes) != len(intercepts) or len(slopes) + 1 != len(_SILU_BREAKPOINTS):
        raise ValueError("SiLU PWL table dimensions are inconsistent")
    result = torch.zeros_like(values, dtype=torch.float32)
    rounded = bf16(values)
    for index in range(len(slopes)):
        (lower, upper) = (_SILU_BREAKPOINTS[index], _SILU_BREAKPOINTS[index + 1])
        mask = (rounded >= lower) & (rounded < upper)
        result = torch.where(
            mask,
            bf16_add(
                bf16_mul(rounded, torch.full_like(rounded, slopes[index])),
                torch.full_like(rounded, intercepts[index]),
            ),
            result,
        )
    result = torch.where(rounded <= -8.0, torch.zeros_like(result), result)
    return torch.where(rounded >= 8.0, rounded, result)


def silu_pwl_bf16(values: torch.Tensor) -> torch.Tensor:
    return _silu_pwl_bf16(values, _SILU_SLOPES, _SILU_INTERCEPTS)


def rope_apply_bf16(
    values: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """Apply the duplicated-half LLaDA RoPE table to [..., T, 128]."""
    if values.shape[-1] != 128 or sin.shape != cos.shape or sin.shape[-1] != 128:
        raise ValueError("RoPE requires matching [..., 128] values and tables")
    half = values.shape[-1] // 2
    rotated = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
    return bf16_add(bf16_mul(values, cos), bf16_mul(rotated, sin))


def softmax_lut_bf16(scores: torch.Tensor) -> torch.Tensor:
    """BF16 softmax with a 256-entry exp LUT on [-16, 0] and fixed pairwise sum."""
    if scores.ndim < 1:
        raise ValueError("softmax scores must have a last dimension")
    rounded_scores = bf16(scores)
    valid_keys = ~torch.isneginf(rounded_scores)
    if not bool(valid_keys.any(dim=-1).all()):
        raise ValueError("softmax requires at least one valid key per query")
    row_max = rounded_scores.max(dim=-1, keepdim=True).values
    delta = torch.clamp(bf16_add(rounded_scores, -row_max), min=-16.0, max=0.0)
    index = torch.clamp(
        torch.round((delta + 16.0) * (255.0 / 16.0)).to(torch.long), 0, 255
    )
    cache_key = (scores.device.type, scores.device.index)
    cached = _SOFTMAX_LUT_CACHE.get(cache_key)
    if cached is None:
        cached = (
            bf16(
                torch.exp(
                    torch.linspace(
                        -16.0, 0.0, 256, device=scores.device, dtype=torch.float32
                    )
                )
            ),
            bf16(
                1.0
                / torch.linspace(
                    1.0, 2.0, 256, device=scores.device, dtype=torch.float32
                )
            ),
        )
        _SOFTMAX_LUT_CACHE[cache_key] = cached
    (lut, reciprocal_lut) = cached
    exponent = torch.where(valid_keys, lut[index], torch.zeros_like(rounded_scores))
    row_sum = bf16_reduce_sum(exponent, dim=-1).unsqueeze(-1)
    exponent_bits = torch.floor(torch.log2(row_sum)).to(torch.int32)
    mantissa = row_sum * torch.pow(
        torch.tensor(2.0, device=scores.device), -exponent_bits.to(torch.float32)
    )
    reciprocal_index = torch.clamp(
        torch.round((mantissa - 1.0) * 255.0).to(torch.long), 0, 255
    )
    reciprocal = bf16(
        reciprocal_lut[reciprocal_index]
        * torch.pow(
            torch.tensor(2.0, device=scores.device), -exponent_bits.to(torch.float32)
        )
    )
    return bf16_mul(exponent, reciprocal)


def _row_modes_from_bits(row_bits: torch.Tensor) -> torch.Tensor:
    if row_bits.dtype != torch.int8 or not bool(
        torch.all((row_bits == 4) | (row_bits == 8))
    ):
        raise ValueError("row_bits must be torch.int8 values 4 or 8")
    return torch.where(row_bits == 4, 0, 1).to(torch.uint8)


def quantize_activation_per_row_bf16(
    values: torch.Tensor, activation_bits: int
) -> Bf16QuantizedTensor:
    if activation_bits not in {4, 8}:
        raise ValueError("activation_bits must be 4 or 8")
    row_bits = torch.full(
        (values.shape[0],), activation_bits, dtype=torch.int8, device=values.device
    )
    return quantize_activation_per_row_bits_bf16(values, row_bits)


def quantize_activation_per_row_bits_bf16(
    values: torch.Tensor, row_bits: torch.Tensor
) -> Bf16QuantizedTensor:
    if row_bits.shape != values.shape[:1] or row_bits.device != values.device:
        raise ValueError("row_bits must contain one value per row on the same device")
    return quantize_mixed_rows_bf16(values, _row_modes_from_bits(row_bits))
