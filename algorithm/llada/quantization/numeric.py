"""Signed symmetric W4 numeric primitives for SpinQuant weight tensors."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from numerics.bf16 import bf16

W4_SCALE_MODES = {-1: "per_output_channel"}


@dataclass(frozen=True)
class SpinQuantW4Tensor:
    codes: torch.Tensor
    scale_bf16: torch.Tensor
    scale_mode: str = "per_output_channel"

    @property
    def group_size(self) -> int:
        for group_size, scale_mode in W4_SCALE_MODES.items():
            if self.scale_mode == scale_mode:
                return group_size
        raise ValueError(f"unsupported W4 scale mode {self.scale_mode}")


def pack_signed_int4(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.int8 or codes.ndim < 1:
        raise ValueError("signed INT4 codes must be an int8 tensor")
    if bool(torch.any(codes < -8)) or bool(torch.any(codes > 7)):
        raise ValueError("signed INT4 code outside [-8,7]")
    if codes.shape[-1] % 2:
        codes = torch.cat(
            (
                codes,
                torch.zeros(
                    *codes.shape[:-1], 1, dtype=torch.int8, device=codes.device
                ),
            ),
            dim=-1,
        )
    low = torch.bitwise_and(codes[..., 0::2].to(torch.int16), 15)
    high = torch.bitwise_left_shift(
        torch.bitwise_and(codes[..., 1::2].to(torch.int16), 15), 4
    )
    return torch.bitwise_or(low, high).to(torch.uint8).contiguous()


def unpack_signed_int4(packed: torch.Tensor, logical_k: int) -> torch.Tensor:
    if (
        packed.dtype != torch.uint8
        or logical_k <= 0
        or packed.shape[-1] != (logical_k + 1) // 2
    ):
        raise ValueError(
            f"invalid packed signed INT4 shape {tuple(packed.shape)} for K={logical_k}"
        )
    low = torch.bitwise_and(packed, 15).to(torch.int8)
    high = torch.bitwise_right_shift(packed, 4).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    output = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device
    )
    output[..., 0::2] = low
    output[..., 1::2] = high
    if logical_k & 1 and bool(torch.any(output[..., -1] != 0)):
        raise ValueError("nonzero high-nibble padding in packed signed INT4 tensor")
    return output[..., :logical_k].contiguous()


def _validate_group_size(columns: int, group_size: int) -> int:
    if group_size not in W4_SCALE_MODES:
        raise ValueError(
            f"W4 group size must be one of {tuple(W4_SCALE_MODES)}, got {group_size}"
        )
    return 1


def _expand_w4_scales(
    scale_bf16: torch.Tensor, *, rows: int, columns: int, group_size: int
) -> torch.Tensor:
    _validate_group_size(columns, group_size)
    expected_shape = (rows,)
    if tuple(scale_bf16.shape) != expected_shape:
        raise ValueError(
            f"W4 scale shape must be {expected_shape} for group size {group_size}, got {tuple(scale_bf16.shape)}"
        )
    return scale_bf16.to(torch.float32).unsqueeze(1)


def symmetric_w4_scale_bf16(
    weight: torch.Tensor, *, group_size: int = -1
) -> torch.Tensor:
    if weight.ndim != 2 or not torch.is_floating_point(weight):
        raise ValueError("GPTQ W4 weight must be a rank-2 floating tensor")
    (rows, columns) = weight.shape
    groups = _validate_group_size(int(columns), int(group_size))
    grouped = weight.detach().to(torch.float32).reshape(rows, groups, -1)
    scale = bf16(grouped.abs().amax(dim=2) * (2.0 / 15.0))
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return scale.squeeze(1)


def quantize_symmetric_w4(
    values: torch.Tensor, scale_bf16: torch.Tensor, *, group_size: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("W4 values must be a rank-2 tensor")
    expanded_scales = _expand_w4_scales(
        scale_bf16,
        rows=int(values.shape[0]),
        columns=int(values.shape[1]),
        group_size=int(group_size),
    )
    codes = quantize_symmetric_w4_codes(values, scale_bf16, group_size=group_size)
    dequantized = codes.to(torch.float32) * expanded_scales
    return (codes, dequantized)


def quantize_symmetric_w4_codes(
    values: torch.Tensor, scale_bf16: torch.Tensor, *, group_size: int = -1
) -> torch.Tensor:
    """Emit signed W4 codes without allocating an unused dequantized tensor."""
    if values.ndim != 2:
        raise ValueError("W4 values must be a rank-2 tensor")
    expanded_scales = _expand_w4_scales(
        scale_bf16,
        rows=int(values.shape[0]),
        columns=int(values.shape[1]),
        group_size=int(group_size),
    )
    return torch.clamp(
        torch.round(values.to(torch.float32) / expanded_scales), -8, 7
    ).to(torch.int8)


@dataclass(frozen=True)
class SpinQuantW8Tensor:
    codes: torch.Tensor
    scale_bf16: torch.Tensor


def quantize_symmetric_w8(values: torch.Tensor) -> SpinQuantW8Tensor:
    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError("W8 weight must be a floating rank-2 tensor")
    weight = values.detach().float()
    scale = (weight.abs().amax(dim=1) / 127.0).to(torch.bfloat16).float()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round(weight / scale.unsqueeze(1)), -127, 127).to(
        torch.int8
    )
    return SpinQuantW8Tensor(codes.cpu(), scale.cpu())
