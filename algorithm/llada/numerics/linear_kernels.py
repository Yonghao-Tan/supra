"""Exact CUDA kernels for the hardware-numeric Linear boundaries."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import torch

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as tl_math
    except ImportError:
        from triton.language import math as tl_math
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    tl_math = None
    _TRITON_AVAILABLE = False
if _TRITON_AVAILABLE:

    @triton.jit
    def _ieee_divide_fp32(numerator, denominator):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [numerator, denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _quantize_per_row_bf16_kernel(
        values_ptr,
        codes_ptr,
        scales_ptr,
        row_bits_ptr,
        rows,
        columns: tl.constexpr,
        values_row_stride,
        values_column_stride,
        row_bits_stride,
        codes_row_stride,
        QUANT_MAX: tl.constexpr,
        USE_ROW_BITS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        valid = (row < rows) & (offsets < columns)
        values = tl.load(
            values_ptr + row * values_row_stride + offsets * values_column_stride,
            mask=valid, other=0.0
        ).to(tl.bfloat16).to(tl.float32)
        maximum = tl.max(tl.abs(values), axis=0)
        if USE_ROW_BITS:
            row_bits = tl.load(row_bits_ptr + row * row_bits_stride, mask=row < rows, other=8)
            quant_max = tl.where(row_bits == 4, 7.0, 127.0)
        else:
            quant_max = QUANT_MAX * 1.0
        scale = _ieee_divide_fp32(maximum, quant_max).to(tl.bfloat16).to(tl.float32)
        scale = tl.where(scale == 0.0, 1.0, scale)
        normalized = _ieee_divide_fp32(values, scale)
        rounded = tl_math.float2int_rn(normalized)
        rounded = tl.maximum(-quant_max, tl.minimum(quant_max, rounded))
        tl.store(
            codes_ptr + row * codes_row_stride + offsets,
            rounded,
            mask=offsets < columns,
        )
        tl.store(scales_ptr + row, scale, mask=row < rows)

    @triton.jit
    def _linear_rescale_bf16_kernel(
        sums_ptr,
        activation_scales_ptr,
        weight_scales_ptr,
        output_ptr,
        elements,
        activation_scale_stride,
        weight_scale_stride,
        columns: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        rows = offsets // columns
        columns_offsets = offsets - rows * columns
        sums = tl.load(sums_ptr + offsets, mask=valid, other=0).to(tl.float32)
        activation_scales = tl.load(
            activation_scales_ptr + rows * activation_scale_stride, mask=valid, other=1.0
        ).to(tl.float32)
        weight_scales = tl.load(
            weight_scales_ptr + columns_offsets * weight_scale_stride, mask=valid, other=1.0
        ).to(tl.float32)
        first_bf16 = (sums * activation_scales).to(tl.bfloat16).to(tl.float32)
        output_bf16 = (first_bf16 * weight_scales).to(tl.bfloat16)
        tl.store(output_ptr + offsets, output_bf16, mask=valid)


@dataclass
class LinearNumericWorkspace:
    """Reusable A8 code and scale storage for sequential Linear calls."""

    code_buffers: dict[tuple[str, int | None], torch.Tensor] = field(
        default_factory=dict
    )
    scale_buffers: dict[tuple[str, int | None], torch.Tensor] = field(
        default_factory=dict
    )

    def acquire_codes(
        self, rows: int, columns: int, device: torch.device
    ) -> torch.Tensor:
        required = rows * columns
        key = (device.type, device.index)
        buffer = self.code_buffers.get(key)
        if buffer is None or buffer.numel() < required:
            buffer = torch.empty(required, device=device, dtype=torch.int8)
            self.code_buffers[key] = buffer
        return buffer[:required].view(rows, columns)

    def acquire_scales(self, rows: int, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        buffer = self.scale_buffers.get(key)
        if buffer is None or buffer.numel() < rows:
            buffer = torch.empty(rows, device=device, dtype=torch.float32)
            self.scale_buffers[key] = buffer
        return buffer[:rows]


def triton_linear_numeric_available() -> bool:
    return (
        _TRITON_AVAILABLE
        and os.environ.get("SUPRA_DISABLE_TRITON_LINEAR_NUMERIC") != "1"
    )


def quantize_per_row_bf16_cuda(
    values: torch.Tensor, workspace: LinearNumericWorkspace, *, activation_bits: int = 8
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize rank-2 CUDA values and zero-fill rows required by ``_int_mm``."""
    if values.ndim != 2 or not torch.is_floating_point(values) or (not values.is_cuda):
        raise ValueError(
            "fused per-row activation quantization requires a rank-2 floating CUDA tensor"
        )
    if activation_bits not in {4, 8}:
        raise ValueError("activation_bits must be 4 or 8")
    if not triton_linear_numeric_available():
        raise RuntimeError("Triton Linear numeric kernels are unavailable")
    (rows, columns) = (int(extent) for extent in values.shape)
    padded_rows = max(32, (rows + 15) // 16 * 16)
    codes = workspace.acquire_codes(padded_rows, columns, values.device)
    scales = workspace.acquire_scales(rows, values.device)
    block_k = triton.next_power_of_2(columns)
    _quantize_per_row_bf16_kernel[padded_rows,](
        values,
        codes,
        scales,
        codes,
        rows,
        columns=columns,
        values_row_stride=values.stride(0),
        values_column_stride=values.stride(1),
        row_bits_stride=0,
        codes_row_stride=codes.stride(0),
        QUANT_MAX=7 if activation_bits == 4 else 127,
        USE_ROW_BITS=False,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return (codes, scales, rows)


def quantize_per_row_bits_bf16_cuda(
    values: torch.Tensor, row_bits: torch.Tensor, workspace: LinearNumericWorkspace
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize each CUDA row independently as A4 or A8."""
    if values.ndim != 2 or not torch.is_floating_point(values) or (not values.is_cuda):
        raise ValueError(
            "mixed per-row activation quantization requires a rank-2 floating CUDA tensor"
        )
    if (
        row_bits.shape != values.shape[:1]
        or row_bits.dtype != torch.int8
        or row_bits.device != values.device
    ):
        raise ValueError(
            "row_bits must be torch.int8 with one value per CUDA activation row"
        )
    if not triton_linear_numeric_available():
        raise RuntimeError("Triton Linear numeric kernels are unavailable")
    (rows, columns) = (int(extent) for extent in values.shape)
    padded_rows = max(32, (rows + 15) // 16 * 16)
    codes = workspace.acquire_codes(padded_rows, columns, values.device)
    scales = workspace.acquire_scales(rows, values.device)
    block_k = triton.next_power_of_2(columns)
    _quantize_per_row_bf16_kernel[padded_rows,](
        values,
        codes,
        scales,
        row_bits,
        rows,
        columns=columns,
        values_row_stride=values.stride(0),
        values_column_stride=values.stride(1),
        row_bits_stride=row_bits.stride(0),
        codes_row_stride=codes.stride(0),
        QUANT_MAX=127,
        USE_ROW_BITS=True,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return (codes, scales, rows)


def linear_rescale_bf16_cuda(
    sums: torch.Tensor,
    activation_scales_bf16: torch.Tensor,
    weight_scales_bf16: torch.Tensor,
) -> torch.Tensor:
    """Apply two BF16 rounding boundaries to an INT32 Linear sum."""
    if (
        sums.ndim != 2
        or sums.dtype != torch.int32
        or (not sums.is_cuda)
        or (not sums.is_contiguous())
    ):
        raise ValueError(
            "fused Linear rescale requires contiguous rank-2 CUDA INT32 sums"
        )
    (rows, columns) = (int(extent) for extent in sums.shape)
    if activation_scales_bf16.shape != (rows,) or weight_scales_bf16.shape != (
        columns,
    ):
        raise ValueError(
            "fused Linear rescale scale shapes do not match the accumulator"
        )
    output = torch.empty((rows, columns), device=sums.device, dtype=torch.bfloat16)
    elements = rows * columns
    block = 256
    _linear_rescale_bf16_kernel[triton.cdiv(elements, block),](
        sums,
        activation_scales_bf16,
        weight_scales_bf16,
        output,
        elements,
        activation_scale_stride=activation_scales_bf16.stride(0),
        weight_scale_stride=weight_scales_bf16.stride(0),
        columns=columns,
        BLOCK=block,
        num_warps=4,
    )
    return output


def accelerated_linear_bf16(
    values: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales_bf16: torch.Tensor,
    workspace: LinearNumericWorkspace,
    *,
    activation_bits: int = 8,
) -> torch.Tensor:
    """Run dynamic A4/A8, exact INT32 GEMM and two-stage BF16 rescale."""
    if os.environ.get("SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC") == "1" and (
        not triton_linear_numeric_available()
    ):
        raise RuntimeError(
            "SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC=1 but the Triton Linear kernels are unavailable"
        )
    (codes, activation_scales, rows) = quantize_per_row_bf16_cuda(
        values, workspace, activation_bits=activation_bits
    )
    sums = torch._int_mm(codes, weight_codes.transpose(0, 1))[:rows]
    return linear_rescale_bf16_cuda(sums, activation_scales, weight_scales_bf16)


def accelerated_mixed_a4a8_linear_bf16(
    values: torch.Tensor,
    row_bits: torch.Tensor,
    weight_codes: torch.Tensor,
    weight_scales_bf16: torch.Tensor,
    workspace: LinearNumericWorkspace,
) -> torch.Tensor:
    """Run per-row A4/A8 quantization, one INT32 GEMM, and BF16 rescale."""
    if os.environ.get("SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC") == "1" and (
        not triton_linear_numeric_available()
    ):
        raise RuntimeError(
            "SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC=1 but the Triton Linear kernels are unavailable"
        )
    (codes, activation_scales, rows) = quantize_per_row_bits_bf16_cuda(
        values, row_bits, workspace
    )
    sums = torch._int_mm(codes, weight_codes.transpose(0, 1))[:rows]
    return linear_rescale_bf16_cuda(sums, activation_scales, weight_scales_bf16)
