"""Batched INT8 matrix multiplication with exact INT32 accumulation."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
if _TRITON_AVAILABLE:

    @triton.jit
    def _batched_int8_mm_kernel(
        lhs_ptr,
        rhs_ptr,
        output_ptr,
        rows,
        columns,
        reduction,
        lhs_batch_stride,
        lhs_head_stride,
        lhs_row_stride,
        lhs_reduction_stride,
        rhs_batch_stride,
        rhs_head_stride,
        rhs_reduction_stride,
        rhs_column_stride,
        output_batch_stride,
        output_head_stride,
        output_row_stride,
        output_column_stride,
        heads: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        column_block = tl.program_id(1)
        group = tl.program_id(2)
        batch = group // heads
        head = group - batch * heads
        row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        column_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction_offsets = tl.arange(0, BLOCK_K)
        lhs_base = lhs_ptr + batch * lhs_batch_stride + head * lhs_head_stride
        rhs_base = rhs_ptr + batch * rhs_batch_stride + head * rhs_head_stride
        lhs_offsets = (
            row_offsets[:, None] * lhs_row_stride
            + reduction_offsets[None, :] * lhs_reduction_stride
        )
        rhs_offsets = (
            reduction_offsets[:, None] * rhs_reduction_stride
            + column_offsets[None, :] * rhs_column_stride
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for reduction_start in range(0, tl.cdiv(reduction, BLOCK_K)):
            current_reduction = reduction_start * BLOCK_K + reduction_offsets
            lhs = tl.load(
                lhs_base + lhs_offsets,
                mask=(row_offsets[:, None] < rows)
                & (current_reduction[None, :] < reduction),
                other=0,
            )
            rhs = tl.load(
                rhs_base + rhs_offsets,
                mask=(current_reduction[:, None] < reduction)
                & (column_offsets[None, :] < columns),
                other=0,
            )
            accumulator += tl.dot(lhs, rhs, out_dtype=tl.int32)
            lhs_offsets += BLOCK_K * lhs_reduction_stride
            rhs_offsets += BLOCK_K * rhs_reduction_stride
        output_base = (
            output_ptr + batch * output_batch_stride + head * output_head_stride
        )
        output_offsets = (
            row_offsets[:, None] * output_row_stride
            + column_offsets[None, :] * output_column_stride
        )
        tl.store(
            output_base + output_offsets,
            accumulator,
            mask=(row_offsets[:, None] < rows) & (column_offsets[None, :] < columns),
        )


@dataclass
class Int8MatmulWorkspace:
    """Two reusable INT32 buffers shared by all sequential attention layers."""

    buffers: dict[tuple[str, str, int | None], torch.Tensor] = field(
        default_factory=dict
    )

    def acquire(
        self, slot: str, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        required = 1
        for extent in shape:
            required *= int(extent)
        key = (slot, device.type, device.index)
        buffer = self.buffers.get(key)
        if buffer is None or buffer.numel() < required:
            buffer = torch.empty(required, device=device, dtype=torch.int32)
            self.buffers[key] = buffer
        return buffer[:required].view(shape)


def triton_int8_bmm_available() -> bool:
    return _TRITON_AVAILABLE and os.environ.get("SUPRA_DISABLE_TRITON_INT8_BMM") != "1"


def reference_int8_batched_matmul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Existing per-head CUDA reference used for exact regression checks."""
    (batches, heads, rows, _) = lhs.shape
    columns = rhs.shape[-1]
    output = torch.empty(
        (batches, heads, rows, columns), device=lhs.device, dtype=torch.int32
    )
    for batch in range(batches):
        for head in range(heads):
            if lhs.is_cuda and hasattr(torch, "_int_mm"):
                padded_rows = max(32, (rows + 15) // 16 * 16)
                padded_k = max(8, (lhs.shape[-1] + 7) // 8 * 8)
                padded_lhs = torch.zeros(
                    (padded_rows, padded_k), device=lhs.device, dtype=torch.int8
                )
                padded_lhs[:rows, : lhs.shape[-1]] = lhs[batch, head]
                padded_columns = max(8, (columns + 7) // 8 * 8)
                padded_rhs = torch.zeros(
                    (padded_k, padded_columns), device=rhs.device, dtype=torch.int8
                )
                padded_rhs[: rhs.shape[-2], :columns] = rhs[batch, head]
                output[batch, head] = torch._int_mm(padded_lhs, padded_rhs)[
                    :rows, :columns
                ]
            else:
                output[batch, head] = torch.matmul(
                    lhs[batch, head].to(torch.int32), rhs[batch, head].to(torch.int32)
                )
    return output


def int8_batched_matmul(
    lhs: torch.Tensor, rhs: torch.Tensor, *, output: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute signed INT8 products with INT32 accumulation."""
    if lhs.ndim != 4 or rhs.ndim != 4:
        raise ValueError("INT8 batched matmul expects rank-4 operands")
    if lhs.dtype != torch.int8 or rhs.dtype != torch.int8:
        raise TypeError("INT8 matmul requires signed int8 operands")
    if lhs.shape[:2] != rhs.shape[:2] or lhs.shape[-1] != rhs.shape[-2]:
        raise ValueError(
            f"incompatible INT8 batched matmul shapes: {tuple(lhs.shape)} and {tuple(rhs.shape)}"
        )
    (batches, heads, rows, reduction) = (int(value) for value in lhs.shape)
    columns = int(rhs.shape[-1])
    expected_shape = (batches, heads, rows, columns)
    if output is None:
        output = torch.empty(expected_shape, device=lhs.device, dtype=torch.int32)
    elif (
        output.shape != expected_shape
        or output.dtype != torch.int32
        or output.device != lhs.device
    ):
        raise ValueError(
            "INT8 batched matmul output has the wrong shape, dtype, or device"
        )
    if (
        lhs.is_cuda
        and (not triton_int8_bmm_available())
        and (os.environ.get("SUPRA_REQUIRE_TRITON_INT8_BMM") == "1")
    ):
        raise RuntimeError(
            "SUPRA_REQUIRE_TRITON_INT8_BMM=1 but the Triton INT8 batched kernel is unavailable"
        )
    if not lhs.is_cuda or not triton_int8_bmm_available():
        output.copy_(reference_int8_batched_matmul(lhs, rhs))
        return output
    if rhs.device != lhs.device:
        raise ValueError("INT8 batched matmul operands must be on the same device")
    block_m = 32
    block_n = 32
    block_k = 32
    grid = (triton.cdiv(rows, block_m), triton.cdiv(columns, block_n), batches * heads)
    _batched_int8_mm_kernel[grid](
        lhs,
        rhs,
        output,
        rows,
        columns,
        reduction,
        lhs.stride(0),
        lhs.stride(1),
        lhs.stride(2),
        lhs.stride(3),
        rhs.stride(0),
        rhs.stride(1),
        rhs.stride(2),
        rhs.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        heads=heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )
    return output
