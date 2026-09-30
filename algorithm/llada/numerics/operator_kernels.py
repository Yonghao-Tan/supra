"""Exact fused CUDA kernels for elementwise hardware-numeric operators."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import torch

_CANDIDATE_LUT_CACHE: dict[
    tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]
] = {}
_SILU_TABLE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
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
    _TRITON_AVAILABLE = False
if _TRITON_AVAILABLE:

    @triton.jit
    def _bf16_rne_explicit(value):
        """Round FP32 to BF16 without a removable float trunc/extend pair."""
        bits = value.to(tl.float32).to(tl.int32, bitcast=True)
        exponent = bits & 2139095040
        mantissa = bits & 8388607
        # Keep the 0xffff0000 mask within the signed input type.
        rounded_bits = (bits + 32767 + ((bits >> 16) & 1)) & -65536
        quiet_nan_bits = (bits | 4194304) & -65536
        output_bits = tl.where(
            (exponent == 2139095040) & (mantissa != 0), quiet_nan_bits, rounded_bits
        )
        return output_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _bf16_pairwise_sum_power_of_two(values, BLOCK_SIZE: tl.constexpr):
        if BLOCK_SIZE >= 2:
            pairs = tl.reshape(values, (BLOCK_SIZE // 2, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 4:
            pairs = tl.reshape(values, (BLOCK_SIZE // 4, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 8:
            pairs = tl.reshape(values, (BLOCK_SIZE // 8, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 16:
            pairs = tl.reshape(values, (BLOCK_SIZE // 16, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 32:
            pairs = tl.reshape(values, (BLOCK_SIZE // 32, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 64:
            pairs = tl.reshape(values, (BLOCK_SIZE // 64, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 128:
            pairs = tl.reshape(values, (BLOCK_SIZE // 128, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 256:
            pairs = tl.reshape(values, (BLOCK_SIZE // 256, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 512:
            pairs = tl.reshape(values, (BLOCK_SIZE // 512, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 1024:
            pairs = tl.reshape(values, (BLOCK_SIZE // 1024, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 2048:
            pairs = tl.reshape(values, (BLOCK_SIZE // 2048, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        if BLOCK_SIZE >= 4096:
            pairs = tl.reshape(values, (BLOCK_SIZE // 4096, 2))
            values = tl.sum(pairs, axis=1).to(tl.bfloat16).to(tl.float32)
        return values

    @triton.jit
    def _silu_pwl_bf16_kernel(
        values_ptr,
        breakpoints_ptr,
        slopes_ptr,
        intercepts_ptr,
        output_ptr,
        elements,
        SEGMENTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        values = (
            tl.load(values_ptr + offsets, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        result = tl.zeros((BLOCK,), tl.float32)
        for segment in range(SEGMENTS):
            lower = tl.load(breakpoints_ptr + segment).to(tl.bfloat16).to(tl.float32)
            upper = (
                tl.load(breakpoints_ptr + segment + 1).to(tl.bfloat16).to(tl.float32)
            )
            slope = tl.load(slopes_ptr + segment).to(tl.bfloat16).to(tl.float32)
            intercept = tl.load(intercepts_ptr + segment).to(tl.bfloat16).to(tl.float32)
            product = (values * slope).to(tl.bfloat16).to(tl.float32)
            candidate = (product + intercept).to(tl.bfloat16).to(tl.float32)
            result = tl.where((values >= lower) & (values < upper), candidate, result)
        lower_tail = tl.load(breakpoints_ptr).to(tl.bfloat16).to(tl.float32)
        upper_tail = tl.load(breakpoints_ptr + SEGMENTS).to(tl.bfloat16).to(tl.float32)
        result = tl.where(
            values <= lower_tail, 0.0, tl.where(values >= upper_tail, values, result)
        )
        tl.store(output_ptr + offsets, result, mask=valid)

    @triton.jit
    def _rope_products_bf16_kernel(
        values_ptr,
        sin_ptr,
        cos_ptr,
        direct_ptr,
        rotated_ptr,
        elements,
        tokens,
        values_batch_stride,
        values_head_stride,
        values_token_stride,
        values_channel_stride,
        table_token_stride,
        table_channel_stride,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        channel = offsets % head_dim
        token_linear = offsets // head_dim
        token = token_linear % tokens
        batch_head = token_linear // tokens
        head = batch_head % heads
        batch = batch_head // heads
        pair_channel = (channel + head_dim // 2) % head_dim
        pair_sign = tl.where(channel < head_dim // 2, -1.0, 1.0)
        value_base = (
            values_ptr
            + batch * values_batch_stride
            + head * values_head_stride
            + token * values_token_stride
        )
        value = (
            tl.load(value_base + channel * values_channel_stride, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        pair = (
            tl.load(
                value_base + pair_channel * values_channel_stride, mask=valid, other=0.0
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        sine = (
            tl.load(
                sin_ptr + token * table_token_stride + channel * table_channel_stride,
                mask=valid,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        cosine = (
            tl.load(
                cos_ptr + token * table_token_stride + channel * table_channel_stride,
                mask=valid,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        direct = (value * cosine).to(tl.bfloat16)
        rotated = (pair_sign * pair * sine).to(tl.bfloat16)
        tl.store(direct_ptr + offsets, direct, mask=valid)
        tl.store(rotated_ptr + offsets, rotated, mask=valid)

    @triton.jit
    def _rope_add_bf16_kernel(
        direct_ptr, rotated_ptr, output_ptr, elements, BLOCK: tl.constexpr
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        direct = (
            tl.load(direct_ptr + offsets, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rotated = (
            tl.load(rotated_ptr + offsets, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        result = (direct + rotated).to(tl.bfloat16)
        tl.store(output_ptr + offsets, result, mask=valid)

    @triton.jit
    def _rms_square_pair_bf16_kernel(
        values_ptr,
        output_ptr,
        output_elements,
        columns: tl.constexpr,
        values_row_stride,
        values_column_stride,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < output_elements
        pairs_per_row = columns // 2
        rows = offsets // pairs_per_row
        pairs = offsets - rows * pairs_per_row
        value_base = (
            values_ptr + rows * values_row_stride + pairs * 2 * values_column_stride
        )
        lhs = tl.load(value_base, mask=valid, other=0.0).to(tl.bfloat16).to(tl.float32)
        rhs = (
            tl.load(value_base + values_column_stride, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        lhs_squared = (lhs * lhs).to(tl.bfloat16).to(tl.float32)
        rhs_squared = (rhs * rhs).to(tl.bfloat16).to(tl.float32)
        result = (lhs_squared + rhs_squared).to(tl.bfloat16).to(tl.float32)
        tl.store(output_ptr + offsets, result, mask=valid)

    @triton.jit
    def _pair_reduce_bf16_kernel(
        input_ptr, output_ptr, output_elements, BLOCK: tl.constexpr
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < output_elements
        lhs = (
            tl.load(input_ptr + offsets * 2, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        rhs = (
            tl.load(input_ptr + offsets * 2 + 1, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        result = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
        tl.store(output_ptr + offsets, result, mask=valid)

    @triton.jit
    def _rms_normalize_bf16_kernel(
        values_ptr,
        totals_ptr,
        weight_ptr,
        output_ptr,
        elements,
        columns: tl.constexpr,
        epsilon: tl.constexpr,
        values_row_stride,
        values_column_stride,
        APPLY_WEIGHT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < elements
        rows = offsets // columns
        channels = offsets - rows * columns
        values = (
            tl.load(
                values_ptr + rows * values_row_stride + channels * values_column_stride,
                mask=valid,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        totals = (
            tl.load(totals_ptr + rows, mask=valid, other=0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        mean = _bf16_rne_explicit(totals * (1.0 / columns))
        epsilon_bf16 = _bf16_rne_explicit(tl.full((BLOCK,), epsilon, tl.float32))
        x = _bf16_rne_explicit(mean + epsilon_bf16)
        bits = x.to(tl.int32, bitcast=True)
        estimate_bits = 1597463007 - (bits >> 1)
        y = _bf16_rne_explicit(estimate_bits.to(tl.float32, bitcast=True))
        y_squared = _bf16_rne_explicit(y * y)
        half_x = _bf16_rne_explicit(0.5 * x)
        correction_product = _bf16_rne_explicit(half_x * y_squared)
        correction = _bf16_rne_explicit(1.5 - correction_product)
        reciprocal_root = _bf16_rne_explicit(y * correction)
        normalized = _bf16_rne_explicit(values * reciprocal_root)
        if APPLY_WEIGHT:
            weight = (
                tl.load(weight_ptr + channels, mask=valid, other=0.0)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            result = _bf16_rne_explicit(normalized * weight)
        else:
            result = normalized
        tl.store(output_ptr + offsets, result, mask=valid)

    @triton.jit
    def _candidate_exp_delta(delta, exp_lut_ptr):
        clipped = tl.maximum(
            -16.0, tl.minimum(0.0, delta.to(tl.bfloat16).to(tl.float32))
        )
        index = tl_math.float2int_rn((clipped + 16.0) * (255.0 / 16.0))
        index = tl.maximum(0, tl.minimum(255, index))
        return tl.load(exp_lut_ptr + index).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _streaming_candidate_bf16_kernel(
        logits_ptr,
        exp_lut_ptr,
        reciprocal_lut_ptr,
        token_ptr,
        top_logit_ptr,
        confidence_ptr,
        rows,
        columns,
        LANE_COUNT: tl.constexpr,
    ):
        row = tl.program_id(0)
        lanes = tl.arange(0, LANE_COUNT)
        lane_max = tl.full((LANE_COUNT,), -float("inf"), tl.float32)
        lane_sum = tl.zeros((LANE_COUNT,), tl.float32)
        lane_id = tl.full((LANE_COUNT,), columns, tl.int32)
        lane_valid = tl.zeros((LANE_COUNT,), tl.int1)
        base = 0
        while base < columns:
            token = base + lanes
            valid = (row < rows) & (token < columns)
            value = (
                tl.load(logits_ptr + row * columns + token, mask=valid, other=0.0)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            new_max = tl.maximum(lane_max, value)
            old_scaled = (
                (lane_sum * _candidate_exp_delta(lane_max - new_max, exp_lut_ptr))
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            new_term = _candidate_exp_delta(value - new_max, exp_lut_ptr)
            new_sum = (old_scaled + new_term).to(tl.bfloat16).to(tl.float32)
            updated_id = tl.where(
                value > lane_max,
                token,
                tl.where(value == lane_max, tl.minimum(lane_id, token), lane_id),
            )
            lane_max = tl.where(valid, tl.where(lane_valid, new_max, value), lane_max)
            lane_sum = tl.where(valid, tl.where(lane_valid, new_sum, 1.0), lane_sum)
            lane_id = tl.where(valid, tl.where(lane_valid, updated_id, token), lane_id)
            lane_valid = lane_valid | valid
            base += LANE_COUNT
        global_max = tl.max(tl.where(lane_valid, lane_max, -float("inf")), axis=0)
        winner_id = tl.min(
            tl.where(lane_valid & (lane_max == global_max), lane_id, columns), axis=0
        )
        scaled_sum = tl.where(
            lane_valid,
            (lane_sum * _candidate_exp_delta(lane_max - global_max, exp_lut_ptr))
            .to(tl.bfloat16)
            .to(tl.float32),
            0.0,
        )
        total = _bf16_pairwise_sum_power_of_two(scaled_sum, BLOCK_SIZE=LANE_COUNT)
        total_bits = total.to(tl.int32, bitcast=True)
        binary_exponent = (total_bits >> 23 & 255) - 127
        inverse_power_bits = 127 - binary_exponent << 23
        inverse_power = inverse_power_bits.to(tl.float32, bitcast=True)
        mantissa = total * inverse_power
        reciprocal_index = tl_math.float2int_rn((mantissa - 1.0) * 255.0)
        reciprocal_index = tl.maximum(0, tl.minimum(255, reciprocal_index))
        reciprocal_mantissa = (
            tl.load(reciprocal_lut_ptr + reciprocal_index)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        confidence = (reciprocal_mantissa * inverse_power).to(tl.bfloat16)
        output_offset = row + tl.arange(0, 1)
        tl.store(token_ptr + output_offset, winner_id)
        tl.store(top_logit_ptr + output_offset, global_max.to(tl.bfloat16))
        tl.store(confidence_ptr + output_offset, confidence)


def triton_operator_numeric_available() -> bool:
    return (
        _TRITON_AVAILABLE
        and os.environ.get("SUPRA_DISABLE_TRITON_OPERATOR_NUMERIC") != "1"
    )


def streaming_candidate_bf16_cuda(
    logits: torch.Tensor, *, lane_count: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        not logits.is_cuda
        or not logits.is_contiguous()
        or logits.ndim < 2
        or (logits.shape[-1] == 0)
    ):
        raise ValueError(
            "streaming candidate reducer requires contiguous CUDA logits with a vocabulary axis"
        )
    from numerics.candidate import CANDIDATE_LANES

    if lane_count is None:
        lane_count = CANDIDATE_LANES
    if lane_count <= 0 or lane_count & lane_count - 1:
        raise ValueError("candidate lane count must be a positive power of two")
    columns = int(logits.shape[-1])
    rows = logits.numel() // columns
    key = (logits.device.type, logits.device.index)
    tables = _CANDIDATE_LUT_CACHE.get(key)
    if tables is None:
        from numerics.bf16 import bf16

        tables = (
            bf16(torch.exp(torch.linspace(-16.0, 0.0, 256, device=logits.device))),
            bf16(1.0 / torch.linspace(1.0, 2.0, 256, device=logits.device)),
        )
        _CANDIDATE_LUT_CACHE[key] = tables
    tokens = torch.empty(rows, device=logits.device, dtype=torch.int32)
    top_logits = torch.empty(rows, device=logits.device, dtype=torch.bfloat16)
    confidence = torch.empty(rows, device=logits.device, dtype=torch.bfloat16)
    _streaming_candidate_bf16_kernel[rows,](
        logits,
        tables[0],
        tables[1],
        tokens,
        top_logits,
        confidence,
        rows,
        columns,
        LANE_COUNT=lane_count,
        num_warps=4,
    )
    output_shape = logits.shape[:-1]
    return (
        tokens.to(torch.long).reshape(output_shape),
        top_logits.reshape(output_shape),
        confidence.reshape(output_shape),
    )


@dataclass
class RMSNormNumericWorkspace:
    buffers: dict[tuple[str, int | None, int], torch.Tensor] = field(
        default_factory=dict
    )

    def acquire(self, slot: int, elements: int, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index, slot)
        buffer = self.buffers.get(key)
        if buffer is None or buffer.numel() < elements:
            buffer = torch.empty(elements, device=device, dtype=torch.bfloat16)
            self.buffers[key] = buffer
        return buffer[:elements]


def silu_pwl_bf16_cuda(values: torch.Tensor, *, coefficients=None) -> torch.Tensor:
    if (
        not values.is_cuda
        or not torch.is_floating_point(values)
        or (not values.is_contiguous())
    ):
        raise ValueError("fused SiLU PWL requires a contiguous floating CUDA tensor")
    key = (values.device.type, values.device.index, coefficients)
    tables = _SILU_TABLE_CACHE.get(key)
    if tables is None:
        from numerics.bf16 import silu_pwl_table_tensors

        tables = silu_pwl_table_tensors(values.device)
        if coefficients is not None:
            tables = (
                tables[0],
                *(
                    torch.tensor(row, device=values.device, dtype=torch.float32)
                    for row in coefficients
                ),
            )
        _SILU_TABLE_CACHE[key] = tables
    output = torch.empty_like(values, dtype=torch.bfloat16)
    elements = values.numel()
    block = 256
    _silu_pwl_bf16_kernel[triton.cdiv(elements, block),](
        values,
        tables[0],
        tables[1],
        tables[2],
        output,
        elements,
        SEGMENTS=int(tables[1].numel()),
        BLOCK=block,
        num_warps=4,
    )
    return output


def rope_apply_bf16_cuda(
    values: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    if values.ndim != 4 or values.shape[-1] != 128 or (not values.is_cuda):
        raise ValueError("fused RoPE requires CUDA values shaped [B,H,T,128]")
    if sin.shape != cos.shape or sin.shape[-2:] != values.shape[-2:]:
        raise ValueError(
            "fused RoPE tables must match the value token and channel dimensions"
        )
    table_sin = sin.reshape(-1, 128)
    table_cos = cos.reshape(-1, 128)
    if table_sin.shape[0] != values.shape[-2]:
        raise ValueError("fused RoPE supports one shared table row per token")
    direct = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
    rotated = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
    output = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
    elements = values.numel()
    block = 256
    _rope_products_bf16_kernel[triton.cdiv(elements, block),](
        values,
        table_sin,
        table_cos,
        direct,
        rotated,
        elements,
        int(values.shape[-2]),
        values.stride(0),
        values.stride(1),
        values.stride(2),
        values.stride(3),
        table_sin.stride(0),
        table_sin.stride(1),
        heads=int(values.shape[1]),
        head_dim=128,
        BLOCK=block,
        num_warps=4,
    )
    _rope_add_bf16_kernel[triton.cdiv(elements, block),](
        direct, rotated, output, elements, BLOCK=block, num_warps=4
    )
    return output.to(torch.float32)


def rms_norm_bf16_cuda(
    values: torch.Tensor,
    weight_bf16: torch.Tensor | None,
    epsilon: float,
    workspace: RMSNormNumericWorkspace,
) -> torch.Tensor:
    columns = int(values.shape[-1])
    if columns < 2 or columns & columns - 1 or (not values.is_cuda):
        raise ValueError(
            "fused RMSNorm requires a CUDA tensor with a power-of-two channel count"
        )
    matrix = values.reshape(-1, columns)
    if weight_bf16 is not None and (
        weight_bf16.shape != (columns,) or weight_bf16.device != values.device
    ):
        raise ValueError("fused RMSNorm weight does not match the input")
    return _rms_norm_bf16_cuda_staged(
        matrix, weight_bf16, epsilon, workspace
    ).reshape_as(values)


def _rms_norm_bf16_cuda_staged(
    values: torch.Tensor,
    weight_bf16: torch.Tensor | None,
    epsilon: float,
    workspace: RMSNormNumericWorkspace,
) -> torch.Tensor:
    """Compute the BF16 pairwise reduction and normalization in separate kernels."""
    columns = int(values.shape[-1])
    rows = values.numel() // columns
    matrix = values.reshape(rows, columns)
    first_elements = rows * (columns // 2)
    buffers = (
        workspace.acquire(0, first_elements, values.device),
        workspace.acquire(1, first_elements, values.device),
    )
    block = 256
    _rms_square_pair_bf16_kernel[triton.cdiv(first_elements, block),](
        matrix,
        buffers[0],
        first_elements,
        columns=columns,
        values_row_stride=matrix.stride(0),
        values_column_stride=matrix.stride(1),
        BLOCK=block,
        num_warps=4,
    )
    current_elements = first_elements
    input_slot = 0
    width = columns // 2
    while width > 1:
        output_elements = current_elements // 2
        output_slot = 1 - input_slot
        _pair_reduce_bf16_kernel[triton.cdiv(output_elements, block),](
            buffers[input_slot],
            buffers[output_slot],
            output_elements,
            BLOCK=block,
            num_warps=4,
        )
        current_elements = output_elements
        input_slot = output_slot
        width //= 2
    output = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
    elements = values.numel()
    _rms_normalize_bf16_kernel[triton.cdiv(elements, block),](
        matrix,
        buffers[input_slot],
        weight_bf16 if weight_bf16 is not None else matrix,
        output,
        elements,
        columns=columns,
        epsilon=float(epsilon),
        values_row_stride=matrix.stride(0),
        values_column_stride=matrix.stride(1),
        APPLY_WEIGHT=weight_bf16 is not None,
        BLOCK=block,
        num_warps=4,
    )
    return output
