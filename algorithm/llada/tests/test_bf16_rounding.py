"""Explicit rounding must preserve signed zero, ties and NaN payload bits."""

import pytest
import torch

from numerics import operator_kernels as kernels
from numerics.bf16 import rms_norm_bf16

if kernels._TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def round_kernel(source, output, count, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(source + index, index < count, other=0)
        rounded = kernels._bf16_rne_explicit(value.to(tl.float32, bitcast=True))
        tl.store(output + index, rounded.to(tl.int32, bitcast=True), index < count)


@pytest.mark.skipif(not kernels._TRITON_AVAILABLE or not torch.cuda.is_available(),
                    reason="requires CUDA and Triton")
def test_explicit_rounding_boundaries_and_rmsnorm():
    high = torch.arange(65536, dtype=torch.int64)
    low = torch.tensor([0, 1, 32767, 32768, 32769, 65535], dtype=torch.int64)
    bits = ((high[:, None] << 16) | low).flatten()
    expected = (bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000
    is_nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x7FFFFF) != 0)
    expected = torch.where(is_nan, (bits | 0x400000) & 0xFFFF0000, expected)
    source = bits.to(torch.int32).cuda()
    actual = torch.empty_like(source)
    round_kernel[triton.cdiv(source.numel(), 256),](source, actual, source.numel(), BLOCK=256)
    assert torch.equal(actual.cpu(), expected.to(torch.int32))

    generator = torch.Generator().manual_seed(20260906)
    values = torch.randn(17, 4096, generator=generator).to(torch.bfloat16)
    values[0] = 0
    values[1] = -0.0
    values[2] *= 0.0001
    weight = torch.randn(4096, generator=generator).to(torch.bfloat16)
    workspace = kernels.RMSNormNumericWorkspace()
    for multiplier in (None, weight):
        expected = rms_norm_bf16(values.float(), multiplier, 1e-5).to(torch.bfloat16)
        actual = kernels.rms_norm_bf16_cuda(
            values.cuda(), None if multiplier is None else multiplier.cuda(), 1e-5, workspace
        )
        assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))


@pytest.mark.skipif(not kernels._TRITON_AVAILABLE or not torch.cuda.is_available(),
                    reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_quantization_strides_and_bf16_input(dtype):
    from numerics.linear_kernels import (LinearNumericWorkspace, quantize_per_row_bf16_cuda,
        quantize_per_row_bits_bf16_cuda, linear_rescale_bf16_cuda)
    from numerics.bf16 import quantize_mixed_rows_bf16, bf16
    storage = torch.full((4, 256), 2., device="cuda", dtype=dtype)
    values = storage[:, ::2]
    values.fill_(1.)
    values[:, 1] = .4999
    modes = torch.tensor([8, 0, 4, 0, 8, 0, 4, 0], device="cuda", dtype=torch.int8)[::2]
    for bits in (None, 4, 8):
        row_bits = modes if bits is None else torch.full_like(modes, bits)
        expected = quantize_mixed_rows_bf16(values, (row_bits == 8).long())
        workspace = LinearNumericWorkspace()
        codes, scales, rows = (quantize_per_row_bits_bf16_cuda(values, modes, workspace)
            if bits is None else quantize_per_row_bf16_cuda(values, workspace, activation_bits=bits))
        assert torch.equal(codes[:rows], expected.codes)
        assert torch.equal(scales.view(torch.int32), expected.scale_bf16.view(torch.int32))
    sums = torch.arange(4 * 128, device="cuda", dtype=torch.int32).reshape(4, 128)
    weight_storage = torch.arange(1, 257, device="cuda", dtype=torch.float32)
    weights = weight_storage[::2]
    # Division before slicing preserves a non-unit scale stride.
    activation_storage = torch.arange(1, 9, device="cuda", dtype=torch.float32) / 16
    activation = activation_storage[::2]
    expected = bf16(bf16(sums.float() * activation[:, None]) * weights[None, :]).bfloat16()
    actual = linear_rescale_bf16_cuda(sums, activation, weights)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    contiguous = linear_rescale_bf16_cuda(sums, activation.contiguous(), weights.contiguous())
    assert torch.equal(contiguous.view(torch.int16), expected.view(torch.int16))
