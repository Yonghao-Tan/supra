"""Hardware-numeric streaming candidate reduction.

The reducer processes one BF16 logit stream per logical token row.  It keeps
64 independent lane states by default while vocabulary entries arrive in increasing
token-ID order, then merges those states with a fixed pairwise tree.  The
state is sufficient for top-1 token selection and LUT-softmax confidence; no
full-vocabulary probability tensor is retained.

"""

from __future__ import annotations
from typing import Sequence
import torch
from numerics.bf16 import bf16, bf16_add, bf16_mul

CANDIDATE_LANES = 64
CANDIDATE_NUMERIC_SCHEDULE = f"candidate-online-bf16-lut-lanes{CANDIDATE_LANES}/v1"
CANDIDATE_SUPPRESSION_SCHEDULE = "raw-confidence-plus-zero-action/v1"


def _exp_lut(device: torch.device) -> torch.Tensor:
    return bf16(
        torch.exp(torch.linspace(-16.0, 0.0, 256, device=device, dtype=torch.float32))
    )


def _reciprocal_lut(device: torch.device) -> torch.Tensor:
    return bf16(1.0 / torch.linspace(1.0, 2.0, 256, device=device, dtype=torch.float32))


def _lut_exp_delta(delta: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(bf16(delta), min=-16.0, max=0.0)
    index = torch.clamp(
        torch.round((clipped + 16.0) * (255.0 / 16.0)).to(torch.long), 0, 255
    )
    return table[index]


def streaming_candidate_bf16_oracle(
    logits: torch.Tensor, *, lane_count: int = CANDIDATE_LANES
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top token, BF16 top logit, and BF16 LUT-softmax confidence."""
    if logits.ndim < 2 or not torch.is_floating_point(logits) or logits.shape[-1] == 0:
        raise ValueError(
            "candidate logits must be a non-empty floating tensor with a vocabulary axis"
        )
    if lane_count <= 0 or lane_count & lane_count - 1:
        raise ValueError("candidate lane count must be a positive power of two")
    original_shape = logits.shape[:-1]
    vocab = int(logits.shape[-1])
    matrix = bf16(logits).reshape(-1, vocab)
    rows = matrix.shape[0]
    device = matrix.device
    exp_table = _exp_lut(device)
    lane_max = torch.full(
        (rows, lane_count), -float("inf"), device=device, dtype=torch.float32
    )
    lane_sum = torch.zeros_like(lane_max)
    lane_id = torch.full((rows, lane_count), vocab, device=device, dtype=torch.long)
    lane_valid = torch.zeros((rows, lane_count), device=device, dtype=torch.bool)
    for base in range(0, vocab, lane_count):
        count = min(lane_count, vocab - base)
        value = matrix[:, base : base + count]
        old_max = lane_max[:, :count]
        old_sum = lane_sum[:, :count]
        valid = lane_valid[:, :count]
        new_max = torch.maximum(old_max, value)
        scaled_old = bf16_mul(old_sum, _lut_exp_delta(old_max - new_max, exp_table))
        new_term = _lut_exp_delta(value - new_max, exp_table)
        new_sum = bf16_add(scaled_old, new_term)
        token_id = torch.arange(
            base, base + count, device=device, dtype=torch.long
        ).unsqueeze(0)
        new_id = torch.where(
            value > old_max,
            token_id,
            torch.where(
                value == old_max,
                torch.minimum(lane_id[:, :count], token_id),
                lane_id[:, :count],
            ),
        )
        lane_max[:, :count] = torch.where(valid, new_max, value)
        lane_sum[:, :count] = torch.where(valid, new_sum, torch.ones_like(value))
        lane_id[:, :count] = torch.where(valid, new_id, token_id)
        lane_valid[:, :count] = True
    global_max = lane_max.max(dim=1, keepdim=True).values
    winner_id = (
        torch.where(
            lane_valid & (lane_max == global_max),
            lane_id,
            torch.full_like(lane_id, vocab),
        )
        .min(dim=1)
        .values
    )
    reduced = torch.where(
        lane_valid,
        bf16_mul(lane_sum, _lut_exp_delta(lane_max - global_max, exp_table)),
        torch.zeros_like(lane_sum),
    )
    while reduced.shape[1] > 1:
        reduced = bf16_add(reduced[:, 0::2], reduced[:, 1::2])
    total = reduced[:, 0]
    total_bits = total.contiguous().view(torch.int32)
    binary_exponent = (
        torch.bitwise_and(torch.bitwise_right_shift(total_bits, 23), 255) - 127
    )
    inverse_power = torch.pow(
        torch.tensor(2.0, device=device), -binary_exponent.to(torch.float32)
    )
    mantissa = total * inverse_power
    reciprocal_index = torch.clamp(
        torch.round((mantissa - 1.0) * 255.0).to(torch.long), 0, 255
    )
    confidence = bf16(_reciprocal_lut(device)[reciprocal_index] * inverse_power)
    return (
        winner_id.reshape(original_shape),
        global_max[:, 0].reshape(original_shape),
        confidence.reshape(original_shape),
    )


def streaming_candidate_bf16(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the same BF16 logit/confidence storage on CPU and CUDA.

    The oracle retains FP32 containers for numerical inspection. Generation
    must match CUDA's BF16 scalar comparisons and score-add rounding.
    """
    if logits.is_cuda:
        from numerics.operator_kernels import (
            streaming_candidate_bf16_cuda,
            triton_operator_numeric_available,
        )

        if triton_operator_numeric_available():
            return streaming_candidate_bf16_cuda(logits.contiguous())
    (token, top_logit, confidence) = streaming_candidate_bf16_oracle(logits)
    return (token, top_logit.to(torch.bfloat16), confidence.to(torch.bfloat16))


def candidate_token_probability_bf16(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    top_logit_bf16: torch.Tensor,
    top_confidence_bf16: torch.Tensor,
) -> torch.Tensor:
    """Return selected-token probability using the candidate reducer's BF16/LUT schedule."""
    if logits.ndim < 2 or logits.shape[:-1] != token_ids.shape:
        raise ValueError("logits rows and token_ids must have matching shapes")
    if (
        top_logit_bf16.shape != token_ids.shape
        or top_confidence_bf16.shape != token_ids.shape
    ):
        raise ValueError("top logit and confidence must match token_ids")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must have dtype torch.long")
    vocab = int(logits.shape[-1])
    if bool(torch.any((token_ids < 0) | (token_ids >= vocab))):
        raise ValueError("token_ids contain a value outside the vocabulary")
    selected_logit = torch.gather(bf16(logits), -1, token_ids.unsqueeze(-1)).squeeze(-1)
    relative_exp = _lut_exp_delta(
        selected_logit - top_logit_bf16, _exp_lut(logits.device)
    )
    return bf16_mul(top_confidence_bf16, relative_exp)


def candidate_action_confidence(
    proposal: torch.Tensor,
    raw_confidence: torch.Tensor,
    suppressed_token_ids: Sequence[int],
) -> torch.Tensor:
    """Preserve raw confidence and return zero action confidence for a suppressed winner."""
    if proposal.shape != raw_confidence.shape or proposal.dtype != torch.long:
        raise ValueError(
            "proposal and raw confidence must have matching shapes and int64 IDs"
        )
    action_confidence = raw_confidence
    for token_id in suppressed_token_ids:
        action_confidence = torch.where(
            proposal == int(token_id),
            torch.zeros_like(action_confidence),
            action_confidence,
        )
    return action_confidence
