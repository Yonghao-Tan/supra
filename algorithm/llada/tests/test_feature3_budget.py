"""Suppressing new future work must preserve already selected positions."""

import torch
import pytest
from collections import Counter
from dataclasses import asdict
from generation.lookahead import DynamicJointWindowScheduler


@pytest.mark.parametrize("optional_bits", [4, 8])
def test_optional_a8_group_rejection_preserves_reuse_and_confirmation(optional_bits):
    selection = DynamicJointWindowScheduler(
        max_next_rows=4, source_b_a4_only=True
    ).select(
        base_positions=torch.tensor([0, 1, 2, 32]),
        base_row_bits=torch.tensor([4, 4, 8, 4], dtype=torch.int8),
        next_block_start=32,
        next_row_bits=torch.tensor([4, 8, optional_bits, 4], dtype=torch.int8),
        next_unresolved=torch.ones(4, dtype=torch.bool),
        next_tentative=torch.tensor([False, True, False, False]),
        next_priority=torch.tensor([1, 1, .625, .875], dtype=torch.bfloat16),
        current_prediction_rows=2,
    )
    if optional_bits == 8:
        assert selection.progress_local_positions.tolist() == [0, 1]
        assert selection.added_local_positions.tolist() == [1]
        assert selection.source_b_optional_reject_reason == "a8"
    else:
        assert set(selection.progress_local_positions.tolist()) == {0, 1, 2, 3}
        assert set(selection.added_local_positions.tolist()) == {1, 2, 3}
        assert selection.source_b_optional_reject_reason == "none"


def test_source_b_dependency_tie_rank_never_crosses_distinct_scores():
    scheduler = DynamicJointWindowScheduler(
        max_next_rows=1, source_b_dependency_tie_rank=True
    )
    inputs = dict(
        base_positions=torch.tensor([0]),
        base_row_bits=torch.tensor([4], dtype=torch.int8),
        next_block_start=32,
        next_row_bits=torch.tensor([4, 4], dtype=torch.int8),
        next_unresolved=torch.ones(2, dtype=torch.bool),
        next_tentative=torch.zeros(2, dtype=torch.bool),
        next_priority=torch.tensor([0.9, 0.35]),
        step_index=2,
    )
    with pytest.raises(ValueError, match="matching BF16 scores"):
        scheduler.select(**inputs)
    tied = scheduler.select(
        **inputs, next_dependency_score=torch.tensor([0.0, 0.0], dtype=torch.bfloat16)
    )
    assert tied.added_local_positions.tolist() == [1]
    distinct = scheduler.select(
        **inputs, next_dependency_score=torch.tensor([0.0, 0.1], dtype=torch.bfloat16)
    )
    assert distinct.added_local_positions.tolist() == [0]


def test_deep_precision_protects_required_rows_and_uses_stable_rank():
    from generation.engine import boundary_deep_precision_bits

    bits = torch.tensor([4, 8, 8, 8, 8, 8], dtype=torch.int8)
    scores = torch.tensor([0, 0, 9, 9, 2, 1], dtype=torch.uint8)
    protected = torch.tensor([True, True, False, False, False, True])
    selected = boundary_deep_precision_bits(bits, scores, protected, 3)
    assert selected.tolist() == [4, 8, 8, 4, 4, 8]
    assert bits.tolist() == [4, 8, 8, 8, 8, 8]
    assert boundary_deep_precision_bits(bits, scores, protected, 1).tolist() == [
        4,
        8,
        4,
        4,
        4,
        8,
    ]
    assert torch.equal(boundary_deep_precision_bits(bits, scores, protected, 5), bits)


def test_attempt_limit_preserves_reused_and_tentative_rows_without_reordering():
    scheduler = DynamicJointWindowScheduler(target_joint_rows=48, max_next_rows=4)
    inputs = dict(
        base_positions=torch.tensor([0, 33]),
        base_row_bits=torch.tensor([8, 4], dtype=torch.int8),
        next_block_start=32,
        next_row_bits=torch.tensor([4, 4, 8, 4], dtype=torch.int8),
        next_unresolved=torch.ones(4, dtype=torch.bool),
        next_tentative=torch.tensor([False, False, True, False]),
        next_priority=torch.tensor([1.0, 0.5, 0.75, 0.9]),
        source_b_attempts=torch.tensor([2, 7, 5, 1], dtype=torch.int32),
    )
    limited = scheduler.select(**inputs, max_source_b_attempts=2)
    assert limited.progress_local_positions.tolist() == [1, 2, 3]
    assert limited.added_local_positions.tolist() == [2, 3]
    default = scheduler.select(**inputs)
    uncapped = scheduler.select(
        **{k: v for (k, v) in inputs.items() if k != "source_b_attempts"}
    )
    assert torch.equal(default.added_local_positions, uncapped.added_local_positions)
    with pytest.raises(ValueError, match="matching nonnegative int32"):
        scheduler.select(
            **{k: v for (k, v) in inputs.items() if k != "source_b_attempts"},
            max_source_b_attempts=2,
        )


@pytest.mark.parametrize("attempt_limit,retry_confidence,head_target", [
    (attempts, confidence, -1)
    for attempts in (1, 2) for confidence in (0.0, 1.0)
] + [(1, 1.0, 3)])
def test_generator_retains_attempt_count_across_steps(
    attempt_limit, retry_confidence, head_target
):
    from generation.engine import generate
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import (
        ActivationType,
        BlockType,
        LLaDAModel,
        LLaDAModelLM,
        ModelConfig,
    )
    from numerics.precision import RowPrecisionContext

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        config = ModelConfig(
            d_model=16,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            mlp_hidden_size=32,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=16,
            vocab_size=4,
            embedding_size=4,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModelLM(
            LLaDAConfig(**asdict(config)), model=LLaDAModel(config)
        ).eval()
    (_, _, trace, _) = generate(
        model,
        torch.tensor([[2]]),
        steps=12,
        gen_length=6,
        block_length=2,
        mask_id=3,
        tau_high=0.9,
        tau_low=0.5,
        confirm_tau=0.5,
        budget_scale=1.0,
        row_precision_context=RowPrecisionContext(),
        packed_attention_refresh=True,
        packed_attention_force_full_current=True,
        packed_attention_full_current_context_rows=0,
        dynamic_block_lookahead=True,
        dynamic_block_canonical_future=True,
        dynamic_block_target_joint_rows=4,
        dynamic_block_max_next_rows=2,
        dynamic_block_next_admission_budget=2,
        dynamic_block_max_handoff_verification_rows=2,
        dynamic_block_canonical_direct_tau=0.99,
        dynamic_block_next_min_confidence=0.99,
        dynamic_block_source_b_max_attempts=attempt_limit,
        dynamic_block_source_b_retry_min_confidence=retry_confidence,
        dynamic_block_target_prediction_rows=head_target,
        dynamic_block_allow_deferred_verification=True,
    )
    attempted = Counter(
        (position for event in trace for position in event.next_added_positions)
    )
    assert attempted and max(attempted.values()) <= attempt_limit
    if retry_confidence:
        assert set(attempted.values()) == {1}
    else:
        assert set(attempted.values()) == {attempt_limit}
    assert not any((event.next_admitted_positions for event in trace))
    if head_target >= 0:
        for event in trace:
            current = set(event.prediction_positions)
            joint = current | set(event.next_progress_positions)
            assert len(joint) <= max(head_target, len(current))
    assert {event.block_index for event in trace if event.next_added_positions} == {
        0,
        1,
    }


def test_retry_confidence_keeps_first_attempt_reuse_and_confirmation():
    scheduler = DynamicJointWindowScheduler(max_next_rows=8)
    inputs = dict(
        base_positions=torch.tensor([0, 33]),
        base_row_bits=torch.tensor([8, 4], dtype=torch.int8),
        next_block_start=32,
        next_row_bits=torch.tensor([4, 4, 8, 4, 4], dtype=torch.int8),
        next_unresolved=torch.ones(5, dtype=torch.bool),
        next_tentative=torch.tensor([False, False, True, False, False]),
        next_priority=torch.ones(5),
        source_b_attempts=torch.tensor([1, 2, 2, 0, 1], dtype=torch.int32),
        next_last_confidence=torch.tensor([0.49, 0.1, 0.1, -1, 0.5], dtype=torch.bfloat16),
        max_source_b_attempts=2,
    )
    result = scheduler.select(**inputs, source_b_retry_min_confidence=0.5)
    assert result.progress_local_positions.tolist() == [1, 2, 3, 4]
    assert inputs['base_row_bits'].tolist() == [8, 4]
    with pytest.raises(ValueError, match='confidence history'):
        scheduler.select(**dict(inputs, next_last_confidence=None), source_b_retry_min_confidence=0.5)


def test_joint_prediction_target_preserves_base_and_pending_confirmations():
    scheduler = DynamicJointWindowScheduler(max_next_rows=8)
    inputs = dict(
        base_positions=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 32, 33]),
        base_row_bits=torch.full((10,), 4, dtype=torch.int8),
        next_block_start=32,
        next_row_bits=torch.tensor([4, 4, 8, 8], dtype=torch.int8),
        next_unresolved=torch.ones(4, dtype=torch.bool),
        next_tentative=torch.zeros(4, dtype=torch.bool),
        next_priority=torch.ones(4),
        current_prediction_rows=8,
    )
    limited = scheduler.select(**inputs, target_prediction_rows=10)
    assert limited.progress_local_positions.tolist() == [0, 1]
    assert limited.added_local_positions.numel() == 0
    assert inputs['base_positions'].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 32, 33]
    assert inputs['base_row_bits'].tolist() == [4] * 10
    full_current = scheduler.select(**inputs, target_prediction_rows=4)
    assert full_current.progress_local_positions.numel() == 0
    due = scheduler.select(**dict(inputs, next_tentative=torch.tensor([False, False, True, True])),
                           target_prediction_rows=4)
    assert due.progress_local_positions.tolist() == [2, 3]
    assert due.added_local_positions.tolist() == [2, 3]
