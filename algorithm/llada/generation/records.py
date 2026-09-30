"""Generation events and counters shared by evaluation and capture."""

from __future__ import annotations
from typing import List
from dataclasses import asdict, dataclass, field
from typing import Any, Tuple
import torch


@dataclass(frozen=True)
class ForwardCaptureEvent:
    """Compact, replay-oriented snapshot of one executed model forward."""

    capture_index: int
    block_index: int
    step_index: int
    forward_kind: str
    nfe_before: int
    cache_initialized_before: bool
    tokens_before: torch.Tensor
    block_state_before: torch.Tensor
    model_input_ids: torch.Tensor
    input_positions: torch.Tensor
    refresh_positions: torch.Tensor
    row_bits: torch.Tensor
    prediction_positions: torch.Tensor
    prediction_mask: torch.Tensor
    teacher_top1_token_ids: torch.Tensor
    teacher_top2_token_ids: torch.Tensor
    teacher_top1_margin: torch.Tensor
    teacher_top1_confidence: torch.Tensor
    teacher_top1_action_confidence: torch.Tensor
    teacher_current_token_probability: torch.Tensor
    full_sequence_recompute: bool = False
    layer0_global_selected_deep: bool = False
    layer0_keep_global_cache: bool = False
    boundary_global_layers: int = 1
    boundary_deep_bits: int = 0
    layer0_current_row_bits: Tuple[int, ...] = ()
    boundary_deep_clip_ratio: float = 0.0
    layer1_global_context_bits: int = 4
    layer1_global_prefix_bits: int = 4
    future_prediction_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    future_state_before: torch.Tensor = field(
        default_factory=lambda: torch.empty((1, 0), dtype=torch.int8)
    )
    future_top1_token_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty((1, 0), dtype=torch.int32)
    )
    future_action_confidence: torch.Tensor = field(
        default_factory=lambda: torch.empty((1, 0), dtype=torch.bfloat16)
    )


@dataclass(frozen=True)
class ForwardReplayResult:
    """Student outputs aligned to the active decisions in one capture event."""

    capture_index: int
    prediction_positions: torch.Tensor
    prediction_logits: torch.Tensor
    future_prediction_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    future_prediction_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class GenerationTraceEvent:
    block_index: int
    step_index: int
    forward_kind: str
    input_start: int
    input_length: int
    cache_sequence_length: int
    transferred_positions: Tuple[int, ...]
    direct_locked_positions: Tuple[int, ...]
    stable_tentative_positions: Tuple[int, ...]
    fallback_tentative_positions: Tuple[int, ...]
    confirmed_positions: Tuple[int, ...]
    remasked_positions: Tuple[int, ...]
    forced_finish_positions: Tuple[int, ...]
    masked_before: int
    masked_after: int
    tentative_before: int
    tentative_after: int
    nfe: int
    cache_initialization_activation_policy: str = "default"
    linear_activation_rows: dict[str, dict[str, int]] = field(default_factory=dict)
    selective_a4_direct_positions: Tuple[int, ...] = ()
    a4_rows: int = 0
    a8_rows: int = 0
    masked_a4_rows: int = 0
    tentative_a8_rows: int = 0
    young_locked_a8_rows: int = 0
    mature_locked_a4_rows: int = 0
    tail_bypassed_positions: Tuple[int, ...] = ()
    layer0_rows: int = 0
    layer0_a4_rows: int = 0
    layer0_a8_rows: int = 0
    layer0_keep_global_cache: bool = False
    boundary_global_layers: int = 1
    boundary_deep_bits: int = 0
    layer0_current_row_bits: Tuple[int, ...] = ()
    boundary_deep_clip_ratio: float = 0.0
    prediction_rows: int = 0
    refresh_rows: int = 0
    active_rows: int = 0
    early_exit_rows: int = 0
    mandatory_refresh_rows: int = 0
    optional_refresh_rows: int = 0
    dependency_rows_updated: int = 0
    dependency_table_bytes: int = 0
    pending_dependency_rows: int = 0
    changed_positions: Tuple[int, ...] = ()
    mandatory_refresh_positions: Tuple[int, ...] = ()
    optional_refresh_positions: Tuple[int, ...] = ()
    cross_block_handoff_changed_positions: Tuple[int, ...] = ()
    cross_block_selected_positions: Tuple[int, ...] = ()
    cross_block_ranked_positions: Tuple[int, ...] = ()
    boundary_context_selected_positions: Tuple[int, ...] = ()
    boundary_context_selected_scores: Tuple[float, ...] = ()
    block_initialization_score_floor_bf16: float = 0.0
    block_initialization_score_scan_rows: int = 0
    boundary_context_candidate_rows: int = 0
    boundary_context_score_entries_read: int = 0
    boundary_context_topk_comparisons: int = 0
    dependency_score_by_row: Tuple[float, ...] = ()
    dependency_shape: Tuple[int, int] = ()
    dependency_dtype: str = ""
    dependency_entries_read: int = 0
    direct_dependency_entries_read: int = 0
    changed_confidence_by_position: Tuple[float, ...] = ()
    cross_block_changed_confidence_by_position: Tuple[float, ...] = ()
    change_confidence_entries_read: int = 0
    change_risk_subtractions: int = 0
    change_risk_multiplications: int = 0
    causal_union_additions: int = 0
    remask_status_entries_read: int = 0
    joint_relation_multiplications: int = 0
    dependency_entries_written: int = 0
    selector_candidate_rows: int = 0
    top_budget_comparisons: int = 0
    refresh_start: int = -1
    refresh_end: int = -1
    active_a4_rows: int = 0
    active_a8_rows: int = 0
    cache_refresh_due_positions_before: Tuple[int, ...] = ()
    cache_refresh_due_positions_after: Tuple[int, ...] = ()
    input_positions: Tuple[int, ...] = ()
    region_start: int = -1
    region_end: int = -1
    refresh_positions: Tuple[int, ...] = ()
    prediction_positions: Tuple[int, ...] = ()
    next_progress_positions: Tuple[int, ...] = ()
    next_added_positions: Tuple[int, ...] = ()
    # Pre-selection candidates and quota after handoff-capacity restriction.
    # None denotes a forward without a future-admission decision.
    next_admission_eligible_a_tokens: int | None = None
    next_admission_eligible_b_tokens: int | None = None
    next_admission_quota: int | None = None
    next_admitted_positions: Tuple[int, ...] = ()
    next_direct_locked_positions: Tuple[int, ...] = ()
    next_tentative_positions: Tuple[int, ...] = ()
    next_confirmed_positions: Tuple[int, ...] = ()
    next_remasked_positions: Tuple[int, ...] = ()
    feature2_local_tail_forward: bool = False
    handoff_verification_positions: Tuple[int, ...] = ()
    current_required_rows: int = 0
    next_progress_rows: int = 0
    next_added_rows: int = 0
    joint_rows: int = 0
    joint_issued_rows: int = 0
    joint_issued_slice_units: int = 0
    joint_pe_issue_groups: int = 0
    joint_a4_rows: int = 0
    joint_a8_rows: int = 0
    qkv_activation_bytes: int = 0
    mixed_k4096_activation_bytes: int = 0
    ffn_down_activation_bytes: int = 0
    next_step_ffn_down_activation_bytes: int = 0
    base_weight_read_multiplier: int = 0
    max_weight_read_multiplier: int = 0
    feature3_extra_weight_read_multiplier: int = 0
    buffer_valid: bool = True
    next_rejected_for_buffer: int = 0
    next_rejected_for_current_step_capacity: int = 0
    next_rejected_for_verification_capacity: int = 0
    next_rejected_for_budget: int = 0
    source_b_optional_proposed_rows: int = 0
    source_b_optional_proposed_a8_rows: int = 0
    source_b_optional_reject_reason: str = "none"
    regular_row_budget_target_active_rows: float = 0.0
    regular_row_budget_allowed_active_rows: int = 0
    regular_row_budget_regular_steps: int = 0
    regular_row_budget_cumulative_active_rows: int = 0


@dataclass
class GenerationStats:
    nfe: int = 0
    direct_lock_tokens: int = 0
    selective_a4_direct_tokens: int = 0
    stable_tentative_tokens: int = 0
    fallback_tentative_tokens: int = 0
    one_step_confirm_lock_tokens: int = 0
    one_step_remask_tokens: int = 0
    forced_finish_tokens: int = 0
    tail_bypass_tokens: int = 0
    tail_bypass_forwards_saved: int = 0
    per_block_nfe: List[int] = field(default_factory=list)
    feature3_a4_rows: int = 0
    feature3_a8_rows: int = 0
    feature3_masked_a4_rows: int = 0
    feature3_tentative_a8_rows: int = 0
    feature3_young_locked_a8_rows: int = 0
    feature3_mature_locked_a4_rows: int = 0
    feature3_activation_bit_sum: int = 0
    feature1_layer0_rows: int = 0
    feature1_prediction_rows: int = 0
    feature1_refresh_rows: int = 0
    feature1_active_rows: int = 0
    feature1_early_exit_rows: int = 0
    feature1_mandatory_refresh_rows: int = 0
    feature1_dependency_rows_updated: int = 0
    feature1_max_dependency_table_bytes: int = 0
    feature1_active_a4_rows: int = 0
    feature1_active_a8_rows: int = 0
    feature1_active_activation_bit_sum: int = 0
    feature2_a4_rows: int = 0
    feature2_a8_rows: int = 0
    feature2_activation_bit_sum: int = 0
    packed_full_sequence_forwards: int = 0
    packed_full_sequence_rows: int = 0
    packed_boundary_forwards: int = 0
    packed_boundary_rows: int = 0
    packed_boundary_a4_rows: int = 0
    packed_boundary_a8_rows: int = 0
    packed_boundary_activation_bit_sum: int = 0
    packed_regular_forwards: int = 0
    dynamic_block_next_progress_rows: int = 0
    dynamic_block_next_added_rows: int = 0
    dynamic_block_next_admitted_tokens: int = 0
    dynamic_block_canonical_direct_tokens: int = 0
    dynamic_block_canonical_handoff_locked_tokens: int = 0
    dynamic_block_next_confirmed_tokens: int = 0
    dynamic_block_next_remasked_tokens: int = 0
    dynamic_block_handoff_locked_tokens: int = 0
    dynamic_block_handoff_verification_tokens: int = 0
    dynamic_block_future_a4_proposals: int = 0
    dynamic_block_spec_tentative_tokens: int = 0
    dynamic_block_future_a8_verifications: int = 0
    dynamic_block_dirty_invalidations: int = 0
    dynamic_block_handoff_a8_confirmed_tokens: int = 0
    dynamic_block_handoff_a8_remasked_tokens: int = 0
    dynamic_block_saved_next_proposals: int = 0
    dynamic_block_saved_handoff_a8_verifications: int = 0
    dynamic_block_max_a8_equivalent_half_rows: int = 0
    dynamic_block_zero_forward_blocks: int = 0
    dynamic_block_current_required_rows: int = 0
    dynamic_block_joint_rows: int = 0
    dynamic_block_joint_issued_rows: int = 0
    dynamic_block_joint_issued_slice_units: int = 0
    dynamic_block_joint_pe_issue_groups: int = 0
    dynamic_block_joint_a4_rows: int = 0
    dynamic_block_joint_a8_rows: int = 0
    dynamic_block_regular_forwards: int = 0
    dynamic_block_buffer_invalid_forwards: int = 0
    dynamic_block_base_weight_read_traversals: int = 0
    dynamic_block_joint_weight_read_traversals: int = 0
    dynamic_block_extra_weight_read_traversals: int = 0
    dynamic_block_max_weight_read_multiplier: int = 0
    dynamic_block_next_rejected_for_buffer: int = 0
    dynamic_block_next_rejected_for_current_step_capacity: int = 0
    dynamic_block_next_rejected_for_verification_capacity: int = 0
    dynamic_block_next_rejected_for_budget: int = 0
    feature2_local_tail_forwards: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
