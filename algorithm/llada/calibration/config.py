"""Generation settings for the two training-source calibration banks."""

from __future__ import annotations
from typing import Any
from numerics.precision import RowPrecisionContext


def generator_arguments(
    task: str, context: RowPrecisionContext, *, feature3: bool,
    steps: int = 256, gen_length: int = 256
) -> dict[str, Any]:
    if task not in {"gsm8k", "humaneval"}:
        raise ValueError(f"unsupported calibration task: {task}")
    if type(gen_length) is not int or gen_length < 32 or gen_length % 32:
        raise ValueError("calibration gen_length must be a positive multiple of 32")
    if type(steps) is not int or steps < 1 or steps % (gen_length // 32):
        raise ValueError("calibration steps must be positive and divisible by the block count")
    gsm = task == "gsm8k"
    arguments = {
        "steps": steps,
        "gen_length": gen_length,
        "block_length": 32,
        "temperature": 0.0,
        "remasking": "low_confidence",
        "mask_id": 126336,
        "tau_high": 0.9,
        "tau_high_tail": 0.75,
        "tau_high_tail_after_step": 8,
        "tau_low": 0.75,
        "confirm_tau": 0.75,
        "selective_a4_direct_tau": 0.9,
        "budget_scale": 26.0 if gsm else 20.0,
        "stability_bonus": 0.05,
        "row_precision_context": context,
        "feature3_maturity_age": 3,
        "feature3_precision_policy": "original",
        "tail_confirmation_policy": "all",
        "candidate_numeric_mode": "bf16_lut",
        "suppressed_candidate_token_ids": (),
        "packed_attention_refresh": True,
        "packed_attention_neighbor_scope": "tri",
        "packed_attention_layer_mode": "all",
        "packed_attention_target_active_rows": 27.5 if gsm else 39.5,
        "packed_attention_initial_burst_rows": 0.0,
        "packed_attention_force_full_current": not gsm,
        "packed_attention_full_current_context_rows": 8,
        "packed_attention_context_a8_rows": 12 if gsm else 10,
        "cross_block_cache_handoff": True,
        "cross_block_boundary_scope": "layer0_attention_two_stage",
        "cross_block_boundary_target_rows": 80,
        "cross_block_full_prefix_oracle_block": -1 if gsm else 1,
        "cross_block_pending_confidence_mode": "all_changes",
        "cross_block_pending_relation_mode": "direct",
        "cross_block_dependency_policy": "initial_keys_selected_rows",
    }
    if feature3:
        arguments.update(
            dynamic_block_lookahead=True,
            dynamic_block_target_joint_rows=48,
            dynamic_block_max_next_rows=32,
            dynamic_block_next_admission_budget=8,
            dynamic_block_min_reuse_score=0.0,
            dynamic_block_next_candidate_policy="high_stable",
            dynamic_block_next_relation_preference="low_dependency",
            dynamic_block_max_current_unresolved=32,
            dynamic_block_max_handoff_verification_rows=32,
            dynamic_block_canonical_future=True,
            dynamic_block_future_horizon=1,
            dynamic_block_canonical_direct_tau=0.75,
            dynamic_block_allow_deferred_verification=True,
            dynamic_block_next_min_confidence=0.75,
        )
    return arguments
