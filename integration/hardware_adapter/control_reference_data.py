"""Generate ATSE, PSME and UAPS control references."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ALIGNMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALIGNMENT_ROOT))

from layer_reference_data import ALGORITHM_ROOT, PROJECT_ROOT, raw_array, compare_arrays
sys.path.insert(0, str(ALGORITHM_ROOT))
import generation.engine as generation
from generation.engine import boundary_deep_precision_bits
from generation.lookahead import (
    DynamicJointWindowScheduler as UtilizationAwarePrefetchScheduler,
)
from generation.refresh import (
    CrossBlockPrefixAttentionState as CrossBlockRefreshState,
    TriBlockAttentionState as AttentionRefreshState,
)
from generation.state import Feature2BlockState as DraftVerifyBlockState
from numerics.precision import RowPrecisionContext as ActivationPrecisionContext
import numpy as np
import torch


_ACTIVATION_TO_ALGORITHM_BITS = {
    "base_activation_bits": "base_row_bits",
    "next_step_base_activation_bits": "next_step_base_row_bits",
    "next_activation_bits": "next_row_bits",
}


def _algorithm_selection_inputs(inputs: dict) -> dict:
    """Translate SUPRA activation fields at the algorithm adapter boundary."""
    return {_ACTIVATION_TO_ALGORITHM_BITS.get(name, name): value
            for name, value in inputs.items()}


def _hardware_selection_inputs(inputs: dict) -> dict:
    reverse = {value: key for key, value in _ACTIVATION_TO_ALGORITHM_BITS.items()}
    return {reverse.get(name, name): value for name, value in inputs.items()}


def _select_prefetch(scheduler, **inputs):
    return scheduler.select(**_algorithm_selection_inputs(inputs))


def _added_activation_bits(selection):
    return selection.added_row_bits


def packed(tensor: torch.Tensor) -> dict:
    if tensor.dtype == torch.float32:
        value = tensor.detach().cpu().contiguous()
        return dict(dtype=str(value.dtype), shape=list(value.shape), raw_dtype="fp32",
            bf16_materialized=torch.equal(value, value.bfloat16().float()),
            raw=value.view(torch.int32).numpy().view("<u4").tolist())
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                raw=raw_array(tensor).tolist())


def make_future_admission_reference_data(*, full_current_context_rows: int = 0,
                                  context_a8_rows: int = 10,
                                  source_b_max_attempts: int = -1,
                                  canonical_direct_tau: float = 0.75,
                                  source_a_handoff: bool | None = None,
                                  feature12_only: bool = False) -> dict:
    """Run canonical future/current decisions with synthetic candidate inputs."""
    from unittest.mock import patch
    records, captures, state_ids, joint_priorities = [], [], {}, []
    precision_records, pending_precision = [], []
    attempt_records, pending_attempts = [], []

    class ScriptedModel:
        device = torch.device("cpu")
        config = SimpleNamespace(n_layers=1)

        def __init__(self):
            self._LLaDAModel__cache = {}
            self.calls = 0

        def __call__(self, input_ids, **kwargs):
            past = kwargs.get("past_key_values")
            query = kwargs.get("query_position_ids")
            if query is None:
                query = (torch.arange(input_ids.shape[1])[None, :] if past is None else
                         kwargs["replace_position"][0].nonzero().flatten()[None, :])
            values = input_ids.float()[:, None, :, None]
            if past is None:
                cache = ((values.clone(), values.clone()),)
            else:
                positions = kwargs.get("kv_write_position_ids")
                if positions is None:
                    positions = kwargs.get("replace_position_kv", kwargs["replace_position"])[0].nonzero().flatten()[None, :]
                local = torch.isin(query[0], positions[0])
                key, value = (tensor.clone() for tensor in past[0])
                key[:, :, positions[0]] = values[:, :, local]
                value[:, :, positions[0]] = values[:, :, local]
                cache = ((key, value),)
            logits = torch.zeros(*input_ids.shape, 4, dtype=torch.bfloat16)
            confidence = torch.full_like(query, 0.8 if self.calls < 2 else 1.0, dtype=torch.bfloat16)
            if source_a_handoff is not None and self.calls == 0:
                confidence[query < 17] = 0.95
            future = query >= 33
            boundary = torch.tensor([0x3f3f, 0x3f40, 0x3f41, 0x3f60], dtype=torch.int16).view(torch.bfloat16)
            confidence[future] = boundary[(query[future]-33) % 4]
            logits[..., 0] = confidence
            region = self._LLaDAModel__cache.get("attn_monitor_query_range")
            if region is not None:
                begin, end = map(int, region.tolist())
                selected = query[0][(query[0] >= begin) & (query[0] < end)]
                dependency = torch.zeros(1, selected.numel(), end-begin, dtype=torch.bfloat16)
                if full_current_context_rows or source_a_handoff is not None or feature12_only:
                    dependency = (((selected[:, None]*3+torch.arange(end-begin)[None, :]*5)%64).float()/256).bfloat16()[None, :]
                self._LLaDAModel__cache["attn_monitor_current"] = dict(
                    query_positions=selected,
                    dependency_mean=dependency,
                    layer_reduction="max", layer_count=torch.tensor(1))
            self.calls += 1
            return SimpleNamespace(logits=logits, past_key_values=cache)

    def snapshot(state):
        return {name: packed(getattr(state, name)) for name in
                ("state", "last_top1", "precision_age", "commit_origin")}

    def capture(event):
        for record, attempts in pending_attempts:
            record["expected"]["source_b_attempts_before_forward"] = packed(attempts)
            record["forward_kind"] = event.forward_kind
        pending_attempts.clear()
        for record in pending_precision:
            positions = record.pop("_positions")
            frame = record.pop("_frame")
            local = frame.f_locals
            physical = {int(position): index for index, position in enumerate(event.input_positions.reshape(-1))}
            indices = torch.tensor([physical[int(position)] for position in positions], dtype=torch.long)
            record.update(capture_index=event.capture_index, block_index=event.block_index,
                step_index=event.step_index, forward_kind=event.forward_kind,
                expected_activation_bits=packed(event.row_bits.reshape(-1).index_select(0, indices)),
                tail_transfer_active=bool(local.get("tail_transfer_active", False)))
            if record["expected_upgrade_count"] and "expected_upgrade_order" not in record:
                raise RuntimeError("context ranking was not observed before forward capture")
            precision_records.append(record)
        pending_precision.clear()
        captures.append(dict(capture_index=event.capture_index, block_index=event.block_index,
            step_index=event.step_index, forward_kind=event.forward_kind,
            input_positions=packed(event.input_positions), activation_bits=packed(event.row_bits),
            prediction_positions=packed(event.prediction_positions),
            state=packed(event.block_state_before), tokens=packed(event.tokens_before)))
        if source_a_handoff is not None:
            frame = sys._getframe(1)
            while frame is not None and "source_a_handoff_pending" not in frame.f_locals:
                frame = frame.f_back
            if frame is None:
                raise RuntimeError("forward capture lacks Source A handoff state")
            local = frame.f_locals
            captures[-1].update(
                refresh_positions=packed(event.refresh_positions),
                source_a_pending=packed(local["source_a_handoff_pending"]),
                future_prediction_positions=packed(local["next_progress_positions"]),
                source_a_positions=packed(local["next_source_a_positions"]))
            if local.get("next_block_state") is not None:
                captures[-1]["future_state"] = snapshot(local["next_block_state"])

    original_admit = DraftVerifyBlockState.admit
    original_age = DraftVerifyBlockState.advance_locked_age
    original_select = generation.select_admissions
    original_joint_select = UtilizationAwarePrefetchScheduler.select
    original_context_upgrade = generation.context_a8_upgrade_count
    original_argsort = torch.argsort

    def observe_context_sort(value, *args, **kwargs):
        order = original_argsort(value, *args, **kwargs)
        local = sys._getframe(1).f_locals
        if pending_precision and value is local.get("attention_values"):
            record = pending_precision[-1]
            record["context_scores"] = packed(value)
            record["expected_upgrade_order"] = packed(order[:record["expected_upgrade_count"]])
        return order

    def context_upgrade(activation_bits, context_candidate_count, **kwargs):
        count = original_context_upgrade(activation_bits, context_candidate_count, **kwargs)
        frame = sys._getframe(1)
        local = frame.f_locals
        if "mandatory_current" in local and "packed_input_positions" in local:
            positions = local["packed_input_positions"].detach().clone()
            pending_precision.append(dict(_frame=frame, _positions=positions,
                input_positions=packed(positions), input_activation_bits=packed(activation_bits),
                mandatory_current=packed(local["mandatory_current"]),
                context_candidates=packed(local["context_candidates"]),
                context_candidate_count=context_candidate_count,
                parameters={("fixed_context_a8_token_count" if key == "fixed_context_a8_rows" else key): value
                            for key, value in kwargs.items()}, expected_upgrade_count=count))
        return count

    def joint_select(scheduler, **kwargs):
        # Read the actual generator's rank inputs at its scheduler call site.
        caller = sys._getframe(1).f_locals
        available = int(caller["available"])
        region = int(caller["next_region_start"])
        dependency = caller["packed_state"].dependency_score[region:region + available]
        result = original_joint_select(scheduler, **kwargs)
        selection_inputs = dict(kwargs)
        if selection_inputs.get("next_service_count") is None:
            selection_inputs["next_service_count"] = torch.zeros_like(kwargs["next_row_bits"], dtype=torch.int32)
        joint_priorities.append(dict(
            name=f"generator_forward_{len(joint_priorities)}",
            available=available,
            low_dependency=caller["dynamic_block_next_relation_preference"] == "low_dependency",
            dependency=packed(dependency),
            tentative=packed(kwargs["next_tentative"]),
            existing_local=packed(caller["existing_local"]),
            expected_priority=packed(kwargs["next_priority"]),
            inputs={key: packed(value) if isinstance(value, torch.Tensor) else value
                    for key, value in _hardware_selection_inputs(selection_inputs).items()},
            expected=dict(progress_local_positions=packed(result.progress_local_positions),
                added_local_positions=packed(result.added_local_positions),
                base_residency=asdict(result.base_residency), joint_residency=asdict(result.joint_residency),
                next_step_verification_residency=asdict(result.next_step_verification_residency))))
        if source_b_max_attempts >= 0:
            record = dict(name=f"canonical_select_{len(attempt_records)}", kind="source_b_generator",
                block_index=int(caller["block_index"]), step_index=int(caller["step_index"]),
                parameters=dict(target_joint_rows=scheduler.target_joint_rows,
                    max_next_rows=scheduler.max_next_rows),
                inputs={key: packed(value) if isinstance(value, torch.Tensor) else value
                        for key, value in _hardware_selection_inputs(kwargs).items()},
                expected=dict(progress_local_positions=packed(result.progress_local_positions),
                    added_local_positions=packed(result.added_local_positions),
                    added_activation_bits=packed(_added_activation_bits(result))))
            attempt_records.append(record)
            pending_attempts.append((record, kwargs["source_b_attempts"]))
        return result

    def admit(state, tokens, proposal, selected, high, stable_low):
        identifier = state_ids.setdefault(id(state), len(state_ids))
        record = dict(event="admit", state_id=identifier, forward_capture=len(captures)-1,
            token_storage_offset=tokens.storage_offset(), before=snapshot(state), tokens_before=packed(tokens),
            proposal=packed(proposal), selected=packed(selected), high=packed(high), stable_low=packed(stable_low))
        result = original_admit(state, tokens, proposal, selected, high, stable_low)
        record.update(after=snapshot(state), tokens_after=packed(tokens),
            direct=packed(result.direct_locked), tentative=packed(result.stable_tentative),
            fallback=packed(result.fallback_tentative))
        records.append(record)
        return result

    def age(state, observed):
        before = snapshot(state)
        original_age(state, observed)
        records.append(dict(event="age", state_id=state_ids.setdefault(id(state), len(state_ids)),
            forward_capture=len(captures)-1, observed_locked=packed(observed), before=before, after=snapshot(state)))

    def select(*args, **kwargs):
        scores = []
        result = original_select(*args, score_observer=lambda value: scores.append(packed(value)), **kwargs)
        records.append(dict(event="select", forward_capture=len(captures)-1,
            admission_mask=packed(args[0]), high=packed(args[1]), stable_low=packed(args[2]),
            stable=packed(args[3]), confidence=packed(args[4]), quota=packed(args[5]),
            score=scores[0], selected=packed(result), parameters=kwargs))
        return result

    def candidate(logits):
        token = logits.argmax(dim=-1)
        confidence = logits.gather(-1, token[..., None]).squeeze(-1).bfloat16()
        return token, confidence, confidence

    config = dict(steps=8, gen_length=64, block_length=32, mask_id=3,
        tau_high=0.90, tau_low=0.75, confirm_tau=0.75, selective_a4_direct_tau=0.90,
        stability_bonus=0.05, budget_scale=1.0, candidate_numeric_mode="bf16_lut",
        packed_attention_refresh=True, packed_attention_force_full_current=True,
        packed_attention_full_current_context_rows=full_current_context_rows,
        packed_attention_target_active_rows=39.5 if full_current_context_rows else 32.0,
        packed_attention_initial_burst_rows=0.0, packed_attention_neighbor_scope="current_next",
        packed_attention_context_a8_rows=context_a8_rows,
        dynamic_block_lookahead=True, dynamic_block_target_joint_rows=48, dynamic_block_max_next_rows=32,
        dynamic_block_next_admission_budget=8, dynamic_block_min_reuse_score=0.0,
        dynamic_block_next_candidate_policy="high_stable", dynamic_block_next_relation_preference="low_dependency",
        dynamic_block_max_current_unresolved=32, dynamic_block_max_handoff_verification_rows=32,
        dynamic_block_canonical_future=True, dynamic_block_canonical_direct_tau=canonical_direct_tau,
        dynamic_block_next_min_confidence=0.75, dynamic_block_allow_deferred_verification=True,
        suppressed_candidate_token_ids=(3,))
    if source_a_handoff is not None:
        config.update(packed_attention_force_full_current=False,
            packed_attention_full_current_context_rows=0,
            packed_attention_neighbor_scope="tri", packed_attention_target_active_rows=44.0,
            budget_scale=16.0, packed_attention_context_a8_rows=0,
            dynamic_block_source_a_confirm_at_handoff=source_a_handoff)
    if source_b_max_attempts >= 0:
        config["dynamic_block_source_b_max_attempts"] = source_b_max_attempts
    if feature12_only:
        if source_a_handoff is not None or full_current_context_rows or source_b_max_attempts >= 0:
            raise ValueError("Feature12 precision inputs require source_a_handoff=None, full_current_context_rows=0 and source_b_max_attempts=-1")
        config = {key: value for key, value in config.items() if not key.startswith("dynamic_block_")}
        config.update(packed_attention_force_full_current=False,
            packed_attention_full_current_context_rows=0,
            packed_attention_neighbor_scope="tri", packed_attention_target_active_rows=27.5,
            budget_scale=26.0, dynamic_block_lookahead=False)
    with patch.object(generation, "streaming_candidate_bf16", side_effect=candidate),\
         patch.object(generation, "select_admissions", side_effect=select),\
         patch.object(UtilizationAwarePrefetchScheduler, "select", joint_select),\
         patch.object(generation, "context_a8_upgrade_count", context_upgrade),\
         patch.object(torch, "argsort", observe_context_sort),\
         patch.object(DraftVerifyBlockState, "admit", admit),\
         patch.object(DraftVerifyBlockState, "advance_locked_age", age):
        tokens, nfe, trace, stats = generation.generate(
            ScriptedModel(), torch.tensor([[2]]), row_precision_context=ActivationPrecisionContext(),
            state_capture_callback=capture, **config)
    transitions = [{key: value for key, value in asdict(item).items() if
        key in ("block_index", "step_index", "forward_kind") or
        "dynamic_block" in key or "handoff" in key} for item in trace]
    return dict(schema="supra-uaps-future-token-control/v1", config=config,
        captures=captures, records=records, transitions=transitions, joint_priorities=joint_priorities,
        packed_precision_records=precision_records,
        source_b_attempt_records=attempt_records,
        nfe=nfe, final_tokens=packed(tokens),
        stats={key: value for key, value in asdict(stats).items() if
               "canonical_direct" in key or "next_admitted" in key or "handoff_locked" in key})


def make_feature12_precision_reference_data() -> dict:
    records, configurations = [], []
    for quota in (12, 0):
        generated = make_future_admission_reference_data(feature12_only=True, context_a8_rows=quota)
        configurations.append(generated["config"])
        for item in generated["packed_precision_records"]:
            if not item["context_candidate_count"]:
                continue
            record = dict(item, task=f"feature12_quota{quota}")
            records.append(record)
    return dict(schema="supra-psme-context-precision/v1",
        source="generation.engine.generate/context_a8_upgrade_count; actual forward capture",
        configurations=configurations,
        records=records)


def make_feature12_boundary_precision_reference_data() -> dict:
    """Observe actual two-stage boundary precision with synthetic layer/scout data."""
    from unittest.mock import patch
    sequence = 128
    current = torch.arange(64, 96)
    mandatory = torch.tensor([2, 37, 112])
    eligible = torch.arange(sequence)[~torch.isin(torch.arange(sequence), current)]
    scores = ((torch.arange(sequence) * 17) % 31).to(torch.uint8)
    current_bits = torch.tensor([4, 8] * 16, dtype=torch.int8)
    records = []
    for context_bits, limit in ((4, 0), (8, 24)):
        precision = ActivationPrecisionContext()
        observed = []

        class Layer:
            def __call__(self, hidden, **kwargs):
                observed.append(precision.row_bits.clone())
                return hidden, kwargs["layer_past"]

        backbone = SimpleNamespace(config=SimpleNamespace(input_emb_norm=False, scale_logits=False),
            transformer=SimpleNamespace(wte=lambda value: value.unsqueeze(-1).float(),
                emb_drop=lambda value: value, blocks=[Layer(), Layer()],
                ln_f=lambda value: value, ff_out=lambda value: value))
        cache = tuple((torch.zeros(1, 1, sequence, 1), torch.zeros(1, 1, sequence, 1)) for _ in range(2))
        # Only Attention and layer arithmetic are replaced. The complete algorithm
        # boundary function produces the selection and both precision vectors.
        with patch.object(generation, "begin_boundary_layer0_scout"),\
             patch.object(generation, "end_boundary_layer0_scout"),\
             patch.object(generation, "read_boundary_layer0_scout", return_value=(scores, 0)):
            result = generation._layer0_global_then_attention_guided_deep(
                SimpleNamespace(model=backbone), precision, tokens=torch.zeros(1, sequence, dtype=torch.long),
                stale_cache=cache, current_positions=current, current_row_bits=current_bits,
                target_rows=88, deep_candidate_positions=eligible,
                mandatory_candidate_positions=mandatory, global_context_bits=context_bits,
                global_prefix_bits=context_bits, deep_a8_row_limit=limit)
        positions, bits = result[2:4]
        if len(observed) != 2 or not torch.equal(observed[1], bits):
            raise ValueError("boundary capture is missing deep-layer precision")
        records.append(dict(name=f"feature12_boundary_context{context_bits}_limit{limit}",
            inputs=dict(score_q8=packed(scores), current_positions=packed(current),
                mandatory_candidate_positions=packed(mandatory), eligible_positions=packed(eligible),
                required_candidate_positions=packed(current[:0]), required_candidate_count=0,
                initial_activation_bits=packed(observed[0]), target_token_count=88, deep_a8_limit=limit),
            expected=dict(deep_positions=packed(positions), deep_bits=packed(bits))))
    return dict(schema="supra-atse-cross-block-token-selection/v1",
        source="generation.engine._layer0_global_then_attention_guided_deep",
        records=records)


def make_saved_dependency_reference_data(seed: int = 20260905) -> dict:
    generator = torch.Generator().manual_seed(seed)
    region_start, rows, total_length = 5, 12, 23
    positions = torch.arange(region_start, region_start + rows)

    def profile(query_positions: torch.Tensor) -> dict:
        values = torch.randint(0, 65, (1, query_positions.numel(), rows),
                               generator=generator).float().div(256).bfloat16()
        return dict(query_positions=query_positions.clone(), dependency_mean=values)

    full_sequence = profile(positions)
    state = AttentionRefreshState(region_start=region_start,
        region_end=region_start + rows, total_length=total_length,
        initial_attention_profile=full_sequence, target_active_rows=6.5)
    records = []
    query_positions = positions
    for step, (changed_rows, remasked_rows) in enumerate(
            [([3], []), ([2, 7, 10], [2, 10]), ([], [])]):
        observed = profile(query_positions)
        changed = torch.zeros((1, total_length), dtype=torch.bool)
        remasked = torch.zeros_like(changed)
        confidence = torch.ones((1, total_length), dtype=torch.bfloat16)
        changed[0, [region_start + row for row in changed_rows]] = True
        remasked[0, [region_start + row for row in remasked_rows]] = True
        confidence[0, region_start + 2] = 0.75
        confidence[0, region_start + 10] = 0.5
        predicted = torch.zeros(rows, dtype=torch.bool)
        predicted[4:6] = True
        before = dict(dependency=packed(state.dependency), pending=packed(state.pending),
                      refresh=packed(state.refresh), regular_steps=state.regular_steps,
                      cumulative_active_rows=state.cumulative_active_rows)
        observation = state.observe(observed, predicted=predicted,
            changed_global=changed, changed_confidence_global=confidence,
            changed_remask_global=remasked)
        records.append(dict(step_index=step, inputs={
            **{name: packed(value) for name, value in observed.items()},
            "predicted": packed(predicted), "changed_global": packed(changed),
            "changed_confidence_global": packed(confidence),
            "changed_remask_global": packed(remasked)}, before=before,
            expected=dict(dependency=packed(state.dependency), pending=packed(state.pending),
                dependency_score=packed(observation.dependency_score),
                new_invalidation=packed(observation.new_invalidation),
                refreshed=packed(observation.refreshed),
                next_refresh=packed(observation.next_refresh),
                mandatory=packed(observation.mandatory),
                optional_selected=packed(observation.optional_selected),
                dependency_rows_updated=observation.dependency_rows_updated,
                allowed_active_rows=state.allowed_active_rows,
                regular_steps=state.regular_steps,
                cumulative_active_rows=state.cumulative_active_rows)))
        query_positions = positions[state.refresh]
    return dict(schema="supra-atse-attention-dependencies/v1", seed=seed,
        source="generation.refresh.AttentionRefreshState",
        region_start=region_start, region_end=region_start + rows,
        total_length=total_length, target_active_rows=6.5,
        initial_attention_profile={name: packed(value) for name, value in full_sequence.items()}, records=records)


def make_regular_boundary_reference_data() -> dict:
    begin, rows, length = 5, 12, 23
    positions = torch.arange(begin, begin + rows)
    records = []
    for name in ("zero_scores", "mandatory_over_budget"):
        profile = dict(query_positions=positions,
            dependency_mean=torch.zeros((1, rows, rows), dtype=torch.bfloat16))
        state = AttentionRefreshState(region_start=begin, region_end=begin + rows,
            total_length=length, initial_attention_profile=profile, target_active_rows=6.5)
        predicted = torch.zeros(rows, dtype=torch.bool)
        if name == "mandatory_over_budget":
            state.account_initial_regular_forward(rows)
            predicted[:8] = True
        changed = torch.zeros((1, length), dtype=torch.bool)
        confidence = torch.ones((1, length), dtype=torch.bfloat16)
        before = dict(dependency=packed(state.dependency), pending=packed(state.pending),
            refresh=packed(state.refresh), regular_steps=state.regular_steps,
            cumulative_active_rows=state.cumulative_active_rows)
        observation = state.observe(profile, predicted=predicted, changed_global=changed,
            changed_confidence_global=confidence, changed_remask_global=changed)
        records.append(dict(name=name, step_index=0, before=before,
            inputs={**{key: packed(value) for key, value in profile.items()},
                "predicted": packed(predicted), "changed_global": packed(changed),
                "changed_remask_global": packed(changed), "changed_confidence_global": packed(confidence)},
            expected=dict(dependency=packed(state.dependency), pending=packed(state.pending),
                new_invalidation=packed(observation.new_invalidation), next_refresh=packed(observation.next_refresh),
                regular_steps=state.regular_steps, cumulative_active_rows=state.cumulative_active_rows)))
    return dict(schema="supra-atse-in-block-boundaries/v1", region_start=begin, region_end=begin+rows,
        total_length=length, target_active_rows=6.5,
        records=records)


def _hardware_residency(value) -> dict:
    fields = asdict(value)
    fields["a4_token_count"] = fields.pop("a4_rows")
    fields["a8_token_count"] = fields.pop("a8_rows")
    return fields


def make_regular_joint_reference_data() -> dict:
    scheduler = UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=32)
    records = []
    root = Path(__file__).resolve().parents[2] / "hardware/cases/control"
    for source_name, record_index, future_rows in (
            ("atse_attention_dependencies.json", 1, 32),
            ("atse_attention_dependencies.json", 0, 8),
            ("atse_in_block_boundaries.json", 1, 0)):
        source = json.loads((root / source_name).read_text())
        regular = source["records"][record_index]
        positions = torch.tensor(regular["expected"]["next_refresh"]["raw"]).nonzero().flatten() + source["region_start"]
        bits = torch.where(positions % 3 == 0, 4, 8).to(torch.int8)
        inputs = dict(base_positions=positions, base_activation_bits=bits,
            next_step_base_activation_bits=torch.full_like(bits, 8), next_block_start=64,
            next_activation_bits=torch.where(torch.arange(future_rows) % 3 == 0, 8, 4).to(torch.int8),
            next_unresolved=torch.ones(future_rows, dtype=torch.bool),
            next_tentative=torch.arange(future_rows) % 3 == 0,
            next_priority=torch.tensor([1.0, 0.25, 0.5, 0.00390625] * 8, dtype=torch.bfloat16)[:future_rows],
            next_service_count=torch.arange(future_rows, dtype=torch.int32) % 3)
        selected = _select_prefetch(scheduler, **inputs)
        records.append(dict(name=f"regular_joint_{len(records)}", regular_source=source_name,
            regular_record_index=record_index,
            inputs={key: packed(value) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()},
            expected=dict(progress_local_positions=packed(selected.progress_local_positions),
                added_local_positions=packed(selected.added_local_positions),
                added_activation_bits=packed(_added_activation_bits(selected)),
                base_residency=_hardware_residency(selected.base_residency), joint_residency=_hardware_residency(selected.joint_residency),
                next_step_verification_residency=_hardware_residency(selected.next_step_verification_residency))))
    return dict(schema="supra-atse-uaps-in-block/v1", target_joint_rows=48, max_next_rows=32,
        records=records)


def make_precision_joint_reference_data() -> dict:
    source = Path(__file__).resolve().parents[2] / "hardware/cases/control/psme_context_precision.json"
    precision = json.loads(source.read_text())["records"]
    scheduler = UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=8)
    records = []
    for index, record in enumerate(precision):
        if record["block_index"] != 0 or record["step_index"] != 1:
            continue
        bits = torch.tensor(record["expected_activation_bits"]["raw"], dtype=torch.int8)
        inputs = dict(base_positions=torch.tensor(record["input_positions"]["raw"]),
            base_activation_bits=bits, next_step_base_activation_bits=bits.clone(), next_block_start=64,
            next_activation_bits=torch.full((8,), 8, dtype=torch.int8),
            next_unresolved=torch.ones(8, dtype=torch.bool), next_tentative=torch.zeros(8, dtype=torch.bool),
            next_priority=torch.ones(8, dtype=torch.bfloat16), next_service_count=torch.zeros(8, dtype=torch.int32))
        selected = _select_prefetch(scheduler, **inputs)
        old = dict(inputs, base_activation_bits=torch.tensor(record["input_activation_bits"]["raw"], dtype=torch.int8),
            next_step_base_activation_bits=torch.tensor(record["input_activation_bits"]["raw"], dtype=torch.int8))
        unallocated = _select_prefetch(scheduler, **old)
        if torch.equal(selected.added_local_positions, unallocated.added_local_positions):
            raise RuntimeError("precision reference_data does not distinguish stale joint input bits")
        parameters = record["parameters"]
        quota = parameters["fixed_context_a8_token_count"]
        records.append(dict(name=f"precision_capacity_quota{quota}", precision_record_index=index,
            inputs={key: packed(value) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()},
            expected=dict(progress_local_positions=packed(selected.progress_local_positions),
                added_local_positions=packed(selected.added_local_positions),
                added_activation_bits=packed(_added_activation_bits(selected)),
                base_residency=_hardware_residency(selected.base_residency), joint_residency=_hardware_residency(selected.joint_residency),
                next_step_verification_residency=_hardware_residency(selected.next_step_verification_residency),
                rejected_for_current_step_capacity=selected.rejected_for_current_step_capacity,
                rejected_for_next_step_verification_capacity=selected.rejected_for_next_step_verification_capacity,
                rejected_for_budget=selected.rejected_for_budget),
            unallocated_added_local_positions=packed(unallocated.added_local_positions)))
    return dict(schema="supra-psme-uaps-context-precision/v1", target_joint_rows=48, max_next_rows=8,
        precision_source=source.name,
        records=records)


def _joint_state_program():
    """Execute the generator's eligibility/forecast block and scheduler.

    Extract the actual AST through its selection call, not a rewritten rule.
    Synthetic full block states exercise limits without a model/GPU forward.
    """
    import ast
    import copy
    import inspect
    source_path = inspect.getsourcefile(inspect.unwrap(generation.generate))
    tree = ast.parse(Path(source_path).read_text())
    choices = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and "current_unresolved_count" in ast.dump(node.test)
               and "dynamic_block_max_current_unresolved" in ast.dump(node.test)]
    if len(choices) != 1:
        raise ValueError("generator UAPS selection trigger was not found")
    branch = copy.deepcopy(choices[0])
    selections = [i for i, node in enumerate(branch.body) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "selection" for t in node.targets)]
    if len(selections) != 1:
        raise ValueError("generator UAPS selection boundary was not found")
    branch.body = branch.body[:selections[0]+1]
    branch.orelse = []
    return source_path, compile(ast.Module(body=[branch], type_ignores=[]), source_path, "exec")


def select_live_joint_reference(inputs: dict) -> dict:
    """Run the generator on independently evolved C numerical state.

    Inputs provide states, dependencies and Source B flags.
    """
    source_path, program = _joint_state_program()
    current = DraftVerifyBlockState((1, 32), device="cpu")
    future = DraftVerifyBlockState((1, 32), device="cpu")
    current.state[0] = torch.tensor(inputs["states"][:32], dtype=torch.int8)
    future.state[0] = torch.tensor(inputs["states"][32:], dtype=torch.int8)
    future.precision_age[0] = torch.tensor(inputs["ages"][32:], dtype=torch.int16)
    base = torch.tensor(inputs["base_positions"], dtype=torch.long)
    bits = torch.tensor(inputs["base_bits"], dtype=torch.int8)
    start, sequence = inputs["current_begin"], inputs["sequence_length"]
    if len(inputs["states"]) != 64 or start + 64 > sequence:
        raise ValueError("live joint reference requires two complete blocks")
    extension, joint = inputs["extension"], inputs["joint"]
    def bf16(raw):
        return torch.tensor(raw, dtype=torch.int32).short().view(torch.bfloat16)
    pending = torch.zeros((1, sequence), dtype=torch.bool)
    pending[0, start:start+64] = torch.tensor(inputs["source_a_pending"], dtype=torch.bool)
    scheduler = UtilizationAwarePrefetchScheduler(target_joint_rows=joint["target_token_count"],
        max_next_rows=joint["max_next_tokens"], min_reuse_score=float(bf16(extension.get("min_reuse_score_bf16", 0))),
        source_b_dependency_tie_rank=bool(extension.get("source_b_flags", 0) & 1),
        source_b_a4_only=bool(extension.get("source_b_flags", 0) & 2))
    scope = dict(vars(generation))
    scope.update(next_block_state=future, block_state=current,
        current_unresolved_count=int((current.state != 2).sum()),
        dynamic_block_max_current_unresolved=extension["max_current_unresolved"], block_length=32,
        packed_row_bits=bits, packed_input_positions=base, joint_scheduler=scheduler,
        block_start=start, block_end=start+32, next_block_rows=32, region_start=start, region_length=64,
        feature3_maturity_age=3, feature3_precision_policy="original", tokens=torch.zeros((1,sequence), dtype=torch.long),
        packed_state=SimpleNamespace(dependency_score=bf16(inputs["dependency_bf16"])),
        dynamic_block_next_relation_preference="low_dependency", future_block_count=1,
        next_preconfirmed=torch.zeros((1,32),dtype=torch.bool),
        dynamic_block_next_admission_budget=extension["admission_budget"],
        dynamic_block_max_handoff_verification_rows=extension["max_handoff_tokens"],
        dynamic_block_source_a_confirm_at_handoff=bool(joint["future_block_flags"] & 128),
        source_a_handoff_pending=pending,
        dynamic_block_allow_deferred_verification=bool(extension["state_control"] & 2),
        next_source_b_attempts=torch.tensor(inputs["attempts"],dtype=torch.int32),
        dynamic_block_source_b_max_attempts=inputs["max_attempts"],
        dynamic_block_source_b_retry_min_confidence=float(bf16(extension.get("retry_min_confidence_bf16",0))),
        next_last_confidence=bf16(inputs["last_confidence_bf16"]),
        dynamic_block_target_prediction_rows=extension.get("prediction_target",0) or -1,
        prediction_positions=torch.tensor(inputs["prediction_positions"],dtype=torch.long),
        total_length=sequence, step_index=inputs["step_index"], next_frontier_service_count=None)
    exec(program,scope)
    selection = scope.get("selection")
    baseline = scheduler.profile.analyze(bits)
    def residency(value):
        return dict(a4_tokens=value.a4_rows,a8_tokens=value.a8_rows)
    return dict(source=source_path, progress=[] if selection is None else selection.progress_local_positions.tolist(),
        added=[] if selection is None else selection.added_local_positions.tolist(),
        added_bits=[] if selection is None else selection.added_row_bits.tolist(),
        base=residency(baseline if selection is None else selection.base_residency),
        joint=residency(baseline if selection is None else selection.joint_residency),
        forecast=residency(baseline if selection is None else selection.next_step_verification_residency))


def make_joint_state_reference_data() -> dict:
    source_path, program = _joint_state_program()
    records = []
    configurations = [(*case, 0, 1) for case in (
            ("capacity_full", 3, 3, 0, True, 0.),
            ("capacity_open", 3, 4, 0, True, 0.),
            ("overlap_sufficient", 3, 3, 1, True, 0.),
            ("current_over_limit", 2, 4, 1, True, 0.),
            ("forecast_reserve", 32, 4, 1, False, 0.),
            ("pending_over_limit", 32, 1, 1, True, 0.),
            ("rank_reuse_floor", 32, 32, 0, True, .75))]
    configurations += [(f"source_b_flags{flags}_step{step}", 32, 32, 1, True, 0., flags, step)
                       for flags in range(4) for step in (1, 2)]
    for name, max_current, max_handoff, context_a8, deferred, reuse, flags, step in configurations:
        current = DraftVerifyBlockState((1, 32), device="cpu")
        current.state.fill_(2); current.state[0, :2] = 0; current.state[0, 2] = 1
        current.precision_age.fill_(3)
        future = DraftVerifyBlockState((1, 32), device="cpu")
        future.state[0, [0, 2, 3]] = 1; future.state[0, 31] = 2
        future.precision_age.fill_(3)
        base = torch.tensor([32, 33, 34, 64, 65, 66])
        base_bits = torch.tensor([4, 4, 8, 8 if context_a8 else 4, 4, 4], dtype=torch.int8)
        score = torch.zeros(64, dtype=torch.bfloat16)
        score[32:] = torch.arange(32, dtype=torch.float32).div(128).bfloat16()
        score[32:35] = torch.tensor([.75, .625, .5], dtype=torch.bfloat16)
        if name.startswith("source_b_"):
            score[35:] = 0.25  # Raw dependency ties with distinct stable-rank priorities.
        pending = torch.zeros((1, 96), dtype=torch.bool); pending[0, 66] = True
        scope = dict(vars(generation))
        scope.update(next_block_state=future, block_state=current, current_unresolved_count=3,
            dynamic_block_max_current_unresolved=max_current, block_length=32,
            packed_row_bits=base_bits.clone(), packed_input_positions=base,
            joint_scheduler=UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=8, min_reuse_score=reuse,
                source_b_dependency_tie_rank=bool(flags & 1), source_b_a4_only=bool(flags & 2)),
            block_start=32, block_end=64, next_block_rows=32, region_start=32, region_length=64,
            feature3_maturity_age=3, feature3_precision_policy="original", tokens=torch.zeros((1,96), dtype=torch.long),
            packed_state=SimpleNamespace(dependency_score=score), dynamic_block_next_relation_preference="low_dependency",
            future_block_count=1, next_preconfirmed=torch.zeros((1,32), dtype=torch.bool),
            dynamic_block_next_admission_budget=2, dynamic_block_max_handoff_verification_rows=max_handoff,
            dynamic_block_source_a_confirm_at_handoff=True, source_a_handoff_pending=pending,
            dynamic_block_allow_deferred_verification=deferred, next_source_b_attempts=None,
            dynamic_block_source_b_max_attempts=-1, dynamic_block_source_b_retry_min_confidence=0.,
            next_last_confidence=None, dynamic_block_target_prediction_rows=-1,
            prediction_positions=torch.tensor([32,33,34]), total_length=96, step_index=step,
            next_frontier_service_count=None)
        exec(program, scope)
        selection = scope.get("selection")
        records.append(dict(name=name, current_states=current.state[0].tolist(),
            future_states=future.state[0].tolist(), future_ages=future.precision_age[0].tolist(),
            source_b_flags=flags, block_step_index=step, source_a_pending_positions=[66],
            base_positions=base.tolist(), base_bits=base_bits.tolist(),
            dependency_bf16=packed(score)["raw"], max_current=max_current, max_handoff=max_handoff,
            context_a8=context_a8, allow_deferred=deferred, admission_budget=2,
            min_reuse_score_bf16=int(torch.tensor(reuse,dtype=torch.bfloat16).view(torch.int16)),
            expected=dict(progress=[] if selection is None else selection.progress_local_positions.tolist(),
                added=[] if selection is None else selection.added_local_positions.tolist(),
                added_bits=[] if selection is None else selection.added_row_bits.tolist(),
                forecast_base_bits=scope.get("next_step_base_row_bits",base_bits).tolist(),
                quota=scope.get("next_admission_quota"), triggered=selection is not None)))
    return dict(schema="supra-joint-state-selection/v1", source=str(Path(source_path).relative_to(PROJECT_ROOT)),
         records=records)


def make_source_b_selection_reference_data() -> dict:
    """Scheduler decisions for independent B flags and step ties."""
    def residency(value):
        fields = asdict(value)
        fields["a4_token_count"] = fields.pop("a4_rows")
        fields["a8_token_count"] = fields.pop("a8_rows")
        return fields

    records = []
    for shape in ("dependency_distinct", "dependency_ties", "optional_a8_winner"):
        for flags in range(4):
            for step in (1, 2):
                base = torch.arange(32)
                base[-1] = 64  # Source A, which must survive B rejection.
                bits = torch.full((32,), 4, dtype=torch.int8)
                bits[-1] = 8
                future = torch.full((12,), 4, dtype=torch.int8)
                due = torch.zeros(12, dtype=torch.bool)
                due[1] = True
                future[0] = future[1] = 8
                if shape == "optional_a8_winner":
                    future[2] = 8
                priority = torch.tensor([1., 1., 1., .248046875, .25, .498046875,
                    .5, .74609375, .75, .875, .125, .375], dtype=torch.bfloat16)
                dependency = (torch.arange(12).to(torch.bfloat16) / 16 if
                    shape == "dependency_distinct" else torch.zeros(12, dtype=torch.bfloat16))
                inputs = dict(base_positions=base, base_activation_bits=bits,
                    next_step_base_activation_bits=bits.clone(), next_block_start=64,
                    next_activation_bits=future, next_unresolved=torch.ones(12, dtype=torch.bool),
                    next_tentative=due, next_priority=priority, next_dependency_score=dependency,
                    next_service_count=torch.zeros(12, dtype=torch.int32), step_index=step)
                selected = _select_prefetch(UtilizationAwarePrefetchScheduler(
                    target_joint_rows=48, max_next_rows=8,
                    source_b_dependency_tie_rank=bool(flags & 1),
                    source_b_a4_only=bool(flags & 2)), **inputs)
                records.append(dict(name=f"{shape}_flags{flags}_step{step}", source_b_flags=flags,
                    inputs={k: packed(v) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()},
                    expected=dict(progress_local_positions=packed(selected.progress_local_positions),
                        added_local_positions=packed(selected.added_local_positions),
                        added_activation_bits=packed(selected.added_row_bits),
                        base_residency=residency(selected.base_residency),
                        joint_residency=residency(selected.joint_residency),
                        next_step_verification_residency=residency(selected.next_step_verification_residency))))
    return dict(schema="supra-uaps-token-selection/v1", target_joint_tokens=48, max_next_tokens=8,
        records=records)


def make_joint_selection_reference_data(case_names: set[str] | None = None) -> dict:
    scheduler = UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=32)
    records = []
    cases = (
            ("base27_add8", 27, 14, 0, 8),
            ("forecast_capacity", 46, 16, 0, 8),
            ("overlap_mixed_order", 32, 3, 3, 8),
            ("no_future", 27, 14, 0, 0),
            ("current_capacity", 40, 20, 0, 8),
            ("overlap_forecast_reject", 46, 18, 3, 8),
            ("priority_rounding", 27, 14, 0, 16),
            ("reuse_floor_equal", 27, 14, 3, 8),
            ("reuse_floor_above", 27, 14, 3, 8))
    if case_names is not None and not case_names <= {case[0] for case in cases}:
        raise ValueError("unknown joint selection case")
    for name, base_rows, a8_rows, overlap, future_count in cases:
        if case_names is not None and name not in case_names:
            continue
        base_positions = torch.arange(base_rows)
        base_bits = torch.full((base_rows,), 4, dtype=torch.int8)
        base_bits[:a8_rows] = 8
        next_bits = torch.full((32,), 4, dtype=torch.int8)
        tentative = torch.zeros(32, dtype=torch.bool)
        if overlap:
            base_positions[-overlap:] = torch.arange(64, 64 + overlap)
            base_bits.fill_(4)
            base_bits[-overlap:] = 8
            tentative[1] = tentative[4] = True
            next_bits[1] = next_bits[4] = 8
        if name == "overlap_forecast_reject":
            base_bits.fill_(4)
            base_bits[:a8_rows] = 8
        unresolved = torch.arange(32) < future_count
        inputs = dict(base_positions=base_positions, base_activation_bits=base_bits,
            next_step_base_activation_bits=base_bits.clone(), next_block_start=64,
            next_activation_bits=next_bits, next_unresolved=unresolved,
            next_tentative=tentative, next_priority=torch.ones(32, dtype=torch.bfloat16),
            next_service_count=torch.zeros(32, dtype=torch.int32))
        if name == "current_capacity":
            inputs["next_step_base_activation_bits"].fill_(4)
            next_bits.fill_(8)
        elif name == "priority_rounding":
            next_bits[8:] = 8
            inputs["next_priority"] = torch.tensor(
                ([1.0, 0.00390625, 0.501953125, 0.0078125] * 8), dtype=torch.bfloat16)
            inputs["next_service_count"] = torch.arange(32, dtype=torch.int32) % 3
            tentative[3] = tentative[9] = True
        reuse_floor = .5 if name == "reuse_floor_equal" else .50390625 if name == "reuse_floor_above" else 0.
        if name.startswith("reuse_floor_"):
            inputs["next_priority"] = torch.tensor([.5, .25, 1., .5, .25, .75, .5, 1.] * 4, dtype=torch.bfloat16)
        selected = _select_prefetch(UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=32,
            min_reuse_score=reuse_floor) if reuse_floor else scheduler, **inputs)
        records.append(dict(name=name,
            min_reuse_score_bf16=int(torch.tensor(reuse_floor).bfloat16().view(torch.int16)),
            inputs={key: packed(value) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()},
            expected=dict(progress_local_positions=packed(selected.progress_local_positions),
                added_local_positions=packed(selected.added_local_positions),
                added_activation_bits=packed(_added_activation_bits(selected)),
                base_residency=asdict(selected.base_residency),
                joint_residency=asdict(selected.joint_residency),
                next_step_verification_residency=asdict(selected.next_step_verification_residency),
                rejected_for_current_step_capacity=selected.rejected_for_current_step_capacity,
                rejected_for_next_step_verification_capacity=selected.rejected_for_next_step_verification_capacity,
                rejected_for_budget=selected.rejected_for_budget)))
    return dict(schema="supra-uaps-token-selection/v1",
        source="generation.lookahead.UtilizationAwarePrefetchScheduler",
        target_joint_rows=48, max_next_rows=32, records=records)


def select_boundary_reference(inputs: dict) -> dict:
    """Independent block_initialization reference for the layer-to-selector connection.

    Inputs contain historical BF16 dependencies and independent L0 scout scores,
    never the DUT's chosen positions. Selection calls the maintained helpers.
    """
    from generation.engine import block_initialization_relative_selection, _select_boundary_layer0_deep_positions
    pending = torch.tensor(inputs["pending_bits"], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    future = torch.tensor(inputs["future_pending_bits"], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    current = torch.tensor(inputs["current_positions"], dtype=torch.long)
    mandatory = torch.tensor(inputs["mandatory_positions"], dtype=torch.long)
    sequence = pending.numel()
    if future.shape != pending.shape or not 0 < sequence <= 2048:
        raise ValueError("block initialization dependency shapes exceed the hardware sequence range")
    dependency_only = bool(inputs["flags"] & 2)
    if not dependency_only and (inputs["flags"] & 1 or inputs["relative_score_floor_bf16"]):
        raise ValueError("future dependency and relative floor require dependency-only block initialization")
    protection = torch.arange(inputs["protected_begin"], inputs["protected_end"])
    mandatory = torch.unique(torch.cat((mandatory, protection)), sorted=True)
    mandatory = mandatory[~torch.isin(mandatory, current)]
    scores = torch.maximum(pending, future) if inputs["flags"] & 1 else pending
    candidates = torch.arange(sequence)
    candidates = candidates[~torch.isin(candidates, current)]
    ranked = candidates[torch.argsort(scores[candidates], descending=True, stable=True)]
    target = min(sequence, inputs["target_token_count"])
    floor = float(torch.tensor(inputs["relative_score_floor_bf16"], dtype=torch.int16).view(torch.bfloat16))
    context = torch.tensor(inputs.get("context_positions", []), dtype=torch.long)
    required_count = inputs.get("required_context_count", 0)
    if floor:
        eligible, target, _, _ = block_initialization_relative_selection(
            ranked, scores[ranked], mandatory, current.numel(), target, floor)
    elif dependency_only:
        target = max(target, current.numel() + mandatory.numel())
        optional = ranked[~torch.isin(ranked, mandatory)]
        eligible = torch.cat((mandatory, optional[:target-current.numel()-mandatory.numel()]))
    else:
        target = max(target, current.numel() + mandatory.numel())
        count = min(ranked.numel(), 2 * (target-current.numel()))
        eligible = torch.unique(torch.cat((ranked[:count], context, mandatory)), sorted=True)
        if inputs["flags"] & 8:
            eligible = ranked[torch.isin(ranked, eligible)]
    deep, optional, _ = _select_boundary_layer0_deep_positions(
        torch.tensor(inputs["score_q8"], dtype=torch.uint8), current_positions=current,
        target_rows=target, eligible_positions=eligible, mandatory_candidate_positions=mandatory,
        required_candidate_positions=context, required_candidate_count=required_count)
    return dict(deep_positions=deep.tolist(), selected_optional=optional.tolist(), target_token_count=target)


def make_shortlist_reference_data(*, block_initialization_context_quota: int = 8) -> dict:
    from generation.engine import _select_boundary_layer0_deep_positions
    records = []
    for sequence, current, target, count, context, due in (
            (257, list(range(64, 96)), 80, 96, list(range(32, 64))+list(range(96, 128)), [0, 256]),
            (2048, [2000], 4, 5, [1, 2], [2047]),
            (257, list(range(64, 96)), 96, 128, list(range(32, 64)), [256]),
            (2048, list(range(1536, 1568)), 432, 800, list(range(1504, 1536)), [2047])):
        positions = torch.arange(sequence)
        state = CrossBlockRefreshState(total_length=sequence, block_length=len(current),
            boundary_target_rows=target, device=torch.device("cpu"))
        state.pending.copy_((((positions*19)%257).float()/256).bfloat16() if count != 5 else torch.zeros(sequence))
        candidates = positions[~torch.isin(positions, torch.tensor(current))]
        ranked = candidates[torch.argsort(state.score_positions(candidates), descending=True, stable=True)[:count]]
        eligible = torch.sort(torch.unique(torch.cat((ranked, torch.tensor(context), torch.tensor(due))))).values
        scores = ((positions*13)%255).to(torch.uint8)
        scores[current] = 0
        inputs = dict(current_positions=torch.tensor(current), target_rows=target, eligible_positions=eligible,
            mandatory_candidate_positions=torch.tensor(due), required_candidate_positions=torch.tensor(context),
            required_candidate_count=8 if sequence == 257 else 1)
        deep, optional, fresh_ranked = _select_boundary_layer0_deep_positions(scores, **inputs)
        records.append(dict(name=f"previous_shortlist_s{sequence}_k{count}",
            inputs=dict(previous_pending=packed(state.pending), shortlist_token_count=count, score_q8=packed(scores),
                **{("target_token_count" if key == "target_rows" else key): packed(value) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}),
            expected=dict(deep_positions=packed(deep), selected_optional=packed(optional), ranked_candidates=packed(fresh_ranked))))
    from generation.engine import block_initialization_relative_selection
    # Fixed current block; varying risk fields exercise the same block initialization
    # selection helpers used by the maintained generator. No RTL output enters
    # these expected sets. Fresh scores deliberately disagree with history.
    for name, floor, future, protected_end, zeros in (
            ("block_initialization_dependency", 0., False, 120, False),
            ("block_initialization_future_floor", .25, True, 120, False),
            ("block_initialization_floor_equal", .125, False, 120, False),
            ("block_initialization_zero_scores", .25, True, 120, True),
            ("block_initialization_protection_expands", 0., False, 310, False)):
        sequence, target = 320, 256
        positions = torch.arange(sequence)
        current = torch.arange(64, 96)
        due = torch.tensor([0, 319])
        protection = torch.arange(20, protected_end)
        mandatory = torch.unique(torch.cat((due, protection)), sorted=True)
        mandatory = mandatory[~torch.isin(mandatory, current)]
        pending = ((positions % 8).float() / 128).bfloat16()
        pending[8] = .25
        pending[mandatory] = 1.  # Required scores must not set the optional cutoff.
        future_pending = ((positions % 5).float() / 32).bfloat16()
        if zeros:
            pending.zero_(); future_pending.zero_()
        scores = torch.maximum(pending, future_pending) if future else pending
        optional = positions[~torch.isin(positions, current)]
        ranked = optional[torch.argsort(scores[optional], descending=True, stable=True)]
        effective_target = max(target, len(current)+len(mandatory))
        if floor:
            chosen, effective_target, _, _ = block_initialization_relative_selection(
                ranked, scores[ranked], mandatory, len(current), target, floor)
        else:
            remaining = ranked[~torch.isin(ranked, mandatory)]
            chosen = torch.cat((mandatory, remaining[:effective_target-len(current)-len(mandatory)]))
        eligible = torch.sort(chosen).values
        fresh = ((positions*13)%255).to(torch.uint8)
        deep, selected, fresh_ranked = _select_boundary_layer0_deep_positions(
            fresh, current_positions=current, target_rows=effective_target,
            eligible_positions=chosen, mandatory_candidate_positions=mandatory,
            required_candidate_positions=torch.empty(0,dtype=torch.long), required_candidate_count=0)
        records.append(dict(name=name, inputs=dict(
            previous_pending=packed(pending), previous_future_pending=packed(future_pending),
            shortlist_token_count=448, shortlist_flags=6 | int(future),
            relative_score_floor_bf16=int(torch.tensor(floor).bfloat16().view(torch.int16)),
            protected_begin=20, protected_end=protected_end,
            score_q8=packed(fresh), current_positions=packed(current),
            eligible_positions=packed(eligible), mandatory_candidate_positions=packed(due),
            required_candidate_positions=packed(torch.empty(0,dtype=torch.long)),
            required_candidate_count=0, target_token_count=target),
            expected=dict(deep_positions=packed(deep), selected_optional=packed(selected),
                ranked_candidates=packed(fresh_ranked), shortlist_mandatory_positions=packed(mandatory),
                effective_target=effective_target)))
    for name, tiebreak, context_fresh in (
            ("block_initialization_history_ties_s2048", True, False),
            ("block_initialization_history_tail_s2048", True, True),
            ("block_initialization_position_ties_s2048", False, False)):
        sequence, target = 2048, 256
        positions = torch.arange(sequence)
        current = torch.arange(1536, 1568)
        context = torch.cat((torch.arange(1504, 1536), torch.arange(1568, 1600)))
        due = torch.tensor([2047])
        mandatory = torch.cat((torch.arange(2000, 2010), due))
        pending = (((positions * 19) % 257).float() / 256).bfloat16()
        # Context and protected tokens below top800 must still participate.
        pending[context] = ((context % 8).float() / 4096).bfloat16()
        pending[mandatory] = 0
        optional = positions[~torch.isin(positions, current)]
        ranked = optional[torch.argsort(pending[optional], descending=True, stable=True)]
        eligible = torch.unique(torch.cat((ranked[:2*(target-32)], context, mandatory)), sorted=True)
        ordered = ranked[torch.isin(ranked, eligible)] if tiebreak else eligible
        fresh = torch.full((sequence,), 7, dtype=torch.uint8)
        if context_fresh:
            fresh[context] = 255
        deep, selected, fresh_ranked = _select_boundary_layer0_deep_positions(
            fresh, current_positions=current, target_rows=target,
            eligible_positions=ordered, mandatory_candidate_positions=mandatory,
            required_candidate_positions=context, required_candidate_count=block_initialization_context_quota)
        # Table rank is a compact encoding of relative order, not an algorithm
        # output: top800 then remaining context. Mandatory is selected first.
        orders = positions.clone()
        if tiebreak:
            orders.fill_(2047)
            first = ranked[:800]
            tail = ranked[torch.isin(ranked, context) & ~torch.isin(ranked, first)]
            orders[first] = torch.arange(first.numel())
            orders[tail] = 800 + torch.arange(tail.numel())
        records.append(dict(name=name, inputs=dict(
            previous_pending=packed(pending), shortlist_token_count=448,
            shortlist_flags=4 | (8 if tiebreak else 0),
            protected_begin=2000, protected_end=2010,
            score_q8=packed(fresh), current_positions=packed(current),
            eligible_positions=packed(eligible), mandatory_candidate_positions=packed(due),
            required_candidate_positions=packed(context), required_candidate_count=block_initialization_context_quota,
            target_token_count=target),
            expected=dict(deep_positions=packed(deep), selected_optional=packed(selected),
                ranked_candidates=packed(fresh_ranked), shortlist_mandatory_positions=packed(mandatory),
                shortlist_orders=packed(orders))))
    return dict(schema="supra-atse-cross-block-token-selection/v1",
        records=records)


def make_boundary_selection_reference_data() -> dict:
    from generation.engine import _select_boundary_layer0_deep_positions

    records = []
    for name in ("ranked_shortlist", "zero_score_ties"):
        scores = (torch.arange(160) % 11).to(torch.uint8)
        if name == "zero_score_ties":
            scores.zero_()
        scores[3] = scores[101] = 0
        inputs = dict(current_positions=torch.arange(64, 96), target_rows=80,
            eligible_positions=torch.cat((torch.arange(48), torch.arange(96, 144))),
            mandatory_candidate_positions=torch.tensor([3, 101]),
            required_candidate_positions=torch.arange(8), required_candidate_count=8)
        deep, optional, ranked = _select_boundary_layer0_deep_positions(scores, **inputs)
        records.append(dict(name=name,
            inputs={"score_q8": packed(scores), **{
                ("target_token_count" if key == "target_rows" else key):
                packed(value) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()}},
            expected=dict(deep_positions=packed(deep), selected_optional=packed(optional),
                          ranked_candidates=packed(ranked))))
    return dict(schema="supra-atse-cross-block-token-selection/v1",
        source="generation.engine._select_boundary_layer0_deep_positions",
        records=records)


def make_psme_fixed_k_reference_data() -> dict:
    """Generate fixed-k selection and state transitions from the algorithm."""
    from generation.engine import select_irreversible_transfers
    from numerics.candidate import streaming_candidate_bf16, candidate_action_confidence

    records = []
    for name, quota, tokens, steps in (
            ("zero_quota", 0, [7, 7, 7, 7], 1),
            ("low_confidence_quota1", 1, [7, 7, 7, 7], 1),
            ("low_confidence_quota2", 2, [7, 7, 7, 7], 1),
            ("retained_token", 1, [7, 7, 7, 6], 1),
            ("quota_clamped_to_masked", 32, [7, 7, 5, 6], 1),
            ("suppressed_mask_winner", 1, [7], 2),
            ("stable_equal_scores", 3, [7, 7, 7, 7], 2),
            ("default_k", 3, [7, 7, 7, 7], 3)):
        token_tensor = torch.tensor([tokens])
        state = DraftVerifyBlockState(token_tensor.shape, device="cpu")
        initial_locked = token_tensor != 7
        state.state[initial_locked] = 2
        state.precision_age[initial_locked] = 5
        state.commit_origin[initial_locked] = 1
        logits = torch.zeros(1, len(tokens), 8, dtype=torch.bfloat16)
        for row in range(len(tokens)):
            logits[0, row, row] = .5 if name == "stable_equal_scores" else (row + 1) / 4
        suppressed = [7] if name == "suppressed_mask_winner" else []
        if suppressed:
            logits.zero_()
            logits[0, 0, 7] = 4
        proposal, _, confidence = streaming_candidate_bf16(logits)
        action = candidate_action_confidence(proposal, confidence, suppressed)
        for step in range(steps):
            masked = state.state == 0
            locked = state.state == 2
            before = dict(tokens=packed(token_tensor.clone()),
                state=packed(state.state.clone()), last_top1=packed(state.last_top1.clone()),
                precision_age=packed(state.precision_age.clone()),
                commit_origin=packed(state.commit_origin.clone()))
            selected = (select_irreversible_transfers(masked, action, mode="fixed_k", k=quota)
                        if quota else torch.zeros_like(masked))
            state.admit(token_tensor, proposal, selected, selected, torch.zeros_like(selected))
            state.update_masked_history(proposal, masked)
            state.advance_locked_age(locked)
            records.append(dict(name=f"{name}_{step}", inputs=dict(
                logits=packed(logits), **before, mask=packed(masked),
                scheduled_quota=quota, mask_token_id=7,
                suppressed_candidate_token_ids=suppressed,
                selected_probability=packed(confidence), action_confidence=packed(action)),
                expected=dict(proposal=packed(proposal), selected=packed(selected),
                    tokens=packed(token_tensor.clone()), state=packed(state.state.clone()),
                    last_top1=packed(state.last_top1.clone()),
                    precision_age=packed(state.precision_age.clone()),
                    commit_origin=packed(state.commit_origin.clone()))))
    return dict(schema="supra-psme-fixed-k-decoding/v1", records=records,
        source="generation.engine.select_irreversible_transfers/generation.state.Feature2BlockState")


def make_atse_uaps_limits_reference_data() -> dict:
    """Capture selection, precision and generator counter updates on CPU."""
    from unittest.mock import patch
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import ActivationType, BlockType, LLaDAModel, LLaDAModelLM, ModelConfig

    def encode(values):
        return {key: packed(value) if isinstance(value, torch.Tensor) else value
                for key, value in values.items()}

    def public_residency(value):
        fields = asdict(value)
        fields["a4_token_count"] = fields.pop("a4_rows")
        fields["a8_token_count"] = fields.pop("a8_rows")
        return fields

    records = []
    scheduler = UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=8)
    inputs = dict(base_positions=torch.tensor([0, 33]),
        base_activation_bits=torch.tensor([8, 4], dtype=torch.int8), next_block_start=32,
        next_step_base_activation_bits=torch.tensor([8, 4], dtype=torch.int8),
        next_activation_bits=torch.tensor([4, 4, 8, 4, 4, 4], dtype=torch.int8),
        next_unresolved=torch.ones(6, dtype=torch.bool),
        next_tentative=torch.tensor([False, False, True, False, False, False]),
        next_priority=torch.tensor([1., .5, .75, .9, .8, .7], dtype=torch.bfloat16),
        next_service_count=torch.zeros(6, dtype=torch.int32),
        source_b_attempts=torch.tensor([2, 7, 5, 1, 0, 2], dtype=torch.int32))
    for limit in (-1, 0, 2):
        args = dict(inputs, max_source_b_attempts=limit)
        result = _select_prefetch(scheduler, **args)
        records.append(dict(name=f"source_b_limit_{limit}", kind="source_b_selection",
            inputs=encode(args), expected=encode(dict(
                progress_local_positions=result.progress_local_positions,
                added_local_positions=result.added_local_positions,
                added_activation_bits=_added_activation_bits(result),
                base_residency=public_residency(result.base_residency),
                joint_residency=public_residency(result.joint_residency),
                next_step_verification_residency=public_residency(result.next_step_verification_residency),
                rejected_for_current_step_capacity=result.rejected_for_current_step_capacity,
                rejected_for_next_step_verification_capacity=result.rejected_for_next_step_verification_capacity,
                rejected_for_budget=result.rejected_for_budget))))

    bits = torch.tensor([4, 8, 8, 8, 8, 8], dtype=torch.int8)
    scores = torch.tensor([255, 0, 9, 9, 2, 1], dtype=torch.uint8)
    protected = torch.tensor([True, True, False, False, False, True])
    for name, positions, limit in (
            ("disabled", [0, 1, 2, 3, 4, 5], -1),
            ("zero", [0, 1, 2, 3, 4, 5], 0),
            ("protected_excess", [0, 1, 2, 3, 4, 5], 1),
            ("ranked_tie", [0, 1, 2, 3, 4, 5], 3),
            ("already_within", [0, 1, 2, 3, 4, 5], 5),
            ("stable_input_order", [9, 8, 7, 2, 1, 0], 3)):
        result = bits.clone() if limit < 0 else boundary_deep_precision_bits(bits, scores, protected, limit)
        records.append(dict(name=f"deep_{name}", kind="deep_precision",
            inputs=encode(dict(deep_positions=torch.tensor(positions), activation_bits=bits,
                score_q8=scores, protected=protected, a8_limit=limit)),
            expected=encode(dict(activation_bits=result))))

    # The generator writes the counter. The observer retains its tensor
    # until the actual forward callback, after selection and before admission.
    original_select = UtilizationAwarePrefetchScheduler.select
    pending = []

    def observe_select(scheduler, **kwargs):
        caller = sys._getframe(1).f_locals
        result = original_select(scheduler, **kwargs)
        record = dict(name=f"generator_select_{len(records)}", kind="source_b_generator",
            block_index=int(caller["block_index"]), step_index=int(caller["step_index"]),
            parameters=dict(target_joint_rows=scheduler.target_joint_rows,
                max_next_rows=scheduler.max_next_rows),
            inputs=encode(_hardware_selection_inputs(kwargs)), expected=encode(dict(
                progress_local_positions=result.progress_local_positions,
                added_local_positions=result.added_local_positions,
                added_activation_bits=_added_activation_bits(result))))
        records.append(record)
        pending.append((record, kwargs["source_b_attempts"]))
        return result

    def capture(event):
        for record, attempts in pending:
            record["expected"]["source_b_attempts_before_forward"] = packed(attempts)
            record["forward_kind"] = event.forward_kind
        pending.clear()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        config = ModelConfig(d_model=16, n_heads=2, n_kv_heads=2, n_layers=1,
            mlp_hidden_size=32, activation_type=ActivationType.silu, block_type=BlockType.llama,
            max_sequence_length=16, vocab_size=4, embedding_size=4, rope=True,
            attention_dropout=0., residual_dropout=0., embedding_dropout=0., use_manual_attention=True)
        model = LLaDAModelLM(LLaDAConfig(**asdict(config)), model=LLaDAModel(config)).eval()
    with patch.object(UtilizationAwarePrefetchScheduler, "select", observe_select):
        _, _, trace, _ = generation.generate(model, torch.tensor([[2]]),
            steps=12, gen_length=6, block_length=2, mask_id=3, tau_high=.9, tau_low=.5,
            confirm_tau=.5, budget_scale=1., row_precision_context=ActivationPrecisionContext(),
            packed_attention_refresh=True, packed_attention_force_full_current=True,
            packed_attention_full_current_context_rows=0, dynamic_block_lookahead=True,
            dynamic_block_canonical_future=True, dynamic_block_target_joint_rows=4,
            dynamic_block_max_next_rows=2, dynamic_block_next_admission_budget=2,
            dynamic_block_max_handoff_verification_rows=2, dynamic_block_canonical_direct_tau=.99,
            dynamic_block_next_min_confidence=.99, dynamic_block_source_b_max_attempts=2,
            dynamic_block_allow_deferred_verification=True, state_capture_callback=capture)
    if pending:
        raise RuntimeError("generator selection did not reach a forward capture")
    canonical = make_future_admission_reference_data(source_b_max_attempts=2, canonical_direct_tau=.99)
    records.extend(canonical["source_b_attempt_records"])
    return dict(schema="supra-atse-uaps-limits/v1", records=records,
        target_joint_rows=48, max_next_rows=8,
        source="generation.engine.generate; generation.lookahead.UtilizationAwarePrefetchScheduler",
        generator_seed=8,
        generator_admitted_positions=[list(event.next_admitted_positions) for event in trace])


def make_cross_block_reference_data(seed: int = 20260905, mode: str = "all_changes") -> dict:
    state = CrossBlockRefreshState(total_length=112, block_length=32,
        boundary_target_rows=80, device=torch.device("cpu"),
        pending_confidence_mode=mode, pending_relation_mode="direct",
        track_future_actual_remask=True)
    state.start_initial_block(block_start=48, block_end=80)
    generator = torch.Generator().manual_seed(seed)
    fields = ("relation", "pending", "actual_remask_pending",
              "actual_remask_epoch_pending", "future_pending", "future_actual_remask_pending")
    records = []
    for step, (queries, consumed, changes) in enumerate((
            ([0, 3, 11, 48, 49, 80, 91], [], [(49, False, 0.75)]),
            ([3, 80], [3, 80], [(49, False, 0.5), (52, True, 0.25), (53, True, 0.625)]),
            ([11], [0], []),
            ([3, 80, 91], [3, 80], [(81, False, 0.5), (84, True, 0.25)]))):
        query = torch.tensor(queries, dtype=torch.long)
        relation = torch.randint(0, 65, (1, len(queries), 32), generator=generator).float().div(256).bfloat16()
        if step == 0:
            relation[0, 0, 1] = 0.25
        changed = torch.zeros((1, 112), dtype=torch.bool)
        remasked = torch.zeros_like(changed)
        confidence = torch.ones((1, 112), dtype=torch.bfloat16)
        for position, remask, value in changes:
            changed[0, position] = True
            remasked[0, position] = remask
            confidence[0, position] = value
        before = {name: packed(getattr(state, name)) for name in fields}
        advance = step == 3
        if advance:
            state.advance_block(block_start=80, block_end=112)
        after_advance = {name: packed(getattr(state, name)) for name in fields} if advance else None
        state.selected_regular = torch.tensor(consumed, dtype=torch.long)
        state.observe(dict(prefix_dependency_query_positions=query, prefix_dependency_mean=relation),
            changed_global=changed, changed_confidence_global=confidence,
            changed_remask_global=remasked)
        records.append(dict(step_index=step, block_start=state.block_start, advance_block=advance,
            after_advance=after_advance, inputs=dict(query_positions=packed(query),
            relation=packed(relation), consumed_positions=packed(state.pending.new_tensor(consumed, dtype=torch.long)),
            changed_global=packed(changed), changed_remask_global=packed(remasked),
            changed_confidence_global=packed(confidence)), before=before,
            expected={name: packed(getattr(state, name)) for name in fields}))
    return dict(schema="supra-atse-cross-block-pending-refresh/v1", seed=seed, rows=112,
        block_start=48, block_length=32, pending_confidence_mode=mode,
        pending_relation_mode="direct", track_future_actual_remask=True,
        source="generation.refresh.CrossBlockRefreshState", records=records)


@torch.inference_mode()
def make_pre_p8_relation_reference_data(seed: int = 20260906, device: str = "cpu") -> dict:
    from layer_reference_data import _load_algorithm
    _load_algorithm()
    from layer_reference_data import (
        ActivationType, BlockType, BufferCache, Int8MatmulWorkspace, LayerNormType,
        LLaDALlamaBlock, ModelConfig, SpinQuantV8CacheCodec, _install_native_attention_numeric,
    )
    from generation.refresh import begin_tri_block_attention_monitor, end_tri_block_attention_monitor

    target, sequence = torch.device(device), 17
    generator = torch.Generator().manual_seed(seed)
    positions = torch.arange(sequence, device=target)
    cache = BufferCache()
    runtime = SimpleNamespace(_LLaDAModel__cache=cache, device=positions.device)
    begin_tri_block_attention_monitor(runtime, region_start=0, region_end=sequence,
        prediction_start=5, prediction_end=8)
    config = ModelConfig(d_model=4096, n_heads=32, n_kv_heads=32, n_layers=2,
        mlp_hidden_size=12288, activation_type=ActivationType.silu, block_type=BlockType.llama,
        layer_norm_type=LayerNormType.rms, include_bias=False, include_qkv_bias=False,
        init_device="meta", rope=True, use_manual_attention=True, flash_attention=False,
        attn_monitor_layer=1, attn_monitor_all_layers=True,
        attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0)
    qkv = torch.zeros((1, sequence, 4096), dtype=torch.bfloat16, device=target)
    records = []
    try:
        for layer in range(2):
            block = LLaDALlamaBlock(layer, config, cache).eval()
            block.attn_out = torch.nn.Identity()
            _install_native_attention_numeric(block, workspace=Int8MatmulWorkspace(), qk_int8=True,
                softmax_lut=True, probability_p8=False, k8_cache=True,
                v8_codec=SpinQuantV8CacheCodec(torch.full((32,), 0.015625, device=target).bfloat16()),
                rope_before_k8=True)
            scores = (torch.randn((1, 32, sequence, sequence), generator=generator)*1.5).bfloat16().to(target)
            block._spinquant_score_override = lambda *args, **kwargs: scores
            probability_forward = block._spinquant_probability_override
            observed = []

            def observe_probability(value):
                result = probability_forward(value)
                observed.append(result.detach().clone())
                return result

            block._spinquant_probability_override = observe_probability
            block.attention(qkv, qkv, qkv, None, use_cache=False, query_position_ids=positions[None, :])
            if len(observed) != 1:
                raise RuntimeError("generic monitor did not consume exactly one probability tensor")
            profile = dict(query_positions=cache["attn_monitor_dependency_query_positions"],
                dependency_mean=cache["attn_monitor_dependency_max"])
            saved = AttentionRefreshState(region_start=0, region_end=sequence, total_length=sequence,
                initial_attention_profile=profile, target_active_rows=8.5)
            records.append(dict(layer_index=layer, inputs=dict(scores=packed(scores), probabilities=packed(observed[0])),
                expected=dict(dependency_max_fp32=packed(profile["dependency_mean"]),
                    saved_dependency=packed(saved.dependency), layer_count=cache["attn_monitor_dependency_layer_count"])))
    finally:
        end_tri_block_attention_monitor(runtime)
    return dict(schema="supra-atse-pre-p8-dependency/v1", seed=seed, backend=str(target),
        source="model.modeling_llada",
        records=records)


@torch.inference_mode()
def make_attention_relation_reference_data(seed: int = 20260905, device: str = "cpu") -> dict:
    from layer_reference_data import _load_algorithm
    _load_algorithm()
    from layer_reference_data import (
        ActivationType, BlockType, BufferCache, Int8MatmulWorkspace, LayerNormType,
        LLaDALlamaBlock, ModelConfig, SpinQuantV8CacheCodec, _install_native_attention_numeric,
    )

    target = torch.device(device)
    generator = torch.Generator().manual_seed(seed)
    positions = torch.tensor([[5, 11, 3]], dtype=torch.long, device=target)
    keys = torch.arange(17, device=target).unsqueeze(0)
    values = torch.randint(-127, 128, (1, 32, 17, 128), generator=generator, dtype=torch.int8).to(target)
    scales = ((torch.arange(32).float() + 16) / 4096).bfloat16().to(target)
    cache = BufferCache()
    cache.update(attn_monitor_capture_qk_probe=True, attn_monitor_capture_deployment_p8_relation=True,
        attn_monitor_numeric_position_context=dict(staged_group=0, query_positions=positions,
            key_positions=keys, kv_write_positions=positions))
    records = []
    for layer in range(2):
        config = ModelConfig(d_model=4096, n_heads=32, n_kv_heads=32, n_layers=2,
            mlp_hidden_size=12288, activation_type=ActivationType.silu, block_type=BlockType.llama,
            layer_norm_type=LayerNormType.rms, include_bias=False, include_qkv_bias=False,
            init_device="meta", rope=True, use_manual_attention=True, flash_attention=False,
            attn_monitor_layer=layer)
        block = LLaDALlamaBlock(layer, config, cache).eval()
        _install_native_attention_numeric(block, workspace=Int8MatmulWorkspace(), qk_int8=True,
            softmax_lut=True, probability_p8=False, k8_cache=True,
            v8_codec=SpinQuantV8CacheCodec(scales), rope_before_k8=True)
        scores = (torch.randn((1, 32, 3, 17), generator=generator) * 1.5).bfloat16().to(target)
        probabilities = block._spinquant_probability_override(scores)
        context = block._spinquant_context_override(probabilities, values)
        probe = cache["attn_monitor_numeric_probe_groups"][0]
        records.append(dict(layer_index=layer, inputs=dict(scores=packed(scores)),
            expected=dict(probabilities=packed(probabilities), context=packed(context),
                probability_codes=packed(probe["selected_layer_p_codes"]),
                probability_scale=packed(probe["selected_layer_p_scale_bf16"]),
                relation=packed(probe["selected_layer_p8_relation_bf16"]),
                all_layer_max_relation=packed(probe["all_layer_max_p8_relation_bf16"]),
                relation_layer_count=probe["p8_relation_layer_count"])))
    return dict(schema="supra-atse-attention-dependency/v1", seed=seed, backend=str(target),
        flags=dict(qk_int8=True, softmax_lut=True, probability_p8=False, v8_cache=True),
        source="quantization.model",
        inputs=dict(query_positions=packed(positions), key_positions=packed(keys),
                    value_codes=packed(values), value_scale=packed(scales)), records=records)


@torch.inference_mode()
def make_boundary_scout_reference_data(seed: int = 20260905, device: str = "cpu",
                                probability_boundaries: bool = False,
                                reduction_order_boundary: bool = False) -> dict:
    from layer_reference_data import _load_algorithm
    _load_algorithm()
    from layer_reference_data import (
        ActivationType, BlockType, BufferCache, Int8MatmulWorkspace, LayerNormType,
        LLaDALlamaBlock, ModelConfig, SpinQuantV8CacheCodec, _install_native_attention_numeric,
    )
    from generation.refresh import (
        begin_boundary_layer0_scout, read_boundary_layer0_scout, end_boundary_layer0_scout,
    )

    target = torch.device(device)
    generator = torch.Generator().manual_seed(seed)
    sequence, heads = 17, 32
    positions = torch.arange(sequence, device=target)
    current = torch.tensor([5, 6, 7], device=target)
    scores = (torch.randn((1, heads, sequence, sequence), generator=generator) * 1.5).bfloat16().to(target)
    cache = BufferCache()
    runtime = SimpleNamespace(_LLaDAModel__cache=cache, device=current.device)
    config = ModelConfig(d_model=4096, n_heads=heads, n_kv_heads=heads, n_layers=1,
        mlp_hidden_size=12288, activation_type=ActivationType.silu, block_type=BlockType.llama,
        layer_norm_type=LayerNormType.rms, include_bias=False, include_qkv_bias=False,
        init_device="meta", rope=True, use_manual_attention=True, flash_attention=False,
        attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0)
    block = LLaDALlamaBlock(0, config, cache).eval()
    block.attn_out = torch.nn.Identity()
    _install_native_attention_numeric(block, workspace=Int8MatmulWorkspace(), qk_int8=True,
        softmax_lut=True, probability_p8=False, k8_cache=True,
        v8_codec=SpinQuantV8CacheCodec(torch.full((heads,), 0.015625, device=target).bfloat16()),
        rope_before_k8=True)
    block._spinquant_score_override = lambda *args, **kwargs: scores
    probability_forward = block._spinquant_probability_override
    observed_probabilities = []
    boundary_probability = None
    explicit_probability = probability_boundaries or reduction_order_boundary
    if explicit_probability:
        boundary_probability = torch.eye(sequence, device=target).expand(1, heads, -1, -1).clone()
        value_sets = []
        if reduction_order_boundary:
            # Sequential FP32 additions cross 16, while an exact sum stays below.
            value_sets.append(torch.tensor(
                [1.0] * 15 + [1 - 2**-8, 2**-8 - 2**-16, 2**-17] +
                [129 * 2**-28] * 14, device=target).bfloat16().float())
        else:
            for raw_last in (0x3eff, 0x3f00, 0x3f01):
                last_value = torch.tensor([raw_last], dtype=torch.int16).view(torch.bfloat16).float().item()
                values = torch.full((heads,), 0.5, device=target)
                values[-1] = last_value
                value_sets.append(values)
        for candidate, values in enumerate(value_sets):
            for query, key in ((5 + candidate, candidate), (candidate, 1)):
                boundary_probability[0, :, query].zero_()
                boundary_probability[0, :, query, key] = values
                boundary_probability[0, :, query, 5] = (1.0 - values).bfloat16().float()

    def observe_probability(value):
        result = boundary_probability if explicit_probability else probability_forward(value)
        observed_probabilities.append(result.detach().clone())
        return result

    block._spinquant_probability_override = observe_probability
    qkv = torch.zeros((1, sequence, 4096), dtype=torch.bfloat16, device=target)
    records = []
    for transition in (None, torch.tensor([1, 4], device=target)):
        begin_boundary_layer0_scout(runtime, current_positions=current,
            total_length=sequence, transition_positions=transition)
        try:
            block.attention(qkv, qkv, qkv, None, use_cache=False,
                            query_position_ids=positions[None, :])
            score_q8, entries = read_boundary_layer0_scout(runtime)
        finally:
            end_boundary_layer0_scout(runtime)
        records.append(dict(direction="current_to_candidate" if transition is None else "candidate_to_transition",
            transition_positions=None if transition is None else packed(transition),
            expected=dict(score_q8=packed(score_q8), probability_entries=entries)))
    if len(observed_probabilities) != 2 or not torch.equal(*observed_probabilities):
        raise RuntimeError("scout directions did not consume the same actual probabilities")
    return dict(schema="supra-atse-layer0-scan/v1", seed=seed, backend=str(target),
        source="model.modeling_llada",
        inputs=dict(query_positions=packed(positions), current_positions=packed(current),
                    **({} if explicit_probability else dict(scores=packed(scores)))),
        input_stage="probabilities" if explicit_probability else "scores",
        reduction_order_boundary=reduction_order_boundary,
        probabilities=packed(observed_probabilities[0]), records=records)


class CrossBlockState(ctypes.Structure):
    _fields_ = [("rows", ctypes.c_uint32), ("block_start", ctypes.c_uint32),
                ("relation", ctypes.c_uint16 * (2048 * 32))] + [
        (name, ctypes.c_uint16 * 2048) for name in ("pending", "actual_remask_pending",
            "actual_remask_epoch_pending", "future_pending", "future_actual_remask_pending")]


def _packed_bf16(tensor: dict) -> np.ndarray:
    if tensor.get("raw_dtype") == "fp32":
        if not tensor.get("bf16_materialized"):
            raise ValueError("BF16 C11 input contains true FP32 intermediate precision")
        raw = np.asarray(tensor["raw"], dtype="<u4")
        if np.any(raw & 0xffff):
            raise ValueError("FP32 container does not contain BF16 raw values")
        return np.ascontiguousarray((raw >> 16).astype("<u2"))
    if tensor["dtype"] != "torch.bfloat16":
        raise ValueError("expected BF16 tensor or BF16-materialized FP32 container")
    return np.ascontiguousarray(tensor["raw"], dtype="<u2")


def compare_attention_relation_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-atse-attention-dependency/v1":
        raise ValueError("unsupported Attention relation reference_data")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    u16p = ctypes.POINTER(ctypes.c_uint16)
    lib.rtl_p8_relation_bf16.argtypes = [ctypes.POINTER(ctypes.c_int8), u16p,
        ctypes.c_uint32, ctypes.c_uint32, u16p, u16p]
    lib.rtl_p8_relation_bf16.restype = ctypes.c_int
    layer_max = None
    for index, record in enumerate(reference_data["records"]):
        expected = record["expected"]
        codes = np.ascontiguousarray(expected["probability_codes"]["raw"], dtype=np.int8)
        scales = _packed_bf16(expected["probability_scale"])
        if codes.ndim != 4 or codes.shape[:2] != (1, 32) or scales.shape != codes.shape[:-1]:
            raise ValueError("Attention relation requires one batch, 32 heads and matching scales")
        rows, keys = codes.shape[-2:]
        actual = np.zeros((1, rows, keys), dtype="<u2")
        if layer_max is None:
            layer_max = np.zeros_like(actual)
        if layer_max.shape != actual.shape:
            raise ValueError("layer relation shapes changed")
        if lib.rtl_p8_relation_bf16(codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            scales.ctypes.data_as(u16p), rows, keys, actual.ctypes.data_as(u16p), layer_max.ctypes.data_as(u16p)):
            raise ValueError("C11 rejected P8 relation input")
        for name, values in (("relation", actual), ("all_layer_max_relation", layer_max)):
            result = compare_arrays(_packed_bf16(expected[name]), values)
            if not result["match"]:
                return dict(result, record=index, field=name)
    return dict(match=True, records=len(reference_data["records"]), checked_fields=2*len(reference_data["records"]))


def compare_pre_p8_relation_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-atse-pre-p8-dependency/v1":
        raise ValueError("unsupported pre-P8 relation reference_data")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    u16p = ctypes.POINTER(ctypes.c_uint16)
    lib.rtl_probability_relation_bf16.argtypes = [u16p, ctypes.c_uint32, ctypes.c_uint32, u16p, u16p]
    lib.rtl_probability_relation_bf16.restype = ctypes.c_int
    layer_max = None
    for index, record in enumerate(reference_data["records"]):
        probability = _packed_bf16(record["inputs"]["probabilities"])
        if probability.ndim != 4 or probability.shape[:2] != (1, 32):
            raise ValueError("pre-P8 relation requires batch one and 32 heads")
        queries, keys = probability.shape[-2:]
        relation = np.zeros((queries, keys), dtype="<u2")
        if layer_max is None:
            layer_max = np.zeros_like(relation)
        if layer_max.shape != relation.shape:
            raise ValueError("pre-P8 query/key shape changed across layers")
        if lib.rtl_probability_relation_bf16(probability.ctypes.data_as(u16p), queries, keys,
                relation.ctypes.data_as(u16p), layer_max.ctypes.data_as(u16p)):
            raise ValueError("C11 rejected pre-P8 probabilities")
        result = compare_arrays(_packed_bf16(record["expected"]["saved_dependency"]), layer_max)
        if not result["match"]:
            return dict(result, record=index, field="saved_dependency")
    return dict(match=True, records=len(reference_data["records"]))


def compare_boundary_scout_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-atse-layer0-scan/v1":
        raise ValueError("unsupported boundary scout reference_data")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    u16p, u8p = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint8)
    lib.rtl_boundary_scout_q8.argtypes = [u16p, ctypes.c_uint32, u16p,
        ctypes.c_uint32, u16p, ctypes.c_uint32, u16p, u8p]
    lib.rtl_boundary_scout_q8.restype = ctypes.c_int
    probability = _packed_bf16(reference_data["probabilities"])
    query = np.ascontiguousarray(reference_data["inputs"]["query_positions"]["raw"], dtype="<u2")
    current = np.ascontiguousarray(reference_data["inputs"]["current_positions"]["raw"], dtype="<u2")
    sequence = query.size
    if probability.shape != (1, 32, sequence, sequence):
        raise ValueError("scout requires all query rows, 32 heads and full key positions")
    for index, record in enumerate(reference_data["records"]):
        transition = np.ascontiguousarray([] if record["transition_positions"] is None else
            record["transition_positions"]["raw"], dtype="<u2")
        actual = np.zeros(sequence, dtype=np.uint8)
        if lib.rtl_boundary_scout_q8(probability.ctypes.data_as(u16p), sequence,
            query.ctypes.data_as(u16p), current.size, current.ctypes.data_as(u16p),
            transition.size, transition.ctypes.data_as(u16p), actual.ctypes.data_as(u8p)):
            raise ValueError("C11 rejected scout input")
        result = compare_arrays(np.asarray(record["expected"]["score_q8"]["raw"], dtype=np.uint8), actual)
        if not result["match"]:
            return dict(result, record=index, field="score_q8")
    return dict(match=True, records=len(reference_data["records"]))


def compare_packed_precision_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-psme-context-precision/v1":
        raise ValueError("unsupported packed precision reference_data")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    u8p, u16p = ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16)
    lib.rtl_context_precision.argtypes = [ctypes.c_uint32, u8p, u8p, u16p,
        ctypes.c_uint32, ctypes.c_uint8, u8p, u8p, ctypes.POINTER(ctypes.c_uint32)]
    lib.rtl_context_precision.restype = ctypes.c_int
    for index, record in enumerate(reference_data["records"]):
        bits = np.asarray(record["input_activation_bits"]["raw"], dtype=np.uint8)
        mandatory = np.asarray(record["mandatory_current"]["raw"], dtype=np.uint8)
        candidates = np.asarray(record["context_candidates"]["raw"], dtype=np.int64)
        rows = bits.size
        dependency = np.zeros(rows, dtype="<u2")
        if record["expected_upgrade_count"]:
            dependency[candidates] = np.asarray(record["context_scores"]["raw"], dtype="<u2")
        actual = np.zeros(rows, dtype=np.uint8)
        order = np.zeros(rows, dtype=np.uint8)
        count = ctypes.c_uint32()
        if lib.rtl_context_precision(rows, bits.ctypes.data_as(u8p), mandatory.ctypes.data_as(u8p),
                dependency.ctypes.data_as(u16p), record["parameters"]["fixed_context_a8_token_count"],
                record["tail_transfer_active"], actual.ctypes.data_as(u8p), order.ctypes.data_as(u8p),
                ctypes.byref(count)):
            raise ValueError("C11 rejected packed precision input")
        expected_order = candidates[np.asarray(record.get("expected_upgrade_order", {}).get("raw", []), dtype=np.int64)]
        for field, expected, observed in (
                ("activation_bits", np.asarray(record["expected_activation_bits"]["raw"], dtype=np.uint8), actual),
                ("upgrade_order", expected_order.astype(np.uint8), order[:count.value])):
            result = compare_arrays(expected, observed)
            if not result["match"]:
                return dict(result, record=index, field=field)
        if count.value != record["expected_upgrade_count"]:
            return dict(match=False, record=index, field="upgrade_count")
    return dict(match=True, records=len(reference_data["records"]))


def compare_cross_block_reference_data(reference_data: dict, library: Path) -> dict:
    if (reference_data["schema"] != "supra-atse-cross-block-pending-refresh/v1" or
            reference_data["block_length"] != 32 or reference_data["pending_confidence_mode"] != "all_changes" or
            reference_data["pending_relation_mode"] != "direct" or not reference_data["track_future_actual_remask"]):
        raise ValueError("unsupported cross-block pending reference_data")
    rows, start = reference_data["rows"], reference_data["block_start"]
    if not 32 < rows <= 2048 or not 0 <= start <= rows - 32:
        raise ValueError("invalid cross-block row range")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    lib.rtl_cross_block_state_bytes.restype = ctypes.c_size_t
    if lib.rtl_cross_block_state_bytes() != ctypes.sizeof(CrossBlockState):
        raise ValueError("C11 cross-block state layout differs from binding")
    u16p, u8p = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint8)
    lib.rtl_cross_block_init.argtypes = [ctypes.POINTER(CrossBlockState), ctypes.c_uint32, ctypes.c_uint32]
    lib.rtl_cross_block_advance.argtypes = [ctypes.POINTER(CrossBlockState), ctypes.c_uint32]
    lib.rtl_cross_block_observe.argtypes = [ctypes.POINTER(CrossBlockState), ctypes.c_uint32,
        u16p, u16p, ctypes.c_uint32, u16p, u8p, u8p, u16p]
    state = CrossBlockState()
    if lib.rtl_cross_block_init(ctypes.byref(state), rows, start):
        raise ValueError("C11 rejected cross-block initialization")
    checked = 0
    for record in reference_data["records"]:
        inputs = record["inputs"]
        vectors = {}
        query_count = len(inputs["query_positions"]["raw"])
        consumed_count = len(inputs["consumed_positions"]["raw"])
        for name, shape, dtype, upper in (
                ("query_positions", [query_count], np.uint16, rows - 1),
                ("consumed_positions", [consumed_count], np.uint16, rows - 1),
                ("relation", [1, query_count, 32], np.uint16, 65535),
                ("changed_global", [1, rows], np.uint8, 1),
                ("changed_remask_global", [1, rows], np.uint8, 1),
                ("changed_confidence_global", [1, rows], np.uint16, 65535)):
            value = np.asarray(inputs[name]["raw"], dtype=np.int64)
            if (list(value.shape) != shape or inputs[name]["shape"] != shape or
                    np.any(value < 0) or np.any(value > upper)):
                raise ValueError(f"invalid cross-block vector {name}")
            vectors[name] = value.astype(dtype)
        if record.get("advance_block"):
            if lib.rtl_cross_block_advance(ctypes.byref(state), record["block_start"]):
                raise ValueError("C11 rejected block advance")
        result = lib.rtl_cross_block_observe(ctypes.byref(state), query_count,
            vectors["query_positions"].ctypes.data_as(u16p), vectors["relation"].ctypes.data_as(u16p),
            consumed_count, vectors["consumed_positions"].ctypes.data_as(u16p),
            vectors["changed_global"].ctypes.data_as(u8p),
            vectors["changed_remask_global"].ctypes.data_as(u8p),
            vectors["changed_confidence_global"].ctypes.data_as(u16p))
        if result:
            raise ValueError(f"C11 rejected cross-block step {record['step_index']}")
        for name, target in record["expected"].items():
            value = np.array(getattr(state, name), dtype=np.uint16)
            value = value.reshape(2048, 32)[:rows] if name == "relation" else value[:rows]
            if not np.array_equal(value, target["raw"]):
                return dict(match=False, step_index=record["step_index"], field=name)
            checked += 1
    return dict(match=True, records=len(reference_data["records"]), checked_fields=checked)


class RefreshState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "rows", "target_half_rows", "regular_steps", "cumulative_active_rows",
        "allowed_active_rows")] + [
        ("dependency", ctypes.c_uint16 * (96 * 96)),
        ("pending", ctypes.c_uint16 * 96),
        ("new_invalidation", ctypes.c_uint16 * 96),
        ("refresh", ctypes.c_uint8 * 96),
        ("mandatory", ctypes.c_uint8 * 96),
        ("optional_selected", ctypes.c_uint8 * 96)]


class BoundarySelection(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "deep_count", "optional_count", "ranked_count")] + [
        (name, ctypes.c_uint16 * 2048) for name in (
            "deep_positions", "selected_optional", "ranked_candidates")]


class JointInput(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "base_rows", "future_rows", "target_rows", "max_next_rows", "next_block_start")] + [
        ("base_positions", ctypes.c_uint16 * 48),
        ("base_bits", ctypes.c_uint8 * 48), ("forecast_bits", ctypes.c_uint8 * 48),
        ("future_bits", ctypes.c_uint8 * 32), ("unresolved", ctypes.c_uint8 * 32),
        ("tentative", ctypes.c_uint8 * 32), ("priority", ctypes.c_uint16 * 32),
        ("service_count", ctypes.c_int32 * 32),
        ("prediction_target", ctypes.c_uint32), ("current_prediction_rows", ctypes.c_uint32),
        ("source_b_attempts", ctypes.c_uint32 * 32), ("last_confidence", ctypes.c_uint16 * 32),
        ("retry_min_confidence", ctypes.c_uint16), ("min_reuse_score", ctypes.c_uint16),
        ("dependency", ctypes.c_uint16 * 32), ("block_step_index", ctypes.c_uint16),
        ("source_b_dependency_tie_rank", ctypes.c_uint8), ("source_b_a4_only", ctypes.c_uint8)]


class ActivationResidency(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "a4_rows", "a8_rows", "issued_slice_units", "pe_issue_groups")] + [
        (name, ctypes.c_uint32 * 3) for name in (
            "activation_bytes", "capacity_bytes", "fragments")]


class JointSelection(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "progress_count", "added_count", "rejected_current", "rejected_forecast", "rejected_budget")] + [
        (name, ctypes.c_uint8 * 32) for name in ("progress", "added", "added_bits")] + [
        (name, ActivationResidency) for name in ("base", "joint", "forecast")]


def compare_joint_selection_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-uaps-token-selection/v1":
        raise ValueError("expected supra-uaps-token-selection/v1 reference schema")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    for name, structure in (("rtl_joint_input_bytes", JointInput),
                            ("rtl_joint_selection_bytes", JointSelection)):
        function = getattr(lib, name)
        function.restype = ctypes.c_size_t
        if function() != ctypes.sizeof(structure):
            raise ValueError("C11 joint selection layout differs from binding")
    lib.rtl_joint_select.argtypes = [ctypes.POINTER(JointInput), ctypes.POINTER(JointSelection)]
    lib.rtl_joint_select.restype = ctypes.c_int
    checked = 0
    for record in reference_data["records"]:
        inputs, expected = record["inputs"], record["expected"]
        request, response = JointInput(), JointSelection()
        request.base_rows = len(inputs["base_positions"]["raw"])
        request.future_rows = len(inputs["next_activation_bits"]["raw"])
        request.prediction_target = max(0, inputs.get("prediction_token_target", inputs.get("target_prediction_rows", -1)))
        request.current_prediction_rows = inputs.get("current_prediction_tokens", inputs.get("current_prediction_rows", 0))
        request.min_reuse_score = record.get("min_reuse_score_bf16", 0)
        flags = record.get("source_b_flags", 0)
        request.source_b_dependency_tie_rank = flags & 1
        request.source_b_a4_only = (flags >> 1) & 1
        request.block_step_index = inputs.get("step_index", 0)
        if "next_dependency_score" in inputs:
            request.dependency[:request.future_rows] = inputs["next_dependency_score"]["raw"]
        if not 1 <= request.base_rows <= 48 or request.future_rows > 32:
            raise ValueError("invalid joint selection row counts")
        for field, value, lower, upper in (
                ("target_rows", reference_data.get("target_joint_tokens", reference_data.get("target_joint_rows")), 1, 48),
                ("max_next_rows", reference_data.get("max_next_tokens", reference_data.get("max_next_rows")), 1, 32),
                ("next_block_start", inputs["next_block_start"], 0, 2048)):
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"invalid joint scalar {field}")
            setattr(request, field, value)
        for field, source, dtype, lower, upper in (
                ("base_positions", "base_positions", "torch.int64", 0, 2047),
                ("base_bits", "base_activation_bits", "torch.int8", 4, 8),
                ("forecast_bits", "next_step_base_activation_bits", "torch.int8", 4, 8),
                ("future_bits", "next_activation_bits", "torch.int8", 4, 8),
                ("unresolved", "next_unresolved", "torch.bool", 0, 1),
                ("tentative", "next_tentative", "torch.bool", 0, 1),
                ("priority", "next_priority", "torch.bfloat16", 0, 65535),
                ("service_count", "next_service_count", "torch.int32", -(2**31), 2**31 - 1)):
            vector = inputs[source]
            values = np.asarray(vector["raw"], dtype=np.int64)
            count = request.base_rows if field in ("base_positions", "base_bits", "forecast_bits") else request.future_rows
            if (values.shape != (count,) or vector["shape"] != [count] or
                    vector["dtype"] != dtype or np.any(values < lower) or np.any(values > upper)):
                raise ValueError(f"invalid joint vector {source}")
            getattr(request, field)[:count] = values.tolist()
        limit = inputs.get("max_future_admission_attempts", inputs.get("max_source_b_attempts", -1))
        retry = inputs.get("future_admission_retry_min_confidence", inputs.get("source_b_retry_min_confidence", 0.0))
        threshold = int(torch.tensor(retry, dtype=torch.bfloat16).view(torch.int16).item())
        confidence = inputs.get("next_last_confidence", {}).get("raw", [-1] * request.future_rows)
        request.retry_min_confidence = threshold
        request.last_confidence[:request.future_rows] = [value & 0xffff for value in confidence]
        attempts = inputs.get("future_admission_attempts", inputs.get("source_b_attempts", {})).get("raw", [0]*request.future_rows)
        request.source_b_attempts[:request.future_rows] = attempts
        allowed = (ctypes.c_uint8 * request.future_rows)(*[
            limit < 0 or attempt < limit
            for attempt in attempts])
        if limit >= 0 or retry:
            lib.rtl_joint_select_allowed.argtypes = [ctypes.POINTER(JointInput),
                ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(JointSelection)]
            lib.rtl_joint_select_allowed.restype = ctypes.c_int
            status = lib.rtl_joint_select_allowed(ctypes.byref(request), allowed, ctypes.byref(response))
        else:
            status = lib.rtl_joint_select(ctypes.byref(request), ctypes.byref(response))
        if status:
            raise ValueError(f"C11 rejected joint selection {record['name']}")
        if response.progress_count > 32 or response.added_count > response.progress_count:
            raise ValueError("C11 joint result count exceeds storage")
        actual = dict(
            progress_local_positions=list(response.progress)[:response.progress_count],
            added_local_positions=list(response.added)[:response.added_count],
            added_activation_bits=list(response.added_bits)[:response.added_count],
            rejected_for_current_step_capacity=response.rejected_current,
            rejected_for_next_step_verification_capacity=response.rejected_forecast,
            rejected_for_budget=response.rejected_budget)
        for name, value in actual.items():
            # Public captures retain decisions/residencies, while direct
            # scheduler captures also include rejection diagnostics.
            if name.startswith("rejected_for_") and name not in expected:
                continue
            target = expected[name]["raw"] if isinstance(expected[name], dict) else expected[name]
            if not np.array_equal(value, target):
                return dict(match=False, record=record["name"], field=name,
                            actual=value, expected=target)
            checked += 1
        for field, name in (("base", "base_residency"), ("joint", "joint_residency"),
                            ("forecast", "next_step_verification_residency")):
            value, target = getattr(response, field), expected[name]
            scalars = ("a4_rows", "a8_rows", "issued_slice_units", "pe_issue_groups")
            public_fields = {"a4_rows": "a4_token_count", "a8_rows": "a8_token_count",
                             "issued_slice_units": "issued_activation_slots"}
            match = all(getattr(value, item) == target.get(public_fields.get(item, item), target.get(item)) for item in scalars)
            operators = target.get("operators", [])
            if "operators" in target:
                match &= len(operators) == 3
            for i, operator in enumerate(operators):
                if i >= 3:
                    break
                match &= (value.activation_bytes[i] == operator["activation_bytes"] and
                          value.capacity_bytes[i] == operator["capacity_bytes"] and
                          value.fragments[i] == operator["fragments"] == operator["weight_read_multiplier"])
            if not match:
                return dict(match=False, record=record["name"], field=name)
            checked += 1
    return dict(match=True, records=len(reference_data["records"]), checked_fields=checked)


def compare_atse_uaps_limits_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-atse-uaps-limits/v1":
        raise ValueError("expected supra-atse-uaps-limits/v1 reference schema")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.rtl_boundary_deep_precision.argtypes = [ctypes.c_uint32, u8p, u8p, u8p, ctypes.c_uint32, u8p]
    lib.rtl_boundary_deep_precision.restype = ctypes.c_int
    checked = 0
    joint_records = []
    for record in reference_data["records"]:
        inputs, expected = record["inputs"], record["expected"]
        if record["kind"] == "source_b_selection":
            converted = dict(inputs)
            converted["next_step_base_activation_bits"] = inputs["base_activation_bits"]
            converted["next_service_count"] = dict(dtype="torch.int32",
                shape=inputs["next_activation_bits"]["shape"], raw=[0] * len(inputs["next_activation_bits"]["raw"]))
            joint_records.append(dict(record, inputs=converted))
            continue
        if record["kind"] != "deep_precision":
            continue
        count = len(inputs["activation_bits"]["raw"])
        arrays = [(ctypes.c_uint8 * count)(*inputs[key]["raw"])
                  for key in ("activation_bits", "score_q8", "protected")]
        actual = (ctypes.c_uint8 * count)(*inputs["activation_bits"]["raw"])
        if inputs["a8_limit"] >= 0:
            if lib.rtl_boundary_deep_precision(count, actual, arrays[1], arrays[2], inputs["a8_limit"], actual):
                raise ValueError(f"C11 rejected deep precision {record['name']}")
        if list(actual) != expected["activation_bits"]["raw"]:
            return dict(match=False, record=record["name"], field="activation_bits")
        checked += 1
    result = compare_joint_selection_reference_data(dict(schema="supra-uaps-token-selection/v1",
        target_joint_rows=48, max_next_rows=8, records=joint_records), library)
    if not result["match"]:
        return result
    return dict(match=True, records=checked + len(joint_records),
        checked_fields=checked + result["checked_fields"])


def compare_boundary_selection_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-atse-cross-block-token-selection/v1":
        raise ValueError("unsupported boundary-selection reference_data")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    lib.rtl_boundary_selection_bytes.restype = ctypes.c_size_t
    if lib.rtl_boundary_selection_bytes() != ctypes.sizeof(BoundarySelection):
        raise ValueError("C11 boundary selection layout differs from binding")
    u16p, u8p = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint8)
    lib.rtl_boundary_select.argtypes = [ctypes.c_uint32, u8p, ctypes.c_uint32,
        ctypes.c_uint32, u16p, ctypes.c_uint32, u16p, ctypes.c_uint32, u16p,
        ctypes.c_uint32, u16p, ctypes.c_uint32, ctypes.POINTER(BoundarySelection)]
    lib.rtl_boundary_select.restype = ctypes.c_int
    checked = 0
    for record in reference_data["records"]:
        inputs = record["inputs"]
        vectors = {}
        for name in ("score_q8", "current_positions", "eligible_positions",
                     "mandatory_candidate_positions", "required_candidate_positions"):
            field = inputs[name]
            value = np.asarray(field["raw"], dtype=np.int64)
            limit = 256 if name == "score_q8" else 2048
            dtype = "torch.uint8" if name == "score_q8" else "torch.int64"
            if (value.ndim != 1 or list(value.shape) != field["shape"] or
                    field["dtype"] != dtype or len(value) > 2048 or
                    np.any(value < 0) or np.any(value >= limit)):
                raise ValueError(f"invalid boundary vector {name}")
            vectors[name] = value.astype(np.uint8 if name == "score_q8" else np.uint16)
        for name in ("target_token_count", "required_candidate_count"):
            if not 0 <= inputs[name] <= 2048:
                raise ValueError(f"invalid boundary scalar {name}")
        scores = vectors["score_q8"]
        args = []
        for name in ("current_positions", "eligible_positions",
                     "mandatory_candidate_positions", "required_candidate_positions"):
            vector = vectors[name]
            args.extend((len(vector), vector.ctypes.data_as(u16p)))
        actual = BoundarySelection()
        status = lib.rtl_boundary_select(len(scores), scores.ctypes.data_as(u8p),
            inputs["target_token_count"], *args, inputs["required_candidate_count"],
            ctypes.byref(actual))
        if status:
            raise ValueError(f"C11 rejected boundary selection {record['name']}")
        for name, count in (("deep_positions", actual.deep_count),
                            ("selected_optional", actual.optional_count),
                            ("ranked_candidates", actual.ranked_count)):
            if count > 2048:
                raise ValueError("C11 boundary result count exceeds storage")
            expected = record["expected"][name]
            if expected["shape"] != [count] or not np.array_equal(
                    np.array(getattr(actual, name))[:count], expected["raw"]):
                return dict(match=False, record=record["name"], field=name)
            checked += 1
    return dict(match=True, records=len(reference_data["records"]), checked_fields=checked)


def compare_saved_dependency_reference_data(reference_data: dict, library: Path) -> dict:
    if reference_data["schema"] != "supra-atse-attention-dependencies/v1":
        raise ValueError("unsupported saved-dependency reference_data")
    lib = ctypes.CDLL(str(library.resolve(strict=True)))
    lib.rtl_refresh_state_bytes.restype = ctypes.c_size_t
    if lib.rtl_refresh_state_bytes() != ctypes.sizeof(RefreshState):
        raise ValueError("C11 refresh state layout does not match binding")
    u16p, u8p = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint8)
    lib.rtl_refresh_init.argtypes = [ctypes.POINTER(RefreshState), ctypes.c_uint32,
                                    ctypes.c_uint32, u16p]
    lib.rtl_refresh_observe.argtypes = [ctypes.POINTER(RefreshState), ctypes.c_uint32,
        u16p, u16p, u8p, u8p, u8p, u16p]
    lib.rtl_refresh_init.restype = lib.rtl_refresh_observe.restype = ctypes.c_int
    start, end = reference_data["region_start"], reference_data["region_end"]
    rows = end - start
    if not 1 <= rows <= 96 or reference_data["target_active_rows"] * 2 % 1:
        raise ValueError("C11 reference requires <=96 rows and half-row budget units")
    full_sequence = np.array(reference_data["initial_attention_profile"]["dependency_mean"]["raw"], dtype=np.uint16)
    if full_sequence.shape != (1, rows, rows):
        raise ValueError("initial attention dependency shape does not match region")
    state = RefreshState()
    if lib.rtl_refresh_init(ctypes.byref(state), rows,
            int(reference_data["target_active_rows"] * 2), full_sequence.ctypes.data_as(u16p)):
        raise ValueError("C11 refresh initialization rejected reference_data")
    checked = 0
    for record in reference_data["records"]:
        inputs, expected = record["inputs"], record["expected"]
        query = np.array(inputs["query_positions"]["raw"], dtype=np.int64) - start
        if np.any(query < 0) or np.any(query >= rows):
            raise ValueError("query outside reference_data region")
        query = query.astype(np.uint16)
        dependency = np.array(inputs["dependency_mean"]["raw"], dtype=np.uint16)
        predicted = np.array(inputs["predicted"]["raw"], dtype=np.uint8)
        if query.ndim != 1 or dependency.shape != (1, len(query), rows) or predicted.shape != (rows,):
            raise ValueError("refresh query/dependency/prediction shapes do not match")
        if any(np.shape(inputs[name]["raw"]) != (1, reference_data["total_length"])
               for name in ("changed_global", "changed_remask_global", "changed_confidence_global")):
            raise ValueError("refresh change masks do not cover the full sequence")
        changed, remasked, confidence = [np.array(inputs[name]["raw"], dtype=dtype)[
            0, start:end].copy() for name, dtype in (
                ("changed_global", np.uint8), ("changed_remask_global", np.uint8),
                ("changed_confidence_global", np.uint16))]
        refreshed = np.array(state.refresh, dtype=np.uint8)[:rows].copy()
        result = lib.rtl_refresh_observe(ctypes.byref(state), len(query),
            query.ctypes.data_as(u16p), dependency.ctypes.data_as(u16p),
            predicted.ctypes.data_as(u8p), changed.ctypes.data_as(u8p),
            remasked.ctypes.data_as(u8p), confidence.ctypes.data_as(u16p))
        if result:
            raise ValueError(f"C11 rejected refresh step {record['step_index']}")
        actual = dict(dependency=np.array(state.dependency).reshape(96, 96)[:rows, :rows],
            pending=np.array(state.pending)[:rows],
            dependency_score=np.array(state.pending)[:rows],
            new_invalidation=np.array(state.new_invalidation)[:rows], refreshed=refreshed,
            next_refresh=np.array(state.refresh)[:rows], mandatory=np.array(state.mandatory)[:rows],
            optional_selected=np.array(state.optional_selected)[:rows],
            allowed_active_rows=state.allowed_active_rows, regular_steps=state.regular_steps,
            cumulative_active_rows=state.cumulative_active_rows)
        for name, value in actual.items():
            target = expected[name]["raw"] if isinstance(expected[name], dict) else expected[name]
            if not np.array_equal(value, target):
                return dict(match=False, step_index=record["step_index"], field=name)
            checked += 1
    return dict(match=True, records=len(reference_data["records"]), checked_fields=checked)


def read_captured_control(index: Path, *, payload_root=None):
    """Format captured generator inputs and results."""
    from model_capture import read_tensor
    metadata = json.loads(index.read_text())
    control = metadata["control_observation"]
    if metadata["validation"]["status"] != "PASS":
        raise ValueError("capture numeric replay must pass before control formatting")
    entries = {entry["name"]: entry for entry in metadata["tensors"] if entry["role"] == "control"}
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"tensor"}:
                root = Path(payload_root) if payload_root is not None else Path(control.get("payload_root", "."))
                if not root.is_absolute():
                    root = index.parent / root
                return read_tensor(entries[value["tensor"]], index=index, payload_root=root)
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    checkpoints = decode(control["checkpoints"])
    prediction = decode(control["prediction"])
    return metadata, checkpoints, prediction


def captured_feature1_reference(index: Path, *, payload_root=None) -> dict:
    """Replay maintained pending/selection functions on observed reduced dependencies."""
    import copy
    metadata, checkpoints, prediction = read_captured_control(index, payload_root=payload_root)
    before, after = checkpoints["before_postprocess"], checkpoints["after_postprocess"]
    phase = prediction["forward_kind"]
    if phase not in ("local_block", "full_sequence", "boundary_refresh", "local_confirmation", "local_forced_finish"):
        raise ValueError(f"pending reference has no postprocessing snapshot for {phase}")
    state = AttentionRefreshState.__new__(AttentionRefreshState)
    state.__dict__.update(copy.deepcopy(before["packed_state"]))
    if state.initial_burst_rows:
        raise ValueError("initial_burst_rows must be zero for this DDR layout")
    begin, end = before["block_start"], before["block_end"]
    changed = before["tokens"] != after["tokens"]
    confidence = torch.ones_like(changed, dtype=torch.bfloat16)
    confidence[:, begin:end] = after["changed_confidence"]
    remasked = changed & (confidence < 1.0)
    predicted = torch.zeros(state.region_end-state.region_start, dtype=torch.bool)
    predicted[begin-state.region_start:end-state.region_start] = after["block_state"]["state"][0] != 2
    profile = before["packed_profile"]
    if profile is None:
        if phase not in ("full_sequence", "boundary_refresh"):
            raise ValueError("regular step is missing its dependency profile")
        state.plan_initial(predicted=predicted, changed_global=changed,
                           changed_confidence_global=confidence, changed_remask_global=remasked)
        positions = prediction["input_positions"].reshape(-1)
        positions = positions[(positions >= state.region_start) & (positions < state.region_end)]
        profile = dict(query_positions=positions, dependency_mean=state.dependency.index_select(
            0, positions.long() - state.region_start).unsqueeze(0))
    else:
        state.observe(profile, predicted=predicted, changed_global=changed,
                      changed_confidence_global=confidence, changed_remask_global=remasked)
    from model_capture import raw_equal
    for name, expected in after["packed_state"].items():
        actual = getattr(state, name)
        if isinstance(expected, torch.Tensor):
            same = raw_equal(actual, expected)
        else:
            same = actual == expected
        if not same:
            raise ValueError(f"Feature1 CPU replay differs from captured selection: {name}")
    selected = state.refresh.nonzero().flatten()+state.region_start
    next_inputs = checkpoints["next_layer_inputs"]
    joint = checkpoints.get("next_joint_selection")
    selected_positions = (joint["inputs"]["base_positions"] if joint else next_inputs["query_position_ids"])
    selected_bits = joint["inputs"]["base_row_bits"] if joint else next_inputs["activation_bits"]
    if selected.tolist() != selected_positions.reshape(-1).tolist():
        raise ValueError("Feature1 selected positions differ from the next actual base query")
    def record_state(value):
        return dict(dependency=packed(value["dependency"]), pending=packed(value["pending"]),
            refresh=packed(value["refresh"]), regular_steps=value["regular_steps"],
            cumulative_active_token_count=value["cumulative_active_rows"])
    expected = record_state(after["packed_state"])
    expected["next_refresh"] = expected.pop("refresh")
    return dict(schema="supra-atse-in-block-boundaries/v1", source_index=str(index.resolve()),
        region_start=state.region_start, region_end=state.region_end, total_length=state.total_length,
        target_active_token_count=state.target_active_rows,
        records=[dict(name=f"observed_event{prediction['capture_index']}", before=record_state(before["packed_state"]),
            inputs=dict(query_positions=packed(profile["query_positions"]), dependency_mean=packed(profile["dependency_mean"]),
                dependency_bf16=packed(state.dependency.index_select(
                    0, profile["query_positions"].long()-state.region_start).unsqueeze(0)),
                predicted=packed(predicted), changed_global=packed(changed), changed_confidence_global=packed(confidence),
                changed_remask_global=packed(remasked)), expected=expected,
            current_begin=begin, current_end=end,
            current_bits=packed(checkpoints["transition"]["next_current_row_bits"].reshape(-1)),
            context_a8=(0 if metadata["generation_config"].get("feature3_precision_policy") == "all_a8"
                        else metadata["generation_config"]["packed_attention_context_a8_rows"]),
            selected_bits=packed(selected_bits))])


def captured_block_initialization_reference(index: Path, *, payload_root=None) -> dict:
    """Replay dependency-only block initialization from captured state and request configuration."""
    from generation.engine import block_initialization_relative_selection
    metadata, checkpoints, prediction = read_captured_control(index, payload_root=payload_root)
    config, before = metadata["generation_config"], checkpoints["before_forward"]
    if (not prediction["layer0_global_selected_deep"] or
            before["block_index"] != config["cross_block_full_prefix_oracle_block"] or
            not config["cross_block_block_initialization_dependency_only"]):
        raise ValueError("capture is not a configured dependency-only sparse block initialization")
    sequence, begin, end = before["total_length"], before["block_start"], before["block_end"]
    current = torch.arange(begin, end)
    state = before["cross_block_prefix_state"]
    pending, future = state["pending"], state["future_pending"]
    due = before["cache_refresh_due"][0].nonzero().flatten()
    due = due[~torch.isin(due, current)]
    protected_begin = config["question_start_token"]
    if protected_begin < 0:
        protected_begin = before["prompt_length"]
    protected_end = sequence if config["cross_block_block_initialization_protect_generation"] else before["prompt_length"]
    blocks = config["cross_block_block_initialization_protected_future_blocks"]
    if blocks >= 0:
        protected_end = min(protected_end, end+(end-begin)*blocks)
    mandatory = torch.unique(torch.cat((due, torch.arange(protected_begin,protected_end))), sorted=True)
    mandatory = mandatory[~torch.isin(mandatory,current)]
    include_future = config["cross_block_block_initialization_include_future_dependency"]
    if include_future and future is None:
        raise ValueError("block initialization requires observed future pending")
    future = torch.zeros_like(pending) if future is None else future
    scores = torch.maximum(pending,future) if include_future else pending
    optional = torch.arange(sequence)
    optional = optional[~torch.isin(optional,current)]
    ranked = optional[torch.argsort(scores[optional],descending=True,stable=True)]
    target, floor = config["cross_block_block_initialization_deep_rows"], config["cross_block_block_initialization_relative_score_floor"]
    if floor:
        eligible,effective,_,_ = block_initialization_relative_selection(ranked,scores[ranked],mandatory,len(current),target,floor)
    else:
        effective = max(target,len(current)+len(mandatory))
        remaining = ranked[~torch.isin(ranked,mandatory)]
        eligible = torch.cat((mandatory,remaining[:effective-len(current)-len(mandatory)]))
    # In this branch candidate count exactly fills the target. L0 scores cannot
    # change membership; compare that independent result with actual L1 inputs.
    expected = torch.sort(torch.cat((current,eligible))).values
    actual = prediction["input_positions"].reshape(-1).long()
    if not torch.equal(expected,actual):
        raise ValueError("block initialization selection differs from captured L1")
    source = dict(previous_pending=packed(pending), previous_future_pending=packed(future),
        shortlist_token_count=min(800,2*(target-len(current))),
        shortlist_flags=6 | int(include_future) | (8 if config["cross_block_block_initialization_dependency_tiebreak"] else 0),
        relative_score_floor_bf16=int(torch.tensor(floor).bfloat16().view(torch.int16)) & 65535,
        protected_begin=protected_begin, protected_end=protected_end,
        score_q8=packed(torch.zeros(sequence,dtype=torch.uint8)), current_positions=packed(current),
        eligible_positions=packed(torch.sort(eligible).values), mandatory_candidate_positions=packed(due),
        required_candidate_positions=packed(torch.empty(0,dtype=torch.long)), required_candidate_count=0,
        target_token_count=target, deep_activation_bits=config["cross_block_block_initialization_deep_bits"])
    return dict(schema="supra-atse-cross-block-token-selection/v1", source_index=str(index.resolve()),
        records=[dict(name=f"observed_block_initialization{prediction['capture_index']}",inputs=source,
            expected=dict(deep_positions=packed(expected),shortlist_mandatory_positions=packed(mandatory),effective_target=effective))])


def captured_boundary_reference(index: Path, *, payload_root=None) -> dict:
    """Use captured scout scores and state to replay ordinary boundary selection."""
    from generation.engine import _select_boundary_layer0_deep_positions
    metadata, checkpoints, prediction = read_captured_control(index, payload_root=payload_root)
    config, before = metadata["generation_config"], checkpoints["before_forward"]
    if (not prediction["layer0_global_selected_deep"] or
            before["block_index"] == config["cross_block_full_prefix_oracle_block"] or
            prediction["boundary_deep_bits"]):
        raise ValueError("ordinary boundary reference requires a two-stage event")
    scout = checkpoints["after_l0_scout"]
    sequence, begin, end = before["total_length"], before["block_start"], before["block_end"]
    current = torch.arange(begin, end)
    if not torch.equal(scout["current_positions"], current):
        raise ValueError("scout current positions differ from the generator state")
    transition = before["cross_block_carry_changed_positions"]
    observed_transition = scout["transition_positions"]
    if transition.numel():
        if not isinstance(observed_transition, torch.Tensor) or not torch.equal(observed_transition, transition):
            raise ValueError("observed boundary transitions differ from the generator state")
    elif observed_transition is not None:
        raise ValueError("empty transition history requires current-to-context scout scores")
    pending = before["cross_block_prefix_state"]["pending"]
    due = before["cross_block_carry_refresh_due_positions"]
    mandatory = due[~torch.isin(due, current)]
    context = torch.cat((torch.arange(before["region_start"], begin),
                         torch.arange(end, before["region_end"])))
    optional = torch.arange(sequence)
    optional = optional[~torch.isin(optional, current)]
    target = config["cross_block_boundary_target_rows"]
    shortlist_count = min(len(optional), 2 * (target - len(current)))
    ranked = optional[torch.argsort(pending[optional], descending=True, stable=True)]
    eligible = torch.unique(torch.cat((ranked[:shortlist_count], context, mandatory)), sorted=True)
    if config.get("packed_attention_force_full_current", False):
        raise ValueError("boundary capture requires packed_attention_force_full_current=False")
    required_count = 0
    scores = scout["score_q8"]
    positions, _, _ = _select_boundary_layer0_deep_positions(scores,
        current_positions=current, target_rows=target, eligible_positions=eligible,
        mandatory_candidate_positions=mandatory, required_candidate_positions=context if required_count else context[:0],
        required_candidate_count=required_count)
    if positions.tolist() != prediction["input_positions"].reshape(-1).tolist():
        raise ValueError("boundary selection differs from captured L1 positions")
    from model_capture import read_tensor
    entry = next(e for e in metadata["tensors"] if e["role"] == "input" and e["name"] in ("layer0.activation_bits", "activation_bits"))
    initial_bits = read_tensor(entry, index=index).reshape(-1).to(torch.int8)
    bits = initial_bits[positions]
    limit = config["cross_block_boundary_deep_a8_row_limit"]
    if limit >= 0:
        bits = boundary_deep_precision_bits(bits, scores[positions], torch.isin(positions, torch.cat((current, mandatory))), limit)
    if not torch.equal(bits, prediction["row_bits"].reshape(-1).to(torch.int8)):
        raise ValueError("boundary precision differs from captured L1")
    return dict(schema="supra-atse-cross-block-token-selection/v1", source_index=str(index.resolve()),
        records=[dict(name=f"observed_boundary{prediction['capture_index']}", inputs=dict(
            previous_pending=packed(pending), shortlist_token_count=shortlist_count, shortlist_flags=0,
            current_positions=packed(current), mandatory_candidate_positions=packed(mandatory),
            context_positions=packed(context), required_candidate_positions=packed(context),
            eligible_positions=packed(eligible), required_candidate_count=required_count,
            initial_activation_bits=packed(initial_bits),
            transition_positions=packed(transition), score_q8=packed(scores),
            target_token_count=target, deep_a8_limit=limit,
            keep_global_l0_cache=prediction["layer0_keep_global_cache"]),
            expected=dict(deep_positions=packed(positions), deep_bits=packed(bits), effective_target=target))])


def generated_boundary_reference(index: Path) -> dict:
    """Format actual CPU layer scout observations for an online RTL boundary."""
    from prepare_layer_testcase import ReferenceData
    from generation.engine import _select_boundary_layer0_deep_positions
    first = ReferenceData(index)
    info, scout = first.metadata, first.metadata["boundary_scout"]
    following = Path(info["continuation_index"])
    second = ReferenceData(following if following.is_absolute() else index.parent / following)
    sequence = info["sequence"]
    transition = scout.get("transition_positions")
    if "transition_positions" not in scout or "shortlist_pending_raw" not in scout:
        raise ValueError("online generated boundary requires recorded scout direction and actual shortlist history")
    current, mandatory, context, eligible = [torch.tensor(scout[name], dtype=torch.long) for name in (
        "current_positions", "mandatory_candidate_positions", "shortlist_context_positions", "eligible_positions")]
    pending = torch.tensor(scout["shortlist_pending_raw"], dtype=torch.int16).view(torch.bfloat16)
    optional = torch.tensor([p for p in range(sequence) if p not in current], dtype=torch.long)
    shortlist_count = min(len(optional), 2 * (scout["target_token_count"] - len(current)))
    ranked = optional[torch.argsort(pending[optional], descending=True, stable=True)]
    if not torch.equal(eligible, torch.unique(torch.cat((ranked[:shortlist_count], context, mandatory)), sorted=True)):
        raise ValueError("generated boundary candidates differ from the configured shortlist")
    scores = torch.tensor(scout["score_q8"], dtype=torch.uint8)
    positions, _, _ = _select_boundary_layer0_deep_positions(scores,
        current_positions=current, target_rows=scout["target_token_count"], eligible_positions=eligible,
        mandatory_candidate_positions=mandatory, required_candidate_positions=context,
        required_candidate_count=scout["required_candidate_count"])
    initial_bits = torch.from_numpy(first.load("input", "activation_bits").copy()).reshape(-1)
    bits = initial_bits[positions]
    if (positions.tolist() != second.load("input", "positions").reshape(-1).tolist() or
            bits.tolist() != second.load("input", "activation_bits").reshape(-1).tolist()):
        raise ValueError("generated boundary reference differs from the actual deep input")
    return dict(schema="supra-atse-cross-block-token-selection/v1", source_index=str(index.resolve()),
        records=[dict(name="generated_boundary", inputs=dict(
            previous_pending=packed(pending), shortlist_token_count=shortlist_count, shortlist_flags=0,
            current_positions=packed(current), mandatory_candidate_positions=packed(mandatory),
            context_positions=packed(context), required_candidate_positions=packed(context),
            eligible_positions=packed(eligible), required_candidate_count=scout["required_candidate_count"],
            initial_activation_bits=packed(initial_bits), transition_positions=packed(torch.tensor(transition or [], dtype=torch.long)),
            score_q8=packed(scores), target_token_count=scout["target_token_count"], deep_a8_limit=-1,
            keep_global_l0_cache=False), expected=dict(deep_positions=packed(positions), deep_bits=packed(bits),
                effective_target=len(positions)))])


def captured_feature2_reference(index: Path, *, payload_root=None) -> dict:
    """Format captured generator inputs and results."""
    hardware = ALIGNMENT_ROOT.parent / "hardware"
    sys.path.insert(0, str(hardware / "scripts"))
    from build_forward_postprocess_config import encode_admission_budget
    metadata, checkpoints, prediction = read_captured_control(index, payload_root=payload_root)
    before, after = checkpoints["before_postprocess"], checkpoints["transition"]
    phase = prediction["forward_kind"]
    closeout_kind = {"local_confirmation": 1, "local_forced_finish": 2}.get(phase, 0)
    if phase not in ("local_block", "full_sequence", "boundary_refresh", "local_confirmation", "local_forced_finish"):
        raise ValueError("control reference requires a generator forward with a completed transition")
    begin, end = before["block_start"], before["block_end"]
    if end - begin != 32 or prediction["prediction_positions"].reshape(-1).tolist() != list(range(begin, end)):
        raise ValueError("control capture must expose the complete current block")
    state = DraftVerifyBlockState((1, 32), device="cpu")
    for name, value in before["block_state"].items():
        setattr(state, name, value.clone())
    precision_policy = metadata["generation_config"].get("feature3_precision_policy", "original")
    bits = state.row_bits(maturity_age=3, policy=precision_policy).reshape(-1)
    for position, precision in zip(prediction["input_positions"].reshape(-1).tolist(), prediction["row_bits"].reshape(-1).tolist()):
        if begin <= position < end:
            bits[position - begin] = precision
    def codes(tensor):
        value = tensor.to(torch.bfloat16)
        if not torch.equal(value.float(), tensor.float()):
            raise ValueError("observed candidate probability is not BF16 materialized")
        return (value.contiguous().view(torch.int16).reshape(-1).int() & 65535).tolist()
    def code(value):
        return int(torch.as_tensor(value, dtype=torch.bfloat16).view(torch.int16)) & 65535
    def values(value):
        return value.reshape(-1).tolist()
    def bitmask(values):
        return sum(1 << (int(position) - begin) for position in values)
    trace = after["trace"]
    before_tokens = before["tokens"][:, begin:end].reshape(-1).tolist()
    after_tokens = values(after["tokens"])
    expected = {key: values(after[key]) for key in ("state", "tokens", "last_top1", "precision_age", "commit_origin")}
    expected["bits"] = values(after["next_current_row_bits"])
    expected["masks"] = {field: bitmask(trace[key]) for field, key in (
        ("selected", "transferred_positions"), ("direct", "direct_locked_positions"),
        ("stable", "stable_tentative_positions"), ("fallback", "fallback_tentative_positions"),
        ("confirmed", "confirmed_positions"), ("remasked", "remasked_positions"))}
    expected["masks"]["token_changed"] = sum(1 << i for i in range(32) if before_tokens[i] != after_tokens[i])
    if closeout_kind == 2:
        expected["masks"]["tail_closed"] = expected["masks"]["selected"]
    request_end = checkpoints.get("request_end") is True
    if request_end or checkpoints["next_forward"]["block_index"] != before["block_index"]:
        final = checkpoints.get("current_state_at_request_end" if request_end else "current_state_at_next_forward")
        if request_end and final is None:
            raise ValueError("request-end control capture lacks the actual final block state")
        if final is None and bool((after["state"] != 2).any()):
            raise ValueError("terminal control capture lacks the actual post-tail current state; recapture")
        if final is not None:
            if final["block_start"] != begin or final["block_end"] != end:
                raise ValueError("post-tail capture belongs to a different current block")
            tail_state = DraftVerifyBlockState((1, 32), device="cpu")
            for name in ("state", "last_top1", "precision_age", "commit_origin"):
                setattr(tail_state, name, after[name].clone())
            closed = tail_state.bypass_tail_confirmation(before["tail_confirmation_policy"])
            for name, value in final["block_state"].items():
                if not torch.equal(getattr(tail_state, name), value):
                    raise ValueError(f"actual block exit differs from tail rule: {name}")
                expected[name] = values(value)
            if not torch.equal(final["tokens"], after["tokens"]):
                raise ValueError("block exit unexpectedly changed current token IDs")
            expected["bits"] = values(tail_state.row_bits(maturity_age=3, policy=precision_policy))
            expected["masks"]["tail_closed"] = expected["masks"].get("tail_closed", 0) | sum(1 << i for i,v in enumerate(values(closed)) if v)
    raw_probability = codes(prediction["teacher_current_token_probability"])
    confidence = codes(prediction["teacher_top1_action_confidence"])
    raw_confidence = codes(prediction["teacher_top1_confidence"])
    proposal = values(prediction["teacher_top1_token_ids"])
    suppressed = set(metadata["generation_config"]["suppressed_candidate_token_ids"])
    rows = [dict(token=before_tokens[i], last_top1=int(state.last_top1[0, i]), age=int(state.precision_age[0, i]),
                 logical=begin+i, state=int(state.state[0, i]), origin=int(state.commit_origin[0, i]),
                 bits=int(bits[i]), top1=proposal[i], selected_probability=raw_probability[i],
                 action_confidence=confidence[i], raw_confidence=raw_confidence[i], suppressed=proposal[i] in suppressed,
                 refresh_required=bool(before["cache_refresh_due"][0, begin+i])) for i in range(32)]
    config = dict(mask_token=before["mask_id"], vocabulary=126464, high=code(before["tau_high"]),
        low=code(before["tau_low"]), verify=code(before["confirm_tau"]), bonus=code(before["stability_bonus"]),
        budget=encode_admission_budget(before["budget_scale"]),
        scheduled=0 if closeout_kind else int(before["schedule"][0, prediction["step_index"]]),
        remaining=1 if closeout_kind else before["steps_per_block"]-prediction["step_index"], step=prediction["step_index"],
        tail_after=before["tau_high_tail_after_step"], tail_enable=before["tau_high_tail"] is not None,
        tail=code(before["tau_high_tail"] if before["tau_high_tail"] is not None else before["tau_high"]),
        tail_bypass_all=before["tail_confirmation_policy"] == "all",
        tail_bypass_stable_only=before["tail_confirmation_policy"] == "stable_only")
    if closeout_kind:
        config.update(closeout_kind=closeout_kind, tail_enable=False, tail_bypass_all=False, tail_bypass_stable_only=False)
    decoding = before.get("decoding_mode", metadata["generation_config"].get("decoding_mode", "feature2"))
    if decoding == "fixed_k":
        config.update(transfer_only=True, scheduled=min(32, int(before["decode_k"])),
                      tail_enable=False, tail_bypass_all=False, tail_bypass_stable_only=False)
    elif decoding != "feature2":
        raise ValueError("observed control preparation requires feature2 or fixed_k decoding")
    reference = dict(schema="supra-observed-feature2-step/v1", source_index=str(index.resolve()),
        source="generation.engine.forward_end_callback",
        sequence_length=before["total_length"], block_start=begin, block_id=before["block_index"],
        suppressed_tokens=sorted(suppressed),
        capture_index=prediction["capture_index"], config=config, rows=rows, expected=expected)
    for i, row in enumerate(rows):
        row["source_a_pending"] = bool(before["source_a_handoff_pending"][0, begin+i])
    generated = metadata["generation_config"]
    future_positions = prediction.get("future_prediction_positions")
    if generated["dynamic_block_lookahead"] and before.get("next_block_state") is not None:
        if before["future_block_count"] != 1 or not generated["dynamic_block_canonical_future"]:
            raise ValueError("captured future control requires the supported one-block canonical path")
        future = DraftVerifyBlockState((1, 32), device="cpu")
        future.__dict__.update(before["next_block_state"])
        future_bits = future.row_bits(maturity_age=3, policy="original").reshape(-1).clone()
        actual_bits = dict(zip(values(prediction["input_positions"]), values(prediction["row_bits"])))
        candidates = before.get("future_candidates")
        observed_future = future_positions is not None and future_positions.numel() != 0
        if observed_future:
            if candidates is None or not torch.equal(
                    candidates["positions"].reshape(-1).int(), future_positions.reshape(-1).int()):
                raise ValueError("future candidate order differs from the observed forward")
            if values(candidates["top1"]) != values(prediction["future_top1_token_ids"]) or codes(
                    candidates["action_confidence"]) != codes(prediction["future_action_confidence"]):
                raise ValueError("future candidates differ from the original generator observation")
        elif candidates is not None and candidates["positions"].numel():
            raise ValueError("unobserved future block has nonempty captured candidates")
        future_after = checkpoints["after_postprocess"]
        final_state = DraftVerifyBlockState((1, 32), device="cpu")
        final_state.__dict__.update(future_after["next_block_state"])
        future_rows = []
        # A live but unobserved future block still supplies state/history to
        # the next joint selection. It receives no head work or state aging.
        ordered = values(candidates["positions"]) if observed_future else []
        for i in range(32):
            position = end+i
            ordinal = ordered.index(position) if position in ordered else None
            row = dict(logical=position, token=int(before["tokens"][0,position]),
                last_top1=int(future.last_top1[0,i]), age=int(future.precision_age[0,i]),
                state=int(future.state[0,i]), origin=int(future.commit_origin[0,i]),
                bits=actual_bits.get(position, int(future_bits[i])),
                refresh_required=bool(before["cache_refresh_due"][0,position]),
                source_a_pending=bool(before["source_a_handoff_pending"][0,position]),
                observed=ordinal is not None)
            history = before.get("next_last_confidence")
            row["last_action_valid"] = bool(history is not None and history[i] >= 0)
            row["last_action_confidence"] = code(float(history[i])) if row["last_action_valid"] else 0
            if ordinal is not None:
                row.update(top1=values(candidates["top1"])[ordinal],
                    raw_confidence=codes(candidates["raw_confidence"])[ordinal],
                    action_confidence=codes(candidates["action_confidence"])[ordinal],
                    selected_probability=codes(candidates["probability"])[ordinal])
            future_rows.append(row)
        future_expected = {key: values(future_after["next_block_state"][key])
            for key in ("state", "last_top1", "precision_age", "commit_origin")}
        future_expected.update(tokens=values(future_after["tokens"][:,end:end+32]),
            bits=values(final_state.row_bits(maturity_age=3, policy="original")),
            source_a_pending=values(future_after["source_a_handoff_pending"][:,end:end+32]))
        masks = {field: sum(1 << (int(p)-end) for p in trace[key]) for field, key in (
            ("selected", "next_admitted_positions"), ("direct", "next_direct_locked_positions"),
            ("confirmed", "next_confirmed_positions"), ("remasked", "next_remasked_positions"))}
        for field, origin in (("stable", 2), ("fallback", 3)):
            masks[field] = sum(1 << i for i in range(32) if masks["selected"] & (1 << i)
                and future_expected["state"][i] == 1 and future_expected["commit_origin"][i] == origin)
        masks["token_changed"] = sum(1 << i for i in range(32)
            if future_rows[i]["token"] != future_expected["tokens"][i])
        future_expected["masks"] = masks
        reference["future"] = dict(block_start=end, block_id=before["block_index"]+1,
            prediction_positions=ordered, rows=future_rows, expected=future_expected,
            scheduled=generated["dynamic_block_next_admission_budget"],
            max_handoff=generated["dynamic_block_max_handoff_verification_rows"],
            source_a_handoff=generated["dynamic_block_source_a_confirm_at_handoff"],
            canonical_direct=code(generated["dynamic_block_canonical_direct_tau"]))
    return reference


def make_psme_state_updates() -> dict:
    """Construct control inputs; derive expected with the algorithm state functions.

    A4 inherited confirmation is deliberately deferred, as in an A4 refresh.
    """
    from generation.state import select_admissions
    from copy import deepcopy
    def code(value):
        return int(torch.as_tensor(value, dtype=torch.bfloat16).view(torch.int16)) & 65535
    def values(value):
        return value.reshape(-1).tolist()
    def mask(value):
        return sum(1 << i for i, flag in enumerate(values(value)) if flag)
    records = []
    for name in ("mixed_admission", "a4_inherited_confirmation", "tail_closeout"):
        inherited = name == "a4_inherited_confirmation"
        tail = name == "tail_closeout"
        state = DraftVerifyBlockState((1, 32), device="cpu")
        state.state.fill_(2); state.precision_age.fill_(3); state.commit_origin.fill_(1)
        count = 1 if tail else 8
        state.state[:, :count] = 0; state.precision_age[:, :count] = -1
        state.commit_origin[:, :count] = 0
        if not tail:
            state.state[:, 8:12] = 1; state.precision_age[:, 8:12] = -1
            state.commit_origin[:, 8:12] = 3
            state.last_top1[:, 1] = 0
        tokens = torch.where(state.state == 0, 3, 1).long()
        proposal = torch.where(state.state == 0, 0, tokens).long()
        confidence = torch.full((1, 32), 0.25, dtype=torch.bfloat16)
        if not tail:
            confidence[:, 0] = 0.953125; confidence[:, 1] = 0.80078125
        probability = torch.ones((1, 32), dtype=torch.bfloat16)
        bits = torch.full((1, 32), 4, dtype=torch.int8) if inherited else state.row_bits(maturity_age=3, policy="original")
        before, before_tokens = deepcopy(state), tokens.clone()
        masked, locked = state.state == 0, state.state == 2
        confirmed = remasked = torch.zeros_like(masked)
        if not inherited:
            confirmation = state.confirm(tokens, proposal, probability, confirm_tau=0.75, mask_id=3)
            confirmed, remasked = confirmation.locked, confirmation.remasked
        stable = masked & (state.last_top1 == proposal)
        high = masked & (confidence >= 0.9)
        stable_low = masked & stable & (confidence >= 0.75) & ~high
        selected = select_admissions(masked, high, stable_low, stable, confidence,
            torch.tensor([1 if tail else 2]), remaining_forwards=1 if tail else 4,
            budget_scale=1., stability_bonus=0.05)
        admitted = state.admit(tokens, proposal, selected, high, stable_low)
        state.update_masked_history(proposal, masked)
        state.advance_locked_age(locked)
        closed = state.bypass_tail_confirmation("all" if tail else "none")
        expected = {field: values(getattr(state, field)) for field in
                    ("state", "last_top1", "precision_age", "commit_origin")}
        expected.update(tokens=values(tokens), bits=values(state.row_bits(maturity_age=3, policy="original")),
            masks={key: mask(value) for key, value in dict(selected=selected,
                direct=admitted.direct_locked, stable=admitted.stable_tentative,
                fallback=admitted.fallback_tentative, confirmed=confirmed, remasked=remasked,
                token_changed=tokens != before_tokens, tail_closed=closed).items()})
        rows = [dict(token=int(before_tokens[0,i]), last_top1=int(before.last_top1[0,i]),
            age=int(before.precision_age[0,i]), logical=32+i, state=int(before.state[0,i]),
            origin=int(before.commit_origin[0,i]), bits=int(bits[0,i]), top1=int(proposal[0,i]),
            selected_probability=code(probability[0,i]), action_confidence=code(confidence[0,i]),
            raw_confidence=code(confidence[0,i]), suppressed=False, refresh_required=False,
            prediction_flag=bool(before.state[0,i] != 2), source_a_pending=False) for i in range(32)]
        records.append(dict(schema="supra-feature2-step/v1", name=name,
            source="generation.state.select_admissions/Feature2BlockState",
            sequence_length=64, block_start=32, block_id=0, suppressed_tokens=[],
            config=dict(mask_token=3, vocabulary=4, high=code(.9), low=code(.75), verify=code(.75),
                bonus=code(.05), budget=code(1.), scheduled=1 if tail else 2,
                remaining=1 if tail else 4, step=0, tail_after=8, tail_enable=False,
                tail=code(.75), tail_bypass_all=tail, tail_bypass_stable_only=False),
            rows=rows, expected=expected))
    return dict(schema="supra-psme-state-updates/v1", records=records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--library", type=Path, help="compare existing reference_data with this C11 library")
    parser.add_argument("--kind", choices=("psme-state-updates", "uaps-source-b-selection", "atse-saved-dependencies", "atse-in-block-boundaries", "atse-uaps-in-block", "uaps-token-selection", "psme-uaps-context-precision", "atse-cross-block-token-selection", "atse-previous-shortlist", "atse-cross-block-pending-refresh", "atse-attention-dependencies", "atse-pre-p8-dependencies", "atse-cross-block-scout", "uaps-future-token-admission", "uaps-priority", "psme-context-precision", "feature12-context-precision", "feature12-boundary-precision", "feature12-block-initialization", "atse-uaps-limits", "psme-fixed-k-decoding", "uaps-live-selection", "uaps-live-state", "atse-live-block-initialization", "captured-feature2", "captured-feature1", "atse-captured-block-initialization", "atse-captured-block-boundary", "atse-generated-block-boundary"), default="atse-saved-dependencies")
    parser.add_argument("--input", type=Path, help="complete independent state for uaps-live-selection verification")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--pending-confidence-mode", default="all_changes",
                        choices=("all_changes", "stable_unmask", "remask_only", "all"),
                        help="cross-block pending reference mode; all emits three independent sequences")
    parser.add_argument("--probability-boundaries", action="store_true",
                        help="use explicit scout probability inputs at UINT8 half-integer boundaries")
    parser.add_argument("--reduction-order-boundary", action="store_true",
                        help="use explicit scout probabilities distinguishing FP32 reduction orders")
    parser.add_argument("--case", action="append", help="generate only these uaps-token-selection cases")
    parser.add_argument("--source-a-handoff", choices=("off", "on", "both"),
                        help="uaps-future-token-admission reference with current supported tri-block controls")
    args = parser.parse_args()
    target = args.index.resolve()
    if args.kind in ("captured-feature2", "captured-feature1", "atse-captured-block-initialization", "atse-captured-block-boundary", "atse-generated-block-boundary"):
        if args.input is None or args.library is not None or target.exists():
            parser.error("captured reference requires --input and a fresh --index in the allowed output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        generate = {"captured-feature2": captured_feature2_reference, "captured-feature1": captured_feature1_reference,
                    "atse-captured-block-initialization": captured_block_initialization_reference, "atse-captured-block-boundary": captured_boundary_reference,
                    "atse-generated-block-boundary": generated_boundary_reference}[args.kind]
        reference = generate(args.input.resolve(strict=True))
        reference["source_index"] = os.path.relpath(args.input.resolve(strict=True), target.parent)
        reference["source_index_base"] = "reference_directory"
        target.write_text(json.dumps(reference, indent=2) + "\n")
        return
    if args.kind in ("uaps-live-selection", "atse-live-block-initialization"):
        if args.input is None or args.library is not None:
            parser.error("live references require --input and no C library")
        hardware_source = ALIGNMENT_ROOT.parent / "hardware"
        sys.path.insert(0, str(hardware_source / "scripts"))
        from artifact_paths import artifact_root
        hardware_root = artifact_root()
        if hardware_root not in target.parents or target.exists():
            parser.error("live reference output must be a new hardware artifact file")
        torch.set_num_threads(2)
        select = select_live_joint_reference if args.kind == "uaps-live-selection" else select_boundary_reference
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(select(json.loads(args.input.read_text()))) + "\n")
        return
    if args.input is not None:
        parser.error("--input is only for live references")
    if args.source_a_handoff and (args.kind != "uaps-future-token-admission" or args.library is not None):
        parser.error("--source-a-handoff applies only to uaps-future-token-admission generation")
    if args.pending_confidence_mode != "all_changes" and (args.kind != "atse-cross-block-pending-refresh" or args.library is not None):
        parser.error("nondefault pending confidence modes apply only to atse-cross-block-pending-refresh generation")
    if (args.probability_boundaries or args.reduction_order_boundary) and (
            args.kind != "atse-cross-block-scout" or args.library is not None):
        parser.error("probability boundary options apply only to atse-cross-block-scout generation")
    if args.probability_boundaries and args.reduction_order_boundary:
        parser.error("choose one probability boundary input")
    if args.device != "cpu" and args.kind not in ("atse-attention-dependencies", "atse-pre-p8-dependencies", "atse-cross-block-scout"):
        parser.error("--device applies only to Attention relation/scout generation")
    if args.case and (args.kind != "uaps-token-selection" or args.library is not None):
        parser.error("--case applies only to uaps-token-selection generation")
    if args.library is not None:
        comparisons = {"atse-saved-dependencies": compare_saved_dependency_reference_data,
                       "atse-cross-block-token-selection": compare_boundary_selection_reference_data,
                       "uaps-token-selection": compare_joint_selection_reference_data,
                       "psme-uaps-context-precision": compare_joint_selection_reference_data,
                       "atse-cross-block-pending-refresh": compare_cross_block_reference_data,
                       "atse-attention-dependencies": compare_attention_relation_reference_data,
                       "atse-pre-p8-dependencies": compare_pre_p8_relation_reference_data,
                       "psme-context-precision": compare_packed_precision_reference_data,
                       "atse-cross-block-scout": compare_boundary_scout_reference_data,
                       "atse-uaps-limits": compare_atse_uaps_limits_reference_data}
        if args.kind not in comparisons:
            parser.error("this reference_data has no C11 comparison entry yet")
        result = comparisons[args.kind](json.loads(target.read_text()), args.library)
        print(json.dumps(result))
        raise SystemExit(0 if result["match"] else 1)
    if target.exists():
        parser.error("--index must be a new file in the allowed output directory")
    torch.set_num_threads(2)
    if args.kind == "psme-state-updates":
        reference_data = make_psme_state_updates()
    elif args.kind == "uaps-source-b-selection":
        reference_data = make_source_b_selection_reference_data()
    elif args.kind == "uaps-live-state":
        reference_data = make_joint_state_reference_data()
    elif args.kind == "psme-fixed-k-decoding":
        reference_data = make_psme_fixed_k_reference_data()
    elif args.kind == "atse-uaps-limits":
        reference_data = make_atse_uaps_limits_reference_data()
    elif args.kind == "atse-previous-shortlist":
        reference_data = make_shortlist_reference_data()
    elif args.kind == "atse-uaps-in-block":
        reference_data = make_regular_joint_reference_data()
    elif args.kind == "psme-uaps-context-precision":
        reference_data = make_precision_joint_reference_data()
    elif args.kind == "uaps-priority":
        generated = make_future_admission_reference_data()
        reference_data = dict(schema="supra-uaps-priority/v1",
            target_joint_rows=48, max_next_rows=32,
            records=generated["joint_priorities"])
    elif args.kind == "uaps-future-token-admission":
        if args.source_a_handoff == "both":
            reference_data = dict(schema="supra-source-a-handoff/v1",
                variants=[make_future_admission_reference_data(source_a_handoff=enabled)
                          for enabled in (False, True)])
        else:
            reference_data = make_future_admission_reference_data(
                source_a_handoff=None if args.source_a_handoff is None else args.source_a_handoff == "on")
    elif args.kind == "feature12-block-initialization":
        reference_data = make_shortlist_reference_data(block_initialization_context_quota=0)
        reference_data["records"] = [record for record in reference_data["records"]
            if record["name"] in ("block_initialization_history_tail_s2048", "block_initialization_position_ties_s2048")]
    elif args.kind == "feature12-boundary-precision":
        reference_data = make_feature12_boundary_precision_reference_data()
    elif args.kind == "feature12-context-precision":
        reference_data = make_feature12_precision_reference_data()
    elif args.kind == "psme-context-precision":
        reference_data = dict(schema="supra-psme-context-precision/v1",
            records=[])
        for count in (10, 12):
            generated = make_future_admission_reference_data(full_current_context_rows=8, context_a8_rows=count)
            for record in generated["packed_precision_records"]:
                reference_data["records"].append(record)
    elif args.kind == "atse-in-block-boundaries":
        reference_data = make_regular_boundary_reference_data()
    elif args.kind == "atse-saved-dependencies":
        reference_data = make_saved_dependency_reference_data()
    elif args.kind == "uaps-token-selection":
        reference_data = make_joint_selection_reference_data(set(args.case) if args.case else None)
    elif args.kind == "atse-cross-block-pending-refresh":
        selected_mode = "all_changes" if args.pending_confidence_mode == "all" else args.pending_confidence_mode
        reference_data = make_cross_block_reference_data(mode=selected_mode)
        if args.pending_confidence_mode == "all":
            reference_data["variants"] = [make_cross_block_reference_data(mode=mode)
                                          for mode in ("remask_only", "stable_unmask")]
    elif args.kind == "atse-attention-dependencies":
        reference_data = make_attention_relation_reference_data(device=args.device)
    elif args.kind == "atse-pre-p8-dependencies":
        reference_data = make_pre_p8_relation_reference_data(device=args.device)
    elif args.kind == "atse-cross-block-scout":
        reference_data = make_boundary_scout_reference_data(device=args.device,
            probability_boundaries=args.probability_boundaries,
            reduction_order_boundary=args.reduction_order_boundary)
    else:
        reference_data = make_boundary_selection_reference_data()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(reference_data, indent=2) + "\n")
    count = len(reference_data.get("records", [])) + sum(len(item["records"]) for item in reference_data.get("variants", []))
    print(f"REFERENCE_DATA_GENERATED records={count} index={target}")


if __name__ == "__main__":
    main()
