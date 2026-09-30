"""Runtime control for changed-token-conditioned packed row refresh."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from numerics.bf16 import bf16, bf16_add, bf16_mul


def _bf16_state(values: torch.Tensor | float) -> torch.Tensor:
    """Round one deployment-visible value and store its BF16 bits."""
    return bf16(values).to(torch.bfloat16)


def _bf16_state_add(
    lhs: torch.Tensor | float, rhs: torch.Tensor | float
) -> torch.Tensor:
    reference = lhs if isinstance(lhs, torch.Tensor) else rhs
    device = reference.device if isinstance(reference, torch.Tensor) else None
    lhs_tensor = (
        lhs
        if isinstance(lhs, torch.Tensor)
        else torch.tensor(lhs, dtype=torch.float32, device=device)
    )
    rhs_tensor = (
        rhs
        if isinstance(rhs, torch.Tensor)
        else torch.tensor(rhs, dtype=torch.float32, device=device)
    )
    return bf16_add(lhs_tensor, rhs_tensor).to(torch.bfloat16)


def _bf16_state_mul(
    lhs: torch.Tensor | float, rhs: torch.Tensor | float
) -> torch.Tensor:
    reference = lhs if isinstance(lhs, torch.Tensor) else rhs
    device = reference.device if isinstance(reference, torch.Tensor) else None
    lhs_tensor = (
        lhs
        if isinstance(lhs, torch.Tensor)
        else torch.tensor(lhs, dtype=torch.float32, device=device)
    )
    rhs_tensor = (
        rhs
        if isinstance(rhs, torch.Tensor)
        else torch.tensor(rhs, dtype=torch.float32, device=device)
    )
    return bf16_mul(lhs_tensor, rhs_tensor).to(torch.bfloat16)


@dataclass(frozen=True)
class TriBlockAttentionObservation:
    dependency_score: torch.Tensor
    new_invalidation: torch.Tensor
    refreshed: torch.Tensor
    next_refresh: torch.Tensor
    mandatory: torch.Tensor
    optional_selected: torch.Tensor
    dependency_rows_updated: int


class CrossBlockPrefixAttentionState:
    """Carry changed-token risk for prompt and completed generation rows."""

    def __init__(
        self,
        *,
        total_length: int,
        block_length: int,
        boundary_target_rows: int,
        device: torch.device,
        pending_confidence_mode: str = "remask_only",
        pending_relation_mode: str = "direct",
        track_future_actual_remask: bool = False,
    ) -> None:
        if total_length <= block_length:
            raise ValueError("cross-block prefix state requires prompt rows")
        if not block_length <= boundary_target_rows <= total_length:
            raise ValueError("boundary_target_rows must cover one generation block")
        if pending_confidence_mode not in {
            "remask_only",
            "all_changes",
            "stable_unmask",
        }:
            raise ValueError(
                "pending_confidence_mode must be remask_only, all_changes, or stable_unmask"
            )
        if pending_relation_mode != "direct":
            raise ValueError("the deployed dependency rule is direct")
        self.total_length = int(total_length)
        self.block_length = int(block_length)
        self.boundary_target_rows = int(boundary_target_rows)
        self.pending_confidence_mode = str(pending_confidence_mode)
        self.pending_relation_mode = str(pending_relation_mode)
        self.track_future_actual_remask = bool(track_future_actual_remask)
        self.pending = torch.zeros(total_length, dtype=torch.bfloat16, device=device)
        self.actual_remask_pending = torch.zeros(
            total_length, dtype=torch.bfloat16, device=device
        )
        self.actual_remask_epoch_pending = torch.zeros(
            total_length, dtype=torch.bfloat16, device=device
        )
        self.future_actual_remask_pending = (
            torch.zeros(total_length, dtype=torch.bfloat16, device=device)
            if self.track_future_actual_remask
            else None
        )
        self.future_pending = (
            torch.zeros(total_length, dtype=torch.bfloat16, device=device)
            if self.track_future_actual_remask
            else None
        )
        self.relation = torch.zeros(
            (total_length, block_length), dtype=torch.bfloat16, device=device
        )
        self.block_start = -1
        self.block_end = -1
        self.prompt_end = -1
        self.selected_regular = torch.empty(0, dtype=torch.long, device=device)

    def start_initial_block(self, *, block_start: int, block_end: int) -> None:
        if self.prompt_end >= 0 and self.prompt_end != block_start:
            raise ValueError("initial block start must preserve the prompt boundary")
        self.prompt_end = int(block_start)
        self.selected_regular = torch.empty(
            0, dtype=torch.long, device=self.pending.device
        )
        self._set_block(block_start=block_start, block_end=block_end)

    def score_positions(
        self, positions: torch.Tensor, *, pending_source: str = "all"
    ) -> torch.Tensor:
        if positions.ndim != 1 or positions.dtype != torch.long:
            raise ValueError("positions must be a rank-1 long tensor")
        if positions.device != self.pending.device:
            raise ValueError("positions must use the cross-block state device")
        if positions.numel() and bool(
            ((positions < 0) | (positions >= self.total_length)).any()
        ):
            raise ValueError("positions are outside the cross-block score table")
        if pending_source not in {"all", "actual_remask", "actual_remask_consistent"}:
            raise ValueError(
                "pending_source must be all, actual_remask, or actual_remask_consistent"
            )
        pending = (
            _bf16_state_mul(
                self.actual_remask_pending, self.actual_remask_epoch_pending
            )
            if pending_source == "actual_remask_consistent"
            else self.actual_remask_pending
            if pending_source == "actual_remask"
            else self.pending
        )
        return pending.index_select(0, positions)

    def _set_block(self, *, block_start: int, block_end: int) -> None:
        if block_end - block_start != self.block_length:
            raise ValueError("cross-block prefix state requires one full block")
        if not 0 <= block_start < block_end <= self.total_length:
            raise ValueError("invalid cross-block prefix range")
        self.block_start = int(block_start)
        self.block_end = int(block_end)
        if self.relation is not None:
            self.relation.zero_()

    def advance_block(self, *, block_start: int, block_end: int) -> None:
        """Retain unresolved refresh scores while changing the tracked key block."""
        if self.block_start < 0 or block_start < self.block_end:
            raise ValueError("dependency block must advance after initialization")
        self._set_block(block_start=block_start, block_end=block_end)
        if self.future_pending is not None:
            self.pending[:block_end] = torch.maximum(
                self.pending[:block_end], self.future_pending[:block_end]
            )
            self.future_pending[:block_end] = 0
        if self.future_actual_remask_pending is not None:
            self.actual_remask_pending[:block_end] = torch.maximum(
                self.actual_remask_pending[:block_end],
                self.future_actual_remask_pending[:block_end],
            )
            self.future_actual_remask_pending[:block_end] = 0

    def observe(
        self,
        profile: dict[str, torch.Tensor],
        *,
        changed_global: torch.Tensor,
        changed_confidence_global: torch.Tensor,
        changed_remask_global: torch.Tensor,
    ) -> None:
        if self.selected_regular.numel():
            self.pending.index_fill_(0, self.selected_regular, 0.0)
            self.actual_remask_pending.index_fill_(0, self.selected_regular, 0.0)
            self.actual_remask_epoch_pending.index_fill_(0, self.selected_regular, 0.0)
            if self.future_pending is not None:
                self.future_pending.index_fill_(0, self.selected_regular, 0.0)
            if self.future_actual_remask_pending is not None:
                self.future_actual_remask_pending.index_fill_(
                    0, self.selected_regular, 0.0
                )
            self.selected_regular = torch.empty(
                0, dtype=torch.long, device=self.pending.device
            )
        positions = profile.get("prefix_dependency_query_positions")
        if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
            raise ValueError("prefix dependency positions must be rank 1")
        positions = positions.to(device=self.pending.device, dtype=torch.long)
        query_limit = (
            self.total_length if self.track_future_actual_remask else self.block_end
        )
        if positions.numel() and bool(
            ((positions < 0) | (positions >= query_limit)).any()
        ):
            raise ValueError("prefix dependency includes a future query row")
        if positions.unique().numel() != positions.numel():
            raise ValueError("prefix dependency query positions must be unique")
        values = profile.get("prefix_dependency_mean")
        if not isinstance(values, torch.Tensor) or values.shape != (
            1,
            positions.numel(),
            self.block_length,
        ):
            raise ValueError("prefix dependency values have the wrong shape")
        relation = (
            torch.nan_to_num(
                values[0].to(device=self.pending.device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            .clamp_(0.0, 1.0)
            .to(torch.bfloat16)
        )
        if self.relation is None:
            raise RuntimeError("current-block relation table is unavailable")
        if positions.numel():
            self.relation.index_copy_(0, positions, relation)
        self.accumulate_changes(
            profile=profile,
            changed_global=changed_global,
            changed_confidence_global=changed_confidence_global,
            changed_remask_global=changed_remask_global,
        )

    def accumulate_changes(
        self,
        *,
        profile: dict[str, torch.Tensor] | None,
        changed_global: torch.Tensor,
        changed_confidence_global: torch.Tensor,
        changed_remask_global: torch.Tensor,
    ) -> None:
        """Accumulate invalidation without consuming rows or updating relations."""
        for tensor, name, dtype in (
            (changed_global, "changed_global", torch.bool),
            (changed_remask_global, "changed_remask_global", torch.bool),
        ):
            if tensor.shape != (1, self.total_length) or tensor.dtype != dtype:
                raise ValueError(f"{name} has the wrong shape or dtype")
        if bool((changed_remask_global & ~changed_global).any()):
            raise ValueError("changed remask must be a subset of changed rows")
        if (
            changed_confidence_global.shape != (1, self.total_length)
            or not changed_confidence_global.dtype.is_floating_point
        ):
            raise ValueError("changed_confidence_global has the wrong shape or dtype")
        changed = changed_global[0, self.block_start : self.block_end]
        if not bool(changed.any()):
            return
        remasked = changed_remask_global[0, self.block_start : self.block_end]
        remask_in_changed = remasked[changed]
        if self.relation is None:
            raise RuntimeError("current-block relation table is unavailable")
        direct = self.relation[:, changed]
        ordinary = direct[:, ~remask_in_changed]
        if ordinary.shape[1] and self.pending_confidence_mode in {
            "all_changes",
            "stable_unmask",
        }:
            ordinary_confidence = _bf16_state(
                changed_confidence_global[0, self.block_start : self.block_end][
                    changed & ~remasked
                ].to(device=self.pending.device, dtype=torch.float32)
            )
            ordinary_confidence = torch.nan_to_num(
                ordinary_confidence, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
            if self.pending_confidence_mode == "all_changes":
                ordinary_factor = _bf16_state_add(2.0, -ordinary_confidence)
            else:
                ordinary_factor = ordinary_confidence
            ordinary = _bf16_state_mul(ordinary, ordinary_factor.unsqueeze(0))
            ordinary.clamp_(0.0, 1.0)
        risk = (
            ordinary.amax(dim=1)
            if ordinary.shape[1]
            else torch.zeros_like(self.pending)
        )
        actual_remask_risk = torch.zeros_like(self.pending)
        remask_risk = direct[:, remask_in_changed].clone()
        if remask_risk.shape[1]:
            confidence = _bf16_state(
                changed_confidence_global[0, self.block_start : self.block_end][
                    remasked
                ].to(device=self.pending.device, dtype=torch.float32)
            )
            severity = _bf16_state_add(
                1.0,
                -torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp_(
                    0.0, 1.0
                ),
            )
            remask_risk = _bf16_state_mul(
                remask_risk, _bf16_state_add(1.0, severity).unsqueeze(0)
            ).clamp_(0.0, 1.0)
            union = remask_risk[:, 0].clone()
            for item in remask_risk[:, 1:].unbind(dim=1):
                union = _bf16_state_add(
                    union, _bf16_state_add(item, -_bf16_state_mul(union, item))
                )
            actual_remask_risk = union
            risk = torch.maximum(risk, union)
        if self.future_actual_remask_pending is not None:
            future_all_risk = risk.clone()
            future_all_risk[: self.block_end] = 0.0
            if self.future_pending is None:
                raise RuntimeError("future all-change pending is unavailable")
            self.future_pending = torch.maximum(self.future_pending, future_all_risk)
            future_risk = actual_remask_risk.clone()
            future_risk[: self.block_end] = 0.0
            self.future_actual_remask_pending = torch.maximum(
                self.future_actual_remask_pending, future_risk
            )
        risk[self.block_end :] = 0.0
        actual_remask_risk[self.block_end :] = 0.0
        self.pending = torch.maximum(self.pending, risk)
        self.actual_remask_pending = torch.maximum(
            self.actual_remask_pending, actual_remask_risk
        )
        self.actual_remask_epoch_pending = torch.maximum(
            self.actual_remask_epoch_pending, actual_remask_risk
        )

class TriBlockAttentionState:
    """Select rows whose cached state depends on changed token positions."""

    def __init__(
        self,
        *,
        region_start: int,
        region_end: int,
        total_length: int,
        initial_attention_profile: dict[str, torch.Tensor],
        target_active_rows: float = 39.5,
        initial_burst_rows: float = 0.0,
    ) -> None:
        if not 0 <= region_start < region_end <= total_length:
            raise ValueError("invalid tri-block region")
        self.region_start = int(region_start)
        self.region_end = int(region_end)
        self.total_length = int(total_length)
        rows = self.region_end - self.region_start
        if float(target_active_rows) < 1.0:
            raise ValueError("target_active_rows must be positive")
        if float(initial_burst_rows) < 0.0:
            raise ValueError("initial_burst_rows must be nonnegative")
        self.target_active_rows = min(float(target_active_rows), float(rows))
        self.initial_burst_rows = float(initial_burst_rows)
        device = self._profile_device(initial_attention_profile)
        self.dependency = torch.zeros((rows, rows), dtype=torch.bfloat16, device=device)
        self.dependency_score = torch.zeros(rows, dtype=torch.bfloat16, device=device)
        self.new_invalidation = torch.zeros_like(self.dependency_score)
        self.pending = torch.zeros_like(self.dependency_score)
        self.mandatory = torch.zeros(rows, dtype=torch.bool, device=device)
        self.optional_selected = torch.zeros(rows, dtype=torch.bool, device=device)
        self.refresh = torch.zeros(rows, dtype=torch.bool, device=device)
        self.dependency_entries_read = 0
        self.direct_dependency_entries_read = 0
        self.change_confidence_entries_read = 0
        self.change_risk_subtractions = 0
        self.change_risk_multiplications = 0
        self.remask_status_entries_read = 0
        self.causal_union_additions = 0
        self.joint_relation_multiplications = 0
        self.selector_candidate_rows = 0
        self.top_budget_comparisons = 0
        self.allowed_active_rows = 0
        self.regular_steps = 0
        self.cumulative_active_rows = 0
        self.dependency_rows_updated = self._update_dependency(initial_attention_profile)

    @staticmethod
    def _profile_device(profile: dict[str, torch.Tensor]) -> torch.device:
        value = profile.get("dependency_mean")
        if not isinstance(value, torch.Tensor):
            raise ValueError("attention profile must contain dependency_mean")
        return value.device

    def _update_dependency(self, profile: dict[str, torch.Tensor]) -> int:
        rows = self.region_end - self.region_start
        positions = profile.get("query_positions")
        value = profile.get("dependency_mean")
        if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
            raise ValueError("attention profile query_positions must be rank 1")
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError("attention profile dependency_mean must be rank 3")
        if value.shape != (1, positions.numel(), rows):
            raise ValueError(
                "attention profile dependency_mean must have shape [1, query_rows, region_rows]"
            )
        positions = positions.detach().to(device=value.device, dtype=torch.long)
        if positions.numel() and bool(
            ((positions < self.region_start) | (positions >= self.region_end)).any()
        ):
            raise ValueError("attention profile query positions are outside the region")
        if positions.unique().numel() != positions.numel():
            raise ValueError("attention profile query positions must be unique")
        local = positions - self.region_start
        dependency = value[0].detach().to(torch.float32).clone()
        dependency = torch.nan_to_num(dependency, nan=0.0, posinf=0.0, neginf=0.0)
        dependency = dependency.clamp_(0.0, 1.0).to(torch.bfloat16)
        if local.numel():
            self.dependency.index_copy_(0, local, dependency)
        return int(local.numel())

    def _accumulate_invalidation(
        self,
        changed_global: torch.Tensor,
        changed_confidence_global: torch.Tensor | None = None,
        changed_remask_global: torch.Tensor | None = None,
    ) -> None:
        if (
            changed_global.shape != (1, self.total_length)
            or changed_global.dtype != torch.bool
        ):
            raise ValueError("changed_global must be a full-sequence bool mask")
        if changed_confidence_global is not None:
            if changed_confidence_global.shape != changed_global.shape:
                raise ValueError("changed confidence must match changed_global")
            if not changed_confidence_global.dtype.is_floating_point:
                raise ValueError("changed confidence must use a floating-point dtype")
        if changed_remask_global is not None:
            if (
                changed_remask_global.shape != changed_global.shape
                or changed_remask_global.dtype != torch.bool
            ):
                raise ValueError(
                    "changed remask must match changed_global as a bool mask"
                )
            if bool((changed_remask_global & ~changed_global).any()):
                raise ValueError("changed remask must be a subset of changed_global")
        changed_local = changed_global[0, self.region_start : self.region_end]
        remasked_local = (
            torch.zeros_like(changed_local)
            if changed_remask_global is None
            else changed_remask_global[0, self.region_start : self.region_end]
        )
        rows = int(self.dependency.shape[0])
        changed_rows = int(changed_local.sum().item())
        remasked_rows = int(remasked_local.sum().item())
        self.direct_dependency_entries_read = rows * changed_rows
        self.change_confidence_entries_read = 0
        self.change_risk_subtractions = 0
        self.change_risk_multiplications = 0
        self.remask_status_entries_read = (
            changed_rows if changed_remask_global is not None else 0
        )
        self.causal_union_additions = 0
        self.joint_relation_multiplications = 0
        self.dependency_entries_read = self.direct_dependency_entries_read
        if bool(changed_local.any()):
            dependency = self.dependency
            direct = dependency[:, changed_local]
            remask_in_changed = remasked_local[changed_local]
            ordinary = direct[:, ~remask_in_changed]
            ordinary_max = (
                ordinary.amax(dim=1)
                if ordinary.shape[1]
                else torch.zeros(rows, dtype=torch.bfloat16, device=dependency.device)
            )
            remask_risk = direct[:, remask_in_changed].clone()
            if remasked_rows:
                if changed_confidence_global is None:
                    raise ValueError("actual remask requires changed confidence")
                confidence = _bf16_state(
                    changed_confidence_global[0, self.region_start : self.region_end][
                        remasked_local
                    ].to(device=dependency.device, dtype=torch.float32)
                )
                confidence = torch.nan_to_num(
                    confidence, nan=0.0, posinf=1.0, neginf=0.0
                ).clamp_(0.0, 1.0)
                severity = _bf16_state_add(1.0, -confidence)
                remask_risk = _bf16_state_mul(
                    remask_risk, _bf16_state_add(1.0, severity).unsqueeze(0)
                )
                self.change_confidence_entries_read = remasked_rows
                self.change_risk_subtractions = remasked_rows
                self.change_risk_multiplications = rows * remasked_rows
                remask_risk.clamp_(0.0, 1.0)
                remask_union = remask_risk[:, 0].clone()
                for risk in remask_risk[:, 1:].unbind(dim=1):
                    remask_union = _bf16_state_add(
                        remask_union,
                        _bf16_state_add(risk, -_bf16_state_mul(remask_union, risk)),
                    )
                self.new_invalidation = torch.maximum(ordinary_max, remask_union)
            else:
                self.new_invalidation = ordinary_max
            union_steps = max(remasked_rows - 1, 0)
            self.causal_union_additions = 2 * rows * union_steps
            self.joint_relation_multiplications = rows * union_steps
        else:
            self.new_invalidation.zero_()
        self.pending = torch.maximum(self.pending, self.new_invalidation)
        self.dependency_score = self.pending.detach().clone()

    def _select_next(
        self, *, predicted: torch.Tensor, changed_global: torch.Tensor
    ) -> None:
        rows = self.region_end - self.region_start
        if predicted.shape != (rows,) or predicted.dtype != torch.bool:
            raise ValueError("predicted must be a tri-block-local bool vector")
        if (
            changed_global.shape != (1, self.total_length)
            or changed_global.dtype != torch.bool
        ):
            raise ValueError("changed_global must be a full-sequence bool mask")
        changed_local = changed_global[0, self.region_start : self.region_end]
        mandatory = predicted | changed_local
        refresh = mandatory.clone()
        optional_selected = torch.zeros_like(refresh)
        self.selector_candidate_rows = int((~mandatory).sum().item())
        cumulative_limit = (
            self.target_active_rows * (self.regular_steps + 1) + self.initial_burst_rows
        )
        allowed_active = max(
            int(mandatory.sum().item()),
            int(cumulative_limit - self.cumulative_active_rows),
        )
        allowed_active = min(rows, allowed_active)
        self.allowed_active_rows = allowed_active
        top_budget_comparisons = 0
        for index in torch.argsort(
            self.dependency_score, descending=True, stable=True
        ).tolist():
            if bool(refresh[index]):
                continue
            if int(refresh.sum().item()) >= allowed_active:
                break
            top_budget_comparisons += 1
            if float(self.dependency_score[index].item()) <= 0.0:
                break
            refresh[index] = True
            optional_selected[index] = True
        self.mandatory = mandatory
        self.optional_selected = optional_selected
        self.top_budget_comparisons = top_budget_comparisons
        self.regular_steps += 1
        self.refresh = refresh
        self.cumulative_active_rows += int(refresh.sum().item())

    def account_initial_regular_forward(self, active_rows: int) -> None:
        if self.regular_steps or self.cumulative_active_rows:
            raise RuntimeError("initial regular forward was already accounted")
        if not 1 <= active_rows <= self.region_end - self.region_start:
            raise ValueError("initial regular active rows are outside the region")
        self.regular_steps = 1
        self.cumulative_active_rows = int(active_rows)

    def plan_initial(
        self,
        *,
        predicted: torch.Tensor,
        changed_global: torch.Tensor,
        changed_confidence_global: torch.Tensor | None = None,
        changed_remask_global: torch.Tensor | None = None,
    ) -> None:
        self._accumulate_invalidation(
            changed_global, changed_confidence_global, changed_remask_global
        )
        self._select_next(predicted=predicted, changed_global=changed_global)

    def refresh_local_indices(self) -> torch.Tensor:
        return torch.nonzero(self.refresh, as_tuple=False).flatten().to(torch.long)

    def observe(
        self,
        profile: dict[str, torch.Tensor],
        *,
        predicted: torch.Tensor,
        changed_global: torch.Tensor,
        changed_confidence_global: torch.Tensor | None = None,
        changed_remask_global: torch.Tensor | None = None,
    ) -> TriBlockAttentionObservation:
        rows = self.region_end - self.region_start
        if predicted.shape != (rows,) or predicted.dtype != torch.bool:
            raise ValueError("predicted must be a tri-block-local bool vector")
        if (
            changed_global.shape != (1, self.total_length)
            or changed_global.dtype != torch.bool
        ):
            raise ValueError("changed_global must be a full-sequence bool mask")
        refreshed = self.refresh.detach().clone()
        self.pending[refreshed] = 0
        self.dependency_rows_updated = self._update_dependency(profile)
        self._accumulate_invalidation(
            changed_global, changed_confidence_global, changed_remask_global
        )
        self._select_next(predicted=predicted, changed_global=changed_global)
        return TriBlockAttentionObservation(
            dependency_score=self.dependency_score.detach().clone(),
            new_invalidation=self.new_invalidation.detach().clone(),
            refreshed=refreshed,
            next_refresh=self.refresh.detach().clone(),
            mandatory=self.mandatory.detach().clone(),
            optional_selected=self.optional_selected.detach().clone(),
            dependency_rows_updated=self.dependency_rows_updated,
        )


def _unwrap_runtime_owner(model: Any) -> Any:
    owner = model
    for _ in range(4):
        if hasattr(owner, "_LLaDAModel__cache"):
            return owner
        wrapped = getattr(owner, "module", None)
        if wrapped is None:
            wrapped = getattr(owner, "model", None)
        if wrapped is None or wrapped is owner:
            break
        owner = wrapped
    raise ValueError("model does not expose the LLaDA runtime cache")


def _runtime_cache(model: Any) -> dict[str, Any]:
    owner = _unwrap_runtime_owner(model)
    cache = getattr(owner, "_LLaDAModel__cache")
    if not isinstance(cache, dict):
        raise ValueError("LLaDA runtime cache must be a dictionary")
    return cache


def begin_boundary_layer0_scout(
    model: Any,
    *,
    current_positions: torch.Tensor,
    total_length: int,
    transition_positions: torch.Tensor | None = None,
) -> None:
    """Capture one Layer0 current-query to all-key P8 score vector."""
    cache = _runtime_cache(model)
    if "boundary_layer0_scout" in cache:
        raise RuntimeError("a boundary Layer0 scout is already active")
    owner = _unwrap_runtime_owner(model)
    if (
        current_positions.ndim != 1
        or current_positions.dtype != torch.long
        or current_positions.device != owner.device
        or (current_positions.numel() == 0)
    ):
        raise ValueError("boundary Layer0 current positions are invalid")
    if (
        total_length <= int(current_positions.max().item())
        or current_positions.unique().numel() != current_positions.numel()
    ):
        raise ValueError("boundary Layer0 current positions are out of range")
    cache["boundary_layer0_scout"] = {
        "current_positions": current_positions.detach().clone(),
        "total_length": int(total_length),
        "transition_positions": None
        if transition_positions is None
        else transition_positions.detach().clone(),
    }


def read_boundary_layer0_scout(model: Any) -> tuple[torch.Tensor, int]:
    cache = _runtime_cache(model)
    state = cache.get("boundary_layer0_scout")
    if not isinstance(state, dict):
        raise RuntimeError("boundary Layer0 scout is not active")
    score = state.get("score_q8")
    total_length = int(state.get("total_length", 0))
    if (
        not isinstance(score, torch.Tensor)
        or score.shape != (total_length,)
        or score.dtype != torch.uint8
    ):
        raise RuntimeError("boundary Layer0 scout did not produce a P8 score")
    return (score.detach().clone(), int(state.get("probability_entries", 0)))


def end_boundary_layer0_scout(model: Any) -> None:
    _runtime_cache(model).pop("boundary_layer0_scout", None)


def configure_tri_block_attention_monitor(
    model: Any, *, layer_mode: str = "last"
) -> None:
    if layer_mode not in {"first", "last", "all"}:
        raise ValueError("attention dependency layer_mode must be first, last, or all")
    outer_config = getattr(model, "config", None)
    owner = _unwrap_runtime_owner(model)
    inner_config = getattr(owner, "config", None)
    if outer_config is None or inner_config is None:
        raise ValueError("model does not expose a config")
    for config in {
        id(outer_config): outer_config,
        id(inner_config): inner_config,
    }.values():
        config.attn_monitor_layer = (
            0 if layer_mode == "first" else int(config.n_layers) - 1
        )
        config.attn_monitor_all_layers = layer_mode == "all"
        config.attn_monitor_layer_mode = layer_mode
        config.noncausal_cached_attention = True


def begin_tri_block_attention_monitor(
    model: Any,
    *,
    region_start: int,
    region_end: int,
    prediction_start: int,
    prediction_end: int,
    prompt_end: int | None = None,
    track_prefix_dependency: bool = False,
    track_prefix_reverse_dependency: bool = False,
    track_prefix_future_dependency: bool = False,
    track_future_query_rows: bool = False,
    trace_attention_relation_diagnostics: bool = False,
    capture_qk_probe: bool = False,
    capture_deployment_p8_relation: bool = False,
    segment_key_groups: Sequence[Sequence[int]] = (),
    capture_segment_head_mass: bool = False,
) -> None:
    cache = _runtime_cache(model)
    _clear_monitor_step(cache)
    owner = _unwrap_runtime_owner(model)
    cache["attn_monitor_query_range"] = torch.tensor(
        [int(region_start), int(region_end)], device=owner.device, dtype=torch.long
    )
    cache["attn_monitor_prediction_range"] = torch.tensor(
        [int(prediction_start), int(prediction_end)],
        device=owner.device,
        dtype=torch.long,
    )
    if prompt_end is None:
        cache.pop("attn_monitor_prompt_end", None)
    else:
        cache["attn_monitor_prompt_end"] = int(prompt_end)
    cache["attn_monitor_track_prefix_dependency"] = bool(track_prefix_dependency)
    cache["attn_monitor_track_prefix_reverse_dependency"] = bool(
        track_prefix_reverse_dependency
    )
    cache["attn_monitor_track_prefix_future_dependency"] = bool(
        track_prefix_future_dependency
    )
    cache["attn_monitor_track_future_query_rows"] = bool(track_future_query_rows)
    cache["attn_monitor_trace_attention_relation_diagnostics"] = bool(
        trace_attention_relation_diagnostics
    )
    cache["attn_monitor_capture_qk_probe"] = bool(capture_qk_probe)
    cache["attn_monitor_capture_deployment_p8_relation"] = bool(
        capture_deployment_p8_relation
    )
    if capture_segment_head_mass:
        if not segment_key_groups:
            raise ValueError("segment head-mass capture requires key groups")
        normalized_groups = tuple(
            (
                torch.tensor(
                    tuple((int(position) for position in group)),
                    dtype=torch.long,
                    device=owner.device,
                )
                for group in segment_key_groups
            )
        )
        if any((group.numel() == 0 for group in normalized_groups)):
            raise ValueError("segment head-mass key groups must be nonempty")
        cache["attn_monitor_segment_key_groups"] = normalized_groups
        cache["attn_monitor_capture_segment_head_mass"] = True
    else:
        cache.pop("attn_monitor_segment_key_groups", None)
        cache.pop("attn_monitor_capture_segment_head_mass", None)


def prepare_tri_block_attention_step(
    model: Any,
    *,
    prediction_positions: torch.Tensor,
    excluded_query_positions: torch.Tensor | None = None,
) -> None:
    cache = _runtime_cache(model)
    _clear_monitor_step(cache)
    if prediction_positions.ndim != 1:
        raise ValueError("prediction_positions must be rank 1")
    if excluded_query_positions is not None:
        if (
            excluded_query_positions.ndim != 1
            or excluded_query_positions.dtype != torch.long
        ):
            raise ValueError("excluded_query_positions must be a rank-1 long tensor")
        cache["attn_monitor_excluded_query_positions"] = (
            excluded_query_positions.detach().clone()
        )


def _clear_monitor_step(cache: dict[str, Any]) -> None:
    for key in (
        "attn_monitor_current",
        "attn_monitor_dependency_sum",
        "attn_monitor_dependency_max",
        "attn_monitor_dependency_top1",
        "attn_monitor_dependency_top2",
        "attn_monitor_dependency_query_positions",
        "attn_monitor_dependency_layer_count",
        "attn_monitor_prompt_key_max",
        "attn_monitor_prompt_change_max",
        "attn_monitor_prefix_dependency_max",
        "attn_monitor_prefix_dependency_query_positions",
        "attn_monitor_prefix_reverse_dependency_max",
        "attn_monitor_prefix_future_dependency_max",
        "attn_monitor_future_prefix_dependency_query_positions",
        "attn_monitor_future_prefix_dependency_max",
        "attn_monitor_future_prefix_key_end",
        "attn_monitor_numeric_position_context",
        "attn_monitor_numeric_probe_groups",
        "attn_monitor_segment_key_groups",
        "attn_monitor_capture_segment_head_mass",
        "attn_monitor_excluded_query_positions",
    ):
        cache.pop(key, None)


def read_tri_block_attention_profile(model: Any) -> dict[str, torch.Tensor]:
    profile = _runtime_cache(model).get("attn_monitor_current")
    if not isinstance(profile, dict):
        raise RuntimeError("model did not publish a tri-block attention profile")
    return profile


def end_tri_block_attention_monitor(model: Any) -> None:
    cache = _runtime_cache(model)
    for key in (
        "attn_monitor_current",
        "attn_monitor_query_range",
        "attn_monitor_prediction_range",
        "attn_monitor_dependency_sum",
        "attn_monitor_dependency_max",
        "attn_monitor_dependency_top1",
        "attn_monitor_dependency_top2",
        "attn_monitor_dependency_query_positions",
        "attn_monitor_dependency_layer_count",
        "attn_monitor_prompt_end",
        "attn_monitor_prompt_key_max",
        "attn_monitor_prompt_change_max",
        "attn_monitor_track_prefix_dependency",
        "attn_monitor_track_prefix_reverse_dependency",
        "attn_monitor_track_prefix_future_dependency",
        "attn_monitor_track_future_query_rows",
        "attn_monitor_trace_attention_relation_diagnostics",
        "attn_monitor_capture_qk_probe",
        "attn_monitor_capture_deployment_p8_relation",
        "attn_monitor_prefix_dependency_max",
        "attn_monitor_prefix_dependency_query_positions",
        "attn_monitor_prefix_reverse_dependency_max",
        "attn_monitor_prefix_future_dependency_max",
        "attn_monitor_future_prefix_dependency_query_positions",
        "attn_monitor_future_prefix_dependency_max",
        "attn_monitor_future_prefix_key_end",
        "attn_monitor_numeric_position_context",
        "attn_monitor_numeric_probe_groups",
        "attn_monitor_segment_key_groups",
        "attn_monitor_capture_segment_head_mass",
    ):
        cache.pop(key, None)
