"""Hardware-valid scheduling for persistent current/next block progression."""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from numerics.bf16 import bf16_add



def _bf16_control(
    values: torch.Tensor | float, *, device: torch.device | None = None
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(torch.float32).to(torch.bfloat16)
    return torch.tensor(values, dtype=torch.bfloat16, device=device)


@dataclass(frozen=True)
class OperatorResidency:
    name: str
    activation_bytes: int
    capacity_bytes: int
    fragments: int
    weight_read_multiplier: int

    @property
    def fits_one_weight_pass(self) -> bool:
        return self.fragments == 1 and self.weight_read_multiplier == 1


@dataclass(frozen=True)
class ActivationResidency:
    a4_rows: int
    a8_rows: int
    issued_slice_units: int
    pe_issue_groups: int
    operators: tuple[OperatorResidency, ...]

    @property
    def active_rows(self) -> int:
        return self.a4_rows + self.a8_rows

    @property
    def useful_slice_units(self) -> int:
        return self.a4_rows + 2 * self.a8_rows

    @property
    def max_weight_read_multiplier(self) -> int:
        return max((item.weight_read_multiplier for item in self.operators), default=0)

    @property
    def fits_one_weight_pass(self) -> bool:
        return bool(self.operators) and all(
            (item.fits_one_weight_pass for item in self.operators)
        )

    def operator(self, name: str) -> OperatorResidency:
        for item in self.operators:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class ActivationBufferProfile:
    """Compact activation residency for one weight pass."""

    max_joint_rows: int = 48
    pe_issue_slice_units: int = 16
    k4096_capacity_bytes: int = 384 * 1024
    k12288_capacity_bytes: int = 384 * 1024

    @staticmethod
    def _fragments(byte_count: int, capacity: int) -> int:
        if byte_count <= 0:
            return 0
        return math.ceil(byte_count / capacity)

    def analyze(self, row_bits: torch.Tensor) -> ActivationResidency:
        if row_bits.ndim != 1 or row_bits.dtype != torch.int8:
            raise ValueError("row_bits must be a rank-1 int8 tensor")
        if bool(((row_bits != 4) & (row_bits != 8)).any()):
            raise ValueError("each physical row must use A4 or A8")
        a4_rows = int((row_bits == 4).sum().item())
        a8_rows = int((row_bits == 8).sum().item())
        rows = a4_rows + a8_rows
        if rows == 0:
            return ActivationResidency(0, 0, 0, 0, ())
        useful_slice_units = a4_rows + 2 * a8_rows
        pe_issue_groups = math.ceil(useful_slice_units / self.pe_issue_slice_units)
        issued_slice_units = pe_issue_groups * self.pe_issue_slice_units
        qkv_bytes = 4096 * rows
        mixed_k4096_bytes = 2048 * a4_rows + 4096 * a8_rows
        down_bytes = 6144 * useful_slice_units
        raw = (
            ("qkv_k4096_all_a8", qkv_bytes, self.k4096_capacity_bytes),
            ("mixed_linear_k4096", mixed_k4096_bytes, self.k4096_capacity_bytes),
            ("ffn_down_k12288", down_bytes, self.k12288_capacity_bytes),
        )
        operators = tuple(
            (
                OperatorResidency(
                    name=name,
                    activation_bytes=byte_count,
                    capacity_bytes=capacity,
                    fragments=self._fragments(byte_count, capacity),
                    weight_read_multiplier=self._fragments(byte_count, capacity),
                )
                for (name, byte_count, capacity) in raw
            )
        )
        return ActivationResidency(
            a4_rows=a4_rows,
            a8_rows=a8_rows,
            issued_slice_units=issued_slice_units,
            pe_issue_groups=pe_issue_groups,
            operators=operators,
        )


def _source_b_tie_rank_value(step_index: int, priority: float) -> float:
    if step_index < 2:
        return priority
    bucket = min(3, max(0, int(priority * 4)))
    bucket_order = (0.0, 3.0, 1.0, 2.0)
    return bucket_order[bucket] + priority * 0.125


@dataclass(frozen=True)
class JointRowSelection:
    progress_local_positions: torch.Tensor
    added_local_positions: torch.Tensor
    added_row_bits: torch.Tensor
    base_residency: ActivationResidency
    joint_residency: ActivationResidency
    next_step_verification_residency: ActivationResidency
    rejected_for_current_step_capacity: int
    rejected_for_next_step_verification_capacity: int
    rejected_for_budget: int
    source_b_optional_proposed_rows: int = 0
    source_b_optional_proposed_a8_rows: int = 0
    source_b_optional_reject_reason: str = "none"

    @property
    def rejected_for_buffer(self) -> int:
        """Total of current and next-step FFN-down capacity rejects."""
        return (
            self.rejected_for_current_step_capacity
            + self.rejected_for_next_step_verification_capacity
        )


class DynamicJointWindowScheduler:
    """Fill a byte-constrained joint window with useful next-block work."""

    def __init__(
        self,
        *,
        target_joint_rows: int = 48,
        max_next_rows: int = 8,
        min_reuse_score: float = 0.0,
        source_b_a4_only: bool = False,
        source_b_dependency_tie_rank: bool = False,
        profile: ActivationBufferProfile | None = None,
    ) -> None:
        self.profile = profile or ActivationBufferProfile()
        if not 1 <= target_joint_rows <= self.profile.max_joint_rows:
            raise ValueError(
                f"target_joint_rows must be in [1, {self.profile.max_joint_rows}]"
            )
        if max_next_rows <= 0:
            raise ValueError("max_next_rows must be positive")
        if not 0.0 <= min_reuse_score <= 1.0:
            raise ValueError("min_reuse_score must be in [0, 1]")
        self.target_joint_rows = int(target_joint_rows)
        self.max_next_rows = int(max_next_rows)
        self.min_reuse_score = float(_bf16_control(min_reuse_score).item())
        self.source_b_a4_only = bool(source_b_a4_only)
        self.source_b_dependency_tie_rank = bool(source_b_dependency_tie_rank)

    @staticmethod
    def _validate_vector(value: torch.Tensor, name: str, dtype: torch.dtype) -> None:
        if value.ndim != 1 or value.dtype != dtype:
            raise ValueError(f"{name} must be a rank-1 {dtype} tensor")

    def select(
        self,
        *,
        base_positions: torch.Tensor,
        base_row_bits: torch.Tensor,
        next_step_base_row_bits: torch.Tensor | None = None,
        next_block_start: int,
        next_row_bits: torch.Tensor,
        next_unresolved: torch.Tensor,
        next_tentative: torch.Tensor,
        next_priority: torch.Tensor,
        next_dependency_score: torch.Tensor | None = None,
        next_service_count: torch.Tensor | None = None,
        source_b_attempts: torch.Tensor | None = None,
        max_source_b_attempts: int = -1,
        source_b_retry_min_confidence: float = 0.0,
        next_last_confidence: torch.Tensor | None = None,
        target_prediction_rows: int = -1,
        current_prediction_rows: int = 0,
        step_index: int = -1,
    ) -> JointRowSelection:
        self._validate_vector(base_positions, "base_positions", torch.long)
        self._validate_vector(base_row_bits, "base_row_bits", torch.int8)
        if next_step_base_row_bits is None:
            next_step_base_row_bits = base_row_bits
        else:
            self._validate_vector(
                next_step_base_row_bits, "next_step_base_row_bits", torch.int8
            )
        self._validate_vector(next_row_bits, "next_row_bits", torch.int8)
        self._validate_vector(next_unresolved, "next_unresolved", torch.bool)
        self._validate_vector(next_tentative, "next_tentative", torch.bool)
        if next_priority.ndim != 1 or not next_priority.dtype.is_floating_point:
            raise ValueError("next_priority must be a rank-1 floating tensor")
        next_priority_bf16 = _bf16_control(next_priority)
        if self.source_b_dependency_tie_rank:
            if (
                next_dependency_score is None
                or next_dependency_score.ndim != 1
                or next_dependency_score.dtype != torch.bfloat16
                or next_dependency_score.shape != next_row_bits.shape
                or next_dependency_score.device != next_row_bits.device
            ):
                raise ValueError("Source B dependency tie rank requires matching BF16 scores")
        if next_service_count is None:
            next_service_count = torch.zeros_like(next_row_bits, dtype=torch.int32)
        elif next_service_count.ndim != 1 or next_service_count.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("next_service_count must be a rank-1 integer tensor")
        block_length = int(next_row_bits.numel())
        if max_source_b_attempts < -1:
            raise ValueError("Source B attempt limit must be -1 or nonnegative")
        if not 0.0 <= source_b_retry_min_confidence <= 1.0:
            raise ValueError("Source B retry confidence must be in [0,1]")
        if target_prediction_rows != -1 and not 1 <= target_prediction_rows <= 64:
            raise ValueError("joint prediction target must be -1 or in [1,64]")
        if not 0 <= current_prediction_rows <= base_positions.numel():
            raise ValueError("current predictions must be a subset of the base rows")
        if max_source_b_attempts >= 0 or source_b_retry_min_confidence > 0:
            if (
                source_b_attempts is None
                or source_b_attempts.shape != next_row_bits.shape
                or source_b_attempts.dtype != torch.int32
                or (source_b_attempts.device != next_row_bits.device)
                or bool((source_b_attempts < 0).any())
            ):
                raise ValueError(
                    "Source B attempt limit requires matching nonnegative int32 counts"
                )
        if source_b_retry_min_confidence > 0 and (
            next_last_confidence is None
            or next_last_confidence.shape != next_row_bits.shape
            or next_last_confidence.dtype != torch.bfloat16
            or next_last_confidence.device != next_row_bits.device
        ):
            raise ValueError("Source B retry requires matching BF16 confidence history")
        if any(
            (
                int(value.numel()) != block_length
                for value in (
                    next_unresolved,
                    next_tentative,
                    next_priority,
                    next_service_count,
                )
            )
        ):
            raise ValueError("next-block vectors must have matching lengths")
        if int(base_positions.numel()) != int(base_row_bits.numel()):
            raise ValueError("base positions and row bits must match")
        if base_positions.numel() > self.profile.max_joint_rows:
            raise ValueError(
                f"base rows exceed the joint row limit of {self.profile.max_joint_rows}"
            )
        if (
            int(next_step_base_row_bits.numel()) != int(base_row_bits.numel())
            or next_step_base_row_bits.device != base_row_bits.device
        ):
            raise ValueError(
                "next-step base row bits must match base row bits in shape and device"
            )
        base_residency = self.profile.analyze(base_row_bits)
        base_down_multiplier = base_residency.operator(
            "ffn_down_k12288"
        ).weight_read_multiplier
        initial_next_step_residency = self.profile.analyze(next_step_base_row_bits)
        next_step_down_multiplier = initial_next_step_residency.operator(
            "ffn_down_k12288"
        ).weight_read_multiplier
        existing_global = base_positions[
            (base_positions >= int(next_block_start))
            & (base_positions < int(next_block_start) + block_length)
        ]
        existing_local = existing_global - int(next_block_start)
        existing_mask = torch.zeros(
            block_length, dtype=torch.bool, device=next_row_bits.device
        )
        if existing_local.numel():
            existing_mask[existing_local] = True
        min_reuse_score_bf16 = _bf16_control(
            self.min_reuse_score, device=next_priority.device
        )
        eligible = next_unresolved & (next_priority_bf16 >= min_reuse_score_bf16)
        existing_eligible = torch.nonzero(
            eligible & existing_mask, as_tuple=False
        ).flatten()
        added_mask = eligible & ~existing_mask
        if max_source_b_attempts >= 0:
            added_mask &= next_tentative | (source_b_attempts < max_source_b_attempts)
        if source_b_retry_min_confidence > 0:
            added_mask &= (
                next_tentative
                | (source_b_attempts == 0)
                | (next_last_confidence >= _bf16_control(
                    source_b_retry_min_confidence, device=next_row_bits.device
                ))
            )
        added_eligible = torch.nonzero(added_mask, as_tuple=False).flatten()
        progress_limit = self.max_next_rows
        existing_limit = progress_limit
        if target_prediction_rows >= 0:
            due_added = int(next_tentative[added_eligible].sum().item())
            due_existing = int(next_tentative[existing_eligible].sum().item())
            progress_limit = min(self.max_next_rows, max(
                target_prediction_rows - current_prediction_rows, due_added + due_existing
            ))
            # Optional reuse must leave prediction capacity for pending confirmations.
            existing_limit = max(0, progress_limit - due_added)

        def order(indices: torch.Tensor, *, source_b: bool = False) -> list[int]:
            def value(index: int) -> float:
                priority = float(next_priority_bf16[index].item())
                if source_b and self.source_b_dependency_tie_rank:
                    return _source_b_tie_rank_value(step_index, priority)
                return priority

            return sorted(
                (int(value) for value in indices.tolist()),
                key=lambda index: (
                    -int(next_tentative[index].item()),
                    int(next_service_count[index].item()),
                    float(next_dependency_score[index].item())
                    if source_b and self.source_b_dependency_tie_rank
                    and not bool(next_tentative[index].item())
                    else 0.0,
                    -value(index),
                    index,
                ),
            )

        progress: list[int] = []
        added: list[int] = []
        added_bits: list[int] = []
        existing_rejected_for_next_step_capacity = 0
        for local_position in order(existing_eligible):
            if len(progress) >= existing_limit:
                break
            candidate_progress = progress + [local_position]
            candidate_positions = next_block_start + torch.tensor(
                candidate_progress, dtype=torch.long, device=base_positions.device
            )
            base_next_step_bits = next_step_base_row_bits.clone()
            base_next_step_bits[torch.isin(base_positions, candidate_positions)] = 8
            if (
                self.profile.analyze(base_next_step_bits)
                .operator("ffn_down_k12288")
                .weight_read_multiplier
                > next_step_down_multiplier
            ):
                existing_rejected_for_next_step_capacity += 1
                continue
            progress.append(local_position)
        progress_positions = next_block_start + torch.tensor(
            progress, dtype=torch.long, device=base_positions.device
        )
        base_next_step_bits = next_step_base_row_bits.clone()
        if progress_positions.numel():
            base_next_step_bits[torch.isin(base_positions, progress_positions)] = 8
        base_next_step_residency = self.profile.analyze(base_next_step_bits)
        ordered_added = order(added_eligible, source_b=True)
        a4_candidates = [
            index for index in ordered_added if int(next_row_bits[index].item()) == 4
        ]
        a8_candidates = [
            index for index in ordered_added if int(next_row_bits[index].item()) == 8
        ]
        row_slots = max(0, self.target_joint_rows - base_residency.active_rows)
        progress_slots = max(0, progress_limit - len(progress))
        add_limit = min(row_slots, progress_slots)
        best_score: tuple[float, ...] | None = None
        best_indices: list[int] = []
        best_residency = base_residency
        best_next_step_residency = base_next_step_residency
        max_current_step_capacity_rows = 0
        max_next_step_capacity_rows = 0
        for take_a4 in range(min(len(a4_candidates), add_limit) + 1):
            max_a8 = min(len(a8_candidates), add_limit - take_a4)
            for take_a8 in range(max_a8 + 1):
                selected = a4_candidates[:take_a4] + a8_candidates[:take_a8]
                selected_bits = torch.tensor(
                    [int(next_row_bits[index].item()) for index in selected],
                    dtype=torch.int8,
                    device=base_row_bits.device,
                )
                candidate_bits = (
                    torch.cat((base_row_bits, selected_bits))
                    if selected_bits.numel()
                    else base_row_bits
                )
                candidate = self.profile.analyze(candidate_bits)
                if (
                    candidate.operator("ffn_down_k12288").weight_read_multiplier
                    > base_down_multiplier
                ):
                    continue
                max_current_step_capacity_rows = max(
                    max_current_step_capacity_rows, len(selected)
                )
                next_step_bits = (
                    torch.cat((base_next_step_bits, torch.full_like(selected_bits, 8)))
                    if selected_bits.numel()
                    else base_next_step_bits
                )
                next_step_residency = self.profile.analyze(next_step_bits)
                if (
                    next_step_residency.operator(
                        "ffn_down_k12288"
                    ).weight_read_multiplier
                    > next_step_down_multiplier
                ):
                    continue
                max_next_step_capacity_rows = max(
                    max_next_step_capacity_rows, len(selected)
                )
                tentative_count = sum(
                    (int(next_tentative[index].item()) for index in selected)
                )
                useful_slice_units = candidate.useful_slice_units
                priority_sum = torch.zeros(
                    (), dtype=torch.bfloat16, device=next_priority.device
                )
                for index in selected:
                    priority_sum = bf16_add(priority_sum, next_priority_bf16[index]).to(
                        torch.bfloat16
                    )
                # With a fixed base, selected count and useful units determine
                # both issued units and take_a4, so neither adds a tie-breaker.
                score = (
                    float(tentative_count),
                    float(priority_sum.item()),
                    float(len(selected)),
                    float(useful_slice_units),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_indices = selected
                    best_residency = candidate
                    best_next_step_residency = next_step_residency
        if self.source_b_a4_only and best_indices:
            optional = [
                index
                for index in best_indices
                if not bool(next_tentative[index].item())
            ]
            optional_a8_rows = sum(
                int(next_row_bits[index].item()) == 8 for index in optional
            )
            optional_reject_reason = "none"
            if optional:
                due = [
                    index
                    for index in best_indices
                    if bool(next_tentative[index].item())
                ]
                due_bits = [int(next_row_bits[index].item()) for index in due]
                due_tensor = torch.tensor(
                    due_bits, dtype=torch.int8, device=base_row_bits.device
                )
                anchor_bits = torch.cat((base_row_bits, due_tensor))
                if optional_a8_rows:
                    optional_reject_reason = "a8"
                    best_indices = due
                    best_residency = self.profile.analyze(anchor_bits)
                    best_next_step_residency = self.profile.analyze(
                        torch.cat(
                            (base_next_step_bits, torch.full_like(due_tensor, 8))
                        )
                    )
        else:
            optional = []
            optional_a8_rows = 0
            optional_reject_reason = "none"
        selected_set = set(best_indices)
        added = [index for index in ordered_added if index in selected_set]
        added_bits = [int(next_row_bits[index].item()) for index in added]
        progress.extend(added)
        maximum_without_capacity = min(len(ordered_added), add_limit)
        rejected_for_current_step_capacity = max(
            0, maximum_without_capacity - max_current_step_capacity_rows
        )
        rejected_for_next_step_verification_capacity = (
            max(0, max_current_step_capacity_rows - max_next_step_capacity_rows)
            + existing_rejected_for_next_step_capacity
        )
        selected_set = set(progress)
        total_eligible = int(eligible.sum().item())
        rejected_for_budget = max(
            0,
            total_eligible
            - len(selected_set)
            - rejected_for_current_step_capacity
            - rejected_for_next_step_verification_capacity,
        )
        added_tensor = torch.tensor(
            added, dtype=torch.long, device=next_row_bits.device
        )
        added_bits_tensor = torch.tensor(
            added_bits, dtype=torch.int8, device=next_row_bits.device
        )
        return JointRowSelection(
            progress_local_positions=torch.tensor(
                progress, dtype=torch.long, device=next_row_bits.device
            ),
            added_local_positions=added_tensor,
            added_row_bits=added_bits_tensor,
            base_residency=base_residency,
            joint_residency=best_residency,
            next_step_verification_residency=best_next_step_residency,
            rejected_for_current_step_capacity=rejected_for_current_step_capacity,
            rejected_for_next_step_verification_capacity=rejected_for_next_step_verification_capacity,
            rejected_for_budget=rejected_for_budget,
            source_b_optional_proposed_rows=len(optional),
            source_b_optional_proposed_a8_rows=optional_a8_rows,
            source_b_optional_reject_reason=optional_reject_reason,
        )
