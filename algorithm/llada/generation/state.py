"""Pure Feature 2 token state and deterministic admission selection."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Callable
import torch

MASKED = 0
TENTATIVE = 1
LOCKED = 2
ORIGIN_NONE = 0
ORIGIN_HIGH = 1
ORIGIN_STABLE = 2
ORIGIN_FALLBACK = 3


@dataclass(frozen=True)
class ConfirmationResult:
    locked: torch.Tensor
    remasked: torch.Tensor


@dataclass(frozen=True)
class AdmissionResult:
    direct_locked: torch.Tensor
    stable_tentative: torch.Tensor
    fallback_tentative: torch.Tensor


class Feature2BlockState:
    """Track masked, one-forward tentative, and locked positions in one block."""

    def __init__(self, shape: tuple[int, int], *, device: torch.device | str) -> None:
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"shape must be positive [batch, block], got {shape}")
        self.state = torch.full(shape, MASKED, dtype=torch.int8, device=device)
        self.last_top1 = torch.full(shape, -1, dtype=torch.long, device=device)
        self.commit_origin = torch.full(
            shape, ORIGIN_NONE, dtype=torch.int8, device=device
        )
        self.precision_age = torch.full(shape, -1, dtype=torch.int16, device=device)

    def row_bits(
        self, *, maturity_age: int = 3, policy: str = "original"
    ) -> torch.Tensor:
        if maturity_age <= 0:
            raise ValueError("maturity_age must be positive")
        if policy not in {"all_a8", "mature_only", "masked_only", "original"}:
            raise ValueError("unsupported Feature 3 precision policy")
        mature = (self.state == LOCKED) & (self.precision_age >= int(maturity_age))
        masked = self.state == MASKED
        use_a4 = {
            "all_a8": torch.zeros_like(masked),
            "mature_only": mature,
            "masked_only": masked,
            "original": masked | mature,
        }[policy]
        return torch.where(use_a4, 4, 8).to(torch.int8)

    def advance_locked_age(self, locked_at_forward_start: torch.Tensor) -> None:
        self._check_shape(locked_at_forward_start, "locked_at_forward_start")
        advance = locked_at_forward_start & (self.state == LOCKED)
        self.precision_age[advance] += 1

    def confirm(
        self,
        tokens: torch.Tensor,
        current_top1: torch.Tensor,
        keep_probability: torch.Tensor,
        *,
        confirm_tau: float,
        mask_id: int,
    ) -> ConfirmationResult:
        self._check_shape(tokens, "tokens")
        self._check_shape(current_top1, "current_top1")
        self._check_shape(keep_probability, "keep_probability")
        due = self.state == TENTATIVE
        supported = (
            due & (current_top1 == tokens) & (keep_probability >= float(confirm_tau))
        )
        remasked = due & ~supported
        self.state[supported] = LOCKED
        self.state[remasked] = MASKED
        self.precision_age[supported] = 0
        self.precision_age[remasked] = -1
        self.commit_origin[remasked] = ORIGIN_NONE
        self.last_top1[remasked] = -1
        tokens[remasked] = int(mask_id)
        return ConfirmationResult(supported, remasked)

    def bypass_tail_confirmation(self, policy: str) -> torch.Tensor:
        """Lock an otherwise complete block only when its tentative origins are allowed."""
        if policy not in {"none", "stable_only", "all"}:
            raise ValueError("unsupported tail confirmation policy")
        bypassed = torch.zeros_like(self.state, dtype=torch.bool)
        if policy == "none" or bool((self.state == MASKED).any()):
            return bypassed
        due = self.state == TENTATIVE
        if not bool(due.any()):
            return bypassed
        if policy == "stable_only" and bool(
            (due & (self.commit_origin != ORIGIN_STABLE)).any()
        ):
            return bypassed
        self.state[due] = LOCKED
        self.precision_age[due] = 0
        return due

    def admit(
        self,
        tokens: torch.Tensor,
        proposal: torch.Tensor,
        selected: torch.Tensor,
        high_candidate: torch.Tensor,
        stable_low_candidate: torch.Tensor,
    ) -> AdmissionResult:
        for name, value in (
            ("tokens", tokens),
            ("proposal", proposal),
            ("selected", selected),
            ("high_candidate", high_candidate),
            ("stable_low_candidate", stable_low_candidate),
        ):
            self._check_shape(value, name)
        selected = selected & (self.state == MASKED)
        direct = selected & high_candidate
        stable = selected & ~direct & stable_low_candidate
        fallback = selected & ~direct & ~stable_low_candidate
        tokens[selected] = proposal[selected]
        self.state[direct] = LOCKED
        self.state[stable | fallback] = TENTATIVE
        self.precision_age[direct] = 0
        self.precision_age[stable | fallback] = -1
        self.commit_origin[direct] = ORIGIN_HIGH
        self.commit_origin[stable] = ORIGIN_STABLE
        self.commit_origin[fallback] = ORIGIN_FALLBACK
        self.last_top1[selected] = -1
        return AdmissionResult(direct, stable, fallback)

    def update_masked_history(
        self, proposal: torch.Tensor, eligible_mask: torch.Tensor
    ) -> None:
        self._check_shape(proposal, "proposal")
        self._check_shape(eligible_mask, "eligible_mask")
        keep = eligible_mask & (self.state == MASKED)
        self.last_top1[keep] = proposal[keep]
        self.last_top1[~(self.state == MASKED)] = -1

    def _check_shape(self, value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor) or value.shape != self.state.shape:
            raise ValueError(f"{name} must have shape {tuple(self.state.shape)}")


def select_admissions(
    admission_mask: torch.Tensor,
    high_candidate: torch.Tensor,
    stable_low_candidate: torch.Tensor,
    stable: torch.Tensor,
    confidence: torch.Tensor,
    scheduled_quota: torch.Tensor,
    *,
    remaining_forwards: int,
    budget_scale: float,
    stability_bonus: float,
    score_observer: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Select proposals by score, resolving equal scores by lower logical position."""
    if admission_mask.ndim != 2 or admission_mask.dtype != torch.bool:
        raise ValueError("admission_mask must be a rank-2 bool tensor")
    if any(
        (
            value.shape != admission_mask.shape
            for value in (high_candidate, stable_low_candidate, stable, confidence)
        )
    ):
        raise ValueError("candidate tensors must match admission_mask")
    if scheduled_quota.shape != admission_mask.shape[:1]:
        raise ValueError("scheduled_quota must contain one value per batch item")
    if remaining_forwards <= 0 or budget_scale < 1.0:
        raise ValueError(
            "remaining_forwards must be positive and budget_scale at least one"
        )
    selected = torch.zeros_like(admission_mask)
    eligible = high_candidate | stable_low_candidate
    score = confidence + stable.to(confidence.dtype) * float(stability_bonus)
    if score_observer is not None:
        score_observer(score.detach().clone())
    for batch_index in range(admission_mask.shape[0]):
        active = (
            torch.nonzero(admission_mask[batch_index], as_tuple=False)
            .flatten()
            .tolist()
        )
        if not active:
            continue
        minimum_required = math.ceil(len(active) / remaining_forwards)
        base_quota = max(1, int(scheduled_quota[batch_index].item()), minimum_required)
        max_quota = min(len(active), math.ceil(base_quota * float(budget_scale)))

        def ordered(indices: list[int]) -> list[int]:
            return sorted(
                indices,
                key=lambda position: (
                    -float(score[batch_index, position].item()),
                    position,
                ),
            )

        eligible_positions = ordered(
            [position for position in active if bool(eligible[batch_index, position])]
        )
        chosen = eligible_positions[:max_quota]
        if len(chosen) < base_quota:
            chosen_set = set(chosen)
            fallback = ordered(
                [position for position in active if position not in chosen_set]
            )
            chosen.extend(fallback[: base_quota - len(chosen)])
        selected[batch_index, chosen] = True
    return selected
