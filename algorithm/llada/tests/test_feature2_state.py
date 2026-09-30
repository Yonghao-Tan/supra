"""CPU tests for pure Feature 2 state and admission behavior."""

from __future__ import annotations
import unittest
import torch
from generation.state import (
    LOCKED,
    MASKED,
    ORIGIN_FALLBACK,
    ORIGIN_HIGH,
    ORIGIN_STABLE,
    TENTATIVE,
    Feature2BlockState,
    select_admissions,
)


class Feature2StateTest(unittest.TestCase):
    def test_stable_only_tail_bypass_rejects_fallback(self) -> None:
        state = Feature2BlockState((1, 2), device="cpu")
        tokens = torch.tensor([[1, 1]])
        proposal = torch.tensor([[2, 2]])
        selected = torch.tensor([[True, True]])
        state.admit(
            tokens,
            proposal,
            selected,
            torch.tensor([[False, False]]),
            torch.tensor([[True, False]]),
        )
        self.assertFalse(bool(state.bypass_tail_confirmation("stable_only").any()))
        bypassed = state.bypass_tail_confirmation("all")
        self.assertEqual(int(bypassed.sum().item()), 2)
        self.assertTrue(bool((state.state == LOCKED).all()))

    def test_precision_policies_select_only_requested_states(self) -> None:
        state = Feature2BlockState((1, 4), device="cpu")
        state.state[:] = torch.tensor([[MASKED, TENTATIVE, LOCKED, LOCKED]])
        state.precision_age[:] = torch.tensor([[-1, -1, 1, 3]])
        self.assertEqual(state.row_bits(policy="all_a8").tolist(), [[8, 8, 8, 8]])
        self.assertEqual(state.row_bits(policy="mature_only").tolist(), [[8, 8, 8, 4]])
        self.assertEqual(state.row_bits(policy="masked_only").tolist(), [[4, 8, 8, 8]])
        self.assertEqual(state.row_bits(policy="original").tolist(), [[4, 8, 8, 4]])

    def test_precision_age_and_row_bits(self) -> None:
        state = Feature2BlockState((1, 4), device="cpu")
        tokens = torch.full((1, 4), 99, dtype=torch.long)
        state.admit(
            tokens,
            torch.tensor([[10, 11, 12, 13]]),
            torch.tensor([[True, True, False, False]]),
            torch.tensor([[True, False, False, False]]),
            torch.tensor([[False, True, False, False]]),
        )
        self.assertEqual(state.precision_age.tolist(), [[0, -1, -1, -1]])
        self.assertEqual(state.row_bits().tolist(), [[8, 8, 4, 4]])
        locked_at_start = state.state == LOCKED
        state.advance_locked_age(locked_at_start)
        state.advance_locked_age(locked_at_start)
        state.advance_locked_age(locked_at_start)
        self.assertEqual(state.precision_age.tolist(), [[3, -1, -1, -1]])
        self.assertEqual(state.row_bits().tolist(), [[4, 8, 4, 4]])
        confirmation = state.confirm(
            tokens,
            torch.tensor([[10, 11, 12, 13]]),
            torch.tensor([[0.0, 0.8, 0.0, 0.0]]),
            confirm_tau=0.75,
            mask_id=99,
        )
        self.assertTrue(confirmation.locked[0, 1])
        self.assertEqual(state.precision_age.tolist(), [[3, 0, -1, -1]])
        self.assertEqual(state.row_bits().tolist(), [[4, 8, 4, 4]])

    def test_direct_lock_tentative_confirm_and_remask(self) -> None:
        state = Feature2BlockState((1, 4), device="cpu")
        tokens = torch.full((1, 4), 99, dtype=torch.long)
        admission = state.admit(
            tokens,
            torch.tensor([[10, 11, 12, 13]]),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[True, False, False, False]]),
            torch.tensor([[False, True, False, False]]),
        )
        self.assertEqual(state.state.tolist(), [[LOCKED, TENTATIVE, TENTATIVE, MASKED]])
        self.assertEqual(
            state.commit_origin.tolist(),
            [[ORIGIN_HIGH, ORIGIN_STABLE, ORIGIN_FALLBACK, 0]],
        )
        self.assertEqual(int(admission.direct_locked.sum()), 1)
        confirmation = state.confirm(
            tokens,
            torch.tensor([[10, 11, 7, 13]]),
            torch.tensor([[0.2, 0.68, 0.99, 0.2]]),
            confirm_tau=0.68,
            mask_id=99,
        )
        self.assertEqual(state.state.tolist(), [[LOCKED, LOCKED, MASKED, MASKED]])
        self.assertEqual(tokens.tolist(), [[10, 11, 99, 99]])
        self.assertTrue(confirmation.locked[0, 1])
        self.assertTrue(confirmation.remasked[0, 2])

    def test_threshold_boundaries_and_masked_history(self) -> None:
        state = Feature2BlockState((1, 3), device="cpu")
        proposal = torch.tensor([[4, 5, 6]])
        state.update_masked_history(proposal, torch.ones((1, 3), dtype=torch.bool))
        stable = state.last_top1 == proposal
        confidence = torch.tensor([[0.86, 0.68, 0.679]])
        high = confidence >= 0.86
        stable_low = stable & (confidence >= 0.68) & (confidence < 0.86)
        self.assertEqual(high.tolist(), [[True, False, False]])
        self.assertEqual(stable_low.tolist(), [[False, True, False]])

    def test_quota_fallback_and_equal_score_position_order(self) -> None:
        active = torch.ones((1, 6), dtype=torch.bool)
        high = torch.tensor([[True, True, True, False, False, False]])
        stable_low = torch.zeros_like(active)
        stable = torch.tensor([[False, True, False, False, False, False]])
        confidence = torch.tensor([[0.9, 0.85, 0.9, 0.8, 0.8, 0.8]])
        selected = select_admissions(
            active,
            high,
            stable_low,
            stable,
            confidence,
            torch.tensor([1]),
            remaining_forwards=3,
            budget_scale=1.0,
            stability_bonus=0.05,
        )
        self.assertEqual(torch.nonzero(selected[0]).flatten().tolist(), [0, 1])
        fallback = select_admissions(
            active,
            torch.zeros_like(active),
            torch.zeros_like(active),
            torch.zeros_like(active),
            torch.full_like(confidence, 0.8),
            torch.tensor([2]),
            remaining_forwards=3,
            budget_scale=16.0,
            stability_bonus=0.05,
        )
        self.assertEqual(torch.nonzero(fallback[0]).flatten().tolist(), [0, 1])
        max_quota = select_admissions(
            active,
            torch.ones_like(active),
            torch.zeros_like(active),
            torch.zeros_like(active),
            confidence,
            torch.tensor([1]),
            remaining_forwards=6,
            budget_scale=3.0,
            stability_bonus=0.05,
        )
        self.assertEqual(int(max_quota.sum()), 3)


if __name__ == "__main__":
    unittest.main()
