"""CPU tests for buffer-valid current/next block scheduling."""

from __future__ import annotations
import unittest
import torch
from generation.lookahead import ActivationBufferProfile, DynamicJointWindowScheduler


class ActivationBufferProfileTest(unittest.TestCase):
    def test_all_a8_32_rows_exactly_fit_the_384_kib_down_buffer(self) -> None:
        result = ActivationBufferProfile().analyze(
            torch.full((32,), 8, dtype=torch.int8)
        )
        self.assertTrue(result.fits_one_weight_pass)
        self.assertEqual(result.operator("qkv_k4096_all_a8").activation_bytes, 131072)
        self.assertEqual(result.operator("ffn_down_k12288").activation_bytes, 393216)
        self.assertEqual(result.max_weight_read_multiplier, 1)

    def test_all_a4_64_rows_fit_the_shared_384_kib_buffer(self) -> None:
        result = ActivationBufferProfile().analyze(
            torch.full((64,), 4, dtype=torch.int8)
        )
        self.assertTrue(result.fits_one_weight_pass)
        self.assertEqual(result.operator("qkv_k4096_all_a8").activation_bytes, 262144)
        self.assertEqual(result.operator("ffn_down_k12288").activation_bytes, 393216)

    def test_all_a4_65_rows_exceed_the_down_buffer(self) -> None:
        result = ActivationBufferProfile().analyze(
            torch.full((65,), 4, dtype=torch.int8)
        )
        self.assertFalse(result.fits_one_weight_pass)
        self.assertEqual(result.operator("ffn_down_k12288").fragments, 2)
        self.assertEqual(result.operator("qkv_k4096_all_a8").fragments, 1)

    def test_mixed_storage_and_pe_issue_groups_use_distinct_formulas(self) -> None:
        result = ActivationBufferProfile().analyze(
            torch.tensor([4] * 14 + [8] * 11, dtype=torch.int8)
        )
        self.assertEqual(result.operator("mixed_linear_k4096").activation_bytes, 73728)
        self.assertEqual(result.operator("ffn_down_k12288").activation_bytes, 221184)
        self.assertEqual(result.useful_slice_units, 36)
        self.assertEqual(result.issued_slice_units, 48)
        self.assertEqual(result.pe_issue_groups, 3)

    def test_a4_and_a8_share_one_pe_issue_group(self) -> None:
        result = ActivationBufferProfile().analyze(
            torch.tensor([4, 8], dtype=torch.int8)
        )
        self.assertEqual(result.useful_slice_units, 3)
        self.assertEqual(result.issued_slice_units, 16)
        self.assertEqual(result.pe_issue_groups, 1)
        self.assertEqual(result.operator("ffn_down_k12288").activation_bytes, 18432)


class DynamicJointWindowSchedulerTest(unittest.TestCase):
    def test_joint_row_limit_is_independent_of_activation_capacity(self) -> None:
        scheduler = DynamicJointWindowScheduler(max_next_rows=32)
        arguments = dict(
            base_positions=torch.arange(40, dtype=torch.long),
            base_row_bits=torch.full((40,), 4, dtype=torch.int8),
            next_block_start=64,
            next_row_bits=torch.full((32,), 4, dtype=torch.int8),
            next_unresolved=torch.ones(32, dtype=torch.bool),
            next_tentative=torch.zeros(32, dtype=torch.bool),
            next_priority=torch.ones(32),
        )
        selection = scheduler.select(**arguments)
        self.assertEqual(selection.joint_residency.active_rows, 48)
        self.assertEqual(selection.joint_residency.a4_rows, 48)
        self.assertEqual(scheduler.profile.k12288_capacity_bytes, 393216)
        with self.assertRaisesRegex(ValueError, "\\[1, 48\\]"):
            DynamicJointWindowScheduler(target_joint_rows=49)
        arguments.update(
            base_positions=torch.arange(49, dtype=torch.long),
            base_row_bits=torch.full((49,), 4, dtype=torch.int8),
        )
        with self.assertRaisesRegex(ValueError, "base rows exceed"):
            scheduler.select(**arguments)

    def test_a4_proposals_reserve_a8_verification_capacity(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=40, max_next_rows=32)
        selection = scheduler.select(
            base_positions=torch.arange(25, dtype=torch.long),
            base_row_bits=torch.tensor([4] * 14 + [8] * 11, dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.full((32,), 4, dtype=torch.int8),
            next_unresolved=torch.ones(32, dtype=torch.bool),
            next_tentative=torch.zeros(32, dtype=torch.bool),
            next_priority=torch.linspace(1.0, 0.0, 32),
        )
        self.assertEqual(selection.base_residency.active_rows, 25)
        self.assertEqual(selection.joint_residency.active_rows, 39)
        self.assertEqual(int(selection.added_local_positions.numel()), 14)
        self.assertEqual(
            selection.next_step_verification_residency.operator(
                "ffn_down_k12288"
            ).activation_bytes,
            393216,
        )
        self.assertEqual(selection.rejected_for_current_step_capacity, 0)
        self.assertEqual(selection.rejected_for_next_step_verification_capacity, 1)
        self.assertTrue(selection.joint_residency.fits_one_weight_pass)

    def test_next_step_current_a8_upgrade_reduces_future_admission(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=40, max_next_rows=16)
        common = {
            "base_positions": torch.arange(24, dtype=torch.long),
            "base_row_bits": torch.full((24,), 4, dtype=torch.int8),
            "next_block_start": 32,
            "next_row_bits": torch.full((16,), 4, dtype=torch.int8),
            "next_unresolved": torch.ones(16, dtype=torch.bool),
            "next_tentative": torch.zeros(16, dtype=torch.bool),
            "next_priority": torch.ones(16),
        }
        unchanged_current = scheduler.select(**common)
        upgraded_current = scheduler.select(
            **common, next_step_base_row_bits=torch.full((24,), 8, dtype=torch.int8)
        )
        self.assertEqual(unchanged_current.added_local_positions.numel(), 16)
        self.assertEqual(upgraded_current.added_local_positions.numel(), 8)
        self.assertEqual(
            upgraded_current.next_step_verification_residency.operator(
                "ffn_down_k12288"
            ).activation_bytes,
            393216,
        )
        self.assertEqual(
            upgraded_current.rejected_for_next_step_verification_capacity, 8
        )

    def test_all_a8_capacity_stops_the_40_row_window_at_32(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=40, max_next_rows=32)
        selection = scheduler.select(
            base_positions=torch.arange(25, dtype=torch.long),
            base_row_bits=torch.full((25,), 8, dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.full((32,), 8, dtype=torch.int8),
            next_unresolved=torch.ones(32, dtype=torch.bool),
            next_tentative=torch.zeros(32, dtype=torch.bool),
            next_priority=torch.linspace(1.0, 0.0, 32),
        )
        self.assertEqual(selection.joint_residency.active_rows, 32)
        self.assertEqual(selection.joint_residency.a8_rows, 32)
        self.assertGreater(selection.rejected_for_buffer, 0)
        self.assertGreater(selection.rejected_for_current_step_capacity, 0)
        self.assertEqual(selection.rejected_for_next_step_verification_capacity, 0)
        self.assertTrue(selection.joint_residency.fits_one_weight_pass)

    def test_future_can_use_an_existing_second_down_traversal(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=48, max_next_rows=8)
        selection = scheduler.select(
            base_positions=torch.arange(33, dtype=torch.long),
            base_row_bits=torch.full((33,), 8, dtype=torch.int8),
            next_block_start=64,
            next_row_bits=torch.full((8,), 4, dtype=torch.int8),
            next_unresolved=torch.ones(8, dtype=torch.bool),
            next_tentative=torch.zeros(8, dtype=torch.bool),
            next_priority=torch.ones(8),
        )
        self.assertEqual(
            selection.base_residency.operator("ffn_down_k12288").weight_read_multiplier,
            2,
        )
        self.assertEqual(selection.added_local_positions.numel(), 8)
        self.assertEqual(
            selection.joint_residency.operator(
                "ffn_down_k12288"
            ).weight_read_multiplier,
            2,
        )

    def test_mixed_selection_prefers_a8_with_verification_reservation(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=40, max_next_rows=32)
        selection = scheduler.select(
            base_positions=torch.arange(25, dtype=torch.long),
            base_row_bits=torch.tensor([4] * 14 + [8] * 11, dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.tensor([4] * 16 + [8] * 16, dtype=torch.int8),
            next_unresolved=torch.ones(32, dtype=torch.bool),
            next_tentative=torch.zeros(32, dtype=torch.bool),
            next_priority=torch.ones(32),
        )
        self.assertEqual(selection.joint_residency.a4_rows, 14)
        self.assertEqual(selection.joint_residency.a8_rows, 25)
        self.assertEqual(selection.joint_residency.issued_slice_units, 64)
        self.assertEqual(selection.next_step_verification_residency.a8_rows, 25)
        self.assertEqual(selection.joint_residency.max_weight_read_multiplier, 1)

    def test_priority_precedes_segment_fill_for_non_tentative_rows(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=40, max_next_rows=32)
        selection = scheduler.select(
            base_positions=torch.arange(25, dtype=torch.long),
            base_row_bits=torch.tensor([4] * 14 + [8] * 11, dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.tensor([4] * 16 + [8] * 16, dtype=torch.int8),
            next_unresolved=torch.ones(32, dtype=torch.bool),
            next_tentative=torch.zeros(32, dtype=torch.bool),
            next_priority=torch.tensor([1.0] * 16 + [0.1] * 16),
        )
        self.assertTrue(selection.added_local_positions.numel())
        self.assertTrue(bool((selection.added_local_positions < 16).all()))

    def test_admission_uses_only_the_ffn_down_capacity(self) -> None:
        scheduler = DynamicJointWindowScheduler(
            target_joint_rows=2,
            max_next_rows=1,
            profile=ActivationBufferProfile(k4096_capacity_bytes=1),
        )
        selection = scheduler.select(
            base_positions=torch.tensor([0], dtype=torch.long),
            base_row_bits=torch.tensor([4], dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.tensor([4], dtype=torch.int8),
            next_unresolved=torch.tensor([True]),
            next_tentative=torch.tensor([False]),
            next_priority=torch.tensor([1.0]),
        )
        self.assertEqual(selection.added_local_positions.tolist(), [0])
        self.assertFalse(selection.joint_residency.fits_one_weight_pass)
        self.assertTrue(
            selection.joint_residency.operator("ffn_down_k12288").fits_one_weight_pass
        )
        self.assertEqual(selection.rejected_for_buffer, 0)

    def test_existing_next_rows_progress_without_consuming_an_extra_slot(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=32, max_next_rows=4)
        selection = scheduler.select(
            base_positions=torch.tensor([0, 1, 32, 34], dtype=torch.long),
            base_row_bits=torch.tensor([8, 8, 4, 8], dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.tensor([4, 4, 8, 4], dtype=torch.int8),
            next_unresolved=torch.ones(4, dtype=torch.bool),
            next_tentative=torch.tensor([False, True, False, False]),
            next_priority=torch.ones(4),
        )
        self.assertEqual(selection.progress_local_positions.tolist(), [0, 2, 1, 3])
        self.assertEqual(selection.added_local_positions.tolist(), [1, 3])
        self.assertEqual(selection.joint_residency.active_rows, 6)

    def test_existing_a4_future_proposal_reserves_a8_verification(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=32, max_next_rows=1)
        selection = scheduler.select(
            base_positions=torch.cat(
                (torch.arange(31, dtype=torch.long), torch.tensor([32]))
            ),
            base_row_bits=torch.tensor([8] * 31 + [4], dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.tensor([4], dtype=torch.int8),
            next_unresolved=torch.tensor([True]),
            next_tentative=torch.tensor([False]),
            next_priority=torch.tensor([1.0]),
        )
        self.assertEqual(selection.progress_local_positions.tolist(), [0])
        self.assertEqual(selection.added_local_positions.numel(), 0)
        self.assertEqual(selection.next_step_verification_residency.a4_rows, 0)
        self.assertEqual(selection.next_step_verification_residency.a8_rows, 32)
        self.assertEqual(
            selection.next_step_verification_residency.operator(
                "ffn_down_k12288"
            ).activation_bytes,
            393216,
        )

    def test_tentative_rows_have_priority_over_new_a4_proposals(self) -> None:
        scheduler = DynamicJointWindowScheduler(target_joint_rows=32, max_next_rows=2)
        selection = scheduler.select(
            base_positions=torch.arange(30, dtype=torch.long),
            base_row_bits=torch.full((30,), 8, dtype=torch.int8),
            next_block_start=32,
            next_row_bits=torch.tensor([4, 8, 4, 8], dtype=torch.int8),
            next_unresolved=torch.ones(4, dtype=torch.bool),
            next_tentative=torch.tensor([False, True, False, True]),
            next_priority=torch.tensor([1.0, 0.2, 0.9, 0.1]),
        )
        self.assertEqual(selection.progress_local_positions.tolist(), [1, 3])
        self.assertEqual(selection.joint_residency.active_rows, 32)


if __name__ == "__main__":
    unittest.main()
