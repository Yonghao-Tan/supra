from __future__ import annotations
import inspect
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
from numerics.bf16 import quantize_activation_per_row_bits_bf16
from numerics.linear_kernels import LinearNumericWorkspace
from numerics.precision import RowPrecisionContext
from quantization.model import SpinQuantW4A8Linear
from quantization.rotation import structured_hadamard_12288, structured_hadamard_12288_bf16
from quantization.numeric import (
    quantize_symmetric_w4,
    symmetric_w4_scale_bf16,
)
from calibration.initialize import (
    SymmetricLWC,
    _replace_with_lwc_result,
    calibration_row_bits,
    run_quantization,
)


class SpinQuantOmniQuantTest(unittest.TestCase):
    def test_cuda_device_is_resolved_before_model_loading(self) -> None:
        for requested, expected in (("cuda", "cuda:3"), ("cuda:1", "cuda:1")):
            with self.subTest(device=requested), patch(
                "calibration.initialize.torch.cuda.current_device", return_value=3
            ), patch("calibration.initialize.torch.cuda.set_device") as select, patch(
                "calibration.initialize.LLaDAModelLM.from_pretrained",
                side_effect=RuntimeError("stop before model loading"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                    run_quantization(
                        Path("unused-checkpoint"),
                        Path("unused-output"),
                        variant="fixed-r1r2-r4-only",
                        device=requested,
                    )
                select.assert_called_once_with(torch.device(expected))

    def test_replaced_ff_out_retains_initialization_fp32_h12288(self) -> None:
        block = nn.Module()
        block.ff_out = nn.Linear(12288, 2, bias=False).to(torch.bfloat16)
        solver = SymmetricLWC(block.ff_out, group_size=-1)
        solver.add_batch(
            torch.linspace(-1.0, 1.0, steps=12288, dtype=torch.bfloat16).reshape(
                1, 12288
            ),
            row_bits=torch.tensor([4], dtype=torch.int8),
        )
        writes = []
        writer = SimpleNamespace(
            variant="fixed-r1r2-r4-only",
            write_w4=lambda *args, **kwargs: writes.append((args, kwargs)),
        )
        context = RowPrecisionContext()
        _replace_with_lwc_result(
            block,
            "ff_out",
            solver,
            writer,
            0,
            torch.device("cpu"),
            LinearNumericWorkspace(),
            context,
        )
        self.assertIsInstance(block.ff_out, SpinQuantW4A8Linear)
        self.assertEqual(len(block.ff_out._forward_pre_hooks), 1)
        values = torch.randn(2, 12288, generator=torch.Generator().manual_seed(17)).to(torch.bfloat16)
        hook = next(iter(block.ff_out._forward_pre_hooks.values()))
        rotated = hook(block.ff_out, (values,))[0]
        self.assertTrue(torch.equal(rotated, structured_hadamard_12288(values).to(torch.bfloat16)))
        self.assertFalse(torch.equal(rotated, structured_hadamard_12288_bf16(values)))
        context.activate(torch.tensor([4], dtype=torch.int8))
        output = block.ff_out(torch.ones(1, 12288, dtype=torch.bfloat16))
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertEqual(len(writes), 1)

    def test_production_generator_defaults_to_gminus1(self) -> None:
        signature = inspect.signature(run_quantization)
        self.assertEqual(signature.parameters["group_size"].default, -1)
        self.assertNotIn("dense_proxy", signature.parameters)
        with self.assertRaisesRegex(ValueError, "G-1"):
            SymmetricLWC(nn.Linear(128, 2, bias=False), group_size=128)

    def test_state_ratio_proxy_is_deterministic_across_batches(self) -> None:
        first = calibration_row_bits(
            10, device=torch.device("cpu"), start=0, a8_period=4
        )
        second = calibration_row_bits(
            6, device=torch.device("cpu"), start=10, a8_period=4
        )
        combined = torch.cat((first, second))
        expected = torch.tensor(
            [4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8, 4, 4, 4, 8], dtype=torch.int8
        )
        self.assertTrue(torch.equal(combined, expected))

    def test_lwc_objective_is_no_worse_than_rtn_candidate(self) -> None:
        torch.manual_seed(71)
        layer = nn.Linear(13, 7, bias=False)
        with torch.no_grad():
            layer.weight[0, 0] = 4.5
            layer.weight[1, 3] = -3.75
        source = layer.weight.detach().float().clone()
        solver = SymmetricLWC(layer, group_size=-1, a8_period=4)
        solver.add_batch(torch.randn(3, 9, 13))
        (quantized, statistics) = solver.quantize(
            clip_min=0.6, clip_max=1.2, clip_steps=13, output_chunk=3
        )
        optimized = quantized.codes.float() * quantized.scale_bf16.unsqueeze(1)
        (direct_codes, direct) = quantize_symmetric_w4(
            source, symmetric_w4_scale_bf16(source)
        )

        def objective(candidate: torch.Tensor) -> torch.Tensor:
            return (
                source.square() * solver.teacher_square.unsqueeze(0)
                - 2.0 * source * candidate * solver.teacher_student.unsqueeze(0)
                + candidate.square() * solver.student_square.unsqueeze(0)
            ).sum()

        self.assertLessEqual(
            float(objective(optimized)), float(objective(direct)) + 0.001
        )
        self.assertEqual(tuple(quantized.codes.shape), tuple(source.shape))
        self.assertGreaterEqual(int(quantized.codes.min()), -8)
        self.assertLessEqual(int(quantized.codes.max()), 7)
        self.assertEqual(statistics["calibration_rows"], 27)
        self.assertEqual(statistics["a8_rows"], 6)
        self.assertEqual(direct_codes.shape, quantized.codes.shape)

    def test_lwc_requires_calibration_rows(self) -> None:
        solver = SymmetricLWC(nn.Linear(4, 3, bias=False), group_size=-1)
        with self.assertRaises(ValueError):
            solver.quantize()

    def test_lwc_explicit_row_bits_match_manual_moments(self) -> None:
        values = torch.tensor(
            [[[0.13, -0.71, 2.25], [4.0, 0.03, -1.0], [-2.0, 0.51, 0.07]]]
        )
        bits = torch.tensor([[4, 8, 4]], dtype=torch.int8)
        solver = SymmetricLWC(nn.Linear(3, 2, bias=False), group_size=-1)
        solver.add_batch(values, row_bits=bits)
        rows = values.reshape(-1, 3)
        quantized = quantize_activation_per_row_bits_bf16(rows, bits.reshape(-1))
        student = (
            (quantized.codes.float() * quantized.scale_bf16.unsqueeze(1))
            .to(torch.bfloat16)
            .float()
        )
        self.assertTrue(
            torch.equal(solver.teacher_square, rows.float().square().sum(0))
        )
        self.assertTrue(
            torch.equal(solver.teacher_student, (rows.float() * student).sum(0))
        )
        self.assertTrue(torch.equal(solver.student_square, student.square().sum(0)))
        (_, statistics) = solver.quantize()
        self.assertEqual(statistics["a4_rows"], 2)
        self.assertEqual(statistics["a8_rows"], 1)

    def test_lwc_explicit_row_bits_are_validated(self) -> None:
        values = torch.randn(1, 2, 3)
        solver = SymmetricLWC(nn.Linear(3, 2, bias=False), group_size=-1)
        for bits in (
            torch.tensor([4, 8], dtype=torch.int8),
            torch.tensor([[4, 6]], dtype=torch.int8),
            torch.tensor([[4, 8]], dtype=torch.int32),
        ):
            with self.assertRaises(ValueError):
                solver.add_batch(values, row_bits=bits)


if __name__ == "__main__":
    unittest.main()
