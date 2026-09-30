"""CPU oracles for the LLaDA SpinQuant rotation implementation."""

from __future__ import annotations
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from quantization import rotation as spinquant_rotation
from quantization.rotation import (
    checkpoint_identity,
    fixed_rotation_signs,
    normalized_hadamard,
    paley_hadamard_12,
    require_algo_output,
    rotate_ff_out_weight,
    rotate_input_weight_fixed,
    rotate_input_weight,
    rotate_o_weight,
    rotate_residual_output_weight,
    rotate_v_weight,
    random_hadamard_dense,
    structured_hadamard_12288,
    structured_hadamard_12288_bf16,
)


def _orthogonal(size: int) -> torch.Tensor:
    (q, _) = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64))
    return q.to(torch.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SpinQuantRotationTest(unittest.TestCase):
    def test_checkpoint_metadata_can_move_and_detects_content_changes(self) -> None:
        config = dict(architectures=["LLaDAModelLM"], d_model=4096, n_heads=32,
                      n_kv_heads=32, n_layers=32, mlp_hidden_size=12288,
                      include_bias=False, weight_tying=False)
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "model"
            original.mkdir()
            for name, content in (("config.json", json.dumps(config)),
                                  ("model.safetensors.index.json", "{}"),
                                  ("tokenizer.json", '{"tokens":[]}')):
                (original / name).write_text(content)
            profile = dict(config_sha256=_sha256(original / "config.json"),
                           index_sha256=_sha256(original / "model.safetensors.index.json"),
                           tokenizer_sha256=_sha256(original / "tokenizer.json"),
                           model_identity="LLaDA-8B-Instruct-original-bf16")
            moved = Path(temporary) / "model_metadata"
            shutil.copytree(original, moved)
            with patch.object(spinquant_rotation, "CHECKPOINT_PROFILES", {original: profile}):
                initial = checkpoint_identity(original)
                relocated = checkpoint_identity(moved)
                self.assertEqual(relocated, dict(initial, path=str(moved)))
                (moved / "tokenizer.json").write_text('{"tokens":[1]}')
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    checkpoint_identity(moved)
            with self.assertRaises(FileNotFoundError):
                checkpoint_identity(Path(temporary) / "missing")

    def test_output_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(spinquant_rotation, "ALGO_ROOT", root):
                output = root / "quantization/test"
                self.assertEqual(require_algo_output(output), output)
                with self.assertRaises(ValueError):
                    require_algo_output(root.parent / "outside")

    def test_h128_and_h12288_are_orthogonal(self) -> None:
        torch.manual_seed(20260813)
        x128 = torch.randn(3, 128)
        self.assertTrue(
            torch.allclose(
                normalized_hadamard(normalized_hadamard(x128)),
                x128,
                atol=2e-05,
                rtol=2e-05,
            )
        )
        h12 = paley_hadamard_12(dtype=torch.float64)
        self.assertTrue(
            torch.equal(h12 @ h12.T, torch.eye(12, dtype=torch.float64) * 12)
        )
        x = torch.randn(2, 12288)
        rotated = structured_hadamard_12288(x)
        restored = structured_hadamard_12288(rotated, transpose=True)
        self.assertTrue(torch.allclose(restored, x, atol=3e-05, rtol=3e-05))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(rotated, dim=-1),
                torch.linalg.vector_norm(x, dim=-1),
                atol=0.0003,
                rtol=3e-05,
            )
        )

    def test_h12288_bf16_target_is_finite_and_close_to_fp32(self) -> None:
        generator = torch.Generator().manual_seed(20260828)
        values = torch.randn(2, 12288, generator=generator).to(torch.bfloat16)
        target = structured_hadamard_12288_bf16(values)
        reference = structured_hadamard_12288(values.float()).to(torch.bfloat16)
        blocks = spinquant_rotation.bf16(values).reshape(-1, 12, 1024)
        stride = 1
        while stride < 1024:
            stage = blocks.reshape(-1, 12, 1024 // (2 * stride), 2, stride)
            (left, right) = stage.unbind(dim=-2)
            blocks = torch.stack(
                (
                    spinquant_rotation.bf16_add(left, right),
                    spinquant_rotation.bf16_add(left, -right),
                ),
                dim=-2,
            ).reshape_as(blocks)
            stride *= 2
        signs = paley_hadamard_12(dtype=torch.float32).reshape(1, 12, 12, 1)
        terms = spinquant_rotation.bf16_mul(blocks.unsqueeze(1), signs)
        terms = torch.cat(
            (terms, torch.zeros(*terms.shape[:2], 4, terms.shape[-1])), dim=2
        )
        while terms.shape[2] > 1:
            terms = spinquant_rotation.bf16_add(terms[:, :, 0::2], terms[:, :, 1::2])
        wide_reference = spinquant_rotation.bf16_mul(
            terms[:, :, 0], spinquant_rotation.bf16(torch.tensor(12288.0 ** (-0.5)))
        ).reshape_as(values)
        self.assertEqual(target.shape, values.shape)
        self.assertEqual(target.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(target, wide_reference))
        self.assertTrue(bool(torch.all(torch.isfinite(target))))
        self.assertLess(float((target.float() - reference.float()).abs().max()), 0.1)
        norm_ratio = torch.linalg.vector_norm(
            target.float()
        ) / torch.linalg.vector_norm(values.float())
        self.assertLess(abs(float(norm_ratio) - 1.0), 0.01)

    def test_r1_input_and_output_absorption(self) -> None:
        torch.manual_seed(3)
        r1 = _orthogonal(8)
        x = torch.randn(4, 8)
        input_weight = torch.randn(6, 8)
        output_weight = torch.randn(8, 6)
        x_rotated = x @ r1
        hidden = F.linear(x, input_weight)
        hidden_rotated = F.linear(x_rotated, rotate_input_weight(input_weight, r1))
        self.assertTrue(torch.allclose(hidden_rotated, hidden, atol=2e-05, rtol=2e-05))
        reference = F.linear(hidden, output_weight) @ r1
        actual = F.linear(hidden, rotate_residual_output_weight(output_weight, r1))
        self.assertTrue(torch.allclose(actual, reference, atol=2e-05, rtol=2e-05))

    def test_fixed_random_hadamard_sampling_and_absorption(self) -> None:
        (r1_a, r2_a) = fixed_rotation_signs()
        (r1_b, r2_b) = fixed_rotation_signs()
        self.assertTrue(torch.equal(r1_a, r1_b))
        self.assertTrue(torch.equal(r2_a[31], r2_b[31]))
        self.assertEqual(tuple(r1_a.shape), (4096,))
        self.assertEqual(len(r2_a), 32)
        sign = torch.tensor([1, -1, 1, -1, -1, 1, 1, -1], dtype=torch.int8)
        dense = random_hadamard_dense(sign)
        weight = torch.randn(5, 8)
        self.assertTrue(
            torch.allclose(
                rotate_input_weight_fixed(weight, sign),
                weight @ dense,
                atol=2e-05,
                rtol=2e-05,
            )
        )

    def test_r2_v_o_absorption(self) -> None:
        torch.manual_seed(4)
        r1 = _orthogonal(8)
        r2 = _orthogonal(4)
        heads = 2
        x = torch.randn(3, 8)
        v_weight = torch.randn(8, 8)
        o_weight = torch.randn(8, 8)
        v = F.linear(x, v_weight).reshape(3, heads, 4)
        v_rotated = F.linear(x @ r1, rotate_v_weight(v_weight, r1, r2, heads)).reshape(
            3, heads, 4
        )
        self.assertTrue(torch.allclose(v_rotated, v @ r2, atol=3e-05, rtol=3e-05))
        reference = F.linear(v.reshape(3, 8), o_weight) @ r1
        actual = F.linear(
            v_rotated.reshape(3, 8), rotate_o_weight(o_weight, r1, r2, heads)
        )
        self.assertTrue(torch.allclose(actual, reference, atol=3e-05, rtol=3e-05))

    def test_r4_ff_out_absorption(self) -> None:
        torch.manual_seed(5)
        r1 = _orthogonal(8)
        gated = torch.randn(1, 12288)
        weight = torch.randn(8, 12288)
        reference = F.linear(gated, weight) @ r1
        rotated = structured_hadamard_12288(gated)
        actual = F.linear(rotated, rotate_ff_out_weight(weight, r1, had=True))
        self.assertTrue(torch.allclose(actual, reference, atol=0.002, rtol=0.0002))


if __name__ == "__main__":
    unittest.main()
