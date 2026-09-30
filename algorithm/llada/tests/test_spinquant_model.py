from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn as nn
from quantization import model as spinquant_model
from generation.refresh import (
    begin_tri_block_attention_monitor,
    configure_tri_block_attention_monitor,
    end_tri_block_attention_monitor,
    read_tri_block_attention_profile,
)
from numerics.int8_matmul import Int8MatmulWorkspace
from numerics.bf16 import (
    bf16,
    bf16_add,
    bf16_mul,
    quantize_per_row_bf16,
    rope_apply_bf16,
    silu_pwl_bf16,
)
from numerics.linear_kernels import LinearNumericWorkspace
from quantization.model import (
    SpinQuantK8CacheCodec,
    SpinQuantV8CacheCodec,
    SpinQuantW4A8Linear,
    _install_target_r4,
    _install_native_attention_numeric,
    _apply_rotary_positions,
    build_target_numeric_coverage,
    dequantize_spinquant_w4_bf16,
)
from quantization.numeric import SpinQuantW4Tensor
from quantization.numeric import quantize_symmetric_w8
from numerics.precision import RowPrecisionContext
from numerics.bf16 import quantize_activation_per_row_bits_bf16
from model.modeling_llada import (
    ActivationType,
    BlockType,
    LLaDAModel,
    LLaDALlamaBlock,
    ModelConfig,
    _explicit_write_local_mask,
)


class SpinQuantModelTest(unittest.TestCase):
    def test_attention_masks_exclude_keys_and_keep_visible_context(self) -> None:
        for manual in (False, True):
            with self.subTest(manual=manual):
                torch.manual_seed(7)
                config = ModelConfig(
                    d_model=16, n_heads=2, n_kv_heads=2, n_layers=1,
                    mlp_hidden_size=32, activation_type=ActivationType.silu,
                    block_type=BlockType.llama, max_sequence_length=8,
                    vocab_size=128, embedding_size=128, rope=True,
                    attention_dropout=0.0, residual_dropout=0.0,
                    embedding_dropout=0.0, use_manual_attention=manual,
                )
                model = LLaDAModel(config).eval()
                bias = torch.zeros(1, 1, 3, 3)
                bias[:, :, :, -1] = float("-inf")
                with torch.no_grad():
                    left = model(torch.tensor([[1, 2, 3]]), attention_bias=bias).logits[:, :2]
                    right = model(torch.tensor([[1, 2, 4]]), attention_bias=bias).logits[:, :2]
                    padded = model(torch.tensor([[1, 2, 3]]),
                                   attention_mask=torch.tensor([[1, 1, 0]])).logits[:, :2]
                    visible = model(torch.tensor([[1, 2]])).logits
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                torch.testing.assert_close(padded, visible, rtol=1e-5, atol=1e-6)

    def test_target_rope_uses_staged_bf16_when_full_precision_is_enabled(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            rope_full_precision=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).eval()
        rotary = model.transformer.blocks[0].rotary_emb
        q = torch.linspace(-3.7, 4.1, steps=4 * 128, dtype=torch.bfloat16).reshape(
            1, 1, 4, 128
        )
        k = torch.linspace(2.9, -4.3, steps=4 * 128, dtype=torch.bfloat16).reshape(
            1, 1, 4, 128
        )
        positions = torch.arange(4)
        rotary._target_numeric_bf16 = False
        (native_q, native_k) = _apply_rotary_positions(rotary, q, k, positions)
        rotary._target_numeric_bf16 = True
        (target_q, target_k) = _apply_rotary_positions(rotary, q, k, positions)
        (sin, cos) = rotary.get_rotary_embedding(4, q.device)
        expected_q = rope_apply_bf16(q, sin.to(torch.bfloat16), cos.to(torch.bfloat16))
        expected_k = rope_apply_bf16(k, sin.to(torch.bfloat16), cos.to(torch.bfloat16))
        self.assertTrue(torch.equal(target_q, expected_q))
        self.assertTrue(torch.equal(target_k, expected_k))
        self.assertFalse(torch.equal(target_q, native_q))
        self.assertFalse(torch.equal(target_k, native_k))

    def test_first_layer_deployment_p8_capture_is_not_overwritten(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=3,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).to(torch.bfloat16).eval()
        for block in model.transformer.blocks:
            _install_native_attention_numeric(
                block,
                workspace=Int8MatmulWorkspace(),
                qk_int8=True,
                softmax_lut=True,
                probability_p8=False,
                k8_cache=True,
                v8_codec=SpinQuantV8CacheCodec(torch.tensor([0.25])),
                rope_before_k8=True,
            )
        configure_tri_block_attention_monitor(model, layer_mode="first")
        begin_tri_block_attention_monitor(
            model,
            region_start=0,
            region_end=4,
            prediction_start=0,
            prediction_end=4,
            prompt_end=1,
            capture_deployment_p8_relation=True,
        )
        try:
            with torch.no_grad():
                model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
            profile = read_tri_block_attention_profile(model)
        finally:
            end_tri_block_attention_monitor(model)
        probe = profile["numeric_probe"]
        self.assertEqual(profile["layer_reduction"], "first")
        self.assertEqual(int(profile["layer_count"].item()), 1)
        self.assertEqual(probe["selected_layer_id"], 0)
        self.assertEqual(probe["p8_relation_layer_count"], 1)
        self.assertTrue(
            torch.equal(
                probe["selected_layer_p8_relation_bf16"], probe["l0_p8_relation_bf16"]
            )
        )
        self.assertNotIn("l31_p8_relation_bf16", probe)

    def test_deployment_p8_relation_capture_omits_raw_probe_operands(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).to(torch.bfloat16).eval()
        block = model.transformer.blocks[0]
        _install_native_attention_numeric(
            block,
            workspace=Int8MatmulWorkspace(),
            qk_int8=True,
            softmax_lut=True,
            probability_p8=False,
            k8_cache=True,
            v8_codec=SpinQuantV8CacheCodec(torch.tensor([0.25])),
            rope_before_k8=True,
        )
        configure_tri_block_attention_monitor(model, layer_mode="all")
        begin_tri_block_attention_monitor(
            model,
            region_start=0,
            region_end=4,
            prediction_start=0,
            prediction_end=4,
            prompt_end=1,
            capture_deployment_p8_relation=True,
        )
        try:
            with torch.no_grad():
                model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
            probe = read_tri_block_attention_profile(model)["numeric_probe"]
        finally:
            end_tri_block_attention_monitor(model)
        self.assertEqual(probe["p8_relation_layer_count"], 1)
        self.assertIn("l31_p8_relation_bf16", probe)
        self.assertIn("all_layer_max_p8_relation_bf16", probe)
        self.assertNotIn("q_codes", probe)
        self.assertNotIn("k_codes_qk_operand", probe)
        self.assertNotIn("l31_p_codes", probe)

    def test_target_numeric_probe_reads_qk_and_pv_integer_operands(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).to(torch.bfloat16).eval()
        block = model.transformer.blocks[0]
        _install_native_attention_numeric(
            block,
            workspace=Int8MatmulWorkspace(),
            qk_int8=True,
            softmax_lut=True,
            probability_p8=False,
            k8_cache=True,
            v8_codec=SpinQuantV8CacheCodec(torch.tensor([0.25])),
            rope_before_k8=True,
        )
        configure_tri_block_attention_monitor(model, layer_mode="all")
        begin_tri_block_attention_monitor(
            model,
            region_start=0,
            region_end=4,
            prediction_start=0,
            prediction_end=4,
            prompt_end=1,
            capture_qk_probe=True,
        )
        integer_operands: list[tuple[torch.Tensor, torch.Tensor]] = []
        original_matmul = spinquant_model.int8_batched_matmul

        def capture_matmul(
            lhs: torch.Tensor, rhs: torch.Tensor, *, output: torch.Tensor
        ) -> torch.Tensor:
            integer_operands.append((lhs.detach().clone(), rhs.detach().clone()))
            return original_matmul(lhs, rhs, output=output)

        try:
            with patch.object(
                spinquant_model, "int8_batched_matmul", side_effect=capture_matmul
            ):
                with torch.no_grad():
                    model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
            profile = read_tri_block_attention_profile(model)
        finally:
            end_tri_block_attention_monitor(model)
        self.assertEqual(len(integer_operands), 2)
        probe = profile["numeric_probe"]
        self.assertEqual(probe["layer_id"], 0)
        self.assertEqual(probe["staged_group"], -1)
        self.assertEqual(probe["p8_relation_layer_count"], 1)
        self.assertTrue(
            torch.equal(probe["query_positions"], torch.arange(4).view(1, 4))
        )
        self.assertTrue(torch.equal(probe["key_positions"], torch.arange(4).view(1, 4)))
        self.assertEqual(probe["kv_write_positions"].shape, (1, 0))
        self.assertTrue(torch.equal(probe["q_codes"], integer_operands[0][0]))
        self.assertTrue(
            torch.equal(
                probe["k_codes_qk_operand"], integer_operands[0][1].transpose(-2, -1)
            )
        )
        self.assertTrue(torch.equal(probe["l31_p_codes"], integer_operands[1][0]))
        reconstructed_p = bf16(
            probe["l31_p_codes"].float() * probe["l31_p_scale_bf16"].unsqueeze(-1)
        ).to(torch.bfloat16)
        expected_relation = reconstructed_p.mean(dim=1).to(torch.bfloat16)
        self.assertTrue(torch.equal(probe["l31_p8_relation_bf16"], expected_relation))
        self.assertTrue(
            torch.equal(probe["all_layer_max_p8_relation_bf16"], expected_relation)
        )

    def test_target_k8_cache_encodes_rope_output(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).eval()
        block = model.transformer.blocks[0]
        _install_native_attention_numeric(
            block,
            workspace=Int8MatmulWorkspace(),
            qk_int8=True,
            softmax_lut=True,
            probability_p8=True,
            k8_cache=True,
            rope_before_k8=True,
        )
        captured: list[torch.Tensor] = []
        hook = block.k_proj.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach())
        )
        with torch.no_grad():
            result = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        hook.remove()
        self.assertEqual(len(captured), 1)
        raw_k = captured[0].view(1, 4, 1, 128).transpose(1, 2)
        (sin, cos) = block.rotary_emb.get_rotary_embedding(4, raw_k.device)
        rotated_k = rope_apply_bf16(
            raw_k, sin.to(torch.bfloat16), cos.to(torch.bfloat16)
        ).to(raw_k.dtype)
        (expected_codes, expected_scales) = block._spinquant_k_cache_codec.encode(
            rotated_k
        )
        (actual_codes, actual_scales, _) = result.attn_key_values[0]
        self.assertTrue(torch.equal(actual_codes, expected_codes))
        self.assertTrue(torch.equal(actual_scales, expected_scales))

    def test_target_bf16_cache_stores_rope_output(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).eval()
        block = model.transformer.blocks[0]
        _install_native_attention_numeric(
            block,
            workspace=None,
            qk_int8=False,
            softmax_lut=True,
            probability_p8=False,
            target_rope_before_cache=True,
        )
        captured: list[torch.Tensor] = []
        hook = block.k_proj.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach())
        )
        try:
            with torch.no_grad():
                result = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        finally:
            hook.remove()
        raw_k = captured[0].view(1, 4, 1, 128).transpose(1, 2)
        (sin, cos) = block.rotary_emb.get_rotary_embedding(4, raw_k.device)
        expected = rope_apply_bf16(
            raw_k, sin.to(torch.bfloat16), cos.to(torch.bfloat16)
        ).to(raw_k.dtype)
        (actual, _) = result.attn_key_values[0]
        self.assertTrue(torch.equal(actual, expected))
        with torch.no_grad():
            appended = model(torch.tensor([[5]]), past_key_values=result.attn_key_values,
                             use_cache=True)
        appended_key, appended_value = appended.attn_key_values[0]
        self.assertEqual(appended_key.shape[-2], 5)
        self.assertEqual(appended_value.shape[-2], 5)
        self.assertTrue(torch.equal(appended_key[:, :, :4], expected))

    def test_target_k8_cache_append_replace_and_handoff(self) -> None:
        config = ModelConfig(
            d_model=128,
            n_heads=1,
            n_kv_heads=1,
            n_layers=1,
            mlp_hidden_size=256,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=8,
            vocab_size=32,
            embedding_size=32,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModel(config).eval()
        block = model.transformer.blocks[0]
        _install_native_attention_numeric(
            block,
            workspace=Int8MatmulWorkspace(),
            qk_int8=True,
            softmax_lut=True,
            probability_p8=True,
            k8_cache=True,
            rope_before_k8=True,
        )
        captured: list[torch.Tensor] = []
        hook = block.k_proj.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output.detach())
        )
        try:
            with torch.no_grad():
                full_sequence = model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
                appended = model(
                    torch.tensor([[5]]),
                    past_key_values=full_sequence.attn_key_values,
                    use_cache=True,
                )
            append_raw_k = captured[-1].view(1, 1, 1, 128)
            (sin, cos) = block.rotary_emb.get_rotary_embedding(5, append_raw_k.device)
            append_rotated_k = rope_apply_bf16(
                append_raw_k,
                sin[:, :, 4:5].to(torch.bfloat16),
                cos[:, :, 4:5].to(torch.bfloat16),
            ).to(append_raw_k.dtype)
            (append_codes, append_scales) = block._spinquant_k_cache_codec.encode(
                append_rotated_k
            )
            (cache_codes, cache_scales, _) = appended.attn_key_values[0]
            self.assertTrue(torch.equal(cache_codes[:, :, 4:5], append_codes))
            self.assertTrue(torch.equal(cache_scales[:, :, 4:5], append_scales))
            before_replace_codes = cache_codes.clone()
            before_replace_scales = cache_scales.clone()
            query_positions = torch.tensor([[1, 3]], dtype=torch.long)
            with torch.no_grad():
                replaced = model(
                    torch.tensor([[6, 7]]),
                    past_key_values=appended.attn_key_values,
                    use_cache=True,
                    query_position_ids=query_positions,
                    kv_write_position_ids=query_positions,
                )
            replace_raw_k = captured[-1].view(1, 1, 2, 128)
            (sin, cos) = block.rotary_emb.get_rotary_embedding(4, replace_raw_k.device)
            replace_rotated_k = rope_apply_bf16(
                replace_raw_k,
                sin.index_select(2, query_positions[0]).to(torch.bfloat16),
                cos.index_select(2, query_positions[0]).to(torch.bfloat16),
            ).to(replace_raw_k.dtype)
            (replace_codes, replace_scales) = block._spinquant_k_cache_codec.encode(
                replace_rotated_k
            )
            (replaced_codes, replaced_scales, _) = replaced.attn_key_values[0]
            self.assertTrue(
                torch.equal(
                    replaced_codes.index_select(2, query_positions[0]), replace_codes
                )
            )
            self.assertTrue(
                torch.equal(
                    replaced_scales.index_select(2, query_positions[0]), replace_scales
                )
            )
            untouched = torch.tensor([0, 2, 4], dtype=torch.long)
            self.assertTrue(
                torch.equal(
                    replaced_codes.index_select(2, untouched),
                    before_replace_codes.index_select(2, untouched),
                )
            )
            self.assertTrue(
                torch.equal(
                    replaced_scales.index_select(2, untouched),
                    before_replace_scales.index_select(2, untouched),
                )
            )
            handoff_codes = replaced_codes.clone()
            handoff_scales = replaced_scales.clone()
            with torch.no_grad():
                handed_off = model(
                    torch.tensor([[8]]),
                    past_key_values=replaced.attn_key_values,
                    use_cache=True,
                    query_position_ids=torch.tensor([[4]], dtype=torch.long),
                    kv_write_position_ids=torch.empty((1, 0), dtype=torch.long),
                )
            (final_codes, final_scales, _) = handed_off.attn_key_values[0]
            self.assertTrue(torch.equal(final_codes, handoff_codes))
            self.assertTrue(torch.equal(final_scales, handoff_scales))
        finally:
            hook.remove()

    def test_target_numeric_coverage_rejects_missing_operators(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(weight_tying=True),
            model=SimpleNamespace(transformer=SimpleNamespace(blocks=[], ff_out=None)),
        )
        coverage = build_target_numeric_coverage(
            model,
            candidate_numeric_mode="native_fp64",
            candidate_numeric_schedule="candidate-native-fp64-softmax/v1",
            candidate_suppression="missing",
            require_dynamic_a4a8=True,
            rotation_source="",
            rotation_sha256="missing",
        )
        self.assertFalse(coverage["valid"])
        self.assertIn("Transformer inventory", coverage["errors"][0])
        self.assertTrue(any(("lm_head" in error for error in coverage["errors"])))
        self.assertTrue(any(("candidate" in error for error in coverage["errors"])))

    def test_target_numeric_coverage_records_complete_gminus1_contract(self) -> None:
        context = RowPrecisionContext()

        def make_w4() -> SpinQuantW4A8Linear:
            return SpinQuantW4A8Linear(
                SpinQuantW4Tensor(
                    codes=torch.tensor(
                        [[-8, -1, 0, 7], [7, 1, 0, -8]], dtype=torch.int8
                    ),
                    scale_bf16=torch.tensor(
                        [0.125, 0.25], dtype=torch.bfloat16
                    ).float(),
                ),
                workspace=LinearNumericWorkspace(),
                row_precision_context=context,
            )

        blocks = []
        for _ in range(32):
            block = SimpleNamespace(
                **{
                    name: make_w4()
                    for name in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "attn_out",
                        "ff_proj",
                        "up_proj",
                        "ff_out",
                    )
                }
            )
            block._spinquant_k_cache_codec = SpinQuantK8CacheCodec()
            block._spinquant_v_cache_codec = SpinQuantV8CacheCodec(
                torch.ones(32, dtype=torch.bfloat16)
            )
            block._spinquant_score_override = lambda *_args: None
            block._spinquant_probability_override = lambda *_args: None
            block._spinquant_context_override = lambda *_args: None
            block.rotary_emb = SimpleNamespace(_target_numeric_bf16=True)
            blocks.append(block)
        lm_head = SpinQuantW4A8Linear(
            quantize_symmetric_w8(
                torch.tensor(
                    [[-2.0, -1.0, 1.0, 2.0], [1.0, 0.5, -0.5, -1.0]],
                    dtype=torch.bfloat16,
                )
            ),
            workspace=LinearNumericWorkspace(),
            activation_bits=8,
            weight_bits=8,
        )
        model = SimpleNamespace(
            config=SimpleNamespace(weight_tying=False, n_heads=32, n_kv_heads=32),
            model=SimpleNamespace(
                transformer=SimpleNamespace(blocks=blocks, ff_out=lm_head)
            ),
            _target_numeric_k_cache_order="rope-before-k8/v1",
            _target_numeric_rope="bf16-mul-add-frozen-table/v1",
            _target_numeric_rope_table_shape=[4096, 128],
            _target_numeric_rope_table_sha256="1" * 64,
            _target_numeric_rms_norm_count=65,
            _target_numeric_bf16_residual_block_count=32,
            _target_numeric_silu_pwl_segments=16,
            _target_numeric_swiglu_multiply_block_count=32,
            _target_numeric_r4="h12288-bf16-staged/v1",
            _spinquant_lm_head_weight_bits=8,
        )
        arguments = {
            "candidate_numeric_mode": "bf16_lut",
            "candidate_numeric_schedule": "candidate-online-bf16-lut-lanes64/v1",
            "candidate_suppression": "raw-confidence-plus-zero-action/v1",
            "require_dynamic_a4a8": True,
            "rotation_source": "spinquant_artifact:fixed-r1r2-r4-only:absorbed_r1_r2",
            "rotation_sha256": "a" * 64,
        }
        coverage = build_target_numeric_coverage(model, **arguments)
        self.assertTrue(coverage["valid"], msg=coverage["errors"])
        self.assertEqual(coverage["schema_version"], "llada-target-numeric-coverage/v3")
        self.assertEqual(coverage["transformer_weight"]["group_size"], -1)
        self.assertEqual(
            coverage["transformer_weight"]["scale_shape_histogram"],
            [{"shape": [2], "count": 224}],
        )
        self.assertEqual(
            coverage["transformer_activation"]["code_ranges"],
            {"a4": [-7, 7], "a8": [-127, 127]},
        )
        self.assertEqual(coverage["attention_cache"]["scales"]["v8"]["shape"], [32, 32])
        self.assertEqual(coverage["residual"]["attention"], "bf16_add")
        self.assertEqual(coverage["residual"]["ffn"], "bf16_add")
        self.assertEqual(coverage["rotation"]["sha256"], "a" * 64)
        self.assertEqual(
            coverage["rotation"]["r1"]["artifact_manifest_sha256"], "a" * 64
        )
        self.assertEqual(coverage["rotation"]["r2"]["layer_count"], 32)
        self.assertEqual(coverage["lm_head"]["weight_scale_shape"], [2])
        self.assertEqual(len(coverage["candidate"]["lut_sha256"]), 64)
        from unittest.mock import patch

        with patch("numerics.candidate.CANDIDATE_LANES", 512), patch(
            "numerics.candidate.CANDIDATE_NUMERIC_SCHEDULE",
            "candidate-online-bf16-lut-lanes512/v1",
        ):
            self.assertFalse(build_target_numeric_coverage(model, **arguments)["valid"])
            diagnostic = build_target_numeric_coverage(
                model,
                **dict(
                    arguments,
                    candidate_numeric_schedule="candidate-online-bf16-lut-lanes512/v1",
                ),
            )
            self.assertTrue(diagnostic["valid"], msg=diagnostic["errors"])
            self.assertEqual(diagnostic["candidate"]["logical_state_count"], 512)
            self.assertFalse(diagnostic["candidate"]["matches_64_lane_reference"])
        blocks[0].rotary_emb._target_numeric_bf16 = False
        rejected = build_target_numeric_coverage(model, **arguments)
        self.assertFalse(rejected["valid"])
        self.assertTrue(
            any(("execute BF16 mul/add" in error for error in rejected["errors"]))
        )
        blocks[0].rotary_emb._target_numeric_bf16 = True
        model.config.n_kv_heads = 8
        rejected = build_target_numeric_coverage(model, **arguments)
        self.assertFalse(rejected["valid"])
        self.assertTrue(
            any(
                (
                    "requires 32 query and KV heads" in error
                    for error in rejected["errors"]
                )
            )
        )
        model.config.n_kv_heads = 32
        blocks[0].q_proj.weight_group_size = 128
        rejected = build_target_numeric_coverage(model, **arguments)
        self.assertFalse(rejected["valid"])
        self.assertTrue(
            any(("declared group arithmetic" in error for error in rejected["errors"]))
        )

        with self.assertRaisesRegex(ValueError, "unsupported Transformer W4 arithmetic"):
            build_target_numeric_coverage(model, **arguments, transformer_weight_arithmetic="g128_bf16_serial/v1")

    def test_target_block_residual_and_swiglu_use_bf16_boundaries(self) -> None:
        block = SimpleNamespace(
            _target_numeric_bf16_residual=True, _target_numeric_silu_pwl16=True
        )
        lhs = torch.tensor([[1.0, -0.5]], dtype=torch.bfloat16)
        rhs = torch.tensor([[0.25, 0.125]], dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                LLaDALlamaBlock._residual_add(block, lhs, rhs),
                bf16_add(lhs, rhs).to(torch.bfloat16),
            )
        )
        gate = torch.tensor([[-1.0, 0.5]], dtype=torch.bfloat16)
        up = torch.tensor([[0.75, -0.25]], dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                LLaDALlamaBlock._silu_multiply(block, gate, up),
                bf16_mul(silu_pwl_bf16(gate), up).to(torch.bfloat16),
            )
        )

    def test_transactional_native_cache_maps_a8_writes_to_local_queries(self) -> None:
        query_positions = torch.tensor([0, 2, 4, 6])
        write_positions = torch.tensor([2, 6])
        self.assertTrue(
            torch.equal(
                _explicit_write_local_mask(query_positions, write_positions),
                torch.tensor([False, True, False, True]),
            )
        )
        with self.assertRaises(ValueError):
            _explicit_write_local_mask(query_positions, torch.tensor([0, 3]))

    def test_signed_attention_matmul_rejects_unsigned_operands(self) -> None:
        lhs = torch.tensor([[[[127, -127, 3]]]], dtype=torch.int8)
        rhs = torch.tensor([[[[127], [-127], [-7]]]], dtype=torch.int8)
        actual = spinquant_model.int8_batched_matmul(lhs, rhs)
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(actual.item(), 32237)
        with self.assertRaisesRegex(TypeError, "signed int8"):
            spinquant_model.int8_batched_matmul(lhs.to(torch.uint8), rhs)

    def test_spinquant_linear_uses_state_driven_a4_a8_rows(self) -> None:
        context = RowPrecisionContext()
        context.activate(torch.tensor([4, 8], dtype=torch.int8))
        weight = SpinQuantW4Tensor(
            codes=torch.tensor([[7, -8, 3, 1]], dtype=torch.int8),
            scale_bf16=torch.tensor([0.25]),
        )
        module = SpinQuantW4A8Linear(
            weight, workspace=LinearNumericWorkspace(), row_precision_context=context
        )
        values = torch.tensor(
            [[0.11, -0.37, 0.83, 1.25], [0.11, -0.37, 0.83, 1.25]], dtype=torch.bfloat16
        )
        actual = module(values)
        self.assertEqual(tuple(actual.shape), (2, 1))
        self.assertNotEqual(float(actual[0, 0]), float(actual[1, 0]))

    def test_spinquant_linear_applies_deployment_bf16_scales(self) -> None:
        codes = torch.tensor([[7, -8, 3, 1]], dtype=torch.int8)
        values = torch.tensor([[0.11, -0.37, 0.83, 1.25]], dtype=torch.bfloat16)
        outputs = []
        for scale in (0.25, 0.5):
            module = SpinQuantW4A8Linear(
                SpinQuantW4Tensor(codes, torch.tensor([scale], dtype=torch.bfloat16)),
                workspace=LinearNumericWorkspace(),
            )
            outputs.append(module(values))
        self.assertTrue(
            torch.equal(outputs[1], (outputs[0].float() * 2.0).to(torch.bfloat16))
        )
        for scale in (0.0, float("nan"), 0.1):
            module = SpinQuantW4A8Linear(
                SpinQuantW4Tensor(codes, torch.tensor([scale], dtype=torch.float32)),
                workspace=LinearNumericWorkspace(),
            )
            with self.assertRaisesRegex(ValueError, "finite positive BF16-materialized"):
                module(values)

    def test_grouped_w4_scales_are_rejected(self) -> None:
        weight = SpinQuantW4Tensor(torch.zeros((2, 128), dtype=torch.int8), torch.ones((2, 1)))
        with self.assertRaisesRegex(ValueError, "weight scale"):
            SpinQuantW4A8Linear(weight, workspace=LinearNumericWorkspace())

    def test_spinquant_w8_quantization_and_mixed_activation_rows(self) -> None:
        weight = quantize_symmetric_w8(
            torch.tensor(
                [[-3.5, -0.2, 1.0, 4.0], [0.0, 0.5, -1.5, 2.0]], dtype=torch.bfloat16
            )
        )
        self.assertEqual(weight.codes.dtype, torch.int8)
        self.assertGreaterEqual(int(weight.codes.min()), -127)
        self.assertLessEqual(int(weight.codes.max()), 127)
        context = RowPrecisionContext()
        context.activate(torch.tensor([4, 8], dtype=torch.int8))
        module = SpinQuantW4A8Linear(
            weight,
            workspace=LinearNumericWorkspace(),
            row_precision_context=context,
            weight_bits=8,
        )
        values = torch.tensor(
            [[0.11, -0.37, 0.83, 1.25], [0.11, -0.37, 0.83, 1.25]], dtype=torch.bfloat16
        )
        actual = module(values)
        self.assertEqual(tuple(actual.shape), (2, 2))
        self.assertFalse(torch.equal(actual[0], actual[1]))

    def test_target_r4_only_does_not_modify_rope(self) -> None:
        class Rotary(nn.Module):
            def forward(self, q, k, block_end_index=None):
                return (q, k)

        block = nn.Module()
        block.rotary_emb = Rotary()
        original_rotary = block.rotary_emb.forward.__func__
        block.ff_out = nn.Identity()
        _install_target_r4(block)
        self.assertIs(block.rotary_emb.forward.__func__, original_rotary)
        self.assertEqual(len(block.ff_out._forward_pre_hooks), 1)

    def test_k8_cache_codec_is_per_token_and_bf16_bounded(self) -> None:
        values = torch.tensor(
            [[[[1.0, -2.0, 4.0], [0.25, -0.5, 1.0]]]], dtype=torch.bfloat16
        )
        codec = SpinQuantK8CacheCodec()
        (codes, scales) = codec.encode(values)
        restored = codec.decode(codes, scales)
        self.assertEqual(codes.shape, values.shape)
        self.assertEqual(codes.dtype, torch.int8)
        self.assertEqual(scales.shape, (1, 1, 2, 1))
        self.assertEqual(scales.dtype, torch.float32)
        self.assertNotEqual(float(scales[0, 0, 0, 0]), float(scales[0, 0, 1, 0]))
        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertLessEqual(
            float((restored.float() - values.float()).abs().max()), 4.0 / 127.0
        )

    def test_k8_cache_codec_rejects_wrong_scale_axis(self) -> None:
        codes = torch.zeros((1, 2, 3, 4), dtype=torch.int8)
        with self.assertRaises(ValueError):
            SpinQuantK8CacheCodec().decode(codes, torch.ones((1, 2, 1, 1)))

    def test_dequant_bf16_uses_persisted_codes_and_row_scales(self) -> None:
        weight = SpinQuantW4Tensor(
            codes=torch.tensor([[-8, 0, 7], [2, -3, 4]], dtype=torch.int8),
            scale_bf16=torch.tensor([0.5, 2.0], dtype=torch.float32),
        )
        actual = dequantize_spinquant_w4_bf16(weight)
        expected = torch.tensor(
            [[-4.0, 0.0, 3.5], [4.0, -6.0, 8.0]], dtype=torch.bfloat16
        )
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_dequant_bf16_rejects_group_scales(self) -> None:
        weight = SpinQuantW4Tensor(
            codes=torch.tensor([[-8, 0, 7, 2], [2, -3, 4, -6]], dtype=torch.int8),
            scale_bf16=torch.tensor([[0.5, 2.0], [0.25, 0.75]], dtype=torch.float32),
            scale_mode="per_output_channel_per_input_group",
        )
        with self.assertRaisesRegex(ValueError, "scale tensor"):
            dequantize_spinquant_w4_bf16(weight)

    def test_dequant_bf16_rejects_out_of_range_codes(self) -> None:
        weight = SpinQuantW4Tensor(
            codes=torch.tensor([[-9, 0]], dtype=torch.int8),
            scale_bf16=torch.tensor([1.0], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            dequantize_spinquant_w4_bf16(weight)

    def test_v8_codec_uses_static_per_head_scales(self) -> None:
        codec = SpinQuantV8CacheCodec(torch.tensor([0.5, 0.25]))
        values = torch.tensor([[[[1.0, -2.0]], [[0.5, -0.75]]]], dtype=torch.bfloat16)
        codes = codec.encode(values)
        self.assertTrue(
            torch.equal(codes, torch.tensor([[[[2, -4]], [[2, -3]]]], dtype=torch.int8))
        )
        self.assertTrue(torch.equal(codec.decode(codes), values))
        self.assertTrue(
            torch.equal(
                codec.expanded_scales(4, torch.device("cpu")),
                torch.tensor([0.5, 0.5, 0.25, 0.25]),
            )
        )

    def test_p8_v8_context_matches_explicit_integer_matmul(self) -> None:
        block = nn.Module()
        codec = SpinQuantV8CacheCodec(torch.tensor([0.25, 0.5]))
        workspace = Int8MatmulWorkspace()
        _install_native_attention_numeric(
            block,
            workspace=workspace,
            qk_int8=False,
            softmax_lut=False,
            probability_p8=False,
            v8_codec=codec,
        )
        probabilities = torch.tensor(
            [[[[0.2, 0.8]], [[0.6, 0.4]], [[0.3, 0.7]], [[0.9, 0.1]]]],
            dtype=torch.float32,
        )
        v_codes = torch.tensor(
            [[[[2, -1], [4, 3]], [[-2, 1], [3, -4]]]], dtype=torch.int8
        ).repeat_interleave(2, dim=1)
        actual = block._spinquant_context_override(probabilities, v_codes)
        quantized = quantize_per_row_bf16(probabilities.reshape(-1, 2))
        p_codes = quantized.codes.reshape_as(probabilities)
        sums = torch.matmul(p_codes.to(torch.int32), v_codes.to(torch.int32))
        p_scale = quantized.scale_bf16.reshape(1, 4, 1, 1)
        v_scale = torch.tensor([0.25, 0.25, 0.5, 0.5]).view(1, 4, 1, 1)
        expected = bf16(sums.float() * p_scale * v_scale).to(torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
