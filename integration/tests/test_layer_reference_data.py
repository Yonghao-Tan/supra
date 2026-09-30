"""CPU verification for the independently runnable hardware-alignment kit."""

from unittest.mock import patch
from pathlib import Path
import json

import numpy as np
import pytest
import torch

import layer_reference_data
from quantization import model as spinquant_model
from numerics.int8_matmul import Int8MatmulWorkspace
from model.modeling_llada import ActivationType, BlockType, BufferCache, LLaDALlamaBlock, ModelConfig
from layer_reference_data import SCHEMA, compare_arrays, compare_reference_data, export_reference_data, make_control_boundaries, make_reference_data, make_multilayer_reference_data, raw_array
from quantization.model import SpinQuantV8CacheCodec, _install_native_attention_numeric
from numerics.bf16 import bf16, softmax_lut_bf16


def test_control_loop_keeps_count_token_ids_positions_and_hidden_distinct():
    import struct
    descriptor = bytearray(64)
    descriptor[0] = 2
    for index, (token, position, bits) in enumerate(((17, 93, 8), (4, 21, 4))):
        struct.pack_into("<IH", descriptor, 32+16*index, token, position)
        descriptor[42+16*index] = bits
    hidden = torch.arange(8192).reshape(1, 2, 4096).bfloat16()
    with patch.object(layer_reference_data, "make_reference_data", return_value=({}, {"block_output": hidden}, {})) as layer,\
         patch.object(layer_reference_data, "make_forward_postprocess_reference_data", return_value=({}, {}, {})) as head:
        inputs, _, metadata = layer_reference_data.make_control_loop_reference_data(bytes(descriptor))
    assert layer.call_args.kwargs["tokens"] == head.call_args.kwargs["tokens"] == 2
    assert layer.call_args.kwargs["_positions"].tolist() == [93, 21]
    assert layer.call_args.kwargs["activation_bits"] == [8, 4]
    assert inputs["tokens"].tolist() == head.call_args.kwargs["_selected_tokens"].tolist() == [17, 4]
    assert torch.equal(layer.call_args.kwargs["_hidden_input"], inputs["embedding"][[17, 4]][None])
    assert torch.equal(head.call_args.kwargs["_hidden_input"], hidden.reshape(2, 4096))
    for invalid in (b"", bytes([0]), bytes([49]), bytes(descriptor[:63])):
        with pytest.raises(ValueError, match="complete one-round"):
            layer_reference_data.make_control_loop_reference_data(invalid)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_qk_consumes_cache_once_after_rope_dense_and_rewrite(kv_heads):
    config = ModelConfig(d_model=256, n_heads=2, n_kv_heads=kv_heads, n_layers=1,
                         mlp_hidden_size=256, activation_type=ActivationType.silu,
                         block_type=BlockType.llama, max_sequence_length=8,
                         include_bias=False, rope=True, use_manual_attention=True,
                         attention_dropout=0.0, residual_dropout=0.0)
    block = LLaDALlamaBlock(0, config, BufferCache()).bfloat16().eval()
    block.reset_parameters()
    _install_native_attention_numeric(
        block, workspace=Int8MatmulWorkspace(), qk_int8=True, softmax_lut=True,
        probability_p8=False, k8_cache=True, rope_before_k8=True,
        v8_codec=SpinQuantV8CacheCodec(torch.full((kv_heads,), 0.02)),
    )
    score = block._spinquant_score_override
    consumed = []

    def record(q, k, **kwargs):
        consumed.append((kwargs["key_codes"].clone(), kwargs["key_scales"].clone()))
        return score(q, k, **kwargs)

    block._spinquant_score_override = record
    generator = torch.Generator().manual_seed(92)
    with torch.inference_mode(), patch.object(
        spinquant_model, "quantize_per_row_bf16", wraps=spinquant_model.quantize_per_row_bf16
    ) as quantize:
        _, cache = block(torch.randn(1, 5, 256, generator=generator).bfloat16(), use_cache=True)
        assert quantize.call_count == 3  # New K, Q, P. No decoded-K requantization.
        assert torch.equal(consumed[-1][0], cache[0].repeat_interleave(2 // kv_heads, dim=1))
        # A legal external cache need not have a saturated maximum code.
        cache[0][:, :, 0] = 3
        cache[1][:, :, 0] = 0.03125
        before = tuple(t.clone() for t in cache)
        quantize.reset_mock()
        positions = torch.tensor([[4, 1]])
        _, updated = block(torch.randn(1, 2, 256, generator=generator).bfloat16(),
                           layer_past=cache, use_cache=True,
                           query_position_ids=positions, kv_write_position_ids=positions)
        assert quantize.call_count == 3
    assert torch.equal(consumed[-1][0], updated[0].repeat_interleave(2 // kv_heads, dim=1))
    assert torch.equal(consumed[-1][1], updated[1].repeat_interleave(2 // kv_heads, dim=1))
    for new, old in zip(updated, before):
        assert torch.equal(new[:, :, [0, 2, 3]], old[:, :, [0, 2, 3]])
    assert torch.all(consumed[-1][0][:, :, 0] == 3)


def test_reference_data_uses_actual_r4_and_preserves_cache_and_seed():
    torch.set_num_threads(2)
    inputs, expected, metadata = make_reference_data(hidden=128)
    again_inputs, again, _ = make_reference_data(hidden=128)
    assert metadata["ffn"] == 12288
    assert metadata["standard_layer"] is False
    assert len(expected) == 61
    for name, value in expected.items():
        assert compare_arrays(raw_array(value), raw_array(again[name]))["match"]
    assert torch.equal(inputs["weight.ff_out.codes"], again_inputs["weight.ff_out.codes"])
    assert not torch.equal(expected["product_before_r4"], expected["ff_out.input"])
    assert torch.equal(expected["qk_key_codes"], expected["cache_key_codes"])
    positions = inputs["positions"]
    assert torch.equal(expected["cache_key_codes"][:, :, positions], expected["new_key_codes"])
    unchanged = torch.ones(metadata["sequence"], dtype=torch.bool)
    unchanged[positions] = False
    for suffix, source in (("key_codes", "retained_key_codes"),
                           ("key_scale", "retained_key_scale"),
                           ("value_codes", "retained_value_codes")):
        assert torch.equal(expected[f"cache_{suffix}"][:, :, unchanged], inputs[source][:, :, unchanged])
    partial_key = inputs["retained_key_codes"].clone()
    partial_scale = inputs["retained_key_scale"].clone()
    partial_key[:, :, positions[:1]] = expected["new_key_codes"][:, :, :1]
    partial_scale[:, :, positions[:1]] = expected["new_key_scale"][:, :, :1]
    q = expected["query_codes"][:, :, :1]
    accum = q.int() @ partial_key.int().transpose(-2, -1)
    partition_scores = bf16(accum.float() * expected["query_scale"][:, :, :1]
                            * partial_scale.transpose(-2, -1) * bf16(128 ** -0.5))
    assert not torch.equal(partition_scores, expected["scores"][:, :, :1])


def test_multilayer_reference_data_uses_distinct_layers_and_real_output_chain():
    torch.set_num_threads(2)
    inputs, expected, metadata = make_multilayer_reference_data(
        layers=2, hidden=128, tokens=3, sequence=5, activation_bits=[4, 4, 4])
    assert metadata["layers"] == 2
    assert [entry["expected_checkpoints"] for entry in metadata["layer_metadata"]] == [61, 61]
    assert len(expected) == 122
    assert torch.equal(inputs["layer1.hidden"], expected["layer0.block_output"])
    assert not torch.equal(inputs["layer0.weight.q_proj.codes"],
                           inputs["layer1.weight.q_proj.codes"])
    assert "layer0.cache_key_codes" in expected
    assert "layer1.cache_key_codes" in expected


def test_reference_data_separates_query_visible_writes_and_persistent_commit():
    torch.set_num_threads(2)
    original_inputs, original, _ = make_reference_data(hidden=128)
    inputs, scout, meta = make_reference_data(hidden=128, cache_commit_token_indices=[1])
    assert meta["cache_commit_token_indices"] == [1]
    for name in ("scores", "context", "block_output", "qk_key_codes", "pv_value_codes"):
        assert np.array_equal(raw_array(scout[name]), raw_array(original[name]))
    inputs, partial, meta = make_reference_data(hidden=128, kv_write_token_indices=[2, 0], cache_commit_token_indices=[2])
    assert meta["kv_write_token_indices"] == [0, 2]
    assert not torch.equal(partial["scores"], original["scores"])
    for actual, commits in ((scout, [1]), (partial, [2])):
        keep_old = torch.ones(17, dtype=torch.bool)
        keep_old[inputs["positions"][commits]] = False
        for suffix in ("key_codes", "key_scale", "value_codes"):
            assert np.array_equal(raw_array(actual[f"cache_{suffix}"][:, :, keep_old]),
                                  raw_array(original_inputs[f"retained_{suffix}"][:, :, keep_old]))
            assert torch.equal(actual[f"cache_{suffix}"][:, :, inputs["positions"][commits]],
                               actual[f"new_{suffix}"][:, :, commits])
    excluded_key = inputs["positions"][1]
    assert torch.equal(partial["qk_key_codes"][:, :, excluded_key],
                       inputs["retained_key_codes"][:, :, excluded_key])
    assert torch.equal(partial["qk_key_scale"][:, :, excluded_key],
                       inputs["retained_key_scale"][:, :, excluded_key])
    assert torch.equal(partial["pv_value_codes"][:, :, excluded_key],
                       inputs["retained_value_codes"][:, :, excluded_key])
    _, read_only, _ = make_reference_data(hidden=128, kv_write_token_indices=[], cache_commit_token_indices=[])
    assert torch.equal(read_only["qk_key_codes"], original_inputs["retained_key_codes"])


@pytest.mark.parametrize("transition", [None, [1, 4, 16]])
@pytest.mark.parametrize("deep_bits", [None, 4])
def test_boundary_scout_restores_rejected_cache_and_gathers_actual_hidden(transition, deep_bits, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    first, second = layer_reference_data.make_boundary_reference_data(
        hidden=128, sequence=17, current_positions=[5, 6, 7, 8], target_token_count=12,
        mandatory_candidate_positions=[0], transition_positions=transition,
        shortlist_pending_raw=[0] * 17,
        deep_activation_bits=deep_bits,
        activation_bits=[4 if 5 <= token <= 8 else 8 for token in range(17)])
    inputs, actual, meta = first
    selected = torch.tensor(meta["boundary_scout"]["deep_positions"])
    assert len(selected) == 12 and set([0, 5, 6, 7, 8]) <= set(selected.tolist())
    assert torch.equal(second[0]["hidden"], actual["block_output"][:, selected])
    assert torch.equal(second[0]["positions"], selected)
    expected_bits = (inputs["activation_bits"][selected] if deep_bits is None else
                     torch.full_like(inputs["activation_bits"][selected], deep_bits))
    assert torch.equal(second[0]["activation_bits"], expected_bits)
    assert second[2]["model_layer_index"] == 1
    rejected = torch.tensor([p for p in range(17) if p not in selected])
    for suffix in ("key_codes", "key_scale", "value_codes"):
        assert torch.equal(actual[f"cache_{suffix}"][:, :, rejected],
                           inputs[f"retained_{suffix}"][:, :, rejected])
        assert torch.equal(actual[f"cache_{suffix}"][:, :, selected],
                           actual[f"new_{suffix}"][:, :, selected])
    assert torch.equal(actual["qk_key_codes"], actual["new_key_codes"])
    assert not torch.equal(actual["qk_key_codes"][:, :, rejected], actual["cache_key_codes"][:, :, rejected])
    from hardware_adapter.control_reference_data import generated_boundary_reference
    monkeypatch.setattr(layer_reference_data, "ALGO_ROOT", tmp_path / "payloads")
    monkeypatch.setattr(layer_reference_data, "PROJECT_ROOT", tmp_path / "source")
    index, following = tmp_path / "source/l0.json", tmp_path / "source/l1.json"
    meta["continuation_index"] = str(following)
    export_reference_data(*first, tmp_path / "payloads/l0", index)
    export_reference_data(*second, tmp_path / "payloads/l1", following)
    if deep_bits is None:
        record = generated_boundary_reference(index)["records"][0]
        assert record["expected"]["deep_positions"]["raw"] == selected.tolist()
        assert record["expected"]["deep_bits"]["raw"] == expected_bits.tolist()
    else:
        with pytest.raises(ValueError, match="actual deep input"):
            generated_boundary_reference(index)


@pytest.mark.parametrize("options", [
    {"kv_write_token_indices": [0, 0]}, {"kv_write_token_indices": [-1]},
    {"cache_commit_token_indices": [3]}, {"kv_write_token_indices": [0], "cache_commit_token_indices": [1]},
])
def test_reference_data_rejects_invalid_cache_sets(options):
    with pytest.raises(ValueError, match="unique local|subset"):
        make_reference_data(hidden=128, **options)


def test_raw_checker_detects_signed_zero_nan_payload_shape_and_integer_changes():
    expected = np.array([0x0000, 0x8000, 0x7fc1, 0x7fc2], dtype="<u2")
    assert compare_arrays(expected, expected.copy())["match"]
    actual = expected.copy()
    actual[1] = 0
    assert compare_arrays(expected, actual)["first_index"] == [1]
    assert json.loads(json.dumps(compare_arrays(expected, actual)))["first_index"] == [1]
    actual = expected.copy()
    actual[3] = 0x7fc1
    assert not compare_arrays(expected, actual)["match"]
    assert not compare_arrays(expected, actual.reshape(2, 2))["match"]
    tensor = torch.from_numpy(expected.view(np.int16)).view(torch.bfloat16)
    assert np.array_equal(raw_array(tensor), expected)
    assert not compare_arrays(np.array([1], dtype="<i4"), np.array([2], dtype="<i4"))["match"]


def test_lut_softmax_excludes_masked_keys_before_reduction():
    scores = torch.tensor([[0.0, -float("inf"), -1.0, -float("inf")]])
    actual = softmax_lut_bf16(scores)
    assert torch.equal(actual[:, [1, 3]], torch.zeros(1, 2))
    assert torch.equal(actual[:, [0, 2]], softmax_lut_bf16(scores[:, [0, 2]]))
    with pytest.raises(ValueError, match="at least one valid key"):
        softmax_lut_bf16(torch.full((1, 3), -float("inf")))


def test_export_and_compare_reject_missing_truncated_and_changed_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(layer_reference_data, "ALGO_ROOT", tmp_path / "payload")
    monkeypatch.setattr(layer_reference_data, "PROJECT_ROOT", tmp_path / "project")
    output, index = tmp_path / "payload" / "case", tmp_path / "project" / "case.json"
    expected = {"block_output": torch.tensor([[1.0, -0.0]], dtype=torch.bfloat16),
                "q_proj.output": torch.tensor([[2.0]], dtype=torch.bfloat16)}
    export_reference_data({}, expected, {"schema": SCHEMA}, output, index)
    actual = tmp_path / "actual"
    actual.mkdir()
    assert not compare_reference_data(index, actual)["match"]
    path = actual / "block_output.bin"
    payload = (output / "expected.block_output.bin").read_bytes()
    path.write_bytes(payload)
    assert not compare_reference_data(index, actual)["match"]
    selected = compare_reference_data(index, actual, ["block_output"])
    assert selected["match"] and selected["scope"] == "selected_checkpoints"
    (actual / "q_proj.output.bin").write_bytes((output / "expected.q_proj.output.bin").read_bytes())
    assert compare_reference_data(index, actual)["match"]
    assert all(not Path(item["path"]).is_absolute() for item in json.loads(index.read_text())["tensors"])
    moved = tmp_path / "moved"
    moved.mkdir()
    (tmp_path / "payload").rename(moved / "payload")
    index.parent.rename(moved / "project")
    actual.rename(moved / "actual")
    index, actual = moved / "project/case.json", moved / "actual"
    output, path = moved / "payload/case", actual / "block_output.bin"
    monkeypatch.setattr(layer_reference_data, "ALGO_ROOT", moved / "payload")
    monkeypatch.setattr(layer_reference_data, "PROJECT_ROOT", moved / "project")
    assert compare_reference_data(index, actual)["match"]
    with pytest.raises(ValueError, match="absent"):
        compare_reference_data(index, actual, ["not_a_checkpoint"])
    path.write_bytes(payload[:-1])
    assert not compare_reference_data(index, actual)["match"]
    path.write_bytes(payload[:2] + b"\x00\x00")
    assert not compare_reference_data(index, actual)["match"]
    with pytest.raises(FileExistsError):
        export_reference_data({}, expected, {"schema": SCHEMA}, output, index)


def test_candidate_cpu_dispatch_preserves_cuda_dtype_without_changing_raw_values():
    from numerics.candidate import streaming_candidate_bf16, streaming_candidate_bf16_oracle
    logits = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]])
    cpu = streaming_candidate_bf16(logits)
    oracle = streaming_candidate_bf16_oracle(logits)
    assert cpu[1].dtype == cpu[2].dtype == torch.bfloat16
    assert torch.equal(cpu[0], oracle[0])
    assert np.array_equal(raw_array(cpu[1]), raw_array(oracle[1]))
    assert np.array_equal(raw_array(cpu[2]), raw_array(oracle[2]))


def test_real_generation_closeout_snapshots():
    torch.set_num_threads(2)
    result = make_control_boundaries(closeouts_only=True)
    assert result["schema"] == "supra-psme-block-closeouts/v1"
    confirmed, forced = result["draft_verify"]
    for case in (confirmed, forced):
        assert case["config"]["tail_confirmation_policy"] == "none"
        assert case["config"]["tau_high_tail"] is None
        assert not case["config"]["packed_attention_force_full_current"]
    assert confirmed["model_evaluations"] == 2 and forced["model_evaluations"] == 3
    assert confirmed["final_phase"] == "local_confirmation"
    assert confirmed["final_cache_refresh_due_positions"] == []
    assert forced["final_phase"] == "local_forced_finish"
    assert forced["final_cache_refresh_due_positions"] == [1]
    assert forced["final_state"]["state"]["raw"] == [[2]*32]
    assert forced["final_state"]["precision_age"]["raw"] == [[0]+[1]*31]
    assert forced["final_state"]["commit_origin"]["raw"] == [[0]+[3]*31]
    confirmation = forced["captures"][1]["confirmation"]
    assert confirmation["remasked"]["raw"] == [[True]+[False]*31]
    assert forced["captures"][2]["state_fields_at_forward_start"]["state"]["raw"] == [[0]+[2]*31]
    assert forced["captures"][2]["token_positions"]["raw"] == [1]
    assert forced["captures"][2]["state_fields_at_forward_start"]["row_bits"]["raw"] == [[4]+[8]*31]
    assert forced["final_state"]["commit_origin"] == forced["captures"][2]["state_fields_at_forward_start"]["commit_origin"]
    assert forced["final_state"]["last_top1"] == forced["captures"][2]["state_fields_at_forward_start"]["last_top1"]


def test_real_generation_boundary_and_dynamic_scale_exports():
    result = make_control_boundaries()
    scales = {(case["input_bf16_raw"]["raw"][0][0], case["activation_bits"]): case
              for case in result["dynamic_scale"]}
    assert scales[0x0080, 4]["scale_bf16_raw"] == [0x0012]
    assert scales[0x0080, 8]["scale_bf16_raw"] == [0x0001]
    assert scales[0x0001, 4]["scale_bf16_raw"] == [0x3f80]
    assert scales[0x0001, 8]["scale_bf16_raw"] == [0x3f80]
    for case in result["draft_verify"][:2]:
        first = case["decisions"][0]
        assert first["state_after_admit"]["raw"][0][:3] == [0, 2, 2]
        assert first["high"]["raw"][0][:3] == [False, True, True]
        assert first["direct"]["raw"][0][:3] == [False, True, True]
    confirm, remask = result["draft_verify"][-2:]
    suppressed = result["draft_verify"][-3]
    assert suppressed["decisions"][0]["proposal"]["raw"][0][0] == 3
    assert suppressed["decisions"][0]["action_confidence"]["raw"][0][0] == 0
    assert suppressed["decisions"][0]["direct"]["raw"][0][0] is False
    assert confirm["transitions"][1]["confirmed"] == list(range(1, 9))
    assert remask["transitions"][1]["remasked"] == [1]
    assert 1 not in remask["transitions"][1]["direct"]
    for case in result["draft_verify"]:
        captures = {c["capture_index"]: c for c in case["captures"]}
        assert len(captures) == len(case["captures"])
        for decision in case["decisions"]:
            capture = captures[decision["capture_index"]]
            end = decision["forward_end"]
            assert decision["block_index"] == capture["block_index"] == end["block_index"]
            assert decision["step_index"] == capture["step_index"] == end["step_index"]
            assert decision["prediction_positions"] == capture["prediction_positions"]
            assert decision["decision_positions"]["raw"] == list(range(1, 33))
            assert decision["score"]["dtype"] == decision["action_confidence"]["dtype"]
            assert end["last_top1"]["shape"] == end["precision_age"]["shape"] == [1, 32]
            assert end["commit_origin"]["shape"] == end["next_current_activation_bits"]["shape"] == [1, 32]
            next_captures = [c for c in case["captures"] if c["capture_index"] == decision["capture_index"] + 1]
            if next_captures:
                assert end["state"] == next_captures[0]["state_at_forward_start"]
        for transition in case["transitions"]:
            capture = captures[transition["capture_index"]]
            assert transition["prediction_positions"] == capture["prediction_positions"]
    first, second = result["draft_verify"][1]["decisions"][:2]
    assert first["forward_end"]["last_top1"]["raw"][0][0] == 0
    assert first["forward_end"]["precision_age"]["raw"][0][1] == 0
    assert second["forward_end"]["precision_age"]["raw"][0][1] == 1


@pytest.mark.parametrize("kwargs", [{"tokens": 2049}, {"tokens": 40, "sequence": 40, "a8_token_count": 41},
                                    {"sequence": 2}, {"a8_token_count": -1}])
def test_invalid_layer_shape_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_reference_data(hidden=128, **kwargs)


def test_algorithm_layer_is_not_limited_by_rtl_round_capacity():
    activation_bits = [8, 4, 4] * 16 + [8]
    inputs, expected, metadata = make_reference_data(hidden=128, tokens=49, sequence=51, activation_bits=activation_bits)
    assert metadata["tokens"] == 49
    assert metadata["a4_token_count"] == 32 and metadata["a8_token_count"] == 17
    assert metadata["activation_slice_units"] == 66
    assert inputs["activation_bits"].tolist() == activation_bits
    assert expected["block_output"].shape == (1, 49, 128)


@pytest.mark.parametrize("bits", [[4, 8], [4, 8, 6]])
def test_explicit_precision_rejects_missing_rows_and_invalid_bits(bits):
    with pytest.raises(ValueError, match="activation_bits"):
        make_reference_data(hidden=128, activation_bits=bits)


def test_generated_clipping_records_preserve_actual_inputs_and_fp32(tmp_path, monkeypatch):
    import layer_reference_data as reference
    torch.set_num_threads(2)
    ratios = {name: (0.625 if name in ("attn_out", "ff_out") else 0.80078125)
              for name in reference.LINEARS}
    inputs, expected, metadata = make_reference_data(hidden=128, a4_clip_ratios=ratios)
    for name, ratio in ratios.items():
        assert inputs[name + ".clip.installed"].item()
        assert inputs[name + ".clip.ratio"].item() == ratio
        assert torch.equal(expected[name + ".input"], expected[name + ".clip.output"])
        a8 = inputs["activation_bits"] == 8
        assert torch.equal(expected[name + ".clip.input"][:, a8],
                           expected[name + ".clip.output"][:, a8])
    assert torch.equal(expected["ff_out.r4_output"], expected["ff_out.clip.input"])
    monkeypatch.setattr(reference, "ALGO_ROOT", tmp_path / "payloads")
    monkeypatch.setattr(reference, "PROJECT_ROOT", tmp_path / "source")
    index = tmp_path / "source/index.json"
    export_reference_data(inputs, expected, metadata, tmp_path / "payloads/data", index)
    written = json.loads(index.read_text())
    reference.validate_numeric_capabilities(written)
    for item in written["tensors"]:
        if item["name"].endswith((".clip.row_max", ".clip.limit_fp32")):
            assert item["encoding"] == "float32_raw"
            actual = np.fromfile(layer_reference_data.tensor_path(item, index), dtype="<u4")
            raw = expected[item["name"]].contiguous().view(torch.int32).numpy().view("<u4")
            assert np.array_equal(actual, raw.reshape(-1))
    from prepare_layer_testcase import ReferenceData
    ReferenceData(index)
    for suffix, dtype, bad, message in (
            ("installed", "|b1", False, "installed payload"),
            ("ratio", "<u2", 0x3f00, "ratio payload")):
        entry = next(e for e in written["tensors"] if e["name"] == "q_proj.clip." + suffix)
        path = layer_reference_data.tensor_path(entry, index)
        original = path.read_bytes()
        np.asarray(bad, dtype=dtype).tofile(path)
        with pytest.raises(ValueError, match=message):
            ReferenceData(index)
        path.write_bytes(original)
    written["a4_clip_ratios"]["q_proj"] = 0.5
    index.write_text(json.dumps(written))
    with pytest.raises(ValueError, match="declared A4 clipping differs"):
        ReferenceData(index)


@torch.inference_mode()
def test_capture_replay_block_reconstructs_quantized_layer():
    from capture.layers import replay_block, CACHE_NAMES, LINEARS
    from model.modeling_llada import LayerNormType
    from model_capture import activation_field_names

    inputs, expected, _ = make_reference_data(
        hidden=128, a4_clip_ratios=dict.fromkeys(LINEARS, 0.8))
    inputs = activation_field_names(inputs, to_algorithm=True)
    for name in LINEARS:
        inputs[name + ".row_bits"] = inputs["row_bits"]
    for name in ("attn_norm", "ff_norm"):
        inputs[name + ".weight"] = torch.ones(128, dtype=torch.bfloat16)
    config = dict(d_model=128, n_heads=1, n_kv_heads=1, n_layers=1,
        mlp_hidden_size=12288, activation_type=ActivationType.silu,
        block_type=BlockType.llama, layer_norm_type=LayerNormType.rms,
        include_bias=False, include_qkv_bias=False, rope=True,
        rope_full_precision=True, max_sequence_length=17,
        attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
        use_manual_attention=True, flash_attention=False)
    block = replay_block(config, inputs, device="cpu")
    positions = inputs["positions"][None, :]
    output, cache = block(inputs["hidden"], use_cache=True,
        layer_past=tuple(inputs["retained_" + name].clone() for name in CACHE_NAMES),
        query_position_ids=positions, kv_write_position_ids=positions)
    assert compare_arrays(raw_array(expected["block_output"]), raw_array(output))["match"]
    for name, actual in zip(CACHE_NAMES, cache):
        assert compare_arrays(raw_array(expected["cache_" + name]), raw_array(actual))["match"]
