"""Logical workload export keeps layer and L31 subset semantics explicit."""

import pytest

from evaluation.workload import _phase, _task_family, convert_request


def test_workload_cli_exports_current_schema(tmp_path, monkeypatch):
    import json
    from evaluation import workload

    shard = tmp_path / "evaluation" / "shard_00"
    shard.mkdir(parents=True)
    (shard / "exit_code").write_text("0\n")
    trace = trace_fixture()
    trace["doc_hash"] = "request-7"
    (shard / "trace.jsonl").write_text(json.dumps(trace) + "\n")
    output = tmp_path / "workload.json"
    monkeypatch.setenv("SUPRA_ALGORITHM_ROOT", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["workload", "--root", str(shard.parent),
                                   "--task", "gsm8k", "--output", str(output)])
    workload.main()
    document = json.loads(output.read_text())
    assert document["schema"] == "supra-algorithm-workload/v2"
    assert [event["phase"] for event in document["requests"][0]["events"]] == [
        "full_sequence", "block_initialization"]
    assert _phase({"forward_kind": "full_sequence", "block_index": 2}, trace) == "full_sequence"


def test_workload_cli_reports_missing_artifact_root(monkeypatch, capsys):
    from evaluation import workload

    monkeypatch.delenv("SUPRA_ALGORITHM_ROOT", raising=False)
    monkeypatch.setattr("sys.argv", ["workload", "--root", "traces",
                                   "--task", "gsm8k", "--output", "workload.json"])
    with pytest.raises(SystemExit) as error:
        workload.main()
    assert error.value.code == 2
    assert "set SUPRA_ALGORITHM_ROOT" in capsys.readouterr().err


def test_future_age_advances_on_head_observation():
    from evaluation.workload import _PrecisionReplay

    trace = trace_fixture()
    replay = _PrecisionReplay(trace)
    replay.advance(trace["trace"][0])
    event = {"block_index": 0, "refresh_positions": [800]}
    for _ in range(4):
        replay.advance(event)
    assert replay.age[32] == 0 and replay._bits(800) == 8


def trace_fixture():
    return {
        "doc_id": 7,
        "task_name": "gsm8k",
        "prompt_token_count": 768,
        "generated_token_ids": [[1] * 256],
        "nfe": 2,
        "generation_protocol": {"block_length": 32},
        "feature1_parameters": {"cross_block_full_prefix_oracle_block": 1},
        "trace": [
            {
                "forward_kind": "full_sequence",
                "block_index": 0,
                "input_start": 0,
                "input_length": 1024,
                "input_positions": list(range(1024)),
                "refresh_positions": list(range(1024)),
                "prediction_positions": list(range(768, 800)),
                "next_progress_positions": [800],
                "next_admitted_positions": [800],
                "next_direct_locked_positions": [800],
                "next_tentative_positions": [],
                "cache_sequence_length": 1024,
                "layer0_rows": 1024,
                "layer0_a4_rows": 1024,
                "layer0_a8_rows": 0,
                "active_a4_rows": 0,
                "active_a8_rows": 0,
                "a4_rows": 32,
                "a8_rows": 0,
            },
            {
                "forward_kind": "boundary_refresh",
                "block_index": 1,
                "input_positions": list(range(480, 800)) + list(range(800, 832)),
                "refresh_positions": list(range(480, 800)) + list(range(800, 832)),
                "prediction_positions": list(range(801, 832)),
                "next_progress_positions": [],
                "cache_sequence_length": 1024,
                "layer0_rows": 1024,
                "layer0_a4_rows": 0,
                "layer0_a8_rows": 1024,
                "active_a4_rows": 352,
                "active_a8_rows": 0,
                "boundary_deep_bits": 4,
                "a4_rows": 31,
                "a8_rows": 1,
            },
        ],
    }


@pytest.mark.parametrize("keep_global", [False, True])
def test_boundary_workload_preserves_cache_commit_setting(keep_global):
    trace = trace_fixture()
    trace["trace"][1]["layer0_keep_global_cache"] = keep_global
    event = convert_request(trace)["events"][1]
    assert event["layer0_keep_global_cache"] is keep_global
    assert event["segments"][0]["kv_write_tokens"] == 1024


def test_export_splits_block_initialization_and_derives_retained_hidden():
    result = convert_request(trace_fixture())
    assert result["request_id"] == "shard_00:7"
    assert result["generated_tokens"] == 256
    full_sequence, block_initialization = result["events"]
    assert full_sequence["phase"] == "full_sequence"
    assert full_sequence["segments"][0]["layer_count"] == 31
    assert full_sequence["segments"][0]["kv_write_tokens"] == 1024
    assert full_sequence["segments"][1]["l31_output_tokens"] == 33
    assert full_sequence["segments"][1]["l31_output_a4_tokens"] == 33
    assert block_initialization["phase"] == "block_initialization"
    assert [segment["layer_count"] for segment in block_initialization["segments"]] == [1, 30, 1]
    assert [segment["kv_write_tokens"] for segment in block_initialization["segments"]] == [1024, 352, 352]
    assert block_initialization["prediction_tokens"] == 31
    assert block_initialization["retained_hidden_positions"] == [800]
    assert block_initialization["segments"][-1]["l31_output_tokens"] == 32
    assert block_initialization["segments"][-1]["l31_output_a4_tokens"] == 32
    assert block_initialization["segments"][-1]["l31_output_a8_tokens"] == 0
    assert block_initialization["lm_head_tokens"] == 31


def test_default_full_sequence_mixed_precision_tracks_l31_output_positions():
    trace = trace_fixture()
    full_sequence = trace["trace"][0]
    full_sequence.update(layer0_a4_rows=32, layer0_a8_rows=992,
                cache_initialization_activation_policy="default")
    converted = convert_request(trace)["events"][0]
    assert converted["segments"][0]["a4_tokens"] == 32
    assert converted["segments"][0]["a8_tokens"] == 992
    assert converted["segments"][-1]["l31_output_a4_tokens"] == 32
    assert converted["segments"][-1]["l31_output_a8_tokens"] == 1


def test_default_full_sequence_mixed_precision_rejects_inconsistent_counts():
    trace = trace_fixture()
    full_sequence = trace["trace"][0]
    full_sequence.update(layer0_a4_rows=31, layer0_a8_rows=993,
                cache_initialization_activation_policy="default")
    with pytest.raises(ValueError, match="mixed full-sequence precision differs"):
        convert_request(trace)


def test_full_a8_block_initialization_uses_executed_deep_precision():
    trace = trace_fixture()
    block_initialization = trace["trace"][1]
    block_initialization.update(input_positions=list(range(1024)),
                     refresh_positions=list(range(1024)),
                     active_a4_rows=0, active_a8_rows=1024,
                     boundary_deep_bits=0)
    converted = convert_request(trace)["events"][1]
    assert converted["segments"][-1]["query_tokens"] == 1024
    assert converted["segments"][-1]["l31_output_tokens"] == 32
    assert converted["segments"][-1]["l31_output_a4_tokens"] == 0
    assert converted["segments"][-1]["l31_output_a8_tokens"] == 32


def test_future_predictions_enter_l31_output_and_lm_head():
    trace = trace_fixture()
    event = trace["trace"][1]
    event["input_positions"].append(832)
    event["refresh_positions"].append(832)
    event["next_progress_positions"] = [832]
    event["active_a4_rows"] += 1
    converted = convert_request(trace)["events"][1]
    assert converted["future_prediction_positions"] == [832]
    assert converted["lm_head_tokens"] == 32
    assert converted["segments"][-1]["l31_output_tokens"] == 33


def test_partial_persistent_kv_write_is_exported_for_every_regular_layer():
    trace = trace_fixture()
    event = trace["trace"][0]
    event["forward_kind"] = "local_block"
    event["input_positions"] = list(range(768, 800))
    event["refresh_positions"] = list(range(768, 799))
    event["input_start"] = 768
    event["input_length"] = 32
    event["prediction_positions"] = list(range(768, 800))
    event["next_progress_positions"] = []
    event["next_admitted_positions"] = []
    event["next_direct_locked_positions"] = []
    event["layer0_rows"] = 32
    event["layer0_a4_rows"] = 32
    event["active_a4_rows"] = 32
    event["a4_rows"] = 32
    event["dependency_score_by_row"] = [0.0] * 32
    event["region_start"] = 768
    event["region_end"] = 800
    trace["trace"] = [event]
    trace["nfe"] = 1

    converted = convert_request(trace)["events"][0]
    assert [segment["kv_write_tokens"] for segment in converted["segments"]] == [31, 31]


def test_rejects_kv_write_for_nonqueried_position():
    trace = trace_fixture()
    trace["trace"][0]["refresh_positions"].append(1024)
    with pytest.raises(ValueError, match="not executed query"):
        convert_request(trace)


def test_rejects_nonexecuted_l31_consumer():
    trace = trace_fixture()
    trace["trace"][1]["next_progress_positions"] = [900]
    with pytest.raises(ValueError, match="executed query"):
        convert_request(trace)


def test_native_task_names_map_to_task_family():
    assert _task_family("gsm8k_native") == "gsm8k"
    with pytest.raises(ValueError, match="unsupported task"):
        _task_family("unknown")


@pytest.mark.parametrize("kind", ["local_confirmation", "local_forced_finish"])
def test_local_completion_forwards_are_explicit_repairs(kind):
    trace = trace_fixture()
    assert _phase({"forward_kind": kind}, trace) == "repair"


def test_confirmation_repair_keeps_unpredicted_masked_queries_a4():
    from evaluation.workload import _PrecisionReplay, TENTATIVE, LOCKED

    trace = trace_fixture()
    trace['feature2_precision_parameters'] = {'context_a8_rows': 12}
    replay = _PrecisionReplay(trace)
    replay.state[:18] = [TENTATIVE] * 18
    replay.state[29:32] = [LOCKED] * 3
    replay.age[29:32] = [0] * 3
    query = list(range(768, 797))
    prediction = query[:18]
    event = dict(block_index=0, a4_rows=11, a8_rows=21,
                 active_a4_rows=11, active_a8_rows=18,
                 dependency_score_by_row=[0.] * 32, region_start=768, region_end=800)
    assert replay.output_counts(event, 'repair', query, (prediction, [], query[18:])) == (11, 18)


def test_repair_output_precision_is_not_implied_by_query_totals():
    from evaluation.workload import _PrecisionReplay, MASKED, TENTATIVE, LOCKED

    trace = trace_fixture()
    trace['feature2_precision_parameters'] = {'context_a8_rows': 1}
    replay = _PrecisionReplay(trace)
    replay.state[:] = [LOCKED] * len(replay.state)
    replay.age[:] = [0] * len(replay.age)
    replay.state[:2] = [MASKED, TENTATIVE]
    scores = [0.] * 800
    scores[768], scores[0] = 1., 0.5
    event = dict(block_index=0, a4_rows=1, a8_rows=31,
                 active_a4_rows=2, active_a8_rows=2,
                 dependency_score_by_row=scores, region_start=0, region_end=800)
    # Treating unpredicted MASKED as context would upgrade768 instead of0.
    # Query totals would still match, but the L31 output split would be wrong.
    assert replay.output_counts(event, 'repair', [0, 1, 768, 769],
                                ([769], [], [768])) == (1, 1)


def test_explicit_future_drafts_require_confirmation_before_locking():
    from evaluation.workload import _PrecisionReplay, TENTATIVE, LOCKED, MASKED

    trace = trace_fixture()
    trace['feature3_dynamic_block_parameters'] = {'canonical_future': True}
    for event in trace['trace']:
        event.update(next_direct_locked_positions=[], next_tentative_positions=[])
    first = trace['trace'][0]
    first.update(next_progress_positions=[800, 801, 802],
                 next_admitted_positions=[800, 801, 802],
                 next_direct_locked_positions=[800], next_tentative_positions=[801, 802])
    replay = _PrecisionReplay(trace)
    replay.advance(first)
    assert replay.state[32:35] == [LOCKED, TENTATIVE, TENTATIVE]
    assert replay.age[32:35] == [0, -1, -1]
    second = dict(block_index=0, next_progress_positions=[801, 802],
                  next_admitted_positions=[], next_direct_locked_positions=[],
                  next_tentative_positions=[], next_confirmed_positions=[801],
                  next_remasked_positions=[802])
    replay.advance(second)
    assert replay.state[32:35] == [LOCKED, LOCKED, MASKED]
    assert replay.age[32:35] == [0, 0, -1]


def test_future_state_export_rejects_unclassified_admission():
    from evaluation.workload import _PrecisionReplay

    trace = trace_fixture()
    trace['feature3_dynamic_block_parameters'] = {'canonical_future': True}
    for event in trace['trace']:
        event.update(next_direct_locked_positions=[], next_tentative_positions=[])
    replay = _PrecisionReplay(trace)
    with pytest.raises(ValueError, match='partition admitted'):
        replay.advance(trace['trace'][0])


def test_future_admissions_require_recorded_state_classification():
    from evaluation.workload import _PrecisionReplay

    trace = trace_fixture()
    del trace['trace'][0]['next_direct_locked_positions']
    replay = _PrecisionReplay(trace)
    with pytest.raises(ValueError, match='partition admitted'):
        replay.advance(trace['trace'][0])
