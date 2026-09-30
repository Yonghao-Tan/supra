"""Benchmark collection must retain scoring and reject incomplete requests."""

import json
from types import SimpleNamespace
import pytest
from evaluation import summary as summary


def test_duplicate_samples_rejected_without_merging_filters():
    strict = dict(doc_id=7, filter="strict-match", exact_match=0)
    flexible = dict(doc_id=7, filter="flexible-extract", exact_match=1)
    for rows in ([strict, flexible], [flexible, strict]):
        assert summary._sample_index(rows, "exact_match", "strict-match")[7] == strict
        with pytest.raises(ValueError, match="Duplicate sample"):
            summary._sample_index([*rows, dict(strict, exact_match=1)],
                                  "exact_match", "strict-match")


def test_multiple_run_files_are_ambiguous(tmp_path):
    first = tmp_path / "results_a.json"
    first.write_text("{}")
    assert summary._unique_file(tmp_path, "results_*.json") == first
    (tmp_path / "results_b.json").write_text("{}")
    with pytest.raises(ValueError, match="Ambiguous"):
        summary._unique_file(tmp_path, "results_*.json")


def case_root(tmp_path, monkeypatch, task="gsm8k"):
    shard = tmp_path / "shard_00"
    shard.mkdir()
    (shard / "exit_code").write_text("0\n")
    result = shard / "results_fixture.json"
    result.write_text(
        json.dumps({"config": {"model_args": {"source_commit": "source"}}})
    )
    sample = {
        "filter": "strict-match",
        "exact_match": 0.0,
        "doc": {"question": "fixture"},
    }
    trace = {
        "doc_hash": "document",
        "nfe": 1,
        "generation_mode": "feature1_packed_feature2_fused_dynamic_block",
        "precision": "W4A4A8",
        "artifact_manifest_sha256": "artifact",
        "lm_head_weight_bits": 8,
        "trace": [
            {
                "forward_kind": "local_block",
                "input_length": 2,
                "input_positions": [3, 4, 8],
                "a4_rows": 2,
                "a8_rows": 1,
                "active_a4_rows": 2,
                "active_a8_rows": 1,
                "layer0_rows": 3,
                "layer0_a4_rows": 2,
                "layer0_a8_rows": 1,
                "masked_after": 0,
            }
        ],
    }
    monkeypatch.setattr(
        summary,
        "load_config",
        lambda *_: (
            {"document": {"sample": sample, "trace": trace}},
            {"shard_00": str(result)},
            {"document": "shard_00"},
        ),
    )
    monkeypatch.setattr(
        summary, "load_config_filter", lambda *_: {"document": {"exact_match": 1.0}}
    )
    return SimpleNamespace(root=tmp_path, trace=trace, sample=sample, shard=shard)


def test_gsm_scores_and_packed_rows_have_their_original_meaning(tmp_path, monkeypatch):
    state = case_root(tmp_path, monkeypatch)
    result = summary.summarize(state.root, "gsm8k", 1, 1, partial=True)
    assert result["quality"] == {"strict_count": 0, "flexible_count": 1}
    assert result["forward_work"]["local_block"]["input_rows"] == 3
    assert result["evaluation_scope"] == "partial"


@pytest.mark.parametrize("feature3", [False, True])
def test_forward_work_uses_executed_rows_and_separates_scout(feature3):
    events = [
        dict(
            forward_kind="full_sequence",
            input_length=1024,
            layer0_rows=1024,
            layer0_a4_rows=32,
            layer0_a8_rows=992,
            active_a4_rows=0,
            active_a8_rows=0,
            a4_rows=32,
            a8_rows=0,
        ),
        dict(
            forward_kind="boundary_refresh",
            input_length=1024,
            input_positions=list(range(80)),
            layer0_rows=1024,
            layer0_a4_rows=32,
            layer0_a8_rows=992,
            active_a4_rows=32,
            active_a8_rows=48,
            a4_rows=32,
            a8_rows=0,
        ),
        dict(
            forward_kind="local_block",
            input_length=32,
            input_positions=list(range(27)),
            layer0_rows=27,
            layer0_a4_rows=19,
            layer0_a8_rows=8,
            active_a4_rows=19,
            active_a8_rows=8,
            a4_rows=23,
            a8_rows=9,
        ),
    ]
    if feature3:
        for event in events:
            event.update(joint_a4_rows=0, joint_a8_rows=0)
        events[-1].update(joint_a4_rows=19, joint_a8_rows=8)
    full_sequence, boundary, regular = [summary.execution_rows(event) for event in events]
    assert (full_sequence["a4_rows"], full_sequence["a8_rows"]) == (32, 992)
    assert (boundary["layer0_rows"], boundary["deep_rows"]) == (1024, 80)
    assert (boundary["deep_a4_rows"], boundary["deep_a8_rows"]) == (32, 48)
    assert regular["a4_rows"] + regular["a8_rows"] == regular["input_rows"] == 27
    events[-1]["active_a4_rows"] = 23
    with pytest.raises(ValueError, match="precision counts"):
        summary.execution_rows(events[-1])


def test_summary_requires_one_global_boundary_layer():
    event = dict(forward_kind="boundary_refresh", input_positions=list(range(320)),
                 layer0_rows=1072, layer0_a4_rows=0, layer0_a8_rows=1072,
                 active_a4_rows=320, active_a8_rows=0, boundary_deep_bits=4, boundary_global_layers=2)
    with pytest.raises(ValueError, match="same set in all deep layers"):
        summary.execution_rows(event)




def test_incomplete_filter_trace_and_shards_are_rejected(tmp_path, monkeypatch):
    state = case_root(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="every benchmark"):
        summary.summarize(state.root, "gsm8k", 1, 1)
    state.trace["nfe"] = 2
    with pytest.raises(ValueError, match="trace/NFE"):
        summary.summarize(state.root, "gsm8k", 1, 1, partial=True)
    state.trace["nfe"] = 1
    state.sample["filter"] = "flexible-extract"
    with pytest.raises(ValueError, match="missing strict-match"):
        summary.summarize(state.root, "gsm8k", 1, 1, partial=True)
    (state.shard / "exit_code").write_text("1\n")
    with pytest.raises(ValueError, match="unfinished"):
        summary.summarize(state.root, "gsm8k", 1, 1, partial=True)


def test_missing_exit_recovery_keeps_completeness_and_failure_checks(tmp_path, monkeypatch):
    state = case_root(tmp_path, monkeypatch)
    (state.shard / "exit_code").unlink()
    with pytest.raises(ValueError, match="unfinished"):
        summary.summarize(state.root, "gsm8k", 1, 1, partial=True)
    kwargs = dict(partial=True, recover_missing_exit_code=True)
    result = summary.summarize(state.root, "gsm8k", 1, 1, **kwargs)
    assert result["quality"] == {"strict_count": 0, "flexible_count": 1}
    assert result["completion_recovery"]["original_process_exit_code"] is None
    assert not (state.shard / "exit_code").exists()
    state.trace["trace"][-1]["masked_after"] = 1
    with pytest.raises(ValueError, match="unfinished generation"):
        summary.summarize(state.root, "gsm8k", 1, 1, **kwargs)
    state.trace["trace"][-1]["masked_after"] = 0
    state.trace["nfe"] = 2
    with pytest.raises(ValueError, match="trace/NFE"):
        summary.summarize(state.root, "gsm8k", 1, 1, **kwargs)
    state.trace["nfe"] = 1
    (state.shard / "exit_code").write_text("1\n")
    with pytest.raises(ValueError, match="unfinished"):
        summary.summarize(state.root, "gsm8k", 1, 1, **kwargs)
