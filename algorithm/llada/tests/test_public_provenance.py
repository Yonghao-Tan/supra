"""Reject mixed or incomplete calibration/evaluation inputs before costly work."""

from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import torch
from calibration.data import capture_sample_ordinal
from calibration.capture import (
    linear_probe_event_indices,
    validate_replay_source,
    TARGET_NUMERIC_CAPTURE_GRAPH,
)


@pytest.mark.parametrize("length", [32, 64, 256])
def test_capture_context_capacity_uses_requested_generation_length(length):
    from calibration.capture import target_prompt_ids

    class Tokenizer:
        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(input_ids=torch.ones((1, int(prompt)), dtype=torch.long))

    record = SimpleNamespace(task="gsm8k", prompt=str(4096 - length))
    ids = target_prompt_ids(Tokenizer(), record, torch.device("cpu"), generation_length=length)
    assert ids.shape[1] + length == 4096
    record.prompt = str(4097 - length)
    with pytest.raises(ValueError, match="exceeds the deployment context"):
        target_prompt_ids(Tokenizer(), record, torch.device("cpu"), generation_length=length)


@pytest.mark.parametrize("task", ["gsm8k", "humaneval"])
def test_calibration_records_explicit_dependency_behavior(task):
    from calibration.capture import generation_arguments
    from numerics.precision import RowPrecisionContext

    arguments = generation_arguments(task, RowPrecisionContext())
    assert arguments["cross_block_dependency_policy"] == "initial_keys_selected_rows"
    assert arguments.get("cache_initialization_activation_policy", "default") == "default"


def test_silu_samples_follow_request_forward_and_layer_not_worker_history():
    from calibration.capture import silu_error_observers
    from numerics.bf16 import silu_pwl_bf16

    class Block:
        _target_numeric_silu_pwl16 = True

        def _silu_multiply(self, gate, up):
            return (silu_pwl_bf16(gate) * up.float()).to(torch.bfloat16)

    blocks = [Block(), Block()]
    model = SimpleNamespace(
        model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))
    )
    gate = torch.arange(16384).to(torch.int16).view(torch.bfloat16)
    up = torch.ones_like(gate)
    expected = blocks[0]._silu_multiply(gate, up)

    def observe(identity, statistics, forward=3):
        with silu_error_observers(
            model,
            enabled=True,
            sample_identity=("code", identity),
            forward_index=lambda: forward,
            statistics=statistics,
        ):
            for block in blocks:
                actual = block._silu_multiply(gate, up)
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    first, second, together, reversed_order = {}, {}, {}, {}
    observe("first", first)
    observe("second", second)
    observe("first", together)
    observe("second", together)
    observe("second", reversed_order)
    observe("first", reversed_order)
    for layer in range(2):
        for key in ("gate_counts", "up_squared_sum"):
            assert torch.equal(
                first[layer][key] + second[layer][key], together[layer][key]
            )
            assert torch.equal(together[layer][key], reversed_order[layer][key])
        assert together[layer]["calls"] == 2
        assert int(together[layer]["gate_counts"].sum()) == 16384
    assert not torch.equal(first[0]["gate_counts"], first[1]["gate_counts"])
    next_forward = {}
    observe("first", next_forward, forward=4)
    assert not torch.equal(first[0]["gate_counts"], next_forward[0]["gate_counts"])
    assert all("_silu_multiply" not in vars(block) for block in blocks)
    with pytest.raises(ValueError, match="request identity"):
        with silu_error_observers(model, enabled=True):
            pass
    with pytest.raises(ValueError, match="nonfinite"):
        with silu_error_observers(
            model,
            enabled=True,
            sample_identity=("code", "bad"),
            forward_index=lambda: 0,
        ):
            blocks[0]._silu_multiply(torch.full_like(gate, float("nan")), up)
    assert all("_silu_multiply" not in vars(block) for block in blocks)


@pytest.mark.parametrize("split", ["test", "validation"])
def test_calibration_loader_rejects_test_or_unverified_sources(split):
    from calibration.capture import load_records

    with pytest.raises(ValueError, match="training-source"):
        load_records(SimpleNamespace(split=split, train_only_calibration=False))


def test_code_calibration_uses_training_source_and_full_task_signature(monkeypatch):
    from calibration import capture as calibration

    called = {}

    def records(**kwargs):
        called.update(kwargs)
        return ["training-code"]

    monkeypatch.setattr(calibration, "load_code_training_records", records)
    args = SimpleNamespace(
        split="train",
        train_only_calibration=True,
        dataset_manifest="manifest",
        samples_per_task=128,
        data_seed=20260908,
        seed=20260806,
        data_dir="data",
        record_task="humaneval",
    )
    assert calibration.load_records(args) == ["training-code"]
    assert called["prompt_from_task"] is True
    assert called["allowed_sources"] == ("ai2-adapt-dev/personahub_code_v2_34999",)
    assert called["split"] == "train"


def test_joint_gptq_cli_uses_both_task_hessians(tmp_path, monkeypatch):
    import sys
    from calibration import solve as solver
    from quantization.numeric import SpinQuantW4Tensor

    layer = torch.nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(64).reshape(4, 16).float().sin())
    prior = SpinQuantW4Tensor(torch.zeros((4, 16), dtype=torch.int8), torch.ones(4))
    baseline = SimpleNamespace(manifest_sha256="parent", read_w4=lambda name: prior)
    monkeypatch.setattr(solver, "SpinQuantArtifactReader", lambda *a, **kw: baseline)
    monkeypatch.setattr(
        "quantization.artifact.require_train_only_artifact", lambda reader: None
    )
    definition = tmp_path / "coverage.json"
    definition.write_text(
        json.dumps(
            {
                "definition": "sum X.T@X; X=BF16(A_codes*A_scale); unweighted sampled rows"
            }
        )
    )
    monkeypatch.setattr(
        solver, "hessian_sources", lambda root, **kw: ([definition], {str(root)})
    )
    primary = torch.eye(16)
    secondary = torch.diag(torch.arange(1, 17, dtype=torch.float32))
    calls = []

    def hessian(*a, **kw):
        calls.append(len(calls))
        return ((primary if len(calls) == 1 else secondary).clone(), 32)

    monkeypatch.setattr(solver, "load_hessian", hessian)
    model = SimpleNamespace(eval=lambda: model, get_submodule=lambda name: layer)
    monkeypatch.setattr(solver.LLaDAModelLM, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(solver, "apply_fixed_spinquant", lambda *a, **kw: None)
    monkeypatch.setattr(solver, "gptq_module_order", lambda model: ["linear.weight"])
    monkeypatch.setattr(solver, "require_algo_output", lambda path: path)
    output = tmp_path / "fit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "solver",
            "--checkpoint",
            str(tmp_path),
            "--artifact-dir",
            str(tmp_path),
            "--hessian-root",
            str(tmp_path / "primary"),
            "--preserve-hessian-root",
            str(tmp_path / "secondary"),
            "--output-dir",
            str(output),
            "--expected-requests",
            "1",
            "--preserve-expected-requests",
            "1",
            "--shard-count",
            "1",
            "--scale-search",
            "--joint-relative-error",
            "--train-only-calibration",
            "--device",
            "cpu",
        ],
    )
    solver.main()
    result = torch.load(output / "linear.pt", weights_only=True)
    report = result["statistics"]
    assert len(calls) == 2
    assert report["relative_joint_new_error"] < report["relative_joint_old_error"]
    assert report["preserve_token_rows"] == 32
    assert report["preserve_old_error"] != report["old_error"]
    assert result["codes"].shape == (4, 16)
    assert torch.isfinite(result["scales_bf16"]).all()


def numeric_metadata():
    return dict(
        anchor="n4",
        actual_linear_a8=False,
        numeric_capture_graph=TARGET_NUMERIC_CAPTURE_GRAPH,
        artifact_manifest_sha256="weight",
        w8_head_manifest_sha256="head",
        head_weight_bits=8,
        a4_clip_ratio_bf16=0.80078125,
        a4_output_clip_ratio_bf16=0.6015625,
        silu_table=None,
    )


def replay_args():
    return SimpleNamespace(
        train_only_calibration=True,
        split="train",
        anchor="n4",
        linear_a8=False,
        _artifact_manifest_sha256="weight",
        _w8_head_manifest_sha256="head",
        _head_weight_bits=8,
        a4_clip_ratio=0.80078125,
        a4_output_clip_ratio=0.6015625,
        _silu_table_info=None,
    )


def test_event_selection_uses_capture_ordinal_across_directory_and_solver_shards():
    events = [
        SimpleNamespace(
            block_index=i,
            step_index=0,
            forward_kind="boundary_refresh",
            future_prediction_positions=torch.empty(0),
        )
        for i in range(7)
    ]
    records = [
        dict(task="gsm8k", sample_id=f"train/{i}", ordinal=i) for i in range(128)
    ]
    expected = {
        r["sample_id"]: linear_probe_event_indices(
            events, stratified=True, sample_ordinal=capture_sample_ordinal(r)
        )
        for r in records
    }
    for capture_shards in (1, 6, 7):
        ordered = [
            r for shard in range(capture_shards) for r in records[shard::capture_shards]
        ]
        for solver_shards in (1, 7):
            actual = {
                r["sample_id"]: linear_probe_event_indices(
                    events, stratified=True, sample_ordinal=capture_sample_ordinal(r)
                )
                for shard in range(solver_shards)
                for r in ordered[shard::solver_shards]
            }
            assert actual == expected
    with pytest.raises(ValueError, match="ordinal"):
        capture_sample_ordinal(dict(task="gsm8k", sample_id="train/0"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_manifest_sha256", "other"),
        ("w8_head_manifest_sha256", "other"),
        ("a4_clip_ratio_bf16", 1.0),
        ("a4_output_clip_ratio_bf16", 1.0),
        ("silu_table", {"slopes": [1.0]}),
        ("numeric_capture_graph", "other"),
        ("actual_linear_a8", True),
        ("head_weight_bits", 4),
    ],
)
def test_fitting_rejects_numeric_mismatch_but_diagnostic_can_compare(field, value):
    metadata = dict(
        numeric_metadata(), train_only_calibration=True, source_split="train"
    )
    validate_replay_source(metadata, replay_args(), fitting=True)
    metadata[field] = value
    with pytest.raises(ValueError, match="numerical provenance"):
        validate_replay_source(metadata, replay_args(), fitting=True)
    validate_replay_source(metadata, replay_args(), fitting=False)


def write_coverage(root, shard, sample, *, numeric=None):
    from quantization.artifact import expected_w4_ids

    directory = root / f"shard_{shard:02d}"
    directory.mkdir(parents=True)
    (root / "exit_code").write_text("0\n")
    record = dict(
        task="gsm8k",
        sample_id=sample,
        train_only_calibration=True,
        source_split="train",
        numeric_source=numeric or numeric_metadata(),
        generation={"arguments": {"steps": 256}},
    )
    path = directory / "hessian_coverage.json"
    path.write_text(
        json.dumps(
            dict(
                banks=[str(directory / "sample.pt")],
                source_records=[record],
                train_only_calibration=True,
                event_sampling="stratified_max6_capture_ordinal/v2",
                modules={
                    name.removesuffix(".weight"): {} for name in expected_w4_ids()
                },
            )
        )
    )
    return path


def test_hessian_rejects_duplicate_identity_at_different_paths(tmp_path):
    from calibration.solve import hessian_sources

    write_coverage(tmp_path, 0, "train/1")
    assert (
        len(hessian_sources(tmp_path, train_only=True, artifact_identity="weight")[1])
        == 1
    )
    write_coverage(tmp_path, 1, "train/1")
    with pytest.raises(ValueError, match="duplicate calibration sample"):
        hessian_sources(tmp_path, train_only=True)


def test_hessian_rejects_missing_mixed_or_wrong_parent_provenance(tmp_path):
    from calibration.solve import hessian_sources

    path = write_coverage(tmp_path, 0, "train/1")
    with pytest.raises(ValueError, match="solver baseline"):
        hessian_sources(tmp_path, train_only=True, artifact_identity="other")
    second = write_coverage(
        tmp_path, 1, "train/2", numeric=dict(numeric_metadata(), a4_clip_ratio_bf16=1.0)
    )
    with pytest.raises(ValueError, match="numerical provenance differs"):
        hessian_sources(tmp_path, train_only=True)
    second.unlink()
    incomplete = json.loads(path.read_text())
    incomplete["source_records"][0].pop("numeric_source")
    path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="missing capture numerical"):
        hessian_sources(tmp_path, train_only=True)


@pytest.mark.parametrize(
    "missing", ["samples.jsonl", "silu_errors.json", "silu_histograms.pt"]
)
def test_silu_missing_shard_member_rejected_before_histogram_load(
    tmp_path, monkeypatch, missing
):
    from calibration.silu import statistics_shards, load_statistics

    (tmp_path / "exit_code").write_text("0\n")
    for index in range(2):
        shard = tmp_path / f"shard_{index:02d}"
        shard.mkdir()
        for name in ("samples.jsonl", "silu_errors.json", "silu_histograms.pt"):
            (shard / name).touch()
    assert len(statistics_shards(tmp_path)) == 2
    (shard / missing).unlink()
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("must reject before tensor load")
    )
    with pytest.raises(ValueError, match="shard members"):
        load_statistics(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("numeric_coverage", {"clip": 1.0, "silu": [1.0]}),
        ("numeric_coverage", {"clip": 0.8, "silu": [2.0]}),
        ("instruct_prompt_mode", "chat"),
        ("generation_protocol", {"steps": 128}),
    ],
)
def test_trace_merge_rejects_changed_numerics_prompt_or_generation(field, value):
    from evaluation.summary import (
        effective_trace_configuration,
        require_matching_configuration,
    )

    row = dict(
        numeric_coverage={"clip": 0.8, "silu": [1.0]},
        instruct_prompt_mode="raw",
        generation_protocol={"steps": 256},
        deployment_diagnostic={"silu_table": "/old/table"},
    )
    expected = effective_trace_configuration(row)
    moved = deepcopy(row)
    moved["deployment_diagnostic"]["silu_table"] = "/new/table"
    moved.update(doc_id=33, trace_output_path="/different/shard")
    require_matching_configuration(
        expected, effective_trace_configuration(moved), source="relocated"
    )
    moved[field] = value
    with pytest.raises(ValueError, match="configuration differs"):
        require_matching_configuration(
            expected, effective_trace_configuration(moved), source="changed"
        )


def test_result_merge_allows_relocation_and_rejects_generation_change():
    from evaluation.summary import (
        effective_result_configuration,
        require_matching_configuration,
    )

    result = dict(
        config=dict(
            model_args="model_path=/old,spinquant_artifact_dir=/old/w,trace_output_path=/s0,steps=256",
            gen_kwargs={"temperature": 0},
            num_fewshot=4,
        )
    )
    expected = effective_result_configuration(result)
    moved = deepcopy(result)
    moved["config"]["model_args"] = (
        "model_path=/new,spinquant_artifact_dir=/new/w,trace_output_path=/s1,steps=256"
    )
    require_matching_configuration(
        expected, effective_result_configuration(moved), source="relocated"
    )
    moved["config"]["num_fewshot"] = 5
    with pytest.raises(ValueError, match="configuration differs"):
        require_matching_configuration(
            expected, effective_result_configuration(moved), source="changed"
        )


def test_result_task_generation_settings_are_compared_without_shard_names():
    from evaluation.summary import (
        effective_result_configuration,
        require_matching_configuration,
    )

    task = dict(
        task="gsm8k_shard_00", generation_kwargs={"until": ["Question:"]}, num_fewshot=4
    )
    first = dict(configs={"gsm8k_shard_00": task})
    second = dict(configs={"gsm8k_shard_01": dict(task, task="gsm8k_shard_01")})
    expected = effective_result_configuration(first)
    require_matching_configuration(
        expected, effective_result_configuration(second), source="next shard"
    )
    second["configs"]["gsm8k_shard_01"]["generation_kwargs"] = {"until": ["\n"]}
    with pytest.raises(ValueError, match="configuration differs"):
        require_matching_configuration(
            expected, effective_result_configuration(second), source="stop rule"
        )


def test_result_filter_function_address_is_not_a_configuration_change():
    from evaluation.summary import effective_result_configuration

    def result(name, address):
        return dict(
            configs={
                "example_task": {
                    "filter_list": [{"filter_fn": f"<function {name} at {address}>"}]
                }
            }
        )

    expected = effective_result_configuration(result("example_filter", "0x123"))
    assert expected == effective_result_configuration(
        result("example_filter", "0x456")
    )
    assert expected != effective_result_configuration(result("other_function", "0x456"))


def test_collector_rejects_mixed_clipping_before_copy_or_scoring(tmp_path, monkeypatch):
    from evaluation import collect as collector
    import sys

    root = tmp_path / "inputs"
    root.mkdir()
    (root / "exit_code").write_text("0\n")
    for index, clip in enumerate((0.8, 1.0)):
        shard = root / f"shard_{index:02d}"
        shard.mkdir()
        (shard / "results_test.json").write_text(
            json.dumps({"config": {"model_args": {"steps": 256}}})
        )
        row = dict(
            artifact_manifest_sha256="weight",
            numeric_coverage=dict(
                clip=clip,
                candidate={"schedule": "candidate-online-bf16-lut-lanes64/v1"},
            ),
        )
        (shard / "trace.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "collected"
    monkeypatch.setattr(collector, "external_output", lambda *a: (output, tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collector",
            "--task",
            "gsm8k",
            "--inputs",
            str(root),
            "--artifact-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="configuration differs"):
        collector.main()
    assert not output.exists()


def test_wikitext_loader_pins_and_records_revision(monkeypatch):
    from quantization import gptq as module

    class Dataset(dict):
        _fingerprint = "test-fingerprint"

    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return Dataset(text=["training text"])

    monkeypatch.setattr(module, "load_dataset", load)
    tokenizer = lambda *a, **k: {"input_ids": torch.arange(4096)[None]}
    (examples, metadata) = module._calibration_examples(tokenizer)
    assert (
        calls[0]["revision"] == metadata["dataset_revision"] == module.WIKITEXT_REVISION
    )
    assert calls[0]["split"] == "train"
    assert metadata["dataset_fingerprint"] == "test-fingerprint"
    assert len(examples) == 128


@pytest.mark.parametrize("task", ["gsm8k", "humaneval"])
def test_short_calibration_trajectory_and_replay(task):
    from calibration.capture import generation_arguments, select_shard
    from capture.replay import ForwardReplayer
    from generation.engine import generate
    from numerics.precision import RowPrecisionContext
    from numerics.candidate import streaming_candidate_bf16
    from test_cache_initialization import model

    captures = []
    arguments = generation_arguments(task, RowPrecisionContext(), gsm_feature3=True,
        human_feature3=True, full_prefix_oracle_block=1, steps=64, gen_length=64)
    arguments["mask_id"] = 7
    tokens, _, trace, _ = generate(model(), torch.full((1, 64), 2),
        state_capture_callback=captures.append, **arguments)
    assert tokens.shape == (1, 128)
    assert {event.block_index for event in trace} == {0, 1}
    assert any(event.forward_kind == "boundary_refresh" for event in trace)
    replayer = ForwardReplayer(model(), RowPrecisionContext(), device=torch.device("cpu"))
    for event in captures:
        result = replayer.step(event)
        if result.prediction_logits.shape[1]:
            proposal, _, _ = streaming_candidate_bf16(result.prediction_logits)
            expected = event.teacher_top1_token_ids[event.prediction_mask].reshape(1, -1)
            assert torch.equal(proposal.cpu(), expected)
    for count in (1, 2):
        shards = [select_shard(list(range(count)), index, count) for index in range(count)]
        assert all(shards)
        assert sorted(item for shard in shards for item in shard) == list(enumerate(range(count)))
