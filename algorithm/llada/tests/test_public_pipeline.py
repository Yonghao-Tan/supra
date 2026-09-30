"""Exercise the end-to-end driver's command graph and process cleanup on CPU."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location(
    "public_pipeline", Path(__file__).resolve().parents[2] / "run.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def args(tmp_path, mode="all"):
    return SimpleNamespace(
        mode=mode,
        model_path=tmp_path / "checkpoint",
        artifact_root=tmp_path,
        run_dir=tmp_path / "run",
        gpus="0,1,2,3,4,5,6",
        model_artifacts=tmp_path / "weights",
        tasks=["gsm8k"],
        feature3="on",
        source_version="test-source",
    )


def test_pipeline_connects_calibration_export_and_gsm_evaluation(tmp_path):
    arguments = args(tmp_path)
    stages = pipeline.build_stages(arguments)
    names = [name for (name, *_) in stages]
    assert names == [
        "prepare-data",
        "split-data",
        "initial-weights",
        "initial-head",
        "capture",
        "hessians",
        "gptq",
        "silu",
        "export",
        "evaluate-gsm8k",
        "collect-gsm8k",
    ]
    for name, commands, gpus, roots in stages:
        if name == "capture":
            assert len(commands) == 28 and len(roots) == 4
            assert all(("--train-only-calibration" in command for command in commands))
        if name == "hessians":
            assert len(commands) == 14 and len(roots) == 2
        if name.startswith("evaluate-"):
            assert len(commands) == 1
            assert commands[0][commands[0].index("--native-processes") + 1] == "7"
            assert all(
                (
                    command[command.index("--weight-artifact") + 1]
                    == str(arguments.run_dir / "model/artifact_w4")
                    for command in commands
                )
            )
            assert all(("--limit" not in command for command in commands))
        if name == "gptq":
            assert len(commands) == len(gpus) == 7


def test_one_gpu_evaluates_full_native_task(tmp_path):
    arguments = args(tmp_path, mode="evaluate")
    arguments.gpus = "GPU-example"
    stages = pipeline.build_stages(arguments)
    assert len(stages) == 2
    assert len(stages[0][1]) == 1 and stages[0][2] == ["GPU-example"]
    assert all(
        (
            command[command.index("--head-artifact") + 1]
            == str(arguments.model_artifacts / "artifact_w8_head")
            for command in stages[0][1]
        )
    )


def test_native_driver_connects_one_distributed_run_to_existing_collection(tmp_path):
    arguments = args(tmp_path, mode="evaluate")
    arguments.gpus = "0,1,2,3"
    arguments.limit_per_process = 1
    stages = pipeline.build_stages(arguments)
    for name, commands, slots, _ in stages:
        if name.startswith("evaluate-"):
            assert len(commands) == 1 and slots == ["0,1,2,3"]
            assert commands[0][commands[0].index("--native-processes") + 1] == "4"
            assert commands[0][commands[0].index("--limit") + 1] == "4"
            assert "--shard" not in commands[0]
        else:
            assert commands[0][commands[0].index("--inputs") + 1].endswith(
                f"/evaluation/{name.removeprefix('collect-')}"
            )
            assert "--partial" in commands[0]


def test_default_driver_builds_one_native_six_process_evaluation(tmp_path):
    arguments = args(tmp_path, mode="evaluate")
    arguments.gpus = "0,1,2,3,4,5"
    default = pipeline.build_stages(arguments)
    assert default == pipeline.build_stages(arguments)
    for name, commands, slots, _ in default:
        if name.startswith("evaluate-"):
            assert len(commands) == 1 and slots == [arguments.gpus]
            assert commands[0][commands[0].index("--native-processes") + 1] == "6"
            assert "--shard" not in commands[0]


def test_cli_uses_one_process_per_device(tmp_path, monkeypatch, capsys):
    import shlex

    argv = [
        "run.py", "--mode", "evaluate", "--model-path", str(tmp_path / "model"),
        "--model-artifacts", str(tmp_path / "weights"), "--artifact-root", str(tmp_path),
        "--run-dir", str(tmp_path / "run"), "--gpus", "0,1,2,3,4,5",
        "--tasks", "gsm8k", "--source-version", "test-source", "--print-commands",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    pipeline.main()
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()
                if not line.startswith("#")]
    evaluations = [cmd for cmd in commands if "evaluation.generate" in cmd]
    assert len(evaluations) == 1
    assert evaluations[0][evaluations[0].index("--native-processes") + 1] == "6"


def test_partial_evaluation_is_explicit_in_generation_and_collection(tmp_path):
    arguments = args(tmp_path, mode="evaluate")
    arguments.limit_per_process = 1
    stages = pipeline.build_stages(arguments)
    for name, commands, _, _ in stages:
        for command in commands:
            if name.startswith("evaluate-"):
                assert command[command.index("--limit") + 1] == "7"
            else:
                assert "--partial" in command
    arguments.limit_per_process = 0
    with pytest.raises(ValueError, match="positive"):
        pipeline.build_stages(arguments)


def test_pipeline_refuses_in_source_outputs_and_duplicate_gpus(tmp_path):
    arguments = args(tmp_path)
    arguments.gpus = "0,0"
    with pytest.raises(ValueError, match="distinct"):
        pipeline.build_stages(arguments)
    arguments = args(tmp_path)
    arguments.artifact_root = pipeline.ROOT
    arguments.run_dir = pipeline.ROOT / "artifacts"
    with pytest.raises(ValueError, match="external"):
        pipeline.build_stages(arguments)


@pytest.mark.parametrize("override", [False, True])
def test_packaged_calibration_indices_use_external_run_directory(tmp_path, monkeypatch, override):
    release = tmp_path / "release"
    (release / "hardware").mkdir(parents=True)
    (release / "algorithm").mkdir()
    monkeypatch.setattr(pipeline, "ROOT", release / "algorithm")
    arguments = args(tmp_path, mode="calibrate")
    if override:
        arguments.data_index_dir = tmp_path / "indices"
    expected = arguments.data_index_dir if override else arguments.run_dir / "data_indices"
    stages = pipeline.build_stages(arguments)
    for name, commands, _, _ in stages:
        if name in ("prepare-data", "split-data"):
            for cmd in commands:
                assert Path(cmd[cmd.index("--manifest") + 1]).parent == expected


@pytest.mark.parametrize("entry", ["prepare", "split"])
def test_packaged_data_builders_accept_external_indices(tmp_path, monkeypatch, entry):
    from calibration import prepare, split

    module = prepare if entry == "prepare" else split
    release = tmp_path / "release"
    (release / "hardware").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    parent = external / "parent.json"
    parent.write_text("{}")
    monkeypatch.setattr(module, "__file__", str(release / "algorithm/llada/calibration" / (entry + ".py")))
    monkeypatch.setenv("SUPRA_ALGORITHM_ROOT", str(external))
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
        output_dir=external / "data", manifest=external / "indices/train.json",
        parent_manifest=parent, ngram_width=13))

    def reached_data_stage(*args):
        raise RuntimeError("reached data stage")

    monkeypatch.setattr(module, "build_protected_keys" if entry == "prepare" else "_read_parent_records",
                        reached_data_stage)
    with pytest.raises(RuntimeError, match="reached data stage"):
        module.main()


@pytest.mark.parametrize("layout", ["standalone", "integrated", "nested_checkout"])
@pytest.mark.parametrize("entry", ["driver", "packager", "quantization", "evaluation", "prepare", "split"])
def test_all_algorithm_entries_reject_outputs_inside_release(tmp_path, monkeypatch, layout, entry):
    project = tmp_path / "source"
    release = project / "release" if layout == "nested_checkout" else project
    algorithm = release if layout == "standalone" else release / "algorithm"
    algorithm.mkdir(parents=True)
    if layout != "standalone":
        (release / "hardware").mkdir()
    if layout == "nested_checkout":
        (project / ".git").mkdir()
    output = (release / "hardware" if layout == "integrated" else project) / "outputs"
    if entry == "driver":
        monkeypatch.setattr(pipeline, "ROOT", algorithm)
        arguments = args(tmp_path, mode="evaluate")
        arguments.artifact_root = project
        arguments.run_dir = output
        with pytest.raises(ValueError, match="external"):
            pipeline.build_stages(arguments)
    elif entry == "quantization":
        from quantization import rotation

        monkeypatch.setattr(rotation, "__file__", str(algorithm / "llada/quantization/rotation.py"))
        monkeypatch.setattr(rotation, "ALGO_ROOT", project)
        with pytest.raises(ValueError, match="outside the source"):
            rotation.require_algo_output(output)
    elif entry == "evaluation":
        from evaluation import generate

        monkeypatch.setattr(generate, "SOURCE", algorithm / "llada")
        with pytest.raises(ValueError, match="external"):
            generate.external_output(output, project)
    elif entry in {"prepare", "split"}:
        from calibration import prepare, split

        module = prepare if entry == "prepare" else split
        monkeypatch.setattr(module, "__file__", str(algorithm / "llada/calibration" / (entry + ".py")))
        monkeypatch.setenv("SUPRA_ALGORITHM_ROOT", str(project))
        monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
            output_dir=output, manifest=algorithm / "data_indices/test.json"))
        with pytest.raises(ValueError, match="outside the source"):
            module.main()
    else:
        spec = importlib.util.spec_from_file_location(
            "package_model", pipeline.ROOT / "package_model.py"
        )
        pack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pack)
        monkeypatch.setattr(pack, "__file__", str(algorithm / "package_model.py"))
        monkeypatch.setattr(sys, "argv", [
            "package_model.py", "--weight-artifact", str(tmp_path / "missing-parent"),
            "--head-artifact", str(tmp_path / "missing-head"),
            "--silu-table", str(tmp_path / "missing-table"),
            "--model-path", str(tmp_path / "missing-model"),
            "--artifact-root", str(project), "--output", str(output),
        ])
        with pytest.raises(ValueError, match="outside the source"):
            pack.main()
    assert not output.exists()


def test_integrated_archive_accepts_external_algorithm_outputs(tmp_path, monkeypatch):
    from quantization import rotation
    from evaluation import generate

    release = tmp_path / "release"
    algorithm = release / "algorithm"
    algorithm.mkdir(parents=True)
    (release / "hardware").mkdir()
    external = tmp_path / "artifacts"
    monkeypatch.setattr(pipeline, "ROOT", algorithm)
    arguments = args(tmp_path, mode="evaluate")
    arguments.artifact_root = external
    arguments.run_dir = external / "evaluation"
    assert pipeline.build_stages(arguments)
    monkeypatch.setattr(rotation, "__file__", str(algorithm / "llada/quantization/rotation.py"))
    monkeypatch.setattr(rotation, "ALGO_ROOT", external)
    assert rotation.require_algo_output(external / "model") == external / "model"
    monkeypatch.setattr(generate, "SOURCE", algorithm / "llada")
    assert generate.external_output(external / "evaluation", external) == (
        external / "evaluation", external)


def test_stage_completes_every_job_before_success_status(tmp_path):
    (tmp_path / "logs").mkdir()
    commands = [[sys.executable, "-c", 'print("done")'] for _ in range(3)]
    root = tmp_path / "stage"
    pipeline.run_stage("cpu", commands, [], dict(os.environ), tmp_path, [root])
    assert (root / "exit_code").read_text() == "0\n"
    assert len(list((tmp_path / "logs").glob("*.log"))) == 3


def test_failed_job_stops_active_peer_and_never_marks_stage_success(
    tmp_path, monkeypatch
):
    (tmp_path / "logs").mkdir()
    processes = []
    popen = subprocess.Popen

    def start(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    commands = [
        [sys.executable, "-c", "raise SystemExit(7)"],
        [sys.executable, "-c", "import time; time.sleep(30)"],
    ]
    root = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="exited 7"):
        pipeline.run_stage(
            "cpu", commands, ["0", "1"], dict(os.environ), tmp_path, [root]
        )
    assert all((process.poll() is not None for process in processes))
    assert (root / "exit_code").read_text() == "1\n"


def test_interruption_reaps_worker_and_marks_incomplete(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    processes = []
    popen = subprocess.Popen

    def start(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    sleep = pipeline.time.sleep

    def interrupt(_seconds):
        monkeypatch.setattr(pipeline.time, "sleep", sleep)
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline.subprocess, "Popen", start)
    monkeypatch.setattr(pipeline.time, "sleep", interrupt)
    root = tmp_path / "stage"
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_stage(
            "cpu",
            [[sys.executable, "-c", "import time; time.sleep(30)"]],
            [],
            dict(os.environ),
            tmp_path,
            [root],
        )
    assert processes[0].poll() is not None
    assert (root / "exit_code").read_text() == "1\n"




def test_model_packaging_preserves_codes_and_rejects_corrupt_copy(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "package_model", pipeline.ROOT / "package_model.py"
    )
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    source = tmp_path / "source"
    source.mkdir()
    codes = bytes([135, 15, 254])
    (source / "weights.int4").write_bytes(codes)
    manifest = {
        "files": {
            "weights.int4": {
                "bytes": len(codes),
                "sha256": hashlib.sha256(codes).hexdigest(),
            }
        }
    }
    (source / "tensor_manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "model"
    assert pack.copy_artifact(source, destination, manifest=manifest) == len(codes)
    assert (destination / "weights.int4").read_bytes() == codes
    assert json.loads((destination / "tensor_manifest.json").read_text()) == manifest
    (source / "weights.int4").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum differs"):
        pack.copy_artifact(source, tmp_path / "corrupt", manifest=manifest)


def test_model_package_has_relative_parent_and_deployment_metadata(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("package_model", pipeline.ROOT / "package_model.py")
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    source = tmp_path / "inputs"
    source.mkdir()
    payload = b"\x87\x0f\xfe"
    files = {"payload.bin": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
    checkpoint = dict(config_sha256="config", index_sha256="index", model_identity="model", path="/private/model")
    parent = dict(schema_version="parent", status="complete", variant="variant", files=files,
                  total_file_bytes=len(payload), checkpoint=checkpoint,
                  gptq=dict(bits=4, group_size=-1, rounding="round_to_nearest_even"),
                  calibration=dict(train_only_calibration=True, output_path="/private/calibration"),
                  tensors=[dict(logical_id="weight", shape=[1, 3], code_min=-8, code_max=7,
                                codes_sha256=files["payload.bin"]["sha256"], calibration_rows=10,
                                quantization_statistics=dict(error=0.1))])
    parent_raw = json.dumps(parent).encode()
    child = dict(parent, schema_version="child", heldout_metrics=dict(error=0.2),
                 parent_artifact=dict(manifest_sha256=hashlib.sha256(parent_raw).hexdigest()))
    for name, raw in (("parent", parent_raw), ("child", json.dumps(child).encode())):
        (source / name).mkdir()
        (source / name / "tensor_manifest.json").write_bytes(raw)
        (source / name / "payload.bin").write_bytes(payload)
    table = dict(schema_version="table", train_only_calibration=True, breakpoints=[0, 1],
                 slopes=[1], intercepts=[0], fit=[dict(error=0.1)], calibration_root="/private/calibration")
    (source / "silu.json").write_text(json.dumps(table))
    (source / "model").mkdir()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        (source / "model" / name).write_text("{}")
    notices = {"README.md": "---\nlicense: mit\n---\nModel attribution\n",
               "LICENSE.txt": "Source model license\n", "NOTICE": "Source model notice\n"}
    for name, content in notices.items():
        (source / "model" / name).write_text(content)
    destination = tmp_path / "outputs" / "model"
    monkeypatch.setattr(sys, "argv", ["package_model.py", "--weight-artifact", str(source / "parent"),
        "--head-artifact", str(source / "child"), "--silu-table", str(source / "silu.json"),
        "--model-path", str(source / "model"), "--artifact-root", str(tmp_path / "outputs"),
        "--output", str(destination)])
    pack.main()
    relocated = tmp_path / "relocated"
    destination.rename(relocated)
    parent_file = relocated / "artifact_w4/tensor_manifest.json"
    packaged_parent = json.loads(parent_file.read_text())
    packaged_child = json.loads((relocated / "artifact_w8_head/tensor_manifest.json").read_text())
    link = packaged_child["parent_artifact"]
    assert (relocated / "artifact_w8_head" / link["path"]).resolve() == parent_file.parent
    assert link["manifest_bytes"] == parent_file.stat().st_size
    assert link["manifest_sha256"] == hashlib.sha256(parent_file.read_bytes()).hexdigest()
    assert packaged_child["checkpoint"] == packaged_parent["checkpoint"]
    assert packaged_parent["calibration"] == {"train_only_calibration": True}
    assert packaged_parent["gptq"] == parent["gptq"]
    assert (relocated / "LICENSE").read_bytes() == (pipeline.ROOT / "LICENSE").read_bytes()
    for name in notices:
        assert (relocated / "upstream" / name).read_bytes() == (source / "model" / name).read_bytes()
    assert packaged_parent["tensors"] == [dict(logical_id="weight", shape=[1, 3], code_min=-8,
        code_max=7, codes_sha256=files["payload.bin"]["sha256"])]
    assert json.loads((relocated / "silu_table_monotone.json").read_text()) == {
        key: table[key] for key in ("schema_version", "train_only_calibration", "breakpoints", "slopes", "intercepts")}
    for name in ("artifact_w4", "artifact_w8_head"):
        assert (relocated / name / "payload.bin").read_bytes() == payload
    assert "/private/" not in parent_file.read_text()
    assert "heldout_metrics" not in packaged_child


def test_top_level_forwards_config_tier_and_explicit_model_args(tmp_path):
    arguments = args(tmp_path, mode="evaluate")
    arguments.tasks = ["gsm8k"]
    arguments.config = tmp_path / "task.json"
    arguments.config.write_text(json.dumps({"task": "gsm8k"}))
    arguments.tier, arguments.model_arg = "baseline", ["decode_k=2", "gen_length=32"]
    command = pipeline.build_stages(arguments)[0][1][0]
    assert command[command.index("--config") + 1] == str(arguments.config)
    assert command[command.index("--tier") + 1] == "baseline"
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--model-arg"] == arguments.model_arg
    assert "--native-processes" in command and "--shard-count" not in command


def test_short_calibration_reuses_data_and_shares_sample_counts(tmp_path):
    arguments = args(tmp_path)
    arguments.tasks = ["gsm8k"]
    arguments.config = Path(__file__).parents[1] / "configs/gsm8k_short.json"
    arguments.calibration_manifest = tmp_path / "existing_train.json"
    arguments.calibration_data_dir = tmp_path / "existing_data"
    stages = pipeline.build_stages(arguments)
    by_name = {name: commands for name, commands, _, _ in stages}
    assert "prepare-data" not in by_name and "split-data" not in by_name
    assert len(by_name["capture"]) == 6 and len(by_name["hessians"]) == 4
    for name in ("capture", "hessians"):
        for command in by_name[name]:
            option = lambda flag: command[command.index(flag) + 1]
            count = 2 if option("--split") == "train" else 1
            assert int(option("--samples-per-task")) == count
            assert int(option("--shard-count")) == count
            assert 0 <= int(option("--shard-index")) < count
            assert option("--dataset-manifest") == str(arguments.calibration_manifest)
            assert option("--data-dir") == str(arguments.calibration_data_dir)
            assert option("--steps") == option("--gen-length") == "64"
            assert "--train-only-calibration" in command
    for command in by_name["gptq"]:
        assert command[command.index("--expected-requests") + 1] == "2"
        assert command[command.index("--preserve-expected-requests") + 1] == "2"
    arguments.calibration_data_dir = None
    capture = dict((name, commands) for name, commands, _, _ in pipeline.build_stages(arguments))["capture"][0]
    assert "--data-dir" not in capture


@pytest.mark.parametrize("settings", [
    {"train_samples_per_task": 0}, {"validation_samples_per_task": True},
    {"gen_length": 63}, {"gen_length": 96, "steps": 64}, {"typo": 2},
])
def test_calibration_rejects_invalid_configuration(tmp_path, settings):
    arguments = args(tmp_path, mode="calibrate")
    arguments.config = tmp_path / "config.json"
    arguments.config.write_text(json.dumps({"calibration": settings}))
    with pytest.raises(ValueError, match="calibration"):
        pipeline.build_stages(arguments)
