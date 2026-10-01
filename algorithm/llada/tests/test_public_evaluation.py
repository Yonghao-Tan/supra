"""The public command preserves task parameters and isolates shell overrides."""

import json
import inspect
from pathlib import Path
import random
from types import SimpleNamespace
import pytest
from evaluation import generate as entry


@pytest.mark.parametrize("config_path", sorted((entry.SOURCE / "configs").glob("*.json")),
                         ids=lambda path: path.stem)
def test_task_model_settings_match_evaluator_arguments(config_path):
    from evaluation.model import QuantizedLLaDALM

    config = json.loads(config_path.read_text())
    arguments = inspect.signature(QuantizedLLaDALM.__init__).parameters
    assert not (set(config.get("model_args", {})) - set(arguments))


def arguments(task="gsm8k"):
    return SimpleNamespace(
        task=task,
        config=None,
        native_processes=1,
        output=Path("/external-algorithm/eval/shard_00"),
        artifact_root=Path("/external-algorithm"),
        model_path=Path("/model/LLaDA-8B-Instruct"),
        head_artifact=Path("/external-algorithm/head"),
        weight_artifact=Path("/external-algorithm/weight"),
        silu_table=Path("/external-algorithm/silu.json"),
        feature3="on",
        source_version="test",
        device="cuda:0",
        limit=None,
    )


@pytest.mark.parametrize("mode", ["feature1_fixed_k", "feature1_fixed_threshold"])
def test_feature3_off_preserves_fixed_decoder_and_refresh_config(tmp_path, mode):
    args = arguments()
    config = json.loads((entry.SOURCE / "configs/gsm8k.json").read_text())
    config["model_args"].update(generation_mode=mode, decode_k=1, decode_threshold=.95)
    args.config = tmp_path / "fixed.json"
    args.config.write_text(json.dumps(config))
    args.feature3 = "off"
    command, _, _ = entry.build_command(args)
    settings = dict(item.split("=", 1) for item in command[command.index("--model_args") + 1].split(","))
    assert settings["generation_mode"] == mode
    assert settings["feature3_precision_policy"] == "all_a8"
    assert settings["feature2_cache_initialization_activation_policy"] == "a4"
    assert settings["feature1_cross_block_block_initialization_deep_rows"] == "384"
    assert settings["feature1_cross_block_block_initialization_deep_bits"] == "4"
    assert settings["a4_output_clip_ratio"] == "0.625"
    assert "feature3_dynamic_block_source_a_confirm_at_handoff" not in settings


def test_native_training_uses_default_seeds_and_excludes_its_target():
    from lm_eval.api.samplers import ContextSampler
    from lm_eval.utils import load_yaml_config

    args = arguments()
    args.native_processes, args.limit, args.split = 6, 128, "train"
    command, _, _ = entry.build_command(args)
    assert command[command.index("--tasks") + 1] == "gsm8k_train_native"
    assert command[command.index("--seed") + 1] == "0,1234,1234,1234"
    config = load_yaml_config(str(entry.SOURCE / "evaluation/tasks/gsm8k_train_native.yaml"))
    assert config['test_split'] == config['fewshot_split'] == 'train'
    assert config['num_fewshot'] == 4
    # Five documents forces the sampler to encounter the target, so this
    # exercises exclusion instead of merely hoping it was not drawn.
    docs = [{'question': f'item_{i}', 'answer': f'answer_{i}'} for i in range(5)]
    cfg = SimpleNamespace(**config)
    task = SimpleNamespace(config=cfg, _config=cfg, doc_to_choice=None,
                           doc_to_text=lambda d: d['question'],
                           doc_to_target=lambda d: d['answer'])
    cfg.target_delimiter, cfg.fewshot_delimiter = ' ', '\n\n'
    cfg.fewshot_config, cfg.doc_to_choice = None, None
    sampler = ContextSampler(docs, task, rnd=random.Random(1234))
    for doc in docs:
        context = sampler.get_context(doc, 4)
        assert doc['question'] not in context and doc['answer'] not in context
        assert sum(d['question'] in context for d in docs) == 4
    args.limit = None
    with pytest.raises(ValueError, match="explicit limit"):
        entry.build_command(args)


@pytest.mark.parametrize("seed", [None, 1234, 4321])
@pytest.mark.parametrize("task_seed", ["0,1234,1234,1234", "1,2,3,4"])
def test_lm_eval_model_seed_default_and_override(monkeypatch, seed, task_seed):
    config = json.loads((entry.SOURCE / "configs/gsm8k.json").read_text())
    config["seed"] = task_seed
    if seed is None:
        config["model_args"].pop("seed")
    else:
        config["model_args"]["seed"] = seed
    monkeypatch.setattr(entry.json, "loads", lambda _: config)
    command, _, _ = entry.build_command(arguments())
    settings = dict(item.split("=", 1) for item in
                    command[command.index("--model_args") + 1].split(","))
    assert settings["seed"] == str(1234 if seed is None else seed)
    assert command[command.index("--seed") + 1] == task_seed


def test_standard_lm_eval_command_preserves_algorithm_and_uses_default_seeds(monkeypatch):
    args = arguments()
    original, _, _ = entry.build_command(args)
    monkeypatch.setenv("SUPRA_EVALUATION_PROTOCOL", "untrusted")
    command, env, _ = entry.build_command(args)
    assert command[command.index("--seed") + 1] == "0,1234,1234,1234"
    assert command[command.index("--num_fewshot") + 1] == "4"
    parse = lambda c: dict(x.split("=", 1) for x in c[c.index("--model_args") + 1].split(","))
    before, after = parse(original), parse(command)
    assert after.pop("seed") == "1234"
    before.pop("seed")
    assert before == after


def test_standard_humaneval_keeps_zero_shot_prompt_and_feature3_pair():
    args = arguments("humaneval")
    on, env, _ = entry.build_command(args)
    assert on[on.index("--seed") + 1] == "0,1234,1234,1234"
    assert on[on.index("--num_fewshot") + 1] == "0"
    assert on[on.index("--tasks") + 1] == "humaneval_native"
    parse = lambda command: dict(
        field.split("=", 1)
        for field in command[command.index("--model_args") + 1].split(",")
    )
    on_args = parse(on)
    assert on_args["seed"] == "1234"
    assert on_args["instruct_prompt_mode"] == "chat"
    assert on_args["humaneval_full_completion"] == "true"
    args.feature3 = "off"
    off_args = parse(entry.build_command(args)[0])
    assert off_args["generation_mode"] == "feature1_packed_feature2_fused_a4a8"
    assert {
        key: value for key, value in on_args.items()
        if not key.startswith("feature3_") and key != "generation_mode"
    } == {
        key: value for key, value in off_args.items()
        if not key.startswith("feature3_") and key != "generation_mode"
    }


def test_code_eval_uses_distinct_cache_files_for_concurrent_scorers(monkeypatch):
    from evaluation import scoring as protocol

    calls = []
    monkeypatch.setattr(protocol.hf_evaluate, "load", lambda *args, **kwargs: calls.append((args, kwargs)))
    protocol.load_code_eval()
    protocol.load_code_eval()

    assert all(args == ("code_eval",) for args, _ in calls)
    assert all(kwargs["revision"] == protocol.CODE_EVAL_REVISION for _, kwargs in calls)
    assert calls[0][1]["experiment_id"] != calls[1][1]["experiment_id"]


def test_external_dependency_path_reaches_model_process(monkeypatch):
    monkeypatch.setenv("SUPRA_EVALUATION_DEPENDENCIES", "/remote/python-overlay")
    _, env, _ = entry.build_command(arguments("humaneval"))
    assert env["PYTHONPATH"] == f"{entry.SOURCE}:/remote/python-overlay"


def test_native_distributed_command_uses_full_tasks_and_accelerate():
    from accelerate.commands.launch import launch_command_parser
    from lm_eval.utils import load_yaml_config

    for task, fewshot in (("gsm8k", 4), ("humaneval", 0)):
        args = arguments(task)
        args.native_processes = 2
        command, env, output = entry.build_command(args)
        assert command[:4] == [entry.sys.executable, "-B", "-m", "accelerate.commands.launch"]
        parsed = launch_command_parser().parse_args(command[4:])
        assert parsed.num_processes == 2
        assert parsed.num_machines == 1
        assert parsed.module is True
        assert parsed.training_script == "evaluation.model"
        assert parsed.training_script_args[parsed.training_script_args.index("--tasks") + 1] == f"{task}_native"
        assert parsed.training_script_args[parsed.training_script_args.index("--num_fewshot") + 1] == str(fewshot)
        assert parsed.training_script_args[parsed.training_script_args.index("--seed") + 1] == "0,1234,1234,1234"
        assert str(output / "trace.jsonl") in parsed.training_script_args[parsed.training_script_args.index("--model_args") + 1]
        yaml = load_yaml_config(str(entry.SOURCE / "evaluation/tasks" / f"{task}_native.yaml"))
        assert "process_docs" not in yaml
        args.limit = 1
    with pytest.raises(ValueError, match="at least one request per process"):
        entry.build_command(args)


def test_distributed_runtime_uses_global_rank_and_separate_traces(monkeypatch):
    import accelerate
    import torch
    from evaluation.model import _evaluation_runtime, _rank_trace_path

    monkeypatch.setenv("WORLD_SIZE", "4")
    fake = SimpleNamespace(num_processes=4, process_index=3, device=torch.device("cuda:1"))
    monkeypatch.setattr(accelerate, "Accelerator", lambda **_: fake)
    assert _evaluation_runtime("cuda") == (fake, 3, 4, torch.device("cuda:1"))
    path = Path("/external-algorithm/trace.jsonl")
    assert _rank_trace_path(path, 3, 4).name == "trace.rank_03.jsonl"
    assert _rank_trace_path(path, 0, 1) == path
    with pytest.raises(ValueError, match="device=cuda"):
        _evaluation_runtime("cuda:0")


@pytest.mark.parametrize("task,feature3,boundary_rows", [
    ("gsm8k", True, None), ("gsm8k", False, None),
    ("humaneval", True, None), ("humaneval", False, None),
    ("gsm8k", True, 128), ("gsm8k", True, 256),
])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_constructor_reaches_model_loading(
    monkeypatch, task, feature3, boundary_rows, device
):
    from evaluation import model as evaluator

    settings = json.loads((entry.SOURCE / "configs" / (task + ".json")).read_text())[
        "model_args"
    ]
    if boundary_rows is not None:
        settings["feature1_cross_block_boundary_target_rows"] = boundary_rows
    if not feature3:
        settings = {
            key: value
            for (key, value) in settings.items()
            if not key.startswith("feature3_dynamic_block")
        }
        settings["generation_mode"] = "feature1_packed_feature2_fused_a4a8"
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: device == "cuda")
    monkeypatch.setattr(evaluator, "checkpoint_identity", lambda _: {})

    def stop_loading(*args, **kwargs):
        raise RuntimeError("reached tokenizer loading")

    monkeypatch.setattr(evaluator.AutoTokenizer, "from_pretrained", stop_loading)
    with pytest.raises(RuntimeError, match="reached tokenizer loading"):
        evaluator.QuantizedLLaDALM(
            model_path="/unused", spinquant_artifact_dir="/unused", device=device, **settings
        )


def test_public_command_keeps_adopted_numerics_and_discards_shell_overrides(
    monkeypatch,
):
    monkeypatch.setenv("SUPRA_DIAGNOSTIC_CANDIDATE_LANES", "32")
    monkeypatch.setenv("SUPRA_DISABLE_TRITON_LINEAR_NUMERIC", "1")
    monkeypatch.setenv("SUPRA_GSM_DIAGNOSTIC_INDICES", "[1,2]")
    for task, target, deep, budget in [
        ("gsm8k", 27.5, 88, 26),
        ("humaneval", 39.5, 64, 19),
    ]:
        args = arguments(task)
        (command, env, output) = entry.build_command(args)
        config = json.loads((entry.SOURCE / "configs" / (task + ".json")).read_text())
        actual = command[command.index("--model_args") + 1]
        for key, value in config["model_args"].items():
            value = str(value).lower() if isinstance(value, bool) else str(value)
            assert f"{key}={value}" in actual.split(",")
        assert config["model_args"]["feature1_tri_target_active_rows"] == target
        assert config["model_args"]["feature1_cross_block_boundary_target_rows"] == deep
        assert config["model_args"]["feature2_budget_scale"] == budget
        assert config["model_args"].get(
            "feature3_dynamic_block_source_b_max_attempts", 0
        ) == 2
        for key in ("source_b_dependency_tie_rank", "source_b_a4_only"):
            assert config["model_args"].get(
                "feature3_dynamic_block_" + key, False
            ) == (task == "gsm8k")
        assert config["model_args"]["feature3_dynamic_block_next_admission_budget"] == (
            12 if task == "gsm8k" else 8
        )
        assert config["model_args"].get(
            "feature1_cross_block_boundary_deep_a8_row_limit", 0
        ) == (40 if task == "gsm8k" else 0)
        assert "feature1_force_full_current" not in config["model_args"]
        assert not config["model_args"]["logical_a4a8"]
        assert "SUPRA_DIAGNOSTIC_CANDIDATE_LANES" not in env
        assert "SUPRA_DISABLE_TRITON_LINEAR_NUMERIC" not in env
        assert "SUPRA_GSM_DIAGNOSTIC_INDICES" not in env
        assert str(output / "trace.jsonl") in actual
        assert command[command.index("--num_fewshot") + 1] == (
            "4" if task == "gsm8k" else "0"
        )


def test_feature3_off_retains_precision_state_and_changes_only_lookahead():
    args = arguments()
    on = entry.build_command(args)[0]
    args.feature3 = "off"
    off = entry.build_command(args)[0]
    parse = lambda c: dict(
        (x.split("=", 1) for x in c[c.index("--model_args") + 1].split(","))
    )
    (on, off) = (parse(on), parse(off))
    assert off["generation_mode"] == "feature1_packed_feature2_fused_a4a8"
    assert off["feature3_precision_policy"] == "original"
    assert off["feature3_maturity_age"] == "3"
    assert not any((k.startswith("feature3_dynamic_block") for k in off))
    assert {
        k: v
        for (k, v) in on.items()
        if not k.startswith("feature3_") and k != "generation_mode"
    } == {
        k: v
        for (k, v) in off.items()
        if not k.startswith("feature3_") and k != "generation_mode"
    }


def test_public_output_and_process_count_errors():
    args = arguments()
    args.output = entry.SOURCE / "eval"
    args.artifact_root = entry.SOURCE
    with pytest.raises(ValueError, match="external artifact root"):
        entry.build_command(args)
    args = arguments()
    args.native_processes = 0
    with pytest.raises(ValueError, match="at least one request"):
        entry.build_command(args)
    args = arguments()
    args.silu_table = Path("/external-algorithm/ambiguous,table.json")
    with pytest.raises(ValueError, match="commas"):
        entry.build_command(args)


def test_export_without_git_rejects_payload_anywhere_in_source_package(
    tmp_path, monkeypatch
):
    source = tmp_path / "public" / "llada"
    source.mkdir(parents=True)
    monkeypatch.setattr(entry, "SOURCE", source)
    with pytest.raises(ValueError, match="external artifact root"):
        entry.external_output(
            source.parent / "artifacts" / "eval", source.parent / "artifacts"
        )
    (output, root) = entry.external_output(
        tmp_path / "data" / "eval", tmp_path / "data"
    )
    assert output == root / "eval"


def test_numeric_exports_without_git_also_reject_in_source_artifacts(
    tmp_path, monkeypatch
):
    from quantization import rotation as spinquant_rotation

    package = tmp_path / "public"
    monkeypatch.setattr(
        spinquant_rotation, "__file__", str(package / "llada/quantization/rotation.py")
    )
    monkeypatch.setattr(spinquant_rotation, "ALGO_ROOT", package / "artifacts")
    with pytest.raises(ValueError, match="outside the source repository"):
        spinquant_rotation.require_algo_output(package / "artifacts" / "weights")


def test_unknown_model_argument_is_rejected():
    from evaluation.model import QuantizedLLaDALM

    with pytest.raises(ValueError, match="Unsupported evaluation model arguments"):
        QuantizedLLaDALM(model_path="/unused", spinquant_artifact_dir="/unused", unknown_setting=0)


def test_progressive_tiers_preserve_default_and_apply_overrides():
    config = json.loads((entry.SOURCE / "configs/gsm8k.json").read_text())
    assert entry.resolve_model_args(config) == config["model_args"]
    for tier, mode in (("baseline", "baseline_fixed_k"), ("feature1", "feature1_fixed_k"),
                       ("feature12", "feature1_packed_feature2_fused_a4a8"),
                       ("feature123", "feature1_packed_feature2_fused_dynamic_block")):
        settings = entry.resolve_model_args(config, tier=tier, overrides={"decode_k": 2})
        assert settings["generation_mode"] == mode and settings["decode_k"] == 2
        assert settings["feature2_cache_initialization_activation_policy"] == ("default" if tier == "baseline" else "a4")
        if tier != "baseline":
            assert settings["feature1_cross_block_block_initialization_deep_rows"] == 384
        if tier != "feature123":
            assert not any(k.startswith("feature3_dynamic") for k in settings)
    with pytest.raises(ValueError, match="unknown generation_mode"):
        entry.resolve_model_args({"model_args": {"generation_mode": "typo"}})
    assert entry.model_overrides(["decode_k=2", "flag=false", "name=example"]) == {
        "decode_k": 2, "flag": False, "name": "example"}


def test_short_generation_overrides_task_generation_kwargs():
    args = arguments()
    args.tier, args.model_arg = "baseline", ["gen_length=32", "steps=32"]
    command, _, _ = entry.build_command(args)
    assert "max_gen_toks=32" in command[command.index("--gen_kwargs") + 1]
    settings = command[command.index("--model_args") + 1]
    assert "gen_length=32" in settings and "steps=32" in settings
