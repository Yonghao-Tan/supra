"""Run a benchmark with lm-eval native task distribution."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import shutil

SOURCE = Path(__file__).resolve().parents[1]


def external_output(path, root):
    (path, root) = (Path(path).resolve(), Path(root).resolve())
    path.relative_to(root)
    algorithm_root = SOURCE.parent
    release_root = (
        algorithm_root.parent
        if algorithm_root.name == "algorithm" and (algorithm_root.parent / "hardware").is_dir()
        else algorithm_root
    )
    project = next((p for p in SOURCE.parents if (p / ".git").exists()), release_root)
    if path == root or path == project or project in path.parents:
        raise ValueError("output must be a child of the external artifact root")
    return (path, root)


def resolve_model_args(config, *, tier=None, feature3=None, overrides=None):
    """Resolve task settings and the progressive execution tier for every caller."""
    values = dict(config.get("model_args", {}))
    values.update(overrides or {})
    selected = tier or config.get("tier")
    if selected is None:
        mode = values.get("generation_mode", "feature1_packed_feature2_fused_dynamic_block")
        selected = {"baseline_fixed_k": "baseline", "feature1_fixed_k": "feature1",
                    "feature1_fixed_threshold": "feature1",
                    "feature1_packed_feature2_fused_a4a8": "feature12",
                    "feature1_packed_feature2_fused_dynamic_block": "feature123"}.get(mode)
        if selected is None:
            raise ValueError(f"unknown generation_mode: {mode!r}")
    if selected not in ("baseline", "feature1", "feature12", "feature123"):
        raise ValueError(f"unknown tier: {selected!r}")
    if feature3 == "off":
        if tier == "feature123":
            raise ValueError("tier=feature123 conflicts with feature3=off")
        if selected == "feature123":
            selected = "feature12"
    if selected != "feature123":
        values = {k: v for k, v in values.items() if not k.startswith("feature3_")}
    fixed = selected in ("baseline", "feature1")
    previous_mode = values.get("generation_mode")
    values["generation_mode"] = {
        "baseline": "baseline_fixed_k", "feature1": "feature1_fixed_k",
        "feature12": "feature1_packed_feature2_fused_a4a8",
        "feature123": "feature1_packed_feature2_fused_dynamic_block",
    }[selected]
    # Preserve explicitly configured fixed-threshold callers of the lower entry.
    if tier is None and "tier" not in config and previous_mode == "feature1_fixed_threshold":
        values["generation_mode"] = previous_mode
    values["feature3_precision_policy"] = "all_a8" if fixed else "original"
    values["feature3_maturity_age"] = 3
    if fixed:
        values.setdefault("decode_k", 3)
    if selected == "baseline":
        values = {k: v for k, v in values.items() if not k.startswith("feature1_cross_block_block_initialization_")}
        values.update(feature2_cache_initialization_activation_policy="default",
                      feature2_a4_direct_tau=-1.0,
                      cross_block_pending_confidence_mode="remask_only",
                      feature1_cross_block_cache_handoff=False,
                      feature1_cross_block_full_prefix_oracle_block=-1,
                      feature1_cross_block_boundary_scope="tri",
                      feature1_cross_block_boundary_deep_a8_row_limit=-1)
    return values


def model_overrides(items):
    result = {}
    for item in items or ():
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError("model-arg uses NAME=JSON_VALUE")
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            result[key] = raw
    return result


def build_command(args):
    if args.task != "gsm8k":
        raise ValueError("Unknown evaluation task")
    config_path = args.config or SOURCE / "configs" / (args.task + ".json")
    config = json.loads(Path(config_path).read_text())
    if config["task"] != args.task:
        raise ValueError("configuration task does not match requested benchmark")
    native_processes = getattr(args, "native_processes", 1)
    split = getattr(args, "split", "test")
    if split != "test" and (
        split != "train" or args.task != "gsm8k"
        or args.limit is None
    ):
        raise ValueError("train diagnosis requires native GSM evaluation with an explicit limit")
    if native_processes < 1 or (args.limit is not None and args.limit < native_processes):
        raise ValueError("native evaluation needs at least one request per process")
    (output, root) = external_output(args.output, args.artifact_root)
    model_args = resolve_model_args(config, tier=getattr(args, "tier", None),
        feature3=getattr(args, "feature3", None), overrides=model_overrides(getattr(args, "model_arg", None)))
    generation_kwargs = dict(config["gen_kwargs"])
    if "gen_length" in model_args:
        generation_kwargs["max_gen_toks"] = int(model_args["gen_length"])
    model_args.setdefault("seed", 1234)
    model_args.update(
        model_path=str(args.model_path.resolve()),
        spinquant_artifact_dir=str(args.head_artifact.resolve()),
        silu_table=str(args.silu_table.resolve()),
        trace_output_path=str(output / "trace.jsonl"),
        source_commit=args.source_version,
    )
    for key, value in model_args.items():
        if any((character in str(value) for character in (",", "\n", "\r"))):
            raise ValueError(
                f"{key}: lm-eval model arguments cannot contain commas or newlines"
            )
    serialized = ",".join(
        (
            f"{key}={(str(value).lower() if isinstance(value, bool) else value)}"
            for (key, value) in model_args.items()
        )
    )
    task_name = f"{args.task}_native"
    if split == "train":
        task_name = "gsm8k_train_native"
    command = [
        sys.executable,
        "-u",
        "-B",
        "-m",
        "evaluation.model",
        "--model",
        "quantized_llada",
        "--include_path",
        str(SOURCE / "evaluation/tasks"),
        "--tasks",
        task_name,
        "--batch_size",
        "1",
        "--device",
        args.device,
        "--gen_kwargs",
        ",".join((f"{k}={v}" for (k, v) in generation_kwargs.items())),
        "--num_fewshot",
        str(config["num_fewshot"]),
        "--seed",
        str(config["seed"]),
        "--model_args",
        serialized,
        "--log_samples",
        "--trust_remote_code",
        "--output_path",
        str(output),
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be a positive sample count")
        command.extend(["--limit", str(args.limit)])
    if native_processes > 1:
        command = [
            sys.executable, "-B", "-m", "accelerate.commands.launch",
            "--multi_gpu", "--num_machines", "1",
            "--num_processes", str(native_processes),
            "--mixed_precision", "no", "--module", "evaluation.model",
            *command[5:],
        ]
    env = {k: v for (k, v) in os.environ.items() if not k.startswith("SUPRA_")}
    dependency_path = os.environ.get("SUPRA_EVALUATION_DEPENDENCIES", "")
    pythonpath = str(SOURCE) + (os.pathsep + dependency_path if dependency_path else "")
    env.update(
        SUPRA_ALGORITHM_ROOT=str(root),
        SUPRA_RELOCATED_INSTRUCT_CHECKPOINT=str(args.model_path.resolve()),
        SUPRA_SPINQUANT_PARENT_ARTIFACT_DIR=str(args.weight_artifact.resolve()),
        PYTHONPATH=pythonpath,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        HF_HOME=str(root / "cache/huggingface"),
        HF_DATASETS_CACHE=str(root / "cache/datasets"),
        TMPDIR=str(root / "cache/tmp"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
        TORCH_EXTENSIONS_DIR=str(root / "cache/torch_extensions"),
    )
    return (command, env, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("gsm8k",), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("test", "train"), default="test",
                        help="dataset split for evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--weight-artifact", type=Path, required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
    parser.add_argument("--silu-table", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-processes", type=int, default=1,
                        help="use lm-eval's native rank partition over this many local processes")
    parser.add_argument("--tier", choices=("baseline", "feature1", "feature12", "feature123"))
    parser.add_argument("--model-arg", action="append", help="override one model setting as NAME=JSON_VALUE")
    parser.add_argument("--feature3", choices=("on", "off"))
    parser.add_argument("--source-version", default="working-tree")
    parser.add_argument(
        "--limit", type=int, help="maximum number of benchmark requests"
    )
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    (command, env, output) = build_command(args)
    if args.print_command:
        print(shlex.join(command))
        return
    for path in (
        args.model_path / "config.json",
        args.weight_artifact / "tensor_manifest.json",
        args.head_artifact / "tensor_manifest.json",
        args.silu_table,
    ):
        if not path.is_file():
            raise ValueError(f"missing required model input: {path}")
    if output.exists():
        raise ValueError("use a fresh output directory")
    for key in tuple(os.environ):
        if key.startswith("SUPRA_"):
            del os.environ[key]
    os.environ.update(env)
    from quantization.artifact import (
        SpinQuantArtifactReader,
        require_train_only_artifact,
    )

    for path in (args.weight_artifact, args.head_artifact):
        require_train_only_artifact(SpinQuantArtifactReader(path, verify_files=False))
    if (
        json.loads(args.silu_table.read_text()).get("train_only_calibration")
        is not True
    ):
        raise ValueError("SiLU requires train-only calibration provenance")
    output.mkdir(parents=True)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    code = 1
    process = None
    interrupted = None
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt

    for sig in previous_handlers:
        signal.signal(sig, stop)
    try:
        with (output / "run.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = process.wait()
        if code == 0 and args.native_processes > 1:
            paths = [output / f"trace.rank_{rank:02d}.jsonl"
                     for rank in range(args.native_processes)]
            if any(not path.is_file() for path in paths):
                raise FileNotFoundError("a distributed rank did not write its trace")
            with (output / "trace.jsonl").open("x") as merged:
                for path in paths:
                    with path.open() as source:
                        shutil.copyfileobj(source, merged)
    except KeyboardInterrupt:
        code = 128 + (interrupted or signal.SIGINT)
        if process is not None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    except Exception:
        code = 1
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        (output / "exit_code").write_text(str(code) + "\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
