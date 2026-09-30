"""Calibrate, export and evaluate quantized LLaDA using the default task configs."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "llada"


def command(script, *arguments):
    module = str(Path(script).with_suffix("")).replace("/", ".")
    return [sys.executable, "-u", "-B", "-m", module, *map(str, arguments)]


def data_index_directory(args):
    run = args.run_dir.resolve()
    default = run / "data_indices"
    return (getattr(args, "data_index_dir", None) or default).resolve()


def build_stages(args):
    if any(task != "gsm8k" for task in args.tasks):
        raise ValueError("Unknown evaluation task")
    run = args.run_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    release_root = (
        ROOT.parent if ROOT.name == "algorithm" and (ROOT.parent / "hardware").is_dir()
        else ROOT
    )
    project = next((p for p in (ROOT, *ROOT.parents) if (p / ".git").exists()), release_root)
    run.relative_to(artifact_root)
    if run == artifact_root or run == project or project in run.parents:
        raise ValueError("run-dir must be a child of an external artifact-root")
    gpus = args.gpus.split(",")
    if (
        not gpus
        or any((not gpu.strip() for gpu in gpus))
        or len(set(gpus)) != len(gpus)
    ):
        raise ValueError("gpus must contain distinct device IDs or UUIDs")
    config_path = getattr(args, "config", None)
    config_document = {} if config_path is None else json.loads(config_path.read_text())
    calibration = dict(train_samples_per_task=128, validation_samples_per_task=16,
                       gen_length=256, steps=256)
    calibration_overrides = config_document.get("calibration", {})
    if not isinstance(calibration_overrides, dict) or set(calibration_overrides) - set(calibration):
        raise ValueError("calibration supports train_samples_per_task, validation_samples_per_task, gen_length and steps")
    calibration.update(calibration_overrides)
    if any(type(value) is not int or value < 1 for value in calibration.values()):
        raise ValueError("calibration counts, gen_length and steps must be positive integers")
    if calibration["gen_length"] % 32 or calibration["steps"] % (calibration["gen_length"] // 32):
        raise ValueError("calibration gen_length must be divisible by 32 and steps by the block count")
    calibration_manifest = getattr(args, "calibration_manifest", None)
    calibration_data_dir = getattr(args, "calibration_data_dir", None)
    if calibration_data_dir is not None and calibration_manifest is None:
        raise ValueError("calibration-data-dir requires calibration-manifest")
    stages = []
    (initial, model) = (run / "initial", run / "model")
    data = run / "data"
    indices = data_index_directory(args)
    common = (
        "--model-path",
        args.model_path.resolve(),
        "--artifact-dir",
        initial / "artifact_w4",
        "--w8-head-dir",
        initial / "artifact_w8_head",
        "--dataset-manifest",
        calibration_manifest.resolve() if calibration_manifest is not None else indices / "train.json",
        *(("--data-dir", calibration_data_dir.resolve()) if calibration_data_dir is not None
          else ("--data-dir", data / "train") if calibration_manifest is None else ()),
        "--gen-length", calibration["gen_length"],
        "--steps", calibration["steps"],
        "--train-only-calibration",
        "--anchor",
        "n4",
        "--seed",
        20260806,
        "--data-seed",
        20260908,
        "--device",
        "cuda",
    )
    if args.mode in ("calibrate", "all"):
        stages.extend(
            [
                (
                    "prepare-data",
                    [
                        command(
                            "calibration/prepare.py",
                            "--output-dir",
                            data / "parent",
                            "--manifest",
                            indices / "parent.json",
                            "--seed",
                            20260825,
                        )
                    ],
                    [],
                    [],
                ),
                (
                    "split-data",
                    [
                        command(
                            "calibration/split.py",
                            "--parent-manifest",
                            indices / "parent.json",
                            "--output-dir",
                            data / "train",
                            "--manifest",
                            indices / "train.json",
                        )
                    ],
                    [],
                    [],
                ),
                (
                    "initial-weights",
                    [
                        command(
                            "calibration/initialize.py",
                            "--checkpoint-dir",
                            args.model_path.resolve(),
                            "--output-dir",
                            initial / "artifact_w4",
                            "--variant",
                            "fixed-r1r2-r4-only",
                            "--group-size",
                            -1,
                            "--a8-period",
                            16,
                            "--device",
                            "cuda:0",
                        )
                    ],
                    gpus[:1],
                    [],
                ),
                (
                    "initial-head",
                    [
                        command(
                            "quantization/head.py",
                            "export-rtn-w8",
                            "--parent-artifact",
                            initial / "artifact_w4",
                            "--output-artifact",
                            initial / "artifact_w8_head",
                        )
                    ],
                    [],
                    [],
                ),
            ]
        )
        if calibration_manifest is not None:
            stages = [stage for stage in stages if stage[0] not in {"prepare-data", "split-data"}]
        (captures, capture_roots, hessians, hessian_roots) = ([], [], [], [])
        for task in ("gsm8k", "humaneval"):
            for split, count in (("train", calibration["train_samples_per_task"]),
                                 ("validation", calibration["validation_samples_per_task"])):
                destination = run / "captures" / task / split
                capture_roots.append(destination)
                shard_count = min(len(gpus), count)
                for shard in range(shard_count):
                    captures.append(
                        command(
                            "calibration/capture.py",
                            "--mode",
                            "capture",
                            *common,
                            "--record-task",
                            task,
                            "--split",
                            split,
                            "--samples-per-task",
                            count,
                            "--shard-index",
                            shard,
                            "--shard-count",
                            shard_count,
                            "--human-feature3",
                            "--gsm-feature3",
                            "--full-prefix-oracle-block",
                            1,
                            "--silu-errors",
                            "--output-dir",
                            destination / f"shard_{shard:02d}",
                        )
                    )
            destination = run / "hessians" / task
            hessian_roots.append(destination)
            shard_count = min(len(gpus), calibration["train_samples_per_task"])
            for shard in range(shard_count):
                hessians.append(
                    command(
                        "calibration/capture.py",
                        "--mode",
                        "linear-hessians",
                        *common,
                        "--record-task",
                        task,
                        "--split",
                        "train",
                        "--samples-per-task",
                        calibration["train_samples_per_task"],
                        "--shard-index",
                        shard,
                        "--shard-count",
                        shard_count,
                        "--capture-root",
                        run / "captures" / task / "train",
                        "--output-dir",
                        destination / f"shard_{shard:02d}",
                    )
                )
        stages.extend(
            [
                ("capture", captures, gpus, capture_roots),
                ("hessians", hessians, gpus, hessian_roots),
            ]
        )
        solves = [
            command(
                "calibration/solve.py",
                "--checkpoint",
                args.model_path.resolve(),
                "--artifact-dir",
                initial / "artifact_w4",
                "--hessian-root",
                run / "hessians/humaneval",
                "--expected-requests",
                calibration["train_samples_per_task"],
                "--preserve-hessian-root",
                run / "hessians/gsm8k",
                "--preserve-expected-requests",
                calibration["train_samples_per_task"],
                "--train-only-calibration",
                "--joint-relative-error",
                "--scale-search",
                "--group-size",
                -1,
                "--shard-index",
                shard,
                "--shard-count",
                len(gpus),
                "--device",
                "cuda",
                "--output-dir",
                run / "solve" / f"shard_{shard:02d}",
            )
            for shard in range(len(gpus))
        ]
        stages.extend(
            [
                ("gptq", solves, gpus, [run / "solve"]),
                (
                    "silu",
                    [
                        command(
                            "calibration/silu.py",
                            "--calibration-root",
                            run / "captures/gsm8k/train",
                            "--check-root",
                            run / "captures/gsm8k/validation",
                            "--secondary-calibration-root",
                            run / "captures/humaneval/train",
                            "--secondary-check-root",
                            run / "captures/humaneval/validation",
                            "--train-only-calibration",
                            "--monotone-positive-branch",
                            "--output",
                            model / "silu_table_monotone.json",
                        )
                    ],
                    [],
                    [],
                ),
                (
                    "export",
                    [
                        command(
                            "quantization/export.py",
                            "--checkpoint",
                            args.model_path.resolve(),
                            "--reference-artifact",
                            initial / "artifact_w8_head",
                            "--quantized-linear-root",
                            run / "solve",
                            "--group-size",
                            -1,
                            "--train-only-calibration",
                            "--device",
                            "cuda",
                            "--output-parent",
                            model / "artifact_w4",
                            "--output-child",
                            model / "artifact_w8_head",
                        )
                    ],
                    gpus[:1],
                    [],
                ),
            ]
        )
    if args.mode in ("evaluate", "all"):
        model = args.model_artifacts.resolve() if args.mode == "evaluate" else model
        config = getattr(args, "config", None)
        if config is not None and (len(args.tasks) != 1 or config_document["task"] != args.tasks[0]):
            raise ValueError("config task must match the single selected task")
        for task in args.tasks:
            destination = run / "evaluation" / task
            evaluations = [
                command(
                    "evaluation/generate.py",
                    "--task",
                    task,
                    "--model-path",
                    args.model_path.resolve(),
                    "--weight-artifact",
                    model / "artifact_w4",
                    "--head-artifact",
                    model / "artifact_w8_head",
                    "--silu-table",
                    model / "silu_table_monotone.json",
                    "--artifact-root",
                    artifact_root,
                    "--output",
                    destination / "shard_00",
                    "--native-processes", len(gpus),
                    *(("--feature3", args.feature3) if getattr(args, "feature3", None) else ()),
                    *(("--tier", args.tier) if getattr(args, "tier", None) else ()),
                    *(("--config", config.resolve()) if config is not None else ()),
                    *(value for item in getattr(args, "model_arg", None) or () for value in ("--model-arg", item)),
                    "--source-version",
                    args.source_version,
                )
            ]
            evaluation_slots = [",".join(gpus)]
            stages.append((f"evaluate-{task}", evaluations, evaluation_slots, [destination]))
            limit = getattr(args, "limit_per_process", None)
            if limit is not None:
                if limit < 1:
                    raise ValueError("limit-per-process must be positive")
                for argv in evaluations:
                    argv.extend(["--limit", str(limit * len(gpus))])
            stages.append(
                (
                    f"collect-{task}",
                    [
                        command(
                            "evaluation/collect.py",
                            "--task",
                            task,
                            "--artifact-root",
                            artifact_root,
                            "--inputs",
                            destination,
                            "--output",
                            run / "results" / task,
                            *(["--partial"] if limit is not None else []),
                        )
                    ],
                    [],
                    [],
                )
            )
    return stages


def run_stage(name, commands, gpus, env, run, status_roots):
    pending = iter(enumerate(commands))
    active = {}
    slots = gpus or [""]
    success = False
    try:
        while True:
            for slot in slots:
                if slot in active:
                    continue
                item = next(pending, None)
                if item is None:
                    continue
                (index, argv) = item
                log = (run / "logs" / f"{name}_{index:02d}.log").open("w")
                try:
                    process = subprocess.Popen(
                        argv,
                        cwd=ROOT,
                        env=dict(env, CUDA_VISIBLE_DEVICES=slot),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except BaseException:
                    log.close()
                    raise
                active[slot] = (process, log, index)
                print(f"{name} job {index}: started on {slot or 'CPU'}", flush=True)
            if not active:
                break
            for slot, (process, log, index) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[slot]
                if code:
                    raise RuntimeError(
                        f"{name} job {index} exited {code}; see logs/{name}_{index:02d}.log"
                    )
                print(f"{name} job {index}: complete", flush=True)
            time.sleep(0.1)
        success = True
    finally:
        for process, _, _ in active.values():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 35
        for process, log, _ in active.values():
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            finally:
                log.close()
        for root in status_roots:
            root.mkdir(parents=True, exist_ok=True)
            (root / "exit_code").write_text("0\n" if success else "1\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("evaluate", "calibrate", "all"), default="evaluate"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--model-artifacts",
        type=Path,
        help="contains artifact_w4, artifact_w8_head and silu_table_monotone.json",
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--data-index-dir",
        type=Path,
        help="directory for calibration dataset indices",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="comma-separated device IDs or UUIDs; one process per device",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("gsm8k",),
        default=["gsm8k"],
    )
    parser.add_argument("--config", type=Path, help="task JSON with optional calibration settings")
    parser.add_argument("--calibration-manifest", type=Path,
                        help="reuse an existing train-source calibration index with disjoint train and validation records")
    parser.add_argument("--calibration-data-dir", type=Path,
                        help="override the reused calibration index data directory")
    parser.add_argument("--tier", choices=("baseline", "feature1", "feature12", "feature123"))
    parser.add_argument("--model-arg", action="append", help="override one model setting as NAME=JSON_VALUE")
    parser.add_argument("--feature3", choices=("on", "off"))
    parser.add_argument(
        "--limit-per-process",
        type=int,
        help="bounded partial evaluation; omit for full benchmarks",
    )
    parser.add_argument(
        "--source-version", required=True, help="release tag or source commit"
    )
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args()
    if args.mode == "evaluate" and args.model_artifacts is None:
        parser.error("evaluate requires --model-artifacts")
    if args.mode != "evaluate" and args.model_artifacts is not None:
        parser.error("calibrate/all create model artifacts inside run-dir")
    args.gpus = ",".join((gpu.strip() for gpu in args.gpus.split(",")))
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must be distinct")
    stages = build_stages(args)
    if args.print_commands:
        for name, commands, gpus, _ in stages:
            print(f"# {name}; devices: {','.join(gpus) or 'CPU'}")
            for argv in commands:
                print(shlex.join(argv))
        return
    run = args.run_dir.resolve()
    if run.exists():
        raise ValueError("use a fresh run-dir")
    if not (args.model_path / "config.json").is_file():
        raise ValueError("model-path is missing config.json")
    if args.mode in ("calibrate", "all") and args.calibration_manifest is None:
        indices = data_index_directory(args)
        if indices.exists():
            raise ValueError("use a fresh data-index-dir")
    if args.mode in ("calibrate", "all") and args.calibration_manifest is not None:
        if not args.calibration_manifest.is_file():
            raise ValueError("calibration-manifest is missing")
        if args.calibration_data_dir is not None and not args.calibration_data_dir.is_dir():
            raise ValueError("calibration-data-dir is missing")
    (run / "logs").mkdir(parents=True)
    env = {
        key: value for (key, value) in os.environ.items() if not key.startswith("SUPRA_")
    }
    root = args.artifact_root.resolve()
    env.update(
        PYTHONPATH=str(SOURCE),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        SUPRA_ALGORITHM_ROOT=str(root),
        SUPRA_RELOCATED_INSTRUCT_CHECKPOINT=str(args.model_path.resolve()),
        SUPRA_SPINQUANT_PARENT_ARTIFACT_DIR=str(run / "initial/artifact_w4"),
        HF_HOME=str(root / "cache/huggingface"),
        HF_DATASETS_CACHE=str(root / "cache/datasets"),
        TMPDIR=str(root / "cache/tmp"),
        TRITON_CACHE_DIR=str(root / "cache/triton"),
        TORCH_EXTENSIONS_DIR=str(root / "cache/torch_extensions"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + args.gpus,
            "--query-gpu=index,name,memory.total,uuid",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    (run / "run.json").write_text(
        json.dumps(
            dict(
                source_version=args.source_version,
                arguments={
                    key: str(value) if isinstance(value, Path) else value
                    for (key, value) in vars(args).items()
                },
                python=sys.version,
                packages={
                    name: metadata.version(name)
                    for name in ("torch", "triton", "transformers", "lm_eval")
                },
                gpu_info=gpu_info.stdout,
            ),
            indent=2,
        )
        + "\n"
    )
    previous = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    code = 1
    try:
        for name, commands, gpus, status_roots in stages:
            run_stage(name, commands, gpus, env, run, status_roots)
        code = 0
    finally:
        (run / "exit_code").write_text(str(code) + "\n")
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
