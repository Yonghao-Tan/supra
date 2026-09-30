"""Collect completed evaluation outputs and benchmark scores."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from evaluation.generate import SOURCE, external_output
from evaluation.summary import (
    effective_trace_configuration,
    effective_result_configuration,
    require_matching_configuration,
)
from evaluation.summary import _unique_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("gsm8k",), required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="collect an explicitly partial evaluation",
    )
    args = parser.parse_args()
    (output, root) = external_output(args.output, args.artifact_root)
    if output.exists():
        raise ValueError("use a fresh output directory")
    shards = []
    for directory in args.inputs:
        if (directory / "exit_code").read_text().strip() != "0":
            raise ValueError(f"evaluation did not complete: {directory}")
        shards.extend(
            [directory]
            if (directory / "trace.jsonl").is_file()
            else sorted(directory.glob("shard_[0-9][0-9]"))
        )
    if not shards or len({p.resolve() for p in shards}) != len(shards):
        raise ValueError("empty or repeated input shards")
    identities = set()
    request_count = 0
    (trace_configuration, result_configuration) = (None, None)
    for shard in shards:
        result = json.loads(_unique_file(shard, "results_*.json").read_text())
        result_configuration = require_matching_configuration(
            result_configuration,
            effective_result_configuration(result),
            source=str(shard),
        )
        with (shard / "trace.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                trace_configuration = require_matching_configuration(
                    trace_configuration,
                    effective_trace_configuration(row),
                    source=str(shard),
                )
                identities.add(row["artifact_manifest_sha256"])
                if (
                    row["numeric_coverage"]["candidate"]["schedule"]
                    != "candidate-online-bf16-lut-lanes64/v1"
                ):
                    raise ValueError("unexpected Candidate numeric schedule")
                request_count += 1
    full_count = 1319
    expected = request_count if args.partial else full_count
    if not 0 < expected <= full_count:
        raise ValueError("invalid request count")
    if request_count != expected or len(identities) != 1:
        raise ValueError(
            f"expected {expected} requests from one artifact, got {request_count}"
        )
    output.mkdir(parents=True)
    for index, shard in enumerate(shards):
        shutil.copytree(shard, output / f"shard_{index:02d}")
    dependencies = os.environ.get("SUPRA_EVALUATION_DEPENDENCIES", "")
    pythonpath = str(SOURCE) + (os.pathsep + dependencies if dependencies else "")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=pythonpath,
        SUPRA_ALGORITHM_ROOT=str(root),
        HF_HOME=str(root / "cache/huggingface"),
        HF_DATASETS_CACHE=str(root / "cache/datasets"),
        TMPDIR=str(root / "cache/tmp"),
        OMP_NUM_THREADS="4",
    )
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    with (output / "collection.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "evaluation.summary",
                "--root",
                str(output),
                "--task",
                args.task,
                "--expected-samples",
                str(expected),
                "--expected-shards",
                str(len(shards)),
                *(["--partial"] if args.partial else []),
                "--output",
                str(output / "summary.json"),
            ],
            env=env,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    summary = json.loads((output / "summary.json").read_text())
    summary["evaluation_scope"] = "partial" if args.partial else "full"
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"quality": summary["quality"], "nfe": summary["nfe"]}, indent=2))


if __name__ == "__main__":
    main()
