"""Summarize matched, disjoint benchmark shards and their actual forward traces."""

from __future__ import annotations
from typing import Any, Dict, Iterable, List
import argparse
from importlib import metadata
import json
from pathlib import Path
import re
import statistics
import datasets
from lm_eval.utils import simple_parse_args_string


def _unique_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} below {directory}")
    if len(files) != 1:
        raise ValueError(f"Ambiguous {pattern} below {directory}: {len(files)} files; select one run")
    return files[0]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_index(
    samples: Iterable[Dict[str, Any]], metric: str, preferred_filter: str
) -> Dict[int, Dict[str, Any]]:
    indexed: Dict[int, Dict[str, Any]] = {}
    seen = set()
    for row in samples:
        if metric not in row:
            continue
        doc_id = int(row["doc_id"])
        key = (doc_id, row.get("filter"))
        if key in seen:
            raise ValueError(f"Duplicate sample doc_id/filter={key}")
        seen.add(key)
        if row.get("filter") == preferred_filter or doc_id not in indexed:
            indexed[doc_id] = row
    return indexed


def _trace_index(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        doc_id = int(row["doc_id"])
        if doc_id in result:
            raise ValueError(f"Duplicate trace doc_id={doc_id}")
        result[doc_id] = row
    return result


def load_case(
    directory: Path, task_name: str, metric: str, filter_name: str
) -> dict[str, dict[str, Any]]:
    samples = _sample_index(
        _read_jsonl(_unique_file(directory, f"samples_{task_name}_*.jsonl")),
        metric,
        filter_name,
    )
    traces = _trace_index(_read_jsonl(directory / "trace.jsonl"))
    if set(samples) != set(traces):
        raise ValueError(f"{directory}: sample/trace mismatch")
    result = {}
    for doc_id, sample in samples.items():
        doc_hash = sample.get("doc_hash")
        if not isinstance(doc_hash, str) or doc_hash in result:
            raise ValueError(f"{directory}: invalid or duplicate doc hash")
        result[doc_hash] = {"sample": sample, "trace": traces[doc_id]}
    return result


def load_filter(
    directory: Path, task_name: str, metric: str, filter_name: str
) -> dict[str, dict[str, Any]]:
    rows = _sample_index(
        _read_jsonl(_unique_file(directory, f"samples_{task_name}_*.jsonl")),
        metric,
        filter_name,
    )
    return {
        row["doc_hash"]: row
        for row in rows.values()
        if row.get("filter") == filter_name
    }


def completed_shards(directory: Path) -> list[Path]:
    shards = []
    for shard in sorted(directory.glob("shard*")):
        if shard.is_dir() and list(shard.glob("**/results_*.json")):
            shards.append(shard)
    if not shards:
        raise FileNotFoundError(f"no completed shards under {directory}")
    return shards


def load_config(
    directory: Path, task_name: str, metric: str, filter_name: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    combined: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    doc_shards: dict[str, str] = {}
    for shard in completed_shards(directory):
        values = load_case(shard, task_name, metric, filter_name)
        duplicate = set(combined) & set(values)
        if duplicate:
            raise ValueError(
                f"{directory}: duplicate documents across shards: {sorted(duplicate)[:3]}"
            )
        combined.update(values)
        sources[shard.name] = str(_unique_file(shard, "results_*.json"))
        doc_shards.update({doc_hash: shard.name for doc_hash in values})
    return (combined, sources, doc_shards)


def load_config_filter(
    directory: Path, task_name: str, metric: str, filter_name: str
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for shard in completed_shards(directory):
        values = load_filter(shard, task_name, metric, filter_name)
        duplicate = set(combined) & set(values)
        if duplicate:
            raise ValueError(
                f"{directory}: duplicate {filter_name} documents across shards"
            )
        combined.update(values)
    return combined


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)




def effective_trace_configuration(trace):
    """Compare executed numerics and generation without per-request positions."""
    fields = (
        "numeric_coverage",
        "generation_mode",
        "decoding",
        "precision",
        "instruct_prompt_mode",
        "confidence_eos_eot_inf",
        "candidate_numeric_schedule",
        "candidate_suppression_schedule",
        "artifact_manifest_sha256",
        "lm_head_weight_bits",
        "seed",
        "mask_id",
        "generation_protocol",
        "logical_a4a8",
        "spinquant_variant",
        "spinquant_execution",
    )
    result = {key: trace.get(key) for key in fields}
    diagnostic = dict(trace.get("deployment_diagnostic") or {})
    if trace.get("numeric_coverage"):
        diagnostic.pop("silu_table", None)
    result["deployment_diagnostic"] = diagnostic
    return result


def effective_result_configuration(result):
    config = result.get("config", {})
    model_args = config.get("model_args", {})
    if isinstance(model_args, str):
        model_args = simple_parse_args_string(model_args)
    model_args = dict(model_args)
    for key in (
        "trace_output_path",
        "device",
        "model_path",
        "spinquant_artifact_dir",
        "silu_table",
        "source_commit",
    ):
        model_args.pop(key, None)
    task_fields = (
        "num_fewshot",
        "generation_kwargs",
        "output_type",
        "doc_to_text",
        "doc_to_target",
        "target_delimiter",
        "fewshot_delimiter",
        "fewshot_split",
        "test_split",
        "metric_list",
        "filter_list",
    )

    def normalize_callable_repr(value):
        if isinstance(value, dict):
            return {key: normalize_callable_repr(item) for (key, item) in value.items()}
        if isinstance(value, list):
            return [normalize_callable_repr(item) for item in value]
        if isinstance(value, str):
            match = re.fullmatch("<function ([\\w.<>]+) at 0x[0-9a-fA-F]+>", value)
            if match:
                return "<function " + match.group(1) + ">"
        return value

    task_configurations = {
        json.dumps(
            normalize_callable_repr({key: task.get(key) for key in task_fields}),
            sort_keys=True,
        )
        for task in result.get("configs", {}).values()
    }
    return dict(
        model_args=model_args,
        task_configurations=sorted(task_configurations),
        evaluation={
            key: config.get(key)
            for key in (
                "batch_size",
                "gen_kwargs",
                "num_fewshot",
                "random_seed",
                "numpy_seed",
                "torch_seed",
                "fewshot_seed",
            )
        },
    )


def require_matching_configuration(expected, actual, *, source):
    if expected is not None and expected != actual:
        differing = [
            key
            for key in expected.keys() | actual.keys()
            if expected.get(key) != actual.get(key)
        ]
        raise ValueError(
            f"effective evaluation configuration differs in {source}: {differing}"
        )
    return actual


def _source_commit_from_results(paths: dict[str, str]) -> str:
    commits: set[str] = set()
    configuration = None
    for path in paths.values():
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        configuration = require_matching_configuration(
            configuration, effective_result_configuration(result), source=path
        )
        model_args = result.get("config", {}).get("model_args", "")
        if isinstance(model_args, str):
            model_args = simple_parse_args_string(model_args)
        source = (
            model_args.get("source_commit") if isinstance(model_args, dict) else None
        )
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"missing source_commit in {path}")
        commits.add(source)
    if len(commits) != 1:
        raise ValueError("source commit differs across shard result files")
    return next(iter(commits))


def execution_rows(event):
    """Count executed rows once per forward, separating L0 from deeper layers."""
    positions = event.get("input_positions", [])
    rows = len(positions) if positions else int(event["input_length"])
    if rows <= 0 or (positions and len(set(positions)) != rows):
        raise ValueError("input row mapping is invalid")
    layer0 = int(event["layer0_rows"])
    l0_a4, l0_a8 = (int(event[f"layer0_a{bits}_rows"]) for bits in (4, 8))
    if event["forward_kind"] == "full_sequence":
        a4, a8 = l0_a4, l0_a8
    else:
        a4, a8 = (int(event[f"active_a{bits}_rows"]) for bits in (4, 8))
    if min(a4, a8, l0_a4, l0_a8) < 0 or a4 + a8 != rows or l0_a4 + l0_a8 != layer0:
        raise ValueError("execution precision counts do not match executed rows")
    if event.get("boundary_global_layers", 1) != 1:
        raise ValueError("workload summary requires one L0 and the same set in all deep layers")
    return dict(
        input_rows=rows,
        a4_rows=a4,
        a8_rows=a8,
        layer0_rows=layer0,
        layer0_a4_rows=l0_a4,
        layer0_a8_rows=l0_a8,
        deep_rows=rows,
        deep_a4_rows=a4,
        deep_a8_rows=a8,
    )


def summarize(
    root: Path, task: str, expected_samples: int, expected_shards: int, *, partial=False,
    recover_missing_exit_code=False,
):
    if task != "gsm8k":
        raise ValueError("Unknown evaluation task")
    full_count = 1319
    if not 0 < expected_samples <= full_count or (
        not partial and expected_samples != full_count
    ):
        raise ValueError("full evaluation requires every benchmark request")
    shards = completed_shards(root)
    if len(shards) != expected_shards:
        raise ValueError(
            f"expected {expected_shards} completed shards, got {len(shards)}"
        )
    missing_exit_codes = []
    for shard in shards:
        exit_code = shard / "exit_code"
        if not exit_code.is_file() and recover_missing_exit_code:
            missing_exit_codes.append(shard.name)
        elif not exit_code.is_file() or exit_code.read_text().strip() != "0":
            raise ValueError(f"unfinished evaluation shard: {shard}")
    metric, filter_name = "exact_match", "strict-match"
    (cases, result_sources, doc_shards) = load_config(root, task, metric, filter_name)
    if len(cases) != expected_samples:
        raise ValueError(f"expected {expected_samples} documents, got {len(cases)}")
    source_commit = _source_commit_from_results(result_sources)
    configuration = None
    parameter_sets = {}
    nfe_values = []
    forward_work = {}
    for doc_hash, case in cases.items():
        trace = case["trace"]
        configuration = require_matching_configuration(
            configuration, effective_trace_configuration(trace), source=doc_hash
        )
        if trace.get("doc_hash") != doc_hash:
            raise ValueError(f"trace document identity mismatch: {doc_hash}")
        if trace.get("source_commit") not in (None, source_commit):
            raise ValueError("trace and result source versions differ")
        if case["sample"].get("filter") != filter_name:
            raise ValueError(f"missing {filter_name} score for {doc_hash}")
        (events, nfe) = (trace.get("trace"), int(trace.get("nfe", -1)))
        if not isinstance(events, list) or len(events) != nfe or nfe <= 0:
            raise ValueError(f"trace/NFE mismatch: {doc_hash}")
        if int(events[-1].get("masked_after", -1)) != 0:
            raise ValueError(f"unfinished generation: {doc_hash}")
        nfe_values.append(nfe)
        for key in (
            "feature1_parameters",
            "feature2_parameters",
            "feature2_precision_parameters",
            "feature3_parameters",
            "feature3_dynamic_block_parameters",
        ):
            value = trace.get(key)
            if key in parameter_sets and parameter_sets[key] != value:
                raise ValueError(f"{key} differs across requests")
            parameter_sets[key] = value
        for event in events:
            phase = event["forward_kind"]
            work = forward_work.setdefault(
                phase,
                {"forwards": 0},
            )
            work["forwards"] += 1
            for key, value in execution_rows(event).items():
                work[key] = work.get(key, 0) + value
    raw_scores = [float(case["sample"][metric]) for case in cases.values()]
    if any((score not in (0.0, 1.0) for score in raw_scores)):
        raise ValueError("single-completion scores must be binary")
    strict_or_raw = int(sum(raw_scores))
    flexible = load_config_filter(root, task, metric, "flexible-extract")
    if set(flexible) != set(cases):
        raise ValueError("strict/flexible document sets differ")
    scores = [float(row[metric]) for row in flexible.values()]
    if any((score not in (0.0, 1.0) for score in scores)):
        raise ValueError("flexible scores must be binary")
    quality = {"strict_count": strict_or_raw, "flexible_count": int(sum(scores))}
    return {
        "schema_version": "quantized-llada-evaluation-summary/v2",
        **({"completion_recovery": {
            "status": "complete_artifacts_validated_exit_status_unavailable",
            "shards_without_exit_code": missing_exit_codes,
            "original_process_exit_code": None,
        }} if missing_exit_codes else {}),
        "task": task,
        "sample_count": len(cases),
        "evaluation_scope": "partial" if partial else "full",
        "completed_shards": len(shards),
        "generation_mode": configuration["generation_mode"],
        "precision": configuration["precision"],
        "quality": quality,
        "nfe": {
            "mean": statistics.fmean(nfe_values),
            "p50": _percentile(nfe_values, 0.5),
            "p90": _percentile(nfe_values, 0.9),
        },
        "forward_work": forward_work,
        "forward_work_semantics": {
            "unit": "rows summed per forward, without layer multiplication",
            "input_rows_a4_a8": "executed packed set; boundary uses the deep set",
            "layer0": "actual L0 execution, including full-sequence scout",
            "deep": "set executed by each later layer, counted once per forward",
        },
        "parameters": parameter_sets,
        "source_commit": source_commit,
        "artifact_manifest_sha256": configuration["artifact_manifest_sha256"],
        "lm_head_weight_bits": configuration["lm_head_weight_bits"],
        "evaluator_versions": {
            "lm_eval": metadata.version("lm_eval"),
            "datasets": datasets.__version__,
        },
        "source_files": {"lm_eval": result_sources},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("gsm8k",), required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument(
        "--recover-missing-exit-code", action="store_true",
        help="collect complete saved outputs after workers exit without writing an exit-code file",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from evaluation.generate import external_output
    import os

    (output, _) = external_output(args.output, Path(os.environ["SUPRA_ALGORITHM_ROOT"]))
    result = summarize(
        args.root,
        args.task,
        args.expected_samples,
        args.expected_shards,
        partial=args.partial,
        recover_missing_exit_code=args.recover_missing_exit_code,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"quality": result["quality"], "nfe": result["nfe"]}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
