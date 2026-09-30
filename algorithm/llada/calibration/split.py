"""Split calibration records by problem group to keep train and validation disjoint."""

from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from datasets import load_dataset
from calibration.prepare import (
    GSM8K_DATASET,
    GSM8K_REVISION,
    HUMANEVAL_DATASET,
    HUMANEVAL_REVISION,
    _sha256_file,
    _sha256_string_set,
    _within,
    normalize_for_overlap,
    record_text,
)

CODE_CATEGORIES = {"code"}
MATH_CATEGORIES = {"gsm8k"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        prompts = [
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")).casefold() == "user"
        ]
        if prompts:
            return "\n".join(prompts)
    return str(record.get("text", ""))


def assistant_text(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        (
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")).casefold() == "assistant"
        )
    )


def normalize_code_problem(text: str) -> str:
    """Normalize obvious identifier-only code variants without erasing prose."""
    normalized = normalize_for_overlap(text)
    normalized = re.sub(
        "\\b(def|class)\\s+[a-z_][a-z0-9_]*",
        lambda match: f"{match.group(1)} <id>",
        normalized,
    )

    def normalize_arguments(match: re.Match[str]) -> str:
        arguments = re.sub("\\b[a-z_][a-z0-9_]*\\b", "<id>", match.group(1))
        arguments = re.sub("\\b\\d+(?:\\.\\d+)?\\b", "<num>", arguments)
        return f"( {arguments.strip()} )"

    normalized = re.sub("\\(\\s*([^()]{0,512})\\s*\\)", normalize_arguments, normalized)
    normalized = re.sub("`\\s*[a-z_][a-z0-9_]*\\s*`", "` <id> `", normalized)
    return " ".join(normalized.split())


def normalize_math_problem(text: str) -> str:
    normalized = normalize_for_overlap(text)
    normalized = re.sub("\\b\\d+(?:\\.\\d+)?\\b", "<num>", normalized)
    normalized = re.sub("\\b[a-z]\\b", "<var>", normalized)
    return " ".join(normalized.split())


def problem_signature(record: dict[str, Any]) -> tuple[str, str]:
    category = str(record["category"])
    prompt = prompt_text(record)
    if category in CODE_CATEGORIES:
        return ("code_identifier_normalized_v1", normalize_code_problem(prompt))
    if category in MATH_CATEGORIES:
        return ("math_number_variable_normalized_v1", normalize_math_problem(prompt))
    return ("normalized_prompt_v1", normalize_for_overlap(prompt))


def _source_problem_key(record: dict[str, Any]) -> str | None:
    upstream_id = record.get("upstream_id")
    source = record.get("source")
    if upstream_id:
        return f"upstream:{source or record.get('dataset', '')}:{upstream_id}"
    url = record.get("url")
    if url:
        return f"url:{url}"
    return None


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def assign_problem_groups(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    dsu = _DisjointSet(len(records))
    first_by_key: dict[str, int] = {}
    signatures: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        (mode, signature) = problem_signature(record)
        signatures.append((mode, signature))
        keys = [f"signature:{mode}:{signature}"]
        source_key = _source_problem_key(record)
        if source_key is not None:
            keys.append(source_key)
        for key in keys:
            if key in first_by_key:
                dsu.union(index, first_by_key[key])
            else:
                first_by_key[key] = index
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members_by_root[dsu.find(index)].append(index)
    groups: dict[str, list[int]] = {}
    for members in members_by_root.values():
        identities = sorted((str(records[index]["sample_id"]) for index in members))
        group_id = "pg:" + _sha256_text("\n".join(identities))[:24]
        groups[group_id] = members
        for index in members:
            (mode, signature) = signatures[index]
            records[index]["problem_group_id"] = group_id
            records[index]["problem_group_normalization"] = mode
            records[index]["problem_signature_sha256"] = _sha256_text(signature)
    return groups


def validate_group_splits(records, groups):
    """Check configured train/check assignments without moving selected requests."""
    counts = defaultdict(lambda: {"train": 0, "validation": 0})
    for group_id, members in groups.items():
        splits = {records[index]["split"] for index in members}
        if len(splits) != 1 or not splits <= {"train", "validation"}:
            raise ValueError(f"problem group crosses train/check splits: {group_id}")
        for index in members:
            record = records[index]
            counts[record["category"]][record["split"]] += 1
    return dict(problem_group_count=len(groups), counts=dict(counts))


def build_family_protected_signatures() -> dict[str, set[str]]:
    code_prompt: set[str] = set()
    code_solution: set[str] = set()
    for row in load_dataset(
        HUMANEVAL_DATASET,
        "openai_humaneval",
        split="test",
        revision=HUMANEVAL_REVISION,
        streaming=True,
    ):
        code_prompt.add(normalize_code_problem(str(row["prompt"])))
        code_solution.add(normalize_code_problem(str(row["canonical_solution"])))
    math_question = {
        normalize_math_problem(str(row["question"]))
        for row in load_dataset(
            GSM8K_DATASET, "main", split="test", revision=GSM8K_REVISION, streaming=True
        )
    }
    return {
        "humaneval_prompt": code_prompt,
        "humaneval_canonical_solution": code_solution,
        "gsm8k_question": math_question,
    }


def family_exclusion_reason(
    record: dict[str, Any], protected: dict[str, set[str]]
) -> str | None:
    category = str(record["category"])
    if category in CODE_CATEGORIES:
        if normalize_code_problem(prompt_text(record)) in protected["humaneval_prompt"]:
            return "identifier_normalized_humaneval_prompt"
        answer = assistant_text(record)
        if (
            answer
            and normalize_code_problem(answer)
            in protected["humaneval_canonical_solution"]
        ):
            return "identifier_normalized_humaneval_solution"
    if category in MATH_CATEGORIES:
        if normalize_math_problem(prompt_text(record)) in protected["gsm8k_question"]:
            return "number_variable_normalized_gsm8k_question"
    return None


def _read_parent_records(
    parent_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    records: list[dict[str, Any]] = []
    indices_by_source: dict[str, list[int]] = defaultdict(list)
    seen_sample_ids: set[str] = set()
    for source_name, source in parent_manifest["sources"].items():
        path = Path(source["external_path"])
        if _sha256_file(path) != source["sha256"]:
            raise ValueError(f"parent source SHA mismatch: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        expected = source["ordered_samples"]
        if len(lines) != len(expected):
            raise ValueError(f"parent source count mismatch: {path}")
        for line, identity in zip(lines, expected):
            record = json.loads(line)
            if (
                record.get("sample_id") != identity["sample_id"]
                or record.get("content_sha256") != identity["content_sha256"]
                or record.get("split") != identity["split"]
                or (record.get("category") != identity["category"])
            ):
                raise ValueError(f"parent ordered sample mismatch: {identity['sample_id']}")
            if record["sample_id"] in seen_sample_ids:
                raise ValueError(f"duplicate sample ID: {record['sample_id']}")
            if _sha256_text(record_text(record)) != record["content_sha256"]:
                raise ValueError(f"parent content SHA mismatch: {record['sample_id']}")
            seen_sample_ids.add(record["sample_id"])
            indices_by_source[source_name].append(len(records))
            records.append(record)
    return (records, indices_by_source)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algo_root = Path(os.environ["SUPRA_ALGORITHM_ROOT"]).resolve()
    algorithm_root = Path(__file__).resolve().parents[2]
    release_root = (
        algorithm_root.parent
        if algorithm_root.name == "algorithm" and (algorithm_root.parent / "hardware").is_dir()
        else algorithm_root
    )
    project_root = next(
        (p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
        release_root,
    )
    if _within(args.output_dir, project_root):
        raise ValueError("dataset payload must be outside the source repository")
    if not _within(args.output_dir, algo_root):
        raise ValueError(f"dataset output must be inside SUPRA_ALGORITHM_ROOT={algo_root}")
    if args.output_dir.exists() or args.manifest.exists():
        raise FileExistsError("dataset output directory and manifest must both be new")
    parent_manifest = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    (records, indices_by_source) = _read_parent_records(parent_manifest)
    protected = build_family_protected_signatures()
    exclusions: dict[str, list[str]] = defaultdict(list)
    for record in records:
        reason = family_exclusion_reason(record, protected)
        if reason is not None:
            exclusions[reason].append(str(record["sample_id"]))
    if exclusions:
        counts = {reason: len(ids) for (reason, ids) in exclusions.items()}
        raise RuntimeError(
            "parent dataset contains family-normalized protected matches: "
            + json.dumps(counts, sort_keys=True)
        )
    groups = assign_problem_groups(records)
    split_summary = validate_group_splits(records, groups)
    args.output_dir.mkdir(parents=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_sources: dict[str, Any] = {}
    for source_name, indices in indices_by_source.items():
        output_path = args.output_dir / f"{source_name}.jsonl"
        source_records = [records[index] for index in indices]
        _write_jsonl(output_path, source_records)
        output_sources[source_name] = {
            "derived_from": parent_manifest["sources"][source_name]["external_path"],
            "external_path": str(output_path.resolve()),
            "byte_count": output_path.stat().st_size,
            "sha256": _sha256_file(output_path),
            "record_count": len(source_records),
            "ordered_samples": [
                {
                    "sample_id": record["sample_id"],
                    "split": record["split"],
                    "category": record["category"],
                    "content_sha256": record["content_sha256"],
                    "problem_group_id": record["problem_group_id"],
                }
                for record in source_records
            ],
        }
    manifest = {
        "schema_version": 4,
        "derived_from": {
            "manifest": str(args.parent_manifest.resolve()),
            "manifest_sha256": _sha256_file(args.parent_manifest),
        },
        "problem_grouping": {
            "code": "identifier-normalized prompt plus exact upstream problem ID",
            "math": "number/variable-normalized prompt plus exact upstream problem ID",
            **split_summary,
        },
        "family_near_duplicate_exclusion": {
            "matched_record_count": 0,
            "humaneval_prompt_signature_count": len(protected["humaneval_prompt"]),
            "humaneval_prompt_signature_sha256": _sha256_string_set(
                protected["humaneval_prompt"]
            ),
            "humaneval_solution_signature_count": len(
                protected["humaneval_canonical_solution"]
            ),
            "humaneval_solution_signature_sha256": _sha256_string_set(
                protected["humaneval_canonical_solution"]
            ),
            "gsm8k_question_signature_count": len(protected["gsm8k_question"]),
            "gsm8k_question_signature_sha256": _sha256_string_set(
                protected["gsm8k_question"]
            ),
        },
        "sources": output_sources,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "output_dir": str(args.output_dir),
                "problem_grouping": manifest["problem_grouping"],
                "source_hashes": {
                    name: source["sha256"] for (name, source) in output_sources.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
