"""Prepare calibration training and validation records outside the source tree.

Benchmark text supplies overlap-exclusion keys; selected records come from
training sources.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
from datasets import load_dataset

TULU_DATASET = "allenai/tulu-3-sft-mixture"
TULU_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
HUMANEVAL_DATASET = "openai/openai_humaneval"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
PERSONAHUB_SOURCE = "ai2-adapt-dev/personahub_code_v2_34999"
DEFAULT_SEED = 20260825
DEFAULT_NGRAM = 13


def normalize_for_overlap(text: str) -> str:
    """Normalize conservatively while retaining code/math punctuation tokens."""
    return " ".join(re.findall("[A-Za-z0-9_]+|[^\\w\\s]", text.casefold()))


def exact_ngrams(text: str, width: int) -> set[tuple[str, ...]]:
    tokens = normalize_for_overlap(text).split()
    if width <= 0:
        raise ValueError("ngram width must be positive")
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def record_text(record: dict[str, Any]) -> str:
    if "messages" in record:
        return "\n".join(
            (str(message.get("content", "")) for message in record["messages"])
        )
    return str(record.get("text", ""))


def has_nonempty_assistant_answer(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if str(message.get("role", "")).casefold() == "assistant":
            return bool(str(message.get("content", "")).strip())
    return False


def contamination_reason(
    text: str,
    *,
    protected_exact: set[str],
    protected_ngrams: set[tuple[str, ...]],
    ngram_width: int,
) -> str | None:
    normalized = normalize_for_overlap(text)
    if any((protected and protected in normalized for protected in protected_exact)):
        return "normalized_full_prompt_or_question"
    if exact_ngrams(text, ngram_width) & protected_ngrams:
        return f"exact_{ngram_width}_gram"
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_string_set(values: set[str]) -> str:
    payload = "".join(
        (
            json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n"
            for value in sorted(values)
        )
    )
    return _sha256_text(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_record_id(
    dataset: str, index: int, record: dict[str, Any], text: str
) -> str:
    upstream = record.get("id") or record.get("url") or ""
    return f"{dataset}:{index}:{upstream}:{_sha256_text(text)[:16]}"


def build_protected_keys(
    ngram_width: int,
) -> tuple[set[str], set[tuple[str, ...]], dict[str, int]]:
    exact: set[str] = set()
    ngrams: set[tuple[str, ...]] = set()
    counts: dict[str, int] = {}
    humaneval_rows = load_dataset(
        HUMANEVAL_DATASET,
        "openai_humaneval",
        split="test",
        revision=HUMANEVAL_REVISION,
        streaming=True,
    )
    humaneval_count = 0
    for row in humaneval_rows:
        for field in ("prompt", "canonical_solution"):
            value = str(row[field])
            protected = normalize_for_overlap(value)
            if protected:
                exact.add(protected)
                ngrams.update(exact_ngrams(value, ngram_width))
        humaneval_count += 1
    counts["humaneval"] = humaneval_count
    gsm8k_rows = load_dataset(
        GSM8K_DATASET, "main", split="test", revision=GSM8K_REVISION, streaming=True
    )
    gsm8k_count = 0
    for row in gsm8k_rows:
        value = str(row["question"])
        protected = normalize_for_overlap(value)
        if protected:
            exact.add(protected)
            ngrams.update(exact_ngrams(value, ngram_width))
        gsm8k_count += 1
    counts["gsm8k"] = gsm8k_count
    return (exact, ngrams, counts)


def _write_jsonl(
    path: Path, rows: Iterator[dict[str, Any]] | list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            )


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def select_training_records(rows, *, dataset, indices, protected_exact,
                            protected_ngrams, ngram_width, seen_content_hashes=None):
    """Select configured rows from the seeded source order, retaining their split."""
    assignments = {}
    for split in ("train", "validation"):
        for index in indices[split]:
            if type(index) is not int or index < 0 or index in assignments:
                raise ValueError(f"duplicate or invalid {dataset} row index: {index!r}")
            assignments[index] = split
    if not assignments:
        raise ValueError(f"{dataset} sample indices are empty")
    seen = set() if seen_content_hashes is None else seen_content_hashes
    selected = []
    for stream_index, row in enumerate(rows):
        if stream_index not in assignments:
            continue
        if dataset == "tulu3":
            if row.get("source") != PERSONAHUB_SOURCE or not has_nonempty_assistant_answer(row):
                raise ValueError(f"Tulu row {stream_index} requires a Personahub assistant response")
            messages = row["messages"]
            text = record_text(row)
            extra = dict(source=row["source"], upstream_id=row.get("id"))
            category, dataset_name = "code", "tulu3"
            exclusion_text = text
        elif dataset == "gsm8k_train":
            question, answer = str(row["question"]), str(row["answer"])
            messages = [dict(role="user", content=question), dict(role="assistant", content=answer)]
            text = question + "\n" + answer
            extra = dict(upstream_split="train")
            category, dataset_name = "gsm8k", "gsm8k"
            exclusion_text = question
        else:
            raise ValueError(f"unknown calibration source: {dataset}")
        reason = contamination_reason(exclusion_text, protected_exact=protected_exact,
            protected_ngrams=protected_ngrams, ngram_width=ngram_width)
        if reason:
            raise ValueError(f"{dataset} row {stream_index} overlaps benchmark text: {reason}")
        digest = _sha256_text(text)
        if digest in seen:
            raise ValueError(f"{dataset} row {stream_index} duplicates selected content")
        seen.add(digest)
        selected.append(dict(sample_id=_stable_record_id(dataset, stream_index, row, text),
            dataset=dataset_name, split=assignments[stream_index], category=category,
            content_sha256=digest, messages=messages, **extra))
        if len(selected) == len(assignments):
            return selected
    raise ValueError(f"{dataset} contains {len(selected)}/{len(assignments)} selected rows")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ngram-width", type=int, default=DEFAULT_NGRAM)
    parser.add_argument("--sample-indices", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/calibration_samples.json")
    return parser.parse_args()


def main():
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
        raise FileExistsError("dataset output directory and index must both be new")
    protected_exact, protected_ngrams, test_counts = build_protected_keys(args.ngram_width)
    selections = json.loads(args.sample_indices.read_text())
    if set(selections) != {"tulu3", "gsm8k_train"}:
        raise ValueError("sample indices require tulu3 and gsm8k_train sources")
    args.output_dir.mkdir(parents=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    sources, seen = {}, set()
    for name, upstream, revision, offset, subset in (
        ("tulu3", TULU_DATASET, TULU_REVISION, 0, None),
        ("gsm8k_train", GSM8K_DATASET, GSM8K_REVISION, 1, "main"),
    ):
        dataset = load_dataset(upstream, *([subset] if subset else []),
                               split="train", revision=revision, streaming=False)
        records = select_training_records(dataset.shuffle(seed=args.seed + offset), dataset=name,
            indices=selections[name], protected_exact=protected_exact,
            protected_ngrams=protected_ngrams, ngram_width=args.ngram_width, seen_content_hashes=seen)
        output = args.output_dir / f"{name}.jsonl"
        _write_jsonl(output, records)
        sources[name] = dict(upstream=upstream, revision=revision, seed=args.seed + offset,
            external_path=str(output), byte_count=output.stat().st_size, sha256=_sha256_file(output),
            record_count=len(records), ordered_samples=[{key: row[key] for key in
                ("sample_id", "split", "category", "content_sha256")} for row in records])
    manifest = dict(schema_version=3, sources=sources, test_exclusion=dict(
        humaneval_dataset=HUMANEVAL_DATASET, humaneval_revision=HUMANEVAL_REVISION,
        gsm8k_dataset=GSM8K_DATASET, gsm8k_revision=GSM8K_REVISION, counts=test_counts,
        ngram_width=args.ngram_width))
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(manifest=str(args.manifest),
                         sources={name: value["record_count"] for name, value in sources.items()})))


if __name__ == "__main__":
    main()
