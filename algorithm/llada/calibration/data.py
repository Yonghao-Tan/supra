"""Train-source calibration records and shared tokenization utilities."""

from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Sequence

SILU_SAMPLING = "shard-layer-call8191-coprime/v1"
CAPTURE_NUMERIC_FIELDS = (
    "anchor",
    "actual_linear_a8",
    "numeric_capture_graph",
    "artifact_manifest_sha256",
    "w8_head_manifest_sha256",
    "head_weight_bits",
    "a4_clip_ratio_bf16",
    "a4_output_clip_ratio_bf16",
    "silu_table",
)


def capture_sample_identity(metadata):
    identity = tuple((metadata.get(key) for key in ("task", "sample_id")))
    if any((not isinstance(value, str) or not value for value in identity)):
        raise ValueError("missing stable capture task/sample_id")
    return identity


def capture_numeric_source(metadata):
    missing = [key for key in CAPTURE_NUMERIC_FIELDS if key not in metadata]
    if missing or any(
        (
            not metadata.get(key)
            for key in ("artifact_manifest_sha256", "w8_head_manifest_sha256")
        )
    ):
        raise ValueError(f"missing capture numerical provenance: {missing}")
    return {key: metadata[key] for key in CAPTURE_NUMERIC_FIELDS}


@dataclass(frozen=True)
class CalibrationRecord:
    sample_id: str
    task: str
    prompt: str
    answer: str
    source_index: int
    category: str = ""
    prompt_messages: tuple[tuple[str, str], ...] = ()
    question_char_start: int | None = None


def gsm8k_position_fewshot_prompt(
    train_rows: Sequence[dict[str, Any]],
    target_row: dict[str, Any],
    *,
    position: int,
    seed: int = 20260806,
    fewshot_count: int = 4,
    exclude_target: bool = False,
) -> str:
    """Reproduce the GSM sampler sequence without reading benchmark test rows."""
    if position < 0:
        raise ValueError("GSM sampler position must be nonnegative")
    if not 0 <= fewshot_count <= 4:
        raise ValueError("GSM fewshot count must be in [0,4]")
    if len(train_rows) < fewshot_count:
        raise ValueError("GSM fewshot pool is smaller than the requested sample")
    replay = random.Random(seed)
    for _ in range(position):
        replay.sample(train_rows, fewshot_count)
    examples = replay.sample(train_rows, fewshot_count)
    if exclude_target and any(row["question"] == target_row["question"] for row in examples):
        eligible = [row for row in train_rows if row["question"] != target_row["question"]]
        if len(eligible) < fewshot_count:
            raise ValueError("GSM fewshot pool cannot exclude the target")
        examples = replay.sample(eligible, fewshot_count)
    prefix = "".join(
        (
            f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
            for row in examples
        )
    )
    return f"{prefix}Question: {target_row['question']}\nAnswer:"


_PYTHON_FENCE = re.compile(
    "```(?:python|py)?\\s*\\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL
)


def _humaneval_style_continuation(
    record: CalibrationRecord, *, prompt_from_task: bool = False
) -> CalibrationRecord | None:
    """Convert a decontaminated code answer into a compilable function continuation."""
    candidates = [
        match.group("code") for match in _PYTHON_FENCE.finditer(record.answer)
    ]
    candidates.append(record.answer)
    source = ""
    tree = None
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
        functions = [
            node
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(functions) == 1:
            (source, tree) = (candidate, parsed)
            break
    if tree is None:
        return None
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    )
    if not function.body or function.end_lineno is None:
        return None
    lines = source.splitlines()
    body_index = 0
    has_docstring = (
        isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    )
    if has_docstring:
        body_index = 1
        if body_index >= len(function.body):
            return None
        first_body_line = function.body[body_index].lineno
        prefix = "\n".join(lines[: first_body_line - 1]).rstrip() + "\n"
    else:
        first_body_line = function.body[0].lineno
        header = "\n".join(lines[: first_body_line - 1]).rstrip()
        prompt_text = " ".join(
            (
                content.strip()
                for (role, content) in record.prompt_messages
                if role == "user" and content.strip()
            )
        )
        prompt_text = prompt_text.replace('"""', "'''")[:512]
        prefix = f'{header}\n    """{prompt_text}"""\n'
    continuation = (
        "\n".join(lines[first_body_line - 1 : function.end_lineno]).rstrip() + "\n"
    )
    if prompt_from_task:
        prompt_text = "\n\n".join(
            (
                content.strip()
                for (role, content) in record.prompt_messages
                if role == "user" and content.strip()
            )
        )
        if not prompt_text:
            return None
        imports = [
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        header = "async " if isinstance(function, ast.AsyncFunctionDef) else ""
        header += f"def {function.name}({ast.unparse(function.args)})"
        if function.returns is not None:
            header += " -> " + ast.unparse(function.returns)
        description = prompt_text.replace('"""', "'''")
        prefix = (
            ("\n".join(imports) + "\n\n" if imports else "")
            + header
            + ':\n    """'
            + description
            + '"""\n'
        )
        body = "\n".join((ast.unparse(node) for node in function.body[body_index:]))
        continuation = (
            "\n".join(("    " + line if line else "" for line in body.splitlines()))
            + "\n"
        )
    try:
        compile(prefix + continuation, record.sample_id, "exec")
    except (SyntaxError, ValueError, TypeError):
        return None
    return CalibrationRecord(
        sample_id=f"he_style:{record.sample_id}",
        task="humaneval",
        prompt=prefix,
        answer=continuation,
        source_index=record.source_index,
        category="code",
    )


def calibration_source_path(source, data_dir_override=None):
    recorded = Path(source["external_path"])
    (filename, root) = (recorded.name, recorded.parent)
    return (
        root if data_dir_override is None else Path(data_dir_override).resolve()
    ) / filename


def load_code_training_records(
    manifest_path: Path,
    *,
    split: str,
    sample_count: int,
    seed: int = 20260904,
    data_dir_override: Path | None = None,
    category_offset: int = 0,
    allowed_sources: tuple[str, ...] | None = None,
    prompt_from_task: bool = False,
) -> list[CalibrationRecord]:
    """Select compilable, decontaminated function-continuation training records."""
    if sample_count <= 0 or category_offset < 0:
        raise ValueError("HumanEval-style sample count must be positive")
    if split not in {"train", "validation"}:
        raise ValueError("HumanEval-style split must be train or validation")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version != 4:
        raise ValueError(f"calibration manifest schema_version={schema_version!r}; expected 4")
    converted: list[CalibrationRecord] = []
    seen_sample_ids: set[str] = set()
    problem_splits: dict[str, str] = {}
    source_index = 0
    for source_name in sorted(manifest["sources"]):
        source = manifest["sources"][source_name]
        path = calibration_source_path(source, data_dir_override)
        payload = path.read_bytes()
        if len(payload) != int(source["byte_count"]):
            raise ValueError(f"calibration source byte count mismatch: {path}")
        if hashlib.sha256(payload).hexdigest() != source.get("sha256"):
            raise ValueError(f"calibration source SHA-256 mismatch: {path}")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        if len(rows) != int(source["record_count"]):
            raise ValueError(f"calibration source record count mismatch: {path}")
        for row in rows:
            sample_id = str(row.get("sample_id", ""))
            row_split = str(row.get("split", ""))
            category = str(row.get("category", ""))
            problem_group = str(row.get("problem_group_id", ""))
            if not sample_id or sample_id in seen_sample_ids:
                raise ValueError(
                    f"invalid or duplicate publication sample ID: {sample_id}"
                )
            if row_split not in {"train", "validation"} or not problem_group:
                raise ValueError(f"invalid publication split/group: {sample_id}")
            previous_split = problem_splits.setdefault(problem_group, row_split)
            if previous_split != row_split:
                raise ValueError(
                    f"publication problem group crosses splits: {problem_group}"
                )
            seen_sample_ids.add(sample_id)
            if row_split != split or category != "code":
                source_index += 1
                continue
            if allowed_sources is not None and row.get("source") not in allowed_sources:
                source_index += 1
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                source_index += 1
                continue
            if str(messages[-1].get("role", "")) != "assistant":
                source_index += 1
                continue
            prompt_messages = tuple(
                (
                    (str(message.get("role", "")), str(message.get("content", "")))
                    for message in messages[:-1]
                )
            )
            if any((not role or not content for (role, content) in prompt_messages)):
                source_index += 1
                continue
            record = CalibrationRecord(
                sample_id=sample_id,
                task="publication_chat",
                prompt="",
                answer=str(messages[-1].get("content", "")),
                source_index=source_index,
                category=category,
                prompt_messages=prompt_messages,
            )
            candidate = _humaneval_style_continuation(
                record, prompt_from_task=prompt_from_task
            )
            if candidate is not None:
                converted.append(candidate)
            source_index += 1
    converted.sort(
        key=lambda record: hashlib.sha256(
            f"{seed}:{split}:he-style:{record.sample_id}".encode("utf-8")
        ).digest()
    )
    if len(converted) < category_offset + sample_count:
        raise ValueError(
            f"only {len(converted)} compilable HumanEval-style records, need {sample_count}"
        )
    return converted[category_offset : category_offset + sample_count]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gsm8k_chat_fields(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(
            f"GSM record must have one user and one assistant: {row.get('sample_id')}"
        )
    if (
        str(messages[0].get("role", "")) != "user"
        or str(messages[1].get("role", "")) != "assistant"
    ):
        raise ValueError(f"GSM record roles are invalid: {row.get('sample_id')}")
    question = str(messages[0].get("content", ""))
    answer = str(messages[1].get("content", ""))
    if not question or not answer or "####" not in answer:
        raise ValueError(
            f"GSM record has an empty question or final answer: {row.get('sample_id')}"
        )
    return (question, answer)


def _gsm8k_upstream_index(sample_id: str) -> int:
    prefix = "gsm8k_train:"
    if not sample_id.startswith(prefix) or "::" not in sample_id:
        raise ValueError(f"unsupported GSM train sample ID: {sample_id}")
    try:
        return int(sample_id[len(prefix) :].split("::", 1)[0])
    except ValueError as error:
        raise ValueError(f"invalid GSM train sample index: {sample_id}") from error


def load_gsm8k_training_records(
    manifest_path: Path,
    *,
    split: str,
    sample_count: int,
    seed: int = 20260829,
    fewshot_seed: int = 20260806,
    fewshot_count: int = 4,
    data_dir_override: Path | None = None,
    category_offset: int = 0,
    fewshot_pool: Sequence[dict[str, Any]] | None = None,
) -> list[CalibrationRecord]:
    """Load GSM train targets with train-only position-dependent 4-shot prompts."""
    if split not in {"train", "validation"}:
        raise ValueError("GSM QAT split must be train or validation")
    if sample_count <= 0 or category_offset < 0:
        raise ValueError("GSM sample count must be positive and offset nonnegative")
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version != 4:
        raise ValueError(f"calibration manifest schema_version={schema_version!r}; expected 4")
    source = manifest.get("sources", {}).get("gsm8k_train")
    if not isinstance(source, dict):
        raise ValueError("publication manifest has no gsm8k_train source")
    path = calibration_source_path(source, data_dir_override)
    if path.stat().st_size != int(source["byte_count"]):
        raise ValueError(f"GSM source byte count mismatch: {path}")
    if _sha256_file(path) != source["sha256"]:
        raise ValueError(f"GSM source SHA-256 mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != int(source["record_count"]):
        raise ValueError(f"GSM source record count mismatch: {path}")
    seen_ids: set[str] = set()
    seen_groups: dict[str, str] = {}
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        row_split = str(row.get("split", ""))
        group_id = str(row.get("problem_group_id", ""))
        if (
            not sample_id
            or sample_id in seen_ids
            or row.get("category") != "gsm8k"
            or (row_split not in by_split)
            or (not group_id)
        ):
            raise ValueError(f"invalid GSM train record: {sample_id}")
        previous = seen_groups.setdefault(group_id, row_split)
        if previous != row_split:
            raise ValueError(f"GSM problem group crosses splits: {group_id}")
        _gsm8k_chat_fields(row)
        _gsm8k_upstream_index(sample_id)
        seen_ids.add(sample_id)
        by_split[row_split].append(row)
    candidates = sorted(
        by_split[split],
        key=lambda row: hashlib.sha256(
            f"{seed}:{split}:gsm8k:{row['sample_id']}".encode("utf-8")
        ).digest(),
    )
    end = category_offset + sample_count
    if len(candidates) < end:
        raise ValueError(
            f"GSM {split} has {len(candidates)} records, needs offset {category_offset} plus {sample_count}"
        )
    selected = candidates[category_offset:end]
    train_pool = [
        {"sample_id": str(row["sample_id"]), "question": question, "answer": answer}
        for row in by_split["train"]
        for (question, answer) in (_gsm8k_chat_fields(row),)
    ]
    if fewshot_pool is not None:
        train_pool = list(fewshot_pool)
        identities = {(example["question"], example["answer"]) for example in train_pool}
        if any(_gsm8k_chat_fields(row) not in identities for row in selected):
            raise ValueError("GSM target content is absent from the supplied train fewshot pool")
    records: list[CalibrationRecord] = []
    for position, row in enumerate(selected):
        sample_id = str(row["sample_id"])
        (question, answer) = _gsm8k_chat_fields(row)
        eligible = train_pool if fewshot_pool is not None else [
            example for example in train_pool if example["sample_id"] != sample_id
        ]
        prompt = gsm8k_position_fewshot_prompt(
            eligible,
            {"question": question, "answer": answer},
            position=category_offset + position,
            seed=fewshot_seed,
            fewshot_count=fewshot_count,
            exclude_target=fewshot_pool is not None,
        )
        records.append(
            CalibrationRecord(
                sample_id=sample_id,
                task="gsm8k",
                prompt=prompt,
                answer=answer,
                source_index=_gsm8k_upstream_index(sample_id),
                category="gsm8k",
                question_char_start=len(prompt) - len(f"Question: {question}\nAnswer:"),
            )
        )
    return records
