"""Prepared and relocated train-source indices preserve calibration requests."""

import hashlib
import json
import pytest
from calibration.data import load_gsm8k_training_records, load_code_training_records


@pytest.mark.parametrize("task", ["gsm8k", "humaneval"])
def test_original_and_relocated_indices_select_identical_requests(tmp_path, task):
    rows = []
    for index in range(9):
        split = "train" if index < 6 else "validation"
        gsm = task == "gsm8k"
        rows.append(
            dict(
                sample_id=f"gsm8k_train:{index}::sample{index}"
                if gsm
                else f"code:{index}",
                category="gsm8k" if gsm else "code",
                split=split,
                problem_group_id=f"problem:{index}",
                messages=[
                    dict(role="user", content=f"{split} question {index}"),
                    dict(
                        role="assistant",
                        content=f"reasoning\n#### {index}"
                        if gsm
                        else f"```python\ndef function_{index}(x):\n    return x + {index}\n```",
                    ),
                ],
            )
        )
    payload = "".join((json.dumps(row) + "\n" for row in rows)).encode()
    (tmp_path / "records.jsonl").write_bytes(payload)
    source = dict(
        byte_count=len(payload),
        record_count=len(rows),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    loader = (
        load_gsm8k_training_records if task == "gsm8k" else load_code_training_records
    )
    collected = []
    for relocated in (False, True):
        entry = dict(source, external_path="/unavailable/records.jsonl"
                     if relocated else str(tmp_path / "records.jsonl"))
        manifest = tmp_path / "index.json"
        manifest.write_text(
            json.dumps(
                dict(
                    schema_version=4,
                    sources={"gsm8k_train" if task == "gsm8k" else "code": entry},
                )
            )
        )
        records = loader(
            manifest,
            split="validation",
            sample_count=2,
            seed=7,
            data_dir_override=tmp_path if relocated else None,
        )
        collected.append(records)
        if gsm:
            pool = [dict(question=f"official question {index}", answer=str(index)) for index in range(12)]
            pool += [dict(question=row['messages'][0]['content'], answer=row['messages'][1]['content']) for row in rows]
            diagnostic = loader(manifest, split='validation', sample_count=2, seed=7,
                                data_dir_override=tmp_path if relocated else None, fewshot_pool=pool)
            assert [(r.sample_id, r.source_index, r.answer) for r in diagnostic] == [
                (r.sample_id, r.source_index, r.answer) for r in records]
            for record in diagnostic:
                target = record.prompt[record.question_char_start:]
                assert target.startswith('Question: validation question')
                assert target not in record.prompt[:record.question_char_start]
            with pytest.raises(ValueError, match='target content'):
                loader(manifest, split='validation', sample_count=2, seed=7,
                       data_dir_override=tmp_path if relocated else None, fewshot_pool=pool[:12])
        with pytest.raises(ValueError, match="split must be train or validation"):
            loader(manifest, split="test", sample_count=1, data_dir_override=tmp_path if relocated else None)
    assert collected[0] == collected[1]
    assert len(collected[0]) == 2
    if task == "gsm8k":
        for record in collected[0]:
            assert record.prompt.count("Question:") == 5
            assert "validation question" not in record.prompt.rsplit("Question:", 1)[0]

    (tmp_path / "records.jsonl").write_bytes(
        payload.replace(b"question", b"tampered", 1)
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        loader(
            manifest,
            split="validation",
            sample_count=2,
            seed=7,
            data_dir_override=tmp_path if relocated else None,
        )
    (tmp_path / "records.jsonl").write_bytes(payload)
    damaged = json.loads(manifest.read_text())
    next(iter(damaged["sources"].values()))["sha256"] = "0" * 64
    manifest.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        loader(
            manifest,
            split="validation",
            sample_count=2,
            seed=7,
            data_dir_override=tmp_path if relocated else None,
        )


def test_dataset_builder_rejects_existing_output_before_fetching_or_writing(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from calibration import prepare as builder

    (package, data) = (tmp_path / "public", tmp_path / "data")
    package.mkdir()
    data.mkdir()
    sentinel = data / "records.jsonl"
    sentinel.write_text("existing calibration records\n")
    monkeypatch.setattr(
        builder, "__file__", str(package / "llada/calibration/prepare.py")
    )
    monkeypatch.setenv("SUPRA_ALGORITHM_ROOT", str(tmp_path))
    monkeypatch.setattr(
        builder,
        "parse_args",
        lambda: SimpleNamespace(output_dir=data, manifest=package / "data.json"),
    )
    with pytest.raises(FileExistsError, match="must both be new"):
        builder.main()
    assert sentinel.read_text() == "existing calibration records\n"
    assert not (package / "data.json").exists()


def test_fewshot_target_exclusion_preserves_noncolliding_sampler():
    import random
    from calibration.data import gsm8k_position_fewshot_prompt

    pool = [dict(question=f'question {i}', answer=f'answer {i}') for i in range(12)]
    rng = random.Random(20260806)
    for position in range(7):
        examples = rng.sample(pool, 4)
        target = next(row for row in pool if row not in examples)
        expected = ''.join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in examples)
        expected += f"Question: {target['question']}\nAnswer:"
        for exclude in (False, True):
            assert gsm8k_position_fewshot_prompt(pool, target, position=position,
                exclude_target=exclude) == expected
        target = examples[0]
        result = gsm8k_position_fewshot_prompt(pool, target, position=position, exclude_target=True)
        assert f"Answer: {target['answer']}\n\n" not in result
        assert result.count('Question:') == 5
        assert result == gsm8k_position_fewshot_prompt(pool, target, position=position, exclude_target=True)
