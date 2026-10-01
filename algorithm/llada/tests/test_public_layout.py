"""Check package structure, payload ignore rules and evaluation CLI imports."""

import os
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[2]


def files():
    return [
        (path, path.relative_to(ROOT)) for path in ROOT.rglob("*") if path.is_file()
    ]


def test_ignore_rules_include_model_source_and_exclude_payload(tmp_path):
    ignore = next(source for source, target in files() if target == Path(".gitignore"))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(ignore.read_text())
    for path, expected in (
        ("llada/model/modeling_llada.py", 1),
        ("model/weights.bin", 0),
        ("artifact_w4/payload/weights.int4", 0),
    ):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", path],
            cwd=tmp_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == expected, (path, result.stderr)


@pytest.mark.parametrize(
    "module, option",
    [("evaluation.model", "--model"), ("evaluation.scoring", "samples")],
)
def test_evaluation_cli_loads_in_a_fresh_interpreter(module, option):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(ROOT / "llada"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-m", module, "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert option in result.stdout
