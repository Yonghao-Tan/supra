"""Check code extraction and entry-point selection for HumanEval scoring."""

import pytest

from evaluation.sanitize import sanitize


@pytest.mark.parametrize("code", [
    "```python\ndef answer():\n    return 42\n```",
    "VALUE = 42\ndef helper():\n    return VALUE\ndef answer():\n    return helper()\n",
    "import math\ndef answer():\n    return math.factorial(4) + 18\n",
])
def test_sanitize_extracts_executable_answer(code):
    cleaned = sanitize(code + "\ndef unused():\n    return -1\n", "answer")
    namespace = {}
    exec(cleaned, namespace)
    assert namespace["answer"]() == 42
    assert "unused" not in namespace


def test_sanitize_without_entrypoint_keeps_functions():
    cleaned = sanitize("def first():\n    return 1\ndef second():\n    return 2\n")
    namespace = {}
    exec(cleaned, namespace)
    assert namespace["first"]() == 1
    assert namespace["second"]() == 2


def test_sanitize_handles_empty_code():
    assert sanitize("", "answer") == ""
