import unittest
from calibration.split import (
    validate_group_splits,
    assign_problem_groups,
    normalize_code_problem,
    normalize_math_problem,
)


class CalibrationSplitTest(unittest.TestCase):
    @staticmethod
    def _record(sample_id, split, category, prompt, upstream_id=None):
        record = {
            "sample_id": sample_id,
            "split": split,
            "category": category,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "answer"},
            ],
        }
        if upstream_id is not None:
            record["source"] = "fixture"
            record["upstream_id"] = upstream_id
        return record

    def test_code_and_math_normalization_detects_obvious_variants(self):
        self.assertEqual(
            normalize_code_problem("def add_values(left, right): return left + right"),
            normalize_code_problem("def sum_items(x, y): return left + right"),
        )
        self.assertEqual(
            normalize_math_problem("If x has 12 apples, how many remain?"),
            normalize_math_problem("If y has 99 apples, how many remain?"),
        )

    def test_configured_split_is_preserved_and_group_overlap_rejected(self):
        records = [
            self._record("m0", "train", "gsm8k", "Alice has 12 apples."),
            self._record("m1", "train", "gsm8k", "Alice has 99 apples."),
            self._record("m2", "validation", "gsm8k", "Bob has 3 pears."),
        ]
        groups = assign_problem_groups(records)
        summary = validate_group_splits(records, groups)
        self.assertEqual(summary["counts"]["gsm8k"], {"train": 2, "validation": 1})
        self.assertEqual([r["split"] for r in records], ["train", "train", "validation"])
        records[1]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "crosses train/check"):
            validate_group_splits(records, groups)

    def test_exact_upstream_id_groups_prompt_variants(self):
        records = [
            self._record("c0", "train", "code", "Write function alpha.", "problem-1"),
            self._record(
                "c1", "validation", "code", "Implement function beta.", "problem-1"
            ),
        ]
        groups = assign_problem_groups(records)
        self.assertEqual(len(groups), 1)


if __name__ == "__main__":
    unittest.main()
