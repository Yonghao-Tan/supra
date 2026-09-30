import unittest
from calibration.prepare import (
    _sha256_string_set,
    contamination_reason,
    exact_ngrams,
    normalize_for_overlap,
    select_training_records,
    PERSONAHUB_SOURCE,
)


class PrepareCalibrationDatasetTest(unittest.TestCase):
    def test_record_id_without_upstream_identifier_is_stable(self):
        from calibration.prepare import _stable_record_id

        missing = _stable_record_id("train", 2, {}, "prompt")
        empty = _stable_record_id("train", 2, {"id": "", "url": ""}, "prompt")
        self.assertEqual(missing, empty)
        self.assertTrue(missing.startswith("train:2::"))

    @staticmethod
    def _messages(prompt, answer):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    def test_normalization_and_overlap(self):
        protected = "Write a function: add_two(a, b)."
        normalized = normalize_for_overlap(protected)
        self.assertEqual(normalized, "write a function : add_two ( a , b ) .")
        reason = contamination_reason(
            "Prefix WRITE a function: add_two(a, b). suffix",
            protected_exact={normalized},
            protected_ngrams=exact_ngrams(protected, 5),
            ngram_width=5,
        )
        self.assertEqual(reason, "normalized_full_prompt_or_question")

    def test_protected_set_hashes_are_order_independent(self):
        strings = {"alpha beta", "gamma"}
        self.assertEqual(
            _sha256_string_set(strings),
            _sha256_string_set(set(reversed(sorted(strings)))),
        )

    def test_selected_personahub_rows_preserve_source_order_and_split(self):
        rows = [
            {"source": "unselected", "messages": self._messages("unused", "answer")},
            {"id": "a", "source": PERSONAHUB_SOURCE, "messages": self._messages("first", "def one(): pass")},
            {"id": "b", "source": PERSONAHUB_SOURCE, "messages": self._messages("second", "def two(): pass")},
        ]
        selected = select_training_records(rows, dataset="tulu3",
            indices={"train": [2], "validation": [1]}, protected_exact=set(),
            protected_ngrams=set(), ngram_width=13)
        self.assertEqual([r["split"] for r in selected], ["validation", "train"])
        self.assertTrue(selected[0]["sample_id"].startswith("tulu3:1:a:"))
        self.assertEqual(selected[1]["messages"], rows[2]["messages"])

    def test_selected_gsm_records_keep_question_answer_and_reject_overlap(self):
        rows = [{"question": "How many pears?", "answer": "Two. #### 2"}]
        kwargs = dict(dataset="gsm8k_train", indices={"train": [0], "validation": []},
            protected_exact=set(), protected_ngrams=set(), ngram_width=13)
        selected = select_training_records(rows, **kwargs)
        self.assertEqual(selected[0]["messages"], self._messages("How many pears?", "Two. #### 2"))
        kwargs["protected_exact"] = {normalize_for_overlap(rows[0]["question"])}
        with self.assertRaisesRegex(ValueError, "overlaps benchmark"):
            select_training_records(rows, **kwargs)

    def test_sample_index_errors_do_not_replace_requests(self):
        for indices, message in [({"train": [0], "validation": [0]}, "duplicate"),
                                 ({"train": [4], "validation": []}, "selected rows")]:
            with self.assertRaisesRegex(ValueError, message):
                select_training_records([], dataset="gsm8k_train", indices=indices,
                    protected_exact=set(), protected_ngrams=set(), ngram_width=13)
