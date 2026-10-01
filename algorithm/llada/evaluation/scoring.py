# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import argparse
import json
import os
from pathlib import Path
import uuid

import evaluate as hf_evaluate

from evaluation.sanitize import sanitize
from quantization.rotation import require_algo_output


CODE_EVAL_REVISION = "262b7e74cf29a715d74f8b02ba1d6ef74e432333"


def load_code_eval():
    return hf_evaluate.load(
        "code_eval",
        revision=CODE_EVAL_REVISION,
        experiment_id=f"supra-{os.getpid()}-{uuid.uuid4().hex}",
    )


def pass_at_k(references: list[str], predictions: list[list[str]], k=(1,)):
    return load_code_eval().compute(references=references, predictions=predictions, k=k)[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [doc["prompt"] + response for response in responses]
        for responses, doc in zip(resps, docs)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Score sanitized HumanEval completions."
    )
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()
    file_path = require_algo_output(args.samples)
    with file_path.open() as handle:
        data = [json.loads(line) for line in handle]
    if not data:
        raise ValueError("sample file is empty")
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    metric = load_code_eval()
    results = []
    for sample in data:
        prediction = [
            sanitize(
                sample["doc"]["prompt"]
                + "\n"
                + sample["resps"][0][0].split("```python\n", 1)[-1].split("```")[0],
                sample["doc"]["entry_point"],
            )
        ]
        score = metric.compute(
            references=[sample["target"]], predictions=[prediction], k=[1]
        )[0]["pass@1"]
        results.append(
            dict(
                task_id=sample["doc"]["task_id"], completion=prediction, pass_at_1=score
            )
        )
    print(sum(row["pass_at_1"] for row in results) / len(results))
    with Path(str(file_path) + ".cleaned").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
