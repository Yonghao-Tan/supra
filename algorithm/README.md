# SUPRA Algorithm and Quantization

SUPRA targets LLaDA-8B-Instruct with attention-guided token skipping (Feature1/ATSE),
draft/verify decoding with mixed-precision execution (Feature2/PSME), and
utilization-aware token prefetching (Feature3/UAPS). Transformer weights use G-1 W4, activations use per-row A4/A8,
Attention uses Q8/K8/P8/V8, and the LM head uses W8.

`run.py` runs calibration, artifact export, benchmark generation and scoring.
Default task settings are in `llada/configs/gsm8k.json` and
`llada/configs/humaneval.json`. Model weights and datasets are obtained separately.

## Source Layout

```text
run.py                 Calibration and evaluation pipeline
package_model.py       Portable model artifact preparation
llada/
  model/               LLaDA Transformer
  generation/          Generation, refresh, admission, lookahead and records
  quantization/        Weights, rotations, artifact readers and installation
  numerics/            BF16 arithmetic, quantization and CUDA kernels
  calibration/         Training data, initial LWC, Hessians and joint GPTQ
  evaluation/          Benchmark execution, tasks and scoring
  capture/             Layer observation and generation replay
  configs/             Default task configurations
  tests/               CPU regression tests
```

The generation entry is `generation.engine.generate`; benchmark execution uses
`evaluation.model.QuantizedLLaDALM`, registered as `quantized_llada`.

## Install

Recommended environment: Python 3.9, PyTorch 2.4.1 with CUDA 11.8.
Run the commands in this guide from `algorithm/`.

```bash
python -m pip install pip==24.2
python -m pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip check
```

The requirements pin the runtime and numerical dependencies.

## Model And Storage

Keep model artifacts outside the source directory:

```bash
export DATA=/absolute/path/algorithm-artifacts
export VERSION=your-source-commit-or-release-tag
```

`DATA` must be outside this source directory. Checkpoints, calibration data,
captures, logs and evaluation outputs are written there. Each run uses a new
output directory. Obtain model and data files under their respective licenses.

Download the [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
BF16 model for calibration:

```bash
export MODEL="$DATA/LLaDA-8B-Instruct"
huggingface-cli download GSAI-ML/LLaDA-8B-Instruct \
  --revision 08b83a6feb34df1a6011b80c3c00c7563e963b07 --local-dir "$MODEL"
```

## Evaluate

A model release directory contains `artifact_w4/`, `artifact_w8_head/`,
`silu_table_monotone.json` and `model_metadata/`. Together these form one
deployment model shared by both tasks. The loader constructs the model from
configuration and installs all artifact weights; evaluation does not need
the original BF16 weight shards. The following native six-process example runs both benchmarks and scoring:

```bash
python run.py --mode evaluate \
  --model-path "$DATA/model/model_metadata" \
  --model-artifacts "$DATA/model" --artifact-root "$DATA" \
  --run-dir "$DATA/evaluation" --gpus 0,1,2,3,4,5 \
  --source-version "$VERSION"
```

Use `--tasks gsm8k` or `--tasks humaneval` for one benchmark. Select an
execution tier with `--tier`:

| Tier | Execution |
| --- | --- |
| `baseline` | Full-sequence W4A8 backbone on every forward, selected output-head rows and irreversible fixed-k decoding. |
| `feature1` | ATSE attention-guided token skipping with fixed-k decoding. |
| `feature12` | ATSE token skipping and PSME draft/verify decoding with mixed-precision execution. |
| `feature123` | Feature1+2 with UAPS utilization-aware token prefetching; the default. |

`--feature3 off` is an alias for the matched Feature1+2 configuration when the full
tier is selected by default. `--config path/to/task.json` selects a task
configuration; its task must match the single benchmark named by `--tasks`.
Repeat `--model-arg NAME=JSON_VALUE` to override model settings. The tier fixes
its decoding mode and activation policy. For example, append
`--tasks gsm8k --tier baseline --model-arg decode_k=2` to the evaluation command.
For a short functional run, additionally use `--model-arg gen_length=32
--model-arg steps=32 --limit-per-process 1`. Generation length must be a positive
multiple of 32. The benchmark default remains 256 generated tokens. One process uses one GPU;
multiple GPUs run independent requests, not distributed model training.

The default native driver starts one Accelerate process per selected GPU and uses the
`gsm8k_native` or `humaneval_native` task. Choose available devices with `--gpus`.
Add `--limit-per-process 1` for one request per process; omit it for full evaluation.
The default GSM protocol uses lm-eval 0.4.8, four training examples and seed tuple
`0,1234,1234,1234`. Process count affects rank-local few-shot sampling, so keep it
consistent when comparing configurations. HumanEval uses zero-shot chat prompts,
complete code outputs and the same seed tuple. Its summary reports raw and
sanitized pass@1; sanitization extracts the generated function and its dependencies.

Limited results are marked `evaluation_scope: partial`.

Outputs are `results/gsm8k/summary.json` and
`results/humaneval/summary.json` under the run directory. Each summary includes
quality and NFE. Per-job logs are in `logs/`; individual evaluation outputs
retain generated tokens and traces. `run.json` records input arguments,
source version, runtime versions, GPU names and memory capacity. A successful
run has `exit_code` equal to zero. Failed or interrupted work is not collected
as a complete result.

`--print-commands` prints the stage commands without loading a model or creating
outputs. For generation and collection commands, use
`PYTHONPATH=llada python -m evaluation.generate --help` and
`PYTHONPATH=llada python -m evaluation.collect --help`.

## Calibrate And Evaluate

Using the BF16 model downloaded above:

```bash
python run.py --mode all --model-path "$MODEL" --artifact-root "$DATA" \
  --run-dir "$DATA/reproduction" --gpus 0,1,2,3,4,5 \
  --source-version "$VERSION"
```

This runs data preparation, initial quantization, train-input capture, Hessian
collection, joint GPTQ, SiLU fitting, artifact export and both benchmarks.
Use `--mode calibrate` to stop after artifact export. See
[CALIBRATION.md](CALIBRATION.md) for data sources, methods and stage outputs.
Evaluation reads the task JSON configurations. Calibration capture uses the
stage-specific profile described in CALIBRATION.md and defined in
`llada/calibration/config.py`.

## Package a Model

Package the calibration outputs into the model directory used by evaluation
and capture:

```bash
python package_model.py --weight-artifact "$DATA/reproduction/model/artifact_w4" \
  --head-artifact "$DATA/reproduction/model/artifact_w8_head" \
  --silu-table "$DATA/reproduction/model/silu_table_monotone.json" \
  --model-path "$MODEL" --artifact-root "$DATA" --output "$DATA/model"
```

This copies codes, scales and metadata, verifies the payloads against their
manifests, preserves the source model's card and license notices, and creates
a model directory for this package's loader.

## GPU Configuration

| Work | Tested hardware and resource requirements |
|---|---|
| Initial quantization | One GPU per process; A100 80GB environment |
| Train capture | A100 80GB or RTX 3090 24GB; one model per worker |
| Hessian collection and GPTQ | A100 80GB; requests or Linear modules divided across workers |
| Benchmark generation | A100 80GB or RTX 3090 24GB; one model per worker |
| SiLU fitting and scoring | CPU |

Use 80GB GPUs for Hessian collection: the full FP32 Hessian bank alone is
about 30 GiB per worker, before model and runtime memory.

## Hardware Workload Export

Completed evaluation traces can be converted to compact logical work without
loading the model or using a GPU. The output records every forward, its actual
sequence length, per-layer-range A4/A8 query counts, and the L31 output and LM
head position sets. Row counts are per layer in the indicated range.
Each layer segment records executed `kv_write_tokens`
separately from `query_tokens`. For global L0 boundary queries,
`layer0_keep_global_cache` distinguishes retaining all updates from restoring
unselected cache rows. The L31 segment also records the
actual A4/A8 split of its output subset for O projection and FFN costing.
The JSON uses the `supra-algorithm-workload/v2` format. Its phases are
`full_sequence`, `block_initialization`, `boundary`, `regular` and `repair`.

Using the evaluation run directory from the example above:

```bash
SUPRA_ALGORITHM_ROOT="$DATA" PYTHONPATH=llada python -m evaluation.workload \
  --root "$DATA/evaluation/evaluation/gsm8k" \
  --task gsm8k \
  --output "$DATA/workloads/gsm8k.json"
```

Use `humaneval` for the task, evaluation directory and output name to export
HumanEval workloads. The exporter reads completed `trace.jsonl` files and writes
per-forward logical workload records.

## Tests And Integration

```bash
python -m pip install 'pytest>=7,<9'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=llada \
  python -B -m pytest -q -p no:cacheprovider llada/tests
```

CPU tests use synthetic tensors and do not require model weights.

`llada/capture/layers.py` provides logical layer capture and replay for
hardware integration. It exports codes, scales, row precision, cache inputs and
numerical checkpoints for the hardware preparation and RTL tools.

## Attribution

This implementation builds on [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
and [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA). See
[UPSTREAM.md](UPSTREAM.md) and `LICENSE` for source attribution and revisions.
