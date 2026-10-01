# Calibration

Run from `algorithm/` with the environment described in `README.md`:

```bash
python run.py --mode calibrate --model-path /absolute/path/LLaDA-8B-Instruct \
  --artifact-root /absolute/path/algorithm-artifacts \
  --run-dir /absolute/path/algorithm-artifacts/calibration \
  --gpus 0,1,2,3,4,5 --source-version your-source-commit-or-release-tag
```

Use `--mode all` to evaluate the exported artifact automatically. Add
`--print-commands` to inspect every command before execution. A new run
directory is required; existing stage outputs are not overwritten.

## Stages

| Stage | Entry | Output under run directory |
|---|---|---|
| Source data | `calibration/prepare.py` | `data/parent` |
| Train/check split | `calibration/split.py` | `data/train` |
| Initial W4 parent | `calibration/initialize.py` | `initial/artifact_w4` |
| W8 head | `quantization/head.py export-rtn-w8` | `initial/artifact_w8_head` |
| Actual train/check inputs | `calibration/capture.py --mode capture` | `captures/<task>/<split>` |
| Full Hessians | Same entry, `--mode linear-hessians` | `hessians/<task>` |
| Joint G-1 GPTQ | `calibration/solve.py` | `solve` |
| Shared SiLU table | `calibration/silu.py` | `model/silu_table_monotone.json` |
| Deployment artifact | `quantization/export.py --quantized-linear-root` | `model/artifact_w4`, `model/artifact_w8_head` |

The stage entries above are relative to `llada/`. Data preparation constructs
the calibration inputs. Export preserves the solved GPTQ codes and scales.
Each GPU job logs to `logs/`; stage success is recorded only after
all its jobs complete successfully.

In SUPRA, `parent.json` and `train.json` are written to `data_indices/` under
the run directory. `--data-index-dir` selects another index directory. Full
data records and numerical payloads are stored under the external run directory.

## Data And Defaults

Initial calibration uses WikiText-2 official train: 128 sequences of 2048
tokens, seed 20260806. The loader fixes dataset revision
`b08601e04326c79dfdd32d625aee71d232d685c3` and records the dataset fingerprint.
Source attribution is in `UPSTREAM.md`.

The joint fit uses 128 GSM8K official-train requests and 128 decontaminated
Personahub code requests from the fixed Tulu source. Another 16 requests per
task are checks and do not enter Hessians or the GPTQ objective. Dataset
preparation uses seed 20260825 and the train/check row indices in
`llada/configs/calibration_samples.json`; capture selection uses data seed
20260908. Preparation checks 13-gram overlap against benchmark text and
problem-group separation between train and check requests.

Code records are checked against the source byte count, existing SHA-256 and
record count while loading. These checks also apply after relocating the data directory.

During calibration, `humaneval` identifies code-task formatting for the
Personahub source; it does not select HumanEval test records. The input contains
the task and reconstructed function signature, not the reference function
body. All fit stages use `--train-only-calibration`.

Initial LWC uses diagonal reconstruction with 25 scale ratios from 0.6 to 1.2.
R1/R2 are fixed and folded. Initialization uses native RMSNorm and SiLU,
FP32 R4 rounded to BF16, and K8 encoding before native RoPE. Quantized
Linear, QK/PV and LUT Softmax retain their integer and BF16 rescale rules.
Subsequent capture uses the deployment BF16 graph, including staged BF16 R4
and RoPE before K8 encoding. The W8 head uses per-output BF16
absmax/127 scales and round-to-nearest-even codes, without sample fitting.
Static V scales have shape `[32 layers,32 heads]` and come from the initial
train-calibrated parent.

The driver selects the capture configuration for each calibration stage.

Hessians use up to six forwards per request and 64 sampled rows per forward.
The joint objective is `E_G / E_G_parent + E_H / E_H_parent`. GPTQ uses one
BF16 scale per output channel (G-1), damping 0.01, algorithm block size 128,
no activation ordering, and scale ratios `1, .9, .8, .7, .6, .5`. The algorithm
block size is not G128 weight grouping. SiLU uses a shared 16-segment BF16 PWL
table, with up-squared weighting and a monotone positive branch.

SiLU observation samples at most 8192 elements per layer and forward with a
coprime stride. Its offset is `layer_call_count * 8191`, modulo the tensor
length. Each layer counter continues across requests in a logical capture
shard. Histograms and fitted tables record this sampling scheme.

The driver preserves seven logical GSM capture shards, six Personahub train
capture shards and seven Personahub check shards. Both Hessian tasks and the
GPTQ solver use seven logical shards. Jobs queue on the caller's selected
physical GPUs. Reduced sample counts use at most one shard per request.

## Replay And Rebuilding

Replay selects events using the ordinal in the sorted capture-file list,
including shard directory order. It verifies parent/head identities, numerical
graph, actual precision, clipping and SiLU against the installed configuration.
Coverage records retain numerical and generation provenance. The solver
rejects duplicate `(task, sample_id)` pairs and mismatched sources. SiLU
samples, observation metadata and histograms cover the same shard set.

Logical sharding determines sampled forwards, SiLU offsets and FP32 Hessian
reduction order. Changing physical GPU assignment preserves these logical
shards. GPU arithmetic and library versions can still affect numerical results.
