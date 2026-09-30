# Capture, Prepare and Run

Run these commands from the SUPRA repository root. Capture uses the
[algorithm environment](../algorithm/README.md) and a matching
[quantized model release](../algorithm/CALIBRATION.md). Layer/head preparation
uses Python and NumPy; observed control and connected steps also use the algorithm
Python dependencies and a C/C++ compiler. Simulation uses the dependencies in the
[hardware guide](../hardware/README.md).

## Capture a Layer and Run It

Set the model paths in `integration/request_example.json`. It contains the
notebook-price prompt used for the single-layer example. Its
`task_config` imports `algorithm/llada/configs/gsm8k.json`; explicit `model_args`
and `generation_args` override the imported fields. Set `tier` to `baseline`, `feature1`, `feature12`
or `feature123` in the request to select the same progression as evaluation.
The default is `feature123`.
[`examples/regular_layer.json`](examples/regular_layer.json) is a capture
selection list for L0 of forward 1. The request uses the GSM8K defaults. Use fresh
capture, preparation and run directories.

```bash
export SUPRA_ALGORITHM_ROOT=/absolute/path/to/algorithm-data
export SUPRA_ARTIFACT_ROOT=/absolute/path/to/hardware-runs
export SUPRA_VERILATOR=/absolute/path/to/verilator-5.050/bin/verilator
python3 -B integration/model_capture.py capture \
  --request-config integration/request_example.json \
  --selections integration/examples/regular_layer.json \
  --output "$SUPRA_ALGORITHM_ROOT/layer/payload" \
  --index "$SUPRA_ALGORITHM_ROOT/layer/index.json"
CASE=$(python3 -B integration/prepare_testcase.py \
  --index "$SUPRA_ALGORITHM_ROOT/layer/index-regular-layers.json" \
  --output "$SUPRA_ARTIFACT_ROOT/prepared/layer")
python3 -B hardware/scripts/run_testcase.py \
  --case "$CASE" --run-id layer --threads 8 --jobs 8
```

The capture-set index is `layer/index.json`; its single layer index is
`layer/index-regular-layers.json`, which the prepare command consumes.
Change the prompt or task settings in the request and the forward/layer in the
selection list to capture another execution.

Capture observes actual CUDA/Triton inputs and outputs, then replays their
numerical checkpoints before returning. Preparation packs those tensors into
DDR inputs and independent expected files. An existing capture starts at the
prepare command; a prepared `case.json` starts at the runner. For a small test
without model weights, use an [included hardware case](../hardware/cases/README.md).
Even a short captured sequence contains model-size weights, so payloads remain
under the external roots.

## More Examples

These configurations select actual inference events using the default Feature
1+2+3 settings. Set the model paths in the selected request. Both initialization
examples share one request; the block transition uses the notebook request above.

| Example (`EXAMPLE`) | Request (`REQUEST`) | Captured execution |
|---|---|---|
| `full_layers_head` | [integration/examples/full_layers_request.json](examples/full_layers_request.json) | L0–L31, full-vocabulary head and step control |
| `cache_initialization` | [integration/examples/initialization_request.json](examples/initialization_request.json) | Initial cache population, L0 |
| `block_initialization` | [integration/examples/initialization_request.json](examples/initialization_request.json) | Block initialization, L0 and selected L1 rows |
| `step_transition` | [integration/examples/step_request.json](examples/step_request.json) | L30–L31 and head/control, then the next step's L0 |
| `block_transition` | [integration/request_example.json](request_example.json) | Block-ending L30–L31 and head/control, then the next block's L0 |

Choose an example and its request from the table, then capture it:

```bash
EXAMPLE=full_layers_head
REQUEST=integration/examples/full_layers_request.json
python3 -B integration/model_capture.py capture \
  --request-config "$REQUEST" \
  --selections "integration/examples/$EXAMPLE.json" \
  --output "$SUPRA_ALGORITHM_ROOT/$EXAMPLE/payload" \
  --index "$SUPRA_ALGORITHM_ROOT/$EXAMPLE/index.json"
```

For `full_layers_head`, `step_transition` or `block_transition`, prepare the
capture set directly:

```bash
CASE=$(python3 -B integration/prepare_testcase.py \
  --steps "$SUPRA_ALGORITHM_ROOT/$EXAMPLE/index.json" \
  --output "$SUPRA_ARTIFACT_ROOT/prepared/$EXAMPLE")
```

For `cache_initialization`, prepare the captured initial layer:

```bash
CASE=$(python3 -B integration/prepare_testcase.py \
  --index "$SUPRA_ALGORITHM_ROOT/$EXAMPLE/index-cache_initialization-layers.json" \
  --layer-options integration/examples/cache_initialization_options.json \
  --output "$SUPRA_ARTIFACT_ROOT/prepared/$EXAMPLE")
```

For `block_initialization`, preparation connects the scout and selected deep rows:

```bash
CASE=$(python3 -B integration/prepare_testcase.py \
  --index "$SUPRA_ALGORITHM_ROOT/$EXAMPLE/index-block_initialization-layers.json" \
  --layer-options integration/examples/block_initialization_options.json \
  --output "$SUPRA_ARTIFACT_ROOT/prepared/$EXAMPLE")
```

Run the prepared case with the same runner:

```bash
python3 -B hardware/scripts/run_testcase.py \
  --case "$CASE" --run-id "$EXAMPLE" --threads 8 --jobs 8
```

Scheduling options are kept in the two JSON files. See
[Data Size and Simulation Time](#data-size-and-simulation-time) when choosing
an example and allocating resources.

## Capture Layers, Head and Control Together

`--selections FILE` accepts a list of named forward selections. Each entry
contains `name` and either a zero-based `capture_index` or the three fields
`block_index`, `step_index` and `phase`. Use actual event identities from the
request's trace. Each selected event occurs once in the list.
Supported phases are `full_sequence`, `boundary_refresh`, `local_block`,
`local_confirmation` and `local_forced_finish`.

`first_layer` and `layer_count` select 1–32 consecutive layers within L0–L31.
`include_head` adds the matching head; `head_only` selects just the head.
`control_state` records the state before and after the selected forward.
For a selected final forward, capture also records `request_end` and
`current_state_at_request_end`, including tail confirmation.

Alternatively, the selections file can contain `start` and `end` objects.
Each endpoint contains an event identity and a `layer` in 0–31. Capture starts
at the first endpoint's layer, includes all intervening forwards with
L0–L31/head/control, and ends at the last endpoint's layer. This preserves
confirmation-only, forced-finish and block-boundary events in their actual order.

The capture-set index lists the generated layer/head indices and their layer
ranges. For named selections, child filenames append `-<name>-layers.json` or
`-<name>-head.json` to the index stem. Each child contains its observations and
replay result. Equal static weights, scales and constants are stored once per
capture request. Move the capture directory together with its sibling payloads;
tensor paths and capture-set references remain relative to their indices.
Numerical replay and layer/head preparation also accept `--payload-root` as an
alternative base for relative tensor paths.

## Prepare a Step or a Sequence

The unified entry prints the resulting `case.json` path. Pass that printed path
to the runner: a regular case resides in `execution/`, and a connected sequence
uses the first step's execution directory.

| Mode | Input and execution |
|---|---|
| `layer` (default) | One layer index; execute its consecutive layer range. `--output-subset` selects the captured L31 consumers when the range ends at L31. |
| `boundary` | Explicit L0 scout/L1 preparation. The `layer` and `regular` modes detect captured scout/deep boundaries and prepare L0 followed by L1 through the selected endpoint. |
| `head` | One head index; execute final normalization and the full captured W8 head. |
| `regular` | A regular layer range ending at L31 plus `--head-index`; capture both with control enabled. Preparation connects layer output, head, state updates and next selection, or terminal completion. |

Pass a completed capture-set index directly to `prepare_testcase.py --steps`
and select a fresh `--output` directory. Include every actual forward between
the first and last selected events. Every non-final event supplies layers,
head and control; the final event can stop after its layers or include its
head/control completion. Later events start at L0. The connection checks actual
selected positions, precision, embedding inputs and persistent cache before
carrying RTL-produced values forward.

At a scout/deep block boundary, the connection carries the completed token and
state tables, pending values, cache-refresh flags and accumulated transition
masks. L0 supplies live scores to the selector; deep layers read its selected
rows from the actual L0 output. A final L0-only endpoint executes the scout and
selector without a deep-layer launch. Confirmation and forced-finish events use
their own postprocess modes and retain their actual capture indices.

## Configuration

Algorithm decisions come from the captured request and control events. Preparation
options select memory layout, DDR timing and scheduling. `--ddr 2400` or `3200`
selects the bundled LPDDR4 model; 3200 is the default. `--layer-options FILE`
supplies a JSON object of explicit preparation keywords. For per-step options,
`--steps` also accepts a list with `index`, optional `head_index`, optional
`layer_range` and optional `layer_options` in each entry. Index paths resolve
relative to that steps file. Regular mode enables L31 output selection and
Attention checkpoints automatically; keep Attention grouping disabled for it.

Layer preparation uses D4096/F12288/H32, sequence lengths up to 2048, G-1 W4
weights and captured A4/A8 rows. A consecutive range shares query positions,
precision and KV write/commit sets. Different KV write and persistent-commit
sets use a single layer with at most 432 committed tokens. L31 output subsets
work for both a single L31 and a multi-layer range ending at L31. Their current,
future and retained-hidden consumers must be present in the capture.

For advanced options, run `integration/prepare_layer_testcase.py --help` or
`hardware/scripts/prepare_head_testcase.py --help`; see also the
[hardware parameter and testcase guide](../hardware/README.md#algorithm-parameter-conversion).
Automatic boundary preparation derives `block_initialization_reference` or
`boundary_reference` from the captured inputs and uses the RTL selector. The
explicit `boundary` mode also accepts those references; without a reference it
replays the captured scout/deep layout. FFN/QKVO reuse options retain their capacity and checkpoint
checks. Synthetic generation and raw inspection are available through
`integration/layer_reference_data.py --help`.

## Checks and Results

Expected tensors come from actual model observations or the stated numerical
reference. Preparation checks producer/consumer raw values and packs existing
expected data. Connected executions consume actual RTL output.
Comparisons preserve raw BF16/FP32 bits, integer codes, scales and accumulator
values. Layer checks cover final hidden and persistent K/V, including unchanged
positions; selected Attention checkpoints and head intermediate tensors add
accepted-event checks. Control connections compare observed decisions and state.

The simulation path is AXI → DPI-C → C++ DRAMSim3. The functional DDR store holds
bytes; DRAMSim3 supplies 32-byte transaction timing, backpressure and completion.
Executions drain accepted transactions before their completion checks. Initial
image loading and inter-execution host copies/configuration have separate
accounting from simulated accelerator cycles.

Results are under `$SUPRA_ARTIFACT_ROOT/testcase_runner/<run-id>/`:
`data/summary.json` contains checks, cycles and physical/effective DDR bytes;
`logs/replay.log` identifies mismatches; `logs/build.log` records compilation;
`run.json` records host simulation wall time. `--threads` controls simulator
threads, while `--jobs` controls compilation. Assign independent CPU sets for
parallel runs. `--validate-only` checks a prepared case before simulation.

## Integration Tests

With the algorithm environment, pytest, NumPy and a C/C++ compiler installed,
run from the repository root:

```bash
mkdir -p "$SUPRA_ARTIFACT_ROOT"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TMPDIR="$SUPRA_ARTIFACT_ROOT" \
  PYTHONPATH=integration:algorithm/llada:hardware/scripts \
  python -B -m pytest -q -p no:cacheprovider \
  integration/test_model_capture.py integration/tests
```

These CPU tests cover capture events, tensor formats, numerical references and
testcase preparation. Model inference and RTL simulation use the examples above.

## Data Size and Simulation Time

TD contains initial memory inputs and reference outputs. Model captures also
store intermediate tensors used for replay. The following ranges help plan a
run; CPU resources, simulator threads and the selected workload affect wall time.

| Test data | Main storage and work | Simulation wall-time scale after compilation |
|---|---|---|
| Small supplied head/control cases | Small memory images and control records | Seconds for the small head/control examples |
| A captured regular layer or full-vocabulary head | Model weights plus selected activations, caches and reference tensors; hundreds of MB or more | Minutes to hours |
| Cache or block initialization layers | Layer weights plus activations and caches for many token rows | Several hours; reserve a long run |
| Many layers or connected steps | Weights for the selected layers plus per-step tensors and state | Hours to a day or longer, depending on selected work and host resources |

File size alone is a poor runtime predictor: weights can be reused across many
token rows or steps. Record layer count, query rows and their A4/A8 split,
sequence length, head output rows and execution count when comparing workloads.
Repeated steps can add substantial computation while sharing the same weights.
Capture and prepared TD are separate storage allocations; intermediate reference
tensors can add substantially to the weight payload.

Budget separately for GPU capture/replay, CPU preparation, simulator compilation
and RTL simulation. The first full-accelerator build takes minutes or longer;
reuse a compatible binary with `--binary` for subsequent cases. During simulation,
use the cycle progress and elapsed wall time to assess the remaining run. Simulator
cycles per wall-clock second depend on the host and execution stage.
