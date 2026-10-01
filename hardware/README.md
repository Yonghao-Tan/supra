# SUPRA Hardware

Run prepared accelerator testcases, compare actual DDR results with fixed expected data.
The package contains the SUPRA RTL configured with a PE M=8, N=8 array, a portable SRAM model,
DRAMSim3, fixed testcases and source-build runners.
Prepared tests run on a CPU with the dependencies below.

## Run an Included Testcase

Requirements: Linux, Python 3.8+, GCC/G++ with C++17 support, make and Verilator 5.050.
Simulation defaults to 8 threads; use `--threads N` to match available CPUs.
Builds and outputs must be outside the source tree.

```bash
export SUPRA_ARTIFACT_ROOT=/absolute/path/to/hardware-runs
export SUPRA_VERILATOR=/absolute/path/to/verilator-5.050/bin/verilator
cd hardware  # from the integrated release root
python3 -B scripts/run_testcase.py --case cases/head_state_update_32tokens/case.json \
  --run-id post32 --threads 8 --jobs 8
```

The runner builds the simulator, loads the supplied DDR image, submits the testcase configuration address, waits for completion and DDR drain, and compares actual bytes with the expected files.
`--threads` controls simulation parallelism; `--jobs` controls compilation parallelism. Use a new run ID for each execution.

An **execution** is one externally launched DDR configuration. It can select
1–32 consecutive Transformer layers and optional final normalization, LM head,
token-state processing and refresh selection; a head-only execution is also
supported. The execution controller advances layers, token rounds and operators
internally. The runner launches subsequent executions in testcase order.

Binary reuse (`--binary`) checks the actual maintained RTL, testbench, C-model
and transitive source dependencies recorded by the build, as well as compiled
SiLU constants and required runner features. A missing record or changed source
requires a fresh build. Interrupted runners terminate their own process group,
including compiler or simulator children.

Results are under `$SUPRA_ARTIFACT_ROOT/testcase_runner/post32/`:

| Output | Content |
|---|---|
| `data/summary.json` | Completion, raw comparison results, effective WSTRB bytes and physical 32-byte DDR read/write transfers for every execution and the complete testcase |
| `data/*.actual.bin` | Actual DDR output ranges |
| `logs/replay.log` | Launch/completion, mismatch details and final status |
| `logs/build.log` | Source-build command and compiler output |

A failed completion, protocol error or raw mismatch returns a nonzero exit code. Mismatches identify the address and expected/actual bytes.

## Testcases

| Test | Entry | Scope |
|---|---|---|
| Prepared testcase | `cases/head_state_update_32tokens/case.json` | 32-token W8 head, 64-token test vocabulary, final normalization and transfer-only state update; fixed input and independent C11 expected |
| C-model numerics | `scripts/run_unit_tests.py --case numeric` | Integer/BF16 arithmetic, residual addition, SiLU, SiLU×up, RMSNorm, RoPE, QK/PV and softmax |
| Datapath RTL | `scripts/run_unit_tests.py --case pe --case quantizer --case sram --case layer-loader` | Mixed PE, A4 clipping, SRAM ports and layer configuration |
| Control | `scripts/run_unit_tests.py --case psme --case atse --case sequencer` | DVSC state transitions, ATSE/UAPS selection, precision budgets, descriptor validation and error drain |
| Dependency state | `scripts/run_unit_tests.py --case invalidation --case refresh-score` | Changed-token confidence, pending-risk modes, key transitions, consumption and DMA error/drain |
| Metadata loader | `scripts/run_unit_tests.py --case token-loader` | Packed token precision, actual loaded-token events and KV-write suppression |
| DDR test model | `scripts/run_unit_tests.py --case ddr-model` | DDR images, read/write progress, backpressure and response errors |
| AXI master | `scripts/run_unit_tests.py --case axi-master` | Burst accounting, backpressure, error/abort drain and restart at 128/256-bit widths |
| Softmax | `scripts/run_unit_tests.py --case softmax` | Batch addressing, BF16/int8 row arithmetic, tail lanes, probability capture and response drain |
| Matmul activation | `scripts/run_unit_tests.py --case matmul-activation` | Scale/quantized payload forwarding, delayed max/SRAM responses, abort and restart |
| Matmul output | `scripts/run_unit_tests.py --case matmul-output` | Local, paired, residual and fused BF16 outputs; batch tails, abort and restart |
| Elementwise | `scripts/run_unit_tests.py --case elementwise` | Arithmetic/SRAM handshakes with supplied arithmetic responses, DMA errors, abort drain and restart |
| Attention context writer | `scripts/run_unit_tests.py --case attention-context-writer` | Exact output payloads, outstanding responses, read/write errors and abort drain |
| Hidden transfer | `scripts/run_unit_tests.py --case hidden-transfer` | Token/stripe layouts, indexed hidden/embedding rows, backpressure and error/abort drain |
| Testcase checker | `scripts/run_unit_tests.py --case testcase-checker` | Head/Attention raw events, token maps, signed zero/NaN, DDR copies and initial segments |

The provided head/state testcase includes synthetic weights and inputs. Expected bytes were generated independently of RTL execution.
Control records under `cases/control/` specify numeric formats and the reference used to generate expected values.

Select focused checks:

```bash
python3 -B scripts/run_unit_tests.py --run-id datapath --jobs 8
python3 -B scripts/run_unit_tests.py --run-id controls \
  --case psme --case atse --case sequencer --jobs 8
python3 -B scripts/run_unit_tests.py --run-id stream --case stream --jobs 8
```

List available checks or prepared cases without creating a build or requiring data downloads:

```bash
python3 -B scripts/run_unit_tests.py --list-cases
python3 -B scripts/run_testcase.py --list-cases
python3 -B scripts/run_testcase.py --case /path/to/case.json --validate-only
```

The default `make` target runs numerical, PE, quantizer, SRAM, layer-loader and raw-checker
checks. The checker includes deliberately incorrect inputs that must be rejected.
Long prepared layers and continuous executions require an explicit testcase run.
`--validate-only` checks the prepared payload references and all execution configurations
before a simulation run.

Run every public model-free unit check with one entry:

```bash
make check RUN_ID=public-check JOBS=8
```

Focused tests use one simulation thread. The full testcase simulator defaults to eight; `--threads` accepts any positive count.
See `cases/README.md` for testcase inputs, coverage and organization.
Successful prepared runs record host `simulation_wall_seconds` in `run.json`, separately
from accelerator cycles in `data/summary.json`. Per-execution cycles exclude
initial reset; the complete summary includes reset and the final DDR drain.

## Algorithm parameter conversion

From this hardware directory, convert a supported task configuration with:

```bash
python3 -B scripts/build_forward_postprocess_config.py \
  --config ../algorithm/llada/configs/gsm8k.json --tier feature12
```

Use `../algorithm/llada/configs/humaneval.json` for HumanEval.
The command prints DDR scalar fields to stdout using only the Python standard library.
It converts the selected algorithm configuration into hardware control fields. Use the head, handoff and layer preparation entries
to allocate addresses and build the DDR image from their documented binary inputs.

The tiers select the following scalar settings. Capture the corresponding
algorithm configuration to obtain its actual workload and control inputs.

| Tier | Precision and selection |
|---|---|
| `baseline` | Full-sequence A8 Transformer execution and fixed-k submission |
| `feature1` | ATSE attention-guided token skipping with fixed-k decoding |
| `feature12` | ATSE token skipping and PSME draft/verify decoding with mixed-precision execution |
| `feature123` | Feature1+2 with UAPS utilization-aware token prefetching |

Fixed-k configurations use `decode_k` (default 3). L31/head work follows the
captured consumers.

For `feature123` configurations with dependency-tie ranking enabled, supply
`--block-step-index` with the actual step in the current block. For example,
the following prints fields for block-local step 2:

```bash
python3 -B scripts/build_forward_postprocess_config.py \
  --config ../algorithm/llada/configs/gsm8k.json --tier feature123 --block-step-index 2
```

## Continuous Testcases

Both included cases use zero Transformer weights and synthetic C11 control inputs:

| Case | Ordered launches |
|---|---|
| `cases/in_block_handoff/case.json` | L31, post-processing, next L0 |
| `cases/cross_block_handoff/case.json` | L31, post-processing, full 164-token L0 scout, 80-token L1 deep selection |

Run either with `python3 -B scripts/run_testcase.py --case <case> --run-id <new-id> --threads 8 --jobs 8`. Later launches use actual DDR state and hidden output. Checks cover accepted embedding reads, independent scout scores and deep metadata, and selected persistent cache bytes.

Every run writes the compact `command_profile.tsv`. Add `--axi-read-trace`
only for focused memory calibration that needs every accepted AXI read address
and burst length; the resulting `axi_ar_trace.tsv` can be large and is omitted
by default. `--ddr-idle-trace` independently records drained DDR intervals.

`--dram-trace` records DRAMSim3 transactions in `data/dramsim3/*addr.trace`.
Each line contains a hexadecimal byte address relative to the DDR aperture,
`READ` or `WRITE`, and a decimal DRAM-cycle timestamp. Each transaction transfers
32 bytes; the DDR configuration defines the cycle period (`tCK`). This option
enables tracing when building the simulator; a reused binary needs the same setting.

## Prepared Input Format

One JSON file describes a case. Paths are relative to that file; the concrete example is `cases/head_state_update_32tokens/case.json`.
`config/*.json` defines the packed records. The corresponding fixed interface
definitions are in `rtl/config/`, `cmodel/generated/` and `tb/generated/`.
When changing a record, update its JSON, SystemVerilog, C and C++ definitions
together, keeping field offsets, widths and constants consistent.
W4 Transformer layer tables use empty enhancement-plane ranges; W8 head data has its separate format.

| Field | Content |
|---|---|
| `ddr_image` | Initial DDR image containing testcase/execution records, address tables, weights, scales and initial state |
| `memory_map` | Existing DDR address ranges and read/write permissions |
| `dramsim3_config` | Device timing and channel configuration |
| `executions` | Ordered execution-configuration addresses, IDs and cycle bounds |
| `executions[].handoff_actions` | Optional host packing of actual post state or RTL-produced metadata before the next execution; see the two continuous cases |
| `executions[].check_embedding_reads` | Check accepted embedding read addresses, lengths and counts against the actual metadata used for that launch |
| `executions[].expected_post_block_completions` | Optional boolean sequence checked at accepted post-processing completions |
| `expected` | Named DDR ranges with address, byte count and expected raw file |
| `initial_segments` | Optional address/bytes/path records loaded once before the first launch, for sparse initial input |
| `head_checkpoints` | Optional captured final-norm, activation codes/scales, INT32 sums and BF16 logits, checked from actual events |
| `head_checkpoints.execution_index` | Required when checking one head inside a sequence of launches; zero-based index |
| `attention_checkpoints` | Optional accepted-event checks for preserved hidden write/refill, Query P8 codes/scales, QK BF16 scores, softmax BF16 probabilities, final P8 codes/scales, PV context and the Attention residual, with an explicit physical-to-logical token map; context requires a full-output layer |
| `attention_checkpoints.execution_index` | Required when checking one layer inside a sequence of launches; zero-based index |
| `silu_mode` | Compile-time coefficient selection: `monotone` (default) or `explicit` |
| `silu_segments` | For `explicit` only: sixteen uint32 words, intercept in bits 31:16 and slope in bits 15:0 |

Multiple executions keep the same DUT and DDR state; the image is not reloaded between launches.
An individual execution may specify `copies`: source address, destination address and byte count.
These copy actual DDR contents after the preceding execution has drained, before starting the next execution.
They support an independent hidden input area for non-identity layer connections, including overlapping copies.
Initial segment loads and inter-launch host copies do not consume simulated accelerator or DDR cycles;
their byte counts are reported separately.
`handoff_actions` prepare the next execution from the completed state and
metadata between drained launches. The continuous cases provide examples.
Completion records and tensor outputs can both be compared as explicit DDR ranges.
`summary.json` reports `read_bytes` as full-width physical DDR reads and `write_bytes` as bytes enabled by WSTRB.
It separately reports `physical_read_bytes` and `physical_write_bytes` from accepted 32-byte DRAM transactions,
both for each launch and for the complete testcase. Use the physical fields for actual DDR DQ traffic.
SiLU coefficients are selected at build time; changing them requires a fresh binary.

## Model-Derived Test Data

Use the [capture and preparation guide](../integration/README.md) to generate
layer, head, boundary and connected-step inputs from model inference. The
shared entry is `integration/prepare_testcase.py`; run its resulting case with
`scripts/run_testcase.py`.

Captured layer tests execute RMSNorm, Q/K/V/O projections, RoPE, Attention,
residual additions and the gate/up, SiLU and down-projection path in RTL.

Preparation uses the captured query positions, A4/A8 precision, KV writes,
commit positions and L31 consumers. It packs the model's raw observations as
independent expected data. During simulation, layer connections read actual
hidden and cache outputs; the checker compares raw tensor and control results.

Scheduling options belong in the preparation configuration. FFN groups contain
four or six batches within 18 activation compute groups (589,824 bytes).
See the preparation guide and
`integration/prepare_layer_testcase.py --help` for configuration fields.

## Hardware Architecture

`supra_top` is the synthesizable accelerator-core top, with launch/completion
and AXI4 interfaces. Confidential chip-level integration, including the MCU,
external memory interface, PLL, and system control, is excluded from this release.
Implementation-specific optimizations, such as clock gating, are also excluded
from this release.

- N8 PE: M8 x N8, W4A4/W4A8 and INT8 Attention operations, with INT32 dot accumulation.
- Transformer weights are W4; the head uses W8A8. Dynamic A4/A8, BF16 scales, R4 and configured A4 clipping are implemented.
- SRAM: 26 macros, 640 KiB, 128-bit words. The portable model preserves dual-port, byte-mask and synchronous-read behavior.
- AXI: 256-bit data, independent read/write channels, bounded outstanding requests, backpressure and error drain.
- ATSE performs attention-guided token skipping; UAPS provides utilization-aware token prefetching.
- PSME combines draft/verify state control (DVSC) with mixed-precision execution
  through the phase-shared compute scheduler (PSCS).

## Source and License

`config/` defines DDR records shared by the hardware modules: refresh records
for ATSE, state/update records for PSME, token metadata for the compute scheduler,
and prefetch records for UAPS.

`rtl/` contains control, compute, memory and operators; `filelists/production.f` is the production source list.
The DRAMSim3 simulation filelist adds SRAM/DDR testbench plumbing. `tb/testcase_runner.cpp` drives prepared inputs and compares results.
`cmodel/` provides hardware numeric and state references; `cases/` contains fixed test inputs and independent expected data.
The algorithm implementation and quantization tools are in `../algorithm/`.
SUPRA contributions use the provisional Non-Commercial License (final terms TBD); see `LICENSE` and `NOTICE`. Third-party licenses remain under `third_party/`.
