# SUPRA

SUPRA is an attention-guided token-skipping diffusion LLM accelerator with
mixed-precision execution and utilization-aware token prefetching. This
repository provides the algorithm implementation, quantization tools, C-model,
synthesizable RTL, and an evaluation framework.

The examples use **LLaDA-8B-Instruct and GSM8K**. Model dimensions, numerical formats and
control settings are documented with each component.

PDK files, proprietary IP, and other confidential materials are excluded from
this release. Portable memory models and integration interfaces cover
process-specific components. Names and formatting have been refined for
readability while preserving functional behavior. The code includes the feature
mechanisms and the engineering details of their implementation.

## Components

| Component | Start here |
|---|---|
| Algorithm implementation | [Algorithm guide](algorithm/README.md): generation, token selection, mixed precision and prefetching |
| Quantization tools | [Calibration guide](algorithm/CALIBRATION.md): BF16 model to W4 Transformer weights and W8 head |
| C-model | [Hardware guide](hardware/README.md): numerical and control references in `hardware/cmodel/` |
| Synthesizable RTL | `hardware/rtl/` and `hardware/filelists/production.f`; portable SRAM simulation |
| Evaluation framework | [Hardware tests](hardware/cases/README.md), [capture and preparation](integration/README.md), and GSM8K evaluation in `algorithm/` |

## Data Availability

The [data example](data/README.md) provides a complete DRAM transaction trace
from a captured GSM8K layer with Feature 1+2+3 enabled. It supplies partial
evaluation data for the memory-access behavior of token skipping and
mixed-precision execution. [Hardware cases](hardware/cases/README.md),
[algorithm tests](algorithm/llada/tests/) and
[model-capture examples](integration/README.md#more-examples) provide further
inputs, reference outputs and workflows for generating data from algorithm
inference through RTL execution.

## From Model Download to RTL Simulation

1. **Install and download.** Set up the [algorithm environment](algorithm/README.md#install)
   and download [LLaDA-8B-Instruct](algorithm/README.md#model-and-storage).
   Set `MODEL` to the BF16 model directory and `DATA` to an external artifact directory.
2. **Quantize and package.** Follow [calibration](algorithm/CALIBRATION.md) with
   `run.py --mode calibrate`, using `$DATA/reproduction` as the run directory.
   Then use the [model packaging command](algorithm/README.md#package-a-model)
   to create `$DATA/model` with W4 weights, a W8 head, the SiLU table and model metadata.
3. **Test and evaluate the algorithm.** Run the [CPU tests](algorithm/README.md#tests-and-integration),
   then [GSM8K evaluation](algorithm/README.md#evaluate) with that model directory.
   Start with `--limit-per-process 1`, then omit it for full evaluation.
4. **Test the hardware.** Install the [simulation dependencies](hardware/README.md#run-an-included-testcase),
   run the included case below, and use the [focused checks](hardware/README.md#testcases)
   for arithmetic, memory and control modules. This step also runs independently of the model steps.
5. **Run algorithm–RTL integration tests.** In `integration/request_example.json`,
   set absolute paths for `model_metadata/`, `artifact_w8_head/` and
   `silu_table_monotone.json` under the packaged `$DATA/model` directory.
   Follow [the single-layer example](integration/README.md#capture-a-layer-and-run-it)
   to capture and replay inference data, prepare `case.json`, and simulate the RTL
   with output comparisons against the captured references. The same entry points
   handle [larger layer ranges and connected steps](integration/README.md#more-examples).

Algorithm commands run from `algorithm/`; capture and hardware commands below
run from the repository root. Set `SUPRA_ALGORITHM_ROOT` to the same external
directory as `DATA`, and use a separate `SUPRA_ARTIFACT_ROOT` for hardware outputs.
The default algorithm settings are in `algorithm/llada/configs/gsm8k.json`;
capture requests import them through `task_config`.

## Run an Included Hardware Test

Requires Linux, Python 3.8+, GCC/G++ with C++17 support, make and Verilator 5.050.
Simulation defaults to 8 threads; use `--threads N` to match available CPUs.
This path needs no model download or GPU. From the repository root:

```bash
export SUPRA_ARTIFACT_ROOT=/absolute/path/to/hardware-runs
export SUPRA_VERILATOR=/absolute/path/to/verilator-5.050/bin/verilator
python3 -B hardware/scripts/run_testcase.py \
  --case hardware/cases/head_state_update_32tokens/case.json --run-id head-test
```

The runner builds the simulator, executes the case with DRAMSim3, drains memory
transactions and compares raw outputs with the supplied reference. A mismatch
or protocol failure returns a nonzero status. Results are written under
`$SUPRA_ARTIFACT_ROOT/testcase_runner/head-test/`.

Use `--list-cases` with `hardware/scripts/run_testcase.py` for prepared tests or
with `hardware/scripts/run_unit_tests.py` for focused checks. See the
[hardware guide](hardware/README.md) for dependencies and test scope.

## Simulation Outputs

RTL tests report accelerator cycles, per-execution command activity, physical
DDR read/write transfers and effective write bytes. Simulation wall time is
reported separately and depends on the host and case size. DRAMSim3 retains its upstream
statistics. Outputs are reported per execution and for the complete testcase.

## Tested Environment

Hardware simulation uses Verilator 5.050, GCC/G++ and the bundled DRAMSim3
revision documented in [its upstream notice](hardware/third_party/dramsim3/UPSTREAM.md).
The algorithm environment and numerical dependencies are listed in
[algorithm/README.md](algorithm/README.md).
Quantization and algorithm evaluation results can vary with GPU model, random
seeds, and library versions.

We welcome synthesis, place-and-route and post-layout studies with other
libraries and implementation flows. Area, timing and power depend on the target
technology, libraries, flow and operating conditions.

## Maintenance

We welcome feedback, bug reports, and contributions. Future releases may include
fixes and improvements to the implementation, tooling, and documentation while
preserving the core mechanisms of each feature.

## License

SUPRA contributions use a **non-commercial license, with final terms TBD**.
See [LICENSE](LICENSE). Third-party components retain their original licenses
and notices; model weights and datasets have separate terms.

## AI-Assisted Development

AI agents from the GPT-5, GPT-6, and Claude 4 model families assisted with
algorithm and hardware development, tape-out workflows, testing, code refinement,
repository organization, and documentation.
