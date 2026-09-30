# UAPS Retry and Prediction Target

Run from the hardware directory:

```bash
python3 -B scripts/run_testcase.py --case cases/uaps_retry_prediction_target/case.json --run-id joint-rules --threads 8 --jobs 8
python3 -B scripts/run_testcase.py --case cases/uaps_retry_prediction_target/suppressed/case.json --run-id suppressed-history --threads 8 --jobs 8
```

Both cases use the full N8 top and DRAMSim3 LPDDR4-3200 (2 x 16-bit channels).
No model, GPU or algorithm environment is required to replay the supplied testcase.

The first case executes a 16-row W8 head with zero weights, candidate reduction,
two blocks of post state (64 rows), pending updates and joint selection. It
enables BF16 retry confidence 0.25 (`0x3e80`) and prediction target 24 through
the DDR joint descriptor suffix. Sixteen candidate rows replace their history;
48 unobserved rows preserve previously supplied confidence. The joint controller
reads the published state. The complete 2 MiB DDR image is compared.

The `suppressed` case executes a 32-row head with the zero-logit winner (token 0)
in the suppression list. The DDR history must contain action confidence zero,
not the unsuppressed probability 1/64. It retains the transfer-only tied-score
quota of the fixed head testcase and compares the complete 2 MiB image.

The initial state and unaffected post expected originate from the
synthetic C11 testcase. Confidence expected comes from the CPU candidate
functions; joint selection expected comes from the CPU scheduler; physical
metadata uses the C11 packer. `integration/hardware_adapter/uaps_reference_data.py`
regenerates these inputs in the unified package when algorithm dependencies are
available.

`scripts/run_unit_tests.py --case atse --case psme --case sequencer` additionally
checks first attempts, below/equal/above-threshold retries, reused future tokens/TENTATIVE
exemptions, replay, new-block counters, target exhaustion, confirmation reservation,
physical/forecast capacity, malformed settings, backpressure and error drain.
