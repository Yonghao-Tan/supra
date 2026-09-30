# Head and State Update

`case.json` runs a 32-row W8 head and transfer-only state update using 4096
BF16 hidden features and a synthetic 64-token vocabulary. It compares the
complete 2 MiB DDR image against the independent C11 reference, including
unchanged inputs and generated outputs.

Run from the hardware directory with the environment in the hardware guide:

```bash
python3 -B scripts/run_testcase.py \
  --case cases/head_state_update_32tokens/case.json --run-id head-state
```

The supplied image, memory map and expected file are ready to run. Paths in
`case.json` are relative to that file. The runner checks completion, DDR drain
and raw bytes, and writes `data/summary.json`, actual output files and
`logs/replay.log` under the run directory.

`mismatch.json` exercises the checker by using the initial image as the expected
output. It returns a nonzero status and identifies the changed DDR addresses.
