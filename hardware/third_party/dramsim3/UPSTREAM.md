# DRAMSim3 Source Record

- Upstream: `https://github.com/umd-memsys/DRAMSim3.git`
- Tag: `1.0.0`
- Commit: `29817593b3389f1337235d63cac515024ab8fd6e`
- License: MIT, preserved in `LICENSE`

Project patches:

1. Move write completion callback from transaction acceptance to
   `WR issue + configured write_delay`.
2. Reject a read or write to an address with a pending write until that write
   completes. This prevents the upstream same-address write merge and gives
   the functional store an unambiguous commit point.
3. Expose read-only effective geometry values needed by the integration
   startup checks.
4. Start write drain when the read queue is empty and at least one write is
   pending. Upstream otherwise leaves a short write-only stream pending
   forever unless its hard-coded high-water mark is crossed.
5. Expose the existing address-to-channel decode as a read-only query so the
   verification backend can close per-channel submitted and completed bytes
   without duplicating the configured address mapping.
6. Account for LPDDR4 command encoding and DQS offsets in read/write completion,
   command spacing, precharge and per-bank refresh timing.

Bundled header-only dependencies and their upstream notices are preserved in
`ext/`. Build products must be written outside the repository through
`SUPRA_ARTIFACT_ROOT`.
