"""Export focused joint selection expected from the public algorithm on CPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import shutil
import subprocess


def _algorithm_scheduler(*, target_joint_tokens, max_next_tokens):
    """Construct the public algorithm scheduler at the integration boundary."""
    from generation.lookahead import DynamicJointWindowScheduler

    return DynamicJointWindowScheduler(
        target_joint_rows=target_joint_tokens,
        max_next_rows=max_next_tokens,
    )


def _algorithm_select(scheduler, public_inputs):
    """Map SUPRA token names to the algorithm package's stable API."""
    algorithm_inputs = dict(public_inputs)
    for public_name, algorithm_name in {
        "base_activation_bits": "base_row_bits",
        "next_step_base_activation_bits": "next_step_base_row_bits",
        "next_activation_bits": "next_row_bits",
        "future_admission_attempts": "source_b_attempts",
        "max_future_admission_attempts": "max_source_b_attempts",
        "future_admission_retry_min_confidence": "source_b_retry_min_confidence",
        "prediction_token_target": "target_prediction_rows",
        "current_prediction_tokens": "current_prediction_rows",
    }.items():
        algorithm_inputs[algorithm_name] = algorithm_inputs.pop(public_name)
    return scheduler.select(**algorithm_inputs)


def _public_residency(residency):
    algorithm_fields = asdict(residency)
    return {
        "a4_token_count": algorithm_fields["a4_rows"],
        "a8_token_count": algorithm_fields["a8_rows"],
        "issued_activation_slots": algorithm_fields["issued_slice_units"],
        "pe_issue_groups": algorithm_fields["pe_issue_groups"],
    }


def make_reference_data():
    parent = Path(__file__).resolve().parents[2]
    algorithm = parent / "algorithm/llada"
    sys.path.insert(0, str(algorithm))
    import torch

    def encode(value):
        if isinstance(value, torch.Tensor):
            raw = value.view(torch.int16) if value.dtype == torch.bfloat16 else value
            return dict(dtype=str(value.dtype), shape=list(value.shape), raw=raw.tolist())
        return value

    records = []
    scheduler = _algorithm_scheduler(target_joint_tokens=48, max_next_tokens=8)

    def capture(name, *, current=20, target=24, threshold=.25, limit=3,
                tentative=(2,), overlap=(1,), attempts=None, confidence=None, base_count=28,
                base_a8=False, forecast_a8=False, count=8):
        positions = list(range(base_count-len(overlap))) + [32+i for i in overlap]
        inputs = dict(base_positions=torch.tensor(positions),
            base_activation_bits=torch.full((base_count,), 8 if base_a8 else 4, dtype=torch.int8),
            next_step_base_activation_bits=torch.full((base_count,), 8 if forecast_a8 else 4, dtype=torch.int8),
            next_block_start=32, next_activation_bits=torch.tensor([8 if i in tentative else 4 for i in range(count)], dtype=torch.int8),
            next_unresolved=torch.ones(count, dtype=torch.bool),
            next_tentative=torch.tensor([i in tentative for i in range(count)]),
            next_priority=torch.tensor([1., .875, .75, .625, .5, .375, .25, .125] + [.0625]*(count-8), dtype=torch.bfloat16),
            next_service_count=torch.zeros(count, dtype=torch.int32),
            future_admission_attempts=torch.tensor(attempts if attempts is not None else [0, 4, 4, 1, 1, 1, 1, 1]+[1]*(count-8), dtype=torch.int32),
            next_last_confidence=torch.tensor(confidence if confidence is not None else [-1., 0., 0., .2490234375, .25, .251953125, 0., .5]+[.5]*(count-8), dtype=torch.bfloat16),
            max_future_admission_attempts=limit,
            future_admission_retry_min_confidence=threshold,
            prediction_token_target=target, current_prediction_tokens=current)
        result = _algorithm_select(scheduler, inputs)
        expected = dict(progress_local_positions=result.progress_local_positions,
            added_local_positions=result.added_local_positions, added_activation_bits=result.added_row_bits,
            base_residency=_public_residency(result.base_residency),
            joint_residency=_public_residency(result.joint_residency),
            next_step_verification_residency=_public_residency(result.next_step_verification_residency))
        records.append(dict(name=name, kind="added_future_token_selection",
            retry_min_confidence_bf16=int(torch.tensor(threshold, dtype=torch.bfloat16).view(torch.int16)),
            inputs={key: encode(value) for key, value in inputs.items()},
            expected={key: encode(value) for key, value in expected.items()}))

    capture("retry_first_reused_future_token_tentative_threshold_edges")
    capture("retry_without_attempt_cap", limit=-1)
    capture("target_disabled", target=-1)
    capture("retry_disabled", threshold=0)
    capture("both_disabled", target=-1, threshold=0, limit=-1)
    capture("target_reached", current=24, tentative=())
    capture("target_exceeded", current=24, target=16, tentative=())
    capture("target_due_added_reservation", current=23, tentative=(2, 3), overlap=(1, 4))
    capture("target_due_existing_and_added", current=24, tentative=(1, 2, 3), overlap=(1, 4))
    capture("target_due_above_max_next", current=24, tentative=tuple(range(12)), overlap=(1, 4), count=12)
    capture("forecast_capacity_rejection", current=20, base_count=32, forecast_a8=True, overlap=(1,))
    capture("all_first_attempts", attempts=[0]*8, confidence=[-1.]*8, tentative=(), overlap=())
    capture("retry_suppressed_zero", attempts=[1]*8, confidence=[0.]*8, tentative=(), overlap=())
    capture("target_64", target=64, limit=-1, threshold=0, tentative=())
    capture("target_only_no_attempt_state", target=24, limit=-1, threshold=0, tentative=())
    return dict(schema="supra-uaps-token-selection/v1", target_joint_tokens=48, max_next_tokens=8,
        records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--testcase-output", type=Path, help="prepare an executable head/postprocess/refresh/UAPS testcase below the hardware artifact root")
    args = parser.parse_args()
    reference_data = make_reference_data()
    if not args.output and not args.testcase_output:
        parser.error("select --output or --testcase-output")
    if args.output:
        args.output.write_text(json.dumps(reference_data, indent=2) + "\n")
        print(f"Exported {len(reference_data['records'])} UAPS token-selection records: {args.output}")
    if args.testcase_output:
        print(prepare_testcase(args.testcase_output))


def prepare_testcase(output):
    import torch
    from numerics.candidate import streaming_candidate_bf16, candidate_action_confidence

    parent = Path(__file__).resolve().parents[2]
    hardware = parent / "hardware"
    sys.path.insert(0, str(hardware / "scripts"))
    from artifact_paths import artifact_root
    from prepare_handoff_testcase import read_record, record, build_reference
    output = output.resolve()
    if artifact_root() not in output.parents or output.exists():
        raise ValueError("testcase output must be a new directory below the hardware artifact root")
    output.mkdir(parents=True)
    seed = hardware / "cases/in_block_handoff"
    initial = bytearray((seed / "initial.bin").read_bytes())
    expected = bytearray((seed / "post_expected.bin").read_bytes())
    # This execution omits the seed's preceding Transformer, so its K/V scale
    # storage must retain initial bytes rather than the Layer31 result.
    expected[0x160000:0x1a0000] = initial[0x160000:0x1a0000]
    base = 0x10000000

    def put(data, offset, size, value):
        data[offset:offset+size] = value.to_bytes(size, "little")

    post = read_record("forward_postprocess_config", initial, 0xd0000)
    post["flags"] = 1
    raw_post = record("forward_postprocess_config", post)
    for image in (initial, expected):
        image[0xd0000:0xd00c0] = raw_post
    predictions = [read_record("prediction_record", initial, 0xd1000+i*32)
                   for i in range(16)]
    # The fixed testcase has zero W8 weights. Its logits are zero irrespective of hidden.
    if any(initial[0x100000:0x140000]):
        raise ValueError("control seed no longer has zero head weights")
    proposal, _, confidence = streaming_candidate_bf16(torch.zeros(1, 16, 64, dtype=torch.bfloat16))
    action = candidate_action_confidence(proposal, confidence, []).to(torch.bfloat16).view(torch.int16).flatten().tolist()
    observed = {p["token_position"]: action[i] for i, p in enumerate(predictions)}
    for i in range(64):
        position = int.from_bytes(initial[0xd6000+i*32:0xd6002+i*32], "little")
        history = 0x3f00 if position % 2 else 0x3e80
        for image in (initial, expected):
            put(image, 0xd6000+i*32+28, 2, history)
            put(image, 0xd6000+i*32+30, 1, 1)
    states = {}
    for i in range(64):
        entry = read_record("token_state_entry", expected, 0xd7000+i*32)
        p = entry["token_position"]
        entry["last_action_confidence_bf16"] = observed.get(p, 0x3f00 if p % 2 else 0x3e80)
        entry["last_action_confidence_valid"] = 1
        expected[0xd7000+i*32:0xd7020+i*32] = record("token_state_entry", entry)
        states[p] = entry
    extension = read_record("uaps_attempt_config", initial, 640)
    extension.update(retry_min_confidence_bf16=0x3e80, prediction_target=24)
    for image in (initial, expected):
        image[640:672] = record("uaps_attempt_config", extension)

    # Base selection/precision is unchanged by the joint options. Use its fixed
    # independent expected records to prepare expected only, never launch data.
    positions = [100+i for i in range(64) if expected[0xe4000+i*16+10] & 1]
    metadata_bits = {}
    cursor = 0xdc000
    while expected[cursor]:
        count = expected[cursor]
        for i in range(count):
            offset = cursor+32+i*16
            metadata_bits[int.from_bytes(expected[offset+4:offset+6], "little")] = expected[offset+10]
        cursor += int.from_bytes(expected[cursor+8:cursor+10], "little")
    bits = [metadata_bits[p] for p in positions]
    forecast = [8 if bits[i] == 8 or (int.from_bytes(expected[0xe0000+p*8:0xe0008+p*8], "little") >> 52) & 1 else 4
                for i, p in enumerate(positions)]
    current_predictions = sum(states[p]["state"] != 2 for p in positions if 100 <= p < 132)
    dependency_raw = [int.from_bytes(expected[0xe4000+(32+i)*16:0xe4002+(32+i)*16], "little") for i in range(32)]
    dependency = torch.tensor(dependency_raw, dtype=torch.int16).view(torch.bfloat16)
    order = torch.argsort(dependency, stable=True)
    ranks = torch.empty_like(dependency)
    ranks[order] = torch.arange(32, dtype=torch.bfloat16)
    priority = 1.0 - (ranks.float()/31).to(torch.bfloat16)
    tentative = torch.tensor([states[132+i]["state"] == 1 for i in range(32)])
    priority[tentative] = 1.0
    priority[[p-132 for p in positions if p >= 132]] = 1.0
    attempts = [int.from_bytes(initial[0xea010+4*i:0xea014+4*i], "little") for i in range(32)]
    scheduler = _algorithm_scheduler(target_joint_tokens=48, max_next_tokens=32)
    result = _algorithm_select(scheduler, dict(
        base_positions=torch.tensor(positions), base_activation_bits=torch.tensor(bits, dtype=torch.int8),
        next_step_base_activation_bits=torch.tensor(forecast, dtype=torch.int8), next_block_start=132,
        next_activation_bits=torch.tensor([states[132+i]["activation_bits"] for i in range(32)], dtype=torch.int8),
        next_unresolved=torch.tensor([states[132+i]["state"] != 2 for i in range(32)]),
        next_tentative=tentative, next_priority=priority,
        future_admission_attempts=torch.tensor(attempts, dtype=torch.int32),
        max_future_admission_attempts=extension["max_attempts"],
        next_last_confidence=torch.tensor([states[132+i]["last_action_confidence_bf16"] for i in range(32)], dtype=torch.int16).view(torch.bfloat16),
        future_admission_retry_min_confidence=.25,
        prediction_token_target=24, current_prediction_tokens=current_predictions))
    progress = sum(1 << i for i in result.progress_local_positions.tolist())
    added = sum(1 << i for i in result.added_local_positions.tolist())
    occupancy = lambda value: value.a4_rows + 2*value.a8_rows
    expected[0xe9020:0xe9030] = record("uaps_result", dict(
        future_prediction_mask=progress,
        added_future_token_mask=added,
        future_prediction_count=result.progress_local_positions.numel(),
        added_future_token_count=result.added_local_positions.numel(),
        base_activation_slots=occupancy(result.base_residency),
        joint_activation_slots=occupancy(result.joint_residency),
        next_pass_activation_slots=occupancy(result.next_step_verification_residency)))
    put(expected, 0xea00c, 4, added)
    for i in range(32):
        put(expected, 0xea010+4*i, 4, attempts[i]+((added>>i)&1))
    tokens = [dict(position=p, bits=bits[i], source=1, source_index=states[p]["token_id"]) for i, p in enumerate(positions)]
    tokens += [dict(position=132+i, bits=states[132+i]["activation_bits"], source=1, source_index=states[132+i]["token_id"])
             for i in result.added_local_positions.tolist()]
    tokens.sort(key=lambda token: token["position"])
    tool = build_reference(output)
    metadata = bytes(json.loads(subprocess.check_output([str(tool), "metadata"],
        input=json.dumps(dict(tokens=tokens, sequence=2048, capture=7)), text=True))["metadata"])
    # No write past the final metadata round; unused bytes retain the input.
    expected[0xdc000:0xdc000+1696] = initial[0xdc000:0xdc000+1696]
    expected[0xdc000:0xdc000+len(metadata)] = metadata
    (output / "initial.bin").write_bytes(initial)
    (output / "expected.bin").write_bytes(expected)
    shutil.copyfile(hardware / "cases/head_state_update_32tokens/memory_map.json", output / "memory_map.json")
    shutil.copyfile(hardware / "config/ddr/lpddr4_3200_2x16_8gib.ini", output / "dram.ini")
    case = dict(schema="supra-testcase/v1", ddr_image="initial.bin", memory_map="memory_map.json", dramsim3_config="dram.ini",
        initial_segments=[], executions=[dict(config_address=base, id=0x50420001, max_cycles=2000000)],
        expected=[dict(name="ddr", address=base, bytes=len(expected), path="expected.bin")])
    (output / "case.json").write_text(json.dumps(case, indent=2)+"\n")
    suppressed_dir = output / "suppressed"
    suppressed_dir.mkdir()
    template = hardware / "cases/head_state_update_32tokens"
    suppressed_input = bytearray((template / "initial.bin").read_bytes())
    suppressed_expected = bytearray((template / "expected.bin").read_bytes())
    execution_config = read_record("execution_config", suppressed_input)
    execution_config["suppressed_token_count"] = 1
    post = read_record("forward_postprocess_config", suppressed_input, 0xd0000)
    post["flags"] = 1
    for image in (suppressed_input, suppressed_expected):
        image[:320] = record("execution_config", execution_config)
        image[0xd0000:0xd00c0] = record("forward_postprocess_config", post)
        put(image, post["suppressed_token_base"]-base, 4, 0)
    proposal, _, confidence = streaming_candidate_bf16(torch.zeros(1, 32, 64, dtype=torch.bfloat16))
    action = candidate_action_confidence(proposal, confidence, [0]).to(torch.bfloat16).view(torch.int16).flatten().tolist()
    if any(action):
        raise ValueError("suppressed zero-logit winner did not produce zero action confidence")
    # All transfer-only scores remain tied, so the fixed quota/state result is
    # unchanged; only the new action history and suppressed input differ.
    for i, raw in enumerate(action):
        put(suppressed_expected, 0xd7000+i*32+28, 2, raw)
        put(suppressed_expected, 0xd7000+i*32+30, 1, 1)
    (suppressed_dir / "initial.bin").write_bytes(suppressed_input)
    (suppressed_dir / "expected.bin").write_bytes(suppressed_expected)
    suppressed_case = dict(schema="supra-testcase/v1", ddr_image="initial.bin", memory_map="../memory_map.json", dramsim3_config="../dram.ini",
        executions=[dict(config_address=base, id=0x50420002, max_cycles=1000000)],
        expected=[dict(name="ddr", address=base, bytes=len(suppressed_expected), path="expected.bin")])
    (suppressed_dir / "case.json").write_text(json.dumps(suppressed_case, indent=2)+"\n")
    return output / "case.json"


if __name__ == "__main__":
    main()
