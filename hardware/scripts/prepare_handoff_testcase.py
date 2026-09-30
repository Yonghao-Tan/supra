#!/usr/bin/env python3
"""Prepare fixed C11 control cases with continuous layer testcases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from artifact_paths import artifact_root
from build_forward_postprocess_config import pack_schema

HARDWARE = Path(__file__).resolve().parents[1]
BASE, CONFIG, TABLE, METADATA = 0x10000000, 0x61000000, 0x61010000, 0x61020000
OUTPUT, TEMP = 0x61200000, 0x61800000
HANDOFF_CASE_DIRECTORIES = {
    "regular": "in_block_handoff",
    "boundary": "cross_block_handoff",
}


def record(name, values):
    schema = json.loads((HARDWARE / "config" / (name + ".json")).read_text())
    initial = {field["name"]: 0 for field in schema["fields"]
               if "constant" not in field and "default" not in field}
    return pack_schema(schema, {**initial, **values}, name)


def read_record(name, image, offset=0):
    schema = json.loads((HARDWARE / "config" / (name + ".json")).read_text())
    return {f["name"]: int.from_bytes(image[offset + f["offset"]:offset + f["offset"] + int(f["type"][1:]) // 8],
                "little", signed=f["type"].startswith("i")) for f in schema["fields"]}


POST_REGIONS = (
    ("post_state", 0x100D7000, 2048), ("post_event", 0x100D8000, 128),
    ("post_completion", 0x100D9000, 16), ("post_metadata", 0x100DB000, 1696),
    ("post_joint_metadata", 0x100DC000, 1696), ("post_pending", 0x100E4000, 1024),
    ("post_budget", 0x100E9000, 16), ("post_joint_result", 0x100E9020, 16),
    ("post_attempts", 0x100EA000, 144),
)


def postprocess_expected_image(initial, expected):
    """Retain scheduling inputs when the current block has completed."""
    post = read_record("forward_postprocess_config", initial, 0xD0000)
    block = read_record("draft_verify_block_config", initial,
                        post["block_configuration_base"] - BASE)
    start = post["next_state_base"] - BASE + block["next_state_entry"] * 32
    # One post completion event reports the current block, including when
    # the same post launch also updates future-block state.
    completions = [all(read_record("token_state_entry", expected, start + i * 32)["state"] == 2
                       for i in range(block["position_count"]))]
    result = bytearray(expected)
    if completions[0]:
        # The generator schedules its next packed forward only while the
        # current block contains an unresolved token.
        for _, address, size in POST_REGIONS[4:]:
            offset = address - BASE
            result[offset:offset + size] = initial[offset:offset + size]
    return bytes(result), completions


def layer_entry(boundary=False, layer=0):
    values = dict(version=2, header_bytes=64, entry_bytes=624, hidden_size=4096, ffn_size=12288,
        head_count=32, head_dimension=128, maximum_sequence_length=2048, kv_head_stride_bytes=262144,
        kv_token_stride_bytes=128, kv_chunk_stride_bytes=8, k_scale_head_stride_bytes=4096,
        k_scale_token_stride_bytes=2, context_row_stride_bytes=8192,
        attention_rms_epsilon_bf16=0x3586, ffn_rms_epsilon_bf16=0x3586)
    schema = json.loads((HARDWARE / "config/layer_address_table.json").read_text())
    for field in schema["fields"]:
        name = field["name"]
        if name.endswith("_weight_base"):
            values[name] = 0x11000000; values[name[:-4] + "limit"] = 0x13000000
        elif (name.endswith("_scale_base") or name.endswith("_gamma_base")) and name[:-4] + "limit" in {f["name"] for f in schema["fields"]}:
            values[name] = 0x13000000; values[name[:-4] + "limit"] = 0x13010000
    regions = dict(rope_cos_lut=(0x13100000, 0x80000), rope_sin_lut=(0x13200000, 0x80000),
        current_k_cache=(0x14000000, 0x800000), current_v_cache=(0x14800000, 0x800000),
        retained_k_cache=(0x15000000, 0x800000), retained_v_cache=(0x15800000, 0x800000),
        k_scale=(0x10160000, 0x20000), v_scale=(0x10180000, 0x20000),
        context=(0x16100000, 0x80000), ffn_gate_up_workspace=(0x16200000, 0x240000),
        ffn_residual_workspace=(0x16480000, 0x60000))
    if boundary:
        values["retained_k_scale_base"] = 0x68080000 + layer * 0x20000
        regions.update(current_k_cache=(0x64000000 + layer * 0x800000, 0x800000),
            current_v_cache=(0x65000000 + layer * 0x800000, 0x800000),
            retained_k_cache=(0x66000000 + layer * 0x800000, 0x800000),
            retained_v_cache=(0x67000000 + layer * 0x800000, 0x800000),
            k_scale=(0x68000000 + layer * 0x20000, 0x20000),
            v_scale=(0x68040000 + layer * 0x20000, 0x20000),
            ffn_gate_up_workspace=(TEMP + 0x400000, 0x240000),
            ffn_residual_workspace=(TEMP + 0x700000, 0x60000))
    for name, (base, size) in regions.items():
        values[name + "_base"] = base; values[name + "_limit"] = base + size
    return record("layer_address_table", values)


def make_execution_config(tokens, sequence, metadata_address, metadata_bytes,
                          start=0, boundary=False, output=OUTPUT):
    values = dict(flags=3, start_layer=start, layer_count=1, total_token_count=tokens, sequence_length=sequence,
        token_metadata_format=3, token_metadata_entry_bytes=16, token_metadata_batch_header_bytes=32,
        embedding_row_bytes=8192, deployment_artifact_id=1, layer_weight_entry_stride=880,
        forward_postprocess_configuration_bytes=192, prediction_entry_bytes=32,
        forward_postprocess_configuration_format=1, prediction_table_format=2,
        layer_weight_table_base=TABLE, layer_weight_table_limit=TABLE + (start + 1) * 880,
        token_metadata_base=metadata_address, token_metadata_limit=metadata_address + metadata_bytes,
        output_hidden_base=output, output_hidden_limit=output + tokens * 8192,
        embedding_base=0x17000000, embedding_limit=0x17000000 + 126464 * 8192)
    regions = dict(bf16_temporary=(0x16200000, 0x300000), current_k_cache=(0x14000000, 0x800000),
        current_v_cache=(0x14800000, 0x800000), retained_k_cache=(0x15000000, 0x800000),
        retained_v_cache=(0x15800000, 0x800000), k_scale=(0x10160000, 0x20000),
        v_scale=(0x10180000, 0x20000), attention_workspace=(0x16100000, 0x80000))
    if boundary:
        regions.update(bf16_temporary=(TEMP, 0x800000), current_k_cache=(0x64000000, 0x1000000),
            current_v_cache=(0x65000000, 0x1000000), retained_k_cache=(0x66000000, 0x1000000),
            retained_v_cache=(0x67000000, 0x1000000), k_scale=(0x68000000, 0x40000),
            v_scale=(0x68040000, 0x40000), retained_k_scale=(0x68080000, 0x40000))
    for name, (base, size) in regions.items():
        values[name + "_base"] = base; values[name + "_limit"] = base + size
    return values


def build_reference(output):
    build = output / "build"; build.mkdir()
    for command in (["test", "-d", str(build)], ["test", "-w", str(build)], ["df", "-h", str(build)], ["df", "-i", str(build)]):
        subprocess.run(command, check=True, stdout=sys.stderr)
    objects = []
    with (output / "reference_build.log").open("w", buffering=1) as log:
        for name in ("rtl_numeric", "forward_postprocess_model", "token_refresh_model"):
            obj = build / (name + ".o"); objects.append(str(obj))
            command = ["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-DSUPRA_SILU_MONOTONE", "-Icmodel", "-c", "cmodel/" + name + ".c", "-o", str(obj)]
            log.write("COMMAND " + repr(command) + "\n")
            subprocess.run(command, cwd=HARDWARE, check=True, stdout=log, stderr=subprocess.STDOUT)
        tool = build / "prepare_reference"
        command = ["g++", "-std=c++17", "-O2", "-I.", "-Ithird_party/dramsim3/ext/headers",
            "cmodel/prepare_handoff_reference.cpp", *objects, "-lm", "-o", str(tool)]
        log.write("COMMAND " + repr(command) + "\n")
        subprocess.run(command, cwd=HARDWARE, check=True, stdout=log, stderr=subprocess.STDOUT)
    return tool


def build_testcase_manifest(executions, segments, expected):
    return dict(
        schema="supra-testcase/v1",
        ddr_image="initial.bin",
        memory_map="memory_map.json",
        dramsim3_config="dram.ini",
        initial_segments=segments,
        executions=executions,
        expected=expected,
    )


def prepare_fixed_k_completion(output):
    """Three real head/post launches; C11 state determines block completion."""
    output = Path(output).resolve()
    if artifact_root() not in output.parents:
        raise ValueError("completion fixture must be below the external hardware artifact root")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tool = build_reference(output)
    seed = HARDWARE / "cases/head_state_update_32tokens"
    cases = []
    for name, quota, mask in (("incomplete_k1", 1, 63), ("complete_k32", 32, 63),
                              ("mask_proposal_k32", 32, 0)):
        directory = output / name
        directory.mkdir()
        image = bytearray((seed / "initial.bin").read_bytes())
        execution = read_record("execution_config", image)
        post = read_record("forward_postprocess_config", image, 0xD0000)
        block = read_record("draft_verify_block_config", image, 0xDA000)
        post["mask_token_id"] = mask
        block.update(flags=32, scheduled_quota=quota)
        image[0xD0000:0xD0000+192] = record("forward_postprocess_config", post)
        image[0xDA000:0xDA000+32] = record("draft_verify_block_config", block)
        states, predictions = [], []
        for i in range(32):
            state = read_record("token_state_entry", image, 0xD6000+i*32)
            state.update(state=0, origin=0, token_id=mask, activation_bits=8,
                         cache_valid=1, refresh_required=0)
            prediction = read_record("prediction_record", image, 0xD1000+i*32)
            prediction.update(current_token_id=mask, tentative_token_id=mask, forward_start_state=0)
            image[0xD6000+i*32:0xD6020+i*32] = record("token_state_entry", state)
            image[0xD1000+i*32:0xD1020+i*32] = record("prediction_record", prediction)
            states.append(state)
            predictions.append(prediction)
        weight_offset = post["raw_w8_base"]-BASE
        scale_offset = post["weight_scale_base"]-BASE
        packed = image[weight_offset:post["raw_w8_limit"]-BASE]
        scales = image[scale_offset:post["weight_scale_limit"]-BASE]
        inputs = dict(states=states, predictions=predictions, vocabulary=post["vocabulary_size"],
            hidden=[0]*(32*4096), packed_weights=[v if v < 128 else v-256 for v in packed],
            weight_scales=[int.from_bytes(scales[i:i+2], "little") for i in range(0,len(scales),2)],
            epsilon=post["final_rms_epsilon_bf16"], mask_token_id=mask, quota=quota,
            remaining_forwards=block["remaining_forwards"], maturity_age=block["maturity_age"],
            sequence=execution["sequence_length"], capture=block["capture_index"],
            global_block_id=block["global_block_id"])
        reference = json.loads(subprocess.check_output([str(tool), "transfer"],
            input=json.dumps(inputs), text=True))
        expected_states = bytes(reference["states"])
        complete = all(read_record("token_state_entry", expected_states, i*32)["state"] == 2
                       for i in range(32))
        # Zero hidden gives zero logits. The C11 candidate function must choose
        # token zero; in the last case it is the MASK and cannot become LOCKED.
        if any(candidate[0] != 0 for candidate in reference["candidates"]):
            raise ValueError("zero-hidden fixture did not produce token-zero proposals")
        if complete != (name == "complete_k32"):
            raise ValueError("C11 fixture does not exercise the intended completion condition")
        (directory / "initial.bin").write_bytes(image)
        (directory / "hidden.bin").write_bytes(bytes(32*8192))
        expected = []
        for field, address, data in (("state", post["next_state_base"], expected_states),
                                     ("event", post["forward_event_base"], bytes(reference["event"]))):
            filename = "expected."+field+".bin"
            (directory / filename).write_bytes(data)
            expected.append(dict(name=field, address=address, bytes=len(data), path=filename))
        shutil.copyfile(seed / "memory_map.json", directory / "memory_map.json")
        shutil.copyfile(HARDWARE / "config/ddr/lpddr4_3200_2x16_8gib.ini", directory / "dram.ini")
        manifest = build_testcase_manifest(
            [dict(config_address=BASE, id=0x50420001, max_cycles=2000000,
                  expected_post_block_completions=[complete])],
            [dict(address=execution["output_hidden_base"], bytes=32*8192, path="hidden.bin")], expected)
        case = directory / "case.json"
        case.write_text(json.dumps(manifest, indent=2)+"\n")
        cases.append(case)
    return cases


def prepare(kind, output):
    root = artifact_root()
    output = output.resolve()
    if root not in output.parents:
        raise ValueError("handoff output must be below the external hardware artifact root")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tool = build_reference(output)
    reference = lambda operation, value: json.loads(subprocess.check_output([str(tool), operation], input=json.dumps(value), text=True))
    seed = HARDWARE / "cases" / HANDOFF_CASE_DIRECTORIES[kind]
    initial = (seed / "initial.bin").read_bytes()
    post_expected, completions = postprocess_expected_image(
        initial, (seed / "post_expected.bin").read_bytes())
    shutil.copyfile(seed / "initial.bin", output / "initial.bin")
    segments, expected, executions = [], [], []
    def segment(name, address, data, offsets=None):
        path = name + ".bin"; (output / path).write_bytes(data)
        for offset in offsets if offsets is not None else [0]:
            segments.append(dict(address=address + offset, bytes=len(data), path=path))
    def compare(name, address, data, offsets=None):
        path = "expected." + name + ".bin"; (output / path).write_bytes(data)
        for i, offset in enumerate(offsets if offsets is not None else [0]):
            expected.append(dict(name=name if offsets is None else name + "_" + str(i),
                                 address=address + offset, bytes=len(data), path=path))
    seed_config = read_record("execution_config", initial)
    sequence, tokens = seed_config["sequence_length"], seed_config["prediction_count"]
    post_execution = dict(config_address=BASE, id=0x50420001, max_cycles=2000000,
                          expected_post_block_completions=completions)
    inputs = []
    for token in range(tokens):
        prediction = read_record("prediction_record", initial, 0xD1000 + token * 32)
        inputs.append(dict(position=prediction["token_position"], source_index=prediction["final_hidden_ddr_row_index"],
                           source=0, bits=prediction["activation_bits"]))
    metadata = bytes(reference("metadata", dict(tokens=inputs, sequence=sequence, capture=6))["metadata"])
    segment("initial_metadata", METADATA, metadata)
    first = make_execution_config(tokens, sequence, METADATA, len(metadata), start=31,
                                  output=seed_config["output_hidden_base"])
    segment("initial_layer31_config", CONFIG, record("execution_config", first))
    segment("initial_layer31_table", TABLE + 31 * 880, layer_entry())
    segment("initial_v_scale", 0x10180000, b"\x80\x3f" * 32)
    executions.append(dict(config_address=CONFIG, id=0x50421000, max_cycles=50000000))
    executions.append(post_execution)
    for name, address, size in POST_REGIONS:
        compare(name, address, post_expected[address - BASE:address - BASE + size])
    next_config = CONFIG + 0x1000
    if kind == "regular":
        next_config_values = make_execution_config(1, sequence, 0x100DC000, 1696)
        next_config_values["flags"] = 1
        segment("regular_layer0_config", next_config,
                record("execution_config", next_config_values))
        segment("regular_layer0_table", TABLE, layer_entry())
        executions.append(dict(config_address=next_config, id=0x50421001, max_cycles=50000000, check_embedding_reads=True,
            handoff_actions=[dict(kind="execution_from_metadata", config_address=next_config,
                metadata_address=0x100DC000, metadata_capacity=1696, output_address=OUTPUT)]))
        compare("next_hidden", OUTPUT, bytes(96 * 8192))
    else:
        sequence, current, capture, target, a8_limit = 164, 132, 7, 80, 48
        scout_metadata, deep_metadata, row_table = METADATA + 0x20000, METADATA + 0x30000, METADATA + 0x40000
        probability, jobs = 0x60000000, 0x60400000
        states = {int.from_bytes(post_expected[0xD7000 + i * 32:0xD7002 + i * 32], "little"):
                  post_expected[0xD7000 + i * 32:0xD7020 + i * 32] for i in range(64)}
        full_tokens, changed, mandatory = [], [], []
        for position in range(sequence):
            bits, token = 8, position % 64
            if position >= current - 32:
                state = states[position]; token = int.from_bytes(state[8:12], "little")
                if position >= current: bits = state[7]
                elif state[25] & 1: changed.append(position)
                if position < current and state[19]: mandatory.append(position)
            full_tokens.append(dict(position=position, source_index=token, source=1, bits=bits))
        boundary = reference("boundary", dict(tokens=full_tokens, sequence=sequence, capture=0,
            current=list(range(current, sequence)), changed=changed, mandatory=mandatory, target=target, a8_limit=a8_limit))
        next_config_values = make_execution_config(
            sequence, sequence, scout_metadata, 4096, boundary=True)
        next_config_values["refresh_configuration_offset"] = 320
        next_config_values["layer_weight_table_limit"] = TABLE + 1760
        segment("scout_layer0_config", next_config,
                record("execution_config", next_config_values))
        segment("scout_layer0_table", TABLE, layer_entry(True))
        segment("deep_layer1_table", TABLE + 880, layer_entry(True, 1))
        refresh = dict(flags=3 | 0x20 | (a8_limit << 8), sequence_length=sequence, target_token_count=target,
            metadata_version=1, table_base=row_table, table_limit=row_table + ((sequence + 7) // 8) * 64,
            metadata_base=deep_metadata, metadata_limit=deep_metadata + 4096,
            head_stride=262144, scale_head_stride=4096, probability_configuration_offset=176,
            relation_job_base=jobs)
        for field, source, destination, size in (("k", 0x64000000, 0x66000000, 0x800000),
                ("v", 0x65000000, 0x67000000, 0x800000), ("scale", 0x68000000, 0x68080000, 0x20000)):
            refresh.update({"source_" + field + "_base": source, "source_" + field + "_limit": source + size,
                "destination_" + field + "_base": destination, "destination_" + field + "_limit": destination + size})
        segment("scout_refresh", next_config + 320, record("token_refresh_config", refresh))
        groups = (sequence + 7) // 8
        prob = dict(output_base=probability, output_limit=jobs, batch_stride_bytes=groups * 128,
            head_stride_bytes=groups * 128 * 6, round_stride_bytes=groups * 128 * 6 * 32,
            layer_mask=1, query_end=sequence, key_groups0=(1 << groups) - 1)
        segment("scout_probability", next_config + 496, record("attention_probability_config", prob))
        executions.append(dict(config_address=next_config, id=0x50421002, max_cycles=70000000, check_embedding_reads=True,
            handoff_actions=[dict(kind="state_to_cross_block_inputs", config_address=next_config,
                metadata_address=scout_metadata, metadata_capacity=4096, state_address=0x100D7000,
                sequence=sequence, current_begin=current, capture=capture, table_address=row_table,
                probability_address=probability, jobs_address=jobs, jobs_capacity=2048,
                refresh_address=next_config + 320, output_address=OUTPUT)]))
        deep_config = CONFIG + 0x2000
        deep_config_values = dict(
            next_config_values,
            start_layer=1,
            total_token_count=target,
            token_metadata_base=deep_metadata,
            token_metadata_limit=deep_metadata + 4096,
            output_hidden_limit=OUTPUT + sequence * 2 * 8192,
        )
        segment("deep_layer1_config", deep_config,
                record("execution_config", deep_config_values))
        refresh.update(flags=1, probability_configuration_offset=0, relation_job_count=0, relation_job_base=0)
        for field in tuple(refresh):
            if field.startswith(("source_", "destination_")):
                refresh[field] += 0x20000 if "scale" in field else 0x800000
        segment("deep_refresh", deep_config + 320, record("token_refresh_config", refresh))
        executions.append(dict(config_address=deep_config, id=0x50421003, max_cycles=50000000,
            copies=[dict(source_address=OUTPUT, destination_address=OUTPUT + sequence * 8192, bytes=sequence * 8192),
                    dict(source_address=deep_metadata, destination_address=0x61080000, bytes=4096),
                    dict(source_address=row_table, destination_address=0x61090000, bytes=((sequence + 7) // 8) * 64)],
            handoff_actions=[dict(kind="execution_from_metadata", config_address=deep_config, metadata_address=deep_metadata,
                metadata_capacity=4096, output_address=OUTPUT, output_capacity_tokens=sequence * 2,
                source_index_offset=sequence, commit_table_address=row_table, sequence=sequence)]))
        compare("scout_selected_metadata", 0x61080000, bytes(boundary["metadata"]))
        scout_table = bytearray(((sequence + 7) // 8) * 64)
        for position, token in enumerate(full_tokens):
            value = boundary["scores"][position] | ((position >= current) << 8) | ((position < current) << 9) | \
                ((position in mandatory) << 10) | (position << 12) | ((token["bits"] == 8) << 34) | (position << 35)
            scout_table[position * 8:position * 8 + 8] = value.to_bytes(8, "little")
        compare("scout_scores_and_inputs", 0x61090000, scout_table)
        for layer in range(2):
            segment(f"boundary_v_scale_{layer}", 0x68040000 + layer * 0x20000, b"\x80\x3f" * 32)
            selected = set(boundary["positions"])
            new_scale = boundary["zero_key_scale_bf16"]
            for base_name, address in (("key", 0x68000000), ("retained", 0x68080000)):
                segment(f"boundary_{base_name}_scale_{layer}", address + layer * 0x20000,
                        (b"\x00\x40" if base_name == "retained" else b"\x80\x3f") * sequence,
                        offsets=[head * 4096 for head in range(32)])
            scale_expected = b"".join((new_scale if position in selected else 0x4000).to_bytes(2, "little")
                                      for position in range(sequence))
            compare(f"retained_key_scale_{layer}", 0x68080000 + layer * 0x20000, scale_expected,
                    offsets=[head * 4096 for head in range(32)])
            for kind_name, address in (("key", 0x66000000), ("value", 0x67000000)):
                width = 128 if kind_name == "key" else 8
                before = bytes([17]) * (sequence * width)
                after = b"".join(bytes([0 if position in selected else 17]) * width for position in range(sequence))
                offsets = [head * 262144 + chunk * 16384 for head in range(32)
                           for chunk in range(1 if kind_name == "key" else 16)]
                segment(f"initial_{kind_name}_cache_{layer}", address + layer * 0x800000, before, offsets)
                compare(f"{kind_name}_cache_{layer}", address + layer * 0x800000, after, offsets)
        compare("scout_and_deep_hidden", OUTPUT, bytes(sequence * 2 * 8192))
        case_reference = dict(deep_positions=boundary["positions"], deep_bits=boundary["bits"],
                              downgraded_a8=boundary["downgraded_a8"], zero_key_scale_bf16=boundary["zero_key_scale_bf16"],
                              source="Independent C11 boundary reference")
        (output / "reference.json").write_text(json.dumps(case_reference, indent=2) + "\n")
    shutil.copyfile(HARDWARE / "cases/head_state_update_32tokens/memory_map.json", output / "memory_map.json")
    shutil.copyfile(HARDWARE / "config/ddr/lpddr4_3200_2x16_8gib.ini", output / "dram.ini")
    case = build_testcase_manifest(executions, segments, expected)
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return output / "case.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("regular", "boundary", "fixed-k-completion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "fixed-k-completion":
        for case in prepare_fixed_k_completion(args.output):
            print(case)
    else:
        print(prepare(args.kind, args.output))


if __name__ == "__main__":
    main()
