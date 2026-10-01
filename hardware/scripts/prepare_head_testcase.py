#!/usr/bin/env python3
"""Prepare captured or CPU reference final-norm/W8-head tensors for simulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

from artifact_paths import artifact_root, reference_source_path
from build_forward_postprocess_config import patch_record
from pack_lm_head_raw_w8 import pack_weights, validate_scales, verify_panels

HARDWARE = Path(__file__).resolve().parents[1]
BASE = 0x10000000
HIDDEN = BASE + 0x10000
POST = BASE + 0xD0000
WEIGHTS = BASE + 0x100000
EXPECTED = {"norm_output": ("<u2", (32, 4096)),
            "activation_codes": ("|i1", (32, 4096)),
            "activation_scale": ("<u2", (32,)),
            "accumulator": ("<i4", None), "output": ("<u2", None)}


def expected_shapes(tokens):
    return {name: (dtype, (tokens, *shape[1:]) if shape else None)
            for name, (dtype, shape) in EXPECTED.items()}


def load_schema(name):
    return json.loads((HARDWARE / "config" / (name + ".json")).read_text())


def load_reference_data(index: Path, tensor_data_root: Path | None = None):
    reference_data = json.loads(index.read_text())
    tokens = reference_data.get("tokens")
    if (reference_data.get("reference_data_kind") not in ("real_cuda_final_output", "cpu_final_output") or
            type(tokens) is not int or not 1 <= tokens <= 96 or
            reference_data.get("hidden") != 4096 or reference_data.get("weight_bits") != 8 or reference_data.get("activation_bits") != 8):
        raise ValueError("head tensors require 1..96 tokens, hidden4096 and W8A8")
    expected_tensors = expected_shapes(tokens)
    vocab = reference_data["vocab"]
    if type(vocab) is not int or not 64 <= vocab <= 126464 or vocab % 8:
        raise ValueError("head vocabulary must be a multiple of eight in [64,126464]")
    entries = {}
    for entry in reference_data["tensors"]:
        # Control snapshots have their own payload root and are consumed by the
        # observed-state adapter, not by the numerical head packer.
        if entry["role"] == "control":
            continue
        key = (entry["role"], entry["name"])
        if key in entries:
            raise ValueError(f"duplicate reference_data tensor: {key}")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = (tensor_data_root or index.parent) / path
        path = path.resolve(strict=True)
        dtype = np.dtype(entry["dtype"])
        count = int(np.prod(entry["shape"], dtype=np.int64))
        if path.stat().st_size != entry["byte_count"] or count * dtype.itemsize != entry["byte_count"]:
            raise ValueError(f"reference_data tensor size differs: {key}")
        entries[key] = {**entry, "path": path}
    required = {("input", name): (dtype, count) for name, dtype, count in (
        ("hidden", "<u2", tokens * 4096), ("norm_weight", "<u2", 4096),
        ("weight_codes", "|i1", vocab * 4096), ("weight_scale", "<u2", vocab),
        ("epsilon", "<u4", 1), ("positions", "<i8", tokens), ("prediction_mask", "|b1", tokens))}
    required.update({("expected", name): (dtype, int(np.prod(shape or (tokens, vocab))))
                     for name, (dtype, shape) in expected_tensors.items()})
    for key, (dtype, count) in required.items():
        entry = entries.get(key)
        if entry is None or np.dtype(entry["dtype"]) != np.dtype(dtype) or entry["byte_count"] != count * np.dtype(dtype).itemsize:
            raise ValueError(f"missing or malformed reference_data tensor: {key}")
        encoding = {"<u2": "bf16_raw", "<u4": "float32_raw"}.get(dtype, "integer")
        if entry.get("encoding") != encoding:
            raise ValueError(f"reference_data tensor encoding differs: {key}: expected {encoding}")
    shapes = {("input", "hidden"): (tokens, 4096), ("input", "weight_codes"): (vocab, 4096),
              ("input", "weight_scale"): (vocab,), ("input", "norm_weight"): (4096,),
              ("input", "positions"): (tokens,), ("input", "prediction_mask"): (tokens,),
              ("input", "epsilon"): ()}
    shapes.update({("expected", name): shape or (tokens, vocab) for name, (_, shape) in expected_tensors.items()})
    for key, shape in shapes.items():
        actual = tuple(entries[key]["shape"])
        if actual != shape and actual != (1, *shape):
            raise ValueError(f"reference_data tensor shape differs: {key}")
    gamma = np.fromfile(entries["input", "norm_weight"]["path"], dtype="<u2")
    if not np.all(gamma == 0x3F80):
        raise ValueError("final norm requires folded unit BF16 gamma")
    if not np.all(np.fromfile(entries["input", "prediction_mask"]["path"], dtype=np.bool_)):
        raise ValueError("this head testcase requires every captured token to be a prediction")
    epsilon = int.from_bytes(entries["input", "epsilon"]["path"].read_bytes(), "little")
    if epsilon & 0x80000000 or epsilon & 0x7F800000 == 0x7F800000 or epsilon == 0:
        raise ValueError("RMS epsilon must be positive and finite")
    epsilon_bf16 = (epsilon + 0x7FFF + ((epsilon >> 16) & 1)) >> 16
    if epsilon_bf16 == 0 or epsilon_bf16 & 0x7F80 == 0x7F80:
        raise ValueError("RMS epsilon must remain positive and finite after BF16 rounding")
    return reference_data, entries, epsilon_bf16


def prepare(index: Path, output: Path, tensor_data_root: Path | None = None,
            vocabulary_prefix: int | None = None, ddr: int = 3200, *,
            preceding_layer_testcase: Path | None = None, layer_output_tokens=None,
            feature2_reference: Path | None = None, base_address: int = BASE):
    index = index.resolve(strict=True)
    output = output.resolve()
    root = artifact_root()
    if root not in output.parents:
        raise ValueError("head testcase output must be a new directory below the hardware artifact root")
    if output.exists():
        raise FileExistsError(output)
    reference_data, entries, epsilon = load_reference_data(index, tensor_data_root)
    if feature2_reference is not None:
        validate_observed_head_reference(reference_data, json.loads(feature2_reference.read_text()), feature2_reference,
                                         head_index=index, payload_root=tensor_data_root)
    tokens = reference_data["tokens"]
    vocab = reference_data["vocab"] if vocabulary_prefix is None else vocabulary_prefix
    if type(vocab) is not int or not 64 <= vocab <= reference_data["vocab"] or vocab % 8:
        raise ValueError("vocabulary prefix must contain whole panels and at least 64 channels")
    scale_base = WEIGHTS + ((vocab * 4096 + 4095) & ~4095)
    image_size = max(0x200000, (scale_base + vocab * 2 + 4095 - BASE) & ~4095)
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < image_size * 3 + 1024**3:
        raise ValueError("insufficient space for head testcase packing")
    template = HARDWARE / "cases/head_state_update_32tokens"
    image = bytearray((template / "initial.bin").read_bytes())
    execution_schema = load_schema("execution_config")
    metadata_schema = json.loads((HARDWARE / "config/token_metadata.json").read_text())
    header_schema = metadata_schema["token_batch_header"]
    entry_schema = metadata_schema["token_entry"]
    header_template = bytes(image[0xD3000:0xD3020])
    entry_template = bytes(image[0xD3020:0xD3030])
    metadata_bytes = 0
    # These head-only source records are all A8: each row occupies two
    # activation slots, so a 64-slot round can contain at most 32 rows.
    for first in range(0, tokens, 32):
        count = min(32, tokens - first)
        inverse_offset = 32 + count * 16
        inverse_bytes = (count + 15) & ~15
        batch_bytes = inverse_offset + inverse_bytes
        offset = 0xD3000 + metadata_bytes
        image[offset:offset + batch_bytes] = bytes(batch_bytes)
        image[offset:offset + 32] = header_template
        patch_record(image, offset, header_schema,
            {"resident_token_count": count, "compute_group_count": (count + 7) // 8,
             "token_batch_index": first // 32, "first_token_ordinal": first,
             "token_batch_bytes": batch_bytes, "inverse_offset": inverse_offset,
             "inverse_bytes": inverse_bytes})
        image[offset+inverse_offset:offset+inverse_offset+count] = bytes(range(count))
        for local in range(count):
            entry = offset + 32 + local * 16
            image[entry:entry+16] = entry_template
            patch_record(image, entry, entry_schema,
                {"source_index": first + local, "token_position": first + local,
                 "kv_index": first + local, "token_ordinal": local,
                 "compute_group": local // 8, "pe_slot": local % 8})
        metadata_bytes += batch_bytes
    patch_record(image, 0, execution_schema, {"output_hidden_base": HIDDEN,
        "output_hidden_limit": HIDDEN + tokens * 8192, "total_token_count": tokens,
        "token_metadata_limit": BASE + 0xD3000 + metadata_bytes,
        "next_token_metadata_limit": BASE + 0xDC000,
        "prediction_count": tokens, "prediction_table_limit": POST + 0x1000 + tokens * 32})
    patch_record(image, POST - BASE, load_schema("forward_postprocess_config"),
        {"vocabulary_size": vocab, "raw_w8_base": WEIGHTS, "raw_w8_limit": WEIGHTS + vocab * 4096,
         "weight_scale_base": scale_base, "weight_scale_limit": scale_base + vocab * 2,
         "final_rms_epsilon_bf16": epsilon})
    blocks = (tokens + 31) // 32
    patch_record(image, 0, execution_schema,
        {"generation_block_count": blocks, "sequence_length": max(64, tokens)})
    patch_record(image, POST - BASE, load_schema("forward_postprocess_config"),
        {"current_state_limit": BASE + 0xD6000 + tokens * 32,
         "next_state_limit": BASE + 0xD7000 + tokens * 32,
         "forward_event_limit": BASE + 0xD8000 + blocks * 64,
         "block_configuration_limit": BASE + 0xDA000 + blocks * 32})
    block_template = bytes(image[0xDA000:0xDA020])
    prediction_template = bytes(image[0xD1000:0xD1020])
    state_template = bytes(image[0xD6000:0xD6020])
    for block in range(blocks):
        offset = 0xDA000 + block * 32
        image[offset:offset+32] = block_template
        patch_record(image, offset, load_schema("draft_verify_block_config"),
            {"block_slot": block, "global_block_id": 3 + block,
             "position_count": min(32, tokens - block * 32), "scheduled_quota": min(2, tokens - block * 32),
             "current_state_entry": block * 32, "next_state_entry": block * 32})
    prediction = load_schema("prediction_record")
    for token in range(tokens):
        offset = 0xD1000 + token * 32
        image[offset:offset+32] = prediction_template
        patch_record(image, offset, prediction,
            {"final_hidden_ddr_row_index": token, "prediction_index": token,
             "token_position": token, "block_slot": token // 32, "global_block_id": 3 + token // 32,
             "block_local_position": token % 32, "source_row_index": token % 32,
             "source_token_batch_index": token // 32,
             "current_state_entry": token, "next_state_entry": token})
        offset = 0xD6000 + token * 32
        image[offset:offset+32] = state_template
        patch_record(image, offset, load_schema("token_state_entry"),
            {"token_position": token, "block_local_position": token % 32,
             "block_slot": token // 32, "global_block_id": 3 + token // 32})
    hidden = entries["input", "hidden"]["path"].read_bytes()
    image[HIDDEN - BASE:HIDDEN - BASE + len(hidden)] = hidden
    output.mkdir()
    initial = output / "initial.bin"
    with initial.open("wb") as stream:
        stream.write(image)
        stream.truncate(image_size)
    weight_source = entries["input", "weight_codes"]["path"]
    if vocab != reference_data["vocab"]:
        source = np.memmap(weight_source, dtype=np.int8, mode="r", shape=(reference_data["vocab"], 4096))
        weight_source = output / "weight_prefix.raw.bin"
        np.asarray(source[:vocab]).tofile(weight_source)
        del source
    packed = output / "packed_weights.bin"
    pack_weights(weight_source, packed, vocab, 4096)
    verify_panels(weight_source, packed, vocab, 4096)
    scales = np.memmap(entries["input", "weight_scale"]["path"], dtype="<u2", mode="r", shape=(reference_data["vocab"],))
    scale_path = output / "weight_scales.bin"
    np.asarray(scales[:vocab]).tofile(scale_path)
    del scales
    validate_scales(scale_path, vocab)
    with initial.open("r+b") as target:
        target.seek(WEIGHTS - BASE)
        with packed.open("rb") as source:
            shutil.copyfileobj(source, target)
        target.seek(scale_base - BASE)
        target.write(scale_path.read_bytes())
    packed.unlink(); scale_path.unlink()
    if weight_source.parent == output:
        weight_source.unlink()
    expected = []
    for name, (dtype, shape) in expected_shapes(tokens).items():
        entry = entries["expected", name]
        path = output / ("expected." + name + ".bin")
        if name in ("accumulator", "output") and vocab != reference_data["vocab"]:
            array = np.memmap(entry["path"], dtype=dtype, mode="r", shape=(tokens, reference_data["vocab"]))
            np.ascontiguousarray(array[:, :vocab]).tofile(path)
            del array
        else:
            shutil.copyfile(entry["path"], path)
        expected.append(dict(name=name, dtype=dtype, shape=list(shape or (tokens, vocab)),
                             bytes=path.stat().st_size, path=path.name))
    shutil.copyfile(entries["input", "hidden"]["path"], output / "expected.hidden_input.bin")
    regions = json.loads((template / "memory_map.json").read_text())
    regions["memory_map"] = [r for r in regions["memory_map"] if r["limit"] <= WEIGHTS]
    for region in regions["memory_map"]:
        if region["name"] in ("token_metadata", "next_token_metadata", "second_next_token_metadata"):
            region.update(limit=region["base"]+4096, span_bytes=4096)
    regions["memory_map"].append(dict(name="raw_w8_weight_and_scale", base=WEIGHTS,
        limit=scale_base + vocab * 2, span_bytes=scale_base + vocab * 2 - WEIGHTS,
        access="read_only", alignment_bytes=32, lifetime="model", source="Prepared real W8 head reference_data"))
    regions.update(current_ddr_image_bytes=image_size, reference_data=f"{tokens}-token W8 head, vocabulary {vocab}")
    (output / "memory_map.json").write_text(json.dumps(regions, indent=2) + "\n")
    shutil.copyfile(HARDWARE / f"config/ddr/lpddr4_{ddr}_2x16_8gib.ini", output / "dram.ini")
    case = dict(schema="supra-testcase/v1", ddr_image="initial.bin", memory_map="memory_map.json",
        dramsim3_config="dram.ini", executions=[dict(config_address=BASE, id=0x50420001,
        max_cycles=max(1000000, vocab * 2048))],
        expected=[dict(name="hidden_input", address=HIDDEN, bytes=len(hidden), path="expected.hidden_input.bin")],
        head_checkpoints=dict(tokens=tokens, vocab=vocab, expected=expected),
        provenance=dict(reference_data_index=str(index), reference_data_kind=reference_data["reference_data_kind"],
            source_vocab=reference_data["vocab"], vocabulary_prefix=vocab))
    if preceding_layer_testcase is None and layer_output_tokens is not None:
        raise ValueError("layer output token indices require a preceding layer testcase")
    if feature2_reference is not None:
        if vocab != reference_data["vocab"]:
            raise ValueError("observed Feature2 requires the full captured vocabulary")
        positions = np.fromfile(entries["input", "positions"]["path"], dtype="<i8").tolist()
        apply_observed_feature2(case, output, feature2_reference, positions)
    if base_address != BASE:
        relocate_head(case, output, base_address)
    if preceding_layer_testcase is not None:
        connect_preceding_layer(case, output, preceding_layer_testcase, layer_output_tokens)
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return output / "case.json"


def relocate_head(case, output, base_address):
    """Relocate head storage and descriptors, preserving linked layer addresses."""
    if type(base_address) is not int or base_address < BASE or base_address % 256:
        raise ValueError("head base must be a 256-byte-aligned DDR address")
    image_path = output / case["ddr_image"]
    size, delta = image_path.stat().st_size, base_address - BASE
    limit = BASE + size
    if base_address + size > BASE + 8 * 1024**3:
        raise ValueError("head storage exceeds the DDR aperture")
    mapping_path = output / case["memory_map"]
    mapping = json.loads(mapping_path.read_text())
    relocated = []
    for region in mapping["memory_map"]:
        region = dict(region)
        if BASE <= region["base"] < limit and region["limit"] <= limit:
            region.update(base=region["base"] + delta, limit=region["limit"] + delta)
        if any(region["base"] < other["limit"] and other["base"] < region["limit"] for other in relocated):
            raise ValueError("relocated head overlaps another DDR region")
        relocated.append(region)
    with image_path.open("r+b") as stream:
        prefix = bytearray(stream.read(0x100000))
        for name, offset in (("execution_config", 0), ("forward_postprocess_config", POST - BASE)):
            schema = load_schema(name)
            fields = {field["name"]: field for field in schema["fields"]}
            for key, field in fields.items():
                if not key.endswith("_base"):
                    continue
                end_key = key[:-4] + "limit"
                if end_key not in fields:
                    continue
                start = offset + field["offset"]
                value = int.from_bytes(prefix[start:start+8], "little")
                end_offset = offset + fields[end_key]["offset"]
                end = int.from_bytes(prefix[end_offset:end_offset+8], "little")
                if BASE <= value < limit and value <= end <= limit:
                    patch_record(prefix, offset, schema, {key: value + delta, end_key: end + delta})
        stream.seek(0); stream.write(prefix)
    for execution in case["executions"]:
        if BASE <= execution["config_address"] < limit:
            execution["config_address"] += delta
    for entry in case["expected"]:
        if BASE <= entry["address"] < limit:
            entry["address"] += delta
    mapping.update(base_address=base_address, limit_address=BASE + 8 * 1024**3, memory_map=relocated)
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")


def validate_observed_head_reference(head, reference, reference_path=None, *, head_index=None, payload_root=None):
    source = reference_source_path(reference, reference_path)
    observed = json.loads(source.read_text())
    for key in ("dataset", "split", "sample_id", "sample_doc_hash", "model_arguments"):
        left, right = head.get("provenance", {}).get(key), observed.get("provenance", {}).get(key)
        if left != right or (left is None and key not in ("split", "sample_doc_hash")):
            raise ValueError(f"head and observed Feature2 capture differ in {key}")
    for key in ("capture_index", "block_index", "step_index", "forward_kind"):
        if head.get("forward", {}).get(key) != observed.get("forward", {}).get(key):
            raise ValueError(f"head and observed Feature2 capture differ in {key}")
    if head.get("forward", {}).get("capture_index") != reference["capture_index"]:
        raise ValueError("head and observed Feature2 refer to different forward events")
    if head["provenance"].get("sample_doc_hash") is None:
        # User prompts have no benchmark document identity. Compare the actual
        # full-sequence token input instead of inventing a dataset identifier.
        if head_index is None:
            raise ValueError("head index is required to compare captured request tokens")
        inputs = []
        for metadata, index in ((head, Path(head_index)), (observed, source)):
            entries = [entry for entry in metadata.get("tensors", [])
                       if entry.get("role") == "control" and
                       entry.get("name") == "control.before_postprocess.tokens"]
            if len(entries) != 1:
                raise ValueError("capture requires one full-sequence request token input")
            entry = entries[0]
            shape = entry["shape"]
            if (entry["dtype"] != "<i8" or entry["encoding"] != "integer" or
                    len(shape) != 2 or shape[0] != 1 or shape[1] < 1):
                raise ValueError("captured request token input has invalid shape or dtype")
            root = Path(payload_root).resolve() if payload_root is not None else Path(metadata["control_observation"].get("payload_root", "."))
            if not root.is_absolute():
                root = index.parent / root
            raw = (root / entry["path"]).read_bytes()
            if len(raw) != entry["byte_count"] or len(raw) != shape[1] * 8:
                raise ValueError("captured request token input has invalid byte count")
            inputs.append(raw)
        if inputs[0] != inputs[1]:
            raise ValueError("head and observed Feature2 capture differ in request tokens")


def apply_observed_feature2(case, output, reference_path, positions):
    """Load observed state and compare head-driven decisions with captured results."""
    reference = json.loads(reference_path.read_text())
    if reference.get("schema") not in ("supra-observed-feature2-step/v1", "supra-feature2-step/v1"):
        raise ValueError("observed Feature2 schema differs")
    rows, config, expected = reference["rows"], reference["config"], reference["expected"]
    completions = [all(state == 2 for state in expected["state"])]
    case["executions"][0]["expected_post_block_completions"] = completions
    begin, block = reference["block_start"], reference["block_id"]
    if len(rows) != 32 or [r["logical"] for r in rows] != list(range(begin, begin+32)):
        raise ValueError("observed Feature2 must contain the complete current block")
    future = reference.get("future")
    closeout_kind = config.get("closeout_kind", 0)
    current_positions = [r["logical"] for r in rows if
                         (r["state"] == 1 if closeout_kind == 1 else r["state"] != 2)]
    future_positions = future["prediction_positions"] if future else []
    if positions != current_positions + future_positions:
        raise ValueError("head inputs must equal observed current then future prediction positions")
    if config["vocabulary"] != case["head_checkpoints"]["vocab"]:
        raise ValueError("observed Feature2 vocabulary differs from head")
    suppressed = reference["suppressed_tokens"]
    if len(suppressed) > 255:
        raise ValueError("suppression list exceeds DDR interface capacity")
    # Standalone fixtures start a local state version; connected captures retain
    # the actual forward index checked by validate_observed_head_reference.
    capture = reference.get("capture_index", 0)
    # Only the control prefix is edited. Weight and numeric expected payloads are untouched.
    with (output / case["ddr_image"]).open("r+b") as stream:
        image = bytearray(stream.read(0x100000))
        def patch(name, offset, **values):
            patch_record(image, offset, load_schema(name), values)
        patch("execution_config", 0, sequence_length=reference["sequence_length"],
              suppressed_token_count=len(suppressed), generation_block_count=2 if future else 1)
        # The subsequent dependency update consumes actual proposal confidence
        # from the published state, including candidates that were not admitted.
        patch("forward_postprocess_config", POST-BASE, flags=1, mask_token_id=config["mask_token"],
              high_confidence_threshold_bf16=config["high"], tail_high_confidence_threshold_bf16=config["tail"],
              low_confidence_threshold_bf16=config["low"], verify_threshold_bf16=config["verify"],
              stability_bonus_bf16=config["bonus"], budget_scale_bf16=config["budget"],
              suppressed_token_base=BASE+0xD9010, suppressed_token_limit=BASE+0xD9010+max(1,len(suppressed))*4,
              current_state_limit=BASE+0xD6000+(64 if future else 32)*32,
              next_state_limit=BASE+0xD7000+(64 if future else 32)*32,
              block_configuration_limit=BASE+0xDA000+(2 if future else 1)*32,
              forward_event_limit=BASE+0xD8000+(2 if future else 1)*64)
        flags = int(config["tail_enable"]) | int(config["tail_bypass_all"])<<2 | int(config["tail_bypass_stable_only"])<<6
        flags |= closeout_kind << 3
        if config.get("transfer_only"):
            flags = 0x20
        patch("draft_verify_block_config", 0xDA000, block_slot=0, global_block_id=block,
              flags=flags, scheduled_quota=config["scheduled"], remaining_forwards=config["remaining"],
              step_index=config["step"], tail_after_step=config["tail_after"], maturity_age=3,
              position_count=32, input_capture_index=capture, capture_index=capture+1,
              current_state_entry=0, next_state_entry=0, observed_mask=0)
        for i, token in enumerate(suppressed):
            image[0xD9010+i*4:0xD9014+i*4] = int(token).to_bytes(4,"little")
        for i, row in enumerate(rows):
            image[0xD6000+i*32:0xD6020+i*32] = bytes(32)
            patch("token_state_entry", 0xD6000+i*32, token_position=row["logical"],
                  block_local_position=i, block_slot=0, global_block_id=block, state=row["state"],
                  origin=row["origin"], activation_bits=row["bits"], token_id=row["token"],
                  last_top1=row["last_top1"] & 0xffffffff, precision_age=row["age"],
                  cache_valid=int(not row["refresh_required"]), refresh_required=int(row["refresh_required"]),
                  capture_index=capture, source_a_pending=int(row.get("source_a_pending", False)))
        for ordinal, position in enumerate(current_positions):
            local = position-begin
            row = rows[local]
            # Preserve final_hidden_ddr_row_index, including an actual preceding L31 mapping.
            patch("prediction_record", 0xD1000+ordinal*32, token_position=position,
                  prediction_index=ordinal, block_slot=0, global_block_id=block, block_local_position=local,
                  activation_bits=row["bits"], forward_start_state=row["state"], current_token_id=row["token"],
                  tentative_token_id=row["token"], current_state_entry=local, next_state_entry=local)
        if future:
            future_flags = 2 | (int(future["source_a_handoff"]) << 7) | (future["max_handoff"] << 8)
            patch("draft_verify_block_config", 0xDA020, block_slot=1, global_block_id=future["block_id"],
                  flags=future_flags, scheduled_quota=future["scheduled"], remaining_forwards=1,
                  step_index=config["step"], tail_after_step=0, maturity_age=3,
                  position_count=32, input_capture_index=capture, capture_index=capture+1,
                  current_state_entry=32, next_state_entry=32, observed_mask=sum(
                      1 << (p-future["block_start"]) for p in future_positions))
            for i, row in enumerate(future["rows"]):
                offset = 0xD6000+(32+i)*32
                image[offset:offset+32] = bytes(32)
                patch("token_state_entry", offset, token_position=row["logical"], block_local_position=i,
                      block_slot=1, global_block_id=future["block_id"], state=row["state"], origin=row["origin"],
                      activation_bits=row["bits"], token_id=row["token"], last_top1=row["last_top1"] & 0xffffffff,
                      precision_age=row["age"], cache_valid=int(not row["refresh_required"]),
                      refresh_required=int(row["refresh_required"]), capture_index=capture,
                      source_a_pending=int(row["source_a_pending"]),
                      last_action_confidence_bf16=row["last_action_confidence"],
                      last_action_confidence_valid=int(row["last_action_valid"]))
            for ordinal, position in enumerate(future_positions, len(current_positions)):
                local = position-future["block_start"]
                row = future["rows"][local]
                patch("prediction_record", 0xD1000+ordinal*32, token_position=position,
                      prediction_index=ordinal, block_slot=1, global_block_id=future["block_id"],
                      block_local_position=local, activation_bits=row["bits"], forward_start_state=row["state"],
                      current_token_id=row["token"], tentative_token_id=row["token"],
                      current_state_entry=32+local, next_state_entry=32+local)
        stream.seek(0); stream.write(image)
    fields = load_schema("token_state_entry")
    history = bytearray()
    predicted_positions = set(positions)
    for i, row in enumerate(rows):
        payload = bytearray(32)
        patch_record(payload, 0, fields, dict(token_position=row["logical"], block_local_position=i,
            block_slot=0, global_block_id=block, state=expected["state"][i], origin=expected["commit_origin"][i],
            activation_bits=expected["bits"][i], token_id=expected["tokens"][i],
            last_top1=expected["last_top1"][i] & 0xffffffff, precision_age=expected["precision_age"][i]))
        path = f"expected.control_state_{i}.bin"
        (output / path).write_bytes(payload[:18])
        case["expected"].append(dict(name=f"control_state_{i}", address=BASE+0xD7000+i*32, bytes=18, path=path))
        remasked = bool(expected["masks"]["remasked"] & (1 << i))
        observed = row["logical"] in predicted_positions
        patch_record(payload, 0, fields, dict(capture_index=capture+1,
            prediction_flag=int(expected["state"][i] != 2),
            change_flags=int(bool(expected["masks"]["token_changed"] & (1 << i))) | (int(remasked) << 1),
            change_confidence_bf16=row["selected_probability"] if remasked else 0x3f80,
            last_action_confidence_bf16=row["action_confidence"] if observed else 0,
            last_action_confidence_valid=int(observed), source_a_pending=0))
        history.extend(payload[20:32])
    history_path = "expected.control_state_history.bin"
    (output / history_path).write_bytes(history)
    case["expected"].append(dict(name="control_state_history", address=BASE+0xD7000+20,
        bytes=len(history), element_bytes=12, stride_bytes=32, path=history_path))
    event_fields = {field["name"]: field for field in load_schema("forward_event")["fields"]}
    for name, source in (("selected_mask","selected"),("confirmed_mask","confirmed"),("remasked_mask","remasked"),
                         ("direct_locked_mask","direct"),("stable_tentative_mask","stable"),
                         ("fallback_tentative_mask","fallback"),("token_changed_mask","token_changed")):
        path = "expected.control_"+name+".bin"
        (output / path).write_bytes(int(expected["masks"][source]).to_bytes(4,"little"))
        case["expected"].append(dict(name="control_"+name,address=BASE+0xD8000+event_fields[name]["offset"],bytes=4,path=path))
    if "tail_closed" in expected["masks"]:
        path = "expected.control_tail_closed_mask.bin"
        (output / path).write_bytes(int(expected["masks"]["tail_closed"]).to_bytes(4,"little"))
        case["expected"].append(dict(name="control_tail_closed_mask",
            address=BASE+0xD8000+event_fields["tail_closed_mask"]["offset"], bytes=4,path=path))
    if future:
        future_expected = future["expected"]
        for i, row in enumerate(future["rows"]):
            payload = bytearray(32)
            observed = row["logical"] in future_positions
            remasked = bool(future_expected["masks"]["remasked"] & (1 << i))
            patch_record(payload, 0, fields, dict(token_position=row["logical"], block_local_position=i,
                block_slot=1, global_block_id=future["block_id"], state=future_expected["state"][i],
                origin=future_expected["commit_origin"][i], activation_bits=future_expected["bits"][i],
                token_id=future_expected["tokens"][i], last_top1=future_expected["last_top1"][i] & 0xffffffff,
                precision_age=future_expected["precision_age"][i], capture_index=capture+1,
                prediction_flag=int(future_expected["state"][i] != 2),
                change_flags=int(bool(future_expected["masks"]["token_changed"] & (1 << i))) | (int(remasked)<<1),
                change_confidence_bf16=row["selected_probability"] if remasked else 0x3f80,
                last_action_confidence_bf16=row["action_confidence"] if observed else row["last_action_confidence"],
                last_action_confidence_valid=int(observed or row["last_action_valid"]),
                source_a_pending=int(future_expected["source_a_pending"][i])))
            for suffix, start, size in (("state", 0, 18), ("history", 20, 12)):
                path = f"expected.future_{suffix}_{i}.bin"
                (output / path).write_bytes(payload[start:start+size])
                case["expected"].append(dict(name=f"future_{suffix}_{i}",
                    address=BASE+0xD7000+(32+i)*32+start, bytes=size, path=path))
        for name, source in (("selected_mask","selected"),("confirmed_mask","confirmed"),
                ("remasked_mask","remasked"),("direct_locked_mask","direct"),
                ("stable_tentative_mask","stable"),("fallback_tentative_mask","fallback"),
                ("token_changed_mask","token_changed")):
            path = "expected.future_"+name+".bin"
            (output / path).write_bytes(int(future_expected["masks"][source]).to_bytes(4,"little"))
            case["expected"].append(dict(name="future_"+name,
                address=BASE+0xD8040+event_fields[name]["offset"],bytes=4,path=path))
    case["provenance"]["control_reference"] = str(reference_path)
    tokens, vocab = len(positions), case["head_checkpoints"]["vocab"]
    logits = np.fromfile(output / "expected.output.bin", dtype="<u2").reshape(tokens, vocab)
    row_by_position = {r["logical"]: r for r in reference["rows"]}
    if future:
        row_by_position.update({r["logical"]: r for r in future["rows"]})
    rows = [row_by_position[p] for p in positions]
    values = [("candidate_top1", "<u4", [r["top1"] for r in rows]),
              ("candidate_logit", "<u2", [int(logits[i,r["top1"]]) for i,r in enumerate(rows)]),
              ("candidate_confidence", "<u2", [r["raw_confidence"] for r in rows]),
              ("candidate_probability", "<u2", [r["selected_probability"] for r in rows]),
              ("candidate_action", "<u2", [r["action_confidence"] for r in rows])]
    candidates = []
    for name, dtype, raw in values:
        path = "expected."+name+".bin"
        array = np.array(raw, dtype=dtype)
        array.tofile(output / path)
        candidates.append(dict(name=name, dtype=dtype, shape=[tokens], bytes=array.nbytes, path=path))
    case["head_checkpoints"]["candidates"] = candidates


def connect_preceding_layer(case, output, layer_path, tokens):
    """Read actual final DDR output from consecutive layers ending at L31."""
    layer_path = layer_path.resolve(strict=True)
    layer = json.loads(layer_path.read_text())
    source = layer.get("source", {})
    boundary = source.get("kind") == "boundary_scout_deep"
    source = source.get("deep", {}) if boundary else source
    executions = layer.get("executions", [])
    if (layer.get("schema") != "supra-testcase/v1" or not executions or
            type(source.get("start_layer")) is not int or type(source.get("layers")) is not int or
            not 0 <= source["start_layer"] < 32 or not 1 <= source["layers"] <= 32 or
            source["start_layer"] + source["layers"] != 32 or
            "head_checkpoints" in layer):
        raise ValueError("head connection requires prepared layers ending at L31")
    head_rows = case["head_checkpoints"]["tokens"]
    if (not isinstance(tokens, list) or len(tokens) != head_rows or len(set(tokens)) != head_rows or
            any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError("one unique packed L31 output token index is required per head input")
    outputs = [item for item in layer["expected"] if item["name"] == ("deep.hidden" if boundary else "hidden")]
    if len(outputs) != 1 or outputs[0]["bytes"] % 8192:
        raise ValueError("preceding L31 must provide its independent expected hidden tensor")
    hidden = outputs[0]
    if max(tokens) >= hidden["bytes"] // 8192:
        raise ValueError("head token selects an output omitted by L31")
    expected_hidden = np.fromfile(layer_path.parent / hidden["path"], dtype="<u2")
    if expected_hidden.size * 2 != hidden["bytes"]:
        raise ValueError("L31 expected hidden byte count differs")
    expected_hidden = expected_hidden.reshape(-1, 4096)[tokens]
    head_input = (output / "expected.hidden_input.bin").read_bytes()
    if expected_hidden.tobytes() != head_input:
        raise ValueError("captured head input differs from the selected L31 output raw bits")
    layer_regions = json.loads((layer_path.parent / layer["memory_map"]).read_text())
    regions = json.loads((output / case["memory_map"]).read_text())
    for region in layer_regions["memory_map"]:
        if any(region["base"] < other["limit"] and other["base"] < region["limit"]
               for other in regions["memory_map"]):
            raise ValueError("layer/head DDR regions overlap; assign separate base addresses")
    layer_image = layer_path.parent / layer["ddr_image"]
    if layer_image.stat().st_size != layer_regions["current_ddr_image_bytes"]:
        raise ValueError("L31 initial image size differs from its regions")
    if (layer_path.parent / layer["dramsim3_config"]).read_bytes() != (output / case["dramsim3_config"]).read_bytes():
        raise ValueError("L31 and head must use the same DDR configuration")
    regions["memory_map"].extend(dict(region, name="layer." + region["name"]) for region in layer_regions["memory_map"])
    image = bytearray((output / case["ddr_image"]).read_bytes())
    fields = {field["name"]: field for field in load_schema("execution_config")["fields"]}
    for name, field in fields.items():
        if not name.endswith("_base") or name == "output_hidden_base":
            continue
        limit_field = fields.get(name[:-4] + "limit")
        if limit_field is None:
            continue
        base = int.from_bytes(image[field["offset"]:field["offset"] + 8], "little")
        limit = int.from_bytes(image[limit_field["offset"]:limit_field["offset"] + 8], "little")
        if base < hidden["address"] + hidden["bytes"] and hidden["address"] < limit:
            raise ValueError(f"L31 hidden overlaps head descriptor {name}; use --base-address 0x60000000")
    image[HIDDEN - BASE:HIDDEN - BASE + len(head_input)] = bytes(len(head_input))
    patch_record(image, 0, load_schema("execution_config"),
        {"output_hidden_base": hidden["address"], "output_hidden_limit": hidden["address"] + hidden["bytes"]})
    for prediction_index, hidden_index in enumerate(tokens):
        patch_record(image, 0xD1000 + prediction_index * 32, load_schema("prediction_record"),
                     {"final_hidden_ddr_row_index": hidden_index})
    (output / case["ddr_image"]).write_bytes(image)
    reference = lambda name: os.path.relpath(layer_path.parent / name, output)
    case["initial_segments"] = [dict(address=layer_regions["base_address"],
        bytes=layer_image.stat().st_size, path=reference(layer["ddr_image"]))] + [
            dict(item, path=reference(item["path"])) for item in layer.get("initial_segments", [])]
    # The standalone head buffer is now unused and zeroed. Check the live
    # producer through layer.hidden and the consumer through head checkpoints.
    case["expected"] = [dict(item, name="layer." + item["name"], path=reference(item["path"]))
                        for item in layer["expected"]] + [
                            item for item in case["expected"] if item["name"] != "hidden_input"]
    case["executions"] = executions + case["executions"]
    case["head_checkpoints"]["execution_index"] = len(executions)
    if "attention_layout" in layer:
        case["attention_layout"] = layer["attention_layout"]
    if "attention_checkpoints" in layer:
        checks = layer["attention_checkpoints"]
        multiple = isinstance(checks, list)
        relocated = [dict(check, execution_index=check.get("execution_index", 0),
            expected=[dict(item, path=reference(item["path"])) for item in check["expected"]])
            for check in (checks if multiple else [checks])]
        case["attention_checkpoints"] = relocated if multiple else relocated[0]
    case["silu_mode"] = layer["silu_mode"]
    if "silu_segments" in layer:
        case["silu_segments"] = layer["silu_segments"]
    case["provenance"]["layer_output_tokens"] = tokens
    (output / case["memory_map"]).write_text(json.dumps(regions, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-data-root", type=Path)
    parser.add_argument("--vocabulary-prefix", type=int)
    parser.add_argument("--ddr", type=int, choices=(2400, 3200), default=3200)
    parser.add_argument("--base-address", type=lambda value: int(value, 0), default=BASE,
                        help="256-byte-aligned head DDR base; use a separate range when connecting layers")
    parser.add_argument("--preceding-layer-testcase", type=Path)
    parser.add_argument("--feature2-reference", type=Path, help="captured current-block inputs and independent decisions")
    parser.add_argument("--regular-control-index", type=Path,
                        help="captured dependency prefix and control state for the selected layer range")
    parser.add_argument("--next-layer-testcase", type=Path,
                        help="prepared next L0, using a separate DDR address range")
    parser.add_argument("--next-layer-index", type=Path,
                        help="independent capture of the actual following L0")
    parser.add_argument("--layer-output-tokens", type=lambda value: [int(token) for token in value.split(",")],
                        help="packed L31 output indices in head input order")
    args = parser.parse_args()
    if args.regular_control_index and not (args.preceding_layer_testcase and args.feature2_reference):
        parser.error("--regular-control-index is missing --preceding-layer-testcase or --feature2-reference")
    if args.feature2_reference and not args.regular_control_index:
        observed = json.loads(args.feature2_reference.read_text())
        if observed.get("future") or any(row["state"] == 2 for row in observed["rows"]):
            parser.error("captured sparse/current+future state is missing --regular-control-index and --preceding-layer-testcase")
    if bool(args.next_layer_testcase) != bool(args.next_layer_index) or (
            args.next_layer_testcase and not args.regular_control_index):
        parser.error("next L0 requires both testcase/index and regular control")
    if args.regular_control_index:
        integration = HARDWARE.parent / "integration"
        sys.path.insert(0, str(integration))
        from prepare_layer_testcase import require_regular_attention, attach_regular_control
        from prepare_testcase import connect_steps
        require_regular_attention(json.loads(args.preceding_layer_testcase.read_text()))
    path = prepare(args.index, args.output, args.tensor_data_root, args.vocabulary_prefix, args.ddr,
                  preceding_layer_testcase=args.preceding_layer_testcase, layer_output_tokens=args.layer_output_tokens,
                  feature2_reference=args.feature2_reference, base_address=args.base_address)
    if args.regular_control_index:
        attach_regular_control(path, args.regular_control_index)
        if args.next_layer_testcase:
            connect_steps(path, args.next_layer_testcase, args.next_layer_index)
    print(path)


if __name__ == "__main__":
    main()
