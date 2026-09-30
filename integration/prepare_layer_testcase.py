#!/usr/bin/env python3
"""Pack captured W4 layer tensors into DDR inputs and comparison files."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HARDWARE = HERE.parent / "hardware"
sys.path.insert(0, str(HARDWARE / "scripts"))
from artifact_paths import artifact_root, reference_source_path
from build_forward_postprocess_config import pack_schema
from layer_reference_data import SCHEMA, validate_numeric_capabilities

BASE = 0x10000000
HIDDEN, FFN, HEADS, DIM, MAX_S = 4096, 12288, 32, 128, 2048
LINEARS = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")
FIELDS = ("query", "key", "value", "attention_output", "ffn_gate", "ffn_up", "ffn_down")


def align(value, alignment=32):
    return (value + alignment - 1) // alignment * alignment


def pack_record(schema, values):
    initial = {field["name"]: 0 for field in schema["fields"]
               if "constant" not in field and "default" not in field}
    return bytearray(pack_schema(schema, {**initial, **values}, schema.get("schema", "record")))


def pack_weights(codes):
    if codes.ndim != 2 or codes.shape[0] % 8 or codes.shape[1] % 4:
        raise ValueError("W4 matrices require output multiples of 8 and input multiples of 4")
    if codes.dtype != np.int8 or np.any(codes < -8) or np.any(codes > 7):
        raise ValueError("weights must be signed W4 codes in int8 containers")
    out, width = codes.shape
    values = (codes.reshape(out // 8, 8, width // 2, 2).astype(np.uint8) & 15)
    packed = values[..., 0] | (values[..., 1] << 4)
    return np.ascontiguousarray(packed.transpose(0, 2, 1))


def pack_tokens(bits, positions, write_token_indices=None, source_indices=None, output_token_indices=None,
              round_token_limit=48):
    if type(round_token_limit) is not int or not 1 <= round_token_limit <= 48:
        raise ValueError("round_token_limit must be an integer in [1, 48]")
    if source_indices is not None and (len(source_indices) != len(bits) or
            len(set(source_indices)) != len(source_indices) or
            any(type(value) is not int or not 0 <= value < 2048 for value in source_indices)):
        raise ValueError("source_indices must be unique resident hidden token indices below 2048")
    if output_token_indices is not None:
        selected_outputs = set(output_token_indices)
        if any(type(token) is not int or not 0 <= token < len(bits) for token in selected_outputs):
            raise ValueError("output_token_indices must contain reference_data token indices")
        preferred = sorted(selected_outputs) + [token for token in range(len(bits)) if token not in selected_outputs]
        preferred_writes = None if write_token_indices is None else {
            local for local, token in enumerate(preferred) if token in write_token_indices}
        preferred_sources = None if source_indices is None else [source_indices[token] for token in preferred]
        candidate, candidate_order, candidate_rounds = pack_tokens(
            [bits[token] for token in preferred], [positions[token] for token in preferred],
            preferred_writes, preferred_sources, round_token_limit=round_token_limit)
        candidate_order = [preferred[token] for token in candidate_order]
        usable = set(candidate_order[:len(selected_outputs)]) == selected_outputs
        cursor = ordinal = 0
        for summary in candidate_rounds:
            tokens = summary["tokens"]
            outputs = sum(token in selected_outputs for token in candidate_order[ordinal:ordinal + tokens])
            output_groups, unused_groups = set(), set()
            for physical in range(tokens):
                address = cursor + 32 + physical * 16
                local, group = candidate[address + 8], candidate[address + 13]
                required = candidate_order[ordinal + local] in selected_outputs
                usable &= required == (physical < outputs) and required == (local < outputs)
                (output_groups if required else unused_groups).add(group)
            usable &= output_groups == set(range(len(output_groups))) and not (output_groups & unused_groups)
            candidate[cursor + 3] |= 2
            candidate[cursor + 30] = outputs
            candidate[cursor + 31] = len(output_groups)
            summary["output_tokens"] = outputs
            ordinal += tokens
            cursor += struct.unpack_from("<H", candidate, cursor + 8)[0]
        if usable:
            if len(candidate_rounds) > 64:
                raise ValueError("output subset requires more than 64 metadata rounds")
            return candidate, candidate_order, candidate_rounds
        metadata, order, rounds = bytearray(), [], []
        # Keep each subset in complete PE groups and preserve the full hidden allocation.
        for required in (True, False):
            subset = [token for token in range(len(bits)) if (token in selected_outputs) == required]
            if not subset:
                continue
            writes = None if write_token_indices is None else {
                local for local, token in enumerate(subset) if token in write_token_indices}
            sources = None if source_indices is None else [source_indices[token] for token in subset]
            block, local_order, local_rounds = pack_tokens(
                [bits[token] for token in subset], [positions[token] for token in subset], writes, sources,
                round_token_limit=round_token_limit)
            cursor = 0
            for index, summary in enumerate(local_rounds):
                tokens = summary["tokens"]
                block[cursor + 3] |= 2
                block[cursor + 30] = tokens if required else 0
                block[cursor + 31] = summary["groups"] if required else 0
                struct.pack_into("<H", block, cursor + 4, len(rounds) + index)
                first = struct.unpack_from("<H", block, cursor + 6)[0]
                struct.pack_into("<H", block, cursor + 6, len(order) + first)
                if source_indices is None:
                    for physical in range(tokens):
                        address = cursor + 32 + physical * 16
                        source = struct.unpack_from("<I", block, address)[0]
                        struct.pack_into("<I", block, address, source + len(order))
                summary["output_tokens"] = tokens if required else 0
                cursor += struct.unpack_from("<H", block, cursor + 8)[0]
            metadata.extend(block)
            order.extend(subset[token] for token in local_order)
            rounds.extend(local_rounds)
        if len(rounds) > 64:
            raise ValueError("output subset requires more than 64 metadata rounds")
        return metadata, order, rounds
    a4 = [i for i, bit in enumerate(bits) if bit == 4]
    a8 = [i for i, bit in enumerate(bits) if bit == 8]
    if len(a4) + len(a8) != len(bits):
        raise ValueError("token bits must be 4 or 8")
    count = max((len(bits) + round_token_limit - 1) // round_token_limit,
                (len(a4) + 2 * len(a8) + 63) // 64)
    if count > 64:
        raise ValueError("input requires more than 64 metadata rounds")
    metadata, order, rounds = bytearray(), [], []
    for ri in range(count):
        left = count - ri - 1
        take4 = min(round_token_limit, len(a4))
        while True:
            take8 = min(len(a8), round_token_limit - take4, (64 - take4) // 2)
            if take4 + take8 and len(a4) + len(a8) - take4 - take8 <= left * round_token_limit and len(a4) + 2 * len(a8) - take4 - 2 * take8 <= left * 64:
                break
            take4 -= 1
            if take4 < 0:
                raise ValueError("cannot fit token metadata capacity")
        selected = a4[:take4] + a8[:take8]
        del a4[:take4]
        del a8[:take8]
        remaining4 = list(range(take4))
        remaining8 = list(range(take4, take4 + take8))
        physical, group = [], 0
        while remaining4 or remaining8:
            chunk8 = remaining8[:8]
            del remaining8[:8]
            physical.extend((token, group, slot, 3) for slot, token in enumerate(chunk8))
            for phase in (1, 2):
                chunk4 = remaining4[:8 - len(chunk8)]
                del remaining4[:len(chunk4)]
                physical.extend((token, group, len(chunk8) + slot, phase) for slot, token in enumerate(chunk4))
            group += 1
        tokens = len(selected)
        inverse_offset = 32 + tokens * 16
        record = bytearray(inverse_offset + align(tokens, 16))
        struct.pack_into("<BBBBHHHHHHIIQ", record, 0, tokens, group, 1, 0, ri, len(order),
                         len(record), 16, inverse_offset, align(tokens, 16), 1, 1, 0)
        for pi, (local, compute_group, slot, mask) in enumerate(physical):
            original = selected[local]
            source_index = len(order) + local if source_indices is None else source_indices[original]
            struct.pack_into("<IHHBBBBBBBB", record, 32 + pi * 16, source_index,
                             int(positions[original]), int(positions[original]), local, 0,
                             int(bits[original]), 0, 0, compute_group, slot, mask)
            record[inverse_offset + local] = pi
        if write_token_indices is not None:
            disabled = sum(1 << pi for pi, (local, _, _, _) in enumerate(physical)
                           if selected[local] not in write_token_indices)
            struct.pack_into("<Q", record, 24, disabled)
        metadata.extend(record)
        order.extend(selected)
        rounds.append(dict(a4=take4, a8=take8, tokens=tokens, groups=group))
    return metadata, order, rounds


def select_ffn_group_batches(rounds, requested=None):
    """Fit fixed Gate/Up groups into the 18-group activation SRAM layout."""
    if not rounds or any(not 1 <= r['groups'] <= 4 for r in rounds):
        raise ValueError('FFN metadata must contain 1-4 compute groups per round')
    if requested is not None and requested not in (4,6):
        raise ValueError('FFN group batch count must be 4 or 6')
    for batches in ((requested,) if requested is not None else (6,4)):
        if all(sum(r['groups'] for r in rounds[i:i+batches]) <= 18
               for i in range(0,len(rounds),batches)):
            return batches
    raise ValueError('FFN grouping exceeds the 18-compute-group activation capacity; use 4 batches')


def mark_qkvo_groups(metadata, rounds, max_batches=6, max_compute_groups=18, *, online=False):
    """Mark contiguous metadata groups accepted by the production Q/K/V/O path."""
    if not rounds:
        raise ValueError("Q/K/V/O grouping requires nonempty metadata")
    offsets, cursor = [], 0
    for summary in rounds:
        offsets.append(cursor)
        cursor += struct.unpack_from("<H", metadata, cursor + 8)[0]
    if cursor != len(metadata):
        raise ValueError("token metadata records do not cover the packed payload")

    groups, first, batches, compute_groups = [], 0, 0, 0
    remaining_a8 = sum(int(r["a8"]) for r in rounds)
    tokens, slots = 0, 0
    for index, summary in enumerate(rounds):
        next_compute_groups = int(summary["groups"])
        if batches and (batches == max_batches or
                        compute_groups + next_compute_groups > max_compute_groups):
            metadata[offsets[index - 1] + 3] |= 0x04
            groups.append(dict(first_round=first, batches=batches,
                               tokens=tokens, activation_slots=slots,
                               compute_groups=compute_groups))
            first, batches, compute_groups = index, 0, 0
            tokens, slots = 0, 0
        batches += 1
        compute_groups += next_compute_groups
        tokens += int(summary["tokens"])
        slots += int(summary["a4"]) + 2 * int(summary["a8"])
        remaining_a8 -= int(summary["a8"])
        if online and index + 1 < len(rounds) and (
                batches == max_batches or compute_groups + (5 if remaining_a8 else 3) > max_compute_groups):
            metadata[offsets[index] + 3] |= 0x04
            groups.append(dict(first_round=first, batches=batches, tokens=tokens,
                               activation_slots=slots, compute_groups=compute_groups))
            first, batches, compute_groups = index + 1, 0, 0
            tokens, slots = 0, 0
    metadata[offsets[-1] + 3] |= 0x04
    groups.append(dict(first_round=first, batches=batches, tokens=tokens,
                       activation_slots=slots,
                       compute_groups=compute_groups))
    if any(group["batches"] > max_batches or
           group["compute_groups"] > max_compute_groups or
           group["activation_slots"] > 288 for group in groups):
        raise ValueError("Q/K/V/O metadata group exceeds hardware capacity")
    return groups


def last_layer_output_token_indices(metadata, positions):
    """Select recorded logit/hidden consumers without changing KV or query membership."""
    first = metadata.get("model_layer_index")
    layers = metadata.get("layer_count", 1)
    if type(first) is not int or type(layers) is not int or layers < 1 or first + layers != 32:
        raise ValueError("output subset requires a consecutive layer range ending at L31")
    forward = metadata.get("forward", {})
    if forward.get("forward_kind") not in ("full_sequence", "local_block", "boundary_refresh", "local_confirmation", "local_forced_finish"):
        raise ValueError("output subset requires a recorded generator forward")
    for name in ("prediction_positions", "future_prediction_positions"):
        if name not in forward:
            raise ValueError(f"output subset requires explicit {name}; missing is not empty")
    if forward.get("isolated_groups") or forward.get("attention_mask"):
        raise ValueError("output subset does not support isolated groups or explicit Attention masks")
    prediction = list(forward["prediction_positions"])
    retained = list(forward.get("retained_hidden_positions", []))
    mask = forward.get("prediction_mask")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.size != len(prediction):
            raise ValueError("captured prediction mask must match prediction positions")
        prediction = [p for p, enabled in zip(prediction, mask.reshape(-1)) if enabled]
        # Live observations carry the exact current block interval. The capture
        # candidate array includes locked positions and is not an output list.
        state = metadata.get("control_observation", {}).get("checkpoints", {}).get("before_postprocess")
        if state is None and "retained_hidden_positions" not in forward:
            raise ValueError("masked capture requires an observed block interval or explicit retained hidden positions")
        if state is not None:
            required = set(prediction) | set(forward["future_prediction_positions"])
            retained = [int(p) for p in positions if state["block_start"] <= p < state["block_end"]
                        and p not in required]
    consumers = prediction + list(forward["future_prediction_positions"]) + retained
    position_to_row = {int(position): token for token, position in enumerate(positions)}
    if len(position_to_row) != len(positions):
        raise ValueError("output subset query positions must be unique")
    if any(type(position) is not int or position not in position_to_row for position in consumers):
        raise ValueError("every output consumer must name an executed query token position")
    return {position_to_row[position] for position in consumers}


def prepare_attention_checkpoints(reference_data, output, metadata, order, tokens, sequence, layer,
                                  *, preserve_input=False):
    """Pack captured Attention tensors and their token map."""
    token_map, residual_map, cursor, first = [], [], 0, 0
    while cursor < len(metadata):
        count = metadata[cursor]
        token_map.append([order[first + metadata[cursor + 32 + p * 16 + 8]]
                          for p in range(count)])
        outputs = metadata[cursor + 30] if metadata[cursor + 3] & 2 else count
        if outputs > count:
            raise ValueError("Attention output subset exceeds query count")
        residual_map.append(token_map[-1][:outputs])
        first += count
        cursor += struct.unpack_from("<H", metadata, cursor + 8)[0]
    if first != tokens or cursor != len(metadata):
        raise ValueError("Attention checkpoint metadata does not cover all tokens")
    prefix = layer_prefix(reference_data, 0, 1)
    expected = []
    query_keys = [("expected", prefix + name) for name in ("query_codes", "query_scale")]
    if any(key in reference_data.entries for key in query_keys) and not all(
            key in reference_data.entries for key in query_keys):
        raise ValueError("Attention checkpoint requires both Query codes and scale")
    tensors = [("probability_codes", "|i1", (HEADS, tokens, sequence)),
               ("probability_scale", "<u2", (HEADS, tokens, 1))]
    if all(key in reference_data.entries for key in query_keys):
        tensors += [("query_codes", "|i1", (HEADS, tokens, DIM)),
                    ("query_scale", "<u2", (HEADS, tokens, 1))]
        intermediate_keys = [("expected", prefix + name)
                             for name in ("scores", "probabilities")]
        if any(key in reference_data.entries for key in intermediate_keys) and not all(
                key in reference_data.entries for key in intermediate_keys):
            raise ValueError("Attention checkpoint requires both scores and probabilities")
        if all(key in reference_data.entries for key in intermediate_keys):
            tensors += [("scores", "<u2", (HEADS, tokens, sequence)),
                        ("probabilities", "<u2", (HEADS, tokens, sequence))]
    # PV is computed only for output rows. The full-query observation below
    # applies to ordinary layers; L31 subset keeps its existing checks.
    if residual_map == token_map and ("expected", prefix + "context") in reference_data.entries:
        tensors.append(("context", "<u2", (HEADS, tokens, DIM)))
    for name, dtype, shape in tensors:
        entry = reference_data.entries["expected", prefix + name]
        if (np.dtype(entry["dtype"]) != np.dtype(dtype) or
                entry.get("encoding") != ("integer" if name.endswith("_codes") else "bf16_raw") or
                tuple(entry["shape"]) not in (shape, (1, *shape))):
            raise ValueError(f"Attention checkpoint shape/dtype/encoding differs: {name}")
        value = reference_data.load("expected", prefix + name, dtype)
        path = output / (name + ".expected.bin")
        value.tofile(path)
        expected.append(dict(name=name, path=path.name, dtype=dtype,
                             shape=list(shape), bytes=value.nbytes))
    for name, role, tensor_name in (
            ("preserved_hidden", "input", "hidden"),
            ("attention_residual", "expected", prefix + "attention_residual")):
        # Input preservation covers multiple initial batches and the layout
        # change before L31.
        if name == "preserved_hidden" and not preserve_input:
            continue
        if (role, tensor_name) not in reference_data.entries:
            continue
        value = reference_data.load(role, tensor_name, "<u2").reshape(tokens, HIDDEN)
        path = output / (name + ".expected.bin")
        value.tofile(path)
        expected.append(dict(name=name, path=path.name,
                             dtype="<u2", shape=[tokens, HIDDEN], bytes=value.nbytes))
    positions = reference_data.load("input", "positions").reshape(-1)
    return dict(layer=layer, tokens=tokens, sequence=sequence, heads=HEADS,
                token_positions=[int(position) for position in positions],
                physical_to_reference_token=token_map,
                attention_residual_token_order=residual_map,
                preserved_hidden_token_order=list(order), expected=expected)


class ReferenceData:
    def __init__(self, index, payload_root=None, *, metadata=None):
        self.index = index.resolve(strict=True)
        self.metadata = json.loads(self.index.read_text()) if metadata is None else metadata
        if self.metadata.get("schema") not in (SCHEMA, "supra-layer-reference-data/v1"):
            raise ValueError("expected SUPRA layer reference data")
        validate_numeric_capabilities(self.metadata)
        self.entries = {(e["role"], e["name"]): e for e in self.metadata["tensors"]}
        if len(self.entries) != len(self.metadata["tensors"]):
            raise ValueError("duplicate tensor names")
        self.root = payload_root.resolve() if payload_root else self.index.parent
        declared_ratios = self.metadata.get("a4_clip_ratios")
        if declared_ratios is not None and not self.metadata.get("numeric_capabilities", {}).get("a4_clipping"):
            raise ValueError("declared A4 clipping is missing its installed/ratio input records")
        # Verify the actual scalar payload, not just the tensor descriptions.
        for layer in self.metadata.get("layers", []) if isinstance(self.metadata.get("layers"), list) else []:
            if not layer.get("clipping"):
                continue
            prefix = ("" if ("input", "weight.q_proj.codes") in self.entries
                      else f"layer{layer['layer']}.")
            for linear, state in layer.get("clipping", {}).items():
                name = prefix + linear
                installed = bool(self.load("input", name + ".clip.installed").item())
                if installed != state["installed"]:
                    raise ValueError(f"clipping installed payload differs from declaration: {name}")
                ratio = int(self.load("input", name + ".clip.ratio", "<u2").item()) if installed else None
                if ratio != state["ratio_bf16_raw"]:
                    raise ValueError(f"clipping ratio payload differs from declaration: {name}")
                if declared_ratios is not None:
                    actual = float(np.array(ratio << 16, dtype="<u4").view("<f4")) if installed else None
                    if linear not in declared_ratios or actual != declared_ratios[linear]:
                        raise ValueError(f"declared A4 clipping differs from installed payload: {name}")

    def load(self, role, name, dtype=None):
        entry = self.entries[role, name]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = self.root / path
        raw_dtype = np.dtype(entry["dtype"])
        if (path.stat().st_size != entry["byte_count"] or
                int(np.prod(entry["shape"], dtype=np.int64)) * raw_dtype.itemsize != entry["byte_count"]):
            raise ValueError(f"tensor byte count differs: {name}")
        if dtype is not None and np.dtype(entry["dtype"]) != np.dtype(dtype):
            raise ValueError(f"unexpected raw dtype: {name}: {entry['dtype']}")
        if dtype == "<u2" and entry["encoding"] != "bf16_raw":
            raise ValueError(f"expected BF16 raw encoding: {name}")
        return np.fromfile(path, dtype=entry["dtype"]).reshape(entry["shape"])


def layer_prefix(reference_data, layer, layers):
    if layers == 1 and ("input", "weight.q_proj.codes") in reference_data.entries:
        return ""
    info = getattr(reference_data, "metadata", {})
    records = info.get("layers")
    if isinstance(records, list) and layer < len(records) and "model_layer" in records[layer]:
        return f"layer{records[layer]['layer']}."
    return f"layer{info.get('model_layer_index', 0) + layer}."


def hardware_softmax_tables():
    """Read literal BF16 tables from the package compiled into the RTL."""
    source = (HARDWARE / "rtl/config/softmax_lut_pkg.sv").read_text()
    tables = {}
    for name in ("exp", "reciprocal"):
        declaration = re.search(
            r"SOFTMAX_" + name.upper() + r"_LUT\s*\[0:255\]\s*=\s*'\{([^}]+)\}", source)
        values = re.findall(r"16'h([0-9a-fA-F]{4})", declaration.group(1)) if declaration else []
        if len(values) != 256:
            raise ValueError(f"cannot read hardware softmax {name} table")
        tables[name] = np.asarray([int(value, 16) for value in values], dtype="<u2")
    return tables


def validate_visibility(options):
    for name in ("attention_mask", "isolated_groups"):
        if options.get(name):
            raise ValueError(f"unsupported visibility/selection: {name}")
    if options.get("attention_bias") is not None:
        raise ValueError("attention_bias has no corresponding input in the prepared Attention descriptor")
    if options.get("use_cache", True) is not True:
        raise ValueError("use_cache=False conflicts with the prepared K/V output regions")


def validate_layer_inputs(reference_data, layers, records, bits, positions, write_token_indices=None, commit_token_indices=None):
    """Check constants and connections before writing any DDR image."""
    info, tokens, sequence = reference_data.metadata, len(bits), reference_data.metadata["sequence"]
    if len(records) != layers:
        raise ValueError("layer descriptor count must equal layer_count")
    forward = info.get("forward", {})
    for options in (info, forward):
        validate_visibility(options)
    writes = positions if write_token_indices is None else positions[write_token_indices]
    commits = positions if commit_token_indices is None else positions[commit_token_indices]
    for name, expected in (("input_positions", positions), ("refresh_positions", writes),
                           ("activation_bits", bits)):
        value = np.asarray(forward.get(name, expected))
        if name == "activation_bits" and value.shape == (1, tokens):
            value = value[0]
        if not np.array_equal(value, expected):
            raise ValueError(f"forward {name} differs from full query mapping")
    connections = info.get("connections")
    if connections is None:
        connections = [info["connection"]] if "connection" in info else []
    if connections and len(connections) != layers - 1:
        raise ValueError("hidden connections must cover every adjacent layer")
    for i, connection in enumerate(connections):
        if (connection.get("mapping") != "identity" or
                connection.get("source") != f"layer{i}.block_output" or
                connection.get("destination") != f"layer{i+1}.hidden" or
                connection.get("source_token_indices", list(range(tokens))) != list(range(tokens))):
            raise ValueError(f"hidden connection {i} differs from the consecutive-layer token mapping: {connection}")

    def hidden(role, name):
        value = reference_data.load(role, name, "<u2")
        if value.shape not in ((tokens, HIDDEN), (1, tokens, HIDDEN)):
            raise ValueError(f"hidden shape mismatch: {name}")
        return value.reshape(tokens, HIDDEN)

    previous = hidden("input", "hidden")
    tables = hardware_softmax_tables()
    # bf16_silu_pwl_pkg.coefficients has these fixed segment boundaries.
    breakpoints = np.asarray([0xc100, 0xc0c0, 0xc080, 0xc040, 0xc000, 0xbfc0,
                              0xbf80, 0xbf00, 0, 0x3f00, 0x3f80, 0x3fc0,
                              0x4000, 0x4040, 0x4080, 0x40c0, 0x4100], dtype="<u2")
    if ("input", "silu_breakpoints") in reference_data.entries:
        if not np.array_equal(reference_data.load("input", "silu_breakpoints", "<u2"), breakpoints):
            raise ValueError("SiLU breakpoints differ from fixed hardware boundaries")
    for name, expected in tables.items():
        if not np.array_equal(reference_data.load("input", "softmax_" + name + "_lut", "<u2"), expected):
            raise ValueError(f"softmax {name} LUT differs from hardware")
    silu_layers, rope_tables = [], []
    for layer, descriptor in enumerate(records):
        prefix = layer_prefix(reference_data, layer, layers)
        if descriptor.get("tokens", tokens) != tokens:
            raise ValueError("per-layer tokens differ from shared token mapping")
        for name, expected in (("activation_bits", bits), ("positions", positions),
                               ("kv_write_positions", writes), ("cache_commit_positions", commits)):
            for tensor_name in (name, prefix + name):
                if ("input", tensor_name) in reference_data.entries and not np.array_equal(
                        reference_data.load("input", tensor_name).reshape(-1), expected):
                    raise ValueError(f"unsupported per-layer {tensor_name}")
        kwargs = descriptor.get("kwargs", {})
        validate_visibility(kwargs)
        for name in ("query_position_ids", "kv_write_position_ids", "replace_position", "replace_position_kv"):
            value = kwargs.get(name)
            if value is None:
                continue
            if isinstance(value, dict) and set(value) == {"tensor"}:
                value = reference_data.load("input", value["tensor"])
            value = np.asarray(value).reshape(-1)
            if name.startswith("replace_"):
                if value.dtype != np.bool_ or len(value) != sequence:
                    raise ValueError(f"invalid per-layer {name}")
                value = np.flatnonzero(value)
                matches = np.array_equal(value, np.sort(writes if name.endswith("_kv") else positions))
            else:
                matches = np.array_equal(value, writes if name == "kv_write_position_ids" else positions)
            if not matches:
                raise ValueError(f"unsupported per-layer {name}")
        if layer or ("input", prefix + "hidden") in reference_data.entries:
            if not np.array_equal(hidden("input", prefix + "hidden"), previous):
                raise ValueError(f"{prefix}hidden differs from preceding hidden output")
        if layer < layers - 1:
            previous = hidden("expected", prefix + "block_output")

        # Check both cache views when the capture exposes QK/PV inputs:
        # Attention consumes retained cache with current writes overlaid, while
        # persistent output overlays only the explicitly committed positions.
        write_indices = np.arange(tokens) if write_token_indices is None else np.asarray(write_token_indices)
        write_positions = positions[write_indices]
        untouched_positions = np.ones(sequence, dtype=np.bool_)
        untouched_positions[write_positions] = False
        for retained_name, consumed_name, produced_name, persistent_name, dtype, width in (
                ("retained_key_codes", "qk_key_codes", "new_key_codes", "cache_key_codes", "|i1", DIM),
                ("retained_key_scale", "qk_key_scale", "new_key_scale", "cache_key_scale", "<u2", 1),
                ("retained_value_codes", "pv_value_codes", "new_value_codes", "cache_value_codes", "|i1", DIM)):
            keys = (("input", prefix + retained_name),
                    ("expected", prefix + consumed_name),
                    ("expected", prefix + produced_name))
            present = [key in reference_data.entries for key in keys]
            if not present[1]:
                continue
            if not all(present):
                raise ValueError(
                    f"Attention cache connection requires {retained_name}, "
                    f"{consumed_name}, and {produced_name}")
            retained = reference_data.load(*keys[0], dtype).reshape(HEADS, sequence, width)
            consumed = reference_data.load(*keys[1], dtype).reshape(HEADS, sequence, width)
            produced = reference_data.load(*keys[2], dtype).reshape(HEADS, tokens, width)
            if (not np.array_equal(consumed[:, untouched_positions],
                                   retained[:, untouched_positions]) or
                    not np.array_equal(consumed[:, write_positions],
                                       produced[:, write_indices])):
                raise ValueError(
                    f"Attention cache overlay differs from captured {consumed_name}")
            commit_indices = np.asarray(commit_token_indices, dtype=np.int64)
            persistent = retained.copy()
            persistent[:, positions[commit_indices]] = produced[:, commit_indices]
            captured_persistent = reference_data.load(
                "expected", prefix + persistent_name, dtype).reshape(HEADS, sequence, width)
            if not np.array_equal(captured_persistent, persistent):
                raise ValueError(
                    f"persistent {persistent_name} differs from retained cache plus committed K/V")
        for name, expected in tables.items():
            key = prefix + "softmax_" + name + "_lut"
            if ("input", key) in reference_data.entries and not np.array_equal(reference_data.load("input", key, "<u2"), expected):
                raise ValueError(f"{key} differs from hardware")
        coefficient_sets = []
        for candidate in (list(dict.fromkeys(["", prefix, f"layer{info.get('model_layer_index', 0)}."])) if layers == 1 else [prefix]):
            names = [candidate + "silu_coefficients." + str(i) for i in range(2)]
            present = [("input", name) in reference_data.entries for name in names]
            if any(present):
                if not all(present):
                    raise ValueError("SiLU requires both slope and intercept arrays")
                values = [reference_data.load("input", name, "<u2") for name in names]
                if any(value.shape != (16,) for value in values):
                    raise ValueError("SiLU slope and intercept must each contain 16 segments")
                coefficient_sets.append((values[0].astype(np.uint32) | (values[1].astype(np.uint32) << 16)).tolist())
        if coefficient_sets and any(value != coefficient_sets[0] for value in coefficient_sets):
            raise ValueError("conflicting single-layer SiLU coefficients")
        if not coefficient_sets:
            raise ValueError("each layer requires explicit SiLU coefficients")
        silu_layers.append(coefficient_sets[0])
        selected_tables = {}
        for name in ("cos", "sin"):
            selected = prefix + "rope_selected_" + name
            if ("input", selected) in reference_data.entries:
                value = reference_data.load("input", selected, "<u2")
                if value.shape not in ((tokens, DIM), (1, tokens, DIM), (1, 1, tokens, DIM)):
                    raise ValueError(f"RoPE shape mismatch: {selected}")
                value = value.reshape(tokens, DIM)
                table = np.zeros((MAX_S, DIM), dtype="<u2")
                table[positions] = value
            else:
                value = reference_data.load("input", "rope_cosine" if name == "cos" else "rope_sine", "<u2")
                if value.shape not in ((sequence, DIM), (1, sequence, DIM), (1, 1, sequence, DIM)):
                    raise ValueError(f"RoPE {name} shape mismatch")
                value = value.reshape(sequence, DIM)
                table = np.zeros((MAX_S, DIM), dtype="<u2")
                table[:sequence] = value
            if not np.array_equal(value[:, :DIM // 2], value[:, DIM // 2:]):
                raise ValueError(f"RoPE {name} requires identical paired halves")
            if layer and not np.array_equal(table, rope_tables[0][name]):
                raise ValueError("per-layer RoPE tables differ")
            selected_tables[name] = table
        rope_tables.append(selected_tables)
    if any(value != silu_layers[0] for value in silu_layers):
        raise ValueError("all layers require the same explicit SiLU table")
    if silu_layers[0] is not None and ("input", "silu_breakpoints") not in reference_data.entries:
        raise ValueError("explicit SiLU table is missing input tensor 'silu_breakpoints'")
    return silu_layers[0], rope_tables


def validate_tensor_shapes(reference_data, layers, tokens, sequence):
    """Require the documented axes, retaining existing batch=1 wrappers."""
    def require(role, name, shapes):
        entry = reference_data.entries[role, name]
        if tuple(entry["shape"]) not in shapes:
            raise ValueError(f"tensor shape mismatch: {name}: {entry['shape']}")

    require("input", "hidden", ((tokens, HIDDEN), (1, tokens, HIDDEN)))
    for name in ("activation_bits", "positions"):
        require("input", name, ((tokens,), (1, tokens)))
    for layer in range(layers):
        prefix = layer_prefix(reference_data, layer, layers)
        for linear in LINEARS:
            width = FFN if linear == "ff_out" else HIDDEN
            height = FFN if linear in ("ff_proj", "up_proj") else HIDDEN
            require("input", prefix + "weight." + linear + ".codes", ((height, width), (1, height, width)))
            require("input", prefix + "weight." + linear + ".scale", ((height,), (1, height), (height, 1)))
        require("input", prefix + "v_scale", ((HEADS,), (1, HEADS), (HEADS, 1), (1, HEADS, 1, 1)))
        for norm in ("attn_norm", "ff_norm"):
            name = prefix + norm + ".weight"
            if ("input", name) in reference_data.entries:
                require("input", name, ((HIDDEN,),))
        for name in ("key_codes", "value_codes", "key_scale"):
            shapes = ((HEADS, sequence, DIM), (1, HEADS, sequence, DIM)) if name != "key_scale" else (
                (HEADS, sequence), (HEADS, sequence, 1), (1, HEADS, sequence, 1))
            require("input", prefix + "retained_" + name, shapes)
            require("expected", prefix + "cache_" + name, shapes)
        require("expected", prefix + "block_output", ((tokens, HIDDEN), (1, tokens, HIDDEN)))


def put_region(image, regions, address, value, base_address=BASE):
    data = np.frombuffer(value, dtype=np.uint8) if isinstance(value, (bytes, bytearray)) else np.ascontiguousarray(value).view(np.uint8).reshape(-1)
    end = address + len(data)
    if not any(region["base"] <= address < region["limit"] and end <= region["limit"] for region in regions):
        raise ValueError(f"DDR write crosses reserved region: address={address:#x}, bytes={len(data)}")
    if address < base_address or end - base_address > len(image):
        raise ValueError("DDR write exceeds image")
    image[address - base_address:end - base_address] = data


def captured_layer_view(source, layer, last_layer=None):
    """Select a consecutive observed range, preserving raw tensors and KV sets."""
    info = source.metadata
    last_layer = layer if last_layer is None else last_layer
    if not 0 <= layer <= last_layer < 32:
        raise ValueError("layer range must satisfy 0 <= first <= last < 32")
    for options in (info, info.get("forward", {})):
        validate_visibility(options)
    descriptors = []
    for model_layer in range(layer, last_layer + 1):
        matches = [d for d in info.get("layers", [])
                   if d.get("model_layer", d["layer"]) == model_layer]
        if len(matches) != 1:
            raise ValueError(f"capture needs one record for model layer {model_layer}")
        descriptors.append(matches[0])
    prefixes = {f"layer{d['layer']}.": f"layer{local}."
                for local, d in enumerate(descriptors)}
    def local_names(value):
        if isinstance(value, dict):
            return {key: local_names(item) for key, item in value.items()}
        if isinstance(value, list):
            return [local_names(item) for item in value]
        if isinstance(value, str):
            for prefix, replacement in prefixes.items():
                if value.startswith(prefix):
                    return replacement + value[len(prefix):]
        return value
    metadata = copy.deepcopy(info)
    for key in ("connection", "connections", "boundary_scout", "continuation_index"):
        metadata.pop(key, None)
    if last_layer != 31:
        metadata.pop("forward", None)
    records = []
    for local, descriptor in enumerate(descriptors):
        records.append(dict(local_names(descriptor), layer=local, model_layer=layer + local))
    metadata.update(layer_count=len(records), layers=records, model_layer_index=layer,
                    tokens=descriptors[0]["tokens"], reference_data_kind="real_cuda_layer_view")
    entries = {(role, local_names(name)): local_names(copy.deepcopy(entry))
               for (role, name), entry in source.entries.items()
               if not name.startswith("layer") or any(name.startswith(p) for p in prefixes)}
    first_prefix = f"layer{descriptors[0]['layer']}."
    for name in ("hidden", "positions", "activation_bits", "softmax_exp_lut", "softmax_reciprocal_lut"):
        entry = copy.deepcopy(source.entries["input", first_prefix + name])
        entry["name"] = name
        entries["input", name] = entry
    metadata["tensors"] = list(entries.values())
    positions = descriptors[0]["positions"]
    for kind in ("kv_write", "cache_commit"):
        metadata[kind + "_token_indices"] = [positions.index(p) for p in descriptors[0][kind + "_positions"]]
    return ReferenceData(source.index, source.root, metadata=metadata)


def boundary_views(index, payload_root=None, last_layer=1):
    source = ReferenceData(index, payload_root)
    info = source.metadata
    # Per-layer views replace global token mappings, but must preserve visibility semantics.
    for options in (info, info.get("forward", {})):
        validate_visibility(options)
    if info.get("reference_data_kind") == "real_cuda_consecutive_layers":
        if not 1 <= last_layer < 32 or info.get("model_layer_index", 0) != 0 or info.get("layer_count", 0) <= last_layer:
            raise ValueError("boundary source requires L0 through its final deep layer")
        connection = info.get("connections", [info.get("connection", {})])[0]
        if (connection.get("source"), connection.get("destination"), connection.get("mapping")) != (
                "layer0.block_output", "layer1.hidden", "index_select"):
            raise ValueError("boundary source requires an explicit index_select connection")
        source_token_indices = connection["source_token_indices"]
        views = [captured_layer_view(source, 0), captured_layer_view(source, 1, last_layer)]
    else:
        if last_layer != 1:
            raise ValueError("multi-layer boundary needs one consecutive actual capture")
        if "boundary_scout" not in info or "continuation_index" not in info:
            raise ValueError("boundary source requires a scout and continuation index")
        continuation = Path(info["continuation_index"])
        if not continuation.is_absolute():
            continuation = index.parent / continuation
        deep = ReferenceData(continuation, payload_root)
        metadata = copy.deepcopy(info)
        metadata.pop("boundary_scout")
        metadata.pop("continuation_index")
        views = [ReferenceData(index, source.root, metadata=metadata), deep]
        first_positions = source.load("input", "positions").reshape(-1).tolist()
        source_token_indices = [first_positions.index(p) for p in deep.load("input", "positions").reshape(-1)]
    first, deep = views
    n0, n1 = first.metadata["tokens"], deep.metadata["tokens"]
    if [v.metadata.get("model_layer_index", 0) for v in views] != [0, 1]:
        raise ValueError("boundary views must execute L0 then L1")
    if (len(source_token_indices) != n1 or len(set(source_token_indices)) != n1 or
            any(type(r) is not int or not 0 <= r < n0 for r in source_token_indices)):
        raise ValueError("invalid boundary source token selection")
    if first.metadata["sequence"] != deep.metadata["sequence"]:
        raise ValueError("boundary layers require the same sequence")
    if not np.array_equal(first.load("input", "positions").reshape(-1)[source_token_indices], deep.load("input", "positions").reshape(-1)):
        raise ValueError("boundary source tokens differ from deep positions")
    if set(first.metadata.get("kv_write_token_indices", range(n0))) != set(range(n0)):
        raise ValueError("L0 scout must write all query cache tokens")
    committed = set(first.metadata.get("cache_commit_token_indices", range(n0)))
    if committed != set(source_token_indices) and committed != set(range(n0)):
        raise ValueError("L0 committed tokens must match the deep selection or all scout tokens")
    prefix0 = layer_prefix(first, 0, 1)
    expected = first.load("expected", prefix0 + "block_output", "<u2").reshape(n0, HIDDEN)
    if not np.array_equal(expected[source_token_indices], deep.load("input", "hidden", "<u2").reshape(n1, HIDDEN)):
        raise ValueError("L0 selected expected output differs from recorded L1 input")
    return views, source_token_indices


def prepare_boundary(index, output, payload_root=None, ddr_config=None,
                     attention_checkpoints=False, *,
                     scout_ffn_group_batches=0, scout_ffn_fused_product=False,
                     deep_ffn_group_batches=0, deep_ffn_fused_product=False,
                     scout_qkvo_group=False, deep_qkvo_group=False,
                     block_initialization_reference=None, boundary_reference=None,
                     deep_round_token_limit=48, last_layer=1, base_address=BASE):
    if last_layer == 0:
        source = ReferenceData(index, payload_root)
        scout_view = captured_layer_view(source, 0)
        reference = block_initialization_reference or boundary_reference
        if reference is None or (block_initialization_reference is not None and boundary_reference is not None):
            raise ValueError("L0 boundary endpoint requires one actual boundary selection reference")
        reference = json.loads(Path(reference).read_text())
        output = output.resolve()
        output.relative_to(artifact_root())
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        scout_path = prepare(index, output / "scout", ddr_config=ddr_config,
            reference_data=scout_view, base_address=base_address,
            attention_checkpoints=attention_checkpoints,
            ffn_group_batches=scout_ffn_group_batches, ffn_fused_product=scout_ffn_fused_product,
            qkvo_group=scout_qkvo_group)
        scout = json.loads(scout_path.read_text())
        mapping = json.loads((scout_path.parent / "memory_map.json").read_text())
        shutil.copyfile(scout_path.parent / "initial.bin", output / "initial.bin")
        (output / "memory_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
        case = copy.deepcopy(scout)
        case["dramsim3_config"] = "scout/ddr.ini"
        case["expected"] = [dict(item, name="scout."+item["name"], path="scout/"+item["path"])
                            for item in scout["expected"]]
        if "attention_checkpoints" in scout:
            case["attention_checkpoints"]["expected"] = [dict(item,path="scout/"+item["path"])
                for item in scout["attention_checkpoints"]["expected"]]
        case["source"] = dict(kind="boundary_scout", scout=scout["source"], index=str(index))
        selected = reference["records"][0]["expected"]["deep_positions"]["raw"]
        attach_boundary_selection(case, output, reference, scout, None, mapping["memory_map"],
            scout_view.metadata["tokens"], len(selected),
            capture_index=source.metadata["forward"]["capture_index"])
        (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
        return output / "case.json"
    views, source_token_indices = boundary_views(index, payload_root, last_layer)
    if block_initialization_reference is not None and boundary_reference is not None:
        raise ValueError("choose block initialization or ordinary boundary reference")
    selection_data = None
    selection_reference = block_initialization_reference or boundary_reference
    if selection_reference is not None:
        selection_data = json.loads(Path(selection_reference).read_text())
        if (selection_data.get("schema") != "supra-atse-cross-block-token-selection/v1" or
                reference_source_path(selection_data, selection_reference) != Path(index).resolve() or
                len(selection_data.get("records", [])) != 1):
            raise ValueError("boundary reference must identify this capture")
        source = selection_data["records"][0]["inputs"]
        if block_initialization_reference is not None and source.get("shortlist_flags", 0) & 6 != 6:
            raise ValueError("block initialization requires dependency-only sparse selection")
        if (selection_data["records"][0]["expected"]["deep_positions"]["raw"] !=
                views[1].load("input", "positions").reshape(-1).tolist()):
            raise ValueError("boundary selection differs from captured L1 positions")
        if (views[0].load("input", "positions").reshape(-1).tolist() !=
                list(range(views[0].metadata["sequence"])) or
                (block_initialization_reference is not None and (
                    not np.all(views[1].load("input", "activation_bits") == source["deep_activation_bits"]) or
                    set(views[0].metadata["cache_commit_token_indices"]) != set(range(views[0].metadata["tokens"]))))):
            raise ValueError("boundary selection requires full ordered L0 cache and uniform captured deep precision")
        if boundary_reference is not None:
            if (source.get("shortlist_flags", 0) != 0 or "raw" not in source.get("transition_positions", {}) or
                    selection_data["records"][0]["expected"]["deep_bits"]["raw"] != views[1].load("input", "activation_bits").reshape(-1).tolist()):
                raise ValueError("ordinary boundary needs recorded transition positions and matching deep precision")
    n0, n1 = (view.metadata["tokens"] for view in views)
    output = output.resolve()
    output.relative_to(artifact_root())
    if output.exists():
        raise FileExistsError(f"use a fresh output directory: {output}")
    output.mkdir(parents=True)
    first_path = prepare(index, output / "scout", ddr_config=ddr_config,
                         reference_data=views[0], attention_checkpoints=attention_checkpoints,
                         base_address=base_address,
                         ffn_group_batches=scout_ffn_group_batches,
                         ffn_fused_product=scout_ffn_fused_product, qkvo_group=scout_qkvo_group)
    first = json.loads(first_path.read_text())
    first_regions = json.loads((first_path.parent / "memory_map.json").read_text())
    first_hidden = next(region for region in first_regions["memory_map"] if region["name"] == "hidden")
    # A single round reads all input before writing hidden. Multiple rounds
    # preserve all selected input during the K/V sweep before any output write.
    # Reuse that existing RTL schedule, without a host copy or extra prefix.
    arena_base = first_hidden["base"]
    arena_limit = first_hidden["limit"]
    inverse = {original: packed for packed, original in enumerate(first["source"]["packed_logical_to_reference_token"])}
    indices = [inverse[token] for token in source_token_indices]
    deep_path = prepare(index, output / "deep", ddr_config=ddr_config, reference_data=views[1],
                        base_address=align(max(r["limit"] for r in first_regions["memory_map"]), 256),
                        hidden_region=(arena_base, arena_limit),
                        attention_checkpoints=attention_checkpoints and not deep_qkvo_group,
                        source_indices=indices, round_token_limit=deep_round_token_limit,
                        ffn_group_batches=deep_ffn_group_batches,
                        ffn_fused_product=deep_ffn_fused_product, qkvo_group=deep_qkvo_group)
    deep = json.loads(deep_path.read_text())
    deep_regions = json.loads((deep_path.parent / "memory_map.json").read_text())
    if (first["silu_mode"], first.get("silu_segments")) != (deep["silu_mode"], deep.get("silu_segments")):
        raise ValueError("boundary launches require the same compiled SiLU constants")
    # The shared hidden rows retain actual runtime results, never expected tensors.
    with (output / "initial.bin").open("wb") as merged:
        for path, base in ((first_path.parent / "initial.bin", base_address),
                           (deep_path.parent / "initial.bin", deep_regions["base_address"])):
            merged.seek(base - base_address)
            with path.open("rb") as image:
                shutil.copyfileobj(image, merged, 8 << 20)
    regions = [dict(region, name="deep_hidden") if region["name"] == "hidden"
               else region for region in first_regions["memory_map"]]
    regions.extend(dict(region, name="deep." + region["name"]) for region in deep_regions["memory_map"])
    (output / "memory_map.json").write_text(json.dumps(dict(first_regions, memory_map=regions,
        current_ddr_image_bytes=(output / "initial.bin").stat().st_size), indent=2) + "\n")
    second_launch = dict(deep["executions"][0], id=2)
    expected = []
    for directory, case in (("scout", first), ("deep", deep)):
        expected.extend(dict(entry, name=directory + "." + entry["name"],
                             path=directory + "/" + entry["path"],
                             **({"execution_index": 0} if directory == "scout" else {}))
                        for entry in case["expected"])
    case = dict(first, dramsim3_config="scout/ddr.ini", executions=[first["executions"][0], second_launch],
                expected=expected, source=dict(kind="boundary_scout_deep", index=str(index),
                    source_token_indices=source_token_indices, deep_source_indices=indices,
                    scout=first["source"], deep=deep["source"],
                    hidden_connection="in-place DDR read of actual L0 output; existing RTL K/V sweep preserves multi-round input"))
    if attention_checkpoints:
        checks = dict(first["attention_checkpoints"], execution_index=0)
        checks["expected"] = [dict(entry, path="scout/" + entry["path"])
                              for entry in checks["expected"]]
        case["attention_checkpoints"] = [checks]
        if "attention_checkpoints" in deep:
            deep_checks = dict(deep["attention_checkpoints"], execution_index=1)
            deep_checks["expected"] = [dict(entry, path="deep/" + entry["path"])
                                       for entry in deep_checks["expected"]]
            case["attention_checkpoints"].append(deep_checks)
    if selection_data is not None:
        attach_boundary_selection(case, output, selection_data, first, deep, regions,
                                   n0, n1, capture_index=json.loads(Path(index).read_text()).get("forward", {}).get("capture_index", 1))
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return output / "case.json"


def attach_boundary_selection(case, output, reference, scout, deep, regions,
                               scout_tokens, deep_tokens, *, capture_index=1):
    """Connect history and optional live L0 scout scores to RTL selection and L1."""
    source = reference["records"][0]["inputs"]
    sequence = scout["source"]["sequence"]
    image_base = json.loads((output / "memory_map.json").read_text())["base_address"]
    current = set(source["current_positions"]["raw"])
    mandatory = set(source["mandatory_candidate_positions"]["raw"])
    historical_only = source.get("shortlist_flags", 0) & 6 == 6
    bits = source.get("deep_activation_bits")
    context = set(source.get("context_positions", {}).get("raw", []))
    if (scout_tokens != sequence or
            len(source["previous_pending"]["raw"]) != sequence or
            (historical_only and (bits not in (4, 8) or len(source["previous_future_pending"]["raw"]) != sequence))):
        raise ValueError("boundary history/precision does not match full L0 capture")
    expected_positions = reference["records"][0]["expected"]["deep_positions"]["raw"]
    if (len(expected_positions) != deep_tokens or
            reference["records"][0]["expected"]["effective_target"] != deep_tokens or
            len(set(expected_positions)) != deep_tokens):
        raise ValueError("historical candidates must exactly fill the actual deep target")
    def schema(name):
        return json.loads((HARDWARE / "config" / (name + ".json")).read_text())
    def unpack(name, raw):
        return {f["name"]: int.from_bytes(raw[f["offset"]:f["offset"] + int(f["type"][1:]) // 8], "little")
                for f in schema(name)["fields"]}
    def region(name):
        return next(r for r in regions if r["name"] == name)
    base = align(max(r["limit"] for r in regions), 256)
    control = bytearray(768)
    def allocate(data):
        offset = align(len(control), 256)
        control.extend(bytes(offset - len(control)))
        control.extend(data)
        return base + offset, base + len(control)
    pending = bytearray(sequence * 16)
    for p in range(sequence):
        struct.pack_into("<H", pending, p * 16, source["previous_pending"]["raw"][p])
        if historical_only:
            struct.pack_into("<H", pending, p * 16 + 6, source["previous_future_pending"]["raw"][p])
    pending_base, pending_limit = allocate(pending)
    inverse = {original: packed for packed, original in enumerate(scout["source"]["packed_logical_to_reference_token"])}
    table = bytearray(align(sequence, 8) * 8)
    # Scalar records from the captured L0 retain its actual current precision.
    bits_by_position = {}
    if not historical_only:
        with (output / "scout/initial.bin").open("rb") as stream:
            metadata_region = region("token_metadata")
            stream.seek(metadata_region["base"] - image_base)
            metadata = stream.read(metadata_region["limit"] - metadata_region["base"])
        cursor = 0
        for _ in scout["source"]["rounds"]:
            count = metadata[cursor]
            if not count or count > 48:
                raise ValueError("invalid captured L0 metadata token count")
            for ti in range(count):
                row = metadata[cursor+32+ti*16:cursor+48+ti*16]
                bits_by_position[int.from_bytes(row[4:6], "little")] = row[10]
            cursor += struct.unpack_from("<H", metadata, cursor + 8)[0]
    for p in range(sequence):
        value = ((p in current) << 8) | ((p in context) << 11) | ((p in mandatory) << 10) | (p << 23) | (((bits if historical_only else bits_by_position[p]) == 8) << 34) | (inverse[p] << 35)
        struct.pack_into("<Q", table, p * 8, value)
    table_base, table_limit = allocate(table)
    # This output starts empty. No selected positions enter the execution image.
    metadata_base, metadata_limit = allocate(bytes(16384))
    full_metadata_base, full_metadata_limit = allocate(bytes(len(pack_tokens([8] * sequence, list(range(sequence)))[0])))
    masks = 0
    jobs = bytearray()
    if historical_only:
        probability_base, probability_limit = allocate(bytes(32))
        batch_stride, head_stride, round_stride = 128, 768, 24576
    else:
        transition = set(source["transition_positions"]["raw"])
        if not transition <= set(range(sequence)):
            raise ValueError("boundary transition positions exceed the sequence")
        # With no previous transition the model uses current queries
        # to score context keys. Each vector job updates eight table records.
        vector_scout = not transition
        keys = set(range(sequence)) - current if vector_scout else transition
        key_groups = sorted({p // 8 for p in keys})
        masks = sum(1 << g for g in key_groups)
        batch_stride, head_stride = len(key_groups) * 128, len(key_groups) * 128 * 6
        round_stride = head_stride * 32
        probability_base, probability_limit = allocate(bytes(len(scout["source"]["rounds"]) * round_stride))
        cursor = 0
        initialized_groups = set()
        for ri in range(len(scout["source"]["rounds"])):
            count = metadata[cursor]
            for physical in range(count):
                row = metadata[cursor+32+physical*16:cursor+48+physical*16]
                position = int.from_bytes(row[4:6], "little")
                if ((vector_scout and position not in current) or
                        (not vector_scout and position in current)):
                    continue
                for gi, group in enumerate(key_groups):
                    jobs.extend(pack_record(schema("attention_dependency_job"), dict(
                        source_base=probability_base + ri * round_stride + physical // 8 * batch_stride + gi * 128 + physical % 8 * 16,
                        source_limit=probability_limit, head_stride=head_stride,
                        lane_mask=sum(1 << lane for lane in range(8) if group*8+lane in keys),
                        operation=(2 if group in initialized_groups else 6) if vector_scout else (7 if gi == 0 else 3),
                        output_base=table_base + (group * 64 if vector_scout else position * 8),
                        output_limit=table_limit)))
                    initialized_groups.add(group)
            cursor += struct.unpack_from("<H", metadata, cursor + 8)[0]
    jobs_base, jobs_limit = allocate(jobs)
    with (output / "scout/initial.bin").open("rb") as stream:
        stream.seek(scout["executions"][0]["config_address"] - image_base)
        config = unpack("execution_config", stream.read(320))
    config["refresh_configuration_offset"] = 320
    control[:320] = pack_record(schema("execution_config"), config)
    refresh = dict(flags=2, sequence_length=sequence, target_token_count=min(sequence, source["target_token_count"]),
                   required_quota=0, head_stride=MAX_S * DIM, scale_head_stride=MAX_S * 2,
                   metadata_version=1, capture_index=capture_index,
                   table_base=table_base, table_limit=table_limit,
                   metadata_base=metadata_base, metadata_limit=metadata_limit,
                   probability_configuration_offset=176)
    deep_grouped = deep is not None and "qkvo_group" in deep["source"].get("reuse", {}).get("enabled_flags", [])
    if deep_grouped:
        refresh["flags"] |= 0x8000
    if not historical_only:
        refresh.update(required_quota=source["required_candidate_count"], relation_job_base=jobs_base,
                       relation_job_count=len(jobs)//64)
        if not source["keep_global_l0_cache"]:
            refresh["flags"] |= 1
        if source["deep_a8_limit"] >= 0:
            refresh["flags"] |= 32 | source["deep_a8_limit"] << 8
    for name, current_name, retained_name in (("k", "cache_k", "retained_k"),
                                            ("v", "cache_v", "retained_v"),
                                            ("scale", "cache_scale", "retained_scale")):
        available = {r["name"] for r in regions}
        retained_name = retained_name if retained_name in available else current_name
        for prefix, key in (("source", current_name), ("destination", retained_name)):
            r = region(key)
            refresh[prefix + "_" + name + "_base"] = r["base"]
            refresh[prefix + "_" + name + "_limit"] = r["limit"]
    control[320:496] = pack_record(schema("token_refresh_config"), refresh)
    probability = dict(output_base=probability_base, output_limit=probability_limit,
                       batch_stride_bytes=batch_stride, head_stride_bytes=head_stride, round_stride_bytes=round_stride,
                       layer_mask=1, query_end=0 if historical_only else sequence,
                       shortlist_configuration_base=base + 576)
    probability.update({"key_groups" + str(i): (masks >> (64*i)) & ((1 << 64)-1) for i in range(4)})
    control[496:576] = pack_record(schema("attention_probability_config"), probability)
    shortlist = dict(pending_base=pending_base, pending_limit=pending_limit,
                     shortlist_token_count=source["shortlist_token_count"], flags=source["shortlist_flags"],
                     relative_score_floor_bf16=source.get("relative_score_floor_bf16",0),
                     protected_begin=source.get("protected_begin",0), protected_end=source.get("protected_end",0))
    control[576:608] = pack_record(schema("cross_block_shortlist_config"), shortlist)
    control.extend(bytes(align(len(control), 256) - len(control)))
    (output / "boundary_control.bin").write_bytes(control)
    case.setdefault("initial_segments", []).append(dict(address=base, bytes=len(control), path="boundary_control.bin"))
    regions.append(dict(name="boundary_control", base=base, limit=base + len(control), access="read_write"))
    map_path = output / "memory_map.json"
    memory_map = json.loads(map_path.read_text())
    memory_map["memory_map"] = regions
    map_path.write_text(json.dumps(memory_map, indent=2) + "\n")
    case["executions"][0]["config_address"] = base
    if deep is not None:
        second = case["executions"][1]
        second["handoff_actions"] = [dict(kind="execution_from_metadata", config_address=second["config_address"],
            metadata_address=metadata_base, metadata_capacity=metadata_limit - metadata_base,
            layout_address=region("deep.token_metadata")["base"],
            layout_capacity=region("deep.token_metadata")["limit"] - region("deep.token_metadata")["base"],
            current_begin=min(current), current_end=max(current)+1)]
    case.setdefault("provenance", {})["boundary_control"] = dict(
        capture_index=capture_index, sequence=sequence, current_begin=min(current),
        historical_only=historical_only, table=[table_base, table_limit],
        pending=[pending_base, pending_limit], metadata=[metadata_base, metadata_limit],
        full_metadata=[full_metadata_base, full_metadata_limit],
        probability=[probability_base, probability_limit], jobs=[jobs_base, jobs_limit],
        refresh_address=base+320, probability_config_address=base+496,
        scout_metadata=[config["token_metadata_base"], config["token_metadata_limit"]])
    raw = None
    if deep is not None:
        # Compare the published descriptor, including its actual hidden row mapping.
        target = region("deep.token_metadata")
        with (output / "initial.bin").open("rb") as stream:
            stream.seek(target["base"] - image_base)
            raw = stream.read(sum(32 + r["tokens"] * 16 + align(r["tokens"], 16) for r in deep["source"]["rounds"]))
    if not historical_only or deep is None:
        import subprocess
        from prepare_handoff_testcase import build_reference
        actual_bits = ([bits] * deep_tokens if historical_only else
                       reference["records"][0]["expected"]["deep_bits"]["raw"])
        encoded = subprocess.check_output([str(build_reference(output)), "metadata"], text=True,
            input=json.dumps(dict(sequence=sequence, capture=capture_index, tokens=[dict(position=p,
                source_index=inverse[p], source=0, bits=b)
                for p,b in zip(expected_positions,actual_bits)])))
        raw = bytes(json.loads(encoded)["metadata"])
        if not historical_only:
            scores_path = "boundary_scores.expected.bin"
            np.asarray(source["score_q8"]["raw"],dtype=np.uint8).tofile(output / scores_path)
            case["expected"].append(dict(name="boundary.scout_scores",address=table_base,
                bytes=sequence,path=scores_path,element_bytes=1,stride_bytes=8))
    if deep_grouped:
        raw = bytearray(raw)
        rounds, cursor = [], 0
        while cursor < len(raw):
            count = raw[cursor]
            a8 = sum(raw[cursor+32+i*16+10] == 8 for i in range(count))
            rounds.append(dict(tokens=count, a4=count-a8, a8=a8, groups=raw[cursor+1]))
            raw[cursor+3] &= ~4
            cursor += struct.unpack_from("<H", raw, cursor+8)[0]
        case["source"]["deep"]["reuse"]["qkvo_groups"] = mark_qkvo_groups(raw, rounds, online=True)
    raw = bytearray(raw)
    cursor = 0
    while cursor < len(raw):
        struct.pack_into("<I", raw, cursor+20, capture_index)
        cursor += struct.unpack_from("<H", raw, cursor+8)[0]
    (output / "boundary_metadata.expected.bin").write_bytes(raw)
    case["expected"].append(dict(name="boundary.actual_metadata", address=metadata_base,
                                 bytes=len(raw), path="boundary_metadata.expected.bin"))
    case["source"]["selection"] = dict(kind="rtl_dependency_only_block_initialization" if historical_only else "rtl_live_boundary",
        reference=reference["source_index"], fresh_l0_score_jobs=len(jobs)//64,
        keep_global_l0_cache=True if historical_only else source["keep_global_l0_cache"])


def require_regular_attention(case):
    mapping = case.get("attention_checkpoints", {})
    if isinstance(mapping, list):
        mapping = max(mapping, key=lambda check: check.get("execution_index", 0)) if mapping else {}
    if not mapping.get("physical_to_reference_token") or not mapping.get("token_positions"):
        raise ValueError("attention_checkpoints is missing a nonempty "
                         "physical_to_reference_token or token_positions mapping")
    return mapping


def pack_future_selection_records(inputs):
    """Convert captured future rows to 16-byte UAPS records.

    Generation initializes unobserved history to -1. Observed zero confidence
    remains valid, including suppressed predictions (generation/engine.py).
    """
    import torch

    history = inputs["next_last_confidence"]
    service = inputs["next_service_count"]
    raw_fields = {
        name: inputs[name].to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<u2")
        for name in ("next_priority", "next_dependency_score")
    }
    records = bytearray()
    for i in range(inputs["next_row_bits"].numel()):
        flags = int(inputs["next_unresolved"][i]) | int(inputs["next_tentative"][i]) << 1 | int(inputs["next_row_bits"][i] == 8) << 2
        confidence = float(history[i]) if history is not None else -1.0
        if confidence != -1.0 and not 0.0 <= confidence <= 1.0:
            raise ValueError("future confidence history must be unobserved (-1) or in [0,1]")
        observed_confidence = confidence >= 0.0
        confidence_raw = int(history[i].to(torch.bfloat16).view(torch.int16)) & 0xffff if observed_confidence else 0
        records.extend(struct.pack("<HHiHBBHH", int(raw_fields["next_priority"][i]), flags,
            int(service[i]) if service is not None else 0,
            confidence_raw, int(observed_confidence), 0,
            int(raw_fields["next_dependency_score"][i]), 0))
    return records


def attach_full_sequence_control(case_path, control_index, case, observed, states, prediction,
                                 first_config, head_config, post, payload_root=None):
    """Join baseline layers and head; preserve tokens for the next full forward."""
    from prepare_handoff_testcase import read_record
    output = case_path.parent
    before = states["before_forward"]
    sequence = before["total_length"]
    if (before.get("decoding_mode") != "fixed_k" or before["packed_state"] is not None or
            before["cross_block_prefix_state"] is not None or before["dynamic_block_lookahead"]):
        raise ValueError("full-sequence control requires baseline fixed-k without packed/future state")
    if (first_config["start_layer"] + first_config["layer_count"] != 32 or
            first_config["total_token_count"] != sequence or first_config["sequence_length"] != sequence or
            prediction["input_positions"].reshape(-1).tolist() != list(range(sequence)) or
            prediction["row_bits"].reshape(-1).tolist() != [8] * sequence):
        raise ValueError("baseline requires all sequence positions at A8 through L31")
    if first_config["refresh_configuration_offset"] or first_config["joint_configuration_offset"]:
        raise ValueError("baseline must not run dependency refresh or future selection")
    head_reference_path = Path(case["provenance"]["control_reference"])
    head_reference = json.loads(head_reference_path.read_text())
    head_source = json.loads(reference_source_path(head_reference, head_reference_path).read_text())
    for name in ("dataset", "split", "sample_id", "sample_doc_hash", "model_arguments"):
        if head_source["provenance"].get(name) != observed["provenance"].get(name):
            raise ValueError("head control and full-sequence capture differ in " + name)
    if (head_reference["capture_index"] != prediction["capture_index"] or
            head_reference["sequence_length"] != sequence or not head_reference["config"].get("transfer_only")):
        raise ValueError("head control differs from the baseline event")
    mapping_path = output / case["memory_map"]
    mapping = json.loads(mapping_path.read_text())
    base = align(max(region["limit"] for region in mapping["memory_map"]), 256)
    first_config = dict(first_config, flags=first_config["flags"] | 4)
    for key, value in head_config.items():
        if key.startswith(("forward_postprocess_", "prediction_")) or key in (
                "generation_block_count", "suppressed_token_count"):
            first_config[key] = value
    data = bytearray(pack_record(json.loads((HARDWARE / "config/execution_config.json").read_text()), first_config))
    data.extend(bytes(align(len(data), 256) - len(data)))
    table_base = base + len(data)
    # This is the initial input token table. Handoff overwrites only the actual
    # completed block, so updates to earlier blocks survive later forwards.
    data.extend(np.asarray(before["tokens"].reshape(-1).tolist(), dtype="<u4").tobytes())
    table_limit = base + len(data)
    data.extend(bytes(align(len(data), 256) - len(data)))
    metadata_base = base + len(data)
    metadata_bytes = len(pack_tokens(np.full(sequence, 8), np.arange(sequence))[0])
    data.extend(bytes(metadata_bytes))
    post_config = read_record("forward_postprocess_config", post)
    if post_config["current_state_limit"] - post_config["current_state_base"] != 32 * 32:
        raise ValueError("baseline requires exactly one current state block")
    case["executions"] = [dict(case["executions"][0],
        expected_post_block_completions=case["executions"][-1]["expected_post_block_completions"], config_address=base,
                               max_cycles=100000000 * first_config["layer_count"])]
    case["head_checkpoints"]["execution_index"] = 0
    # DRAMSim3 transfers complete 32-byte transactions, including the final record.
    data.extend(bytes(align(len(data), 32) - len(data)))
    (output / "regular_control.bin").write_bytes(data)
    case.setdefault("initial_segments", []).append(dict(address=base, bytes=len(data), path="regular_control.bin"))
    mapping["memory_map"].append(dict(name="regular_control", base=base, limit=base + len(data), access="read_write"))
    state_base = post_config["next_state_base"]
    case["provenance"]["regular_control"] = dict(index=str(control_index), payload_root=str(Path(payload_root).resolve()) if payload_root else None, mode="baseline_full_sequence",
        start_layer=first_config["start_layer"], layer_count=first_config["layer_count"],
        metadata_base=metadata_base, metadata_limit=base + len(data),
        storage=dict(token_table=[table_base, table_limit], state=[state_base, state_base + 32 * 32],
            input_state=[post_config["current_state_base"], post_config["current_state_limit"]], joint_result=[0, 0]))
    if states.get("request_end") is True:
        case["provenance"]["regular_control"]["closeout"] = True
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    case_path.write_text(json.dumps(case, indent=2) + "\n")
    return case_path


def attach_regular_control(case_path, control_index, *, payload_root=None):
    """Connect layer probabilities and head state to pending updates and selection."""
    import subprocess
    import torch
    from hardware_adapter.control_reference_data import read_captured_control, captured_feature1_reference
    from prepare_handoff_testcase import build_reference, read_record
    case_path, control_index = Path(case_path), Path(control_index)
    output = case_path.parent
    case = json.loads(case_path.read_text())
    if len(case["executions"]) < 2 or "control_reference" not in case.get("provenance", {}):
        raise ValueError("regular connection requires prepared L31 plus observed Feature2 head")
    attention = require_regular_attention(case)
    head_base = case["executions"][-1].get("config_address", BASE)
    with (output / case["ddr_image"]).open("rb") as stream:
        head_config = read_record("execution_config", stream.read(320))
        post_offset = head_config["forward_postprocess_configuration_base"] - head_base
        if post_offset < 0:
            raise ValueError("regular connection requires the head post configuration in its DDR image")
        stream.seek(post_offset)
        post = stream.read(192)
    if len(post) != 192 or not read_record("forward_postprocess_config", post)["flags"] & 1:
        raise ValueError("regular pending requires post action-confidence history publication (flags bit0)")
    observed, states, prediction = read_captured_control(control_index, payload_root=payload_root)
    first = dict(case["executions"][-2])
    segment = next(item for item in reversed(case["initial_segments"])
                   if item["address"] <= first["config_address"] < item["address"] + item["bytes"])
    with (output / segment["path"]).open("rb") as stream:
        stream.seek(first["config_address"] - segment["address"])
        first_config = read_record("execution_config", stream.read(320))
    first_layer, layer_count = first_config["start_layer"], first_config["layer_count"]
    if observed["generation_config"].get("full_sequence_recompute", False):
        return attach_full_sequence_control(case_path, control_index, case, observed, states, prediction,
                                            first_config, head_config, post, payload_root)
    initial = prediction["forward_kind"] in ("full_sequence", "boundary_refresh")
    boundary = prediction["forward_kind"] == "boundary_refresh"
    if first_layer + layer_count != 32 or prediction["forward_kind"] not in ("local_block", "full_sequence", "boundary_refresh", "local_confirmation", "local_forced_finish"):
        raise ValueError("regular connection requires actual consecutive layers ending at L31")
    prior = states.get("before_first_layer_dependency" if first_layer < 31 else "before_l31_dependency")
    if first_layer and not boundary and (prior is None or prior["attn_monitor_dependency_layer_count"] != first_layer):
        raise ValueError("regular connection lacks the dependency reduction before its first layer")
    before, after = states["before_forward"], states["after_postprocess"]
    local_before = states["before_postprocess"]["packed_state"] if initial else before["packed_state"]
    request_end = states.get("request_end") is True
    closeout = request_end or states["next_forward"]["block_index"] != before["block_index"]
    if closeout:
        final = states.get("current_state_at_request_end" if request_end else "current_state_at_next_forward")
        if final is None or bool((final["block_state"]["state"] != 2).any()):
            raise ValueError("terminal connection requires actual completed current state")
    refresh_reference = (dict(target_active_token_count=local_before["target_active_rows"],
                   records=[dict(context_a8=observed["generation_config"]["packed_attention_context_a8_rows"])])
              if closeout else captured_feature1_reference(control_index, payload_root=payload_root))
    if before["cross_block_dependency_policy"] != "current_keys_committed_rows":
        raise ValueError("this connection requires current-key consumption")
    head_reference = json.loads(Path(case["provenance"]["control_reference"]).read_text())
    head_source = json.loads(reference_source_path(head_reference,
        Path(case["provenance"]["control_reference"])).read_text())
    for key in ("dataset", "split", "sample_id", "sample_doc_hash", "model_arguments"):
        if head_source["provenance"].get(key) != observed["provenance"].get(key):
            raise ValueError("head control and dependency capture differ in " + key)
    if (head_reference["capture_index"] != prediction["capture_index"] or
            head_reference["sequence_length"] != before["total_length"]):
        raise ValueError("head control and dependency event differ")
    sequence, begin, end = before["total_length"], before["region_start"], before["region_end"]
    current, current_end = before["block_start"], before["block_end"]
    count = end - begin
    if not 1 <= count <= 96 or current_end - current != 32:
        raise ValueError("regular dependency region exceeds the hardware range")
    mapping_path = output / case["memory_map"]
    mapping = json.loads(mapping_path.read_text())
    regions = mapping["memory_map"]
    base = align(max(r["limit"] for r in regions), 256)
    data = bytearray(2048)
    def schema(name):
        return json.loads((HARDWARE / "config" / (name + ".json")).read_text())
    def alloc(payload):
        offset = align(len(data), 256)
        data.extend(bytes(offset - len(data)))
        data.extend(payload)
        return base + offset, base + len(data)
    def expected(name, address, payload, element_bytes=None, stride_bytes=None):
        path = "expected.regular_" + name + ".bin"
        (output / path).write_bytes(payload)
        entry = dict(name="regular." + name, address=address, bytes=len(payload), path=path)
        if element_bytes is not None:
            entry.update(element_bytes=element_bytes, stride_bytes=stride_bytes)
        case["expected"].append(entry)
    def raw(value):
        return value.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<u2")
    def padded_relation(value, stride):
        value = raw(value)
        out = np.zeros((value.shape[0], stride // 2), dtype="<u2")
        out[:, :value.shape[1]] = value
        return out
    relation = padded_relation(local_before["dependency"], 256)
    if initial:
        relation.fill(0)
    query = attention["token_positions"]
    if first_layer and not boundary:
        prior_query = prior["attn_monitor_dependency_query_positions"].reshape(-1).tolist()
        if set(prior_query) != {p for p in query if begin <= p < end}:
            raise ValueError("executed query differs from its preceding dependency reduction")
        relation[np.array(prior_query) - begin, :count] = raw(prior["attn_monitor_dependency_max"])[0]
    relation_base, relation_limit = alloc(relation.tobytes())
    cross_value = before["cross_block_prefix_state"]["relation"]
    cross = np.zeros((sequence, 32), dtype="<u2") if cross_value is None else padded_relation(cross_value, 64)
    if first_layer and not boundary:
        prefix_query = prior["attn_monitor_prefix_dependency_query_positions"].reshape(-1).tolist()
        cross[prefix_query] = raw(prior["attn_monitor_prefix_dependency_max"])[0]
    cross_base, cross_limit = alloc(cross.tobytes())
    expected_relation = padded_relation(after["packed_state"]["dependency"], 256)
    if closeout and not initial:
        # Regular closeout skips the state update; RTL still reduces the executed rows.
        profile = states["before_postprocess"]["packed_profile"]
        positions = profile["query_positions"].reshape(-1).numpy()
        expected_relation[positions - begin, :count] = raw(profile["dependency_mean"])[0]
    expected("dependency", relation_base, expected_relation.tobytes())
    expected("cross_relation", cross_base, padded_relation(after["cross_block_prefix_state"]["relation"], 64).tobytes())
    pending = bytearray(count * 16)
    for row, value in enumerate(raw(local_before["pending"]).reshape(-1)):
        struct.pack_into("<H", pending, row * 16, value)
        pending[row * 16 + 10] = 0 if initial else int(local_before["refresh"][row])
    pending_base, pending_limit = alloc(pending)
    cross_pending = bytearray(sequence * 16)
    for field, name in enumerate(("pending", "actual_remask_pending", "actual_remask_epoch_pending",
                                   "future_pending", "future_actual_remask_pending")):
        original, target = before["cross_block_prefix_state"][name], after["cross_block_prefix_state"][name]
        if original is None or target is None:
            raise ValueError("paired pending requires actual values for every vector")
        for row, value in enumerate(raw(original).reshape(-1)):
            struct.pack_into("<H", cross_pending, row * 16 + field * 2, value)
    cross_pending_base, cross_pending_limit = alloc(cross_pending)
    # Each expected vector is compared separately; runtime refresh markers and
    # new-invalidation fields retain their own hardware meaning.
    local_result = local_before if closeout else after["packed_state"]
    expected("pending", pending_base, raw(local_result["pending"]).tobytes(), 2, 16)
    cross_target = np.stack([raw(after["cross_block_prefix_state"][name]).reshape(-1)
        for name in ("pending", "actual_remask_pending", "actual_remask_epoch_pending",
                     "future_pending", "future_actual_remask_pending")], axis=1)
    expected("cross_pending", cross_pending_base, cross_target.tobytes(), 10, 16)
    target_x2 = int(refresh_reference["target_active_token_count"] * 2)
    def budget(state):
        steps, used = state["regular_steps"], state["cumulative_active_rows"]
        return pack_record(schema("in_block_refresh_budget"), dict(target_token_count_x2=target_x2,
            region_start=begin, regular_steps=steps, cumulative_active_token_count=used,
            token_count_x2_credit=(target_x2 * steps - 2 * used) & 0xffffffff))
    budget_base, budget_limit = alloc(budget(local_before))
    expected("budget", budget_base, budget(local_result))
    table = bytearray(align(sequence, 8) * 8)
    tokens_before = before["tokens"].reshape(-1).tolist()
    for position, token in enumerate(tokens_before):
        struct.pack_into("<Q", table, position * 8, int(token) << 35)
    table_base, table_limit = alloc(table)
    history = bytearray(16 + sequence)
    struct.pack_into("<HHII", history, 0, current, sequence, prediction["capture_index"],
        sum(1 << row for row, changed in enumerate(before["block_transition_union"].reshape(-1).tolist()) if changed))
    history[16:] = bytes(before["cache_refresh_due"].reshape(-1).tolist())
    history_base, history_limit = alloc(history)
    metadata_base, metadata_limit = alloc(bytes(4096))
    # Encode the independently observed positions for comparison.
    packed = bytes(4096)
    if not closeout:
        tool = build_reference(output)
        following = states["next_layer_inputs"]
        positions = following["query_position_ids"].reshape(-1).tolist()
        bits = following["activation_bits"].reshape(-1).tolist()
        tokens_after = states["next_forward"]["tokens"].reshape(-1).tolist()
        packed = json.loads(subprocess.check_output([str(tool), "metadata"], text=True,
            input=json.dumps(dict(tokens=[dict(position=p,source_index=tokens_after[p],source=1,bits=b)
                                         for p,b in zip(positions,bits)], sequence=sequence,
                                  capture=prediction["capture_index"] + 1))))["metadata"]
        if observed["generation_config"]["dynamic_block_lookahead"]:
            # Add the original forward's KV suppression to the independently packed
            # expected metadata. It is indexed by physical entry, not logical order.
            cursor = 0
            while cursor < len(packed):
                mask = 0
                for row in range(packed[cursor]):
                    position = int.from_bytes(bytes(packed[cursor+36+row*16:cursor+38+row*16]), "little")
                    if position >= current_end and states["next_forward"]["source_a_handoff_pending"][0,position]:
                        mask |= 1 << row
                packed[cursor+24:cursor+30] = mask.to_bytes(6, "little")
                cursor += int.from_bytes(bytes(packed[cursor+8:cursor+10]), "little")
    expected("selected_metadata", metadata_base, bytes(4096) if closeout else bytes(packed))
    # Capture the union of tri-region keys, which also includes current keys.
    first_group, groups = begin // 8, (end + 7) // 8 - begin // 8
    batch_stride, head_stride = groups * 128, groups * 128 * 6
    rounds = attention["physical_to_reference_token"]
    regular_rounds = rounds
    ref_positions = attention["token_positions"]
    if first_config["flags"] & (1 << 13):
        position_rows = {position: row for row, position in enumerate(ref_positions)}
        with (output / segment["path"]).open("rb") as stream:
            stream.seek(first_config["token_metadata_base"] - segment["address"])
            regular_base, regular_limit, l31_base, l31_limit = struct.unpack("<4Q", stream.read(32))
            stream.seek(regular_base - segment["address"])
            normal_metadata = stream.read(regular_limit - regular_base)
        regular_rounds, cursor = [], 0
        while cursor < len(normal_metadata):
            regular_rounds.append([position_rows[struct.unpack_from("<H", normal_metadata, cursor+36+row*16)[0]]
                                   for row in range(normal_metadata[cursor])])
            cursor += struct.unpack_from("<H", normal_metadata, cursor+8)[0]
    prob_base, prob_limit = alloc(bytes(max(len(rounds),len(regular_rounds)) * head_stride * 32))
    def relation_jobs(layout):
        jobs = bytearray()
        for ri, token_map in enumerate(layout):
            for physical, reference in enumerate(token_map):
                position = ref_positions[reference]
                if position not in query:
                    raise ValueError("query map differs from its dependency snapshot")
                for group in range(groups):
                    key_start = (first_group + group) * 8
                    for low, high, destination, limit in (
                            (begin, end, relation_base + (position - begin) * 256, relation_limit),
                            (current, current_end, cross_base + position * 64, cross_limit)):
                        if limit == relation_limit and not begin <= position < end:
                            continue
                        mask = sum(1 << lane for lane in range(8) if low <= key_start + lane < high)
                        if mask:
                            jobs.extend(pack_record(schema("attention_dependency_job"), dict(
                                source_base=prob_base + ri * head_stride * 32 + physical // 8 * batch_stride + group * 128 + physical % 8 * 16,
                                source_limit=prob_limit, head_stride=head_stride, lane_mask=mask,
                                operation=1 | (16 if first_layer and layer_count > 1 and not boundary else 0),
                                output_base=destination + (key_start - low) * 2, output_limit=limit)))
        return jobs
    jobs = relation_jobs(regular_rounds)
    jobs_base, _ = alloc(jobs)
    execution_extension = bytearray(32)
    if first_config["flags"] & (1 << 13):
        l31_jobs = relation_jobs(rounds)
        l31_jobs_base, _ = alloc(l31_jobs)
        struct.pack_into("<QI", execution_extension, 0, l31_jobs_base, len(l31_jobs)//64)
    first_config["refresh_configuration_offset"] = 320
    first_config["flags"] |= 4
    for key, value in head_config.items():
        if key.startswith(("forward_postprocess_", "prediction_")) or key in (
                "generation_block_count", "suppressed_token_count"):
            first_config[key] = value
    common = dict(sequence_length=sequence, target_token_count=count,
        head_stride=MAX_S * DIM, scale_head_stride=MAX_S * 2,
        table_base=table_base, table_limit=table_limit, metadata_version=1,
        capture_index=prediction["capture_index"] + 1)
    # Reuse probability storage only after each selected layer has reduced it.
    # Partial ranges merge the captured prefix; a full range starts a new reduction.
    masks = sum(1 << g for g in range(first_group, first_group + groups))
    probability = dict(output_base=prob_base, output_limit=prob_limit,
        batch_stride_bytes=batch_stride, head_stride_bytes=head_stride,
        round_stride_bytes=head_stride * 32, layer_mask=(1 << 31) if boundary else ((1 << layer_count) - 1) << first_layer,
        query_begin=0, query_end=sequence)
    probability.update({"key_groups" + str(i): (masks >> (64 * i)) & ((1 << 64) - 1) for i in range(4)})
    # Head publishes the current block state before the selector reads it.
    state_base = read_record("forward_postprocess_config", post)["next_state_base"]
    future = head_reference.get("future")
    common_pending = dict(change_base=state_base, change_limit=state_base + (64 if future else 32) * 32)
    cross_config = dict(common_pending, token_count=sequence, keys=32, block_end=current_end,
        relation_row_shift=6, flags=67 | (32 if prediction["forward_kind"] == "boundary_refresh" else 0), relation_base=cross_base, relation_limit=cross_limit,
        pending_base=cross_pending_base, pending_limit=cross_pending_limit)
    regular_config = dict(common_pending, token_count=count, keys=count, block_end=current_end,
        relation_row_shift=8, flags=6, relation_base=relation_base, relation_limit=relation_limit,
        pending_base=pending_base, pending_limit=pending_limit,
        regular_budget_base=budget_base, regular_budget_limit=budget_limit)
    refresh = dict(common, flags=2 | 4 | 8 | 128 | (refresh_reference["records"][0]["context_a8"] << 8),
        metadata_base=metadata_base, metadata_limit=metadata_limit, pending_configuration_offset=288,
        probability_configuration_offset=208, relation_job_base=jobs_base, relation_job_count=len(jobs) // 64)
    if observed["generation_config"].get("feature3_precision_policy") == "all_a8":
        refresh["flags"] = 2 | 4 | 8 | 16 | 128
    if future:
        config = observed["generation_config"]
        next_joint = states.get("next_joint_selection")
        current_joint = states.get("current_joint_selection")
        if (next_joint is None and not closeout) or (current_joint is None and not initial):
            raise ValueError("future control window requires recorded current and next joint selector calls")
        joint_result_base, joint_result_limit = alloc(bytes(16))
        attempt_base = attempt_limit = 0
        counts = before["next_source_b_attempts"]
        if counts is not None:
            attempts = bytearray(144)
            struct.pack_into("<IHHII", attempts, 0, 0x31425441, current_end, 0,
                             prediction["capture_index"], sum(1 << row for row in
                                 (current_joint["result"]["added_local_positions"].tolist() if current_joint else [])))
            for row, value in enumerate(counts.tolist()):
                struct.pack_into("<i", attempts, 16+row*4, value)
            attempt_base, attempt_limit = alloc(attempts)
        def extension(step, live):
            return dict(state_base=attempt_base, state_limit=attempt_limit,
                max_attempts=max(0,config["dynamic_block_source_b_max_attempts"]) if
                    config["dynamic_block_source_b_max_attempts"] >= 0 else 2147483647,
                retry_min_confidence_bf16=int(raw(torch.tensor(config["dynamic_block_source_b_retry_min_confidence"]))),
                prediction_target=max(0,config["dynamic_block_target_prediction_rows"]),
                source_b_flags=int(config["dynamic_block_source_b_dependency_tie_rank"]) |
                    int(config["dynamic_block_source_b_a4_only"]) << 1,
                min_reuse_score_bf16=int(raw(torch.tensor(config["dynamic_block_min_reuse_score"]))),
                state_control=(1 | (int(config["dynamic_block_allow_deferred_verification"]) << 1)) if live else 0,
                max_current_unresolved=config["dynamic_block_max_current_unresolved"] if live else 0,
                max_handoff_tokens=config["dynamic_block_max_handoff_verification_rows"] if live else 0,
                admission_budget=config["dynamic_block_next_admission_budget"] if live else 0,
                block_step_index=step)
        def joint_bytes(selection, live, explicit_base=0, explicit_limit=0, future_table=0, future_limit=0):
            inputs = selection["inputs"]
            record = bytearray(pack_record(schema("uaps_config"), dict(
                base_token_count=0 if live else inputs["base_positions"].numel(),
                future_token_count=inputs["next_row_bits"].numel(), next_block_start=current_end,
                max_next_tokens=config["dynamic_block_max_next_rows"],
                priority_control=0xe0 if live else 0,
                base_table_base=explicit_base, base_table_limit=explicit_limit,
                future_table_base=future_table, future_table_limit=future_limit,
                result_base=joint_result_base, result_limit=joint_result_limit)))
            struct.pack_into("<H", record, 6, 96)
            return record + pack_record(schema("uaps_attempt_config"),extension(inputs["step_index"],live))
        refresh["target_token_count"] = config["dynamic_block_target_joint_rows"]
        first_config["joint_configuration_offset"] = 0
        if not closeout:
            first_config["joint_configuration_offset"] = 800
            data[800:896] = joint_bytes(next_joint, True)
        def joint_expected(selection):
            result = selection["result"]
            progress, added = result["progress_local_positions"].tolist(), result["added_local_positions"].tolist()
            def slots(name):
                residency = result[name]
                return residency["a4_rows"] + 2 * residency["a8_rows"]
            return pack_record(schema("uaps_result"), dict(
                future_prediction_mask=sum(1 << p for p in progress),
                added_future_token_mask=sum(1 << p for p in added),
                future_prediction_count=len(progress), added_future_token_count=len(added),
                base_activation_slots=slots("base_residency"),
                joint_activation_slots=slots("joint_residency"),
                next_pass_activation_slots=slots("next_step_verification_residency")))
        expected("joint_result", joint_result_base, bytes(16) if closeout else joint_expected(next_joint))
        # Restore the captured entry state once. Later forwards carry the
        # selector result directly through the normal execution path.
        result = current_joint["result"] if current_joint else {}
        progress = set(result["progress_local_positions"].tolist()) if result else set()
        added = set(result["added_local_positions"].tolist()) if result else set()
        source_a_mask = sum(1 << row for row in progress - added)
        struct.pack_into("<IIHHII", execution_extension, 12,
                         (3 if current_joint else 0), source_a_mask, current_end, 0, prediction["capture_index"], 0)
    for offset, name, values in ((0,"execution_config",first_config),(320,"token_refresh_config",refresh),
            (528,"attention_probability_config",probability),(608,"refresh_score_config",cross_config),
            (688,"refresh_score_config",regular_config)):
        packed_record = pack_record(schema(name),values)
        data[offset:offset + len(packed_record)] = packed_record
    struct.pack_into("<H", data, 320+6, 208)
    data[496:528] = execution_extension
    first["config_address"] = base
    for action in first.get("handoff_actions", []):
        action["config_address"] = base
    final_execution = len(case["executions"]) - 2
    case["executions"] = case["executions"][:-2] + [dict(first, max_cycles=100000000 * layer_count,
        expected_post_block_completions=case["executions"][-1]["expected_post_block_completions"])]
    case["head_checkpoints"]["execution_index"] = final_execution
    for item in case["expected"]:
        if item.get("execution_index", final_execution) > final_execution:
            item["execution_index"] = final_execution
    if first_layer and layer_count > 1 and not boundary:
        case.setdefault("required_features", []).append("dependency_merge_initial")
    # DRAMSim3 transfers complete 32-byte transactions, including the final record.
    data.extend(bytes(align(len(data), 32) - len(data)))
    (output / "regular_control.bin").write_bytes(data)
    case.setdefault("initial_segments", []).append(dict(address=base, bytes=len(data), path="regular_control.bin"))
    regions.append(dict(name="regular_control",base=base,limit=base + len(data),access="read_write"))
    case["provenance"]["regular_control"] = dict(index=str(control_index),
        payload_root=str(Path(payload_root).resolve()) if payload_root else None,
        start_layer=first_layer, layer_count=layer_count,
        block_start=current, capture_index=prediction["capture_index"], request_end=request_end,
        closeout_kind={"local_confirmation": 1, "local_forced_finish": 2}.get(prediction["forward_kind"], 0),
        metadata_base=metadata_base, metadata_limit=metadata_limit,
        storage=dict(relation=[relation_base, relation_limit],
            cross_relation=[cross_base, cross_limit], pending=[pending_base, pending_limit],
            cross_pending=[cross_pending_base, cross_pending_limit], budget=[budget_base, budget_limit],
            token_table=[table_base, table_limit],
            history=[history_base, history_limit],
            executed_metadata=[l31_base, l31_limit] if first_config["flags"] & (1 << 13) else
                [first_config["token_metadata_base"], first_config["token_metadata_limit"]],
            event=[read_record("forward_postprocess_config", post)["forward_event_base"],
                   read_record("forward_postprocess_config", post)["forward_event_base"] + 64],
            state=[state_base, state_base + (64 if future else 32) * 32],
            input_state=[read_record("forward_postprocess_config", post)["current_state_base"],
                         read_record("forward_postprocess_config", post)["current_state_limit"]],
            attempts=[attempt_base, attempt_limit] if future else [0, 0],
            joint_result=[joint_result_base, joint_result_limit] if future else [0, 0]))
    if closeout:
        case["provenance"]["regular_control"]["closeout"] = True
    mapping_path.write_text(json.dumps(mapping,indent=2) + "\n")
    case_path.write_text(json.dumps(case,indent=2) + "\n")
    return case_path


def prepare(index, output, payload_root=None, ddr_config=None, *, reference_data=None,
            base_address=BASE, hidden_region=None, source_indices=None,
            last_layer_output_subset=False, round_token_limit=48, attention_checkpoints=False,
            ffn_group_batches=0, ffn_fused_product=False, ffn_down_pair=False,
            attention_output_pair=False, attention_pair=False, kv_pair=False,
            qkvo_group=False):
    reference_data = reference_data or ReferenceData(index, payload_root)
    info = reference_data.metadata
    layers = info.get("layer_count", info.get("layers", 1))
    first_layer = info.get("model_layer_index", 0)
    if type(layers) is not int or not 1 <= layers <= 32 or type(first_layer) is not int or not 0 <= first_layer < 32 or first_layer + layers > 32:
        raise ValueError("prepared layer testcase requires 1..32 consecutive layers within L0..L31")
    if (info["hidden"], info["ffn"], info["heads"]) != (HIDDEN, FFN, HEADS):
        raise ValueError("layer dimensions must be D4096/F12288/H32")
    tokens, sequence = info["tokens"], info["sequence"]
    if type(base_address) is not int or not BASE <= base_address < BASE + 8 * 1024**3 or base_address % 256:
        raise ValueError("base_address must be 256-byte aligned within the DDR aperture")
    if source_indices is not None and hidden_region is None:
        raise ValueError("source_indices require an existing hidden_region")
    if last_layer_output_subset and hidden_region is not None:
        raise ValueError("last-layer output subset requires its own initial hidden allocation")
    if hidden_region is not None and (len(hidden_region) != 2 or hidden_region[0] % 256 or
            hidden_region[1] - hidden_region[0] < max(tokens, max(source_indices or [tokens - 1]) + 1) * HIDDEN * 2):
        raise ValueError("hidden_region cannot contain the input and output tokens")
    if not 1 <= tokens <= sequence <= MAX_S:
        raise ValueError("require 1 <= tokens <= sequence <= 2048")
    bits = reference_data.load("input", "activation_bits").reshape(-1)
    positions = reference_data.load("input", "positions").reshape(-1)
    if len(bits) != tokens or len(positions) != tokens or len(set(positions.tolist())) != tokens or np.any(positions < 0) or np.any(positions >= sequence):
        raise ValueError("invalid token bits or unique logical positions")
    for key in ("attention_mask", "isolated_groups", "boundary_scout"):
        if info.get(key):
            raise ValueError(f"unsupported visibility/selection: {key}")
    records = info.get("layers") if isinstance(info.get("layers"), list) else [{} for _ in range(layers)]
    if len(records) != layers or any(not isinstance(record, dict) for record in records):
        raise ValueError("layer descriptor count/type must cover the consecutive model layer range")
    position_rows = {int(position): row for row, position in enumerate(positions)}
    def selected_rows(field, descriptor_field, default):
        if field in info:
            return info[field]
        declared = records[0].get(descriptor_field)
        if declared is None:
            return default
        if any(position not in position_rows for position in declared):
            raise ValueError(f"{descriptor_field} includes a non-query position")
        return [position_rows[position] for position in declared]
    write_token_indices = selected_rows("kv_write_token_indices", "kv_write_positions", list(range(tokens)))
    commit_token_indices = selected_rows("cache_commit_token_indices", "cache_commit_positions", write_token_indices)
    for name, selected in (("kv_write_token_indices", write_token_indices), ("cache_commit_token_indices", commit_token_indices)):
        if len(set(selected)) != len(selected) or any(type(r) is not int or not 0 <= r < tokens for r in selected):
            raise ValueError(f"invalid {name}")
    if not set(commit_token_indices) <= set(write_token_indices):
        raise ValueError("persistent commit tokens must be a subset of KV write tokens")
    selective = set(write_token_indices) != set(commit_token_indices)
    commit_selected = selective and bool(commit_token_indices)
    if selective and layers != 1:
        raise ValueError("selective persistence requires one layer per testcase")
    if selective and len(commit_token_indices) > 432:
        raise ValueError("selective persistence supports at most 432 committed tokens")
    for layer, descriptor in enumerate(records):
        if "model_layer" in descriptor:
            if descriptor["model_layer"] != first_layer + layer or descriptor.get("layer") != layer:
                raise ValueError("captured layer index/model_layer must match the consecutive model layer range")
        elif descriptor.get("layer", first_layer + layer) != first_layer + layer:
            raise ValueError("layer descriptors must match the consecutive model layer range")
        for key, selected in (("positions", positions), ("kv_write_positions", positions[write_token_indices]),
                              ("cache_commit_positions", positions[commit_token_indices])):
            if key in descriptor and descriptor[key] != selected.tolist():
                raise ValueError(f"unsupported per-layer {key}")
    output_token_indices = last_layer_output_token_indices(info, positions) if last_layer_output_subset else None
    effective_output_subset = output_token_indices is not None and len(output_token_indices) < tokens
    if attention_checkpoints and (attention_pair or qkvo_group):
        raise ValueError("Attention checkpoints require a layer without Attention grouping")
    if type(ffn_group_batches) is not int or ffn_group_batches not in (0, 4, 6):
        raise ValueError("ffn_group_batches must be 0, 4 or 6")
    reuse = bool(ffn_group_batches or ffn_fused_product or ffn_down_pair or
                 attention_output_pair or attention_pair or kv_pair or qkvo_group)
    separate_l31_metadata = effective_output_subset and layers > 1
    if reuse and effective_output_subset and not separate_l31_metadata:
        raise ValueError("reuse cannot be combined with output subset")
    if (hidden_region is not None or source_indices is not None) and (
            ffn_down_pair or attention_output_pair or attention_pair or kv_pair):
        raise ValueError("Down or Attention grouping cannot be combined with boundary hidden selection")
    if qkvo_group and (attention_output_pair or attention_pair or kv_pair):
        raise ValueError("Q/K/V/O grouping cannot be combined with Attention or KV pairing")
    if np.any(bits != 4) and (ffn_down_pair or attention_output_pair or
                             attention_pair or kv_pair):
        raise ValueError("Down or Attention/KV pairing requires every token to use A4")
    if ffn_fused_product and not ffn_group_batches:
        raise ValueError("fused product requires FFN grouping")
    if ffn_down_pair and not ffn_fused_product:
        raise ValueError("Down pairing requires fused product")
    metadata, order, rounds = pack_tokens(bits, positions, set(write_token_indices), source_indices,
                                       output_token_indices if effective_output_subset and not separate_l31_metadata else None,
                                       round_token_limit=round_token_limit)
    input_order = order
    l31_metadata, l31_rounds = None, None
    if separate_l31_metadata:
        source_rows = [0] * tokens
        for packed, token in enumerate(input_order):
            source_rows[token] = packed
        l31_metadata, order, l31_rounds = pack_tokens(
            bits, positions, set(write_token_indices), source_rows, output_token_indices,
            round_token_limit=round_token_limit)
    qkvo_groups = mark_qkvo_groups(metadata, rounds) if qkvo_group else []
    if ffn_group_batches and len(rounds) < 2:
        raise ValueError("FFN grouping requires at least two metadata rounds")
    if ffn_group_batches:
        select_ffn_group_batches(rounds,ffn_group_batches)
    silu, rope_tables = validate_layer_inputs(reference_data, layers, records, bits, positions, write_token_indices, commit_token_indices)
    validate_tensor_shapes(reference_data, layers, tokens, sequence)
    output = output.resolve()
    output.relative_to(artifact_root())
    if output.exists():
        raise FileExistsError(f"use a fresh output directory: {output}")
    output.mkdir(parents=True)
    schemas = {name: json.loads((HARDWARE / "config" / (name + ".json")).read_text())
               for name in ("execution_config", "layer_address_table")}
    cursor = base_address
    regions, pending = [], []

    def reserve(name, size, access="read_write", data=None):
        nonlocal cursor
        # hidden_transfer requires 256-byte bases; use the same alignment for
        # each independently addressed region, including hidden workspaces.
        start, size = align(cursor, 256), align(size)
        cursor = start + size
        if cursor > base_address + 8 * 1024**3:
            raise ValueError("prepared layer regions exceed the DDR aperture")
        regions.append(dict(name=name, base=start, limit=cursor, access=access))
        if data is not None:
            pending.append((start, bytes(data)))
        return start, cursor

    config_region = reserve("configuration", 496 if commit_selected else 320, "read_only")
    refresh_record = (config_region[0] + 320, config_region[0] + 496) if commit_selected else None
    table = reserve("layer_table", (first_layer + layers) * 896, "read_only")
    if separate_l31_metadata:
        row_table = reserve("token_metadata", 32 + len(metadata) + len(l31_metadata), "read_only")
        regular_base = row_table[0] + 32
        l31_base = regular_base + len(metadata)
        directory = struct.pack("<4Q", regular_base, l31_base, l31_base, l31_base + len(l31_metadata))
        pending.append((row_table[0], directory + bytes(metadata) + bytes(l31_metadata)))
    else:
        row_table = reserve("token_metadata", len(metadata), "read_only", metadata)
    next_table = reserve("next_metadata", max(1696, len(metadata)))
    hidden = hidden_region or reserve("hidden", tokens * HIDDEN * 2)
    context_tokens = 96 if attention_output_pair or attention_pair else 48
    if qkvo_groups:
        context_tokens = max(context_tokens,
                             max(group["tokens"] for group in qkvo_groups))
    # Gate/Up may keep several padded batches; KV capture reuses this region
    # before FFN. Allocate it after the preserved hidden/residual images.
    workspace_tokens = 48
    if ffn_group_batches:
        workspace_tokens = max(workspace_tokens, max(
            sum(align(r["tokens"], 2) for r in rounds[start:start + ffn_group_batches])
            for start in range(0, len(rounds), ffn_group_batches)))
    workspace_bytes = max(workspace_tokens * FFN * 4, 96 * HIDDEN * 2 if kv_pair else 0)
    if qkvo_groups:
        workspace_bytes = max(
            workspace_bytes,
            (tokens + max(group["tokens"] for group in qkvo_groups)) * HIDDEN * 2)
    context = reserve("context", context_tokens * HIDDEN * 2)
    temporary = reserve("temporary", tokens * HIDDEN * 4 + workspace_bytes + 48 * HIDDEN * 2)
    gate_up = temporary[0] + tokens * HIDDEN * 4
    residual = gate_up + workspace_bytes
    cache_k = reserve("cache_k", layers * HEADS * MAX_S * DIM)
    cache_v = reserve("cache_v", layers * HEADS * MAX_S * DIM)
    cache_scale = reserve("cache_scale", layers * HEADS * MAX_S * 2)
    retained_k = reserve("retained_k", HEADS * MAX_S * DIM) if selective else cache_k
    retained_v = reserve("retained_v", HEADS * MAX_S * DIM) if selective else cache_v
    retained_scale = reserve("retained_scale", HEADS * MAX_S * 2) if selective else cache_scale
    refresh_rows = reserve("refresh_rows", align(sequence, 8) * 8) if commit_selected else None
    v_scale = reserve("v_scale", layers * HEADS * 2, "read_only")
    gamma = reserve("unit_gamma", HIDDEN * 2, "read_only", np.full(HIDDEN, 0x3f80, dtype="<u2").tobytes())
    cos = reserve("rope_cos", MAX_S * DIM * 2, "read_only")
    sin = reserve("rope_sin", MAX_S * DIM * 2, "read_only")
    layer_entries = []
    weight_payload = []
    for layer in range(layers):
        prefix = layer_prefix(reference_data, layer, layers)
        entry = dict(kv_head_stride_bytes=MAX_S * DIM, kv_token_stride_bytes=DIM,
                     kv_chunk_stride_bytes=8, k_scale_head_stride_bytes=MAX_S * 2,
                     k_scale_token_stride_bytes=2, context_row_stride_bytes=HIDDEN * 2)
        config = records[layer].get("config", info.get("config", {}))
        epsilon = np.array([config.get("rms_norm_eps", info.get("rms_epsilon", 1e-5))], dtype="<f4").view("<u4")
        epsilon_raw = int(((epsilon + 0x7fff + ((epsilon >> 16) & 1)) >> 16)[0])
        entry.update(attention_rms_epsilon_bf16=epsilon_raw, ffn_rms_epsilon_bf16=epsilon_raw)
        def region_fields(name, region):
            entry[name + "_base"], entry[name + "_limit"] = region
        ratios = []
        for linear, field in zip(LINEARS, FIELDS):
            width = FFN if linear == "ff_out" else HIDDEN
            height = FFN if linear in ("ff_proj", "up_proj") else HIDDEN
            codes_name = prefix + "weight." + linear + ".codes"
            codes_entry = reference_data.entries["input", codes_name]
            if int(np.prod(codes_entry["shape"])) != width * height:
                raise ValueError(f"weight shape mismatch: {codes_name}")
            weight = reserve(prefix + linear + ".weight", width * height // 2, "read_only")
            weight_payload.append((weight[0], codes_name, height, width))
            scales = reference_data.load("input", prefix + "weight." + linear + ".scale", "<u2")
            if scales.size != height:
                raise ValueError(f"{linear} weight scale count is {scales.size}; expected {height} output-row scales")
            scale = reserve(prefix + linear + ".scale", scales.nbytes, "read_only", scales.tobytes())
            region_fields(field + "_base_weight", weight)
            region_fields(field + "_enhancement_weight", (0, 0))
            region_fields(field + "_weight_scale", scale)
            bit_key = ("input", prefix + linear + ".activation_bits")
            if bit_key in reference_data.entries and not np.array_equal(reference_data.load(*bit_key).reshape(-1), bits):
                raise ValueError("different per-Linear token bits are unsupported")
            installed_key = ("input", prefix + linear + ".clip.installed")
            ratio = 0
            if installed_key in reference_data.entries and reference_data.load(*installed_key).item():
                ratio = int(reference_data.load("input", prefix + linear + ".clip.ratio", "<u2").item())
            ratios.append(ratio)
            entry[field + "_clip_ratio_bf16"] = ratio
        if ratios[0] != ratios[1] or ratios[0] != ratios[2] or ratios[4] != ratios[5]:
            raise ValueError("shared Q/K/V and Gate/Up inputs require equal clipping ratios")
        entry["flags"] = int(any(ratios))
        for name in ("attn_norm", "ff_norm"):
            if ("input", prefix + name + ".weight") in reference_data.entries and not np.all(reference_data.load("input", prefix + name + ".weight", "<u2") == 0x3f80):
                raise ValueError("folded unit norm gamma is required")
        for name in ("attention_rms_gamma", "ffn_rms_gamma"):
            region_fields(name, gamma)
        region_fields("rope_cos_lut", cos)
        region_fields("rope_sin_lut", sin)
        for name, arena, stride in (("k_cache", cache_k, HEADS * MAX_S * DIM), ("v_cache", cache_v, HEADS * MAX_S * DIM)):
            selected = (arena[0] + layer * stride, arena[0] + (layer + 1) * stride)
            region_fields("current_" + name, selected)
            retained = retained_k if name == "k_cache" else retained_v
            region_fields("retained_" + name, (retained[0] + layer * stride, retained[0] + (layer + 1) * stride))
        region_fields("k_scale", (cache_scale[0] + layer * HEADS * MAX_S * 2, cache_scale[0] + (layer + 1) * HEADS * MAX_S * 2))
        region_fields("v_scale", (v_scale[0] + layer * HEADS * 2, v_scale[0] + (layer + 1) * HEADS * 2))
        if selective:
            entry["retained_k_scale_base"] = retained_scale[0]
        region_fields("context", context)
        region_fields("ffn_gate_up_workspace", (gate_up, residual))
        region_fields("ffn_residual_workspace", (residual, temporary[1]))
        layer_entries.append(pack_record(schemas["layer_address_table"], entry))
    image_path = output / "initial.bin"
    with image_path.open("wb") as stream:
        stream.truncate(cursor - base_address)
    image = np.memmap(image_path, dtype=np.uint8, mode="r+")
    def put(address, value):
        put_region(image, regions, address, value, base_address)
    for address, value in pending:
        put(address, value)
    for address, name, height, width in weight_payload:
        put(address, pack_weights(reference_data.load("input", name, "|i1").reshape(height, width)))
    cfg = dict(flags=3, start_layer=first_layer, layer_count=layers, total_token_count=tokens, sequence_length=sequence,
               layer_weight_entry_stride=896, deployment_artifact_id=0x4831323238384231)
    enabled_flags = ["separate_l31_metadata"] if separate_l31_metadata else []
    if ffn_group_batches:
        cfg["ffn_pair_first_batch_plus1"] = 1
        enabled_flags += ["ffn_pair_repeat", "ffn_four_batch" if ffn_group_batches == 4 else "ffn_six_batch"]
    enabled_flags += [name for name, enabled in (
        ("ffn_fused_product", ffn_fused_product), ("ffn_down_pair", ffn_down_pair),
        ("attention_output_pair", attention_output_pair), ("attention_pair", attention_pair),
        ("kv_pair", kv_pair), ("qkvo_group", qkvo_group)) if enabled]
    for name in enabled_flags:
        cfg["flags"] |= 1 << schemas["execution_config"]["flag_bits"][name]
    for name, region in (("next_token_metadata", next_table), ("output_hidden", hidden),
                         ("bf16_temporary", temporary), ("layer_weight_table", table),
                         ("current_k_cache", cache_k), ("retained_k_cache", retained_k),
                         ("current_v_cache", cache_v), ("retained_v_cache", retained_v),
                         ("k_scale", cache_scale), ("v_scale", v_scale),
                         ("attention_workspace", context), ("token_metadata", row_table)):
        cfg[name + "_base"], cfg[name + "_limit"] = region
    embedding_base = align(base_address + 0x100000000, 8192)
    cfg.update(embedding_base=embedding_base, embedding_limit=embedding_base + 1035993088)
    if selective:
        cfg.update(retained_k_scale_base=retained_scale[0], retained_k_scale_limit=retained_scale[1])
    if commit_selected:
        cfg["refresh_configuration_offset"] = refresh_record[0] - config_region[0]
        refresh_schema = json.loads((HARDWARE / "config/token_refresh_config.json").read_text())
        refresh = dict(flags=1, sequence_length=sequence, target_token_count=len(commit_token_indices),
                       head_stride=MAX_S * DIM, scale_head_stride=MAX_S * 2,
                       table_base=refresh_rows[0], table_limit=refresh_rows[1],
                       metadata_version=1, capture_index=1)
        for kind, current, retained in (("k", cache_k, retained_k), ("v", cache_v, retained_v),
                                         ("scale", cache_scale, retained_scale)):
            refresh["source_" + kind + "_base"], refresh["source_" + kind + "_limit"] = current
            refresh["destination_" + kind + "_base"], refresh["destination_" + kind + "_limit"] = retained
        selected_table = np.zeros(align(sequence, 8), dtype="<u8")
        selected_table[positions[commit_token_indices]] = 1 << 8
        put(refresh_rows[0], selected_table)
        put(refresh_record[0], pack_record(refresh_schema, refresh))
    put(config_region[0], pack_record(schemas["execution_config"], cfg))
    for layer, entry in enumerate(layer_entries):
        put(table[0] + (first_layer + layer) * 896, entry)
    if hidden_region is None:
        put(hidden[0], reference_data.load("input", "hidden", "<u2").reshape(tokens, HIDDEN)[input_order])
    expected = []
    def expect(name, address, value):
        path = output / (name + ".expected.bin")
        value = np.ascontiguousarray(value)
        value.tofile(path)
        expected.append(dict(name=name, address=address, bytes=value.nbytes, path=path.name))
    for layer in range(layers):
        prefix = layer_prefix(reference_data, layer, layers)
        for name, arena, stride, dtype, suffix in (
            ("key_codes", cache_k, HEADS * MAX_S * DIM, "|i1", "key_codes"),
            ("value_codes", cache_v, HEADS * MAX_S * DIM, "|i1", "value_codes"),
            ("key_scale", cache_scale, HEADS * MAX_S * 2, "<u2", "key_scale")):
            initial = reference_data.load("input", prefix + "retained_" + name, dtype)
            golden = reference_data.load("expected", prefix + "cache_" + suffix, dtype)
            if initial.size != golden.size or golden.size != HEADS * sequence * (1 if name == "key_scale" else DIM):
                raise ValueError("cache shape mismatch")
            def physical(value):
                if name == "key_scale":
                    padded = np.zeros((HEADS, MAX_S), dtype="<u2")
                    padded[:, :sequence] = value.reshape(HEADS, sequence)
                    return padded
                padded = np.zeros((HEADS, MAX_S, DIM), dtype=np.int8)
                padded[:, :sequence] = value.reshape(HEADS, sequence, DIM)
                if name == "value_codes":
                    return np.ascontiguousarray(padded.reshape(HEADS, MAX_S, 16, 8).transpose(0, 2, 1, 3))
                return padded
            put(arena[0] + layer * stride, physical(initial))
            persistent = retained_scale if name == "key_scale" else retained_k if name == "key_codes" else retained_v
            if selective:
                put(persistent[0], physical(initial))
            expect(f"layer{first_layer + layer}.cache_{name}", persistent[0] + layer * stride, physical(golden))
        put(v_scale[0] + layer * HEADS * 2, reference_data.load("input", prefix + "v_scale", "<u2"))
        for name, arena in (("cos", cos), ("sin", sin)):
            put(arena[0], rope_tables[layer][name])
        if layer == layers - 1:
            expected_order = order if output_token_indices is None else order[:len(output_token_indices)]
            if expected_order:
                expect("hidden", hidden[0], reference_data.load("expected", prefix + "block_output", "<u2").reshape(tokens, HIDDEN)[expected_order])
            if output_token_indices is not None and len(output_token_indices) < tokens:
                # Earlier layers write every row. Only L31 leaves the unused
                # rows untouched, so their value is the actual L31 input.
                prior_hidden = (reference_data.load("input", "hidden", "<u2") if layers == 1 else
                                reference_data.load("expected", layer_prefix(reference_data, layer - 1, layers) + "block_output", "<u2"))
                expect("hidden_unwritten", hidden[0] + len(output_token_indices) * HIDDEN * 2,
                       prior_hidden.reshape(tokens, HIDDEN)[input_order[len(output_token_indices):]])
    image.flush()
    del image
    ddr_config = ddr_config or HARDWARE / "config/ddr/lpddr4_3200_2x16_8gib.ini"
    shutil.copyfile(ddr_config, output / "ddr.ini")
    region_manifest = dict(base_address=base_address,
                           limit_address=base_address + 8 * 1024**3,
                           logical_aperture_bytes=8 * 1024**3,
                           current_ddr_image_bytes=image_path.stat().st_size,
                           memory_map=regions)
    (output / "memory_map.json").write_text(json.dumps(region_manifest, indent=2) + "\n")
    case = dict(schema="supra-testcase/v1", ddr_image="initial.bin", memory_map="memory_map.json",
                dramsim3_config="ddr.ini",
                executions=[dict(config_address=base_address, id=1,
                    max_cycles=max(50000000, layers * len(rounds) * 40000000))],
                expected=expected, silu_mode="explicit",
                source=dict(kind=info.get("reference_data_kind", "logical_layer_reference_data"), index=index.name,
                            tokens=tokens, sequence=sequence, start_layer=first_layer, layers=layers, rounds=rounds,
                            packed_logical_to_reference_token=order))
    if separate_l31_metadata:
        case["source"].update(input_packed_logical_to_reference_token=input_order,
                              l31_rounds=l31_rounds)
    if silu is not None:
        case["silu_segments"] = silu
    if reuse:
        case["source"]["reuse"] = dict(ffn_group_batches=ffn_group_batches,
            enabled_flags=enabled_flags, context_tokens=context_tokens,
            ffn_workspace_bytes=workspace_bytes,
            qkvo_groups=qkvo_groups,
            attention_sequence_supported=sequence <= 1296)
    if output_token_indices is not None:
        case["source"].update(last_layer_output_subset=effective_output_subset,
            output_reference_tokens=sorted(output_token_indices),
            output_positions=[int(positions[token]) for token in sorted(output_token_indices)])
    if attention_checkpoints:
        final_layer = first_layer + layers - 1
        checkpoint_reference = reference_data if layers == 1 else captured_layer_view(reference_data, final_layer)
        case["attention_checkpoints"] = prepare_attention_checkpoints(
            checkpoint_reference, output, l31_metadata if separate_l31_metadata else metadata,
            order, tokens, sequence, final_layer,
            preserve_input=separate_l31_metadata or (layers == 1 and (len(rounds) > 1 or effective_output_subset)))
    (output / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return output / "case.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Normal testcases use model_layer_index for 1..32 consecutive layers within L0..L31. "
        "Commit tokens must be a subset of write tokens; unequal sets require one layer and at most 432 commits. "
        "Boundary L0 and L1 share the hidden allocation; sequence length is at most 2048."))
    parser.add_argument("--index", type=Path, required=True, help="existing logical layer reference_data index")
    parser.add_argument("--payload-root", type=Path, help="root for relative tensor paths")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory below SUPRA_ARTIFACT_ROOT")
    parser.add_argument("--ddr-config", type=Path)
    parser.add_argument("--base-address", type=lambda value: int(value, 0), default=BASE,
                        help="DDR base for this layer testcase, including its configuration")
    parser.add_argument("--layer", type=int, help="select one observed model layer from the capture")
    parser.add_argument("--boundary", action="store_true", help="prepare nonidentity L0 scout/L1 deep launches sharing actual DDR hidden")
    parser.add_argument("--block-initialization-reference", type=Path,
                        help="captured block initialization reference for RTL deep-row selection")
    parser.add_argument("--boundary-reference", type=Path,
                        help="independent ordinary-boundary reference; live L0 scores drive RTL selection")
    parser.add_argument("--boundary-scout-ffn-group-batches", type=int, choices=(0, 4, 6), default=0)
    parser.add_argument("--boundary-scout-ffn-fused-product", action="store_true")
    parser.add_argument("--boundary-deep-ffn-group-batches", type=int, choices=(0, 4, 6), default=0)
    parser.add_argument("--boundary-deep-ffn-fused-product", action="store_true")
    parser.add_argument("--boundary-scout-qkvo-group", action="store_true")
    parser.add_argument("--boundary-deep-qkvo-group", action="store_true")
    parser.add_argument("--last-layer-output-subset", action="store_true",
                        help="select L31 outputs for recorded current/future predictions and retained hidden consumers")
    parser.add_argument("--round-token-limit", type=int, default=48)
    parser.add_argument("--ffn-group-batches", type=int, choices=(0, 4, 6), default=0,
                        help="mixed or all-A4 Gate/Up reuse in successive groups; 0 preserves the original schedule")
    parser.add_argument("--ffn-fused-product", action="store_true")
    parser.add_argument("--ffn-down-pair", action="store_true")
    parser.add_argument("--attention-output-pair", action="store_true")
    parser.add_argument("--attention-pair", action="store_true")
    parser.add_argument("--kv-pair", action="store_true")
    parser.add_argument("--qkvo-group", action="store_true",
                        help="group two to six metadata batches for Q/K/V/O weight reuse")
    parser.add_argument("--attention-checkpoints", action="store_true",
                        help="compare preserved hidden, Query codes/scales, QK scores, BF16 softmax and final P8 codes/scales from existing RTL observations")
    args = parser.parse_args()
    if (args.block_initialization_reference is not None or args.boundary_reference is not None) and not args.boundary:
        parser.error("selection references require --boundary")
    if args.boundary and args.layer is not None:
        parser.error("--layer and --boundary are mutually exclusive")
    if args.boundary and args.last_layer_output_subset:
        parser.error("boundary scout/deep cannot use last-layer output subset")
    if args.boundary and (args.base_address != BASE or args.round_token_limit != 48 or args.ffn_group_batches or
            args.ffn_fused_product or args.ffn_down_pair or args.attention_output_pair or
            args.attention_pair or args.kv_pair or args.qkvo_group):
        parser.error("boundary FFN options must use the explicit scout/deep arguments")
    if not args.boundary and (args.boundary_scout_ffn_group_batches or
            args.boundary_scout_ffn_fused_product or args.boundary_deep_ffn_group_batches or
            args.boundary_deep_ffn_fused_product or args.boundary_scout_qkvo_group or args.boundary_deep_qkvo_group):
        parser.error("boundary scout/deep FFN options require --boundary")
    if args.boundary:
        print(prepare_boundary(args.index, args.output, args.payload_root, args.ddr_config,
                               attention_checkpoints=args.attention_checkpoints,
                               scout_ffn_group_batches=args.boundary_scout_ffn_group_batches,
                               scout_ffn_fused_product=args.boundary_scout_ffn_fused_product,
                               deep_ffn_group_batches=args.boundary_deep_ffn_group_batches,
                               deep_ffn_fused_product=args.boundary_deep_ffn_fused_product,
                               scout_qkvo_group=args.boundary_scout_qkvo_group,
                               deep_qkvo_group=args.boundary_deep_qkvo_group,
                               block_initialization_reference=args.block_initialization_reference,
                               boundary_reference=args.boundary_reference))
    else:
        print(prepare(args.index, args.output, args.payload_root, args.ddr_config,
                      reference_data=(captured_layer_view(ReferenceData(args.index, args.payload_root), args.layer)
                                      if args.layer is not None else None),
                      base_address=args.base_address,
                      last_layer_output_subset=args.last_layer_output_subset,
                      round_token_limit=args.round_token_limit, ffn_group_batches=args.ffn_group_batches,
                      ffn_fused_product=args.ffn_fused_product, ffn_down_pair=args.ffn_down_pair,
                      attention_output_pair=args.attention_output_pair,
                      attention_pair=args.attention_pair, kv_pair=args.kv_pair,
                      qkvo_group=args.qkvo_group,
                      attention_checkpoints=args.attention_checkpoints))


if __name__ == "__main__":
    main()
