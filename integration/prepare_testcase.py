"""Prepare captured layers or a layer/head/control execution through one entry."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

import prepare_layer_testcase as layers
import prepare_head_testcase as head


def _read_case(case_path, case, address, count):
    """Read initialization overlays, never expected outputs."""
    mapping = json.loads((case_path.parent / case["memory_map"]).read_text())
    image = case_path.parent / case["ddr_image"]
    segments = [dict(address=mapping["base_address"], bytes=image.stat().st_size, path=str(image)),
                *case.get("initial_segments", [])]
    result, present = bytearray(count), bytearray(count)
    for segment in segments:
        lo, hi = max(address, segment["address"]), min(address+count, segment["address"]+segment["bytes"])
        if lo >= hi:
            continue
        with (case_path.parent / segment["path"]).open("rb") as stream:
            stream.seek(lo-segment["address"])
            payload = stream.read(hi-lo)
        if len(payload) != hi-lo:
            raise ValueError("short testcase initialization")
        result[lo-address:hi-address] = payload
        present[lo-address:hi-address] = bytes([1]) * (hi-lo)
    if not all(present):
        raise ValueError(f"uninitialized testcase range {address:#x}+{count}")
    return bytes(result)


def connect_steps(case_path, next_case_path, next_index, *, payload_root=None):
    """Connect a following real step, carrying actual DDR and selector state.

    Each capture supplies static weights and a fixed physical test layout. The
    runtime checks that the RTL-selected positions/precision fit that layout,
    then copies actual token IDs, KV, state and pending values. Expected files
    only check outputs and model producer/consumer consistency here.
    """
    import torch
    from hardware_adapter.control_reference_data import read_captured_control
    from prepare_handoff_testcase import read_record
    case_path, next_case_path, next_index = map(lambda p: Path(p).resolve(strict=True),
                                              (case_path,next_case_path,next_index))
    output = case_path.parent
    case, following = json.loads(case_path.read_text()), json.loads(next_case_path.read_text())
    previous = case["provenance"]["regular_control"]
    full_sequence = previous.get("mode") == "baseline_full_sequence"
    if previous.get("request_end") or previous.get("closeout") == "layer_endpoint":
        raise ValueError("completed testcase has no following captured control")
    source, states, prediction = read_captured_control(Path(previous["index"]), payload_root=previous.get("payload_root"))
    reference = layers.ReferenceData(next_index, payload_root)
    for name in ("dataset","split","sample_id","sample_doc_hash","model_arguments"):
        if source["provenance"].get(name) != reference.metadata["provenance"].get(name):
            raise ValueError("following capture differs in " + name)
    if reference.metadata["forward"]["capture_index"] != prediction["capture_index"] + 1:
        raise ValueError("captures must be consecutive actual forwards")
    new_block = states["next_forward"]["block_index"] != states["before_forward"]["block_index"]
    boundary = following.get("provenance", {}).get("boundary_control")
    if new_block and not full_sequence and boundary is None:
        raise ValueError("packed block transition requires the actual L0 scout/deep capture")
    if boundary is not None and not new_block:
        raise ValueError("boundary capture must begin the following block")
    if full_sequence and not reference.metadata["generation_config"].get("full_sequence_recompute", False):
        raise ValueError("following baseline capture must recompute the full sequence")
    if case.get("silu_segments") != following.get("silu_segments") or case["silu_mode"] != following["silu_mode"]:
        raise ValueError("connected steps require identical compiled SiLU")
    if (output/case["dramsim3_config"]).read_bytes() != (next_case_path.parent/following["dramsim3_config"]).read_bytes():
        raise ValueError("connected steps require identical DDR timing")
    next_control = following.get("provenance",{}).get("regular_control")
    if next_control and full_sequence != (next_control.get("mode") == "baseline_full_sequence"):
        raise ValueError("connected steps change full-sequence/packed execution")
    execution = dict(following["executions"][0])
    config_address = execution["config_address"]
    config = read_record("execution_config",_read_case(next_case_path,following,config_address,320))
    if config["start_layer"] != 0:
        raise ValueError("next regular step must start at L0")
    view = layers.captured_layer_view(reference,0)
    selected = states["next_layer_inputs"]
    positions = view.load("input","positions").reshape(-1).tolist()
    bits = view.load("input","activation_bits").reshape(-1).tolist()
    hidden = view.load("input","hidden","<u2").reshape(-1,layers.HIDDEN)
    selected_positions = selected["query_position_ids"]
    if selected_positions is None and full_sequence:
        selected_positions = torch.arange(states["next_forward"]["total_length"])
    if full_sequence and (positions != list(range(states["next_forward"]["total_length"])) or
                          bits != [8] * len(positions) or config["refresh_configuration_offset"]):
        raise ValueError("following baseline input must cover the full sequence at A8 without refresh")
    if (positions != selected_positions.reshape(-1).tolist() or
            bits != selected["activation_bits"].reshape(-1).tolist() or
            not np.array_equal(hidden,selected["hidden"].contiguous().view(torch.int16).numpy().view("<u2").reshape(hidden.shape))):
        raise ValueError("following layer input differs from actual previous selection/embedding")
    mapping_path = output/case["memory_map"]
    mapping = json.loads(mapping_path.read_text())
    next_map = json.loads((next_case_path.parent/following["memory_map"]).read_text())
    step = len(case.get("provenance",{}).get("connected_steps",[])) + 1
    prefix = f"step{step}."
    relative = lambda name: os.path.relpath(next_case_path.parent/name,output)
    for region in next_map["memory_map"]:
        if any(region["base"] < r["limit"] and r["base"] < region["limit"] for r in mapping["memory_map"]):
            raise ValueError("connected step DDR regions overlap")
        mapping["memory_map"].append(dict(region,name=prefix+region["name"]))
    # All static data is loaded once. Actual copies occur later, after drain.
    image = next_case_path.parent/following["ddr_image"]
    case.setdefault("initial_segments",[]).append(dict(address=next_map["base_address"],
        bytes=image.stat().st_size,path=relative(following["ddr_image"])))
    case["initial_segments"].extend(dict(item,path=relative(item["path"])) for item in following.get("initial_segments",[]))
    storage = case["provenance"]["regular_control"]["storage"]
    copies = []
    def copy(source_range,destination_range):
        size = source_range[1]-source_range[0]
        if size != destination_range[1]-destination_range[0]:
            raise ValueError("carried state shape changed within a regular block")
        if size:
            copies.append(dict(source_address=source_range[0],destination_address=destination_range[0],bytes=size))
    if next_control:
        next_storage = following["provenance"]["regular_control"]["storage"]
        carried = ("token_table",) if full_sequence else (("cross_pending", "token_table", "history") if new_block else
            ("relation", "cross_relation", "pending", "cross_pending", "budget", "token_table", "attempts", "history"))
        for name in carried:
            if name == "history" and name not in storage:
                continue
            copy(storage[name],next_storage[name])
        if not new_block:
            copy(storage["state"],next_storage["input_state"])
    # Preserve every cache already executed in this chain. Layers encountered
    # for the first time retain their captured entrance cache.
    cache_name = lambda name: re.search(r"(layer\d+\.cache_\w+)$",name)
    caches = {cache_name(item["name"]).group(1):item for item in case["expected"] if cache_name(item["name"])}
    for item in (() if full_sequence else following["expected"]):
        name = cache_name(item["name"])
        if name and name.group(1) in caches:
            prior = caches[name.group(1)]
            if prior["bytes"] != item["bytes"]:
                raise ValueError("persistent cache shape changed")
            if (output/prior["path"]).read_bytes() != _read_case(next_case_path,following,item["address"],item["bytes"]):
                raise ValueError("model cache producer/consumer differ: " + name.group(1))
            copy([prior["address"],prior["address"]+prior["bytes"]],[item["address"],item["address"]+item["bytes"]])
    embedding = case["provenance"].get("embedding_table")
    if embedding is None:
        base = layers.align(max(r["limit"] for r in mapping["memory_map"]),8192)
        embedding = dict(base=base,limit=base+126464*8192,tokens={})
        mapping["memory_map"].append(dict(name="embedding",base=base,limit=embedding["limit"],access="read_only"))
        case["provenance"]["embedding_table"] = embedding
    for row,position in enumerate(positions):
        token = int(states["next_forward"]["tokens"].reshape(-1)[position])
        if not 0 <= token < 126464:
            raise ValueError("actual input token exceeds vocabulary")
        payload = hidden[row].tobytes()
        existing = embedding["tokens"].get(str(token))
        if existing:
            if (output/existing).read_bytes() != payload:
                raise ValueError("captured embedding differs for the same token")
        else:
            name = f"embedding_{token}.bin"
            (output/name).write_bytes(payload)
            embedding["tokens"][str(token)] = name
            case["initial_segments"].append(dict(address=embedding["base"]+token*8192,bytes=8192,path=name))
    if max(r["limit"] for r in mapping["memory_map"]) > head.BASE + 8*1024**3:
        raise ValueError("connected steps exceed the 8 GiB DDR aperture")
    config.update(embedding_base=embedding["base"],embedding_limit=embedding["limit"])
    config_name = prefix+"execution.bin"
    (output/config_name).write_bytes(layers.pack_record(json.loads((layers.HARDWARE/"config/execution_config.json").read_text()),config))
    case["initial_segments"].append(dict(address=config_address,bytes=320,path=config_name))
    if next_control and not full_sequence and not boundary:
        refresh_address = config_address + config["refresh_configuration_offset"]
        header = _read_case(next_case_path, following, refresh_address, 8)
        if int.from_bytes(header[6:8], "little") == 208:
            entry = bytearray(_read_case(next_case_path, following, refresh_address+176, 32))
            entry[12:32] = bytes(20)
            name = prefix + "refresh_execution.bin"
            (output/name).write_bytes(entry)
            case["initial_segments"].append(dict(address=refresh_address+176, bytes=32, path=name))
    current = states["next_forward"]["block_start"]
    action = dict(kind="execution_from_metadata",config_address=config_address,
        metadata_address=previous["metadata_base"],metadata_capacity=previous["metadata_limit"]-previous["metadata_base"],
        layout_address=config["token_metadata_base"],layout_capacity=config["token_metadata_limit"]-config["token_metadata_base"],
        current_begin=current,current_end=current+32)
    if full_sequence:
        action.update(kind="full_sequence_from_state", sequence=len(positions),
            capture=prediction["capture_index"] + 1,
            source_state_address=storage["state"][0],
            source_current_begin=states["before_forward"]["block_start"],
            table_address=(next_storage if next_control else storage)["token_table"][0],
            reset_current_state=new_block)
    if not full_sequence and "history" in storage:
        live = next_storage if next_control else storage
        action.update(history_address=live["history"][0], token_table_address=live["token_table"][0],
            source_state_address=storage["state"][0], source_state_count=(storage["state"][1]-storage["state"][0])//32,
            source_event_address=storage["event"][0], source_current_begin=states["before_forward"]["block_start"],
            source_metadata_address=storage["executed_metadata"][0],
            source_metadata_capacity=storage["executed_metadata"][1]-storage["executed_metadata"][0],
            capture=prediction["capture_index"]+1, sequence=len(states["next_forward"]["tokens"].reshape(-1)))
    if boundary:
        if "history_address" not in action:
            raise ValueError("packed boundary requires cumulative transitions and cache-refresh history")
        copy(storage["cross_pending"], boundary["pending"])
        configuration = reference.metadata["generation_config"]
        action.update(kind="packed_boundary_from_state", metadata_address=boundary["full_metadata"][0],
            metadata_capacity=boundary["full_metadata"][1]-boundary["full_metadata"][0],
            table_address=boundary["table"][0], historical_only=boundary["historical_only"],
            probability_address=boundary["probability"][0], probability_limit=boundary["probability"][1],
            jobs_address=boundary["jobs"][0], jobs_capacity=(boundary["jobs"][1]-boundary["jobs"][0])//64,
            refresh_address=boundary["refresh_address"], probability_config_address=boundary["probability_config_address"],
            precision_policy=configuration["feature3_precision_policy"], maturity_age=configuration["feature3_maturity_age"],
            context_bits=configuration["cross_block_boundary_context_row_bits"],
            block_initialization_all_a8=boundary["historical_only"] and configuration["cross_block_block_initialization_all_a8"],
            block_index=states["next_forward"]["block_index"])
        if next_control:
            action.update(next_state_address=next_storage["input_state"][0],
                next_state_count=(next_storage["input_state"][1]-next_storage["input_state"][0])//32)
            deep_action = following["executions"][-1]["handoff_actions"][0]
            deep_action.update(state_address=next_storage["input_state"][0], state_count=action["next_state_count"])
    if config["flags"] & (1 << 13):
        directory = _read_case(next_case_path, following, config["token_metadata_base"], 32)
        regular_base, regular_limit = np.frombuffer(directory, dtype="<u8", count=2).tolist()
        action.update(layout_address=regular_base, layout_capacity=regular_limit-regular_base)
    if storage["joint_result"][0] and not boundary:
        action["future_result_address"] = storage["joint_result"][0]
    if next_control and not boundary:
        action.update(state_address=next_storage["input_state"][0],
                      closeout_kind=next_control.get("closeout_kind", 0),
                      state_count=(next_storage["input_state"][1]-next_storage["input_state"][0])//32)
    # Mark host-written descriptors explicitly. These writes are not DDR traffic
    # generated by the accelerator, and are reported separately by the runner.
    for region in mapping["memory_map"]:
        if region["base"] <= action["layout_address"] < region["limit"]:
            region["access"] = "read_write"
    execution.update(copies=copies+execution.get("copies", []),handoff_actions=[action]+execution.get("handoff_actions", []),check_embedding_reads=True)
    destination_index = len(case["executions"])
    for item in case["expected"]:
        item.setdefault("execution_index", destination_index-1)
    launches = [execution] + following["executions"][1:]
    for offset, launch in enumerate(launches):
        case["executions"].append(dict(launch, id=destination_index+offset+1))
    for item in following["expected"]:
        case["expected"].append(dict(item,name=prefix+item["name"],path=relative(item["path"]),
            execution_index=destination_index+item.get("execution_index", len(launches)-1)))
    for key in ("head_checkpoints","attention_checkpoints"):
        if key not in following:
            continue
        checks = case.setdefault(key,[])
        if isinstance(checks,dict):
            checks = case[key] = [checks]
        incoming = following[key] if isinstance(following[key], list) else [following[key]]
        for entry in incoming:
            check = dict(entry,execution_index=destination_index+entry.get("execution_index", len(launches)-1))
            for field in ("expected","candidates"):
                if field in check:
                    check[field] = [dict(item,path=relative(item["path"])) for item in check[field]]
            checks.append(check)
    case.setdefault("required_features",[]).extend(["multiple_checkpoints","actual_metadata_layout"])
    case["required_features"] = sorted(set(case["required_features"]+following.get("required_features",[])))
    case["provenance"].setdefault("connected_steps",[]).append(dict(index=str(next_index),execution_index=destination_index,
        actual_cache_copies=len(copies)))
    if next_control:
        case["provenance"]["regular_control"] = dict(next_control)
    else:
        case["provenance"]["regular_control"]["closeout"] = "layer_endpoint"
    mapping_path.write_text(json.dumps(mapping,indent=2)+"\n")
    case_path.write_text(json.dumps(case,indent=2)+"\n")
    return case_path


def head_output_rows(layer_case, layer_index, head_index, payload_root=None):
    """Map logical consumers to actual DDR rows; expected is only a connection check."""
    case = json.loads(Path(layer_case).read_text())
    reference = layers.ReferenceData(Path(layer_index), payload_root)
    source = case["source"]
    if source.get("kind") == "boundary_scout_deep":
        source = source["deep"]
        reference = layers.captured_layer_view(reference, source["start_layer"], source["start_layer"]+source["layers"]-1)
    positions = reference.load("input", "positions").reshape(-1).tolist()
    order = source["packed_logical_to_reference_token"]
    packed = {positions[token]: ordinal for ordinal, token in enumerate(order)}
    _, entries, _ = head.load_reference_data(Path(head_index), payload_root)
    consumers = np.fromfile(entries["input", "positions"]["path"], dtype="<i8").tolist()
    if len(set(consumers)) != len(consumers) or any(position not in packed for position in consumers):
        raise ValueError("head consumers must map uniquely to executed layer positions")
    return [packed[position] for position in consumers]


def prepare(index, output, *, mode="layer", head_index=None, payload_root=None,
            ddr=3200, base_address=None, output_subset=None, attention_checkpoints=None,
            layer_options=None, head_base_address=head.BASE, layer_range=None):
    index, output = Path(index).resolve(strict=True), Path(output).resolve()
    if mode not in ("layer", "boundary", "head", "regular"):
        raise ValueError("unknown preparation mode")
    if output.exists():
        raise FileExistsError(output)
    root = layers.artifact_root()
    if root not in output.parents:
        raise ValueError("testcase output must be below the hardware artifact root")
    ddr_config = layers.HARDWARE / f"config/ddr/lpddr4_{ddr}_2x16_8gib.ini"
    explicit_base = base_address is not None or "base_address" in (layer_options or {})
    options = dict(layer_options or {})
    for key, explicit, default in (("base_address", base_address, layers.BASE),
            ("last_layer_output_subset", output_subset, False),
            ("attention_checkpoints", attention_checkpoints, None)):
        value = options.pop(key, default)
        if explicit is not None:
            value = explicit
        if key == "base_address":
            base_address = value
        elif key == "last_layer_output_subset":
            output_subset = value
        else:
            attention_checkpoints = value
    reference = None
    if layer_range is not None:
        if (not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2 or
                any(type(value) is not int for value in layer_range)):
            raise ValueError("layer_range needs first and last model layer indices")
        reference = layers.captured_layer_view(layers.ReferenceData(index, payload_root), *layer_range)
    if mode == "head":
        if head_index is not None or options or output_subset or attention_checkpoints or explicit_base or layer_range is not None:
            raise ValueError("head mode takes only its head index and DDR setting")
        return head.prepare(index, output, payload_root, ddr=ddr, base_address=head_base_address)
    if mode == "boundary":
        if head_index is not None or output_subset or explicit_base or head_base_address != head.BASE or layer_range is not None:
            raise ValueError("boundary mode uses the scout/deep preparation layout")
        return layers.prepare_boundary(index, output, payload_root, ddr_config,
                                       attention_checkpoints=attention_checkpoints, **options)
    source_metadata = json.loads(index.read_text())
    two_stage = source_metadata.get("forward", {}).get("layer0_global_selected_deep", False)
    first_layer = layer_range[0] if layer_range is not None else source_metadata.get("model_layer_index", 0)
    last_layer = layer_range[1] if layer_range is not None else first_layer + source_metadata.get("layer_count", 1) - 1
    boundary = two_stage and first_layer == 0
    def prepare_layers(destination, address, checked, subset):
        if not boundary:
            return layers.prepare(index, destination, payload_root, ddr_config,
                base_address=address, reference_data=reference,
                last_layer_output_subset=subset, attention_checkpoints=checked, **options)
        from hardware_adapter.control_reference_data import captured_block_initialization_reference, captured_boundary_reference
        config = source_metadata["generation_config"]
        block_initialization = source_metadata["forward"]["block_index"] == config["cross_block_full_prefix_oracle_block"]
        producer = captured_block_initialization_reference if block_initialization else captured_boundary_reference
        reference_path = destination.parent / (destination.name + "_selection_reference.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps(producer(index, payload_root=payload_root), indent=2) + "\n")
        return layers.prepare_boundary(index, destination, payload_root, ddr_config,
            base_address=address, last_layer=last_layer, attention_checkpoints=checked,
            **{("block_initialization_reference" if block_initialization else "boundary_reference"): reference_path}, **options)
    if mode == "layer":
        if head_index is not None or head_base_address != head.BASE:
            raise ValueError("head inputs and addresses require head or regular mode")
        return prepare_layers(output, base_address, attention_checkpoints, output_subset)
    if head_index is None:
        raise ValueError("regular mode requires the matching actual head index")
    head_index = Path(head_index).resolve(strict=True)
    from hardware_adapter.control_reference_data import captured_feature2_reference
    observed = captured_feature2_reference(index, payload_root=payload_root)
    head_data = json.loads(head_index.read_text())
    head.validate_observed_head_reference(head_data, observed, index, head_index=head_index, payload_root=payload_root)
    output.mkdir(parents=True)
    control = output / "control_reference.json"
    control.write_text(json.dumps(observed, indent=2) + "\n")
    layer_case = prepare_layers(output / "layers", base_address if explicit_base else 0x60000000,
                                attention_checkpoints, True)
    rows = head_output_rows(layer_case, index, head_index, payload_root)
    case = head.prepare(head_index, output / "execution", payload_root, ddr=ddr,
        preceding_layer_testcase=layer_case, layer_output_tokens=rows, feature2_reference=control,
        base_address=head_base_address)
    if boundary:
        prepared = json.loads(case.read_text())
        prepared["provenance"]["boundary_control"] = json.loads(layer_case.read_text())["provenance"]["boundary_control"]
        case.write_text(json.dumps(prepared, indent=2) + "\n")
    layers.attach_regular_control(case, index, payload_root=payload_root)
    return case


def captured_steps(selections):
    """Use the capture set's actual order, including every intermediate event."""
    if not isinstance(selections, dict):
        return selections
    if selections.get("schema") != "supra-capture-set/v1":
        raise ValueError("steps object must be a completed capture set")
    steps = {}
    for item in selections["captures"]:
        index = item["capture_index"]
        if type(index) is not int or index < 0 or item["kind"] not in ("layers", "head"):
            raise ValueError("invalid capture-set event/index kind")
        step = steps.setdefault(index, {})
        key = "index" if item["kind"] == "layers" else "head_index"
        if key in step:
            raise ValueError("duplicate capture-set event payload")
        step[key] = item["index"]
        if key == "index" and "layer_range" in item:
            step["layer_range"] = item["layer_range"]
    indices = sorted(steps)
    if not indices or indices != list(range(indices[0], indices[-1]+1)):
        raise ValueError("capture set must include every actual intermediate forward")
    if any("index" not in step for step in steps.values()):
        raise ValueError("each prepared event requires captured layers")
    if any("head_index" not in steps[index] for index in indices[:-1]):
        raise ValueError("every intermediate captured event requires its actual head/control")
    return [steps[index] for index in indices]


def prepare_steps(selections, output, *, payload_root=None, ddr=3200):
    """Prepare consecutive captures through the same single-step functions."""
    selections = captured_steps(selections)
    if not isinstance(selections,list) or not selections:
        raise ValueError("steps must list consecutive captured forwards")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result, base = None, head.BASE
    for number, selection in enumerate(selections):
        if set(selection) - {"index","head_index","layer_options","layer_range"} or "index" not in selection:
            raise ValueError("each step needs index and optional head_index/layer_options")
        with_head = "head_index" in selection
        if not with_head and number != len(selections)-1:
            raise ValueError("every non-final step requires its actual head/control capture")
        case = prepare(selection["index"],output/f"step{number}",mode="regular" if with_head else "layer",
            head_index=selection.get("head_index"),payload_root=payload_root,ddr=ddr,
            base_address=base+0x50000000 if with_head else base,head_base_address=base if with_head else head.BASE,
            layer_options=selection.get("layer_options"),
            layer_range=selection.get("layer_range"))
        if result is None:
            result = case
        else:
            connect_steps(result,case,selection["index"], payload_root=payload_root)
        prepared = json.loads(result.read_text())
        mapping = json.loads((result.parent/prepared["memory_map"]).read_text())
        base = layers.align(max(region["limit"] for region in mapping["memory_map"]),256)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--index", type=Path)
    inputs.add_argument("--steps", type=Path, help="JSON list of consecutive layer/head capture indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("layer", "boundary", "head", "regular"), default="layer")
    parser.add_argument("--head-index", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--ddr", type=int, choices=(2400, 3200), default=3200)
    parser.add_argument("--base-address", type=lambda value: int(value, 0))
    parser.add_argument("--head-base-address", type=lambda value: int(value, 0), default=head.BASE)
    parser.add_argument("--output-subset", action="store_true", default=None)
    parser.add_argument("--attention-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.add_argument("--layer-options", type=Path, help="JSON of explicit prepare-layer scheduling arguments")
    args = parser.parse_args()
    if args.steps:
        if (args.head_index or args.base_address is not None or args.layer_options or args.layer_range or args.mode != "layer"
                or args.output_subset or args.attention_checkpoints is not None or args.head_base_address != head.BASE):
            parser.error("--steps carries per-step inputs; do not combine it with single-step options")
        selections = captured_steps(json.loads(args.steps.read_text()))
        for selection in selections:
            for key in ("index","head_index"):
                if key in selection:
                    selection[key] = str((args.steps.parent/selection[key]).resolve(strict=True))
        print(prepare_steps(selections,args.output,payload_root=args.payload_root,ddr=args.ddr))
        return
    options = json.loads(args.layer_options.read_text()) if args.layer_options else None
    print(prepare(args.index, args.output, mode=args.mode, head_index=args.head_index,
        payload_root=args.payload_root, ddr=args.ddr, base_address=args.base_address,
        head_base_address=args.head_base_address,
        output_subset=args.output_subset, attention_checkpoints=args.attention_checkpoints,
        layer_options=options, layer_range=args.layer_range))


if __name__ == "__main__":
    main()
