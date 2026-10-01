"""CPU checks of W4 DDR bytes and execution token metadata against published schemas."""

import json

import numpy as np
import pytest

import prepare_layer_testcase as testcase
import prepare_testcase as unified


@pytest.mark.parametrize("updates,valid", [
    ({"precision_age": -1}, True), ({"activation_bits": 7}, False),
    ({"change_flags": 4}, False), ({"token_position": 2048}, False),
])
def test_record_encoders_share_token_state_rules(updates, valid):
    from build_forward_postprocess_config import pack_schema
    from prepare_handoff_testcase import record
    from prepare_head_testcase import patch_record

    layout = schema("token_state_entry")
    values = {field["name"]: 0 for field in layout["fields"]}
    values["activation_bits"] = 8
    initial = pack_schema(layout, values, "state")
    changed = {**values, **updates}
    encoders = [lambda: pack_schema(layout, changed, "state"),
                lambda: testcase.pack_record(layout, changed),
                lambda: record("token_state_entry", changed)]
    image = bytearray(b"prefix!!" + initial + b"suffix!!")
    before = bytes(image)
    if valid:
        expected = encoders[0]()
        assert all(encode() == expected for encode in encoders[1:])
        assert expected[16:18] == b"\xff\xff"
        patch_record(image, 8, layout, updates)
        assert image == b"prefix!!" + expected + b"suffix!!"
    else:
        for encode in encoders:
            with pytest.raises(ValueError):
                encode()
        with pytest.raises(ValueError):
            patch_record(image, 8, layout, {"precision_age": -1, **updates})
        assert bytes(image) == before


@pytest.mark.parametrize("updates", [
    {"flags": 0x80}, {"flags": 0x100}, {"flags": 0x1200},
    {"embedding_base": 3},
])
def test_record_encoders_share_execution_rules(updates):
    from build_forward_postprocess_config import pack_schema, patch_record
    from prepare_handoff_testcase import record

    layout = schema("execution_config")
    initial = (testcase.HARDWARE / "cases/head_state_update_32tokens/initial.bin").read_bytes()[:320]
    values = {**decode_record(initial, layout), **updates}
    for encode in (lambda: pack_schema(layout, values, "execution"),
                   lambda: testcase.pack_record(layout, values),
                   lambda: record("execution_config", values)):
        with pytest.raises(ValueError):
            encode()
    image = bytearray(initial)
    with pytest.raises(ValueError):
        patch_record(image, 0, layout, updates)
    assert image == initial


def test_boundary_cli_passes_block_initialization_reference(monkeypatch, tmp_path):
    reference = tmp_path / "selection.json"
    calls = []
    monkeypatch.setattr(testcase, "prepare_boundary", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("sys.argv", ["prepare_layer_testcase.py", "--index", "layers.json",
                                   "--output", str(tmp_path / "td"), "--boundary",
                                   "--block-initialization-reference", str(reference)])
    testcase.main()
    assert len(calls) == 1
    assert calls[0][1]["block_initialization_reference"] == reference


def test_capture_set_prepares_every_actual_event_and_partial_endpoints():
    records = []
    for index, first, last in ((4,30,31),(5,0,31),(6,0,2)):
        records.append(dict(capture_index=index,kind="layers",index=f"layers{index}.json",layer_range=[first,last]))
        if index != 6:
            records.append(dict(capture_index=index,kind="head",index=f"head{index}.json"))
    source = dict(schema="supra-capture-set/v1",captures=list(reversed(records)))
    assert unified.captured_steps(source) == [
        dict(index="layers4.json",head_index="head4.json",layer_range=[30,31]),
        dict(index="layers5.json",head_index="head5.json",layer_range=[0,31]),
        dict(index="layers6.json",layer_range=[0,2])]
    source["captures"] = [item for item in records if item["capture_index"] != 5]
    with pytest.raises(ValueError,match="every actual intermediate"):
        unified.captured_steps(source)
    source["captures"] = [item for item in records if item["index"] != "head5.json"]
    with pytest.raises(ValueError,match="actual head/control"):
        unified.captured_steps(source)
    source["captures"] = [records[-1]]
    assert unified.captured_steps(source) == [dict(index="layers6.json",layer_range=[0,2])]


@pytest.mark.parametrize("option", [
    ["--output-subset"], ["--attention-checkpoints"],
    ["--head-base-address", "0x40000000"], ["--base-address", "0x40000000"],
])
def test_steps_reject_single_step_options_before_reading_inputs(monkeypatch, option):
    import prepare_testcase as unified
    monkeypatch.setattr("sys.argv", ["prepare_testcase.py", "--steps", "missing.json",
                                   "--output", "unused", *option])
    with pytest.raises(SystemExit) as error:
        unified.main()
    assert error.value.code == 2


@pytest.mark.parametrize("mode,options", [
    ("head", {"base_address": 0x40000000}),
    ("boundary", {"head_base_address": 0x40000000}),
    ("layer", {"head_base_address": 0x40000000}),
])
def test_prepare_rejects_addresses_unused_by_selected_mode(tmp_path, monkeypatch, mode, options):
    import prepare_testcase as unified
    index = tmp_path / "index.json"
    index.write_text("{}")
    monkeypatch.setattr(unified.layers, "artifact_root", lambda: tmp_path)
    with pytest.raises(ValueError):
        unified.prepare(index, tmp_path / "output", mode=mode, **options)


def test_ffn_activation_capacity_selects_four_batches_for_dense_a8():
    # Six full A8 rounds require 24 x 32768 B, beyond the 589824 B layout.
    # Six full A4 rounds fit exactly; a mixed three-round boundary also fits.
    a8=[dict(groups=4) for _ in range(13)]
    assert testcase.select_ffn_group_batches(a8)==4
    with pytest.raises(ValueError,match='18-compute-group'):
        testcase.select_ffn_group_batches(a8,6)
    assert testcase.select_ffn_group_batches([dict(groups=3)]*12,6)==6
    assert testcase.select_ffn_group_batches([dict(groups=n) for n in (3,4,1)],6)==6


@pytest.mark.parametrize("local_state_version, closeout_kind", [(False, 0), (True, 0), (False, 1), (False, 2)])
def test_observed_head_control_preserves_sparse_predictions_and_hidden_mapping(tmp_path, local_state_version, closeout_kind):
    import prepare_head_testcase as head
    reference = json.loads((head.HARDWARE / "cases/control/psme_state_updates.json").read_text())["records"][0]
    reference["capture_index"] = 7
    source = tmp_path / "state_input.json"
    if local_state_version:
        reference.pop("capture_index", None)
        source = tmp_path / "state_input.json"
    if closeout_kind:
        reference["config"].update(closeout_kind=closeout_kind, tail_enable=False,
                                   tail_bypass_all=False, tail_bypass_stable_only=False)
    source.write_text(json.dumps(reference))
    image = bytearray((head.HARDWARE / "cases/head_state_update_32tokens/initial.bin").read_bytes())
    head.patch_record(image, 0xD1000, head.load_schema("prediction_record"),
                      {"final_hidden_ddr_row_index": 26})
    (tmp_path / "initial.bin").write_bytes(image)
    case = dict(ddr_image="initial.bin", executions=[{}], head_checkpoints=dict(vocab=reference["config"]["vocabulary"]), expected=[], provenance={})
    positions = [row["logical"] for row in reference["rows"] if
                 (row["state"] == 1 if closeout_kind == 1 else row["state"] != 2)]
    np.zeros((len(positions), reference["config"]["vocabulary"]), dtype="<u2").tofile(tmp_path / "expected.output.bin")
    head.apply_observed_feature2(case, tmp_path, source, positions)
    data = (tmp_path / "initial.bin").read_bytes()
    descriptor = decode_record(data[0xD1000:], schema("prediction_record"))
    assert descriptor["final_hidden_ddr_row_index"] == 26
    assert descriptor["token_position"] == positions[0]
    assert descriptor["current_state_entry"] == positions[0]-reference["block_start"]
    block = decode_record(data[0xDA000:], schema("draft_verify_block_config"))
    assert block["position_count"] == 32 > len(positions)
    assert not block["flags"] & 32
    assert (block["flags"] >> 3) & 3 == closeout_kind
    assert block["observed_mask"] == 0  # Nonzero is reserved for a canonical future block.
    assert block["scheduled_quota"] == reference["config"]["scheduled"]
    assert block["input_capture_index"] == reference.get("capture_index", 0)
    assert block["capture_index"] == reference.get("capture_index", 0) + 1
    post = decode_record(data[head.POST-head.BASE:], schema("forward_postprocess_config"))
    assert post["flags"] & 1  # Required by the cross-block pending consumer.
    assert len(case["expected"]) == 41  # Includes explicit tail_closed, even when zero.
    history = next(e for e in case["expected"] if e["name"] == "control_state_history")
    saved = (tmp_path/history["path"]).read_bytes()
    assert (history["element_bytes"], history["stride_bytes"]) == (12,32)
    for i, row in enumerate(reference["rows"]):
        entry = saved[i*12:(i+1)*12]
        observed = row["logical"] in positions
        assert int.from_bytes(entry[:4], "little") == reference.get("capture_index", 0)+1
        assert entry[10] == observed
        assert int.from_bytes(entry[8:10], "little") == (row["action_confidence"] if observed else 0)
    with pytest.raises(ValueError, match="current then future prediction positions"):
        head.apply_observed_feature2(case, tmp_path, source, positions[:-1])


def test_regular_connection_rejects_missing_action_history_before_loading_capture(tmp_path):
    import prepare_head_testcase as head
    (tmp_path / "initial.bin").write_bytes(
        (head.HARDWARE / "cases/head_state_update_32tokens/initial.bin").read_bytes())
    case = dict(ddr_image="initial.bin", executions=[{}, {}],
                provenance=dict(control_reference="unused.json"),
                attention_checkpoints=dict(physical_to_reference_token=[[0]], token_positions=[0]))
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError, match="action-confidence history publication"):
        testcase.attach_regular_control(path, tmp_path / "missing_capture.json")


@pytest.fixture
def future_selection_inputs():
    import torch
    return dict(
        next_row_bits=torch.tensor([4, 8, 4, 8]),
        next_unresolved=torch.tensor([True, True, False, True]),
        next_tentative=torch.tensor([False, True, True, False]),
        next_priority=torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.bfloat16),
        next_dependency_score=torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.bfloat16),
        next_service_count=torch.tensor([0, 1, 2, 3]),
        next_last_confidence=torch.tensor([-1.0, 0.0, 0.50390625, 1.0], dtype=torch.bfloat16),
    )


def test_future_selection_history_sentinel_and_observed_raw_bits(future_selection_inputs):
    # engine initializes history to -1 and publishes actual action confidence.
    # lookahead's retry rule uses this history after the first attempt: observed
    # zero must remain distinct from unobserved, even for a suppressed winner.
    data = testcase.pack_future_selection_records(future_selection_inputs)
    # Literal wire records also check flags, signed service count, BF16 fields,
    # reserved bytes, and the 16-byte stride independently of the pack format.
    assert data == bytes.fromhex(
        "803f 0100 00000000 0000 00 00 0000 0000 "
        "003f 0700 01000000 0000 01 00 803e 0000 "
        "803e 0200 02000000 013f 01 00 003f 0000 "
        "0000 0500 03000000 803f 01 00 803f 0000"
    )


def test_future_selection_missing_history_and_service(future_selection_inputs):
    future_selection_inputs.update(next_last_confidence=None, next_service_count=None)
    data = testcase.pack_future_selection_records(future_selection_inputs)
    assert len(data) == 64
    for offset in range(0, len(data), 16):
        assert data[offset + 4:offset + 12] == bytes(8)


@pytest.mark.parametrize("confidence", [-2.0, -0.5, 1.0078125, float("nan"), float("inf"), -float("inf")])
def test_future_selection_rejects_invalid_history(future_selection_inputs, confidence):
    # Reject malformed captured history before it can initialize DDR state.
    future_selection_inputs["next_last_confidence"][2] = confidence
    with pytest.raises(ValueError, match="unobserved.*or in"):
        testcase.pack_future_selection_records(future_selection_inputs)


def test_head_relocation_preserves_external_layer_and_moves_state(tmp_path):
    import prepare_head_testcase as head
    source = head.HARDWARE / "cases/head_state_update_32tokens"
    image = bytearray((source / "initial.bin").read_bytes())
    external = 0x60000000
    head.patch_record(image, 0, head.load_schema("execution_config"),
                      dict(output_hidden_base=external, output_hidden_limit=external + 8192))
    (tmp_path / "initial.bin").write_bytes(image)
    mapping = dict(memory_map=[dict(name="head", base=head.BASE, limit=head.BASE + len(image)),
                               dict(name="layer", base=external, limit=external + 8192)])
    (tmp_path / "map.json").write_text(json.dumps(mapping))
    case = dict(ddr_image="initial.bin", memory_map="map.json",
                executions=[dict(config_address=external), dict(config_address=head.BASE, expected_post_block_completions=[False])],
                expected=[dict(address=head.BASE + 0xD7000), dict(address=external)])
    head.relocate_head(case, tmp_path, 0x30000000)
    raw = (tmp_path / "initial.bin").read_bytes()
    config = decode_record(raw, schema("execution_config"))
    post = decode_record(raw[0xD0000:], schema("forward_postprocess_config"))
    assert config["output_hidden_base"] == external
    assert config["forward_postprocess_configuration_base"] == 0x300D0000
    assert post["current_state_base"] == 0x300D6000
    assert case["executions"][1]["config_address"] == 0x30000000
    assert case["expected"] == [dict(address=0x300D7000), dict(address=external)]
    assert raw[0xD1000:0xD1020] == image[0xD1000:0xD1020]


def test_regular_connection_rejects_missing_attention_before_payload_or_writes(tmp_path):
    path = tmp_path / "case.json"
    path.write_text(json.dumps(dict(executions=[{}, {}], provenance=dict(control_reference="unused"))))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="attention_checkpoints is missing"):
        testcase.attach_regular_control(path, tmp_path / "missing_capture.json")
    assert list(tmp_path.iterdir()) == [path] and path.read_bytes() == before


def test_head_cli_reports_missing_attention_mapping_before_preparation(tmp_path, monkeypatch):
    import sys
    import prepare_head_testcase as head
    preceding = tmp_path / "layers.json"
    preceding.write_text(json.dumps(dict(executions=[{}])))
    destination = tmp_path / "head"
    monkeypatch.setattr(sys, "argv", ["prepare-head", "--index", "missing-head.json",
        "--output", str(destination), "--preceding-layer-testcase", str(preceding),
        "--regular-control-index", "missing-control.json", "--feature2-reference", "missing-feature2.json"])
    monkeypatch.setattr(head, "prepare", lambda *a, **k: pytest.fail("preparation started with missing Attention mapping"))
    with pytest.raises(ValueError, match="attention_checkpoints is missing"):
        head.main()
    assert not destination.exists()


def test_regular_consumer_rejects_completed_block_before_loading_payload(tmp_path):
    case = tmp_path / "case.json"
    following = tmp_path / "next.json"
    case.write_text(json.dumps(dict(provenance=dict(regular_control=dict(closeout=True, request_end=True)))))
    following.write_text("{}")
    (tmp_path / "unused.json").write_text("{}")
    with pytest.raises(ValueError, match="completed testcase has no following captured control"):
        unified.connect_steps(case, following, tmp_path / "unused.json")


@pytest.mark.parametrize("invalid", [None, "precision", "packed"])
def test_baseline_regular_control_uses_full_sequence_without_dependency(tmp_path, monkeypatch, invalid):
    import torch
    import prepare_head_testcase as head
    from hardware_adapter import control_reference_data
    image = bytearray((head.HARDWARE / "cases/head_state_update_32tokens/initial.bin").read_bytes())
    head.patch_record(image, head.POST-head.BASE, head.load_schema("forward_postprocess_config"),
                      {"flags": 1, "current_state_limit": head.BASE + 0xD6000 + 1024})
    (tmp_path / "initial.bin").write_bytes(image)
    first = decode_record(image, schema("execution_config"))
    first.update(start_layer=31, layer_count=1, total_token_count=34, sequence_length=34,
                 flags=3, refresh_configuration_offset=0, joint_configuration_offset=0)
    (tmp_path / "layer.bin").write_bytes(testcase.pack_record(schema("execution_config"), first))
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(provenance={}, generation_config=dict(full_sequence_recompute=True))))
    reference = tmp_path / "control.json"
    reference.write_text(json.dumps(dict(source_index=str(source), capture_index=2,
                                         sequence_length=34, config=dict(transfer_only=True))))
    before = dict(total_length=34, decoding_mode="fixed_k", packed_state=None,
                  cross_block_prefix_state=None, dynamic_block_lookahead=False,
                  tokens=torch.arange(34).reshape(1, -1))
    prediction = dict(capture_index=2, input_positions=torch.arange(34), row_bits=torch.full((34,), 8))
    if invalid == "precision":
        prediction["row_bits"][0] = 4
    if invalid == "packed":
        before["packed_state"] = {}
    monkeypatch.setattr(control_reference_data, "read_captured_control", lambda _, **kwargs: (
        json.loads(source.read_text()), dict(before_forward=before), prediction))
    case = dict(ddr_image="initial.bin", memory_map="map.json", expected=[],
        executions=[dict(config_address=0x60000000), dict(config_address=head.BASE, expected_post_block_completions=[False])],
        initial_segments=[dict(path="layer.bin", address=0x60000000, bytes=320)],
        provenance=dict(control_reference=str(reference)), head_checkpoints={},
        attention_checkpoints=dict(token_positions=list(range(34)), physical_to_reference_token=[list(range(34))]))
    (tmp_path / "map.json").write_text(json.dumps(dict(memory_map=[
        dict(name="head", base=head.BASE, limit=head.BASE+len(image)),
        dict(name="layer", base=0x60000000, limit=0x60001000)])))
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    if invalid:
        with pytest.raises(ValueError, match="baseline|A8"):
            testcase.attach_regular_control(path, source)
        assert not (tmp_path / "regular_control.bin").exists()
        return
    testcase.attach_regular_control(path, source)
    actual = json.loads(path.read_text())
    control = actual["provenance"]["regular_control"]
    assert control["mode"] == "baseline_full_sequence" and len(actual["executions"]) == 1
    assert set(control["storage"]) == {"token_table", "state", "input_state", "joint_result"}
    data = (tmp_path / "regular_control.bin").read_bytes()
    config = decode_record(data, schema("execution_config"))
    assert config["flags"] == 7 and config["refresh_configuration_offset"] == 0
    assert config["forward_postprocess_configuration_base"] == head.POST
    start, limit = control["storage"]["token_table"]
    base = actual["executions"][0]["config_address"]
    assert np.frombuffer(data[start-base:limit-base], dtype="<u4").tolist() == list(range(34))
    assert not actual["expected"]  # No expected next-token payload initializes execution.


@pytest.mark.parametrize("new_block", [False, True])
def test_baseline_next_step_carries_tokens_not_cache_or_expected_metadata(tmp_path, monkeypatch, new_block):
    import torch
    from types import SimpleNamespace
    from hardware_adapter import control_reference_data
    from prepare_handoff_testcase import read_record
    directories = [tmp_path / name for name in ("previous", "following")]
    for directory in directories:
        directory.mkdir()
        (directory / "dram.ini").write_text("same timing")
    previous_dir, following_dir = directories
    sequence = 66
    hidden = np.full((sequence,4096), 0x3f80, dtype="<u2")
    selected = dict(query_position_ids=None, activation_bits=torch.full((sequence,),8),
        hidden=torch.from_numpy(hidden.view("<i2").copy()).view(torch.bfloat16))
    states = dict(next_layer_inputs=selected, before_forward=dict(block_index=0,block_start=2),
        next_forward=dict(tokens=torch.full((sequence,),3), total_length=sequence,
                          block_index=int(new_block),block_start=34 if new_block else 2))
    previous_root, next_root = tmp_path / "previous_payload", tmp_path / "next_payload"
    def read_control(_, *, payload_root=None):
        assert payload_root == str(previous_root)
        return dict(provenance={}), states, dict(capture_index=2)
    def read_reference(_, payload_root=None):
        assert payload_root == next_root
        return SimpleNamespace(metadata=dict(provenance={}, forward=dict(capture_index=3),
                              generation_config=dict(full_sequence_recompute=True)))
    monkeypatch.setattr(control_reference_data, "read_captured_control", read_control)
    monkeypatch.setattr(testcase, "ReferenceData", read_reference)
    arrays=dict(positions=np.arange(sequence),activation_bits=np.full((sequence,),8),hidden=hidden)
    monkeypatch.setattr(testcase,"captured_layer_view",lambda *_: SimpleNamespace(load=lambda _,name,*a:arrays[name]))
    config=read_record("execution_config",(testcase.HARDWARE/"cases/head_state_update_32tokens/initial.bin").read_bytes()[:320])
    config.update(start_layer=0,flags=3,layer_count=32,sequence_length=sequence,
                  total_token_count=sequence,refresh_configuration_offset=0,joint_configuration_offset=0)
    (following_dir/"initial.bin").write_bytes(testcase.pack_record(schema("execution_config"),config))
    # These expected cache bytes deliberately differ from initialization. Baseline
    # must overwrite the complete cache instead of carrying the previous cache.
    (previous_dir/"cache.bin").write_bytes(b"old")
    (following_dir/"cache.bin").write_bytes(b"new")
    for directory,base in ((previous_dir,0x10000),(following_dir,0x20000)):
        (directory/"map.json").write_text(json.dumps(dict(base_address=base,memory_map=[
            dict(name="step",base=base,limit=base+0x4000)])))
    storage=dict(token_table=[0x11000,0x11000+sequence*4],state=[0x12000,0x12400],
                 input_state=[0x13000,0x13400],joint_result=[0,0])
    next_storage={name:([lo+0x10000,hi+0x10000] if hi else [0,0]) for name,(lo,hi) in storage.items()}
    common=dict(silu_mode="monotone",dramsim3_config="dram.ini",memory_map="map.json",initial_segments=[])
    def control(saved):
        return dict(index="unused",payload_root=str(previous_root),mode="baseline_full_sequence",storage=saved,
                    metadata_base=0x10000,metadata_limit=0x11000)
    case=dict(common,provenance=dict(regular_control=control(storage)),executions=[{}],
              expected=[dict(name="layer0.cache_k",address=0x10500,bytes=3,path="cache.bin")])
    following=dict(common,ddr_image="initial.bin",provenance=dict(regular_control=control(next_storage)),
        executions=[dict(config_address=0x20000)],
        expected=[dict(name="layer0.cache_k",address=0x20500,bytes=3,path="cache.bin")])
    paths=[directory/"case.json" for directory in directories]
    for path,value in zip(paths,(case,following)):
        path.write_text(json.dumps(value))
    index=tmp_path/"index.json"; index.write_text("{}")
    unified.connect_steps(*paths,index,payload_root=next_root)
    actual=json.loads(paths[0].read_text())
    execution=actual["executions"][1]
    assert len(execution["copies"]) == (1 if new_block else 2)
    assert execution["copies"][0] == dict(source_address=storage["token_table"][0],
        destination_address=next_storage["token_table"][0],bytes=sequence*4)
    action=execution["handoff_actions"][0]
    assert action["kind"] == "full_sequence_from_state" and action["sequence"] == sequence
    assert action["source_state_address"] == storage["state"][0]
    assert action["table_address"] == next_storage["token_table"][0]
    assert action["source_current_begin"] == 2 and action["current_begin"] == (34 if new_block else 2)
    assert action["reset_current_state"] is new_block
    assert len([s for s in actual["initial_segments"] if s["path"].startswith("embedding_")]) == 1


def test_regular_consumer_aligns_embedding_rows_after_unaligned_image(tmp_path, monkeypatch):
    import torch
    from types import SimpleNamespace
    from hardware_adapter import control_reference_data
    from prepare_handoff_testcase import read_record
    output, following = tmp_path / "control", tmp_path / "next"
    output.mkdir(); following.mkdir()
    for directory in (output, following):
        (directory / "dram.ini").write_text("same timing")
    hidden = np.repeat(np.array([0x3f80, 0x4000], dtype="<u2")[:, None], 4096, axis=1)
    selected = dict(query_position_ids=torch.tensor([0, 1]), activation_bits=torch.tensor([4, 8]),
                    hidden=torch.from_numpy(hidden.view("<i2").copy()).view(torch.bfloat16))
    states = dict(next_layer_inputs=selected, before_forward=dict(block_index=0),
                  next_forward=dict(tokens=torch.tensor([3, 5]),block_index=0,block_start=0))
    monkeypatch.setattr(control_reference_data, "read_captured_control",
                        lambda _, **kwargs: (dict(provenance={}), states, dict(capture_index=1)))
    monkeypatch.setattr(testcase, "ReferenceData", lambda _, payload_root=None: SimpleNamespace(
        metadata=dict(provenance={}, forward=dict(capture_index=2))))
    arrays = dict(positions=np.array([0, 1]), activation_bits=np.array([4, 8]), hidden=hidden)
    monkeypatch.setattr(testcase, "captured_layer_view", lambda *_:
                        SimpleNamespace(load=lambda stage, name, *args: arrays[name]))
    config = (testcase.HARDWARE / "cases/head_state_update_32tokens/initial.bin").read_bytes()[:320]
    config = read_record("execution_config", config)
    config.update(start_layer=0, flags=0x10, layer_count=2)
    (following / "initial.bin").write_bytes(testcase.pack_record(schema("execution_config"), config))
    hidden[::-1].tofile(following / "hidden.bin")
    (following / "memory_map.json").write_text(json.dumps(dict(base_address=0x20000,
        current_ddr_image_bytes=320, memory_map=[dict(name="next_image", base=0x20000, limit=0x21100)])))
    (output / "memory_map.json").write_text(json.dumps(dict(
        memory_map=[dict(name="control_image", base=0x10000, limit=0x11000)])))
    common = dict(silu_mode="monotone", dramsim3_config="dram.ini", memory_map="memory_map.json")
    case = dict(common, provenance=dict(regular_control=dict(index="unused",
        metadata_base=0x10000, metadata_limit=0x11000, storage=dict(joint_result=[0,0]))), initial_segments=[], expected=[], executions=[{}])
    next_case = dict(common, ddr_image="initial.bin", source=dict(start_layer=0, layers=1,
        packed_logical_to_reference_token=[1, 0]), executions=[dict(config_address=0x20000)],
        expected=[dict(name="hidden", path="hidden.bin", bytes=hidden.nbytes, address=0x21000)])
    path, next_path = output / "case.json", following / "case.json"
    path.write_text(json.dumps(case)); next_path.write_text(json.dumps(next_case))
    (tmp_path / "reference.json").write_text("{}")
    unified.connect_steps(path, next_path, tmp_path / "reference.json")
    actual = json.loads(path.read_text())
    config = read_record("execution_config", (output / "step1.execution.bin").read_bytes())
    assert config["flags"] == 0x10 and config["layer_count"] == 2
    assert config["embedding_base"] == 0x22000
    assert config["embedding_limit"] - config["embedding_base"] == 126464 * 8192
    with pytest.raises(ValueError, match="embedding_base.*aligned"):
        testcase.pack_record(schema("execution_config"), dict(config, embedding_base=0x21100))
    for ordinal, token in enumerate([3, 5]):
        segment = next(x for x in actual["initial_segments"] if x["path"] == f"embedding_{token}.bin")
        assert segment["address"] == config["embedding_base"] + token * 8192
        assert (output / segment["path"]).read_bytes() == hidden[ordinal].tobytes()
    expected = next(item for item in actual["expected"] if item["name"] == "step1.hidden")
    assert (output / expected["path"]).read_bytes() == hidden[::-1].tobytes()
    assert actual["executions"][1]["handoff_actions"][0]["layout_address"] == config["token_metadata_base"]
    path.write_text(json.dumps(case))
    states["next_forward"]["block_index"] = 1
    with pytest.raises(ValueError, match="packed block transition requires the actual L0 scout/deep capture"):
        unified.connect_steps(path, next_path, tmp_path / "reference.json")


def schema(name):
    return json.loads((testcase.HARDWARE / "config" / (name + ".json")).read_text())


def decode_record(data, layout):
    return {
        field["name"]: int.from_bytes(
            data[field["offset"]:field["offset"] + int(field["type"][1:]) // 8],
            "little",
        )
        for field in layout["fields"]
    }


def required_fields(layout):
    return {field["name"]: field["minimum"] for field in layout["fields"]
            if field.get("minimum", 0) > 0 and "constant" not in field}


@pytest.mark.parametrize("count", [96, 352, 384, 432])
def test_refresh_descriptor_preserves_extended_token_count(count):
    layout = schema("token_refresh_config")
    values = dict(required_fields(layout), target_token_count=count, sequence_length=512)
    packed = testcase.pack_record(layout, values)
    assert int.from_bytes(packed[14:16], "little") == count
    with pytest.raises(ValueError, match="target_token_count"):
        testcase.pack_record(layout, dict(values, target_token_count=433))


def test_w4_bytes_preserve_signed_codes_and_output_lane_order():
    # Each output lane starts with a different signed value. The second tile
    # reverses K so tile, lane and nibble swaps cannot pass a constant reference_data.
    codes = np.array([
        [-8, -7, -1, 0, 1, 2, 6, 7],
        [-7, -6, 0, 1, 2, 3, 7, -8],
        [-6, -5, 1, 2, 3, 4, -8, -7],
        [-5, -4, 2, 3, 4, 5, -7, -6],
        [-4, -3, 3, 4, 5, 6, -6, -5],
        [-3, -2, 4, 5, 6, 7, -5, -4],
        [-2, -1, 5, 6, 7, -8, -4, -3],
        [-1, 0, 6, 7, -8, -7, -3, -2],
    ], dtype=np.int8)
    codes = np.concatenate((codes, codes[:, ::-1]))
    packed = testcase.pack_weights(codes)
    payload = packed.tobytes()
    assert len(payload) == codes.size // 2
    assert payload[:8] == bytes.fromhex("98 a9 ba cb dc ed fe 0f")
    assert payload[8:16] == bytes.fromhex("0f 10 21 32 43 54 65 76")
    decoded = np.empty_like(codes)
    for tile in range(2):
        for k_pair in range(4):
            for lane in range(8):
                byte = payload[tile * 32 + k_pair * 8 + lane]
                for half, nibble in enumerate((byte & 15, byte >> 4)):
                    decoded[tile * 8 + lane, 2 * k_pair + half] = (
                        nibble - 16 if nibble >= 8 else nibble)
    np.testing.assert_array_equal(decoded, codes)


@pytest.mark.parametrize("codes", [
    np.zeros((7, 4), dtype=np.int8),
    np.zeros((8, 6), dtype=np.int8),
    np.zeros(32, dtype=np.int8),
    np.zeros((8, 4), dtype=np.int16),
    np.full((8, 4), -9, dtype=np.int8),
    np.full((8, 4), 8, dtype=np.int8),
])
def test_w4_rejects_unrepresentable_or_misaligned_weights(codes):
    with pytest.raises(ValueError):
        testcase.pack_weights(codes)


@pytest.mark.parametrize("required", [set(), {0, 2, 48}, set(range(65))])
def test_output_subset_keeps_positions_sources_and_full_allocation(required):
    bits = [4 if token % 3 else 8 for token in range(65)]
    positions = [1000 - 7 * token for token in range(65)]
    write_token_indices = set(range(0, 65, 2))
    metadata, order, summaries = testcase.pack_tokens(bits, positions, write_token_indices, output_token_indices=required)
    assert set(order[:len(required)]) == required
    assert sorted(order) == list(range(65))
    cursor = ordinal = 0
    layout = schema("token_metadata")
    for token_batch_index, summary in enumerate(summaries):
        header = decode_record(metadata[cursor:], layout["token_batch_header"])
        assert header["flags"] == 2
        assert header["token_batch_index"] == token_batch_index
        assert header["first_token_ordinal"] == ordinal
        tokens = header["resident_token_count"]
        outputs = header["output_token_count"]
        assert outputs == summary["output_tokens"]
        assert 1 <= header["compute_group_count"] <= 4
        required_groups, unused_groups = set(), set()
        for physical in range(tokens):
            entry = decode_record(metadata[cursor + 32 + physical * 16:], layout["token_entry"])
            assert entry["source_index"] == ordinal + entry["token_ordinal"]
            original = order[entry["source_index"]]
            assert entry["token_position"] == entry["kv_index"] == positions[original]
            assert (physical < outputs) == (original in required)
            (required_groups if original in required else unused_groups).add(entry["compute_group"])
            assert bool(header["kv_write_disable_mask"] & (1 << physical)) == (original not in write_token_indices)
        assert required_groups == set(range(header["output_compute_group_count"]))
        assert not required_groups & unused_groups
        ordinal += tokens
        cursor += header["token_batch_bytes"]
    assert ordinal == 65


def test_full_sequence_output_subset_preserves_full_round_count():
    required = set(range(789, 821))
    metadata, order, rounds = testcase.pack_tokens([4] * 1045, list(range(1045)), output_token_indices=required)
    assert len(rounds) == 22
    assert rounds[0] == dict(a4=48, a8=0, tokens=48, groups=3, output_tokens=32)
    assert sum(r["output_tokens"] for r in rounds) == 32
    assert set(order[:32]) == required and sorted(order) == list(range(1045))
    header = decode_record(metadata, schema("token_metadata")["token_batch_header"])
    assert header["output_compute_group_count"] == 2


@pytest.mark.parametrize("limit,bits", [
    (3, [4] * 6), (17, [4] * 33 + [8] * 32), (32, [8] * 65),
])
def test_round_token_limit_preserves_tokens_and_capacity(limit, bits):
    _, order, rounds = testcase.pack_tokens(bits, list(range(len(bits))), round_token_limit=limit)
    assert sorted(order) == list(range(len(bits)))
    assert all(0 < r["tokens"] <= limit and r["a4"] + 2 * r["a8"] <= 64 for r in rounds)
    assert sum(r["tokens"] for r in rounds) == len(bits)
    with pytest.raises(ValueError, match="more than 64"):
        testcase.pack_tokens([4] * 65, list(range(65)), round_token_limit=1)


@pytest.mark.parametrize("limit", [0, 49, True, "3"])
def test_invalid_round_token_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="round_token_limit"):
        testcase.pack_tokens([4], [0], round_token_limit=limit)


@pytest.mark.parametrize("phase", ["local_block", "local_confirmation", "local_forced_finish"])
def test_last_layer_output_consumers_include_future_and_retained_hidden(phase):
    metadata = dict(model_layer_index=31, layer_count=1, forward=dict(
        forward_kind=phase, prediction_positions=[17],
        future_prediction_positions=[29], retained_hidden_positions=[42]))
    assert testcase.last_layer_output_token_indices(metadata, [42, 17, 29, 100]) == {0, 1, 2}
    metadata["forward"].pop("future_prediction_positions")
    with pytest.raises(ValueError, match="explicit future_prediction_positions"):
        testcase.last_layer_output_token_indices(metadata, [42, 17, 29, 100])
    metadata["forward"]["future_prediction_positions"] = []
    metadata["model_layer_index"] = 0
    with pytest.raises(ValueError, match="L31"):
        testcase.last_layer_output_token_indices(metadata, [42, 17, 29, 100])


def test_last_layer_selected_boundary_consumers_are_supported():
    metadata = dict(model_layer_index=31, layer_count=1, forward=dict(
        forward_kind="boundary_refresh", layer0_global_selected_deep=True,
        prediction_positions=[821, 822], future_prediction_positions=[]))
    assert testcase.last_layer_output_token_indices(metadata, [3, 821, 822, 900]) == {1, 2}
    metadata["forward"]["isolated_groups"] = True
    with pytest.raises(ValueError, match="isolated groups"):
        testcase.last_layer_output_token_indices(metadata, [3, 821, 822, 900])


@pytest.mark.parametrize("a4,a8,round_count", [
    (1, 0, 1), (0, 1, 1),
    (48, 0, 1), (49, 0, 2),
    (0, 32, 1), (0, 33, 2),
    (32, 16, 1), (31, 17, 2),
    (40, 40, 2), (16, 64, 3),
    (2048, 0, 43), (0, 2048, 64), (1024, 1024, 48),
])
def test_round_metadata_bijection_capacity_phases_and_positions(a4, a8, round_count):
    layout = schema("token_metadata")
    rng = np.random.default_rng(120)
    bits = rng.permutation(np.array([4] * a4 + [8] * a8, dtype=np.int8))
    positions = rng.permutation(2048)[:len(bits)]
    metadata, order, summaries = testcase.pack_tokens(bits, positions)
    assert len(summaries) == round_count <= layout["maximum_token_batches"]
    assert sorted(order) == list(range(len(bits)))
    cursor, ordinal, decoded_positions = 0, 0, []
    for token_batch_index in range(round_count):
        header = decode_record(metadata[cursor:], layout["token_batch_header"])
        tokens = header["resident_token_count"]
        assert 1 <= tokens <= layout["maximum_resident_tokens"]
        assert 1 <= header["compute_group_count"] <= 4
        assert header["token_batch_index"] == token_batch_index
        assert header["first_token_ordinal"] == ordinal
        assert header["token_entry_bytes"] == layout["token_entry"]["size_bytes"]
        assert header["semantic_group_count"] == 1
        assert header["flags"] == header["kv_write_disable_mask"] == 0
        assert header["metadata_version"] == header["capture_index"] == 1
        assert header["inverse_offset"] == layout["token_batch_header"]["size_bytes"] + tokens * 16
        assert header["inverse_bytes"] == ((tokens + 15) // 16) * 16
        assert header["token_batch_bytes"] == header["inverse_offset"] + header["inverse_bytes"]
        assert header["token_batch_bytes"] % layout["alignment_bytes"] == 0
        record = metadata[cursor:cursor + header["token_batch_bytes"]]
        assert len(record) == header["token_batch_bytes"]
        inverse = record[header["inverse_offset"]:header["inverse_offset"] + tokens]
        assert sorted(inverse) == list(range(tokens))
        assert not any(record[header["inverse_offset"] + tokens:])
        occupied, token_ordinals, counts, groups = set(), set(), {4: 0, 8: 0}, set()
        for physical in range(tokens):
            entry = decode_record(record[32 + physical * 16:], layout["token_entry"])
            local = entry["token_ordinal"]
            assert local not in token_ordinals and 0 <= local < tokens
            token_ordinals.add(local)
            assert inverse[local] == physical
            assert entry["source_index"] == ordinal + local
            original = order[entry["source_index"]]
            assert entry["activation_bits"] == bits[original]
            assert entry["token_position"] == entry["kv_index"] == positions[original]
            decoded_positions.append(entry["token_position"])
            assert entry["source"] == entry["query_group"] == entry["cache_group"] == 0
            group, slot, mask = entry["compute_group"], entry["pe_slot"], entry["phase_mask"]
            assert 0 <= group < header["compute_group_count"] and 0 <= slot < 8
            groups.add(group)
            assert mask == 3 if entry["activation_bits"] == 8 else mask in (1, 2)
            for phase in range(2):
                if mask & (1 << phase):
                    address = (group, slot, phase)
                    assert address not in occupied, "two tokens use the same PE slot and phase"
                    occupied.add(address)
            counts[entry["activation_bits"]] += 1
        assert groups == set(range(header["compute_group_count"]))
        assert len(occupied) == counts[4] + 2 * counts[8] <= 64
        assert summaries[token_batch_index] == dict(a4=counts[4], a8=counts[8], tokens=tokens,
                                               groups=header["compute_group_count"])
        ordinal += tokens
        cursor += header["token_batch_bytes"]
    assert cursor == len(metadata) and ordinal == len(bits)
    assert sorted(decoded_positions) == sorted(positions.tolist())


@pytest.mark.parametrize("name,values", [
    ("execution_config", dict(layer_count=2, total_token_count=80, sequence_length=2048,
                                   output_hidden_base=0x123456789ABCDEF0)),
    ("layer_address_table", dict(query_base_weight_base=0x123456789ABCDEF0)),
])
def test_descriptor_format_matches_schema_constants_sizes_and_endianness(name, values):
    layout = schema(name)
    packed = testcase.pack_record(layout, dict(required_fields(layout), **values))
    assert len(packed) == layout["size_bytes"]
    decoded = decode_record(packed, layout)
    for key, value in values.items():
        assert decoded[key] == value
    for field in layout["fields"]:
        if "constant" in field:
            value = field["constant"]
            assert decoded[field["name"]] == (int(value, 0) if isinstance(value, str) else value)
    assert bytes.fromhex("f0 de bc 9a 78 56 34 12") in packed


@pytest.mark.parametrize("changes", [
    dict(unrecognized=1), dict(magic=0), dict(layer_count=0),
    dict(total_token_count=2049), dict(sequence_length=-1), dict(flags=0x80),
    dict(flags=0x4000),
])
def test_descriptor_rejects_invalid_fields_constants_ranges_and_flag_bits(changes):
    layout = schema("execution_config")
    values = required_fields(layout)
    values.update(changes)
    with pytest.raises(ValueError):
        testcase.pack_record(layout, values)


@pytest.mark.parametrize("flags", [0x10, 0x30, 0x50])
def test_descriptor_accepts_supported_ffn_group_flag_bits(flags):
    layout = schema("execution_config")
    packed = testcase.pack_record(layout, dict(required_fields(layout), flags=flags))
    assert decode_record(packed, layout)["flags"] == flags


@pytest.mark.parametrize("flags", [0x20, 0x40, 0x70])
def test_descriptor_rejects_inconsistent_ffn_group_flag_bits(flags):
    layout = schema("execution_config")
    with pytest.raises(ValueError):
        testcase.pack_record(layout, dict(required_fields(layout), flags=flags))


def minimal_reference_data(root, *, bits=(4, 8), positions=(3, 1), **overrides):
    tensors = []
    for name, values, dtype in (("activation_bits", bits, "|i1"), ("positions", positions, "<i8")):
        array = np.asarray(values, dtype=dtype)
        path = root / (name + ".bin")
        array.tofile(path)
        tensors.append(dict(role="input", name=name, path=path.name, shape=list(array.shape),
                            dtype=dtype, encoding="integer", byte_count=array.nbytes))
    metadata = dict(schema="supra-layer-reference-data/v1", hidden=4096, ffn=12288,
                    heads=32, tokens=2, sequence=4, tensors=tensors)
    metadata.update(overrides)
    index = root / "reference_data.json"
    index.write_text(json.dumps(metadata))
    return index


@pytest.mark.parametrize("changes,message", [
    (dict(schema="unsupported"), "SUPRA layer reference data"),
    (dict(a4_clip_ratios={"q_proj": 0.8}), "installed/ratio input records"),
    (dict(hidden=128), "D4096/F12288/H32"),
    (dict(tokens=0), "1 <= tokens"),
    (dict(sequence=2049), "1 <= tokens"),
    (dict(layer_count=33), "within L0..L31"),
    (dict(model_layer_index=32), "within L0..L31"),
    (dict(bits=(4, 6)), "token bits"),
    (dict(bits=(4,)), "token bits"),
    (dict(positions=(1, 1)), "unique logical positions"),
    (dict(positions=(-1, 1)), "unique logical positions"),
    (dict(positions=(4, 1)), "unique logical positions"),
    (dict(positions=(1,)), "unique logical positions"),
    (dict(attention_mask=True), "visibility/selection"),
    (dict(kv_write_token_indices=[2]), "invalid kv_write_token_indices"),
    (dict(kv_write_token_indices=[0], cache_commit_token_indices=[1]), "subset of KV write tokens"),
    (dict(layer_count=1, layers=[dict(layer=1)]), "consecutive"),
    (dict(layer_count=1, layers=[dict(positions=[1, 3])]), "per-layer positions"),
])
def test_prepare_rejects_invalid_input_before_creating_payload(tmp_path, changes, message):
    index = minimal_reference_data(tmp_path, **changes)
    output = tmp_path / "unused-output"
    with pytest.raises(ValueError, match=message):
        testcase.prepare(index, output)
    assert not output.exists()


def test_reference_data_rejects_duplicate_names_and_corrupt_raw_payload(tmp_path):
    index = minimal_reference_data(tmp_path)
    metadata = json.loads(index.read_text())
    metadata["tensors"].append(metadata["tensors"][0])
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="duplicate tensor names"):
        testcase.ReferenceData(index)
    index = minimal_reference_data(tmp_path)
    reference_data = testcase.ReferenceData(index)
    with pytest.raises(ValueError, match="unexpected raw dtype"):
        reference_data.load("input", "activation_bits", "<u2")
    (tmp_path / "activation_bits.bin").write_bytes(b"\x04")
    with pytest.raises(ValueError, match="byte count differs"):
        reference_data.load("input", "activation_bits")


def test_reference_data_rejects_partial_raw_element(tmp_path):
    index = minimal_reference_data(tmp_path)
    metadata = json.loads(index.read_text())
    entry = metadata["tensors"][1]
    entry["byte_count"] += 1
    with (tmp_path / entry["path"]).open("ab") as stream:
        stream.write(b"\xff")
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="byte count differs"):
        testcase.ReferenceData(index).load("input", "positions")


def add_tensors(index, values):
    metadata = json.loads(index.read_text())
    replaced = set(values)
    metadata["tensors"] = [e for e in metadata["tensors"] if (e["role"], e["name"]) not in replaced]
    for (role, name), array in values.items():
        array = np.asarray(array)
        path = index.parent / (role + "." + name + ".bin")
        array.tofile(path)
        metadata["tensors"].append(dict(role=role, name=name, path=path.name,
            shape=list(array.shape), dtype=array.dtype.str, byte_count=array.nbytes,
            encoding="bf16_raw" if array.dtype == np.dtype("<u2") else "integer"))
    index.write_text(json.dumps(metadata))


def semantic_reference_data(root, layers=1, explicit=True, selected_rope=False):
    index = minimal_reference_data(root, layer_count=layers, layers=[dict(layer=i) for i in range(layers)])
    hidden = np.full((1, 2, 4096), 0x3f80, dtype="<u2")
    values = {("input", "hidden"): hidden}
    for name, table in testcase.hardware_softmax_tables().items():
        values["input", "softmax_" + name + "_lut"] = table
    if explicit:
        values["input", "silu_breakpoints"] = (np.asarray([
            -8., -6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6., 8.
        ], dtype="<f4").view("<u4") >> 16).astype("<u2")
    for layer in range(layers):
        if layer:
            values["input", f"layer{layer}.hidden"] = hidden.copy()
        if layer < layers - 1:
            values["expected", f"layer{layer}.block_output"] = hidden.copy()
        if explicit:
            values["input", f"layer{layer}.silu_coefficients.0"] = np.full(16, 0x3f80, dtype="<u2")
            values["input", f"layer{layer}.silu_coefficients.1"] = np.full(16, 0x8000, dtype="<u2")
        for name, raw in (("cos", 0x3f80), ("sin", 0)):
            key = f"layer{layer}.rope_selected_{name}" if selected_rope else "rope_cosine" if name == "cos" else "rope_sine"
            values["input", key] = np.full((1, 1, 2 if selected_rope else 4, 128), raw, dtype="<u2")
    add_tensors(index, values)
    return index


def validate_semantics(index):
    reference_data = testcase.ReferenceData(index)
    info = reference_data.metadata
    return testcase.validate_layer_inputs(reference_data, info["layer_count"], info["layers"],
                                    reference_data.load("input", "activation_bits"), reference_data.load("input", "positions"))


def combined_boundary_reference_data(root):
    index = semantic_reference_data(root, layers=2)
    reference_data = testcase.ReferenceData(index)
    values = {}
    for layer in range(2):
        for name in ("hidden", "positions", "activation_bits", "softmax_exp_lut", "softmax_reciprocal_lut"):
            values["input", f"layer{layer}." + name] = reference_data.load("input", name)
    add_tensors(index, values)
    metadata = json.loads(index.read_text())
    metadata.update(reference_data_kind="real_cuda_consecutive_layers",
        connection=dict(source="layer0.block_output", destination="layer1.hidden",
                        mapping="index_select", source_token_indices=[0, 1]),
        layers=[dict(layer=i, tokens=2, positions=[3, 1], kv_write_positions=[3, 1],
                     cache_commit_positions=[3, 1]) for i in range(2)])
    index.write_text(json.dumps(metadata))
    return index


def test_captured_layer_view_preserves_selected_cache_and_input_mapping(tmp_path):
    index = combined_boundary_reference_data(tmp_path)
    metadata = json.loads(index.read_text())
    metadata["layers"][1]["cache_commit_positions"] = [1]
    metadata["connections"] = [metadata["connection"]]
    index.write_text(json.dumps(metadata))
    source = testcase.ReferenceData(index)
    view = testcase.captured_layer_view(source, 1)
    assert view.metadata["model_layer_index"] == 1
    assert view.metadata["kv_write_token_indices"] == [0, 1]
    assert view.metadata["cache_commit_token_indices"] == [1]
    assert "connection" not in view.metadata
    assert "connections" not in view.metadata
    assert not any(name.startswith("layer1.") for _, name in view.entries)
    np.testing.assert_array_equal(view.load("input", "layer0.hidden"), source.load("input", "layer1.hidden"))
    np.testing.assert_array_equal(view.load("input", "hidden"), source.load("input", "layer1.hidden"))
    assert source.metadata["layer_count"] == 2
    with pytest.raises(ValueError, match="one record"):
        testcase.captured_layer_view(source, 2)


def test_partial_multilayer_view_remaps_local_names_and_tensor_references(tmp_path):
    index = combined_boundary_reference_data(tmp_path)
    metadata = json.loads(index.read_text())
    metadata["model_layer_index"] = 30
    for i, record in enumerate(metadata["layers"]):
        record["model_layer"] = 30 + i
        record["kwargs"] = {"query_position_ids": {"tensor": f"layer{i}.positions"}}
    index.write_text(json.dumps(metadata))
    source = testcase.ReferenceData(index)
    view = testcase.captured_layer_view(source, 31)
    assert view.metadata["layers"][0]["layer"] == 0
    assert view.metadata["layers"][0]["model_layer"] == 31
    assert view.metadata["layers"][0]["kwargs"]["query_position_ids"]["tensor"] == "layer0.positions"
    np.testing.assert_array_equal(view.load("input", "layer0.hidden"), source.load("input", "layer1.hidden"))
    assert not any(name.startswith("layer1.") for _, name in view.entries)


def test_masked_l31_capture_keeps_executed_current_hidden_only():
    info = dict(model_layer_index=31, layer_count=1,
        forward=dict(forward_kind="local_block", prediction_positions=[8, 9, 10, 11],
                     prediction_mask=[[False, True, False, True]], future_prediction_positions=[]),
        control_observation=dict(checkpoints=dict(before_postprocess=dict(block_start=8, block_end=12))))
    # Position 8 is locked and absent from query; 10 is locked but was refreshed.
    # Prefix position 2 participates in Attention but has no output consumer.
    assert testcase.last_layer_output_token_indices(info, [2, 9, 10, 11]) == {1, 2, 3}
    info.pop("control_observation")
    with pytest.raises(ValueError, match="observed block interval"):
        testcase.last_layer_output_token_indices(info, [2, 9, 10, 11])


@pytest.mark.parametrize("options,message", [
    ({"attention_mask": True}, "visibility/selection"),
    ({"isolated_groups": [[0], [1]]}, "visibility/selection"),
    ({"attention_bias": {"tensor": "bias"}}, "attention_bias"),
    ({"use_cache": False}, "use_cache=False"),
])
def test_boundary_rejects_forward_semantics_before_splitting_views(tmp_path, options, message):
    index = combined_boundary_reference_data(tmp_path)
    testcase.boundary_views(index)
    metadata = json.loads(index.read_text())
    metadata["forward"] = options
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        testcase.boundary_views(index)


def test_boundary_continuation_uses_explicit_payload_root(tmp_path):
    payload = tmp_path / "payload"
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    for layer, name in enumerate(("scout", "deep")):
        directory = payload / name
        directory.mkdir(parents=True)
        index = minimal_reference_data(directory, model_layer_index=layer)
        hidden = np.full((1, 2, 4096), 0x3f80, dtype="<u2")
        add_tensors(index, {("input", "hidden"): hidden,
                            ("expected", f"layer{layer}.block_output"): hidden})
        metadata = json.loads(index.read_text())
        if layer == 0:
            metadata.update(boundary_scout={"selection": "given"}, continuation_index="deep.json")
        for entry in metadata["tensors"]:
            entry["path"] = name + "/" + entry["path"]
        (indexes / (name + ".json")).write_text(json.dumps(metadata))
    views, tokens = testcase.boundary_views(indexes / "scout.json", payload)
    assert tokens == [0, 1]
    assert all(view.root == payload for view in views)


@pytest.mark.parametrize("key,encoding", [
    (("input", "hidden"), "integer"),
    (("input", "epsilon"), "bf16_raw"),
    (("expected", "activation_codes"), "bf16_raw"),
])
def test_head_rejects_incorrect_raw_encoding(tmp_path, key, encoding):
    import prepare_head_testcase as head

    values = {("input", "hidden"): np.zeros((1, 32, 4096), dtype="<u2"),
        ("input", "norm_weight"): np.full(4096, 0x3f80, dtype="<u2"),
        ("input", "weight_codes"): np.zeros((64, 4096), dtype="|i1"),
        ("input", "weight_scale"): np.full(64, 0x3f80, dtype="<u2"),
        ("input", "positions"): np.arange(32, dtype="<i8"),
        ("input", "prediction_mask"): np.ones((1, 32), dtype="|b1"),
        ("input", "epsilon"): np.array(0x3727c5ac, dtype="<u4")}
    for name, (dtype, shape) in head.EXPECTED.items():
        values["expected", name] = np.zeros(shape or (32, 64), dtype=dtype)
    index = minimal_reference_data(tmp_path, reference_data_kind="real_cuda_final_output", tokens=32,
                            weight_bits=8, activation_bits=8, vocab=64)
    add_tensors(index, values)
    metadata = json.loads(index.read_text())
    for entry in metadata["tensors"]:
        if entry["name"] == "epsilon":
            entry["encoding"] = "float32_raw"
    index.write_text(json.dumps(metadata))
    head.load_reference_data(index)
    for entry in metadata["tensors"]:
        if (entry["role"], entry["name"]) == key:
            entry["encoding"] = encoding
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="encoding"):
        head.load_reference_data(index)


@pytest.mark.parametrize("layers,explicit,selected", [(1, True, False), (2, True, True), (32, True, True)])
def test_layer_constants_and_identity_connection_are_preserved(tmp_path, layers, explicit, selected):
    index = semantic_reference_data(tmp_path, layers, explicit, selected)
    segments, rope = validate_semantics(index)
    assert segments == ([0x80003f80] * 16 if explicit else None)
    assert len(rope) == layers
    for tables in rope:
        assert tables["cos"].shape == (2048, 128)
        np.testing.assert_array_equal(tables["cos"][[3, 1]], np.full((2, 128), 0x3f80))
        assert not tables["sin"].any()
        assert not tables["cos"][4:].any()


@pytest.mark.parametrize("kind,message", [
    ("breakpoints", "SiLU breakpoints"),
    ("broadcast_intercept", "each contain 16"),
    ("missing_intercept", "both slope and intercept"),
    ("different_silu", "same explicit SiLU"),
    ("missing_layer_silu", "each layer requires explicit SiLU"),
    ("softmax_exp", "softmax exp LUT"),
    ("layer_softmax_reciprocal", "differs from hardware"),
    ("rope_halves", "identical paired halves"),
    ("layer_rope", "per-layer RoPE tables differ"),
    ("hidden_connection", "preceding hidden output"),
    ("initial_hidden", "preceding hidden output"),
    ("partial_write", "per-layer layer1.kv_write_positions"),
    ("partial_commit", "per-layer layer1.cache_commit_positions"),
    ("changed_bits", "per-layer layer1.activation_bits"),
    ("changed_positions", "per-layer layer1.positions"),
])
def test_prepare_rejects_incompatible_numeric_inputs_before_image(tmp_path, kind, message):
    index = semantic_reference_data(tmp_path, layers=2, explicit=True, selected_rope=True)
    reference_data = testcase.ReferenceData(index)
    if kind == "breakpoints":
        array = reference_data.load("input", "silu_breakpoints")
        array[0] ^= 1
        change = {("input", "silu_breakpoints"): array}
    elif kind == "broadcast_intercept":
        change = {("input", "layer0.silu_coefficients.1"): np.zeros(1, dtype="<u2")}
    elif kind in ("missing_intercept", "missing_layer_silu"):
        metadata = json.loads(index.read_text())
        names = {"layer0.silu_coefficients.1"} if kind == "missing_intercept" else {
            "layer1.silu_coefficients.0", "layer1.silu_coefficients.1"}
        metadata["tensors"] = [e for e in metadata["tensors"] if e["name"] not in names]
        index.write_text(json.dumps(metadata))
        change = {}
    elif kind == "different_silu":
        change = {("input", "layer1.silu_coefficients.0"): np.zeros(16, dtype="<u2")}
    elif kind in ("softmax_exp", "layer_softmax_reciprocal"):
        name = "softmax_exp_lut" if kind == "softmax_exp" else "softmax_reciprocal_lut"
        array = reference_data.load("input", name)
        array[5] ^= 1
        change = {("input", name if kind == "softmax_exp" else "layer1." + name): array}
    elif kind in ("rope_halves", "layer_rope"):
        array = reference_data.load("input", "layer1.rope_selected_cos")
        array[0, 0, 0, 0] ^= 1
        if kind == "layer_rope":
            array[0, 0, 0, 64] ^= 1
        change = {("input", "layer1.rope_selected_cos"): array}
    elif kind in ("hidden_connection", "initial_hidden"):
        array = reference_data.load("input", "hidden")
        array[0, 0, 0] = 0x8000
        change = {("input", ("layer1" if kind == "hidden_connection" else "layer0") + ".hidden"): array}
    else:
        name, values = {
            "partial_write": ("kv_write_positions", [3]),
            "partial_commit": ("cache_commit_positions", [1]),
            "changed_bits": ("activation_bits", [8, 4]),
            "changed_positions": ("positions", [1, 3]),
        }[kind]
        change = {("input", "layer1." + name): np.asarray(values, dtype="<i8")}
    add_tensors(index, change)
    output = tmp_path / "rejected-image"
    with pytest.raises(ValueError, match=message):
        testcase.prepare(index, output)
    assert not output.exists()


@pytest.mark.parametrize("kind,message", [
    ("descriptor_count", "descriptor count"),
    ("index_select", "consecutive-layer token mapping"),
    ("source_token_indices", "consecutive-layer token mapping"),
    ("isolated_groups", "visibility/selection"),
    ("refresh_subset", "full query mapping"),
    ("attention_bias", "attention_bias"),
    ("query_mapping", "query_position_ids"),
])
def test_prepare_reports_inconsistent_forward_and_layer_mapping(tmp_path, kind, message):
    index = semantic_reference_data(tmp_path, layers=2)
    metadata = json.loads(index.read_text())
    if kind == "descriptor_count":
        metadata["layers"].pop()
    elif kind in ("index_select", "source_token_indices"):
        metadata["connection"] = dict(source="layer0.block_output", destination="layer1.hidden",
                                      mapping="identity", source_token_indices=[0, 1])
        metadata["connection"].update(mapping="index_select" if kind == "index_select" else "identity",
                                      source_token_indices=[1, 0])
    elif kind == "isolated_groups":
        metadata["forward"] = {"isolated_groups": True}
    elif kind == "refresh_subset":
        metadata["forward"] = {"refresh_positions": [3]}
    elif kind == "attention_bias":
        metadata["layers"][1]["kwargs"] = {"attention_bias": {"tensor": "unused"}}
    else:
        metadata["layers"][1]["kwargs"] = {"query_position_ids": [[1, 3]]}
    index.write_text(json.dumps(metadata))
    output = tmp_path / "rejected-image"
    with pytest.raises(ValueError, match=message):
        testcase.prepare(index, output)
    assert not output.exists()


def test_explicit_silu_defines_preserve_all_raw_segments():
    from run_testcase import silu_defines
    segments = [0x80003f80 + i for i in range(16)]
    defines = silu_defines(dict(silu_mode="explicit", silu_segments=segments))
    assert len(defines) == 16
    for index, define in enumerate(defines):
        name, literal = define.split("=", 1)
        assert name == "-DSUPRA_SILU_SEGMENT_" + str(index)
        assert literal.startswith("32'h")
        assert int(literal[4:], 16) == segments[index]


@pytest.mark.parametrize("case", [
    dict(silu_mode="unknown"), dict(silu_mode="explicit", silu_segments=[0] * 15),
    dict(silu_mode="explicit", silu_segments=[-1] * 16),
    dict(silu_mode="explicit", silu_segments=[0x100000000] * 16),
    dict(silu_mode="explicit", silu_segments=[True] * 16),
    dict(silu_mode="unknown", silu_segments=[0] * 16),
])
def test_silu_defines_reject_invalid_mode_or_raw_segments(case):
    from run_testcase import silu_defines
    with pytest.raises(ValueError):
        silu_defines(case)


def shaped_reference_data():
    from types import SimpleNamespace
    shapes = {("input", "hidden"): (1, 3, 4096),
              ("input", "activation_bits"): (3,), ("input", "positions"): (3,),
              ("input", "v_scale"): (32,), ("input", "attn_norm.weight"): (4096,),
              ("expected", "block_output"): (1, 3, 4096)}
    for name, shape in (("q_proj", (4096, 4096)), ("k_proj", (4096, 4096)),
                        ("v_proj", (4096, 4096)), ("attn_out", (4096, 4096)),
                        ("ff_proj", (12288, 4096)), ("up_proj", (12288, 4096)),
                        ("ff_out", (4096, 12288))):
        shapes["input", "weight." + name + ".codes"] = shape
        shapes["input", "weight." + name + ".scale"] = (shape[0],)
    for name, shape in (("key_codes", (1, 32, 17, 128)),
                        ("value_codes", (1, 32, 17, 128)), ("key_scale", (1, 32, 17, 1))):
        shapes["input", "retained_" + name] = shape
        shapes["expected", "cache_" + name] = shape
    return SimpleNamespace(entries={key: {"shape": list(value)} for key, value in shapes.items()})


def test_supported_tensor_axes_and_singleton_batch_wrappers():
    reference_data = shaped_reference_data()
    testcase.validate_tensor_shapes(reference_data, 1, 3, 17)
    reference_data.entries["input", "weight.ff_proj.codes"]["shape"] = [1, 12288, 4096]
    reference_data.entries["input", "v_scale"]["shape"] = [1, 32, 1, 1]
    reference_data.entries["input", "retained_key_codes"]["shape"] = [32, 17, 128]
    testcase.validate_tensor_shapes(reference_data, 1, 3, 17)


@pytest.mark.parametrize("role,name,shape", [
    ("input", "weight.ff_proj.codes", [4096, 12288]),
    ("input", "weight.ff_out.codes", [12288, 4096]),
    ("input", "weight.ff_proj.scale", [96, 128]),
    ("input", "retained_value_codes", [1, 17, 32, 128]),
    ("expected", "cache_key_scale", [1, 17, 32, 1]),
    ("input", "v_scale", [4, 8]),
    ("input", "v_scale", [33]),
    ("input", "attn_norm.weight", [32, 128]),
    ("expected", "block_output", [1, 4096, 3]),
])
def test_tensor_shapes_reject_transposed_equal_size_or_oversized_data(role, name, shape):
    reference_data = shaped_reference_data()
    reference_data.entries[role, name]["shape"] = shape
    with pytest.raises(ValueError, match="tensor shape mismatch"):
        testcase.validate_tensor_shapes(reference_data, 1, 3, 17)


def test_ddr_put_stays_inside_one_reserved_region():
    image = np.zeros(32, dtype=np.uint8)
    regions = [dict(base=testcase.BASE, limit=testcase.BASE + 16),
               dict(base=testcase.BASE + 16, limit=testcase.BASE + 32)]
    testcase.put_region(image, regions, testcase.BASE + 12, bytes([1, 2, 3, 4]))
    assert image[12:16].tolist() == [1, 2, 3, 4]
    assert not image[16:].any()
    before = image.copy()
    with pytest.raises(ValueError, match="crosses reserved region"):
        testcase.put_region(image, regions, testcase.BASE + 12, bytes(8))
    np.testing.assert_array_equal(image, before)
    with pytest.raises(ValueError, match="exceeds image"):
        testcase.put_region(image, [dict(base=testcase.BASE, limit=testcase.BASE + 64)], testcase.BASE + 30, bytes(4))


@pytest.mark.parametrize("location,name", [
    ("forward", "attention_mask"), ("forward", "isolated_groups"),
    ("kwargs", "attention_mask"), ("kwargs", "isolated_groups"),
])
def test_nested_masks_and_groups_are_rejected(tmp_path, location, name):
    index = semantic_reference_data(tmp_path, layers=2)
    metadata = json.loads(index.read_text())
    if location == "forward":
        metadata["forward"] = {name: [1]}
    else:
        metadata["layers"][1]["kwargs"] = {name: [1]}
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="visibility/selection"):
        validate_semantics(index)


def small_preparation_reference_data(root, monkeypatch, *, layer=0, positions=(5, 1, 3), bits=(8, 4, 4),
                              writes=None, commits=None, hidden=None, output=None):
    root.mkdir()
    monkeypatch.setattr(testcase, "FFN", 128)
    monkeypatch.setattr(testcase, "HEADS", 1)
    tokens, sequence = len(bits), 6
    index = minimal_reference_data(root, bits=bits, positions=positions, hidden=4096, ffn=128,
                            heads=1, tokens=tokens, sequence=sequence, model_layer_index=layer)
    values = {("input", "hidden"): np.full((1, tokens, 4096), 0x3f80, dtype="<u2") if hidden is None else hidden,
              ("expected", "block_output"): np.full((1, tokens, 4096), 0x4100, dtype="<u2") if output is None else output,
              ("input", "v_scale"): np.array([0x3f80], dtype="<u2"),
              ("input", "rope_cosine"): np.full((1, 1, sequence, 128), 0x3f80, dtype="<u2"),
              ("input", "rope_sine"): np.zeros((1, 1, sequence, 128), dtype="<u2")}
    for name, table in testcase.hardware_softmax_tables().items():
        values["input", "softmax_" + name + "_lut"] = table
    values["input", "silu_breakpoints"] = (np.asarray([
        -8., -6., -4., -3., -2., -1.5, -1., -.5, 0., .5, 1., 1.5, 2., 3., 4., 6., 8.
    ], dtype="<f4").view("<u4") >> 16).astype("<u2")
    values["input", "silu_coefficients.0"] = np.full(16, 0x3f80, dtype="<u2")
    values["input", "silu_coefficients.1"] = np.zeros(16, dtype="<u2")
    for name in testcase.LINEARS:
        height, width = (128 if name in ("ff_proj", "up_proj") else 4096), (128 if name == "ff_out" else 4096)
        values["input", "weight." + name + ".codes"] = np.full((height, width), -3, dtype=np.int8)
        values["input", "weight." + name + ".scale"] = np.full(height, 0x3f80, dtype="<u2")
    for name in ("key_codes", "value_codes", "key_scale"):
        shape = (1, 1, sequence, 1 if name == "key_scale" else 128)
        dtype = "<u2" if name == "key_scale" else "|i1"
        values["input", "retained_" + name] = np.zeros(shape, dtype=dtype)
        values["expected", "cache_" + name] = np.ones(shape, dtype=dtype)
    add_tensors(index, values)
    metadata = json.loads(index.read_text())
    if writes is not None:
        metadata["kv_write_token_indices"] = writes
    if commits is not None:
        metadata["cache_commit_token_indices"] = commits
    index.write_text(json.dumps(metadata))
    return index


def read_prepared_record(case_path, address, layout):
    case = json.loads(case_path.read_text())
    manifest = json.loads((case_path.parent / case["memory_map"]).read_text())
    with (case_path.parent / case["ddr_image"]).open("rb") as stream:
        stream.seek(address - manifest["base_address"])
        return decode_record(stream.read(layout["size_bytes"]), layout)


def test_reduced_reference_data_rejects_pre_l0_cache_for_uncomputed_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(
        tmp_path / "input", monkeypatch, writes=[0], commits=[0])
    tokens, sequence, heads, dim = 3, 6, 1, 128
    key_codes = np.zeros((heads, sequence, dim), dtype="|i1")
    key_scale = np.zeros((heads, sequence, 1), dtype="<u2")
    value_codes = np.zeros((heads, sequence, dim), dtype="|i1")
    new_key_codes = np.ones((heads, tokens, dim), dtype="|i1")
    new_key_scale = np.ones((heads, tokens, 1), dtype="<u2")
    new_value_codes = np.ones((heads, tokens, dim), dtype="|i1")
    key_codes[:, 5] = new_key_codes[:, 0]
    key_scale[:, 5] = new_key_scale[:, 0]
    value_codes[:, 5] = new_value_codes[:, 0]
    # Position 1 is not recomputed. A nonzero captured QK value there cannot
    # be produced from the zero retained cache in this testcase.
    key_codes[:, 1, 0] = 7
    add_tensors(index, {
        ("expected", "qk_key_codes"): key_codes,
        ("expected", "qk_key_scale"): key_scale,
        ("expected", "pv_value_codes"): value_codes,
        ("expected", "new_key_codes"): new_key_codes,
        ("expected", "new_key_scale"): new_key_scale,
        ("expected", "new_value_codes"): new_value_codes,
    })
    with pytest.raises(ValueError, match="cache overlay differs from captured qk_key_codes"):
        testcase.prepare(index, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_reduced_reference_data_rejects_stale_persistent_cache_expected(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(
        tmp_path / "input", monkeypatch, writes=[0], commits=[0])
    tokens, sequence, heads, dim = 3, 6, 1, 128
    key_codes = np.zeros((heads, sequence, dim), dtype="|i1")
    key_scale = np.zeros((heads, sequence, 1), dtype="<u2")
    value_codes = np.zeros((heads, sequence, dim), dtype="|i1")
    new_key_codes = np.ones((heads, tokens, dim), dtype="|i1")
    new_key_scale = np.ones((heads, tokens, 1), dtype="<u2")
    new_value_codes = np.ones((heads, tokens, dim), dtype="|i1")
    key_codes[:, 5] = new_key_codes[:, 0]
    key_scale[:, 5] = new_key_scale[:, 0]
    value_codes[:, 5] = new_value_codes[:, 0]
    add_tensors(index, {
        ("expected", "qk_key_codes"): key_codes,
        ("expected", "qk_key_scale"): key_scale,
        ("expected", "pv_value_codes"): value_codes,
        ("expected", "new_key_codes"): new_key_codes,
        ("expected", "new_key_scale"): new_key_scale,
        ("expected", "new_value_codes"): new_value_codes,
    })
    # The small reference_data's all-one cache expected disagrees with the zero
    # retained cache at every position except the single committed position.
    with pytest.raises(ValueError, match="persistent cache_key_codes differs"):
        testcase.prepare(index, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_reuse_testcase_flags_workspace_and_default_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, bits=(4, 4, 4))
    baseline = testcase.prepare(index, tmp_path / "baseline")
    original = read_prepared_record(baseline, testcase.BASE, schema("execution_config"))
    assert original["flags"] == 3 and original["ffn_pair_first_batch_plus1"] == 0
    path = testcase.prepare(index, tmp_path / "paired", round_token_limit=1,
        ffn_group_batches=6, ffn_fused_product=True, ffn_down_pair=True,
        attention_output_pair=True, attention_pair=True, kv_pair=True)
    cfg = read_prepared_record(path, testcase.BASE, schema("execution_config"))
    assert cfg["flags"] == 0xfd3 and cfg["ffn_pair_first_batch_plus1"] == 1
    entry = read_prepared_record(path, cfg["layer_weight_table_base"], schema("layer_address_table"))
    assert entry["context_limit"] - entry["context_base"] == 96 * 8192
    assert entry["ffn_gate_up_workspace_limit"] - entry["ffn_gate_up_workspace_base"] >= 96 * 8192
    assert entry["ffn_gate_up_workspace_base"] >= cfg["bf16_temporary_base"] + 3 * 4096 * 4
    assert entry["ffn_residual_workspace_base"] == entry["ffn_gate_up_workspace_limit"]
    assert len(json.loads(path.read_text())["source"]["rounds"]) == 3
    before = {e["name"]: (baseline.parent / e["path"]).read_bytes()
              for e in json.loads(baseline.read_text())["expected"]}
    after = {e["name"]: (path.parent / e["path"]).read_bytes()
             for e in json.loads(path.read_text())["expected"]}
    assert before == after
    for options, message in (
            (dict(ffn_fused_product=True), "requires FFN grouping"),
            (dict(ffn_down_pair=True), "requires fused product"),
            (dict(ffn_group_batches=6), "at least two")):
        with pytest.raises(ValueError, match=message):
            testcase.prepare(index, tmp_path / "rejected", **options)
        assert not (tmp_path / "rejected").exists()
    reference_data = testcase.ReferenceData(index)
    original_load = reference_data.load
    reference_data.load = lambda role, name, dtype=None: np.array([8, 4, 4]) if (role, name) == ("input", "activation_bits") else original_load(role, name, dtype)
    with pytest.raises(ValueError, match="at least two metadata rounds"):
        testcase.prepare(index, tmp_path / "mixed", reference_data=reference_data,
                   ffn_group_batches=4, ffn_fused_product=True)
    with pytest.raises(ValueError, match="every token to use A4"):
        testcase.prepare(index, tmp_path / "mixed", reference_data=reference_data, kv_pair=True)


def test_qkvo_group_marks_batches_and_reserves_runtime_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(
        tmp_path / "input", monkeypatch,
        positions=(0, 1, 2, 3, 4, 5), bits=(4, 8, 4, 8, 4, 4))
    path = testcase.prepare(index, tmp_path / "grouped", round_token_limit=1,
                      qkvo_group=True)
    cfg = read_prepared_record(path, testcase.BASE, schema("execution_config"))
    assert cfg["flags"] == 0x1003
    entry = read_prepared_record(
        path, cfg["layer_weight_table_base"], schema("layer_address_table"))
    assert entry["context_limit"] - entry["context_base"] == 48 * 8192
    assert (entry["ffn_gate_up_workspace_limit"] -
            entry["ffn_gate_up_workspace_base"]) >= 12 * 8192
    case = json.loads(path.read_text())
    groups = case["source"]["reuse"]["qkvo_groups"]
    assert groups == [dict(first_round=0, batches=6, tokens=6,
                           activation_slots=8, compute_groups=6)]
    manifest = json.loads((path.parent / case["memory_map"]).read_text())
    flags, cursor = [], cfg["token_metadata_base"] - manifest["base_address"]
    image = (path.parent / case["ddr_image"]).read_bytes()
    for _ in range(6):
        flags.append(image[cursor + 3])
        cursor += int.from_bytes(image[cursor + 8:cursor + 10], "little")
    assert flags == [0, 0, 0, 0, 0, 4]


def test_qkvo_group_splits_six_batches_plus_single_tail():
    metadata, _, rounds = testcase.pack_tokens(
        [4] * 7, list(range(7)), round_token_limit=1)
    groups = testcase.mark_qkvo_groups(metadata, rounds)
    assert [group["batches"] for group in groups] == [6, 1]
    cursor, flags = 0, []
    for _ in rounds:
        flags.append(metadata[cursor + 3])
        cursor += int.from_bytes(metadata[cursor + 8:cursor + 10], "little")
    assert flags == [0, 0, 0, 0, 0, 4, 4]


def test_qkvo_group_rejects_pair_flags_and_accepts_single_batch(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(
        tmp_path / "input", monkeypatch, bits=(4, 4, 4))
    with pytest.raises(ValueError, match="Attention or KV pairing"):
        testcase.prepare(index, tmp_path / "pair-conflict", round_token_limit=1,
                   qkvo_group=True, kv_pair=True)
    case_path = testcase.prepare(index, tmp_path / "single", qkvo_group=True)
    case = json.loads(case_path.read_text())
    assert case["source"]["reuse"]["qkvo_groups"][0]["batches"] == 1


def test_real_l31_capture_local_tensor_numbering(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, layer=31)
    metadata = json.loads(index.read_text())
    metadata["layers"] = [dict(layer=0, model_layer=31)]
    metadata["layer_count"] = 1
    metadata["forward"] = dict(activation_bits=[[8, 4, 4]])
    for entry in metadata["tensors"]:
        if entry["name"] not in ("hidden", "activation_bits", "positions", "softmax_exp_lut", "softmax_reciprocal_lut",
                                 "rope_cosine", "rope_sine", "silu_breakpoints"):
            entry["name"] = "layer0." + entry["name"]
    index.write_text(json.dumps(metadata))
    reference_data = testcase.ReferenceData(index)
    assert testcase.layer_prefix(reference_data, 0, 1) == "layer0."
    path = testcase.prepare(index, tmp_path / "prepared")
    cfg = read_prepared_record(path, testcase.BASE, schema("execution_config"))
    assert cfg["start_layer"] == 31
    metadata["layers"][0]["model_layer"] = 30
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model_layer"):
        testcase.prepare(index, tmp_path / "wrong_layer")
    metadata["layers"][0]["model_layer"] = 31
    metadata["forward"]["activation_bits"] = [[8, 4, 4], [8, 4, 4]]
    index.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="activation_bits"):
        testcase.prepare(index, tmp_path / "wrong_batch")


@pytest.mark.parametrize("positions", [[], [3], [5, 1, 3]])
def test_l31_subset_preparation_preserves_expected_bits_and_bypasses_full_set(tmp_path, monkeypatch, positions):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    hidden = np.repeat(np.array([0x3f81, 0x3f82, 0x3f83], dtype="<u2")[:, None], 4096, axis=1)[None]
    golden = np.repeat(np.array([0x4101, 0x4102, 0x4103], dtype="<u2")[:, None], 4096, axis=1)[None]
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, layer=31, hidden=hidden, output=golden)
    metadata = json.loads(index.read_text())
    metadata["forward"] = dict(forward_kind="local_block", prediction_positions=positions,
                               future_prediction_positions=[])
    index.write_text(json.dumps(metadata))
    path = testcase.prepare(index, tmp_path / "subset", last_layer_output_subset=True)
    case = json.loads(path.read_text())
    order = case["source"]["packed_logical_to_reference_token"]
    assert case["source"]["last_layer_output_subset"] == (len(positions) < 3)
    expected = {entry["name"]: entry for entry in case["expected"]}
    if positions:
        actual = np.fromfile(path.parent / expected["hidden"]["path"], dtype="<u2").reshape(len(positions), 4096)
        np.testing.assert_array_equal(actual, golden[0, order[:len(positions)]])
    else:
        assert "hidden" not in expected
    if len(positions) < 3:
        actual = np.fromfile(path.parent / expected["hidden_unwritten"]["path"], dtype="<u2").reshape(3 - len(positions), 4096)
        np.testing.assert_array_equal(actual, hidden[0, order[len(positions):]])
    else:
        baseline = testcase.prepare(index, tmp_path / "baseline")
        assert (path.parent / "initial.bin").read_bytes() == (baseline.parent / "initial.bin").read_bytes()


def test_attention_checkpoint_preparation_preserves_all_tokens_after_output_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, layer=31)
    metadata = json.loads(index.read_text())
    metadata["forward"] = dict(forward_kind="local_block", prediction_positions=[3],
                               future_prediction_positions=[])
    index.write_text(json.dumps(metadata))
    sequence = metadata["sequence"]
    codes = np.arange(testcase.HEADS * 3 * sequence, dtype=np.uint8).view(np.int8).reshape(1, testcase.HEADS, 3, sequence)
    scales = np.arange(testcase.HEADS * 3, dtype="<u2").reshape(1, testcase.HEADS, 3, 1) + 0x8000
    add_tensors(index, {("expected", "probability_codes"): codes,
                        ("expected", "probability_scale"): scales,
                        ("expected", "context"): np.zeros((1, testcase.HEADS, 3, 128), dtype="<u2")})
    path = testcase.prepare(index, tmp_path / "checked", last_layer_output_subset=True,
                      attention_checkpoints=True)
    case = json.loads(path.read_text())
    checks = case["attention_checkpoints"]
    assert checks["layer"] == 31
    assert "context" not in [entry["name"] for entry in checks["expected"]]
    assert sorted(token for batch in checks["physical_to_reference_token"] for token in batch) == [0, 1, 2]
    hidden = np.fromfile(path.parent / "preserved_hidden.expected.bin", dtype="<u2").reshape(3, 4096)
    source_hidden = testcase.ReferenceData(index).load("input", "hidden", "<u2").reshape(3, 4096)
    np.testing.assert_array_equal(hidden, source_hidden)
    for entry, array in zip(checks["expected"][:2], (codes, scales)):
        assert (path.parent / entry["path"]).read_bytes() == array.tobytes()
    assert checks["expected"][-1] == {
        "name": "preserved_hidden", "path": "preserved_hidden.expected.bin",
        "dtype": "<u2", "shape": [3, 4096], "bytes": 24576}
    baseline = testcase.prepare(index, tmp_path / "unchecked", last_layer_output_subset=True)
    assert (path.parent / "initial.bin").read_bytes() == (baseline.parent / "initial.bin").read_bytes()


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("subset", [False, True])
def test_final_layer_checkpoints_preserve_input_for_l31_layout(tmp_path, monkeypatch, layers, subset):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, layer=32 - layers)
    add_tensors(index, {
        ("expected", "probability_codes"): np.zeros((1, 1, 3, 6), dtype="i1"),
        ("expected", "probability_scale"): np.ones((1, 1, 3, 1), dtype="<u2"),
        ("expected", "attention_residual"): np.full((1, 3, 4096), 0x4200, dtype="<u2"),
    })
    metadata = json.loads(index.read_text())
    metadata["forward"] = dict(forward_kind="local_block", prediction_positions=[3],
                               future_prediction_positions=[])
    if layers == 2:
        entries = list(metadata["tensors"])
        prior_output = next(e for e in entries if e["role"] == "expected" and e["name"] == "block_output")
        for layer in range(layers):
            for entry in entries:
                value = prior_output if layer == 1 and entry["role"] == "input" and entry["name"] == "hidden" else entry
                metadata["tensors"].append(dict(value, role=entry["role"], name=f"layer{layer}." + entry["name"]))
        metadata.update(layer_count=2, layers=[dict(layer=i, model_layer=30+i,
            tokens=3, positions=[5, 1, 3], kv_write_positions=[5, 1, 3],
            cache_commit_positions=[5, 1, 3]) for i in range(2)])
    index.write_text(json.dumps(metadata))
    path = testcase.prepare(index, tmp_path / "checked", attention_checkpoints=True,
                            round_token_limit=2, last_layer_output_subset=subset)
    case = json.loads(path.read_text())
    config = read_prepared_record(path, testcase.BASE, schema("execution_config"))
    checks = case["attention_checkpoints"]
    names = {entry["name"] for entry in checks["expected"]}
    assert config["start_layer"] == 32 - layers
    assert config["layer_count"] == layers
    assert checks["layer"] == 31
    assert ("preserved_hidden" in names) == (layers == 1 or subset)
    assert {"probability_codes", "probability_scale", "attention_residual"} <= names
    assert sorted(token for batch in checks["physical_to_reference_token"] for token in batch) == [0, 1, 2]


def test_attention_single_round_has_no_preservation_spill(tmp_path, monkeypatch):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, layer=0)
    sequence = json.loads(index.read_text())["sequence"]
    add_tensors(index, {
        ("expected", "probability_codes"): np.zeros((1, testcase.HEADS, 3, sequence), dtype="i1"),
        ("expected", "probability_scale"): np.ones((1, testcase.HEADS, 3, 1), dtype="<u2"),
        ("expected", "context"): np.arange(testcase.HEADS * 3 * 128, dtype="<u2").reshape(1, testcase.HEADS, 3, 128),
    })
    path = testcase.prepare(index, tmp_path / "checked", attention_checkpoints=True)
    names = [item["name"] for item in json.loads(path.read_text())["attention_checkpoints"]["expected"]]
    assert "preserved_hidden" not in names
    assert "probability_codes" in names
    assert "context" in names
    assert (path.parent / "context.expected.bin").read_bytes() == testcase.ReferenceData(index).load("expected", "context", "<u2").tobytes()
    assert not (path.parent / "preserved_hidden.expected.bin").exists()


@pytest.mark.parametrize("head_base", [0x10000000, 0x90000000])
def test_l31_actual_output_connects_to_head_without_expected_input(tmp_path, monkeypatch, head_base):
    import prepare_head_testcase as head
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(head, "artifact_root", lambda: tmp_path)
    golden = np.repeat(np.array([0x4101, 0x4102, 0x4103], dtype="<u2")[:, None], 4096, axis=1)[None]
    layer_index = small_preparation_reference_data(tmp_path / "layer_input", monkeypatch,
                                             layer=31, output=golden)
    layer_path = testcase.prepare(layer_index, tmp_path / "layer", base_address=0x60000000)
    layer_case = json.loads(layer_path.read_text())
    layer_regions = json.loads((layer_path.parent / layer_case["memory_map"]).read_text())
    assert layer_regions["base_address"] == 0x60000000
    assert layer_regions["limit_address"] == 0x60000000 + 8 * 1024**3
    assert layer_regions["logical_aperture_bytes"] == 8 * 1024**3
    layer_hidden = next(item for item in layer_case["expected"] if item["name"] == "hidden")
    packed = np.fromfile(layer_path.parent / layer_hidden["path"], dtype="<u2").reshape(3, 4096)
    values = {("input", "hidden"): packed[[1, 0]],
        ("input", "norm_weight"): np.full(4096, 0x3f80, dtype="<u2"),
        ("input", "weight_codes"): np.zeros((64, 4096), dtype="|i1"),
        ("input", "weight_scale"): np.full(64, 0x3f80, dtype="<u2"),
        ("input", "positions"): np.array([1, 0], dtype="<i8"),
        ("input", "prediction_mask"): np.ones(2, dtype="|b1"),
        ("input", "epsilon"): np.array(0x3727c5ac, dtype="<u4")}
    for name, (dtype, shape) in head.expected_shapes(2).items():
        values["expected", name] = np.zeros(shape or (2, 64), dtype=dtype)
    head_root = tmp_path / "head_input"
    head_root.mkdir()
    head_index = minimal_reference_data(head_root, reference_data_kind="cpu_final_output", tokens=2,
                                 weight_bits=8, activation_bits=8, vocab=64)
    add_tensors(head_index, values)
    metadata = json.loads(head_index.read_text())
    next(entry for entry in metadata["tensors"] if entry["name"] == "epsilon")["encoding"] = "float32_raw"
    head_index.write_text(json.dumps(metadata))
    output_alias = tmp_path / "output_alias"
    output_alias.symlink_to(tmp_path, target_is_directory=True)
    path = head.prepare(head_index, output_alias / "connected", preceding_layer_testcase=layer_path,
                        layer_output_tokens=[1, 0], base_address=head_base)
    case = json.loads(path.read_text())
    assert case["executions"][0] == layer_case["executions"][0]
    assert case["head_checkpoints"]["execution_index"] == 1
    image = (path.parent / case["ddr_image"]).read_bytes()
    cfg = decode_record(image, schema("execution_config"))
    assert cfg["output_hidden_base"] == layer_hidden["address"]
    assert not any(item["name"] == "hidden_input" for item in case["expected"])
    connected_hidden = next(item for item in case["expected"] if item["name"] == "layer.hidden")
    assert connected_hidden["address"] == cfg["output_hidden_base"]
    assert (path.parent / connected_hidden["path"]).read_bytes() == packed.tobytes()
    assert cfg["prediction_count"] == 2
    metadata = image[0xD3000:cfg["token_metadata_limit"] - head_base]
    header = decode_record(metadata, schema("token_metadata")["token_batch_header"])
    assert header["resident_token_count"] == cfg["total_token_count"] == 2
    assert header["token_batch_bytes"] == len(metadata) == 80
    assert header["compute_group_count"] == 1
    assert metadata[header["inverse_offset"]:] == bytes([0, 1]) + bytes(14)
    block = decode_record(image[0xDA000:], schema("draft_verify_block_config"))
    assert block["position_count"] == block["scheduled_quota"] == 2
    assert not any(image[head.HIDDEN - head.BASE:head.HIDDEN - head.BASE + 2 * 8192])
    assert decode_record(image[0xD1000:], schema("prediction_record"))["final_hidden_ddr_row_index"] == 1
    assert decode_record(image[0xD1020:], schema("prediction_record"))["final_hidden_ddr_row_index"] == 0
    initialized = {(path.parent / entry["path"]).resolve() for entry in case["initial_segments"]}
    expected = {(path.parent / entry["path"]).resolve() for entry in case["expected"]}
    assert initialized == {layer_path.parent / "initial.bin"}
    assert not initialized & expected
    if head_base == 0x90000000:
        scaffold = json.loads(layer_path.read_text())
        scaffold["source"] = dict(kind="boundary_scout_deep", deep=scaffold["source"])
        scaffold["executions"] = [dict(scaffold["executions"][0], id=1),
                                  dict(scaffold["executions"][0], id=2, copies=[
                                      dict(source=layer_hidden["address"], destination=layer_hidden["address"], bytes=16)])]
        for item in scaffold["expected"]:
            item["name"] = "deep." + item["name"]
        (layer_path.parent / "selector.bin").write_bytes(bytes(16))
        scaffold["initial_segments"] = [dict(address=0x68000000, bytes=16, path="selector.bin")]
        scaffold["attention_checkpoints"] = [dict(execution_index=i, expected=[dict(layer_hidden)]) for i in (0, 1)]
        scaffold_path = layer_path.parent / "boundary.json"
        scaffold_path.write_text(json.dumps(scaffold))
        boundary_path = head.prepare(head_index, tmp_path / "boundary_head",
            preceding_layer_testcase=scaffold_path, layer_output_tokens=[1, 0], base_address=head_base)
        connected = json.loads(boundary_path.read_text())
        assert connected["executions"][:2] == scaffold["executions"]
        assert connected["head_checkpoints"]["execution_index"] == 2
        assert [c["execution_index"] for c in connected["attention_checkpoints"]] == [0, 1]
        assert (boundary_path.parent / connected["initial_segments"][1]["path"]).read_bytes() == bytes(16)
        assert (boundary_path.parent / connected["attention_checkpoints"][1]["expected"][0]["path"]).read_bytes() == packed.tobytes()
        connected_hidden = next(item for item in connected["expected"] if item["name"] == "layer.deep.hidden")
        assert connected_hidden["address"] == layer_hidden["address"]
    overlapping_layer = testcase.prepare(layer_index, tmp_path / "overlapping_layer", base_address=0x30000000)
    with pytest.raises(ValueError, match="hidden overlaps head descriptor embedding_base"):
        head.prepare(head_index, tmp_path / "overlapping_head", preceding_layer_testcase=overlapping_layer,
                     layer_output_tokens=[1, 0])
    for tokens, message in (([0, 0], "unique"), ([0, 3], "omitted"), ([0, 1], "raw bits")):
        with pytest.raises(ValueError, match=message):
            head.prepare(head_index, tmp_path / ("bad" + str(tokens)), preceding_layer_testcase=layer_path,
                         layer_output_tokens=tokens)
    with pytest.raises(ValueError, match="DDR aperture"):
        testcase.prepare(layer_index, tmp_path / "outside", base_address=testcase.BASE + 8 * 1024**3)


@pytest.mark.parametrize("commits", [[2], []])
def test_l31_preparation_and_selective_commit_addresses(tmp_path, monkeypatch, commits):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    index = small_preparation_reference_data(tmp_path / "input", monkeypatch, layer=31, writes=[0, 2], commits=commits)
    if not commits:
        reference_data = testcase.ReferenceData(index)
        add_tensors(index, {("expected", "cache_" + name): reference_data.load("input", "retained_" + name)
                            for name in ("key_codes", "value_codes", "key_scale")})
    path = testcase.prepare(index, tmp_path / "prepared")
    cfg = read_prepared_record(path, testcase.BASE, schema("execution_config"))
    assert cfg["start_layer"] == 31 and cfg["layer_count"] == 1
    entry = read_prepared_record(path, cfg["layer_weight_table_base"] + 31 * 896, schema("layer_address_table"))
    for linear in testcase.FIELDS:
        assert entry[linear + "_enhancement_weight_base"] == 0
        assert entry[linear + "_enhancement_weight_limit"] == 0
    assert entry["current_k_cache_base"] != entry["retained_k_cache_base"]
    if not commits:
        assert cfg["refresh_configuration_offset"] == 0
        case = json.loads(path.read_text())
        with (path.parent / "initial.bin").open("rb") as stream:
            for expected in case["expected"]:
                if ".cache_" in expected["name"]:
                    stream.seek(expected["address"] - testcase.BASE)
                    assert stream.read(expected["bytes"]) == (path.parent / expected["path"]).read_bytes()
        return
    refresh = read_prepared_record(path, testcase.BASE + cfg["refresh_configuration_offset"], schema("token_refresh_config"))
    assert refresh["source_k_base"] == entry["current_k_cache_base"]
    assert refresh["destination_k_base"] == entry["retained_k_cache_base"]
    assert refresh["destination_scale_base"] == cfg["retained_k_scale_base"]
    assert refresh["target_token_count"] == 1
    with (path.parent / "initial.bin").open("rb") as stream:
        stream.seek(refresh["table_base"] - testcase.BASE)
        table = np.frombuffer(stream.read(64), dtype="<u8")
        assert np.flatnonzero(table).tolist() == [3]
        assert table[3] == 1 << 8
        stream.seek(cfg["token_metadata_base"] - testcase.BASE)
        raw = stream.read(128)
    layout = schema("token_metadata")
    header = decode_record(raw, layout["token_batch_header"])
    for physical in range(3):
        token = decode_record(raw[32 + physical * 16:], layout["token_entry"])
        assert bool(header["kv_write_disable_mask"] & (1 << physical)) == (token["token_position"] == 1)
    case = json.loads(path.read_text())
    assert {entry["name"] for entry in case["expected"]} == {
        "layer31.cache_key_codes", "layer31.cache_key_scale", "layer31.cache_value_codes", "hidden"}


def test_l0_endpoint_preserves_original_capture_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    index = tmp_path / "capture.json"
    index.write_text(json.dumps(dict(forward=dict(capture_index=37))))
    source = SimpleNamespace(metadata=json.loads(index.read_text()))
    view = SimpleNamespace(metadata=dict(tokens=64))  # L0 view deliberately omits forward metadata.
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(testcase, "ReferenceData", lambda *args: source)
    monkeypatch.setattr(testcase, "captured_layer_view", lambda *args: view)
    def prepare(index, output, **kwargs):
        output.mkdir()
        (output / "initial.bin").write_bytes(bytes(32))
        (output / "memory_map.json").write_text(json.dumps(dict(memory_map=[])))
        case = output / "case.json"
        case.write_text(json.dumps(dict(expected=[], source={}, executions=[{}])))
        return case
    monkeypatch.setattr(testcase, "prepare", prepare)
    selected = []
    monkeypatch.setattr(testcase, "attach_boundary_selection", lambda *args, **kwargs:
        selected.append((args[4], kwargs["capture_index"])))
    reference = tmp_path / "selection.json"
    reference.write_text(json.dumps(dict(records=[dict(expected=dict(deep_positions=dict(raw=[4,5])))])))
    result = testcase.prepare_boundary(index, tmp_path / "prepared", last_layer=0,
                                      boundary_reference=reference)
    assert selected == [(None, 37)] and len(json.loads(result.read_text())["executions"]) == 1


@pytest.mark.parametrize("keep_global", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_boundary_two_launches_read_actual_hidden_without_host_copy(tmp_path, monkeypatch, keep_global, grouped):
    monkeypatch.setattr(testcase, "artifact_root", lambda: tmp_path)
    first_output = np.repeat(np.array([0x4100, 0x4110, 0x4120], dtype="<u2")[:, None], 4096, axis=1)[None]
    commits = [0, 1, 2] if keep_global else [2, 0]
    first = small_preparation_reference_data(tmp_path / "first", monkeypatch, writes=[0, 1, 2], commits=commits, output=first_output)
    deep = small_preparation_reference_data(tmp_path / "second", monkeypatch, layer=1, positions=(3, 5), bits=(8, 4), hidden=first_output[:, [2, 0]])
    metadata = json.loads(first.read_text())
    metadata.update(boundary_scout={"selection": "given"}, continuation_index=str(deep))
    first.write_text(json.dumps(metadata))
    add_tensors(first, {
        ("expected", "probability_codes"): np.zeros((1, testcase.HEADS, 3, 6), dtype="|i1"),
        ("expected", "probability_scale"): np.zeros((1, testcase.HEADS, 3, 1), dtype="<u2"),
    })
    add_tensors(deep, {
        ("expected", "probability_codes"): np.zeros((1, testcase.HEADS, 2, 6), dtype="|i1"),
        ("expected", "probability_scale"): np.zeros((1, testcase.HEADS, 2, 1), dtype="<u2"),
    })
    views, tokens = testcase.boundary_views(first)
    assert tokens == [2, 0]
    assert views[1].metadata["model_layer_index"] == 1
    assert views[0].metadata["cache_commit_token_indices"] == commits
    path = testcase.prepare_boundary(first, tmp_path / "prepared", attention_checkpoints=True,
        deep_qkvo_group=grouped, deep_round_token_limit=1 if grouped else 48)
    case = json.loads(path.read_text())
    assert len(case["executions"]) == 2
    assert [check["execution_index"] for check in case["attention_checkpoints"]] == ([0] if grouped else [0, 1])
    assert all(entry["path"].startswith(directory + "/")
               for directory, check in zip(("scout", "deep"), case["attention_checkpoints"])
               for entry in check["expected"])
    scout_cfg = read_prepared_record(path, testcase.BASE, schema("execution_config"))
    assert bool(scout_cfg["refresh_configuration_offset"]) == (not keep_global)
    manifest = json.loads((path.parent / case["memory_map"]).read_text())
    assert manifest["current_ddr_image_bytes"] == (path.parent / case["ddr_image"]).stat().st_size
    scout_manifest = json.loads((path.parent / "scout/memory_map.json").read_text())
    assert manifest["current_ddr_image_bytes"] > scout_manifest["current_ddr_image_bytes"]
    for entry in case["expected"]:
        assert (path.parent / entry["path"]).stat().st_size == entry["bytes"]
        assert any(region["base"] <= entry["address"] and entry["address"] + entry["bytes"] <= region["limit"]
                   for region in manifest["memory_map"])
    assert case["source"]["deep_source_indices"] == [1, 2]
    assert case["source"]["scout"]["tokens"] == 3
    assert case["source"]["scout"]["start_layer"] == 0
    assert len(case["source"]["scout"]["rounds"]) == 1
    assert case["source"]["deep"]["tokens"] == 2
    assert case["source"]["deep"]["start_layer"] == 1
    assert len(case["source"]["deep"]["rounds"]) == (2 if grouped else 1)
    assert not case["executions"][1].get("copies")
    cfg = read_prepared_record(path, case["executions"][1]["config_address"], schema("execution_config"))
    assert cfg["start_layer"] == 1
    assert bool(cfg["flags"] & 0x1000) == grouped
    assert case["executions"][1]["config_address"] % 8192 != 0
    assert cfg["embedding_base"] % 8192 == 0
    assert cfg["embedding_limit"] - cfg["embedding_base"] == 1035993088
    assert scout_cfg["output_hidden_base"] == cfg["output_hidden_base"]
    assert scout_cfg["output_hidden_limit"] == cfg["output_hidden_limit"]
    assert all(a["limit"] <= b["base"] for a, b in zip(manifest["memory_map"], manifest["memory_map"][1:]))
    image = np.fromfile(path.parent / "initial.bin", dtype=np.uint8)
    assert all(entry.get("execution_index") == 0 for entry in case["expected"]
               if entry["name"].startswith("scout."))
    source = scout_cfg["output_hidden_base"] - testcase.BASE
    # Model only an L0 write. L1 must see it without any between-launch copy.
    image[source:source + 3 * 4096 * 2] = 0x5a
    raw = image[cfg["token_metadata_base"] - testcase.BASE:].tobytes()
    layout = schema("token_metadata")
    sources, cursor = [], 0
    for _ in case["source"]["deep"]["rounds"]:
        header = decode_record(raw[cursor:], layout["token_batch_header"])
        sources.extend(decode_record(raw[cursor+32+i*16:], layout["token_entry"])["source_index"]
                       for i in range(raw[cursor]))
        cursor += int.from_bytes(raw[cursor+8:cursor+10], "little")
    assert sorted(sources) == [1, 2]
    for index in sources:
        address = cfg["output_hidden_base"] - testcase.BASE + index * 4096 * 2
        assert np.all(image[address:address + 4096 * 2] == 0x5a)


@pytest.mark.parametrize("indices", [[0], [0, 0], [-1, 1], [0, 2048]])
def test_source_index_mapping_rejects_wrong_length_aliasing_and_range(indices):
    with pytest.raises(ValueError, match="source_indices"):
        testcase.pack_tokens([4, 8], [3, 5], source_indices=indices)


@pytest.mark.parametrize("mode,sequence", [("historical", 8), ("transition", 8),
                                        ("current_to_context", 8), ("historical", 395)])
@pytest.mark.parametrize("grouped, deep_present", [(True, True), (False, True), (False, False)])
@pytest.mark.parametrize("requested_target", [3, 12])
def test_boundary_selection_uses_history_and_empty_runtime_metadata(tmp_path, mode, sequence, grouped, deep_present, requested_target):
    historical_only = mode == "historical"
    vector_scout = mode == "current_to_context"
    deep_tokens = 3
    root = testcase.BASE
    def reg(name, offset, size):
        return dict(name=name, base=root+offset, limit=root+offset+size, access="read_write")
    regions = [reg("configuration", 0, 512), reg("cache_k", 0x1000, 256),
               reg("cache_v", 0x2000, 256), reg("cache_scale", 0x3000, 256),
               reg("deep.token_metadata", 0x4000, 256)]
    regions[-1]["access"] = "read_only"
    config = dict(required_fields(schema("execution_config")), total_token_count=sequence,
                  sequence_length=sequence)
    (tmp_path/"scout").mkdir()
    (tmp_path/"scout/initial.bin").write_bytes(testcase.pack_record(schema("execution_config"), config))
    packed, _, rounds = testcase.pack_tokens([4]*deep_tokens, [1, 4, 7], source_indices=[1, 4, 7])
    image = bytearray(0x4100); image[0x4000:0x4000+len(packed)] = packed
    (tmp_path/"initial.bin").write_bytes(image)
    (tmp_path/"memory_map.json").write_text(json.dumps(dict(base_address=root, memory_map=regions)))
    scout = dict(source=dict(sequence=sequence, packed_logical_to_reference_token=list(range(sequence))),
                 executions=[dict(config_address=root)])
    deep = dict(source=dict(rounds=rounds, reuse=dict(enabled_flags=["qkvo_group"] if grouped else [])))
    inputs = dict(current_positions=dict(raw=[4]), mandatory_candidate_positions=dict(raw=[7]),
                  deep_activation_bits=4, previous_pending=dict(raw=[0x3f80]*sequence),
                  previous_future_pending=dict(raw=[0]*sequence), target_token_count=requested_target,
                  shortlist_token_count=4, shortlist_flags=15, relative_score_floor_bf16=0,
                  protected_begin=7, protected_end=8)
    ref = dict(source_index="capture.json", records=[dict(inputs=inputs, expected=dict(
        deep_positions=dict(raw=[1, 4, 7]), effective_target=deep_tokens))])
    case = dict(executions=[dict(config_address=root), dict(config_address=root+512)], expected=[], source=dict(deep=deep["source"]))
    if not historical_only:
        # A mixed L0 maps physical rows differently from sorted logical positions.
        bits = [8, 4, 8, 8, 4, 8, 8, 8]
        scout_meta, order, scout_rounds = testcase.pack_tokens(bits, list(range(sequence)))
        scout["source"].update(packed_logical_to_reference_token=order, rounds=scout_rounds)
        regions.insert(1, reg("token_metadata", 0x200, len(scout_meta)))
        initial = bytearray(0x200 + len(scout_meta))
        initial[:320] = testcase.pack_record(schema("execution_config"),config)
        initial[0x200:] = scout_meta
        (tmp_path/"scout/initial.bin").write_bytes(initial)
        deep["source"]["packed_logical_to_reference_token"] = [1, 0, 2]
        np.repeat(np.array([4, 1, 7],dtype="<u2"),testcase.HIDDEN).tofile(tmp_path/"hidden.bin")
        case["expected"].append(dict(name="deep.hidden",path="hidden.bin",bytes=deep_tokens*8192,address=root+0x8000))
        inputs.update(shortlist_flags=0, transition_positions=dict(raw=[] if vector_scout else [0,7]), context_positions=dict(raw=[2]),
            required_candidate_count=0, deep_a8_limit=2, keep_global_l0_cache=False,
            score_q8=dict(raw=[0,17,0,0,0,0,0,19]))
        ref["records"][0]["expected"]["deep_bits"] = dict(raw=[8,4,8])
    if not deep_present:
        case["executions"] = case["executions"][:1]
    testcase.attach_boundary_selection(case, tmp_path, ref, scout, deep if deep_present else None,
                                       regions, sequence, deep_tokens)
    if sequence == 395:
        # Runtime packing of 15 A4 rows at positions 220..234 needs 7168
        # bytes, exceeding the 7136-byte all-A8 layout for the same sequence.
        begin, end = case["provenance"]["boundary_control"]["full_metadata"]
        assert end - begin >= 7168
    segment = case["initial_segments"][0]
    control = (tmp_path/segment["path"]).read_bytes()
    refresh = decode_record(control[320:496], schema("token_refresh_config"))
    assert refresh["target_token_count"] == min(sequence, requested_target)
    offset = refresh["metadata_base"]-segment["address"]
    assert not any(control[offset:offset+16384])
    assert refresh["flags"] == ((2 if historical_only else (1|2|32|2<<8)) | (1 << 15 if grouped else 0))
    job_count = 0 if historical_only else 1 if vector_scout else 7
    assert refresh["relation_job_count"] == job_count
    probability = decode_record(control[496:576], schema("attention_probability_config"))
    # The descriptor must remain enabled to load shortlist state. An empty query
    # interval prevents historical-only selection from exporting new scores.
    assert probability["layer_mask"] == 1
    assert probability["query_begin"] == 0
    assert probability["query_end"] == (0 if historical_only else sequence)
    assert probability["key_groups0"] == (0 if historical_only else 1)
    assert all(probability[f"key_groups{i}"] == 0 for i in range(1,4))
    table = refresh["table_base"]-segment["address"]
    rows = [int.from_bytes(control[table+p*8:table+p*8+8], "little") for p in range(sequence)]
    assert all(not (r & (1<<9)) for r in rows)  # No host-preselected eligibility.
    assert [p for p,r in enumerate(rows) if r & (1<<8)] == [4]
    if not historical_only:
        assert [p for p,r in enumerate(rows) if r & (1<<11)] == [2]
        assert [p for p,r in enumerate(rows) if r & (1<<34)] == [0,2,3,5,6,7]
        offset = refresh["relation_job_base"]-segment["address"]
        jobs = [decode_record(control[offset+i*64:offset+(i+1)*64],schema("attention_dependency_job")) for i in range(job_count)]
        destinations = {0} if vector_scout else {p*8 for p in range(8) if p != 4}
        assert {j["output_base"]-refresh["table_base"] for j in jobs} == destinations
        assert all(j["operation"] == (6 if vector_scout else 7) and
                   j["lane_mask"] == (239 if vector_scout else 129) for j in jobs)
        # Check actual physical-row mapping rather than assuming logical token order.
        for physical in range(sequence):
            position=int.from_bytes(scout_meta[32+physical*16+4:32+physical*16+6],"little")
            if (vector_scout and position != 4) or (not vector_scout and position == 4):
                continue
            destination = refresh["table_base"] + (0 if vector_scout else position*8)
            job=next(j for j in jobs if j["output_base"]==destination)
            assert job["source_base"] == probability["output_base"] + physical*16
        hidden=next(e for e in case["expected"] if e["name"]=="deep.hidden")
        assert np.fromfile(tmp_path/hidden["path"],dtype="<u2").reshape(-1,testcase.HIDDEN)[:,0].tolist()==[4,1,7]
    if deep_present:
        action = case["executions"][1]["handoff_actions"][0]
        assert action["metadata_address"] == refresh["metadata_base"]
        assert "source_index_offset" not in action
        memory_map = json.loads((tmp_path/"memory_map.json").read_text())["memory_map"]
        assert next(r for r in memory_map if r["name"] == "deep.token_metadata")["access"] == "read_write"
    else:
        assert len(case["executions"]) == 1
        assert case["provenance"]["boundary_control"]["metadata"][0] == refresh["metadata_base"]
    inverse = {original: packed for packed, original in enumerate(scout["source"]["packed_logical_to_reference_token"])}
    assert [(row >> 35) & 2047 for row in rows] == [inverse[p] for p in range(sequence)]
    if historical_only and deep_present:
        if grouped:
            packed[3] |= 4
        assert (tmp_path/case["expected"][0]["path"]).read_bytes() == packed
    assert case["executions"][0]["config_address"] == segment["address"]
    assert all(a["limit"] <= b["base"] for a,b in zip(regions, regions[1:]))


def test_observed_head_reference_rejects_same_step_from_another_request(tmp_path):
    import copy
    import prepare_head_testcase as head
    identity = dict(dataset="example", split="test", sample_id="request-a", sample_doc_hash="request-id",
                    model_arguments=dict(generation_mode="feature12", clip=0.8))
    captured = dict(provenance=identity, forward=dict(capture_index=3, block_index=0,
                                                    step_index=2, forward_kind="local_block"))
    source = tmp_path/"capture.json"; source.write_text(json.dumps(captured))
    reference = dict(source_index=str(source), capture_index=3)
    head.validate_observed_head_reference(captured, reference)
    for key, value in [("sample_id", "request-b"), ("model_arguments", dict(clip=0.625))]:
        wrong = copy.deepcopy(captured); wrong["provenance"][key] = value
        with pytest.raises(ValueError, match=key):
            head.validate_observed_head_reference(wrong, reference)


def test_user_prompt_head_identity_uses_actual_tokens(tmp_path):
    import copy
    import prepare_head_testcase as head
    captured = dict(provenance=dict(dataset="user_request", split=None, sample_id="example",
                    sample_doc_hash=None, model_arguments=dict(seed=1234)),
                    forward=dict(capture_index=1, block_index=0, step_index=1, forward_kind="local_block"),
                    control_observation=dict(payload_root="."), tensors=[dict(role="control",
                        name="control.before_postprocess.tokens", path="tokens.bin", shape=[1, 3],
                        dtype="<i8", encoding="integer", byte_count=24)])
    for name in ("layer", "head"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "index.json").write_text(json.dumps(captured))
        np.array([11, 12, 126336], dtype="<i8").tofile(directory / "tokens.bin")
    source = tmp_path / "layer/index.json"
    head_index = tmp_path / "head/index.json"
    reference = dict(source_index=str(source), capture_index=1)
    head.validate_observed_head_reference(captured, reference, head_index=head_index)
    with pytest.raises(ValueError, match="head index"):
        head.validate_observed_head_reference(captured, reference)
    np.array([11, 13, 126336], dtype="<i8").tofile(tmp_path / "head/tokens.bin")
    with pytest.raises(ValueError, match="request tokens"):
        head.validate_observed_head_reference(captured, reference, head_index=head_index)
    wrong = copy.deepcopy(captured)
    wrong["tensors"][0]["byte_count"] = 16
    with pytest.raises(ValueError, match="byte count"):
        head.validate_observed_head_reference(wrong, reference, head_index=head_index)


def test_relocated_user_prompt_head_identity(tmp_path):
    import prepare_head_testcase as head
    payload = tmp_path / "payload"
    payload.mkdir()
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    metadata = dict(provenance=dict(dataset="user_request", sample_id="example", model_arguments=dict(seed=1234)),
                    forward=dict(capture_index=1, block_index=0, step_index=1, forward_kind="local_block"),
                    control_observation=dict(payload_root="."), tensors=[dict(role="control",
                        name="control.before_postprocess.tokens", path="tokens.bin", shape=[1, 3],
                        dtype="<i8", encoding="integer", byte_count=24)])
    source = indexes / "layer.json"
    source.write_text(json.dumps(metadata))
    index = indexes / "head.json"
    index.write_text(json.dumps(metadata))
    np.array([11, 12, 126336], dtype="<i8").tofile(payload / "tokens.bin")
    reference = dict(source_index=str(source), capture_index=1)
    head.validate_observed_head_reference(metadata, reference, head_index=index, payload_root=payload)


@pytest.mark.parametrize("tokens", [33, 48, 96])
def test_head_capacity_metadata_and_state_ranges(tmp_path, monkeypatch, tokens):
    """Check descriptor counts and storage ranges across metadata batches."""
    import prepare_head_testcase as head
    monkeypatch.setattr(head, "artifact_root", lambda: tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    index = minimal_reference_data(source, reference_data_kind="cpu_final_output",
                                  tokens=tokens, vocab=64, weight_bits=8, activation_bits=8)
    values = {("input", "hidden"): np.zeros((tokens, 4096), dtype="<u2"),
        ("input", "norm_weight"): np.full(4096, 0x3f80, dtype="<u2"),
        ("input", "weight_codes"): np.zeros((64, 4096), dtype="|i1"),
        ("input", "weight_scale"): np.full(64, 0x3f80, dtype="<u2"),
        ("input", "positions"): np.arange(tokens, dtype="<i8"),
        ("input", "prediction_mask"): np.ones(tokens, dtype="|b1"),
        ("input", "epsilon"): np.array(0x358637bd, dtype="<u4")}
    for name, (dtype, shape) in head.expected_shapes(tokens).items():
        values["expected", name] = np.zeros(shape or (tokens, 64), dtype=dtype)
    add_tensors(index, values)
    reference = json.loads(index.read_text())
    next(t for t in reference["tensors"] if t["name"] == "epsilon")["encoding"] = "float32_raw"
    # A separately stored control observation is not a numerical head input.
    reference["tensors"].append(dict(role="control", name="state", path="separate-root.bin"))
    index.write_text(json.dumps(reference))
    path = head.prepare(index, tmp_path / "prepared")
    case = json.loads(path.read_text())
    image = (path.parent / case["ddr_image"]).read_bytes()
    config = decode_record(image, schema("execution_config"))
    cursor, count, rounds = 0xD3000, 0, 0
    while cursor < config["token_metadata_limit"] - head.BASE:
        header = decode_record(image[cursor:], schema("token_metadata")["token_batch_header"])
        assert 1 <= header["resident_token_count"] <= 32
        assert header["first_token_ordinal"] == count
        assert header["token_batch_index"] == rounds
        count += header["resident_token_count"]
        cursor += header["token_batch_bytes"]
        rounds += 1
    assert count == tokens and rounds == (tokens + 31) // 32
    assert config["generation_block_count"] == rounds
    regions = json.loads((path.parent / case["memory_map"]).read_text())["memory_map"]
    for region_name, end in (("token_metadata", config["token_metadata_limit"]),
                             ("next_token_metadata", config["next_token_metadata_limit"])):
        region = next(r for r in regions if r["name"] == region_name)
        assert region["limit"] >= (end + 31) // 32 * 32
    post = decode_record(image[0xD0000:], schema("forward_postprocess_config"))
    assert post["current_state_limit"] - post["current_state_base"] == tokens * 32
    for token in range(tokens):
        descriptor = decode_record(image[0xD1000 + token*32:], schema("prediction_record"))
        assert descriptor["current_state_entry"] == token
        assert descriptor["block_slot"] == token // 32
        assert descriptor["source_token_batch_index"] == token // 32


@pytest.mark.parametrize("relative", [False, True])
def test_observed_head_source_survives_cwd_and_tree_move(tmp_path, monkeypatch, relative):
    import shutil
    import prepare_head_testcase as head
    source_root = tmp_path / "original"
    (source_root / "captures").mkdir(parents=True)
    (source_root / "references").mkdir()
    captured = dict(provenance=dict(dataset="user", split="test", sample_id="example",
                    sample_doc_hash="example-id", model_arguments=dict(clip=.625)),
                    forward=dict(capture_index=4, block_index=0, step_index=3, forward_kind="local_block"))
    source = source_root / "captures/input.json"
    source.write_text(json.dumps(captured))
    ref_path = source_root / "references/control.json"
    reference = dict(source_index="../captures/input.json" if relative else str(source), capture_index=4)
    if relative:
        reference["source_index_base"] = "reference_directory"
    ref_path.write_text(json.dumps(reference))
    monkeypatch.chdir(tmp_path)
    head.validate_observed_head_reference(captured, reference, ref_path)
    if relative:
        moved = tmp_path / "moved"
        shutil.move(str(source_root), moved)
        ref_path = moved / "references/control.json"
        head.validate_observed_head_reference(captured, reference, ref_path)
        with pytest.raises(ValueError, match="sample_id"):
            head.validate_observed_head_reference(dict(captured, provenance=dict(captured["provenance"], sample_id="different")), reference, ref_path)
        with pytest.raises(ValueError, match="capture_index"):
            head.validate_observed_head_reference(dict(captured, forward=dict(captured["forward"], capture_index=5)), reference, ref_path)
        reference.pop("source_index_base")
        with pytest.raises(ValueError, match="relative source_index requires"):
            head.validate_observed_head_reference(captured, reference, ref_path)


def test_captured_range_preserves_independent_layer_tensors(tmp_path):
    index = combined_boundary_reference_data(tmp_path)
    metadata = json.loads(index.read_text())
    metadata["model_layer_index"] = 30
    for local, record in enumerate(metadata["layers"]):
        record["model_layer"] = 30 + local
    index.write_text(json.dumps(metadata))
    source = testcase.ReferenceData(index)
    view = testcase.captured_layer_view(source, 30, 31)
    assert view.metadata["layer_count"] == 2
    assert [record["model_layer"] for record in view.metadata["layers"]] == [30, 31]
    for local in range(2):
        np.testing.assert_array_equal(view.load("input", f"layer{local}.hidden"),
                                      source.load("input", f"layer{local}.hidden"))
    with pytest.raises(ValueError, match="model layer 29"):
        testcase.captured_layer_view(source, 29, 31)


@pytest.mark.parametrize("override", [None, False, True])
def test_unified_prepare_merges_config_and_explicit_values(tmp_path, monkeypatch, override):
    import prepare_testcase as unified
    index = tmp_path / "input.json"
    index.write_text("{}")
    monkeypatch.setattr(unified.layers, "artifact_root", lambda: tmp_path)
    observed = {}
    def pack(*args, **kwargs):
        observed.update(kwargs)
        return args[1] / "case.json"
    monkeypatch.setattr(unified.layers, "prepare", pack)
    options = dict(attention_checkpoints=True, last_layer_output_subset=True,
                   base_address=0x20000000, qkvo_group=False)
    unified.prepare(index, tmp_path / "prepared", layer_options=options,
                    attention_checkpoints=override)
    assert observed["attention_checkpoints"] is (True if override is None else override)
    assert observed["last_layer_output_subset"] is True
    assert observed["base_address"] == 0x20000000
    assert options["attention_checkpoints"] is True


@pytest.mark.parametrize("execution_index", [0, 1])
def test_prepared_case_rejects_actual_output_name_collision(execution_index):
    from run_testcase import RTL_ROOT, validate_prepared_configurations

    case = RTL_ROOT / "cases/in_block_handoff/case.json"
    config = json.loads(case.read_text())
    original = dict(config["expected"][0], execution_index=0)
    config["expected"] = [original, dict(original, execution_index=execution_index)]
    with pytest.raises(ValueError, match="repeats the actual output filename"):
        validate_prepared_configurations(case, config)
