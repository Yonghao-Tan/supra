"""Fixed raw files can be inspected and compared without the model runtime."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import layer_reference_data as reference_data


class TestReferenceDataConsumer:
    @pytest.mark.parametrize("real_capture", [False, True])
    def test_relative_payload_cli_without_torch_or_algorithm(self, tmp_path, real_capture):
        root = tmp_path
        payload = root / "payload"
        payload.mkdir()
        (payload / "activation_bits.bin").write_bytes(bytes([4, 8]))
        (payload / "output.bin").write_bytes(bytes.fromhex("00000080803f"))
        (root / "output.bin").write_bytes(bytes.fromhex("00000080803f"))
        entries = [dict(role="input", name="activation_bits", path="activation_bits.bin",
                        dtype="|i1", shape=[2], encoding="integer", byte_count=2),
                   dict(role="expected", name="output", path="output.bin",
                        dtype="<u2", shape=[3], encoding="bf16_raw", byte_count=6)]
        index = root / "index.json"
        metadata = dict(schema=reference_data.SCHEMA, seed=0, tokens=2,
            sequence=3, hidden=128, ffn=12288, heads=1, tensors=entries)
        if real_capture:
            import ast
            producer = ast.parse((HERE / "model_capture.py").read_text())
            metadata["schema"] = next(ast.literal_eval(node.value) for node in producer.body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "SCHEMA" for target in node.targets))
            metadata["reference_data_kind"] = "real_cuda_consecutive_layers"
        index.write_text(json.dumps(metadata))
        code = '''
import importlib.abc, runpy, sys
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'spinquant_model', 'model',
                                      'numerics', 'quantization', 'evaluation', 'generation', 'capture'):
            raise AssertionError('fixed reference_data consumer imported ' + fullname)
sys.meta_path.insert(0, NoModel())
runpy.run_path(sys.argv.pop(1), run_name='__main__')
'''
        command = [sys.executable, "-B", "-c", code, str(HERE / "layer_reference_data.py")]
        common = ["--index", str(index), "--payload-root", str(payload)]
        inspected = subprocess.run(command + ["inspect"] + common, check=True,
                                   text=True, capture_output=True)
        assert json.loads(inspected.stdout)["a8_token_count"] == 1
        matched = subprocess.run(command + ["compare"] + common + ["--actual-root", str(root)],
                                 check=True, text=True, capture_output=True)
        assert json.loads(matched.stdout)["match"]
        (root / "output.bin").write_bytes(bytes.fromhex("00000000803f"))
        mismatch = reference_data.compare_reference_data(index, root, payload_root=payload)
        assert not mismatch["match"]
        assert mismatch["tensors"][0]["first_index"] == [1]
        assert mismatch["tensors"][0]["expected_raw"] == 0x8000

    def test_new_numeric_fields_require_capability(self):
        with pytest.raises(ValueError, match="explicit numeric_capabilities"):
            reference_data.validate_numeric_capabilities({"tensors": [
                {"role": "expected", "name": "layer0.q_proj.clip.limit_fp32"}]})


@pytest.mark.parametrize("generator,filename", [
    ("make_source_b_selection_reference_data", "uaps_source_b_selection.json"),
    ("make_regular_joint_reference_data", "atse_uaps_in_block.json"),
])
def test_control_regeneration_matches_fixed_records(generator, filename):
    from hardware_adapter import control_reference_data
    hardware = HERE.parent / "hardware"
    expected = json.loads((hardware / "cases/control" / filename).read_text())
    actual = json.loads(json.dumps(getattr(control_reference_data, generator)()))
    assert actual["records"] == expected["records"]


def test_synthetic_feature2_steps_match_algorithm_and_cover_boundaries():
    from hardware_adapter import control_reference_data as ref
    hardware = HERE.parent / "hardware"
    fixed = json.loads((hardware / "cases/control/psme_state_updates.json").read_text())
    assert ref.make_psme_state_updates() == fixed
    mixed, inherited, tail = fixed["records"]
    assert mixed["expected"]["masks"]["confirmed"] == 0xf00
    assert inherited["expected"]["masks"]["confirmed"] == 0
    assert inherited["expected"]["state"][8:12] == [1]*4
    assert tail["expected"]["masks"]["tail_closed"] == 1
    assert tail["expected"]["state"] == [2]*32


def test_precision_joint_fixed_and_generated_expected_match_algorithm():
    import torch
    from hardware_adapter import control_reference_data as ref
    hardware = HERE.parent / "hardware"
    fixed = json.loads((hardware / "cases/control/psme_uaps_context_precision.json").read_text())
    generated = json.loads(json.dumps(ref.make_precision_joint_reference_data()))
    # Regeneration uses the bundled base-precision inputs, which may differ from
    # the self-contained fixed case. Verify both against their own exact inputs.
    for data in (fixed, generated):
        scheduler = ref.UtilizationAwarePrefetchScheduler(target_joint_rows=48, max_next_rows=8)
        for record in data["records"]:
            inputs = {}
            for key, value in record["inputs"].items():
                if isinstance(value, dict) and "raw" in value:
                    if value["dtype"] == "torch.bfloat16":
                        tensor = torch.tensor(value["raw"], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                    else:
                        tensor = torch.tensor(value["raw"], dtype=getattr(torch, value["dtype"].split(".")[-1]))
                    inputs[key] = tensor.reshape(value["shape"])
                else:
                    inputs[key] = value
            selected = ref._select_prefetch(scheduler, **inputs)
            for key, expected in record["expected"].items():
                actual = getattr(selected, {"added_activation_bits": "added_row_bits"}.get(key, key))
                if isinstance(actual, torch.Tensor):
                    actual = ref.packed(actual)
                elif "residency" in key:
                    actual = ref._hardware_residency(actual)
                assert json.loads(json.dumps(actual)) == expected, (record["name"], key)


def test_capture_cli_writes_document_relative_source(tmp_path, monkeypatch):
    import json
    from hardware_adapter import control_reference_data as producer
    source = tmp_path / "captures/input.json"
    source.parent.mkdir()
    source.write_text("{}")
    target = tmp_path / "references/control.json"
    monkeypatch.setattr(producer, "PROJECT_ROOT", tmp_path / "source")
    monkeypatch.setattr(producer, "captured_feature2_reference", lambda p: dict(source_index=str(p)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(producer.sys, "argv", ["reference", "--kind", "captured-feature2",
                        "--input", "captures/input.json", "--index", "references/control.json"])
    producer.main()
    result = json.loads(target.read_text())
    assert result["source_index_base"] == "reference_directory"
    assert (target.parent / result["source_index"]).resolve() == source
    assert not producer.Path(result["source_index"]).is_absolute()
