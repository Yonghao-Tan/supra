"""The sample-free W8 exporter preserves the existing head quantization."""

from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import torch
from quantization import head as head
from quantization import artifact
from quantization.numeric import SpinQuantW4Tensor, quantize_symmetric_w8


def test_w8_export_uses_original_bf16_weight_and_no_captures(monkeypatch):
    source = torch.tensor(
        [[0.0, -0.0, 0.0], [1.0, -0.501, 0.023], [-0.031, 0.003, 0.015]],
        dtype=torch.bfloat16,
    )
    original = quantize_symmetric_w8(source)
    parent = SimpleNamespace(
        head_quantized=False, read_bf16=lambda name: source.clone()
    )
    monkeypatch.setattr(head, "SpinQuantArtifactReader", lambda *a, **kw: parent)
    monkeypatch.setattr(
        "quantization.artifact.require_train_only_artifact", lambda reader: None
    )
    saved = {}

    class Writer:
        def __init__(self, *args, **kwargs):
            saved.update(kwargs)

        def write(self, value, **kwargs):
            saved["quantized"] = value
            saved.update(kwargs)
            return {"total_file_bytes": 0}

    monkeypatch.setattr(head, "SpinQuantLMHeadArtifactWriter", Writer)
    head.export_rtn_head(
        SimpleNamespace(parent_artifact=Path("parent"), output_artifact=Path("child")),
        weight_bits=8,
    )
    actual = saved["quantized"]
    assert torch.equal(actual.codes, original.codes)
    assert torch.equal(
        actual.scale_bf16.view(torch.int32), original.scale_bf16.view(torch.int32)
    )
    assert saved["calibration"]["method"] == "weight_only_no_samples"
    assert saved["quantization"]["bits"] == 8


def test_current_parent_codec_preserves_signed_codes_and_bf16_scales(tmp_path, monkeypatch):
    identity = artifact.supported_checkpoint_identities()[0]
    monkeypatch.setattr(artifact, "checkpoint_identity", lambda path: identity)
    monkeypatch.setattr(artifact, "_load_index", lambda path: {})
    monkeypatch.setattr(artifact, "_shard_hashes", lambda *args: {})
    monkeypatch.setattr(artifact, "require_algo_output", lambda path: Path(path))
    output = tmp_path / "parent"
    writer = artifact.SpinQuantArtifactWriter(
        tmp_path, output, variant="fixed-r1r2-r4-only",
        calibration={"train_only_calibration": True}, rotation={"r1": torch.ones(4096)},
        weight_quantization={"tensor_kind": "rtn_w4", "sym": True,
                             "signed_integer_range": [-8, 7], "signed_zero_point": 0,
                             "qzeros": False, "group_size": -1},
    )
    codes = torch.tensor([[-8, 7, 0, -1], [1, 2, -3, -4]], dtype=torch.int8)
    scales = torch.tensor([0.5, 2.0])
    weight = SpinQuantW4Tensor(codes, scales)
    for name in sorted(artifact.expected_w4_ids()):
        writer.write_w4(name, weight, source_shape=[2, 4], calibration_rows=0)
    for name in (artifact.EMBEDDING, artifact.LM_HEAD):
        writer.write_bf16(name, torch.ones((2, 4), dtype=torch.bfloat16))
    writer.write_v_cache_calibration(
        torch.ones((32, 32)), statistics=[{} for _ in range(32)]
    )
    writer.finish()
    assert (output / artifact.W4_CODES_FILE).read_bytes() == bytes.fromhex("78f021cd") * 224
    assert (output / artifact.W4_SCALES_FILE).read_bytes() == bytes.fromhex("003f0040") * 224
    def reject_output(path):
        raise AssertionError("artifact reads must not use the output boundary")
    monkeypatch.setattr(artifact, "require_algo_output", reject_output)
    reader = artifact.SpinQuantArtifactReader(output)
    assert reader.manifest["schema_version"] == artifact.JOINT_W4_SCHEMA_VERSION
    restored = reader.read_w4(sorted(artifact.expected_w4_ids())[0])
    assert torch.equal(restored.codes, codes)
    assert torch.equal(restored.scale_bf16.view(torch.int32), scales.view(torch.int32))
    assert torch.equal(reader.read_v_cache_scales(), torch.ones((32, 32)))


@pytest.mark.parametrize("schema", ["unknown-format/v1", None])
def test_unrecognized_artifact_schema_is_rejected(tmp_path, monkeypatch, schema):
    monkeypatch.setattr(artifact, "require_algo_output", lambda path: Path(path))
    (tmp_path / artifact.MANIFEST_FILE).write_text(json.dumps({"schema_version": schema, "status": "complete"}))
    with pytest.raises(ValueError, match="unsupported or incomplete"):
        artifact.SpinQuantArtifactReader(tmp_path)


def test_w4_head_export_is_rejected_before_loading():
    with pytest.raises(ValueError, match="requires W8"):
        head.export_rtn_head(SimpleNamespace(), weight_bits=4)
