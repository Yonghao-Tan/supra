"""Copy an existing deployment artifact into a portable model release directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def deployment_manifest(manifest):
    """Keep the numerical format and tensor records used by the model loader."""
    result = {key: manifest[key] for key in (
        "schema_version", "status", "variant", "files", "total_file_bytes"
    )}
    result["checkpoint"] = {key: manifest["checkpoint"][key] for key in (
        "config_sha256", "index_sha256", "model_identity", "tokenizer_sha256"
    ) if key in manifest["checkpoint"]}
    calibration = manifest.get("calibration", {})
    result["calibration"] = {key: calibration[key] for key in (
        "train_only_calibration", "dataset", "new_w4_test_sets_used_for_weight_or_scale_selection"
    ) if key in calibration}
    tensor_fields = (
        "logical_id", "kind", "shape", "dtype", "payload", "offset_bytes", "length_bytes",
        "code_dtype", "scale_dtype", "scale_mode", "group_size", "code_payload", "scale_payload",
        "weight_offset_bytes", "weight_length_bytes", "scale_offset_bytes", "scale_length_bytes",
        "sha256", "codes_sha256", "scales_sha256", "parent_slice_sha256", "source_bf16_sha256",
        "code_min", "code_max",
    )
    result["tensors"] = [{key: entry[key] for key in tensor_fields if key in entry}
                         for entry in manifest["tensors"]]
    for section in ("weight_quantization", "gptq", "head_quantization"):
        if section in manifest:
            result[section] = {key: manifest[section][key] for key in (
                "algorithm", "bits", "sym", "signed_integer_range", "signed_zero_point",
                "qzeros", "group_size", "scale", "arithmetic_id", "rounding",
                "activation_bits", "activation_scale", "accumulator", "epilogue",
                "artifact_selection_metric",
            ) if key in manifest[section]}
    if "attention" in manifest:
        attention = manifest["attention"]
        result["attention"] = {key: attention[key] for key in (
            "q", "k_cache", "probability", "v_cache_dtype"
        ) if key in attention}
        result["attention"]["v_quantization"] = {
            key: attention["v_quantization"][key] for key in (
                "shape", "scale_dtype", "scale_mode", "scale_rule", "code_dtype",
                "signed_integer_range", "signed_zero_point", "qzeros", "rounding",
                "payload", "payload_sha256",
            ) if key in attention["v_quantization"]
        }
    if "rotation" in manifest:
        result["rotation"] = {key: manifest["rotation"][key] for key in (
            "seed", "sampling", "r1_sign_shape", "r2_sign_count", "r2_sign_shape",
            "payload_sha256", "r1", "r2", "r4",
        ) if key in manifest["rotation"]}
    return result


def copy_artifact(source, destination, *, manifest):
    destination.mkdir()
    (destination / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid artifact payload path: {name}")
        (original, copied) = (source / relative, destination / relative)
        if original.stat().st_size != expected["bytes"]:
            raise ValueError(f"payload size differs: {original}")
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        digest = hashlib.sha256()
        with copied.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"copied payload checksum differs: {copied}")
    return sum((entry["bytes"] for entry in manifest["files"].values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-artifact", type=Path, required=True)
    parser.add_argument("--head-artifact", type=Path, required=True)
    parser.add_argument("--silu-table", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(args.artifact_root.resolve())
    source_root = Path(__file__).resolve().parent
    release_root = (
        source_root.parent
        if source_root.name == "algorithm" and (source_root.parent / "hardware").is_dir()
        else source_root
    )
    project = next(
        (p for p in (source_root, *source_root.parents) if (p / ".git").exists()),
        release_root,
    )
    if (
        project == output
        or project in output.parents
        or output == args.artifact_root.resolve()
    ):
        raise ValueError("model payload must be outside the source repository")
    if output.exists():
        raise ValueError("use a new model release directory")
    parent_raw = (args.weight_artifact / "tensor_manifest.json").read_bytes()
    child = json.loads((args.head_artifact / "tensor_manifest.json").read_text())
    if (
        child.get("parent_artifact", {}).get("manifest_sha256")
        != hashlib.sha256(parent_raw).hexdigest()
    ):
        raise ValueError("head artifact does not reference the supplied W4 parent")
    silu = json.loads(args.silu_table.read_text())
    if silu.get("train_only_calibration") is not True:
        raise ValueError("SiLU table requires training-source provenance")
    output.mkdir(parents=True)
    parent = deployment_manifest(json.loads(parent_raw))
    child = deployment_manifest(child)
    size = copy_artifact(args.weight_artifact, output / "artifact_w4", manifest=parent)
    packaged_parent = (output / "artifact_w4/tensor_manifest.json").read_bytes()
    child["parent_artifact"] = dict(
        path="../artifact_w4", manifest_bytes=len(packaged_parent),
        manifest_sha256=hashlib.sha256(packaged_parent).hexdigest(),
        schema_version=parent["schema_version"], files=parent["files"],
    )
    size += copy_artifact(args.head_artifact, output / "artifact_w8_head", manifest=child)
    table = {key: silu[key] for key in (
        "schema_version", "train_only_calibration", "breakpoints", "slopes", "intercepts"
    )}
    (output / "silu_table_monotone.json").write_text(json.dumps(table, indent=2) + "\n")
    metadata = output / "model_metadata"
    metadata.mkdir()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        source = args.model_path / name
        if source.is_file():
            shutil.copyfile(source, metadata / name)
        elif name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            raise FileNotFoundError(source)
    card = source_root / "MODEL_CARD.md"
    if card.is_file():
        shutil.copyfile(card, output / "README.md")
    shutil.copyfile(source_root / "LICENSE", output / "LICENSE")
    upstream = output / "upstream"
    upstream.mkdir()
    for source in sorted(args.model_path.iterdir()):
        name = source.name.lower()
        if source.is_file() and (name == "readme.md" or name.split(".")[0] in
                                {"license", "licence", "notice", "copyright"}):
            shutil.copyfile(source, upstream / source.name)
    print(
        json.dumps(
            dict(output=str(output), payload_bytes=size, payload_checksums_match=True),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
