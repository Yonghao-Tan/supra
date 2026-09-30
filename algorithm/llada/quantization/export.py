"""Export fixed-rotation RTN or precomputed G-1 GPTQ with an unchanged W8 head."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import torch
from model.modeling_llada import LLaDAModelLM
from quantization.artifact import (
    EMBEDDING,
    LM_HEAD,
    SpinQuantArtifactReader,
    SpinQuantArtifactWriter,
    SpinQuantLMHeadArtifactWriter,
    expected_w4_ids,
)
from quantization.gptq import apply_fixed_spinquant
from quantization.numeric import (
    SpinQuantW4Tensor,
    quantize_symmetric_w4_codes,
    symmetric_w4_scale_bf16,
)
from quantization.rotation import require_algo_output, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--output-child", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=(-1,), default=-1)
    parser.add_argument("--quantized-linear-root", type=Path)
    parser.add_argument("--train-only-calibration", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    solved = {}
    if args.quantized_linear_root is not None:
        if (args.quantized_linear_root / "exit_code").read_text().strip() != "0":
            raise ValueError("precomputed GPTQ requires a completed solver run")
        for path in args.quantized_linear_root.glob("shard_*/*.pt"):
            if path.stem in solved:
                raise ValueError("duplicate precomputed Linear")
            solved[path.stem] = path
        if set(solved) != {name.removesuffix(".weight") for name in expected_w4_ids()}:
            raise ValueError(
                "precomputed weights must cover exactly 224 Transformer Linears"
            )
    output_parent = require_algo_output(args.output_parent)
    output_child = require_algo_output(args.output_child)
    if output_parent.exists() or output_child.exists():
        raise FileExistsError("RTN exporter refuses to overwrite an existing artifact")
    output_parent.parent.mkdir(parents=True, exist_ok=True)
    output_child.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset_manifest is not None:
        dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
        if dataset_manifest.get("schema_version") not in {
            3,
            "target-evaluation-records/v1",
        }:
            raise ValueError("unsupported optional dataset provenance manifest")
    reference = SpinQuantArtifactReader(args.reference_artifact, verify_files=False)
    if args.train_only_calibration:
        from quantization.artifact import require_train_only_artifact

        require_train_only_artifact(reference)
    if (
        not reference.head_quantized
        or reference.head_weight_bits != 8
        or reference.head_group_size != -1
        or (reference.variant != "fixed-r1r2-r4-only")
    ):
        raise ValueError(
            "reference artifact must be fixed-R1/R2/R4 W4 plus G-1 W8 head"
        )
    reference_parent = reference._parent_reader
    if reference_parent is None or not reference_parent.joint_v8:
        raise ValueError("reference W8 child must resolve its joint V8 parent")
    solver_summaries = []
    if solved:
        solver_summaries = [
            json.loads(p.read_text())
            for p in sorted(args.quantized_linear_root.glob("shard_*/summary.json"))
        ]
        declared = [m["module"] for s in solver_summaries for m in s["modules"]]
        if len(declared) != len(solved) or set(declared) != set(solved):
            raise ValueError("solver summaries and precomputed tensors differ")
        method = "full_covariance_gminus1_gptq"
        if any(
            (
                s["method"] != method
                or s.get("group_size", -1) != args.group_size
                or s["source_artifact_manifest_sha256"]
                != reference_parent.manifest_sha256
                for s in solver_summaries
            )
        ):
            raise ValueError("solver used a different method or source artifact")
        if (
            len(
                {
                    (
                        s["hessian_root"],
                        s["calibration_requests"],
                        s["percdamp"],
                        s["block_size"],
                        s.get("act_order", False),
                        tuple(s.get("scale_search_ratios") or ()),
                        s.get("preserve_hessian_root"),
                        s.get("preserve_calibration_requests", 0),
                        s.get("joint_relative_error", False),
                    )
                    for s in solver_summaries
                }
            )
            != 1
        ):
            raise ValueError("solver shards used different calibration settings")
    device = torch.device(args.device)
    if (
        args.train_only_calibration
        and solved
        and any((s.get("train_only_calibration") is not True for s in solver_summaries))
    ):
        raise ValueError("train-only export rejects test or unverified solver results")
    model = LLaDAModelLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    rotation = apply_fixed_spinquant(
        model,
        variant="fixed-r1r2-r4-only",
        device=device,
        offload_after_transform=True,
        target_numeric_r4=True,
    )
    calibration = {
        "purpose": "Fresh BF16 fixed-rotation signed W4 RTN; weights determine scales",
        "new_w4_test_sets_used_for_weight_or_scale_selection": False,
        "reused_attention_and_head_source": str(args.reference_artifact.resolve()),
        "reused_attention_and_head_manifest_sha256": reference.manifest_sha256,
        "calibration_rows": 0,
        "train_only_calibration": args.train_only_calibration,
    }
    if args.dataset_manifest is not None:
        calibration.update(
            dataset_manifest=str(args.dataset_manifest.resolve()),
            dataset_manifest_bytes=args.dataset_manifest.stat().st_size,
            dataset_manifest_sha256=sha256_file(args.dataset_manifest),
        )
    if solved:
        calibration.update(
            purpose=f"G{args.group_size} GPTQ on actual A4/A8 three-Feature deployment inputs",
            new_w4_test_sets_used_for_weight_or_scale_selection=not args.train_only_calibration,
            quantized_linear_root=str(args.quantized_linear_root.resolve()),
            solver_shards=solver_summaries,
            calibration_rows=min(
                (m["token_rows"] for s in solver_summaries for m in s["modules"])
            ),
        )
    group_size = int(args.group_size)
    arithmetic_id = "gminus1_bf16_dual_scale/v1"
    writer = SpinQuantArtifactWriter(
        args.checkpoint,
        output_parent,
        variant="fixed-r1r2-r4-only",
        calibration=calibration,
        rotation=rotation,
        weight_quantization={
            "tensor_kind": "gptq_w4" if solved else "rtn_w4",
            "algorithm": (
                "signed_symmetric_per_output_channel_gptq"
                if solved
                else "signed_symmetric_per_output_channel_rtn"
            ),
            "bits": 4,
            "sym": True,
            "signed_integer_range": [-8, 7],
            "signed_zero_point": 0,
            "qzeros": False,
            "group_size": group_size,
            "scale": "bf16_per_output_channel_gptq_range_search"
            if solved and solver_summaries[0].get("scale_search_ratios")
            else "bf16_per_output_channel_2amax_div_15",
            "arithmetic_id": arithmetic_id,
            "rounding": "round_to_nearest_even",
            "target_linear_count": 224,
            "artifact_selection_metric": "none_fixed_parent",
        },
    )
    try:
        writer.write_bf16(EMBEDDING, model.model.transformer.wte.weight)
        writer.write_bf16(LM_HEAD, model.model.transformer.ff_out.weight)
        for index, logical_id in enumerate(sorted(expected_w4_ids()), start=1):
            module = model.get_submodule(logical_id[: -len(".weight")])
            source = module.weight.detach().to(device=device, dtype=torch.bfloat16)
            statistics = {
                "index": index,
                "source_abs_max": float(source.float().abs().max()),
            }
            if solved:
                payload = torch.load(
                    solved[logical_id.removesuffix(".weight")],
                    map_location="cpu",
                    weights_only=True,
                )
                (codes, scale) = (payload["codes"], payload["scales_bf16"])
                if codes.dtype != torch.int8 or codes.shape != source.shape:
                    raise ValueError(f"invalid precomputed codes: {logical_id}")
                expected_scale_shape = (source.shape[0],)
                if (
                    scale.dtype not in (torch.bfloat16, torch.float32)
                    or scale.shape != expected_scale_shape
                    or (
                        not torch.equal(scale.float(), scale.to(torch.bfloat16).float())
                    )
                ):
                    raise ValueError(f"invalid precomputed scales: {logical_id}")
                scale = scale.to(torch.bfloat16)
                statistics.update(payload["statistics"])
            else:
                scale = symmetric_w4_scale_bf16(source, group_size=group_size)
                codes = quantize_symmetric_w4_codes(
                    source, scale, group_size=group_size
                )
            writer.write_w4(
                logical_id,
                SpinQuantW4Tensor(
                    codes.cpu(),
                    scale.cpu(),
                    scale_mode="per_output_channel",
                ),
                source_shape=list(source.shape),
                calibration_rows=statistics.get("token_rows", 0),
                quantization_statistics={
                    **statistics,
                    "scale_min": float(scale.min()),
                    "scale_max": float(scale.max()),
                },
            )
            del source, scale, codes
        v_record = reference_parent.manifest["attention"]["v_quantization"]
        writer.write_v_cache_calibration(
            reference_parent.read_v_cache_scales(), statistics=v_record["statistics"]
        )
        parent_manifest = writer.finish()
    except Exception:
        writer.abort()
        raise
    parent = SpinQuantArtifactReader(output_parent, verify_files=True)
    reference_head_entry = reference.entries[LM_HEAD]
    new_head_entry = parent.entries[LM_HEAD]
    if reference_head_entry.get("source_bf16_sha256") != new_head_entry.get("sha256"):
        raise RuntimeError("fixed rotation produced a different BF16 lm_head")
    child_writer = SpinQuantLMHeadArtifactWriter(
        output_parent,
        output_child,
        calibration=reference.manifest.get("calibration", {}),
        quantization=reference.manifest["head_quantization"],
    )
    try:
        child_manifest = child_writer.write(
            reference.read_quantized_head(),
            quantization_statistics={
                **reference_head_entry.get("quantization_statistics", {}),
                "rebound_from_manifest_sha256": reference.manifest_sha256,
            },
        )
    except Exception:
        child_writer.abort()
        raise
    os.environ["SUPRA_SPINQUANT_PARENT_ARTIFACT_DIR"] = str(output_parent)
    verified = SpinQuantArtifactReader(output_child, verify_files=False)
    print(
        json.dumps(
            {
                "parent": {
                    "path": str(output_parent),
                    "manifest_sha256": parent.manifest_sha256,
                    "total_file_bytes": parent_manifest["total_file_bytes"],
                },
                "child": {
                    "path": str(output_child),
                    "manifest_sha256": verified.manifest_sha256,
                    "total_file_bytes": child_manifest["total_file_bytes"],
                },
                "w4_kind": "gptq_w4" if solved else "rtn_w4",
                "w4_group_size": group_size,
                "transformer_weight_arithmetic": arithmetic_id,
                "lm_head": "rebound_signed_w8_gminus1_rtn",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
