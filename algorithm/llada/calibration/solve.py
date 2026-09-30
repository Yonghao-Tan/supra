"""Solve static-scale GPTQ from complete Hessians of captured deployment inputs."""

import argparse
import json
from pathlib import Path
import time
import torch
from model.modeling_llada import LLaDAModelLM
from quantization.artifact import SpinQuantArtifactReader, expected_w4_ids
from quantization.gptq import NoGroupGPTQ, apply_fixed_spinquant
from quantization.numeric import SpinQuantW4Tensor, symmetric_w4_scale_bf16
from quantization.rotation import require_algo_output
from calibration.data import capture_sample_identity, capture_numeric_source


def gptq_module_order(model):
    def key(logical_id):
        layer = model.get_submodule(logical_id.removesuffix(".weight"))
        return (-layer.in_features, -layer.out_features, logical_id)

    return sorted(expected_w4_ids(), key=key)


def hessian_sources(root, *, train_only=False, artifact_identity=None):
    if (root / "exit_code").read_text().strip() != "0":
        raise ValueError("Hessian collection did not exit successfully")
    sources = sorted(root.glob("shard_*/hessian_coverage.json"))
    if not sources:
        raise ValueError("no completed Hessian shards")
    (banks, names) = (
        set(),
        {name.removesuffix(".weight") for name in expected_w4_ids()},
    )
    (identities, task_numerics) = (set(), {})
    sampling = set()
    for path in sources:
        coverage = json.loads(path.read_text())
        records = coverage.get("source_records", [])
        if len(records) != len(coverage["banks"]):
            raise ValueError("missing Hessian source records")
        sampling.add(coverage.get("event_sampling"))
        if len(sampling) != 1:
            raise ValueError("Hessian event sampling definitions differ")
        for record in records:
            identity = capture_sample_identity(record)
            if identity in identities:
                raise ValueError(f"duplicate calibration sample: {identity}")
            identities.add(identity)
            if train_only or artifact_identity is not None:
                numeric = capture_numeric_source(record.get("numeric_source", {}))
                if not isinstance(record.get("generation"), dict) or not record[
                    "generation"
                ].get("arguments"):
                    raise ValueError("missing Hessian capture generation provenance")
                if (
                    artifact_identity is not None
                    and numeric["artifact_manifest_sha256"] != artifact_identity
                ):
                    raise ValueError("Hessian artifact differs from solver baseline")
                task = identity[0]
                if task in task_numerics and task_numerics[task] != numeric:
                    raise ValueError(
                        f"Hessian numerical provenance differs for task {task}"
                    )
                task_numerics[task] = numeric
        if train_only:
            records = coverage.get("source_records", [])
            if (
                coverage.get("train_only_calibration") is not True
                or len(records) != len(coverage["banks"])
                or any(
                    (
                        r.get("source_split") != "train"
                        or r.get("train_only_calibration") is not True
                        for r in records
                    )
                )
            ):
                raise ValueError("train-only GPTQ rejects test or unverified Hessians")
        if set(coverage["modules"]) != names:
            raise ValueError("Hessian module coverage differs")
        part = coverage["banks"]
        if len(part) != len(set(part)) or banks.intersection(part):
            raise ValueError("duplicate calibration requests")
        banks.update(part)
    return (sources, banks)


def load_hessian(sources, name, columns, device, *, field="hessian"):
    (total, count) = (None, 0)
    for source in sources:
        part = torch.load(
            source.parent / (name + ".pt"), map_location="cpu", weights_only=True
        )
        matrix = part[field]
        if matrix.dtype != torch.float32 or matrix.shape != (columns, columns):
            raise ValueError(f"invalid Hessian shape/dtype: {name}")
        if not torch.isfinite(matrix).all() or part["token_rows"] <= 0:
            raise ValueError(f"invalid Hessian data: {name}")
        if total is None:
            total = matrix.to(device)
        else:
            total.add_(matrix.to(device))
        count += int(part["token_rows"])
    return (total, count)


def representation_error(
    weight, codes, scale, hessian, *, per_output=False
):
    total = 0.0
    channel_errors = []
    for start in range(0, weight.shape[0], 128):
        end = min(start + 128, weight.shape[0])
        scales = (
            scale[start:end, None]
            if scale.ndim == 1
            else scale[start:end].repeat_interleave(
                weight.shape[1] // scale.shape[1], dim=1
            )
        )
        decoded = (codes[start:end].float() * scales).to(torch.bfloat16).float()
        delta = decoded - weight[start:end].float()
        terms = (delta @ hessian * delta).double()
        if not per_output:
            total += float(terms.sum())
        if per_output:
            channel_errors.append(terms.sum(1))
    return torch.cat(channel_errors) if per_output else total


def quantize_with_scale_search(
    layer,
    hessian,
    rows,
    *,
    ratios=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
    incumbent=None,
    preserve_hessian=None,
):
    """Choose each G-1 channel's scale after complete GPTQ error feedback."""
    if not ratios or ratios[0] != 1.0 or any((not 0 < ratio <= 1 for ratio in ratios)):
        raise ValueError(
            "scale search must retain the original scale as its first candidate"
        )
    original = layer.weight.detach().clone()
    base_scale = symmetric_w4_scale_bf16(original)
    (best, best_errors, choices) = (None, None, None)
    preserved_errors = None
    if incumbent is None and preserve_hessian is not None:
        raise ValueError("preservation Hessian requires an incumbent")
    if incumbent is not None:
        if incumbent.group_size != -1 or incumbent.codes.shape != original.shape:
            raise ValueError("constrained search requires matching G-1 incumbent")
        best = SpinQuantW4Tensor(
            incumbent.codes.cpu().clone(), incumbent.scale_bf16.cpu().clone()
        )
        best_errors = representation_error(
            original,
            best.codes.to(original.device),
            best.scale_bf16.to(original.device),
            hessian,
            per_output=True,
        ).cpu()
        if preserve_hessian is not None:
            preserved_errors = representation_error(
                original,
                best.codes.to(original.device),
                best.scale_bf16.to(original.device),
                preserve_hessian,
                per_output=True,
            ).cpu()
        if not torch.isfinite(best_errors).all() or (
            preserved_errors is not None
            and (not torch.isfinite(preserved_errors).all())
        ):
            raise ValueError("nonfinite incumbent reconstruction error")
        choices = torch.full((len(best_errors),), -1, dtype=torch.int64)
    try:
        for index, ratio in enumerate(ratios):
            layer.weight.data.copy_(original)
            solver = NoGroupGPTQ(layer)
            (solver.hessian, solver.token_rows) = (hessian.clone(), rows)
            scales = (base_scale.float() * ratio).to(torch.bfloat16).float()
            candidate = solver.quantize(fixed_scale_bf16=scales)
            errors = representation_error(
                original,
                candidate.codes.to(original.device),
                candidate.scale_bf16.to(original.device),
                hessian,
                per_output=True,
            ).cpu()
            if not torch.isfinite(errors).all():
                raise ValueError("nonfinite scale-search reconstruction error")
            if best is None:
                (best, best_errors) = (candidate, errors)
                choices = torch.zeros(len(errors), dtype=torch.int64)
            else:
                improved = errors < best_errors
                if preserve_hessian is not None:
                    candidate_preserved = representation_error(
                        original,
                        candidate.codes.to(original.device),
                        candidate.scale_bf16.to(original.device),
                        preserve_hessian,
                        per_output=True,
                    ).cpu()
                    if not torch.isfinite(candidate_preserved).all():
                        raise ValueError("nonfinite preservation reconstruction error")
                    improved &= candidate_preserved <= preserved_errors
                best.codes[improved] = candidate.codes[improved]
                best.scale_bf16[improved] = candidate.scale_bf16[improved]
                best_errors[improved] = errors[improved]
                choices[improved] = index
    finally:
        layer.weight.data.copy_(original)
    decoded = (
        best.codes.to(original.device).float()
        * best.scale_bf16.to(original.device)[:, None]
    )
    layer.weight.data.copy_(decoded.to(original.dtype))
    report = dict(
        ratios=list(ratios),
        selected_output_counts=[
            int((choices == index).sum()) for index in range(len(ratios))
        ],
    )
    if incumbent is not None:
        report["incumbent_output_count"] = int((choices == -1).sum())
        report["selection"] = (
            "minimize primary error subject to per-output preservation error <= incumbent"
            if preserve_hessian is not None
            else "minimize primary error including incumbent"
        )
    return (best, report)


def relative_task_hessian(primary, secondary, primary_error, secondary_error):
    """Give equal importance to fractional task-error changes, independent of row counts."""
    import math

    if primary.shape != secondary.shape or not all(
        (math.isfinite(x) and x > 0 for x in (primary_error, secondary_error))
    ):
        raise ValueError(
            "relative task objective requires matching Hessians and positive finite errors"
        )
    return primary / primary_error + secondary / secondary_error


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--hessian-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=7)
    parser.add_argument("--expected-requests", type=int, default=128)
    parser.add_argument("--group-size", type=int, choices=(-1,), default=-1)
    parser.add_argument("--scale-search", action="store_true")
    parser.add_argument("--preserve-hessian-root", type=Path)
    parser.add_argument("--preserve-expected-requests", type=int, default=128)
    parser.add_argument("--joint-relative-error", action="store_true")
    parser.add_argument("--train-only-calibration", action="store_true", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.preserve_hessian_root is not None and (not args.scale_search):
        raise ValueError("preservation requires G-1 scale search")
    if args.joint_relative_error and args.preserve_hessian_root is None:
        raise ValueError("joint relative objective requires the second task Hessian")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid solver shard")
    output = require_algo_output(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    baseline = SpinQuantArtifactReader(args.artifact_dir, verify_files=False)
    (sources, banks) = hessian_sources(
        args.hessian_root, train_only=True, artifact_identity=baseline.manifest_sha256
    )
    if len(banks) != args.expected_requests:
        raise ValueError("unexpected calibration request count")
    (preserve_sources, preserve_banks) = (None, set())
    if args.preserve_hessian_root is not None:
        (preserve_sources, preserve_banks) = hessian_sources(
            args.preserve_hessian_root,
            train_only=True,
            artifact_identity=baseline.manifest_sha256,
        )
        if len(preserve_banks) != args.preserve_expected_requests:
            raise ValueError("unexpected preservation request count")
        definition = "sum X.T@X; X=BF16(A_codes*A_scale); unweighted sampled rows"
        if any(
            (
                json.loads(p.read_text())["definition"] != definition
                for p in sources + preserve_sources
            )
        ):
            raise ValueError("constrained search Hessian definitions differ")
    device = torch.device(args.device)
    from quantization.artifact import require_train_only_artifact

    require_train_only_artifact(baseline)
    model = LLaDAModelLM.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    apply_fixed_spinquant(
        model,
        variant="fixed-r1r2-r4-only",
        device=device,
        offload_after_transform=True,
        target_numeric_r4=True,
    )
    output.mkdir(parents=True)
    metrics = []
    for logical_id in gptq_module_order(model)[args.shard_index :: args.shard_count]:
        started = time.monotonic()
        name = logical_id.removesuffix(".weight")
        layer = model.get_submodule(name).to(device)
        original = layer.weight.detach().clone()
        (hessian, rows) = load_hessian(sources, name, layer.in_features, device)
        prior = baseline.read_w4(logical_id)
        old_error = representation_error(
            original, prior.codes.to(device), prior.scale_bf16.to(device), hessian
        )
        search_report = None
        solver = None
        preserve_hessian = None
        preserve_rows = None
        if preserve_sources is not None:
            (preserve_hessian, preserve_rows) = load_hessian(
                preserve_sources, name, layer.in_features, device
            )
        solve_hessian = hessian
        preserve_old_error = None
        if args.joint_relative_error:
            preserve_old_error = representation_error(
                original,
                prior.codes.to(device),
                prior.scale_bf16.to(device),
                preserve_hessian,
            )
            solve_hessian = relative_task_hessian(
                hessian, preserve_hessian, old_error, preserve_old_error
            )
        if args.scale_search:
            (quantized, search_report) = quantize_with_scale_search(
                layer,
                solve_hessian,
                rows,
                incumbent=prior if preserve_sources is not None else None,
                preserve_hessian=None
                if args.joint_relative_error
                else preserve_hessian,
            )
        else:
            solver = NoGroupGPTQ(layer)
            solver.hessian = hessian.clone()
            solver.token_rows = rows
            quantized = solver.quantize(group_size=-1, act_order=False)
        new_error = representation_error(
            original,
            quantized.codes.to(device),
            quantized.scale_bf16.to(device),
            hessian,
        )
        report = dict(
            module=name,
            token_rows=rows,
            old_error=old_error,
            new_error=new_error,
            seconds=time.monotonic() - started,
        )
        if search_report is not None:
            report["scale_search"] = search_report
        if preserve_hessian is not None:
            report.update(
                preserve_token_rows=preserve_rows,
                preserve_old_error=preserve_old_error
                if preserve_old_error is not None
                else representation_error(
                    original,
                    prior.codes.to(device),
                    prior.scale_bf16.to(device),
                    preserve_hessian,
                ),
                preserve_new_error=representation_error(
                    original,
                    quantized.codes.to(device),
                    quantized.scale_bf16.to(device),
                    preserve_hessian,
                ),
            )
        if args.joint_relative_error:
            report["relative_joint_old_error"] = 2.0
            report["relative_joint_new_error"] = (
                new_error / old_error
                + report["preserve_new_error"] / preserve_old_error
            )
        torch.save(
            dict(
                codes=quantized.codes,
                scales_bf16=quantized.scale_bf16,
                statistics=report,
            ),
            output / (name + ".pt"),
        )
        metrics.append(report)
        print(json.dumps(report), flush=True)
        layer.cpu()
        del solver, hessian, original, prior, quantized, preserve_hessian, solve_hessian
    (output / "summary.json").write_text(
        json.dumps(
            dict(
                method="full_covariance_gminus1_gptq",
                group_size=-1,
                hessian_root=str(args.hessian_root),
                source_artifact_manifest_sha256=baseline.manifest_sha256,
                hessian_coverage_sources=[
                    str(p) for p in sources + (preserve_sources or [])
                ],
                calibration_requests=len(banks),
                percdamp=0.01,
                block_size=128,
                act_order=False,
                scale_search_ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
                if args.scale_search
                else None,
                preserve_hessian_root=str(args.preserve_hessian_root)
                if preserve_sources is not None
                else None,
                preserve_calibration_requests=len(preserve_banks),
                joint_relative_error=args.joint_relative_error,
                train_only_calibration=True,
                scope="fixed incumbent inputs; BF16 rotated reference weights; no activation compensation",
                modules=metrics,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
