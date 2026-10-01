"""Collect train-source deployment trajectories, SiLU statistics and Hessians."""

from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from calibration.data import SILU_SAMPLING, load_gsm8k_training_records, load_code_training_records
from generation.engine import generate
from generation.records import ForwardCaptureEvent
from capture.replay import ForwardReplayer
from numerics.candidate import streaming_candidate_bf16
from model.modeling_llada import LLaDAModelLM
from numerics.precision import RowPrecisionContext
from quantization.model import (
    TARGET_NUMERIC_CAPTURE_GRAPH,
    SpinQuantW4A8Linear,
    install_target_numeric_lm_head,
    replace_with_spinquant_joint_full_w4a8_v8,
)
from quantization.rotation import INSTRUCT_CHECKPOINT, require_algo_output
from calibration.config import generator_arguments as _generator_arguments

SCHEMA_VERSION = "llada-target-numeric-matched-replay/v1"


def generation_arguments(
    task,
    context,
    *,
    gsm_feature3=True,
    human_feature3=False,
    full_prefix_oracle_block=None,
    steps=256,
    gen_length=256,
):
    if task not in ("gsm8k", "humaneval"):
        raise ValueError("unknown calibration task")
    arguments = _generator_arguments(
        task, context, feature3=gsm_feature3 if task == "gsm8k" else human_feature3,
        steps=steps, gen_length=gen_length
    )
    if full_prefix_oracle_block is not None:
        if full_prefix_oracle_block != 1:
            raise ValueError("the calibration profile uses block initialization block 1")
        arguments["cross_block_full_prefix_oracle_block"] = full_prefix_oracle_block
    return arguments


def load_records(args):
    if args.split not in ("train", "validation") or not args.train_only_calibration:
        raise ValueError(
            "calibration accepts only training-source train/validation records"
        )
    common = dict(
        manifest_path=args.dataset_manifest,
        split=args.split,
        sample_count=args.samples_per_task,
        seed=args.data_seed or args.seed,
        data_dir_override=args.data_dir,
        category_offset=0,
    )
    if args.record_task == "gsm8k":
        return load_gsm8k_training_records(**common)
    return load_code_training_records(
        **common,
        prompt_from_task=True,
        allowed_sources=("ai2-adapt-dev/personahub_code_v2_34999",),
    )


@torch.inference_mode()
def capture_record(
    model, context, record, ordinal, tokenizer, args, output, *, silu_statistics=None
):
    prompt_ids = target_prompt_ids(
        tokenizer, record, torch.device(args.device),
        generation_length=getattr(args, "gen_length", 256),
    )
    arguments = generation_arguments(
        record.task,
        context,
        gsm_feature3=args.gsm_feature3,
        human_feature3=args.human_feature3,
        full_prefix_oracle_block=args.full_prefix_oracle_block,
        steps=getattr(args, "steps", 256), gen_length=getattr(args, "gen_length", 256),
    )
    events = []
    started = time.monotonic()
    with silu_error_observers(
        model,
        enabled=silu_statistics is not None,
        statistics=silu_statistics,
    ):
        (tokens, nfe, traces, stats) = generate(
            model, prompt_ids, state_capture_callback=events.append, **arguments
        )
    if not events or len(events) != len(traces):
        raise RuntimeError("capture/trace event count mismatch")
    metadata = dict(
        anchor="n4",
        actual_linear_a8=False,
        source_split=args.split,
        source_category=record.category,
        train_only_calibration=True,
        dataset_manifest=str(args.dataset_manifest.resolve()),
        code_prompt_policy="full_task_and_signature",
        silu_table=getattr(args, "_silu_table_info", None),
        a4_clip_ratio_bf16=args.a4_clip_ratio,
        a4_output_clip_ratio_bf16=args.a4_output_clip_ratio,
        data_use="train_or_validation",
        sample_id=record.sample_id,
        task=record.task,
        ordinal=ordinal,
        numeric_capture_graph=TARGET_NUMERIC_CAPTURE_GRAPH,
        artifact_dir=str(args.artifact_dir.resolve()),
        w8_head_dir=str(args.w8_head_dir.resolve()),
        artifact_manifest_sha256=args._artifact_manifest_sha256,
        w8_head_manifest_sha256=args._w8_head_manifest_sha256,
        head_weight_bits=model.model.transformer.ff_out.weight_bits,
        generation=dict(
            arguments={
                k: v for (k, v) in arguments.items() if k != "row_precision_context"
            },
        ),
        seed=args.seed,
        model_path=str(args.model_path.resolve()),
        prompt_tokens=int(prompt_ids.shape[1]),
        silu_sampling=SILU_SAMPLING if silu_statistics is not None else None,
    )
    bank = output / f"sample_{ordinal:04d}.pt"
    torch.save(
        dict(
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
            events=[asdict(event) for event in events],
            trace=[asdict(t) for t in traces],
            generated_tokens=tokens.cpu(),
        ),
        bank,
    )
    phases = {}
    for event in events:
        row = phases.setdefault(event.forward_kind, dict(events=0, a4=0, a8=0))
        row["events"] += 1
        row["a4"] += int((event.row_bits == 4).sum())
        row["a8"] += int((event.row_bits == 8).sum())
    return dict(
        **metadata,
        nfe=nfe,
        seconds=time.monotonic() - started,
        bank=str(bank.resolve()),
        bank_bytes=bank.stat().st_size,
        phase_rows=phases,
        generation_text=tokenizer.decode(
            tokens[0, prompt_ids.shape[1] :].tolist(), skip_special_tokens=True
        ),
        reference=record.answer,
        stats=asdict(stats),
        metrics={},
        trace=None,
    )


def target_prompt_ids(tokenizer, record, device, *, generation_length=256):
    """Tokenize inference inputs without imposing training-target length rules."""
    prompt = record.prompt
    if record.task == "humaneval":
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    if generation_length <= 0:
        raise ValueError("generation_length must be positive")
    if ids.shape[1] + generation_length > 4096:
        raise ValueError(
            "calibration prompt plus generation exceeds the deployment context"
        )
    return ids.to(device)


def select_shard(records, index, count):
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    return list(enumerate(records))[index::count]


@contextmanager
def silu_error_observers(
    model, *, enabled, statistics=None
):
    if not enabled:
        yield None
        return
    statistics = {} if statistics is None else statistics
    restore = []
    try:
        for layer, block in enumerate(model.model.transformer.blocks):
            if not getattr(block, "_target_numeric_silu_pwl16", False):
                raise ValueError(
                    "SiLU observation requires the deployed sixteen-segment operator"
                )
            original = block._silu_multiply
            had_override = "_silu_multiply" in vars(block)
            previous = vars(block).get("_silu_multiply")
            row = statistics.setdefault(layer, dict(calls=0, sampled_elements=0))

            def observe(gate, up, *, original=original, row=row):
                result = original(gate, up)
                if (
                    gate.dtype != torch.bfloat16
                    or up.shape != gate.shape
                    or not gate.numel()
                ):
                    raise ValueError(
                        "SiLU observation requires matching BF16 gate/up tensors"
                    )
                (flat_gate, flat_up) = (
                    gate.detach().reshape(-1),
                    up.detach().reshape(-1),
                )
                count = min(flat_gate.numel(), 8192)
                stride = max(1, flat_gate.numel() // count)
                while math.gcd(stride, flat_gate.numel()) != 1:
                    stride += 1
                # The counter persists across requests within a logical capture shard.
                offset = row["calls"] * 8191
                indices = (
                    torch.arange(count, device=gate.device) * stride + offset
                ) % flat_gate.numel()
                (x, u) = (flat_gate[indices], flat_up[indices].float())
                actual = result.detach().reshape(-1)[indices].float()
                if not bool(torch.isfinite(x).all() & torch.isfinite(u).all()):
                    raise ValueError("nonfinite SiLU observation input")
                target = (
                    (torch.nn.functional.silu(x.float()).to(torch.bfloat16).float() * u)
                    .to(torch.bfloat16)
                    .float()
                )
                raw = x.contiguous().view(torch.int16).to(torch.int64) & 65535
                if "gate_counts" not in row:
                    row["gate_counts"] = torch.zeros(
                        65536, dtype=torch.int64, device=gate.device
                    )
                    row["up_squared_sum"] = torch.zeros(
                        65536, dtype=torch.float64, device=gate.device
                    )
                row["gate_counts"].scatter_add_(0, raw, torch.ones_like(raw))
                row["up_squared_sum"].scatter_add_(0, raw, u.double().square())
                for key, value in dict(
                    reference_energy=target.double().square().sum(),
                    squared_error=(actual.double() - target.double()).square().sum(),
                ).items():
                    row[key] = row.get(key, 0) + value
                row["calls"] += 1
                row["sampled_elements"] += count
                return result

            restore.append((block, had_override, previous))
            block._silu_multiply = observe
        yield statistics
    finally:
        for block, had_override, previous in restore:
            if had_override:
                block._silu_multiply = previous
            else:
                del block._silu_multiply


def linear_probe_event_indices(events, *, stratified=False, sample_ordinal=0):
    if not events:
        return set()
    selected = {0}
    block_initialization = [
        i for (i, e) in enumerate(events) if e.block_index == 1 and e.step_index == 0
    ]
    regular = [
        i
        for (i, e) in enumerate(events)
        if e.forward_kind == "local_block" and e.future_prediction_positions.numel()
    ]
    if block_initialization:
        selected.add(block_initialization[0])
    if regular:
        selected.update((regular[0], regular[-1]))
    if stratified:
        boundaries = [
            i
            for (i, e) in enumerate(events)
            if e.block_index >= 2
            and e.step_index == 0
            and (e.forward_kind != "local_block")
        ]
        if boundaries:
            selected.add(boundaries[sample_ordinal % len(boundaries)])
        confirmations = [
            i for (i, e) in enumerate(events) if "confirmation" in e.forward_kind
        ]
        current_only = [
            i
            for (i, e) in enumerate(events)
            if e.forward_kind == "local_block"
            and (not e.future_prediction_positions.numel())
            and (e.step_index > 0)
        ]
        late = (
            confirmations
            if confirmations and (sample_ordinal % 2 or not current_only)
            else current_only
        )
        if late:
            selected.add(late[-1])
    return selected


def linear_probe_rows(event, row_count, device):
    positions = event.input_positions.to(device)
    if len(positions) != row_count:
        if row_count != event.tokens_before.shape[1]:
            raise ValueError("cannot map Linear probe rows to logical positions")
        positions = torch.arange(row_count, device=device)
    wanted = torch.cat(
        (
            event.prediction_positions[event.prediction_mask[0]],
            event.future_prediction_positions,
        )
    ).to(device)
    preferred = torch.nonzero(torch.isin(positions, wanted), as_tuple=False).flatten()
    remaining = torch.nonzero(~torch.isin(positions, wanted), as_tuple=False).flatten()
    spare = max(0, 64 - len(preferred))
    if len(remaining) > spare:
        remaining = (
            remaining[
                torch.linspace(0, len(remaining) - 1, spare, device=device).long()
            ]
            if spare
            else remaining[:0]
        )
    return (positions, torch.sort(torch.cat((preferred[:64], remaining))).values)


def validate_replay_source(metadata, args, *, fitting):
    allowed = {"train"} if fitting else {"train", "validation"}
    if getattr(args, "train_only_calibration", False) and (
        metadata.get("train_only_calibration") is not True
        or metadata.get("source_split") not in allowed
        or metadata.get("source_split") != args.split
    ):
        raise ValueError(
            "train-only replay rejects test, unverified or mismatched capture split"
        )
    if fitting:
        from calibration.data import capture_numeric_source

        actual = capture_numeric_source(metadata)
        expected = dict(
            anchor="n4",
            actual_linear_a8=False,
            numeric_capture_graph=TARGET_NUMERIC_CAPTURE_GRAPH,
            artifact_manifest_sha256=getattr(args, "_artifact_manifest_sha256", None),
            w8_head_manifest_sha256=getattr(args, "_w8_head_manifest_sha256", None),
            head_weight_bits=getattr(args, "_head_weight_bits", None),
            a4_clip_ratio_bf16=args.a4_clip_ratio,
            a4_output_clip_ratio_bf16=args.a4_output_clip_ratio,
            silu_table=getattr(args, "_silu_table_info", None),
        )
        for key, value in expected.items():
            if actual[key] != value:
                raise ValueError(f"capture numerical provenance differs: {key}")


@torch.inference_mode()
def replay_linear_hessians(model, context, banks, args, output):
    from numerics.bf16 import quantize_activation_per_row_bits_bf16

    (active, hessians, coverage) = ({}, {}, {})
    source_records = []
    handles = []

    def observe(module, inputs, result):
        if not active.get("enabled"):
            return
        event = active["event"]
        values = inputs[0].reshape(-1, module.in_features)
        bits = context.require(
            len(values), values.device
        )
        (positions, indices) = linear_probe_rows(event, len(values), values.device)
        (values, bits) = (values[indices], bits[indices])
        quantized = quantize_activation_per_row_bits_bf16(values, bits)
        rows = (
            (quantized.codes.float() * quantized.scale_bf16[:, None])
            .to(torch.bfloat16)
            .float()
        )
        name = module.module_name
        if name not in hessians:
            hessians[name] = torch.zeros(
                (module.in_features, module.in_features),
                device=values.device,
                dtype=torch.float32,
            )
        hessians[name].addmm_(rows.T, rows)
        phase = (
            "block_initialization"
            if event.block_index == 1 and event.step_index == 0
            else event.forward_kind
        )
        counts = coverage.setdefault(name, {}).setdefault(
            phase, dict(rows=0, a4_rows=0, a8_rows=0, calls=0)
        )
        counts["rows"] += len(rows)
        counts["a4_rows"] += int((bits == 4).sum())
        counts["a8_rows"] += int((bits == 8).sum())
        counts["calls"] += 1

    for name, module in model.named_modules():
        if isinstance(module, SpinQuantW4A8Linear) and ".blocks." in name:
            if module.weight_group_size != -1 or name != module.module_name:
                raise ValueError(
                    "Hessian collection requires named G-1 Transformer Linears"
                )
            handles.append(module.register_forward_hook(observe))
    if len(handles) != 224:
        raise ValueError("Hessian collection must cover 224 Transformer Linears")
    try:
        for ordinal, bank in banks:
            payload = torch.load(bank, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            validate_replay_source(metadata, args, fitting=True)
            from calibration.data import capture_numeric_source

            source_records.append(
                {
                    key: metadata.get(key)
                    for key in (
                        "sample_id",
                        "task",
                        "ordinal",
                        "source_split",
                        "train_only_calibration",
                        "dataset_manifest",
                        "generation",
                        "seed",
                        "code_prompt_policy",
                    )
                }
            )
            source_records[-1]["numeric_source"] = capture_numeric_source(metadata)
            events = [ForwardCaptureEvent(**r) for r in payload["events"]]
            selected = linear_probe_event_indices(
                events, stratified=True, sample_ordinal=ordinal
            )
            replayer = ForwardReplayer(model, context, device=torch.device(args.device))
            for i, event in enumerate(events):
                active.update(enabled=i in selected, event=event)
                result = replayer.step(event)
                (top1, _, _) = streaming_candidate_bf16(result.prediction_logits)
                if not torch.equal(
                    top1.cpu(),
                    event.teacher_top1_token_ids[:, event.prediction_mask[0]],
                ):
                    raise RuntimeError("Hessian observation changed student replay")
                if event.future_prediction_positions.numel():
                    (top1, _, _) = streaming_candidate_bf16(
                        result.future_prediction_logits
                    )
                    if not torch.equal(top1.cpu(), event.future_top1_token_ids):
                        raise RuntimeError("Hessian observation changed future replay")
                replayer.detach_cache()
            print(
                json.dumps(
                    dict(
                        stage="hessian_sample_complete",
                        sample_id=payload["metadata"]["sample_id"],
                        measured_events=len(selected),
                    )
                ),
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    for name, matrix in hessians.items():
        if not torch.isfinite(matrix).all():
            raise ValueError(f"nonfinite Hessian: {name}")
        field = "hessian"
        torch.save(
            {
                field: matrix.cpu(),
                "token_rows": sum((r["rows"] for r in coverage[name].values())),
            },
            output / (name + ".pt"),
        )
    (output / "hessian_coverage.json").write_text(
        json.dumps(
            dict(
                definition="sum X.T@X; X=BF16(A_codes*A_scale); unweighted sampled rows",
                event_sampling="stratified_max6_sorted_file_ordinal/v1",
                banks=[str(bank) for (_, bank) in banks],
                modules=coverage,
                train_only_calibration=getattr(args, "train_only_calibration", False),
                source_records=source_records,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("prepare", "capture", "linear-hessians"), required=True
    )
    parser.add_argument("--model-path", type=Path, default=INSTRUCT_CHECKPOINT)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--w8-head-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--silu-errors", action="store_true")
    parser.add_argument("--silu-table", type=Path)
    parser.add_argument("--samples-per-task", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--full-prefix-oracle-block", type=int, default=1)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--record-task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--train-only-calibration", action="store_true", required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--a4-clip-ratio", type=float, default=1.0)
    parser.add_argument("--a4-output-clip-ratio", type=float)
    parser.add_argument(
        "--gsm-feature3", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--human-feature3", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples_per_task <= 0:
        parser.error("samples-per-task must be positive")
    _generator_arguments(args.record_task, None, feature3=False, steps=args.steps, gen_length=args.gen_length)
    select_shard([], args.shard_index, args.shard_count)
    if args.mode == "linear-hessians" and (
        args.split != "train" or args.capture_root is None
    ):
        parser.error("Hessians require train captures and capture-root")
    args.a4_clip_ratio = float(torch.tensor(args.a4_clip_ratio, dtype=torch.bfloat16))
    args.a4_output_clip_ratio = (
        args.a4_clip_ratio
        if args.a4_output_clip_ratio is None
        else float(torch.tensor(args.a4_output_clip_ratio, dtype=torch.bfloat16))
    )
    if not 0 < args.a4_clip_ratio <= 1 or not 0 < args.a4_output_clip_ratio <= 1:
        parser.error("A4 clipping ratios must be finite and in (0,1]")
    if args.mode != "prepare" and args.output_dir is None:
        parser.error("capture and Hessians require output-dir")
    output = None if args.output_dir is None else require_algo_output(args.output_dir)
    if output is not None and output.exists():
        parser.error("use a fresh output directory")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.mode == "linear-hessians":
        banks = sorted(args.capture_root.glob("**/sample_*.pt"))
        inputs = select_shard(banks, args.shard_index, args.shard_count)
    else:
        inputs = select_shard(
            load_records(args), args.shard_index, args.shard_count
        )
    if not inputs:
        raise ValueError("shard has no inputs")
    if args.mode == "prepare":
        for ordinal, record in inputs:
            prompt_ids = target_prompt_ids(
                tokenizer, record, torch.device("cpu"), generation_length=args.gen_length
            )
            print(
                json.dumps(
                    dict(
                        ordinal=ordinal,
                        sample_id=record.sample_id,
                        task=record.task,
                        category=record.category,
                        prompt_tokens=prompt_ids.numel(),
                        completion_tokens=None,
                    )
                ),
                flush=True,
            )
        return
    from quantization.artifact import (
        SpinQuantArtifactReader,
        require_train_only_artifact,
    )

    for path, field in (
        (args.artifact_dir, "_artifact_manifest_sha256"),
        (args.w8_head_dir, "_w8_head_manifest_sha256"),
    ):
        reader = SpinQuantArtifactReader(path, verify_files=False)
        require_train_only_artifact(reader)
        setattr(args, field, reader.manifest_sha256)
    output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    context = RowPrecisionContext()
    model = (
        LLaDAModelLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(torch.device(args.device))
    )
    installation = replace_with_spinquant_joint_full_w4a8_v8(
        model,
        args.artifact_dir,
        variant="fixed-r1r2-r4-only",
        row_precision_context=context,
        activation_bits=8,
    )
    if not isinstance(model.model.transformer.ff_out, SpinQuantW4A8Linear):
        install_target_numeric_lm_head(model, artifact_dir=args.w8_head_dir)
    if model.model.transformer.ff_out.weight_bits != 8:
        raise ValueError("calibration requires the W8 head")
    args._head_weight_bits = 8
    print(
        json.dumps(
            dict(stage="model_ready", installation=installation, samples=len(inputs))
        ),
        flush=True,
    )
    if args.silu_table is not None:
        from quantization.model import override_target_silu_table

        table = json.loads(args.silu_table.read_text())
        if table.get("train_only_calibration") is not True:
            raise ValueError("SiLU table lacks training-source provenance")
        override_target_silu_table(model, table)
        args._silu_table_info = {
            k: table[k] for k in ("breakpoints", "slopes", "intercepts")
        }
    if args.a4_clip_ratio != 1.0 or args.a4_output_clip_ratio != 1.0:
        from quantization.model import install_target_a4_clipping

        install_target_a4_clipping(
            model, args.a4_clip_ratio, output_ratio=args.a4_output_clip_ratio
        )
    if args.mode == "linear-hessians":
        replay_linear_hessians(model, context, inputs, args, output)
        return
    silu_statistics = {} if args.silu_errors else None
    with (output / "samples.jsonl").open("w") as summary:
        for ordinal, record in inputs:
            result = capture_record(
                model,
                context,
                record,
                ordinal,
                tokenizer,
                args,
                output,
                silu_statistics=silu_statistics,
            )
            summary.write(json.dumps(result) + "\n")
            summary.flush()
            print(
                json.dumps(
                    dict(
                        stage="sample_complete",
                        ordinal=ordinal,
                        seconds=result["seconds"],
                        nfe=result["nfe"],
                    )
                ),
                flush=True,
            )
        if silu_statistics is not None:
            torch.save(
                {
                    layer: {
                        key: value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                        for (key, value) in row.items()
                    }
                    for (layer, row) in silu_statistics.items()
                },
                output / "silu_histograms.pt",
            )
            measurements = [
                dict(
                    layer=layer,
                    **{
                        key: value.item() if isinstance(value, torch.Tensor) else value
                        for (key, value) in row.items()
                        if key not in ("gate_counts", "up_squared_sum")
                    },
                )
                for (layer, row) in silu_statistics.items()
            ]
            (output / "silu_errors.json").write_text(
                json.dumps(
                    dict(
                        train_only_calibration=True,
                        source_split=args.split,
                        scope="At most 8192 elements per call, coprime stride; offset = layer call count * 8191 within each logical shard; original output returned",
                        sampling=SILU_SAMPLING,
                        reference="BF16(BF16(native SiLU(BF16 gate))*BF16 up)",
                        artifact_dir=str(args.artifact_dir),
                        measurements=measurements,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
