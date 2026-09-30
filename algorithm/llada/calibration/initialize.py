"""Initialize G-1 W4 weights with symmetric LWC and sequential V8 calibration.

WikiText training inputs use a fixed A4/A8 schedule. Deployment trajectory
Hessians and joint GPTQ are collected and solved in the subsequent stages."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from numerics.int8_matmul import Int8MatmulWorkspace
from numerics.linear_kernels import LinearNumericWorkspace
from model.modeling_llada import LLaDAModelLM
from numerics.precision import RowPrecisionContext
from quantization.artifact import R4_VARIANTS, ROTATED_VARIANTS, SpinQuantArtifactWriter
from quantization.gptq import (
    JOINT_INSIDE_LAYER_GROUPS,
    _calibrate_v8_scale,
    _calibration_examples,
    _capture_first_layer_inputs,
    apply_fixed_spinquant,
)
from quantization.model import (
    SpinQuantV8CacheCodec,
    SpinQuantW4A8Linear,
    _install_native_attention_numeric,
    _install_target_numeric_block_ops,
    _record_target_rope_table,
)
from quantization.numeric import (
    W4_SCALE_MODES,
    SpinQuantW4Tensor,
    quantize_symmetric_w4,
)
from quantization.rotation import INSTRUCT_CHECKPOINT, structured_hadamard_12288_bf16
from numerics.bf16 import quantize_activation_per_row_bits_bf16

DEFAULT_A8_PERIOD = 16
DEFAULT_CLIP_MIN = 0.6
DEFAULT_CLIP_MAX = 1.2
DEFAULT_CLIP_STEPS = 25
DEFAULT_OUTPUT_CHUNK = 128


def calibration_row_bits(
    rows: int,
    *,
    device: torch.device,
    start: int = 0,
    a8_period: int = DEFAULT_A8_PERIOD,
) -> torch.Tensor:
    if rows <= 0 or a8_period <= 1:
        raise ValueError("rows must be positive and a8_period must exceed one")
    indices = torch.arange(start, start + rows, device=device)
    return torch.where(indices.remainder(a8_period) == a8_period - 1, 8, 4).to(
        torch.int8
    )


class SymmetricLWC:
    """Symmetric LWC with one BF16 scale per output row."""

    def __init__(
        self, layer: nn.Linear, *, group_size: int, a8_period: int = DEFAULT_A8_PERIOD
    ) -> None:
        if not isinstance(layer, nn.Linear) or layer.bias is not None:
            raise TypeError("OmniQuantLWC requires a bias-free Linear")
        if a8_period <= 1:
            raise ValueError("a8_period must exceed one")
        if group_size != -1:
            raise ValueError("LWC initialization requires G-1 weights")
        self.layer = layer
        self.columns = int(layer.in_features)
        self.group_size = int(group_size)
        self.a8_period = int(a8_period)
        device = layer.weight.device
        self.teacher_square = torch.zeros(
            self.columns, device=device, dtype=torch.float32
        )
        self.teacher_student = torch.zeros_like(self.teacher_square)
        self.student_square = torch.zeros_like(self.teacher_square)
        self.token_rows = 0
        self.a4_rows = 0
        self.a8_rows = 0

    @torch.no_grad()
    def add_batch(
        self, values: torch.Tensor, *, row_bits: torch.Tensor | None = None
    ) -> None:
        rows = values.detach().reshape(-1, self.columns)
        if row_bits is None:
            bits = calibration_row_bits(
                int(rows.shape[0]),
                device=rows.device,
                start=self.token_rows,
                a8_period=self.a8_period,
            )
        else:
            if (
                row_bits.dtype != torch.int8
                or tuple(row_bits.shape) != tuple(values.shape[:-1])
                or row_bits.device != rows.device
                or (not bool(torch.all((row_bits == 4) | (row_bits == 8))))
            ):
                raise ValueError(
                    "row_bits must match the activation leading shape on the same device and contain only torch.int8 values 4 or 8"
                )
            bits = row_bits.detach().reshape(-1)
        self.a4_rows += int((bits == 4).sum().item())
        self.a8_rows += int((bits == 8).sum().item())
        quantized = quantize_activation_per_row_bits_bf16(rows, bits)
        student = (
            (quantized.codes.float() * quantized.scale_bf16.unsqueeze(1))
            .to(torch.bfloat16)
            .float()
        )
        teacher = rows.float()
        self.teacher_square += teacher.square().sum(dim=0)
        self.teacher_student += (teacher * student).sum(dim=0)
        self.student_square += student.square().sum(dim=0)
        self.token_rows += int(rows.shape[0])

    @torch.no_grad()
    def quantize(
        self,
        *,
        clip_min: float = DEFAULT_CLIP_MIN,
        clip_max: float = DEFAULT_CLIP_MAX,
        clip_steps: int = DEFAULT_CLIP_STEPS,
        output_chunk: int = DEFAULT_OUTPUT_CHUNK,
    ) -> tuple[SpinQuantW4Tensor, dict[str, object]]:
        if self.token_rows <= 0:
            raise ValueError("OmniQuant-LWC requires calibration activation rows")
        if not 0 < clip_min <= 1.0 <= clip_max or clip_steps < 2:
            raise ValueError("clip grid must contain 1.0 and at least two candidates")
        if output_chunk <= 0:
            raise ValueError("output_chunk must be positive")
        weight = self.layer.weight.detach().float()
        grouped_weight = weight.reshape(weight.shape[0], 1, -1)
        grouped_teacher_square = self.teacher_square.reshape(1, -1)
        grouped_teacher_student = self.teacher_student.reshape(1, -1)
        grouped_student_square = self.student_square.reshape(1, -1)
        ratios = torch.linspace(
            float(clip_min), float(clip_max), int(clip_steps), device=weight.device
        )
        if not bool(torch.any(torch.isclose(ratios, torch.ones_like(ratios)))):
            ratios = torch.sort(torch.cat((ratios, ratios.new_tensor([1.0])))).values
        codes = torch.empty_like(weight, dtype=torch.int8)
        scales = torch.empty(
            weight.shape[0], 1, device=weight.device, dtype=torch.float32
        )
        selected_ratios = torch.empty_like(scales)
        for start in range(0, int(weight.shape[0]), int(output_chunk)):
            end = min(start + int(output_chunk), int(weight.shape[0]))
            block = weight[start:end]
            grouped_block = grouped_weight[start:end]
            base_scale = (
                (grouped_block.abs().amax(dim=2) * (2.0 / 15.0))
                .to(torch.bfloat16)
                .float()
            )
            base_scale = torch.where(
                base_scale == 0, torch.ones_like(base_scale), base_scale
            )
            teacher_constant = (
                grouped_block.square() * grouped_teacher_square.unsqueeze(0)
            )
            best_loss = torch.full(
                (end - start, 1), 1e309, device=weight.device, dtype=torch.float32
            )
            best_codes = torch.empty_like(block, dtype=torch.int8)
            best_scales = torch.empty_like(base_scale)
            best_ratios = torch.empty_like(best_scales)
            for ratio in ratios:
                candidate_scale = (base_scale * ratio).to(torch.bfloat16).float()
                scale_arg = candidate_scale.squeeze(1)
                (candidate_codes, candidate) = quantize_symmetric_w4(
                    block, scale_arg, group_size=-1
                )
                grouped_candidate = candidate.reshape(end - start, 1, -1)
                loss = (
                    teacher_constant
                    - 2.0
                    * grouped_block
                    * grouped_candidate
                    * grouped_teacher_student.unsqueeze(0)
                    + grouped_candidate.square() * grouped_student_square.unsqueeze(0)
                ).sum(dim=2)
                better = loss < best_loss
                best_loss = torch.where(better, loss, best_loss)
                best_scales = torch.where(better, candidate_scale, best_scales)
                best_ratios = torch.where(
                    better, ratio.expand_as(best_ratios), best_ratios
                )
                best_codes = torch.where(
                    better.unsqueeze(-1)
                    .expand(end - start, 1, grouped_block.shape[-1])
                    .reshape_as(block),
                    candidate_codes,
                    best_codes,
                )
            codes[start:end] = best_codes
            scales[start:end] = best_scales
            selected_ratios[start:end] = best_ratios
        scale_arg = scales.squeeze(1)
        (_, dequantized) = quantize_symmetric_w4(weight, scale_arg, group_size=-1)
        self.layer.weight.data.copy_(dequantized.to(self.layer.weight.dtype))
        statistics = {
            "calibration_rows": self.token_rows,
            "a4_rows": self.a4_rows,
            "a8_rows": self.a8_rows,
            "clip_ratio_min": float(selected_ratios.min().item()),
            "clip_ratio_max": float(selected_ratios.max().item()),
            "clip_ratio_mean": float(selected_ratios.mean().item()),
            "weight_group_size": -1,
            "scale_groups_per_output": 1,
            "output_groups_clipped_below_rtn": int(
                (selected_ratios < 1.0).sum().item()
            ),
            "output_groups_scaled_above_rtn": int((selected_ratios > 1.0).sum().item()),
            "output_groups": int(selected_ratios.numel()),
            "output_channels": int(weight.shape[0]),
        }
        return (
            SpinQuantW4Tensor(
                codes.cpu(), scale_arg.cpu(), scale_mode=W4_SCALE_MODES[-1]
            ),
            statistics,
        )


def _activate_context(
    context: RowPrecisionContext, hidden: torch.Tensor, *, a8_period: int
) -> None:
    rows = int(hidden.numel() // hidden.shape[-1])
    context.activate(
        calibration_row_bits(rows, device=hidden.device, a8_period=a8_period)
    )


def _replace_with_lwc_result(
    block: nn.Module,
    name: str,
    solver: SymmetricLWC,
    writer: SpinQuantArtifactWriter,
    layer_index: int,
    device: torch.device,
    workspace: LinearNumericWorkspace,
    context: RowPrecisionContext,
) -> None:
    module_name = f"model.transformer.blocks.{layer_index}.{name}"
    source_shape = list(solver.layer.weight.shape)
    (quantized, statistics) = solver.quantize()
    writer.write_w4(
        f"{module_name}.weight",
        quantized,
        source_shape=source_shape,
        calibration_rows=solver.token_rows,
        quantization_statistics=statistics,
    )
    replacement = SpinQuantW4A8Linear(
        quantized,
        workspace=workspace,
        row_precision_context=context,
        module_name=module_name,
    ).to(device)
    setattr(block, name, replacement)
    if name == "ff_out" and writer.variant in R4_VARIANTS:

        def rotate_ff_input(
            _module: nn.Module, inputs: tuple[torch.Tensor, ...]
        ) -> tuple[torch.Tensor, ...]:
            if len(inputs) != 1:
                raise ValueError("ff_out expects one activation input")
            return (structured_hadamard_12288_bf16(inputs[0]).to(inputs[0].dtype),)

        block.ff_out.register_forward_pre_hook(rotate_ff_input)


def _install_lwc_group(
    block: nn.Module,
    names: tuple[str, ...],
    layer_inputs: list[torch.Tensor],
    writer: SpinQuantArtifactWriter,
    layer_index: int,
    device: torch.device,
    workspace: LinearNumericWorkspace,
    context: RowPrecisionContext,
    *,
    group_size: int,
    a8_period: int,
) -> None:
    solvers = {
        name: SymmetricLWC(
            getattr(block, name), group_size=group_size, a8_period=a8_period
        )
        for name in names
    }

    def collect_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        *,
        solver: SymmetricLWC,
        module_name: str,
    ) -> None:
        if len(inputs) != 1:
            raise ValueError(f"{module_name} calibration expects one input tensor")
        values = inputs[0]
        rows = int(values.numel() // values.shape[-1])
        bits = context.require(rows, values.device)
        solver.add_batch(values, row_bits=bits.reshape(values.shape[:-1]))

    handles = [
        getattr(block, name).register_forward_pre_hook(
            lambda module, inputs, solver=solver, name=name: collect_input(
                module,
                inputs,
                solver=solver,
                module_name=f"model.transformer.blocks.{layer_index}.{name}",
            )
        )
        for (name, solver) in solvers.items()
    ]
    try:
        for hidden_cpu in layer_inputs:
            hidden = hidden_cpu.to(device)
            _activate_context(context, hidden, a8_period=a8_period)
            block(hidden, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for name, solver in solvers.items():
        _replace_with_lwc_result(
            block, name, solver, writer, layer_index, device, workspace, context
        )


@torch.inference_mode()
def calibrate_initial_layers(
    model: LLaDAModelLM,
    layer_inputs: list[torch.Tensor],
    writer: SpinQuantArtifactWriter,
    device: torch.device,
    *,
    group_size: int,
    a8_period: int,
) -> None:
    linear_workspace = LinearNumericWorkspace()
    attention_workspace = Int8MatmulWorkspace()
    context = RowPrecisionContext()
    v_scales: list[torch.Tensor] = []
    v_statistics: list[dict[str, object]] = []
    for layer_index, block in enumerate(model.model.transformer.blocks):
        print(f"omniquant-lwc: block {layer_index + 1}/32 QKV", flush=True)
        block.to(device)
        _install_lwc_group(
            block,
            JOINT_INSIDE_LAYER_GROUPS[0],
            layer_inputs,
            writer,
            layer_index,
            device,
            linear_workspace,
            context,
            group_size=group_size,
            a8_period=a8_period,
        )
        (scale, statistics) = _calibrate_v8_scale(block, layer_inputs, device)
        v_scales.append(scale)
        v_statistics.append({"layer": layer_index, **statistics})
        _install_native_attention_numeric(
            block,
            workspace=attention_workspace,
            qk_int8=True,
            softmax_lut=True,
            probability_p8=False,
            k8_cache=True,
            v8_codec=SpinQuantV8CacheCodec(scale),
            rope_before_k8=True,
        )
        for names in JOINT_INSIDE_LAYER_GROUPS[1:]:
            print(
                f"omniquant-lwc: block {layer_index + 1}/32 group {','.join(names)}",
                flush=True,
            )
            _install_lwc_group(
                block,
                names,
                layer_inputs,
                writer,
                layer_index,
                device,
                linear_workspace,
                context,
                group_size=group_size,
                a8_period=a8_period,
            )
        next_inputs = []
        for hidden_cpu in layer_inputs:
            hidden = hidden_cpu.to(device)
            _activate_context(context, hidden, a8_period=a8_period)
            (output, _) = block(hidden, use_cache=False)
            next_inputs.append(output.to("cpu", torch.bfloat16))
        layer_inputs = next_inputs
        block.cpu()
        torch.cuda.empty_cache()
    writer.write_v_cache_calibration(torch.stack(v_scales), statistics=v_statistics)


def run_quantization(
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    variant: str,
    device: str,
    group_size: int = -1,
    a8_period: int = DEFAULT_A8_PERIOD,
) -> dict[str, object]:
    if variant not in ROTATED_VARIANTS:
        raise ValueError(f"variant must be one of {ROTATED_VARIANTS}")
    if group_size != -1:
        raise ValueError("LWC initialization requires G-1 weights")
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(target_device)
    model = LLaDAModelLM.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    rotation = apply_fixed_spinquant(
        model, variant=variant, device=target_device, target_numeric_r4=True
    )
    _install_target_numeric_block_ops(model)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    (examples, calibration) = _calibration_examples(tokenizer)
    calibration["kind"] = "wikitext_training_initialization"
    calibration["activation_precision_schedule"] = {
        "kind": "deterministic_state_ratio_proxy",
        "a4_rows_per_period": int(a8_period - 1),
        "a8_rows_per_period": 1,
        "period": int(a8_period),
    }
    writer = SpinQuantArtifactWriter(
        checkpoint_dir,
        output_dir,
        variant=variant,
        calibration=calibration,
        rotation=rotation,
        weight_quantization={
            "tensor_kind": "omniquant_lwc_w4",
            "algorithm": "constrained_omniquant_lwc_diagonal_reconstruction",
            "bits": 4,
            "sym": True,
            "signed_integer_range": [-8, 7],
            "signed_zero_point": 0,
            "qzeros": False,
            "group_size": group_size,
            "scale": "learned_bf16_per_output_channel_2amax_div_15",
            "rounding": "round_to_nearest_even",
            "target_linear_count": 224,
            "lwc": True,
            "let_scale": False,
            "let_shift": False,
            "learned_rounding": False,
            "clip_search": {
                "minimum_ratio": DEFAULT_CLIP_MIN,
                "maximum_ratio": DEFAULT_CLIP_MAX,
                "steps": DEFAULT_CLIP_STEPS,
                "includes_rtn_ratio": True,
            },
            "reconstruction": {
                "scope": "linear_output_diagonal_moment",
                "teacher_input": "rotated_bfloat16",
                "student_input": "fixed_schedule_per_row_a4_a8_dequant_bfloat16",
                "activation_scale": "one_dynamic_scale_per_token_row_no_activation_grouping",
                "sequential_joint_attention": "q8_k8_lut_p8_v8_int32_pv",
                "target_numeric_graph": "bf16_rms_rope_pwl16_swiglu_residual_staged_h12288",
            },
        },
    )
    try:
        writer.write_bf16(
            "model.transformer.wte.weight", model.model.transformer.wte.weight
        )
        writer.write_bf16(
            "model.transformer.ff_out.weight", model.model.transformer.ff_out.weight
        )
        layer_inputs = _capture_first_layer_inputs(model, examples, target_device)
        calibrate_initial_layers(
            model,
            layer_inputs,
            writer,
            target_device,
            group_size=group_size,
            a8_period=a8_period,
        )
        _record_target_rope_table(model)
        return writer.finish()
    except Exception:
        writer.abort()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=INSTRUCT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=ROTATED_VARIANTS, default="fixed-r1r2-r4-only"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-size", type=int, choices=(-1,), default=-1)
    parser.add_argument("--a8-period", type=int, default=DEFAULT_A8_PERIOD)
    args = parser.parse_args()
    manifest = run_quantization(
        args.checkpoint_dir,
        args.output_dir,
        variant=args.variant,
        device=args.device,
        group_size=args.group_size,
        a8_period=args.a8_period,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "schema_version": manifest["schema_version"],
                "total_file_bytes": manifest["total_file_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
