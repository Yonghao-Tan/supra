"""Fixed SpinQuant rotation and no-group GPTQ for supported LLaDA-8B checkpoints."""

from __future__ import annotations
import hashlib
import json
from typing import Iterable
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from quantization.artifact import R4_VARIANTS, ROTATED_VARIANTS
from quantization.numeric import (
    SpinQuantW4Tensor,
    quantize_symmetric_w4,
    symmetric_w4_scale_bf16,
)
from quantization.rotation import (
    ALGO_ROOT,
    fixed_rotation_signs,
    rotate_ff_out_weight_fixed,
    rotate_input_weight_fixed,
    rotate_o_weight_fixed,
    rotate_v_weight_fixed,
    structured_hadamard_12288,
    structured_hadamard_12288_bf16,
)
from numerics.bf16 import (
    quantize_activation_per_row_bf16,
    quantize_activation_per_row_bits_bf16,
)

CALIBRATION_SEED = 20260806
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CALIBRATION_SAMPLES = 128
CALIBRATION_SEQUENCE_LENGTH = 2048
CALIBRATION_BATCH_SIZE = 4
GPTQ_BLOCK_SIZE = 128
GPTQ_PERCDAMP = 0.01
JOINT_INSIDE_LAYER_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("attn_out",),
    ("up_proj", "ff_proj"),
    ("ff_out",),
)


class NoGroupGPTQ:
    """GPTQ with one static BF16 scale per output row."""

    def __init__(self, layer: nn.Linear, *, activation_bits: int | None = None) -> None:
        if not isinstance(layer, nn.Linear) or layer.bias is not None:
            raise TypeError("NoGroupGPTQ requires a bias-free Linear")
        self.layer = layer
        self.columns = int(layer.in_features)
        self.hessian = torch.zeros(
            (self.columns, self.columns),
            device=layer.weight.device,
            dtype=torch.float32,
        )
        self.token_rows = 0
        if activation_bits not in {None, 4, 8}:
            raise ValueError("GPTQ Hessian activation bits must be None, 4, or 8")
        self.activation_bits = activation_bits

    @torch.no_grad()
    def add_batch(
        self, values: torch.Tensor, *, row_bits: torch.Tensor | None = None
    ) -> None:
        """Accumulate an unweighted Hessian from BF16 or deployment A4/A8 rows."""
        rows = values.detach().reshape(-1, self.columns)
        if row_bits is not None:
            if self.activation_bits is not None:
                raise ValueError(
                    "event row_bits cannot be combined with fixed Hessian activation_bits"
                )
            if (
                row_bits.dtype != torch.int8
                or tuple(row_bits.shape) != tuple(values.shape[:-1])
                or row_bits.device != rows.device
                or (not bool(torch.all((row_bits == 4) | (row_bits == 8))))
            ):
                raise ValueError(
                    "row_bits must match the activation leading shape on the same device and contain only torch.int8 values 4 or 8"
                )
            quantized = quantize_activation_per_row_bits_bf16(
                rows, row_bits.detach().reshape(-1)
            )
            rows = (quantized.codes.float() * quantized.scale_bf16.unsqueeze(1)).to(
                torch.bfloat16
            )
        elif self.activation_bits is not None:
            quantized = quantize_activation_per_row_bf16(rows, self.activation_bits)
            rows = (quantized.codes.float() * quantized.scale_bf16.unsqueeze(1)).to(
                torch.bfloat16
            )
        rows = rows.to(torch.float32)
        self.hessian.addmm_(rows.T, rows)
        self.token_rows += int(rows.shape[0])

    @torch.no_grad()
    def quantize(
        self,
        *,
        percdamp: float = GPTQ_PERCDAMP,
        block_size: int = GPTQ_BLOCK_SIZE,
        group_size: int = -1,
        act_order: bool = False,
        fixed_scale_bf16: torch.Tensor | None = None,
    ) -> SpinQuantW4Tensor:
        if self.token_rows <= 0:
            raise ValueError("GPTQ requires at least one calibration activation row")
        if group_size != -1:
            raise ValueError("GPTQ group size must be -1")
        weight = self.layer.weight.detach().to(torch.float32).clone()
        if fixed_scale_bf16 is None:
            scale = symmetric_w4_scale_bf16(weight, group_size=group_size)
        else:
            scale = fixed_scale_bf16.detach().to(
                device=weight.device, dtype=torch.float32
            )
            if (
                group_size != -1
                or scale.shape != (weight.shape[0],)
                or (not bool(torch.all(torch.isfinite(scale) & (scale > 0))))
                or (not torch.equal(scale, scale.to(torch.bfloat16).float()))
            ):
                raise ValueError(
                    "fixed GPTQ scales require one positive finite BF16 value per G-1 output"
                )
            scale = scale.clone()
        hessian = self.hessian
        self.hessian = torch.empty(0, device=weight.device)
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0
        inverse_permutation = None
        if act_order:
            permutation = torch.argsort(
                hessian.diagonal(), descending=True, stable=True
            )
            inverse_permutation = torch.argsort(permutation)
            weight = weight[:, permutation]
            hessian = hessian[permutation][:, permutation]
        diagonal = torch.arange(self.columns, device=weight.device)
        hessian[diagonal, diagonal] += float(percdamp) * torch.mean(torch.diag(hessian))
        inverse_cholesky = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True
        )
        codes = torch.empty_like(weight, dtype=torch.int8)
        dequantized = torch.empty_like(weight)
        for block_start in range(0, self.columns, int(block_size)):
            block_end = min(block_start + int(block_size), self.columns)
            block_weight = weight[:, block_start:block_end].clone()
            block_errors = torch.zeros_like(block_weight)
            block_inverse = inverse_cholesky[
                block_start:block_end, block_start:block_end
            ]
            for local_column in range(block_end - block_start):
                column = block_weight[:, local_column : local_column + 1]
                column_scale = scale
                (column_codes, column_dequantized) = quantize_symmetric_w4(
                    column, column_scale
                )
                codes[:, block_start + local_column] = column_codes[:, 0]
                dequantized[:, block_start + local_column] = column_dequantized[:, 0]
                divisor = block_inverse[local_column, local_column]
                error = (column[:, 0] - column_dequantized[:, 0]) / divisor
                block_weight[:, local_column:] -= error.unsqueeze(1) @ block_inverse[
                    local_column, local_column:
                ].unsqueeze(0)
                block_errors[:, local_column] = error
            weight[:, block_end:] -= (
                block_errors @ inverse_cholesky[block_start:block_end, block_end:]
            )
        if inverse_permutation is not None:
            codes = codes[:, inverse_permutation].contiguous()
            dequantized = dequantized[:, inverse_permutation]
        self.layer.weight.data.copy_(dequantized.to(self.layer.weight.dtype))
        return SpinQuantW4Tensor(
            codes.cpu(),
            scale.cpu(),
            scale_mode="per_output_channel",
        )


def apply_fixed_spinquant(
    model: LLaDAModelLM,
    *,
    variant: str,
    device: torch.device | str = "cpu",
    offload_after_transform: bool = True,
    target_numeric_r4: bool = False,
    rotation_seed: int = CALIBRATION_SEED,
) -> dict[str, torch.Tensor]:
    if variant not in ROTATED_VARIANTS:
        raise ValueError(
            f"fixed SpinQuant requires one of {ROTATED_VARIANTS}, got {variant}"
        )
    (r1_sign, r2_signs) = fixed_rotation_signs(int(rotation_seed))
    target_device = torch.device(device)
    r1_device = r1_sign.to(target_device)
    transformer = model.model.transformer
    with torch.no_grad():
        print("rotation: embedding", flush=True)
        transformer.wte.to(target_device)
        embedding = transformer.wte.weight.data
        transformer.wte.weight.data = rotate_input_weight_fixed(
            embedding, r1_device
        ).to(embedding.dtype)
        if offload_after_transform:
            transformer.wte.cpu()
        print("rotation: lm_head", flush=True)
        transformer.ff_out.to(target_device)
        transformer.ln_f.to(target_device)
        final_norm = transformer.ln_f.weight.data
        head = transformer.ff_out.weight.data * final_norm.to(
            transformer.ff_out.weight.device
        ).unsqueeze(0)
        transformer.ff_out.weight.data = rotate_input_weight_fixed(head, r1_device).to(
            head.dtype
        )
        transformer.ln_f.weight.fill_(1)
        if offload_after_transform:
            transformer.ff_out.cpu()
            transformer.ln_f.cpu()
        for layer, block in enumerate(transformer.blocks):
            print(f"rotation: block {layer + 1}/32", flush=True)
            block.to(target_device)
            r2_device = r2_signs[layer].to(target_device)
            attention_norm = block.attn_norm.weight.data
            feedforward_norm = block.ff_norm.weight.data
            for name in ("q_proj", "k_proj", "v_proj"):
                module = getattr(block, name)
                fused = module.weight.data * attention_norm.to(
                    module.weight.device
                ).unsqueeze(0)
                module.weight.data = (
                    rotate_v_weight_fixed(fused, r1_device, r2_device, 32)
                    if name == "v_proj"
                    else rotate_input_weight_fixed(fused, r1_device)
                ).to(module.weight.dtype)
            for name in ("ff_proj", "up_proj"):
                module = getattr(block, name)
                fused = module.weight.data * feedforward_norm.to(
                    module.weight.device
                ).unsqueeze(0)
                module.weight.data = rotate_input_weight_fixed(fused, r1_device).to(
                    module.weight.dtype
                )
            block.attn_out.weight.data = rotate_o_weight_fixed(
                block.attn_out.weight.data, r1_device, r2_device, 32
            ).to(block.attn_out.weight.dtype)
            r4_enabled = variant in R4_VARIANTS
            block.ff_out.weight.data = rotate_ff_out_weight_fixed(
                block.ff_out.weight.data, r1_device, had=r4_enabled
            ).to(block.ff_out.weight.dtype)
            block.attn_norm.weight.fill_(1)
            block.ff_norm.weight.fill_(1)
            if r4_enabled:

                def rotate_ff_input(
                    _module: nn.Module, inputs: tuple[torch.Tensor, ...]
                ) -> tuple[torch.Tensor, ...]:
                    rotated = (
                        structured_hadamard_12288_bf16(inputs[0])
                        if target_numeric_r4
                        else structured_hadamard_12288(inputs[0]).to(inputs[0].dtype)
                    )
                    return (rotated,)

                block.ff_out.register_forward_pre_hook(rotate_ff_input)
                if target_numeric_r4:
                    block._target_numeric_r4 = "h12288-bf16-staged/v1"
            if offload_after_transform:
                block.cpu()
            if offload_after_transform and target_device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "r1_sign": r1_sign,
        **{f"r2_sign.{index}": sign for (index, sign) in enumerate(r2_signs)},
    }


def _calibration_examples(
    tokenizer: AutoTokenizer,
) -> tuple[list[torch.Tensor], dict[str, object]]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        revision=WIKITEXT_REVISION,
        cache_dir=str(ALGO_ROOT / "cache/datasets"),
    )
    joined = "\n\n".join((str(text) for text in dataset["text"]))
    tokens = tokenizer(joined, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    generator = torch.Generator(device="cpu").manual_seed(CALIBRATION_SEED)
    maximum_start = int(tokens.numel()) - CALIBRATION_SEQUENCE_LENGTH
    starts = torch.randint(
        0, maximum_start + 1, (CALIBRATION_SAMPLES,), generator=generator
    )
    examples = [
        tokens[start : start + CALIBRATION_SEQUENCE_LENGTH].clone()
        for start in starts.tolist()
    ]
    index_payload = json.dumps(starts.tolist(), separators=(",", ":")).encode("ascii")
    return (
        examples,
        {
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "dataset_revision": WIKITEXT_REVISION,
            "dataset_fingerprint": dataset._fingerprint,
            "samples": CALIBRATION_SAMPLES,
            "sequence_length": CALIBRATION_SEQUENCE_LENGTH,
            "batch_size": CALIBRATION_BATCH_SIZE,
            "seed": CALIBRATION_SEED,
            "sample_starts": starts.tolist(),
            "sample_starts_sha256": hashlib.sha256(index_payload).hexdigest(),
            "token_count": int(tokens.numel()),
        },
    )


class _CapturedInput(RuntimeError):
    pass


@torch.inference_mode()
def _capture_first_layer_inputs(
    model: LLaDAModelLM, examples: Iterable[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    transformer = model.model.transformer
    first_block = transformer.blocks[0]
    transformer.wte.to(device)
    transformer.emb_drop.to(device)
    first_block.to(device)
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().to("cpu", torch.bfloat16))
        raise _CapturedInput

    handle = first_block.register_forward_pre_hook(capture)
    try:
        examples = list(examples)
        for start in range(0, len(examples), CALIBRATION_BATCH_SIZE):
            input_ids = torch.stack(examples[start : start + CALIBRATION_BATCH_SIZE])
            try:
                model(input_ids=input_ids.to(device), use_cache=False)
            except _CapturedInput:
                pass
    finally:
        handle.remove()
        transformer.wte.cpu()
        transformer.emb_drop.cpu()
        first_block.cpu()
    return captured


@torch.inference_mode()
def _calibrate_v8_scale(
    block: nn.Module, layer_inputs: list[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, dict[str, object]]:
    kv_heads = int(block.config.effective_n_kv_heads)
    head_dim = int(block.config.d_model // block.config.n_heads)
    maximum = torch.zeros(kv_heads, dtype=torch.float32, device=device)
    rows = 0

    def collect(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        nonlocal rows
        values = output.detach().reshape(-1, output.shape[-2], kv_heads, head_dim)
        maximum.copy_(torch.maximum(maximum, values.float().abs().amax(dim=(0, 1, 3))))
        rows += int(values.shape[0] * values.shape[1])

    handle = block.v_proj.register_forward_hook(collect)
    try:
        for hidden in layer_inputs:
            block(hidden.to(device), use_cache=False)
    finally:
        handle.remove()
    if rows <= 0 or bool(torch.any(maximum <= 0)):
        raise ValueError(
            "V8 calibration requires nonzero observations for every KV head"
        )
    scale = (maximum / 127.0).to(torch.bfloat16).float()
    return (
        scale.cpu(),
        {
            "token_rows": rows,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "max_abs_min": float(maximum.min().item()),
            "max_abs_max": float(maximum.max().item()),
        },
    )
