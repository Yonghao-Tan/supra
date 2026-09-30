"""Persistent fixed-SpinQuant W4 artifacts for supported LLaDA-8B checkpoints."""

from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, BinaryIO, Mapping
import uuid
import numpy as np
import torch
from safetensors.torch import save_file
from quantization.checkpoint import _load_index, _shard_hashes
from quantization.numeric import (
    W4_SCALE_MODES,
    SpinQuantW4Tensor,
    pack_signed_int4,
    unpack_signed_int4,
)
from quantization.rotation import (
    ROTATION_SEED,
    checkpoint_identity,
    require_algo_output,
    sha256_file,
    supported_checkpoint_identities,
)

JOINT_W4_SCHEMA_VERSION = "llada-joint-spinquant-w4a4-k8-v8/v4"
LM_HEAD_W8_SCHEMA_VERSION = "llada-joint-spinquant-w4a4-k8-v8-lm-head-w8a8/v6"
MANIFEST_FILE = "tensor_manifest.json"
W4_CODES_FILE = "payload/block_weights.int4"
W4_SCALES_FILE = "payload/block_weight_scales.bf16"
BF16_FILE = "payload/embedding_lm_head.bf16"
EMBEDDING_FILE = "payload/embedding.bf16"
LM_HEAD_W8_CODES_FILE = "payload/lm_head.int8"
LM_HEAD_W8_SCALES_FILE = "payload/lm_head_w8_scales.bf16"
ROTATION_FILE = "payload/fixed_rotation_signs.safetensors"
V_CACHE_SCALES_FILE = "payload/v_cache_scales.bf16"


def require_train_only_artifact(reader):
    """Reject test-fitted or unidentified initialization in train-only PTQ."""
    if reader.head_quantized:
        if reader._parent_reader is None:
            raise ValueError("train-only head has no parent")
        require_train_only_artifact(reader._parent_reader)
        quantization = reader.manifest["head_quantization"]
        if (
            quantization.get("algorithm") != "symmetric_per_output_row_w8"
            or quantization.get("artifact_selection_metric")
            != "single_predeclared_w8_format_no_selection"
        ):
            raise ValueError(
                "train-only PTQ requires the fixed sample-free W8 head"
            )
        return
    calibration = reader.manifest.get("calibration", {})
    quantization = reader.manifest.get("weight_quantization", {})
    if calibration.get("new_w4_test_sets_used_for_weight_or_scale_selection"):
        raise ValueError("calibration weights were fitted using test data")
    if calibration.get("train_only_calibration") is True:
        return
    if (
        calibration.get("dataset") == "Salesforce/wikitext:wikitext-2-raw-v1:train"
        and quantization.get("algorithm")
        == "constrained_omniquant_lwc_diagonal_reconstruction"
    ):
        return
    raise ValueError("artifact has no supported train-only calibration provenance")


VARIANTS = ("fixed-r1r2-r4-only",)
ROTATED_VARIANTS = VARIANTS
R4_VARIANTS = VARIANTS
REQUIRED_FILES = {W4_CODES_FILE, W4_SCALES_FILE, BF16_FILE, ROTATION_FILE}
EMBEDDING = "model.transformer.wte.weight"
LM_HEAD = "model.transformer.ff_out.weight"
BF16_IDS = {EMBEDDING, LM_HEAD}
LM_HEAD_W8_CHILD_FILES = {EMBEDDING_FILE, LM_HEAD_W8_CODES_FILE, LM_HEAD_W8_SCALES_FILE}
BLOCK_LINEAR = re.compile(
    "^model\\.transformer\\.blocks\\.(\\d+)\\.(attn_out|ff_out|ff_proj|k_proj|q_proj|up_proj|v_proj)\\.weight$"
)
GMINUS1_BF16_ARITHMETIC = "gminus1_bf16_dual_scale/v1"


def expected_w4_ids() -> set[str]:
    modules = ("attn_out", "ff_out", "ff_proj", "k_proj", "q_proj", "up_proj", "v_proj")
    return {
        f"model.transformer.blocks.{layer}.{module}.weight"
        for layer in range(32)
        for module in modules
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bf16_from_le_payload(payload: bytes, *, name: str) -> torch.Tensor:
    """Load little-endian BF16 bits through NumPy's Torch-compatible INT16 dtype."""
    if len(payload) % 2:
        raise ValueError(f"{name} BF16 payload byte length must be even")
    raw_bits = np.frombuffer(payload, dtype="<i2").copy()
    return torch.from_numpy(raw_bits).view(torch.bfloat16)


def _bf16_payload(values: torch.Tensor) -> bytes:
    return (
        values.detach()
        .cpu()
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .numpy()
        .astype("<i2", copy=False)
        .tobytes()
    )


class SpinQuantArtifactWriter:
    def __init__(
        self,
        checkpoint_dir: Path,
        output_dir: Path,
        *,
        variant: str,
        calibration: Mapping[str, object],
        rotation: Mapping[str, torch.Tensor],
        weight_quantization: Mapping[str, object] | None = None,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unsupported SpinQuant variant {variant}")
        self.checkpoint = checkpoint_identity(checkpoint_dir)
        self.output_dir = require_algo_output(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite artifact directory: {self.output_dir}"
            )
        self.variant = variant
        self.weight_quantization = (
            None if weight_quantization is None else dict(weight_quantization)
        )
        self.w4_kind = (
            "gptq_w4"
            if self.weight_quantization is None
            else str(self.weight_quantization.pop("tensor_kind"))
        )
        if self.w4_kind not in {"gptq_w4", "omniquant_lwc_w4", "rtn_w4"}:
            raise ValueError(f"unsupported W4 tensor kind {self.w4_kind}")
        self.w4_group_size = int(
            -1
            if self.weight_quantization is None
            else self.weight_quantization.get("group_size", -1)
        )
        if self.w4_group_size not in W4_SCALE_MODES:
            raise ValueError(f"unsupported W4 group size {self.w4_group_size}")
        if (
            self.w4_group_size == -1
            and self.weight_quantization is not None
            and (
                self.weight_quantization.get("arithmetic_id")
                not in {None, GMINUS1_BF16_ARITHMETIC}
            )
        ):
            raise ValueError("G-1 W4 metadata declares incompatible arithmetic")
        self.schema_version = JOINT_W4_SCHEMA_VERSION
        self.calibration = dict(calibration)
        self.weight_map = _load_index(checkpoint_dir)
        self.source_shards = _shard_hashes(checkpoint_dir, self.weight_map)
        self.temp_dir = self.output_dir.with_name(
            f".{self.output_dir.name}.incomplete-{uuid.uuid4().hex}"
        )
        (self.temp_dir / "payload").mkdir(parents=True, exist_ok=False)
        if not rotation:
            raise ValueError(f"{variant} artifact requires rotation tensors")
        rotation_payload = {
            name: tensor.cpu().contiguous() for (name, tensor) in rotation.items()
        }
        self.rotation_identity = {
            "seed": ROTATION_SEED,
            "sampling": "one_cpu_torch_generator_r1_then_r2_layers_0_to_31",
            "r1_sign_shape": [4096],
            "r2_sign_count": 32,
            "r2_sign_shape": [128],
        }
        save_file(rotation_payload, str(self.temp_dir / ROTATION_FILE))
        self.rotation_identity["payload_sha256"] = sha256_file(
            self.temp_dir / ROTATION_FILE
        )
        self.code_file: BinaryIO = (self.temp_dir / W4_CODES_FILE).open("wb")
        self.scale_file: BinaryIO = (self.temp_dir / W4_SCALES_FILE).open("wb")
        self.bf16_file: BinaryIO = (self.temp_dir / BF16_FILE).open("wb")
        self.offsets = {"code": 0, "scale": 0, "bf16": 0}
        self.entries: dict[str, dict[str, Any]] = {}
        self.v_cache_calibration: dict[str, Any] | None = None
        self.closed = False

    def write_w4(
        self,
        logical_id: str,
        quantized: SpinQuantW4Tensor,
        *,
        source_shape: list[int],
        calibration_rows: int,
        quantization_statistics: Mapping[str, object] | None = None,
    ) -> None:
        if logical_id in self.entries or logical_id not in expected_w4_ids():
            raise ValueError(f"unexpected or duplicate W4 tensor {logical_id}")
        if list(quantized.codes.shape) != list(source_shape):
            raise ValueError(f"W4 code shape mismatch for {logical_id}")
        (rows, columns) = quantized.codes.shape
        expected_scale_shape = (rows,)
        if tuple(quantized.scale_bf16.shape) != expected_scale_shape:
            raise ValueError(f"W4 scale shape mismatch for {logical_id}")
        if quantized.scale_mode != W4_SCALE_MODES[self.w4_group_size]:
            raise ValueError(f"W4 scale mode mismatch for {logical_id}")
        if int(quantized.codes.min()) < -8 or int(quantized.codes.max()) > 7:
            raise ValueError(f"W4 code outside [-8,7] for {logical_id}")
        code_payload = pack_signed_int4(quantized.codes).numpy().tobytes()
        scale_payload = _bf16_payload(quantized.scale_bf16)
        self.code_file.write(code_payload)
        self.scale_file.write(scale_payload)
        self.entries[logical_id] = {
            "logical_id": logical_id,
            "kind": self.w4_kind,
            "shape": source_shape,
            "code_dtype": "signed_int4_twos_complement_low_nibble_first",
            "scale_dtype": "bfloat16_le",
            "scale_mode": W4_SCALE_MODES[self.w4_group_size],
            "group_size": self.w4_group_size,
            "weight_offset_bytes": self.offsets["code"],
            "weight_length_bytes": len(code_payload),
            "scale_offset_bytes": self.offsets["scale"],
            "scale_length_bytes": len(scale_payload),
            "codes_sha256": _sha256(code_payload),
            "scales_sha256": _sha256(scale_payload),
            "code_min": int(quantized.codes.min()),
            "code_max": int(quantized.codes.max()),
            "calibration_rows": int(calibration_rows),
            **(
                {"quantization_statistics": dict(quantization_statistics)}
                if quantization_statistics is not None
                else {}
            ),
        }
        self.offsets["code"] += len(code_payload)
        self.offsets["scale"] += len(scale_payload)

    def write_bf16(self, logical_id: str, values: torch.Tensor) -> None:
        if logical_id in self.entries or logical_id not in BF16_IDS:
            raise ValueError(f"unexpected or duplicate BF16 tensor {logical_id}")
        payload = _bf16_payload(values)
        self.bf16_file.write(payload)
        self.entries[logical_id] = {
            "logical_id": logical_id,
            "kind": "rotated_bf16",
            "shape": list(values.shape),
            "dtype": "bfloat16_le",
            "offset_bytes": self.offsets["bf16"],
            "length_bytes": len(payload),
            "sha256": _sha256(payload),
        }
        self.offsets["bf16"] += len(payload)

    def write_v_cache_calibration(
        self, scales_bf16: torch.Tensor, *, statistics: list[dict[str, Any]]
    ) -> None:
        if self.v_cache_calibration is not None:
            raise ValueError("V8 calibration was already written")
        expected_layers = len(
            {logical_id.split(".")[3] for logical_id in expected_w4_ids()}
        )
        if scales_bf16.ndim != 2 or scales_bf16.shape[0] != expected_layers:
            raise ValueError(f"V8 scales must be [{expected_layers}, num_kv_heads]")
        if len(statistics) != expected_layers:
            raise ValueError("V8 statistics must contain one record per layer")
        scales = scales_bf16.detach().cpu().to(torch.bfloat16).contiguous()
        if not bool(torch.all(torch.isfinite(scales.float()))) or bool(
            torch.any(scales <= 0)
        ):
            raise ValueError("V8 scales must be finite and positive")
        payload = _bf16_payload(scales)
        path = self.temp_dir / V_CACHE_SCALES_FILE
        path.write_bytes(payload)
        self.v_cache_calibration = {
            "shape": list(scales.shape),
            "scale_dtype": "bfloat16_le",
            "scale_mode": "static_per_layer_per_kv_head",
            "scale_rule": "bf16_per_head_max_abs_div_127",
            "code_dtype": "signed_int8",
            "signed_integer_range": [-127, 127],
            "signed_zero_point": 0,
            "qzeros": False,
            "rounding": "round_to_nearest_even",
            "payload": V_CACHE_SCALES_FILE,
            "payload_sha256": _sha256(payload),
            "statistics": statistics,
        }

    def _close_payloads(self) -> None:
        if not self.closed:
            self.code_file.close()
            self.scale_file.close()
            self.bf16_file.close()
            self.closed = True

    def finish(self) -> dict[str, Any]:
        self._close_payloads()
        expected = expected_w4_ids() | BF16_IDS
        if set(self.entries) != expected:
            missing = sorted(expected - set(self.entries))
            extra = sorted(set(self.entries) - expected)
            raise ValueError(
                f"incomplete SpinQuant artifact, missing={missing}, extra={extra}"
            )
        if self.v_cache_calibration is None:
            raise ValueError("joint V8 artifact is missing V cache calibration")
        required_files = REQUIRED_FILES | {V_CACHE_SCALES_FILE}
        files = {
            name: {
                "bytes": (self.temp_dir / name).stat().st_size,
                "sha256": sha256_file(self.temp_dir / name),
            }
            for name in sorted(required_files)
        }
        gptq_metadata = {
            "algorithm": "hessian_error_feedback_sequential",
            "bits": 4,
            "sym": True,
            "signed_integer_range": [-8, 7],
            "signed_zero_point": 0,
            "qzeros": False,
            "group_size": -1,
            "desc_act": False,
            "percdamp": 0.01,
            "solver_block_size": 128,
            "scale": "bf16_per_output_channel_2amax_div_15",
            "rounding": "round_to_nearest_even",
            "target_linear_count": 224,
            "hessian_input": {
                "default": "dynamic_per_row_a8_dequant_bf16",
                "ff_out": "r4_then_dynamic_per_row_a4_dequant_bf16",
            },
            "inside_layer_order": [
                "q_proj+k_proj+v_proj",
                "v8_per_head_calibration",
                "q8k8_lut_p8_v8_attention",
                "attn_out",
                "up_proj+ff_proj",
                "ff_out",
            ],
        }
        schema_version = self.schema_version
        manifest: dict[str, Any] = {
            "schema_version": schema_version,
            "status": "complete",
            "checkpoint": self.checkpoint,
            "source_shards": self.source_shards,
            "variant": self.variant,
            "calibration": self.calibration,
            "calibration_forward": {
                "model": self.checkpoint["model_identity"][: -len("-original-bf16")]
                if self.checkpoint["model_identity"].endswith("-original-bf16")
                else self.checkpoint["model_identity"],
                "attention": "native_bidirectional_is_causal_false",
                "labels": False,
                "loss": False,
            },
            **(
                {"weight_quantization": self.weight_quantization}
                if self.weight_quantization is not None
                else {"gptq": gptq_metadata}
            ),
            "rotation": {
                **self.rotation_identity,
                "r1": "fixed_random_hadamard_absorbed",
                "r2": "fixed_random_hadamard_per_layer_absorbed",
                "r3": "disabled",
                "r4": "normalized_structured_h12288_online",
            },
            "attention": {
                "q": "dynamic_per_head_token_int8",
                "k_cache": "dynamic_per_head_token_int8_scale_shape_b_h_t_1",
                "probability": "dynamic_per_query_row_int8_for_int32_pv",
                "v_cache_dtype": "int8",
                "v_quantization": self.v_cache_calibration,
            },
            "files": files,
            "total_file_bytes": sum((record["bytes"] for record in files.values())),
            "tensors": [self.entries[name] for name in sorted(self.entries)],
        }
        (self.temp_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(self.temp_dir, self.output_dir)
        return manifest

    def abort(self) -> None:
        self._close_payloads()
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class SpinQuantLMHeadArtifactWriter:
    """Write the W8A8 head overlay for one joint W4 artifact."""

    def __init__(
        self,
        parent_artifact_dir: Path,
        output_dir: Path,
        *,
        calibration: Mapping[str, object],
        quantization: Mapping[str, object],
        verify_parent: bool = True,
    ) -> None:
        self.parent = SpinQuantArtifactReader(
            parent_artifact_dir, verify_files=verify_parent
        )
        if not self.parent.joint_v8 or self.parent.variant not in R4_VARIANTS:
            raise ValueError(
                "quantized lm-head child requires a joint R4 SpinQuant parent"
            )
        if self.parent.head_quantized:
            raise ValueError(
                "quantized lm-head child parent must retain the source BF16 head"
            )
        self.output_dir = require_algo_output(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite artifact directory: {self.output_dir}"
            )
        self.temp_dir = self.output_dir.with_name(
            f".{self.output_dir.name}.incomplete-{uuid.uuid4().hex}"
        )
        (self.temp_dir / "payload").mkdir(parents=True, exist_ok=False)
        self.calibration = dict(calibration)
        self.quantization = dict(quantization)
        self.written = False

    def write(
        self,
        quantized_head: Any,
        *,
        quantization_statistics: Mapping[str, object],
    ) -> dict[str, Any]:
        if self.written:
            raise ValueError("quantized lm-head child payload was already written")
        weight_bits = int(self.quantization.get("bits", 0))
        if weight_bits != 8:
            raise ValueError("lm-head child weight bits must be 8")
        source_head = self.parent.read_bf16(LM_HEAD)
        embedding = self.parent.read_bf16(EMBEDDING)
        if list(source_head.shape) != [126464, 4096]:
            raise ValueError(f"unexpected lm-head shape {list(source_head.shape)}")
        if quantized_head.codes.shape != source_head.shape:
            raise ValueError("quantized lm-head code shape mismatch")
        group_size = int(self.quantization.get("group_size", -1))
        if group_size != -1:
            raise ValueError("lm-head child group size must be -1")
        if quantized_head.scale_bf16.shape != source_head.shape[:1]:
            raise ValueError("quantized lm-head scale shape mismatch")
        (code_min, code_max) = (-127, 127)
        if (
            int(quantized_head.codes.min()) < code_min
            or int(quantized_head.codes.max()) > code_max
        ):
            raise ValueError(
                f"W{weight_bits} lm-head code outside [{code_min},{code_max}]"
            )
        embedding_payload = _bf16_payload(embedding)
        code_payload = quantized_head.codes.detach().cpu().contiguous().numpy().tobytes()
        code_file = LM_HEAD_W8_CODES_FILE
        scale_file = LM_HEAD_W8_SCALES_FILE
        tensor_kind = "symmetric_w8"
        code_dtype = "signed_int8_twos_complement"
        schema_version = LM_HEAD_W8_SCHEMA_VERSION
        scale_payload = _bf16_payload(quantized_head.scale_bf16)
        payloads = {
            EMBEDDING_FILE: embedding_payload,
            code_file: code_payload,
            scale_file: scale_payload,
        }
        for name, payload in payloads.items():
            with (self.temp_dir / name).open("wb") as handle:
                handle.write(payload)
        parent_manifest = self.parent.manifest_path
        parent_files = self.parent.manifest.get("files", {})
        files = {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for (name, payload) in payloads.items()
        }
        embedding_entry = {
            "logical_id": EMBEDDING,
            "kind": "rotated_bf16",
            "shape": list(embedding.shape),
            "dtype": "bfloat16_le",
            "payload": EMBEDDING_FILE,
            "offset_bytes": 0,
            "length_bytes": len(embedding_payload),
            "sha256": _sha256(embedding_payload),
            "parent_slice_sha256": self.parent.entries[EMBEDDING]["sha256"],
        }
        head_entry = {
            "logical_id": LM_HEAD,
            "kind": tensor_kind,
            "shape": list(source_head.shape),
            "code_dtype": code_dtype,
            "scale_dtype": "bfloat16_le",
            "scale_mode": "per_output_channel",
            "group_size": group_size,
            "code_payload": code_file,
            "scale_payload": scale_file,
            "weight_offset_bytes": 0,
            "weight_length_bytes": len(code_payload),
            "scale_offset_bytes": 0,
            "scale_length_bytes": len(scale_payload),
            "codes_sha256": _sha256(code_payload),
            "scales_sha256": _sha256(scale_payload),
            "code_min": int(quantized_head.codes.min()),
            "code_max": int(quantized_head.codes.max()),
            "source_bf16_sha256": self.parent.entries[LM_HEAD]["sha256"],
            "quantization_statistics": dict(quantization_statistics),
        }
        manifest: dict[str, Any] = {
            "schema_version": schema_version,
            "status": "complete",
            "checkpoint": self.parent.checkpoint,
            "variant": self.parent.variant,
            "parent_artifact": {
                "path": str(self.parent.artifact_dir),
                "manifest_bytes": parent_manifest.stat().st_size,
                "manifest_sha256": self.parent.manifest_sha256,
                "schema_version": self.parent.manifest["schema_version"],
                "files": parent_files,
            },
            "calibration": self.calibration,
            "head_quantization": self.quantization,
            "files": files,
            "total_file_bytes": sum((record["bytes"] for record in files.values())),
            "tensors": [embedding_entry, head_entry],
        }
        (self.temp_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(self.temp_dir, self.output_dir)
        self.written = True
        return manifest

    def abort(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class SpinQuantArtifactReader:
    def __init__(
        self,
        artifact_dir: Path,
        *,
        verify_files: bool = True,
        expected_checkpoint: Mapping[str, str] | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.manifest_path = self.artifact_dir / MANIFEST_FILE
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        schema_version = self.manifest.get("schema_version")
        self._parent_reader: SpinQuantArtifactReader | None = None
        self.head_weight_bits = 8 if schema_version == LM_HEAD_W8_SCHEMA_VERSION else None
        self.head_group_size = (
            int(self.manifest.get("head_quantization", {}).get("group_size", -1))
            if self.head_weight_bits is not None
            else None
        )
        self.head_quantized = self.head_weight_bits is not None
        if self.head_quantized:
            self._initialize_lm_head_child(
                verify_files=verify_files, expected_checkpoint=expected_checkpoint
            )
            return
        if schema_version != JOINT_W4_SCHEMA_VERSION or self.manifest.get("status") != "complete":
            raise ValueError(f"unsupported or incomplete SpinQuant artifact {artifact_dir}")
        self.joint_v8 = True
        self.head_weight_bits = None
        artifact_checkpoint = self.manifest.get("checkpoint")
        identity_keys = ("config_sha256", "index_sha256", "model_identity")
        if not isinstance(artifact_checkpoint, dict) or not any(
            (
                all(
                    (
                        artifact_checkpoint.get(key) == known.get(key)
                        for key in identity_keys
                    )
                )
                for known in supported_checkpoint_identities()
            )
        ):
            raise ValueError(
                "SpinQuant artifact is not tied to a supported original LLaDA checkpoint"
            )
        if expected_checkpoint is not None and any(
            (
                artifact_checkpoint.get(key) != expected_checkpoint.get(key)
                for key in identity_keys
            )
        ):
            raise ValueError(
                f"SpinQuant artifact/checkpoint mismatch: artifact={artifact_checkpoint.get('model_identity')}, loaded={expected_checkpoint.get('model_identity')}"
            )
        self.checkpoint = artifact_checkpoint
        self.variant = self.manifest.get("variant")
        if self.variant not in VARIANTS:
            raise ValueError("SpinQuant artifact variant mismatch")
        quantization = self.manifest.get("weight_quantization", self.manifest.get("gptq", {}))
        expected_group_size = -1
        if (
            quantization.get("sym") is not True
            or quantization.get("signed_integer_range") != [-8, 7]
            or quantization.get("signed_zero_point") != 0
            or (quantization.get("qzeros") is not False)
            or (quantization.get("group_size") != expected_group_size)
        ):
            raise ValueError("SpinQuant W4 numeric contract mismatch")
        declared_arithmetic = quantization.get("arithmetic_id")
        if expected_group_size == -1 and declared_arithmetic not in {
            None,
            GMINUS1_BF16_ARITHMETIC,
        }:
            raise ValueError("SpinQuant G-1 arithmetic metadata mismatch")
        self.transformer_weight_arithmetic = GMINUS1_BF16_ARITHMETIC
        files = self.manifest.get("files", {})
        required_files = REQUIRED_FILES | {V_CACHE_SCALES_FILE}
        if set(files) != required_files:
            raise ValueError("SpinQuant artifact payload inventory mismatch")
        for name in files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe SpinQuant artifact payload path: {name}")
        self.entries = {
            entry["logical_id"]: entry for entry in self.manifest.get("tensors", [])
        }
        if set(self.entries) != expected_w4_ids() | BF16_IDS:
            raise ValueError("SpinQuant artifact tensor inventory mismatch")
        for logical_id in expected_w4_ids():
            entry = self.entries[logical_id]
            shape = entry.get("shape")
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError(f"SpinQuant W4 tensor shape mismatch: {logical_id}")
            (rows, columns) = shape
            if (
                not isinstance(rows, int)
                or not isinstance(columns, int)
                or rows <= 0
                or (columns <= 0)
            ):
                raise ValueError(
                    f"SpinQuant W4 tensor dimensions mismatch: {logical_id}"
                )
            scale_groups = 1
            expected_kinds = {"gptq_w4", "omniquant_lwc_w4", "rtn_w4"}
            if (
                entry.get("kind") not in expected_kinds
                or entry.get("code_dtype")
                != "signed_int4_twos_complement_low_nibble_first"
                or entry.get("scale_dtype") != "bfloat16_le"
                or (entry.get("scale_mode") != W4_SCALE_MODES[expected_group_size])
                or (entry.get("group_size", -1) != expected_group_size)
                or (entry.get("weight_length_bytes") != rows * ((columns + 1) // 2))
                or (entry.get("scale_length_bytes") != 2 * rows * scale_groups)
            ):
                raise ValueError(f"SpinQuant W4 tensor metadata mismatch: {logical_id}")
        if verify_files:
            for name, record in files.items():
                path = self.artifact_dir / name
                if (
                    path.stat().st_size != record["bytes"]
                    or sha256_file(path) != record["sha256"]
                ):
                    raise ValueError(
                        f"SpinQuant artifact file hash/length mismatch: {name}"
                    )
        self.manifest_sha256 = sha256_file(self.manifest_path)

    def _initialize_lm_head_child(
        self, *, verify_files: bool, expected_checkpoint: Mapping[str, str] | None
    ) -> None:
        if self.manifest.get("status") != "complete":
            raise ValueError("incomplete quantized lm-head child artifact")
        parent_record = self.manifest.get("parent_artifact")
        if not isinstance(parent_record, dict):
            raise ValueError("quantized lm-head child is missing its parent artifact")
        mapped_parent = os.environ.get("SUPRA_SPINQUANT_PARENT_ARTIFACT_DIR")
        parent_path = Path(mapped_parent) if mapped_parent else Path(parent_record["path"])
        if not mapped_parent and not parent_path.is_absolute():
            parent_path = self.artifact_dir / parent_path
        parent_dir = parent_path.expanduser().resolve()
        parent_manifest = parent_dir / MANIFEST_FILE
        if (
            not parent_manifest.is_file()
            or parent_manifest.stat().st_size != parent_record.get("manifest_bytes")
            or sha256_file(parent_manifest) != parent_record.get("manifest_sha256")
        ):
            raise ValueError(
                "quantized lm-head child parent manifest identity mismatch"
            )
        parent = SpinQuantArtifactReader(
            parent_dir,
            verify_files=verify_files,
            expected_checkpoint=expected_checkpoint,
        )
        if (
            parent.head_quantized
            or not parent.joint_v8
            or parent.variant not in R4_VARIANTS
        ):
            raise ValueError(
                "quantized lm-head child requires a non-child joint R4 parent"
            )
        if parent_record.get("schema_version") != parent.manifest.get(
            "schema_version"
        ) or parent_record.get("files") != parent.manifest.get("files"):
            raise ValueError(
                "quantized lm-head child parent payload inventory mismatch"
            )
        if (
            self.manifest.get("checkpoint") != parent.checkpoint
            or self.manifest.get("variant") != parent.variant
        ):
            raise ValueError(
                "quantized lm-head child does not match its parent checkpoint/variant"
            )
        identity_keys = ("config_sha256", "index_sha256", "model_identity")
        if expected_checkpoint is not None and any(
            (
                self.manifest.get("checkpoint", {}).get(key)
                != expected_checkpoint.get(key)
                for key in identity_keys
            )
        ):
            raise ValueError("quantized lm-head child/checkpoint mismatch")
        quantization = self.manifest.get("head_quantization", {})
        weight_bits = self.head_weight_bits
        if weight_bits != 8:
            raise ValueError(
                "quantized lm-head child schema does not define weight bits"
            )
        (code_min, code_max) = (-127, 127)
        expected_group_size = -1
        if (
            quantization.get("bits") != weight_bits
            or quantization.get("sym") is not True
            or quantization.get("signed_integer_range") != [code_min, code_max]
            or (quantization.get("signed_zero_point") != 0)
            or (quantization.get("qzeros") is not False)
            or (quantization.get("group_size") != expected_group_size)
            or (quantization.get("activation_bits") != 8)
        ):
            raise ValueError(f"W{weight_bits} lm-head child numeric contract mismatch")
        files = self.manifest.get("files", {})
        expected_files = LM_HEAD_W8_CHILD_FILES
        if set(files) != expected_files:
            raise ValueError(f"W{weight_bits} lm-head child payload inventory mismatch")
        for name, record in files.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"unsafe W{weight_bits} lm-head child payload path: {name}"
                )
            if verify_files:
                payload_path = self.artifact_dir / name
                if (
                    payload_path.stat().st_size != record["bytes"]
                    or sha256_file(payload_path) != record["sha256"]
                ):
                    raise ValueError(
                        f"W{weight_bits} lm-head child file hash/length mismatch: {name}"
                    )
        child_entries = {
            entry["logical_id"]: entry for entry in self.manifest.get("tensors", [])
        }
        if set(child_entries) != {EMBEDDING, LM_HEAD}:
            raise ValueError(f"W{weight_bits} lm-head child tensor inventory mismatch")
        embedding_entry = child_entries[EMBEDDING]
        head_entry = child_entries[LM_HEAD]
        if embedding_entry.get("parent_slice_sha256") != parent.entries[EMBEDDING].get(
            "sha256"
        ) or head_entry.get("source_bf16_sha256") != parent.entries[LM_HEAD].get(
            "sha256"
        ):
            raise ValueError(
                f"W{weight_bits} lm-head child source tensor identity mismatch"
            )
        embedding_shape = parent.entries[EMBEDDING].get("shape")
        head_shape = parent.entries[LM_HEAD].get("shape")
        if (
            embedding_entry.get("kind") != "rotated_bf16"
            or embedding_entry.get("shape") != embedding_shape
            or embedding_entry.get("dtype") != "bfloat16_le"
            or (embedding_entry.get("payload") != EMBEDDING_FILE)
            or (embedding_entry.get("offset_bytes") != 0)
            or (embedding_entry.get("length_bytes") != 2 * math.prod(embedding_shape))
            or (embedding_entry.get("sha256") != files[EMBEDDING_FILE].get("sha256"))
        ):
            raise ValueError(
                f"W{weight_bits} lm-head child embedding metadata mismatch"
            )
        if not isinstance(head_shape, list) or len(head_shape) != 2:
            raise ValueError(f"W{weight_bits} lm-head child source head shape mismatch")
        (head_rows, head_columns) = head_shape
        group_size = int(quantization["group_size"])
        scale_groups = 1
        code_file = LM_HEAD_W8_CODES_FILE
        scale_file = LM_HEAD_W8_SCALES_FILE
        tensor_kind = "symmetric_w8"
        code_dtype = "signed_int8_twos_complement"
        expected_weight_bytes = head_rows * head_columns
        if (
            head_entry.get("kind") != tensor_kind
            or head_entry.get("shape") != head_shape
            or head_entry.get("code_dtype") != code_dtype
            or (head_entry.get("scale_dtype") != "bfloat16_le")
            or (
                head_entry.get("scale_mode")
                != "per_output_channel"
            )
            or (head_entry.get("group_size", -1) != group_size)
            or (head_entry.get("code_payload") != code_file)
            or (head_entry.get("scale_payload") != scale_file)
            or (head_entry.get("weight_offset_bytes") != 0)
            or (head_entry.get("weight_length_bytes") != expected_weight_bytes)
            or (head_entry.get("scale_offset_bytes") != 0)
            or (head_entry.get("scale_length_bytes") != 2 * head_rows * scale_groups)
            or (head_entry.get("codes_sha256") != files[code_file].get("sha256"))
            or (head_entry.get("scales_sha256") != files[scale_file].get("sha256"))
            or (not isinstance(head_entry.get("code_min"), int))
            or (not isinstance(head_entry.get("code_max"), int))
            or (head_entry["code_min"] < code_min)
            or (head_entry["code_max"] > code_max)
            or (head_entry["code_min"] > head_entry["code_max"])
        ):
            raise ValueError(f"W{weight_bits} lm-head child tensor metadata mismatch")
        if self.manifest.get("total_file_bytes") != sum(
            (int(record.get("bytes", -1)) for record in files.values())
        ):
            raise ValueError(f"W{weight_bits} lm-head child total byte count mismatch")
        self._parent_reader = parent
        self.transformer_weight_arithmetic = parent.transformer_weight_arithmetic
        self.joint_v8 = True
        self.checkpoint = parent.checkpoint
        self.variant = parent.variant
        self.entries = dict(parent.entries)
        self.entries.update(child_entries)
        self.manifest_sha256 = sha256_file(self.manifest_path)

    def read_v_cache_scales(self) -> torch.Tensor:
        if self._parent_reader is not None:
            return self._parent_reader.read_v_cache_scales()
        record = self.manifest.get("attention", {}).get("v_quantization")
        if not isinstance(record, dict):
            raise ValueError("joint artifact is missing V8 calibration metadata")
        shape = record.get("shape")
        if (
            record.get("scale_mode") != "static_per_layer_per_kv_head"
            or record.get("signed_integer_range") != [-127, 127]
            or record.get("signed_zero_point") != 0
            or (record.get("qzeros") is not False)
            or (not isinstance(shape, list))
            or (len(shape) != 2)
            or (shape[0] != 32)
            or (shape[1] <= 0)
            or (record.get("payload") != V_CACHE_SCALES_FILE)
        ):
            raise ValueError("V8 numeric contract mismatch")
        payload = (self.artifact_dir / V_CACHE_SCALES_FILE).read_bytes()
        if _sha256(payload) != record.get("payload_sha256"):
            raise ValueError("V8 scale payload hash mismatch")
        values = _bf16_from_le_payload(payload, name="V8 scale")
        if values.numel() != math.prod(shape):
            raise ValueError("V8 scale payload length mismatch")
        scales = values.reshape(shape).to(torch.float32)
        if not bool(torch.all(torch.isfinite(scales))) or bool(torch.any(scales <= 0)):
            raise ValueError("V8 scales must be finite and positive")
        return scales

    def read_w4(self, logical_id: str) -> SpinQuantW4Tensor:
        if self._parent_reader is not None and logical_id != LM_HEAD:
            return self._parent_reader.read_w4(logical_id)
        entry = self.entries[logical_id]
        if entry.get("kind") not in {
            "gptq_w4",
            "omniquant_lwc_w4",
            "rtn_w4",
        }:
            raise TypeError(f"{logical_id} is not a supported W4 tensor")
        code_file = entry.get("code_payload", W4_CODES_FILE)
        scale_file = entry.get("scale_payload", W4_SCALES_FILE)
        with (self.artifact_dir / code_file).open("rb") as handle:
            handle.seek(entry["weight_offset_bytes"])
            code_payload = handle.read(entry["weight_length_bytes"])
        with (self.artifact_dir / scale_file).open("rb") as handle:
            handle.seek(entry["scale_offset_bytes"])
            scale_payload = handle.read(entry["scale_length_bytes"])
        if (
            _sha256(code_payload) != entry["codes_sha256"]
            or _sha256(scale_payload) != entry["scales_sha256"]
        ):
            raise ValueError(f"SpinQuant tensor slice hash mismatch: {logical_id}")
        (rows, columns) = entry["shape"]
        packed = (
            torch.frombuffer(bytearray(code_payload), dtype=torch.uint8)
            .clone()
            .reshape(rows, (columns + 1) // 2)
        )
        codes = unpack_signed_int4(packed, columns)
        scales = _bf16_from_le_payload(scale_payload, name=f"{logical_id} W4 scale").to(
            torch.float32
        )
        return SpinQuantW4Tensor(
            codes, scales, scale_mode=str(entry.get("scale_mode", "per_output_channel"))
        )

    def read_quantized_head(self) -> Any:
        if self.head_weight_bits != 8 or self._parent_reader is None:
            raise TypeError("artifact does not contain a quantized lm-head")
        entry = self.entries[LM_HEAD]
        code_file = entry["code_payload"]
        scale_file = entry["scale_payload"]
        code_payload = (self.artifact_dir / code_file).read_bytes()
        scale_payload = (self.artifact_dir / scale_file).read_bytes()
        if (
            _sha256(code_payload) != entry["codes_sha256"]
            or _sha256(scale_payload) != entry["scales_sha256"]
        ):
            raise ValueError("SpinQuant W8 lm-head tensor slice hash mismatch")
        (rows, columns) = entry["shape"]
        codes = (
            torch.frombuffer(bytearray(code_payload), dtype=torch.int8)
            .clone()
            .reshape(rows, columns)
        )
        scales = _bf16_from_le_payload(scale_payload, name="W8 lm-head scale").to(
            torch.float32
        )
        from quantization.numeric import SpinQuantW8Tensor

        return SpinQuantW8Tensor(codes, scales)

    def read_bf16(self, logical_id: str) -> torch.Tensor:
        if self._parent_reader is not None and logical_id != EMBEDDING:
            raise TypeError(
                f"{logical_id} is not stored as BF16 in the quantized lm-head child"
            )
        entry = self.entries[logical_id]
        if entry.get("kind") != "rotated_bf16":
            raise TypeError(f"{logical_id} is not a supported BF16 tensor")
        payload_file = entry.get("payload", BF16_FILE)
        with (self.artifact_dir / payload_file).open("rb") as handle:
            handle.seek(entry["offset_bytes"])
            payload = handle.read(entry["length_bytes"])
        if _sha256(payload) != entry["sha256"]:
            raise ValueError(f"SpinQuant BF16 tensor slice hash mismatch: {logical_id}")
        values = _bf16_from_le_payload(payload, name=logical_id)
        return values.reshape(entry["shape"])
