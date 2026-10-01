"""G-1 W4 deployment evaluation with Feature1+2 and optional Feature3."""

from __future__ import annotations
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import handle_non_serializable, hash_string
from transformers import AutoTokenizer
from transformers.modeling_utils import no_init_weights
from numerics.candidate import (
    CANDIDATE_NUMERIC_SCHEDULE,
    CANDIDATE_SUPPRESSION_SCHEDULE,
)
from generation.engine import generate
from numerics.int8_matmul import triton_int8_bmm_available
from numerics.linear_kernels import triton_linear_numeric_available
from numerics.operator_kernels import triton_operator_numeric_available
from numerics.precision import RowPrecisionContext
from quantization.artifact import (
    SpinQuantArtifactReader,
)
from quantization.model import (
    build_target_numeric_coverage,
    replace_with_spinquant_joint_full_w4a8_v8,
)
from quantization.rotation import checkpoint_identity
from model.modeling_llada import LLaDAModelLM
from model.configuration_llada import LLaDAConfig


def _question_start_token(request, context, offsets):
    start = getattr(request, "question_char_start", None)
    if start is None:
        question = request.doc.get("question")
        if not isinstance(question, str):
            raise ValueError("question protection requires input question metadata")
        suffix = f"Question: {question}\nAnswer:"
        if not context.endswith(suffix):
            raise ValueError(
                "question metadata does not match the prompt suffix"
            )
        start = len(context) - len(suffix)
    if (
        type(start) is not int
        or not 0 <= start < len(context)
        or not context[start:].startswith("Question: ")
        or not context.endswith("\nAnswer:")
    ):
        raise ValueError("invalid input question boundary")
    for index, (_, end) in enumerate(offsets):
        if end > start:
            return index
    raise ValueError("tokenizer offsets do not cover the input question")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evaluation_runtime(device: str):
    distributed_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if distributed_size == 1:
        return None, 0, 1, torch.device(device)
    if device != "cuda":
        raise ValueError("distributed evaluation requires device=cuda")
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )
    if accelerator.num_processes != distributed_size:
        raise ValueError("Accelerate world size differs from the launcher")
    return (accelerator, accelerator.process_index,
            accelerator.num_processes, accelerator.device)


def _rank_trace_path(path: Path, rank: int, world_size: int) -> Path:
    if world_size == 1:
        return path
    return path.with_name(f"{path.stem}.rank_{rank:02d}{path.suffix}")


def _tensor_bit_evidence(tensor: torch.Tensor) -> Dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": _sha256_bytes(raw),
    }


def _compact_state_capture(event: Any) -> Dict[str, Any]:
    state_inputs = {
        "tokens_before": _tensor_bit_evidence(event.tokens_before),
        "block_state_before": _tensor_bit_evidence(event.block_state_before),
    }
    decision_inputs = {
        name: _tensor_bit_evidence(getattr(event, name))
        for name in (
            "prediction_positions",
            "prediction_mask",
            "teacher_top1_token_ids",
            "teacher_top2_token_ids",
            "teacher_top1_margin",
            "teacher_top1_confidence",
            "teacher_top1_action_confidence",
            "teacher_current_token_probability",
        )
    }
    row_bits_raw = event.row_bits.detach().cpu().to(torch.int8).contiguous()
    if row_bits_raw.ndim == 2 and row_bits_raw.shape[0] == 1:
        row_bits = row_bits_raw.reshape(-1)
    elif row_bits_raw.ndim == 1:
        row_bits = row_bits_raw
    else:
        raise RuntimeError(
            f"row_bits must have shape [rows] or [1, rows], got {list(row_bits_raw.shape)}"
        )
    return {
        "capture_index": int(event.capture_index),
        "capture_nfe_before": int(event.nfe_before),
        "capture_block_index": int(event.block_index),
        "capture_step_index": int(event.step_index),
        "capture_forward_kind": str(event.forward_kind),
        "cache_initialized_before": bool(event.cache_initialized_before),
        "full_sequence_recompute": bool(event.full_sequence_recompute),
        "state_before_sha256": _sha256_bytes(
            json.dumps(state_inputs, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ),
        "model_input_ids": _tensor_bit_evidence(event.model_input_ids),
        "capture_input_positions": [
            int(value) for value in event.input_positions.detach().cpu().tolist()
        ],
        "capture_refresh_positions": [
            int(value) for value in event.refresh_positions.detach().cpu().tolist()
        ],
        "row_bits": [int(value) for value in row_bits.tolist()],
        "row_bits_source_shape": list(row_bits_raw.shape),
        "row_bits_sha256": _sha256_bytes(row_bits.view(torch.uint8).numpy().tobytes()),
        "decision_inputs_sha256": _sha256_bytes(
            json.dumps(decision_inputs, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ),
    }


def _merge_state_capture_evidence(
    compact_trace: List[Dict[str, Any]], captures: List[Dict[str, Any]]
) -> None:
    if len(compact_trace) != len(captures):
        raise RuntimeError(
            f"state-capture count differs from the generation trace: {len(captures)} != {len(compact_trace)}"
        )
    for index, (trace_event, capture) in enumerate(zip(compact_trace, captures)):
        if capture["capture_index"] != index:
            raise RuntimeError(
                f"state-capture sequence is not contiguous at index {index}"
            )
        expected = (
            int(trace_event["block_index"]),
            int(trace_event["step_index"]),
            str(trace_event["forward_kind"]),
        )
        actual = (
            capture["capture_block_index"],
            capture["capture_step_index"],
            capture["capture_forward_kind"],
        )
        if actual != expected:
            raise RuntimeError(
                f"state-capture event {index} differs from trace: {actual} != {expected}"
            )
        trace_event.update(capture)


@register_model("quantized_llada")
class QuantizedLLaDALM(LM):
    """Batch-one LLaDA evaluator with independent requests per process."""

    def __init__(
        self,
        model_path: str,
        spinquant_artifact_dir: str = "",
        spinquant_variant: str = "fixed-r1r2-r4-only",
        spinquant_execution: str = "joint-full-w4a8-v8",
        trace_output_path: str = "",
        max_length: int = 4096,
        batch_size: int = 1,
        steps: int = 256,
        gen_length: int = 256,
        block_length: int = 32,
        temperature: float = 0.0,
        remasking: str = "low_confidence",
        generation_mode: str = "feature1_packed_feature2_fused_dynamic_block",
        decode_k: int = 3,
        decode_threshold: float = 0.9,
        feature1_tri_neighbor_scope: str = "tri",
        feature1_dependency_layer_mode: str = "all",
        feature1_tri_target_active_rows: float = 39.5,
        feature2_context_a8_rows: int = 10,
        feature2_cache_initialization_activation_policy: str = "default",
        feature1_cross_block_cache_handoff: bool = True,
        feature1_cross_block_boundary_scope: str = "layer0_attention_two_stage",
        feature1_cross_block_boundary_target_rows: int = 64,
        feature1_cross_block_full_prefix_oracle_block: int = 1,
        feature1_cross_block_block_initialization_deep_rows: int = 0,
        feature1_cross_block_block_initialization_dependency_only: bool = False,
        feature1_cross_block_block_initialization_relative_score_floor: float = 0.0,
        feature1_cross_block_block_initialization_include_future_dependency: bool = False,
        feature1_cross_block_block_initialization_protected_future_blocks: int = -1,
        feature1_cross_block_block_initialization_deep_bits: int = 0,
        feature1_cross_block_block_initialization_deep_clip_ratio: float = 0.0,
        feature1_cross_block_block_initialization_dependency_tiebreak: bool = False,
        feature1_cross_block_block_initialization_protect_question: bool = False,
        feature1_cross_block_block_initialization_protect_generation: bool = False,
        feature1_cross_block_block_initialization_all_a8: bool = False,
        feature1_cross_block_block_initialization_keep_global_l0_cache: bool = False,
        feature1_cross_block_block_initialization_global_layers: int = 1,
        feature1_cross_block_boundary_deep_a8_row_limit: int = -1,
        cross_block_boundary_context_row_bits: int = 8,
        cross_block_pending_confidence_mode: str = "remask_only",
        cross_block_pending_relation_mode: str = "direct",
        cross_block_dependency_policy: str = "current_keys_committed_rows",
        feature2_tau_high: float = 0.86,
        feature2_tau_high_tail: float = -1.0,
        feature2_tau_high_tail_after_step: int = 0,
        feature2_tau_low: float = 0.68,
        feature2_confirm_tau: float = 0.68,
        feature2_a4_direct_tau: float = -1.0,
        feature2_budget_scale: float = 16.0,
        feature2_stability_bonus: float = 0.05,
        feature2_tail_confirmation_policy: str = "none",
        feature3_maturity_age: int = 3,
        feature3_precision_policy: str = "original",
        feature3_dynamic_block_target_joint_rows: int = 48,
        feature3_dynamic_block_max_next_rows: int = 4,
        feature3_dynamic_block_source_b_max_attempts: int = -1,
        feature3_dynamic_block_source_b_retry_min_confidence: float = 0.0,
        feature3_dynamic_block_source_b_a4_only: bool = False,
        feature3_dynamic_block_source_b_dependency_tie_rank: bool = False,
        feature3_dynamic_block_source_a_confirm_at_handoff: bool = False,
        feature3_dynamic_block_target_prediction_rows: int = -1,
        feature3_dynamic_block_next_admission_budget: int = 1,
        feature3_dynamic_block_min_reuse_score: float = 0.0,
        feature3_dynamic_block_next_candidate_policy: str = "high_only",
        feature3_dynamic_block_next_relation_preference: str = "high_dependency",
        feature3_dynamic_block_max_current_unresolved: int = 8,
        feature3_dynamic_block_max_handoff_verification_rows: int = 8,
        feature3_dynamic_block_canonical_future: bool = False,
        feature3_dynamic_block_future_horizon: int = 1,
        feature3_dynamic_block_canonical_direct_tau: float = -1.0,
        feature3_dynamic_block_allow_deferred_verification: bool = False,
        feature3_dynamic_block_next_min_confidence: float = 0.92,
        logical_a4a8: bool = False,
        instruct_prompt_mode: str = "raw",
        humaneval_full_completion: bool = False,
        confidence_eos_eot_inf: bool = False,
        candidate_numeric_mode: str = "bf16_lut",
        require_target_numeric_coverage: bool = False,
        a4_clip_ratio: Optional[float] = None,
        a4_output_clip_ratio: Optional[float] = None,
        silu_table: str = "",
        source_commit: str = "",
        seed: int = 20260806,
        mask_id: int = 126336,
        device: str = "cuda",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        if not spinquant_artifact_dir:
            raise ValueError("spinquant_artifact_dir is missing")
        if (
            generation_mode
            not in (
                "feature1_packed_feature2_fused_dynamic_block",
                "feature1_packed_feature2_fused_a4a8",
                "baseline_fixed_k",
                "feature1_fixed_k",
                "feature1_fixed_threshold",
            )
        ):
            raise ValueError(f"unknown generation_mode: {generation_mode!r}")
        if spinquant_execution != "joint-full-w4a8-v8":
            raise ValueError(f"spinquant_execution={spinquant_execution!r}; expected 'joint-full-w4a8-v8'")
        if spinquant_variant != "fixed-r1r2-r4-only":
            raise ValueError(f"spinquant_variant={spinquant_variant!r}; expected 'fixed-r1r2-r4-only'")
        if logical_a4a8 != False:
            raise ValueError(f"logical_a4a8={logical_a4a8!r}; expected False")
        if candidate_numeric_mode != "bf16_lut":
            raise ValueError(f"candidate_numeric_mode={candidate_numeric_mode!r}; expected 'bf16_lut'")
        if block_length != 32:
            raise ValueError(f"block_length={block_length!r}; expected 32")
        if feature1_tri_neighbor_scope != "tri":
            raise ValueError(f"feature1_tri_neighbor_scope={feature1_tri_neighbor_scope!r}; expected 'tri'")
        if feature1_dependency_layer_mode != "all":
            raise ValueError(
                f"feature1_dependency_layer_mode={feature1_dependency_layer_mode!r}; expected 'all'"
            )
        baseline = generation_mode == "baseline_fixed_k"
        if feature1_cross_block_cache_handoff != (not baseline):
            raise ValueError(
                f"feature1_cross_block_cache_handoff={feature1_cross_block_cache_handoff!r}; expected {not baseline}"
            )
        if feature1_cross_block_boundary_scope != ("tri" if baseline else "layer0_attention_two_stage"):
            raise ValueError(
                f"feature1_cross_block_boundary_scope={feature1_cross_block_boundary_scope!r}; "
                f"expected {'tri' if baseline else 'layer0_attention_two_stage'!r}"
            )
        if feature1_cross_block_full_prefix_oracle_block != (-1 if baseline else 1):
            raise ValueError(
                f"feature1_cross_block_full_prefix_oracle_block={feature1_cross_block_full_prefix_oracle_block!r}; expected {-1 if baseline else 1}"
            )
        fixed_decoding = generation_mode in {"baseline_fixed_k", "feature1_fixed_k", "feature1_fixed_threshold"}
        if feature2_cache_initialization_activation_policy not in {"default", "a4"}:
            raise ValueError("cache initialization activation policy must be default or a4")
        if feature1_cross_block_block_initialization_global_layers != 1:
            raise ValueError("sparse block initialization supports one full global layer")
        if feature1_cross_block_block_initialization_deep_rows < 0 or (
            feature1_cross_block_block_initialization_deep_rows
            and feature1_cross_block_block_initialization_deep_rows < 32
        ):
            raise ValueError("sparse block initialization requires at least 32 rows")
        if (
            feature1_cross_block_block_initialization_dependency_tiebreak
            or feature1_cross_block_block_initialization_protect_question
            or feature1_cross_block_block_initialization_protect_generation
            or feature1_cross_block_block_initialization_all_a8
            or feature1_cross_block_block_initialization_keep_global_l0_cache
        ) and not feature1_cross_block_block_initialization_deep_rows:
            raise ValueError("block initialization selection options require sparse block initialization")
        if (
            feature1_cross_block_block_initialization_protect_question
            and instruct_prompt_mode != "raw"
        ):
            raise ValueError("question protection requires the raw GSM prompt")
        if feature3_precision_policy != ("all_a8" if fixed_decoding else "original"):
            raise ValueError("feature3_precision_policy must be all_a8 for fixed decoding or original for state decoding")
        if int(decode_k) < 1 or not 0.0 <= float(decode_threshold) <= 1.0:
            raise ValueError("invalid fixed decoding parameter")
        if feature3_maturity_age != 3:
            raise ValueError("feature3_maturity_age must be 3")
        (resolved_a4_clip_ratio, resolved_a4_output_clip_ratio, resolved_silu_table) = (
            1.0 if a4_clip_ratio is None else a4_clip_ratio,
            a4_output_clip_ratio,
            silu_table,
        )
        if fixed_decoding and cross_block_boundary_context_row_bits != 8:
            raise ValueError("fixed decoding requires A8 boundary scout context")
        super().__init__()
        if kwargs:
            raise ValueError(
                f"Unsupported evaluation model arguments: {sorted(kwargs)}"
            )
        if int(batch_size) != 1:
            raise ValueError("batch_size must be 1")
        if int(gen_length) <= 0 or int(gen_length) % 32:
            raise ValueError("gen_length must be positive and divisible by block_length=32")
        if int(steps) <= 0 or int(steps) % (int(gen_length) // 32):
            raise ValueError("steps must be positive and divisible by the generation block count")
        self.steps, self.gen_length = int(steps), int(gen_length)
        if float(temperature) != 0.0 or str(remasking) != "low_confidence":
            raise ValueError(
                "temperature must be 0.0 and remasking must be low_confidence"
            )
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device={device!r} requires an available CUDA device")
        accelerator, self._rank, self._world_size, self._device = _evaluation_runtime(device)
        if accelerator is not None:
            self.accelerator = accelerator
        if not bool(trust_remote_code):
            raise ValueError("LLaDA checkpoint loading requires trust_remote_code=True")
        self.model_path = str(model_path)
        spinquant_checkpoint = checkpoint_identity(Path(self.model_path))
        self._max_length = int(max_length)
        self.generation_mode = str(generation_mode)
        self.decoding_mode = (
            "fixed_k" if baseline else generation_mode.removeprefix("feature1_") if fixed_decoding else "feature2"
        )
        self.decode_k = int(decode_k)
        self.decode_threshold = float(decode_threshold)
        self.fixed_decoding = fixed_decoding
        self.feature1_tri_target_active_rows = float(feature1_tri_target_active_rows)
        self.feature2_context_a8_rows = int(feature2_context_a8_rows)
        self.feature2_cache_initialization_activation_policy = str(feature2_cache_initialization_activation_policy)
        self.feature1_cross_block_block_initialization_deep_rows = int(
            feature1_cross_block_block_initialization_deep_rows
        )
        self.feature1_cross_block_block_initialization_dependency_tiebreak = bool(
            feature1_cross_block_block_initialization_dependency_tiebreak
        )
        self.feature1_cross_block_block_initialization_dependency_only = bool(feature1_cross_block_block_initialization_dependency_only)
        if not 0 <= feature1_cross_block_block_initialization_relative_score_floor <= 1:
            raise ValueError("relative block initialization score floor must be in [0,1]")
        self.feature1_cross_block_block_initialization_relative_score_floor = float(torch.tensor(
            feature1_cross_block_block_initialization_relative_score_floor, dtype=torch.bfloat16
        ))
        self.feature1_cross_block_block_initialization_include_future_dependency = bool(feature1_cross_block_block_initialization_include_future_dependency)
        self.feature1_cross_block_block_initialization_protected_future_blocks = int(feature1_cross_block_block_initialization_protected_future_blocks)
        self.feature1_cross_block_block_initialization_deep_bits = int(feature1_cross_block_block_initialization_deep_bits)
        self.feature1_cross_block_block_initialization_deep_clip_ratio = float(
            torch.tensor(float(feature1_cross_block_block_initialization_deep_clip_ratio), dtype=torch.bfloat16)
        )
        if self.feature1_cross_block_block_initialization_dependency_only and not feature1_cross_block_block_initialization_dependency_tiebreak:
            raise ValueError("dependency-only block initialization requires dependency ordering")
        if self.feature1_cross_block_block_initialization_include_future_dependency and not self.feature1_cross_block_block_initialization_dependency_only:
            raise ValueError("joint block initialization risk requires dependency-only selection")
        if self.feature1_cross_block_block_initialization_protected_future_blocks < -1 or (
            self.feature1_cross_block_block_initialization_protected_future_blocks >= 0
            and not feature1_cross_block_block_initialization_protect_generation
        ):
            raise ValueError("near-future protection requires generation protection")
        if self.feature1_cross_block_block_initialization_deep_bits not in (0, 4, 8) or (
            self.feature1_cross_block_block_initialization_deep_bits and not feature1_cross_block_block_initialization_all_a8
        ):
            raise ValueError("separate deep bits require an A8 sparse block initialization scout")
        if not 0 <= self.feature1_cross_block_block_initialization_deep_clip_ratio <= 1 or (
            self.feature1_cross_block_block_initialization_deep_clip_ratio and self.feature1_cross_block_block_initialization_deep_bits != 4
        ):
            raise ValueError("block initialization clipping override requires uniform deep A4")
        self.feature1_cross_block_block_initialization_protect_question = bool(
            feature1_cross_block_block_initialization_protect_question
        )
        self.feature1_cross_block_block_initialization_protect_generation = bool(
            feature1_cross_block_block_initialization_protect_generation
        )
        self.feature1_cross_block_block_initialization_all_a8 = bool(
            feature1_cross_block_block_initialization_all_a8
        )
        self.feature1_cross_block_block_initialization_keep_global_l0_cache = bool(
            feature1_cross_block_block_initialization_keep_global_l0_cache
        )
        self.feature1_cross_block_boundary_target_rows = int(
            feature1_cross_block_boundary_target_rows
        )
        self.feature1_cross_block_boundary_deep_a8_row_limit = int(
            feature1_cross_block_boundary_deep_a8_row_limit
        )
        self.cross_block_boundary_context_row_bits = int(
            cross_block_boundary_context_row_bits
        )
        self.cross_block_pending_confidence_mode = str(
            cross_block_pending_confidence_mode
        )
        self.cross_block_pending_relation_mode = str(cross_block_pending_relation_mode)
        if cross_block_dependency_policy not in {"initial_keys_selected_rows", "current_keys_committed_rows"}:
            raise ValueError("unsupported cross-block dependency policy")
        self.cross_block_dependency_policy = cross_block_dependency_policy
        if self.cross_block_pending_relation_mode != "direct":
            raise ValueError("the deployed dependency rule is direct")
        if not 1.0 <= self.feature1_tri_target_active_rows <= 96.0:
            raise ValueError("feature1_tri_target_active_rows must be in [1, 96]")
        if self.feature2_context_a8_rows < 0:
            raise ValueError("feature2_context_a8_rows must be nonnegative")
        if self.feature1_cross_block_boundary_deep_a8_row_limit < -1:
            raise ValueError("deep A8 limit requires Layer0 scout boundaries")
        boundary_row_limit = (
            256 if feature1_cross_block_boundary_scope == "layer0_attention_two_stage" else 96
        )
        if not 32 <= self.feature1_cross_block_boundary_target_rows <= boundary_row_limit:
            raise ValueError(
                "feature1_cross_block_boundary_target_rows exceeds the scope-specific range"
            )
        if self.cross_block_boundary_context_row_bits not in {4, 8}:
            raise ValueError("cross_block_boundary_context_row_bits must be 4 or 8")
        if self.cross_block_pending_confidence_mode not in {
            "remask_only",
            "all_changes",
            "stable_unmask",
        }:
            raise ValueError(
                "cross_block_pending_confidence_mode must be remask_only, all_changes, or stable_unmask"
            )
        self.feature2_parameters = {
            "tau_high": float(feature2_tau_high),
            "tau_high_tail": None
            if float(feature2_tau_high_tail) < 0.0
            else float(feature2_tau_high_tail),
            "tau_high_tail_after_step": int(feature2_tau_high_tail_after_step),
            "tau_low": float(feature2_tau_low),
            "confirm_tau": float(feature2_confirm_tau),
            "a4_direct_tau": float(feature2_a4_direct_tau),
            "budget_scale": float(feature2_budget_scale),
            "stability_bonus": float(feature2_stability_bonus),
            "tail_confirmation_policy": str(feature2_tail_confirmation_policy),
        }
        if self.feature2_parameters["tail_confirmation_policy"] not in {
            "none",
            "stable_only",
            "all",
        }:
            raise ValueError("unsupported feature2_tail_confirmation_policy")
        if self.feature2_parameters["tau_high_tail_after_step"] < 0:
            raise ValueError("feature2_tau_high_tail_after_step must be nonnegative")
        if self.feature2_parameters["tau_high_tail"] is not None and (
            not self.feature2_parameters["tau_low"]
            <= self.feature2_parameters["tau_high_tail"]
            <= self.feature2_parameters["tau_high"]
        ):
            raise ValueError(
                "feature2_tau_high_tail must satisfy tau_low <= value <= tau_high"
            )
        if self.feature2_parameters["a4_direct_tau"] != -1.0 and (
            not 0.0 <= self.feature2_parameters["a4_direct_tau"] <= 1.0
        ):
            raise ValueError("feature2_a4_direct_tau must be -1 or in [0, 1]")
        self.source_commit = str(source_commit)
        self.feature3_dynamic_block_target_joint_rows = int(
            feature3_dynamic_block_target_joint_rows
        )
        self.feature3_dynamic_block_max_next_rows = int(feature3_dynamic_block_max_next_rows)
        self.feature3_dynamic_block_source_b_max_attempts = int(
            feature3_dynamic_block_source_b_max_attempts
        )
        if self.feature3_dynamic_block_source_b_max_attempts < -1:
            raise ValueError("Source B attempt limit must be -1 or nonnegative")
        self.feature3_dynamic_block_source_b_retry_min_confidence = float(
            torch.tensor(
                feature3_dynamic_block_source_b_retry_min_confidence, dtype=torch.bfloat16
            )
        )
        if not 0 <= self.feature3_dynamic_block_source_b_retry_min_confidence <= 1:
            raise ValueError("Source B retry confidence must be in [0,1]")
        self.feature3_dynamic_block_source_b_dependency_tie_rank = bool(
            feature3_dynamic_block_source_b_dependency_tie_rank
        )
        self.feature3_dynamic_block_source_b_a4_only = bool(feature3_dynamic_block_source_b_a4_only)
        self.feature3_dynamic_block_source_a_confirm_at_handoff = bool(
            feature3_dynamic_block_source_a_confirm_at_handoff
        )
        self.feature3_dynamic_block_target_prediction_rows = int(
            feature3_dynamic_block_target_prediction_rows
        )
        if (
            self.feature3_dynamic_block_target_prediction_rows != -1
            and not 1 <= self.feature3_dynamic_block_target_prediction_rows <= 64
        ):
            raise ValueError("joint prediction target must be -1 or in [1,64]")
        self.feature3_dynamic_block_next_admission_budget = int(
            feature3_dynamic_block_next_admission_budget
        )
        self.feature3_dynamic_block_min_reuse_score = float(
            feature3_dynamic_block_min_reuse_score
        )
        self.feature3_dynamic_block_next_candidate_policy = str(
            feature3_dynamic_block_next_candidate_policy
        )
        self.feature3_dynamic_block_next_relation_preference = str(
            feature3_dynamic_block_next_relation_preference
        )
        self.feature3_dynamic_block_max_current_unresolved = int(
            feature3_dynamic_block_max_current_unresolved
        )
        self.feature3_dynamic_block_max_handoff_verification_rows = int(
            feature3_dynamic_block_max_handoff_verification_rows
        )
        self.feature3_dynamic_block_canonical_future = bool(
            feature3_dynamic_block_canonical_future
        )
        self.feature3_dynamic_block_future_horizon = int(
            feature3_dynamic_block_future_horizon
        )
        self.feature3_dynamic_block_canonical_direct_tau = float(
            feature3_dynamic_block_canonical_direct_tau
        )
        self.feature3_dynamic_block_allow_deferred_verification = bool(
            feature3_dynamic_block_allow_deferred_verification
        )
        self.feature3_dynamic_block_next_min_confidence = float(
            feature3_dynamic_block_next_min_confidence
        )
        if not 1 <= self.feature3_dynamic_block_target_joint_rows <= 48:
            raise ValueError("feature3_dynamic_block_target_joint_rows must be in [1, 48]")
        if self.feature3_dynamic_block_max_next_rows <= 0:
            raise ValueError("feature3_dynamic_block_max_next_rows must be positive")
        if self.feature3_dynamic_block_next_admission_budget <= 0:
            raise ValueError(
                "feature3_dynamic_block_next_admission_budget must be positive"
            )
        if not 0.0 <= self.feature3_dynamic_block_min_reuse_score <= 1.0:
            raise ValueError("feature3_dynamic_block_min_reuse_score must be in [0, 1]")
        if not 1 <= self.feature3_dynamic_block_max_current_unresolved <= 32:
            raise ValueError(
                "feature3_dynamic_block_max_current_unresolved must be in [1, block_length]"
            )
        if not 1 <= self.feature3_dynamic_block_max_handoff_verification_rows <= 32:
            raise ValueError(
                "feature3_dynamic_block_max_handoff_verification_rows must be in [1, block_length]"
            )
        if self.feature3_dynamic_block_next_candidate_policy not in {
            "high_only",
            "high_stable",
            "stable_high",
            "stable_low_only",
            "stable_low_seen",
            "stable_any",
            "proposal_verify",
            "observe_only",
        }:
            raise ValueError("unsupported feature3_dynamic_block_next_candidate_policy")
        if self.feature3_dynamic_block_next_relation_preference not in {
            "high_dependency",
            "low_dependency",
        }:
            raise ValueError("unsupported feature3_dynamic_block_next_relation_preference")
        if not 0.0 <= self.feature3_dynamic_block_next_min_confidence <= 1.0:
            raise ValueError(
                "feature3_dynamic_block_next_min_confidence must be in [0, 1]"
            )
        if (
            self.feature3_dynamic_block_canonical_future
            and self.generation_mode != "feature1_packed_feature2_fused_dynamic_block"
        ):
            raise ValueError(
                "canonical future requires dynamic generation, immediate K/V writes, and no shadow handoff"
            )
        if self.feature3_dynamic_block_future_horizon not in {1, 2}:
            raise ValueError("future horizon must be 1 or 2")
        if self.feature3_dynamic_block_future_horizon > 1 and (
            not self.feature3_dynamic_block_canonical_future
        ):
            raise ValueError("multi-block horizon requires canonical future")
        if not (
            self.feature3_dynamic_block_canonical_direct_tau == -1.0
            or 0.0 <= self.feature3_dynamic_block_canonical_direct_tau <= 1.0
        ):
            raise ValueError("canonical future direct tau must be -1 or in [0, 1]")
        if self.feature3_dynamic_block_canonical_direct_tau >= 0.0 and (
            not self.feature3_dynamic_block_canonical_future
        ):
            raise ValueError("canonical direct tau requires canonical future")
        self.instruct_prompt_mode = str(instruct_prompt_mode)
        self.humaneval_full_completion = bool(humaneval_full_completion)
        self.confidence_eos_eot_inf = bool(confidence_eos_eot_inf)
        if self.instruct_prompt_mode not in {"raw", "chat"}:
            raise ValueError("instruct_prompt_mode must be raw or chat")
        if self.humaneval_full_completion and self.instruct_prompt_mode != "chat":
            raise ValueError(
                "humaneval_full_completion requires instruct_prompt_mode=chat"
            )
        self.require_target_numeric_coverage = bool(require_target_numeric_coverage)
        self.seed = int(seed)
        self.mask_id = int(mask_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.checkpoint_identity = {
            "path": spinquant_checkpoint["path"],
            "config_sha256": spinquant_checkpoint["config_sha256"],
            "safetensors_index_sha256": spinquant_checkpoint["index_sha256"],
        }
        with no_init_weights():
            self.model = LLaDAModelLM._from_config(
                LLaDAConfig.from_pretrained(self.model_path), torch_dtype=torch.bfloat16
            ).eval()
        self.model.to(self._device)
        self.artifact_manifest_sha256: Optional[str] = None
        self.spinquant_artifact_dir: Optional[Path] = None
        self.spinquant_variant = ""
        self.spinquant_execution = ""
        self.row_precision_context: Optional[RowPrecisionContext] = None
        self.row_precision_context = RowPrecisionContext()
        self.spinquant_artifact_dir = Path(spinquant_artifact_dir).resolve()
        self.spinquant_variant = "fixed-r1r2-r4-only"
        self.spinquant_execution = "joint-full-w4a8-v8"
        reader = SpinQuantArtifactReader(
            self.spinquant_artifact_dir,
            verify_files=False,
            expected_checkpoint=spinquant_checkpoint,
        )
        replaced = replace_with_spinquant_joint_full_w4a8_v8(
            self.model,
            self.spinquant_artifact_dir,
            variant="fixed-r1r2-r4-only",
            row_precision_context=self.row_precision_context,
            activation_bits=8,
        )
        if (
            self.row_precision_context is not None
            and int(replaced.get("dynamic_a4a8_linear", 0)) != 224
        ):
            raise ValueError("joint SpinQuant Feature 3 must bind all 224 block Linears")
        self.artifact_manifest_sha256 = reader.manifest_sha256
        expected_parameters = {
            "model.transformer.wte.weight",
            "model.transformer.ln_f.weight",
        }
        expected_parameters.update(
            (
                f"model.transformer.blocks.{layer}.{norm}.weight"
                for layer in range(32)
                for norm in ("attn_norm", "ff_norm")
            )
        )
        if set(dict(self.model.named_parameters())) != expected_parameters:
            raise RuntimeError("artifact installation left unexpected model parameters")
        if os.environ.get("SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC") == "1":
            if not triton_linear_numeric_available():
                raise RuntimeError(
                    "SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC=1 but the fused Linear kernels are unavailable"
                )
        if os.environ.get("SUPRA_REQUIRE_TRITON_OPERATOR_NUMERIC") == "1":
            if not triton_operator_numeric_available():
                raise RuntimeError(
                    "SUPRA_REQUIRE_TRITON_OPERATOR_NUMERIC=1 but the fused operator kernels are unavailable"
                )
        self.deployment_diagnostic = {}
        a4_clip_ratio = float(
            torch.tensor(float(resolved_a4_clip_ratio), dtype=torch.bfloat16)
        )
        a4_output_clip_ratio = (
            a4_clip_ratio
            if resolved_a4_output_clip_ratio is None
            else float(
                torch.tensor(float(resolved_a4_output_clip_ratio), dtype=torch.bfloat16)
            )
        )
        if not 0 < a4_clip_ratio <= 1 or not 0 < a4_output_clip_ratio <= 1:
            raise ValueError("A4 clipping ratio must be finite and in (0,1]")
        if a4_clip_ratio != 1.0:
            self.deployment_diagnostic["a4_clip_ratio_bf16"] = a4_clip_ratio
        if a4_output_clip_ratio != a4_clip_ratio:
            self.deployment_diagnostic[
                "a4_output_clip_ratio_bf16"
            ] = a4_output_clip_ratio
        if resolved_silu_table:
            from quantization.model import override_target_silu_table

            override_target_silu_table(
                self.model, json.loads(Path(resolved_silu_table).read_text())
            )
            self.deployment_diagnostic["silu_table"] = str(resolved_silu_table)
        if (
            a4_clip_ratio != 1.0
            or a4_output_clip_ratio != 1.0
            or self.feature1_cross_block_block_initialization_deep_clip_ratio != 0.0
        ):
            from quantization.model import install_target_a4_clipping

            install_target_a4_clipping(
                self.model, a4_clip_ratio, output_ratio=a4_output_clip_ratio
            )
        self.numeric_coverage: Optional[Dict[str, Any]] = None
        self.numeric_coverage_sha256: Optional[str] = None
        if self.spinquant_artifact_dir is not None:
            self.numeric_coverage = build_target_numeric_coverage(
                self.model,
                candidate_numeric_mode="bf16_lut",
                candidate_numeric_schedule=CANDIDATE_NUMERIC_SCHEDULE,
                candidate_suppression=CANDIDATE_SUPPRESSION_SCHEDULE,
                require_dynamic_a4a8=True,
                transformer_weight_arithmetic=reader.transformer_weight_arithmetic,
                rotation_source=f"spinquant_artifact:{self.spinquant_variant}:absorbed_r1_r2",
                rotation_sha256=reader.manifest_sha256,
            )
            if self.feature2_cache_initialization_activation_policy != "default":
                self.numeric_coverage["transformer_activation"][
                    "cache_initialization_activation_policy"
                ] = self.feature2_cache_initialization_activation_policy
            if a4_clip_ratio != 1.0 or a4_output_clip_ratio != 1.0:
                self.numeric_coverage["transformer_activation"].update(
                    scale_mode="dynamic_clipped_max_abs_per_token_row",
                    a4_clip_ratio_bf16=a4_clip_ratio,
                    clip_order="R4_if_present; A4 clamp to +/-BF16(row_max*ratio); existing dynamic quantization",
                    a8_clipping=False,
                )
                if a4_output_clip_ratio != a4_clip_ratio:
                    self.numeric_coverage["transformer_activation"].update(
                        a4_output_clip_ratio_bf16=a4_output_clip_ratio,
                        output_clip_modules=["attn_out", "ff_out"],
                    )
            coverage_payload = json.dumps(
                self.numeric_coverage, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
            self.numeric_coverage_sha256 = _sha256_bytes(coverage_payload)
        if self.require_target_numeric_coverage and (
            self.numeric_coverage is None or not self.numeric_coverage["valid"]
        ):
            errors = (
                ["target numeric coverage requires a SpinQuant artifact"]
                if self.numeric_coverage is None
                else self.numeric_coverage["errors"]
            )
            raise ValueError("target numeric coverage failed: " + "; ".join(errors))
        self.trace_output_path = (
            Path(trace_output_path).resolve() if trace_output_path else None
        )
        if self.trace_output_path is not None:
            self.trace_output_path = _rank_trace_path(
                self.trace_output_path, self.rank, self.world_size
            )
            self.trace_output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.trace_output_path.exists():
                raise FileExistsError(
                    f"Refusing to append to existing trace file: {self.trace_output_path}"
                )

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self.gen_length

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.model_path

    def tok_encode(self, string: str, **_: Any) -> List[int]:
        return self.tokenizer(string, add_special_tokens=True)["input_ids"]

    def tok_decode(self, tokens: Iterable[int], **_: Any) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    def loglikelihood(self, requests: Any) -> List[Any]:
        raise NotImplementedError(
            "Use a generate_until task for diffusion generation"
        )

    def loglikelihood_rolling(self, requests: Any) -> List[Any]:
        raise NotImplementedError(
            "Use a generate_until task for diffusion generation"
        )

    def _trim_stop_sequences(self, text: str, until: Any) -> str:
        stop_sequences = (
            [] if until is None else [until] if isinstance(until, str) else list(until)
        )
        positions = [
            text.find(stop) for stop in stop_sequences if stop and text.find(stop) >= 0
        ]
        return text[: min(positions)] if positions else text

    def _write_trace(
        self,
        request: Any,
        input_ids: torch.Tensor,
        tokens: torch.Tensor,
        nfe: int,
        trace: Any,
        feature2_stats: Any = None,
        state_capture_evidence: Optional[List[Dict[str, Any]]] = None,
        question_start_token: int = -1,
        generation_response: Optional[str] = None,
    ) -> None:
        if self.trace_output_path is None:
            return
        generated = tokens[:, input_ids.shape[1] :].detach().cpu().contiguous()
        compact_trace = [
            {
                "block_index": event.block_index,
                "step_index": event.step_index,
                "forward_kind": event.forward_kind,
                "input_start": event.input_start,
                "input_length": event.input_length,
                "input_positions": event.input_positions,
                "cache_refresh_due_positions_before": event.cache_refresh_due_positions_before,
                "cache_refresh_due_positions_after": event.cache_refresh_due_positions_after,
                "cache_sequence_length": event.cache_sequence_length,
                "transferred_positions": event.transferred_positions,
                "direct_locked_positions": event.direct_locked_positions,
                "selective_a4_direct_positions": event.selective_a4_direct_positions,
                "cross_block_handoff_changed_positions": event.cross_block_handoff_changed_positions,
                "stable_tentative_positions": event.stable_tentative_positions,
                "fallback_tentative_positions": event.fallback_tentative_positions,
                "confirmed_positions": event.confirmed_positions,
                "remasked_positions": event.remasked_positions,
                "handoff_verification_positions": event.handoff_verification_positions,
                "forced_finish_positions": event.forced_finish_positions,
                "tail_bypassed_positions": event.tail_bypassed_positions,
                "masked_before": event.masked_before,
                "masked_after": event.masked_after,
                "tentative_before": event.tentative_before,
                "tentative_after": event.tentative_after,
                "nfe": event.nfe,
                "a4_rows": event.a4_rows,
                "a8_rows": event.a8_rows,
                "masked_a4_rows": event.masked_a4_rows,
                "tentative_a8_rows": event.tentative_a8_rows,
                "young_locked_a8_rows": event.young_locked_a8_rows,
                "mature_locked_a4_rows": event.mature_locked_a4_rows,
                "layer0_rows": event.layer0_rows,
                "cache_initialization_activation_policy": event.cache_initialization_activation_policy,
                "linear_activation_rows": event.linear_activation_rows,
                "layer0_keep_global_cache": event.layer0_keep_global_cache,
                "boundary_global_layers": event.boundary_global_layers,
                "boundary_deep_bits": event.boundary_deep_bits,
                "layer0_current_row_bits": event.layer0_current_row_bits,
                "boundary_deep_clip_ratio": event.boundary_deep_clip_ratio,
                "layer0_a4_rows": getattr(event, "layer0_a4_rows", 0),
                "layer0_a8_rows": getattr(event, "layer0_a8_rows", 0),
                "prediction_rows": event.prediction_rows,
                "refresh_rows": event.refresh_rows,
                "active_rows": event.active_rows,
                "early_exit_rows": event.early_exit_rows,
                "mandatory_refresh_rows": event.mandatory_refresh_rows,
                "optional_refresh_rows": event.optional_refresh_rows,
                "dependency_rows_updated": event.dependency_rows_updated,
                "dependency_table_bytes": event.dependency_table_bytes,
                "pending_dependency_rows": event.pending_dependency_rows,
                "changed_positions": event.changed_positions,
                "mandatory_refresh_positions": event.mandatory_refresh_positions,
                "optional_refresh_positions": event.optional_refresh_positions,
                "dependency_score_by_row": event.dependency_score_by_row,
                "changed_confidence_by_position": getattr(
                    event, "changed_confidence_by_position", ()
                ),
                "cross_block_changed_confidence_by_position": getattr(
                    event, "cross_block_changed_confidence_by_position", ()
                ),
                "dependency_shape": event.dependency_shape,
                "dependency_dtype": event.dependency_dtype,
                "dependency_entries_read": event.dependency_entries_read,
                "direct_dependency_entries_read": getattr(
                    event, "direct_dependency_entries_read", 0
                ),
                "change_confidence_entries_read": getattr(
                    event, "change_confidence_entries_read", 0
                ),
                "change_risk_subtractions": getattr(
                    event, "change_risk_subtractions", 0
                ),
                "change_risk_multiplications": getattr(
                    event, "change_risk_multiplications", 0
                ),
                "causal_union_additions": getattr(event, "causal_union_additions", 0),
                "remask_status_entries_read": getattr(
                    event, "remask_status_entries_read", 0
                ),
                "joint_relation_multiplications": getattr(
                    event, "joint_relation_multiplications", 0
                ),
                "dependency_entries_written": event.dependency_entries_written,
                "selector_candidate_rows": event.selector_candidate_rows,
                "top_budget_comparisons": event.top_budget_comparisons,
                "regular_row_budget_target_active_rows": getattr(
                    event, "regular_row_budget_target_active_rows", 0.0
                ),
                "regular_row_budget_allowed_active_rows": getattr(
                    event, "regular_row_budget_allowed_active_rows", 0
                ),
                "regular_row_budget_regular_steps": getattr(
                    event, "regular_row_budget_regular_steps", 0
                ),
                "regular_row_budget_cumulative_active_rows": getattr(
                    event, "regular_row_budget_cumulative_active_rows", 0
                ),
                "cross_block_selected_positions": list(
                    getattr(event, "cross_block_selected_positions", ())
                ),
                "cross_block_ranked_positions": list(
                    getattr(event, "cross_block_ranked_positions", ())
                ),
                "boundary_context_selected_positions": list(
                    getattr(event, "boundary_context_selected_positions", ())
                ),
                "boundary_context_selected_scores": list(
                    getattr(event, "boundary_context_selected_scores", ())
                ),
                "block_initialization_score_floor_bf16": event.block_initialization_score_floor_bf16,
                "block_initialization_score_scan_rows": event.block_initialization_score_scan_rows,
                "boundary_context_candidate_rows": getattr(
                    event, "boundary_context_candidate_rows", 0
                ),
                "boundary_context_score_entries_read": getattr(
                    event, "boundary_context_score_entries_read", 0
                ),
                "boundary_context_topk_comparisons": getattr(
                    event, "boundary_context_topk_comparisons", 0
                ),
                "refresh_start": event.refresh_start,
                "refresh_end": event.refresh_end,
                "region_start": event.region_start,
                "region_end": event.region_end,
                "refresh_positions": event.refresh_positions,
                "prediction_positions": event.prediction_positions,
                "active_a4_rows": event.active_a4_rows,
                "active_a8_rows": event.active_a8_rows,
                **(
                    {
                        "next_progress_positions": event.next_progress_positions,
                        "next_added_positions": event.next_added_positions,
                        "next_admission_eligible_a_tokens": event.next_admission_eligible_a_tokens,
                        "next_admission_eligible_b_tokens": event.next_admission_eligible_b_tokens,
                        "next_admission_quota": event.next_admission_quota,
                        "next_admitted_positions": event.next_admitted_positions,
                        "next_direct_locked_positions": event.next_direct_locked_positions,
                        "next_tentative_positions": event.next_tentative_positions,
                        "next_confirmed_positions": event.next_confirmed_positions,
                        "next_remasked_positions": event.next_remasked_positions,
                        "current_required_rows": event.current_required_rows,
                        "next_progress_rows": event.next_progress_rows,
                        "next_added_rows": event.next_added_rows,
                        "joint_rows": event.joint_rows,
                        "joint_issued_rows": event.joint_issued_rows,
                        "joint_issued_slice_units": event.joint_issued_slice_units,
                        "joint_pe_issue_groups": event.joint_pe_issue_groups,
                        "joint_a4_rows": event.joint_a4_rows,
                        "joint_a8_rows": event.joint_a8_rows,
                        "qkv_activation_bytes": event.qkv_activation_bytes,
                        "mixed_k4096_activation_bytes": event.mixed_k4096_activation_bytes,
                        "ffn_down_activation_bytes": event.ffn_down_activation_bytes,
                        "next_step_ffn_down_activation_bytes": event.next_step_ffn_down_activation_bytes,
                        "base_weight_read_multiplier": event.base_weight_read_multiplier,
                        "max_weight_read_multiplier": event.max_weight_read_multiplier,
                        "feature3_extra_weight_read_multiplier": event.feature3_extra_weight_read_multiplier,
                        "buffer_valid": event.buffer_valid,
                        "next_rejected_for_buffer": event.next_rejected_for_buffer,
                        "next_rejected_for_current_step_capacity": event.next_rejected_for_current_step_capacity,
                        "next_rejected_for_verification_capacity": event.next_rejected_for_verification_capacity,
                        "next_rejected_for_budget": event.next_rejected_for_budget,
                        "source_b_optional_proposed_rows": event.source_b_optional_proposed_rows,
                        "source_b_optional_proposed_a8_rows": event.source_b_optional_proposed_a8_rows,
                        "source_b_optional_reject_reason": event.source_b_optional_reject_reason,
                    }
                    if self.generation_mode == "feature1_packed_feature2_fused_dynamic_block"
                    else {}
                ),
            }
            for event in trace
        ]
        if state_capture_evidence is not None:
            _merge_state_capture_evidence(compact_trace, state_capture_evidence)
        precision = f"spinquant-joint-w4a8-q8k8-lut-p8-v8-int32pv-k8v8cache-lmhead-w8a8-{self.spinquant_variant}"
        row: Dict[str, Any] = {
            "schema_version": "feature3-dynamic-joint-window-llada-eval-trace/v2"
            if self.generation_mode == "feature1_packed_feature2_fused_dynamic_block"
            else "feature1-packed-feature2-fused-llada-eval-trace/v1"
            if self.generation_mode == "feature1_packed_feature2_fused_a4a8"
            else "feature2-feature3-state-a4a8-llada-eval-trace/v1",
            "candidate_numeric_schedule": CANDIDATE_NUMERIC_SCHEDULE,
            "candidate_suppression_schedule": CANDIDATE_SUPPRESSION_SCHEDULE,
            "numeric_coverage": self.numeric_coverage,
            "numeric_coverage_sha256": self.numeric_coverage_sha256,
            "generation_mode": self.generation_mode,
            "decoding": {
                "mode": self.decoding_mode,
                "k": self.decode_k if self.decoding_mode == "fixed_k" else None,
                "threshold": self.decode_threshold
                if self.decoding_mode == "fixed_threshold"
                else None,
                "feature2_enabled": not self.fixed_decoding,
                "activation_policy": "all_a8_full_sequence"
                if self.generation_mode == "baseline_fixed_k"
                else "regular_a8_with_refresh_overrides"
                if self.fixed_decoding
                else "state_a4a8",
            },
            "instruct_prompt_mode": self.instruct_prompt_mode,
            "humaneval_full_completion": self.humaneval_full_completion,
            "confidence_eos_eot_inf": self.confidence_eos_eot_inf,
            "task_name": request.task_name,
            "question_start_token": question_start_token,
            "doc_id": request.doc_id,
            "request_index": request.idx,
            "precision": precision,
            "logical_a4a8": False,
            "spinquant_variant": self.spinquant_variant,
            "spinquant_execution": self.spinquant_execution,
            "v_cache_dtype": getattr(self.model, "_spinquant_v_cache_dtype", None)
            if self.spinquant_artifact_dir is not None
            else None,
            "r4_tensor_rotation_count_per_nfe": 32,
            "attention_int8_matmul": "triton-batched-int8-int32"
            if triton_int8_bmm_available()
            else "per-head-reference-int8-int32",
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "deployment_diagnostic": self.deployment_diagnostic,
            "lm_head_weight_bits": getattr(
                self.model, "_spinquant_lm_head_weight_bits", None
            ),
            "prompt_token_count": int(input_ids.shape[1]),
            "seed": self.seed,
            "mask_id": self.mask_id,
            "generation_protocol": {
                "steps": self.steps,
                "gen_length": self.gen_length,
                "block_length": 32,
                "temperature": 0.0,
                "remasking": "low_confidence",
                "batch_size": self.batch_size,
            },
            "generated_token_sha256": _sha256_bytes(generated.numpy().tobytes()),
            "generated_token_ids": generated.tolist(),
            "generation_response": generation_response,
            "nfe": int(nfe),
            "trace": compact_trace,
        }
        row["generation_protocol"]["evaluation_protocol"] = "lm_eval"
        row["generation_protocol"]["reseed_per_request"] = False
        row["feature1_parameters"] = {
            "scope": "tri",
            "dependency_layer_mode": "all",
            "dependency_dtype": "bfloat16",
            "selection": "descending_score_under_cumulative_active_row_budget",
            "target_active_rows": self.feature1_tri_target_active_rows,
            "initial_burst_rows": 0.0,
            "proposal_scope": "current_and_next_block"
            if self.generation_mode == "feature1_packed_feature2_fused_dynamic_block"
            else "current_block_only",
            "execution": "packed sparse arbitrary-position queries and K/V scatter",
            "cross_block_boundary_scope": "layer0_attention_two_stage",
            "cross_block_boundary_target_rows": self.feature1_cross_block_boundary_target_rows,
            "cross_block_full_prefix_oracle_block": 1,
            "cross_block_block_initialization_deep_rows": self.feature1_cross_block_block_initialization_deep_rows,
            "cross_block_block_initialization_dependency_only": self.feature1_cross_block_block_initialization_dependency_only,
            "cross_block_block_initialization_relative_score_floor": self.feature1_cross_block_block_initialization_relative_score_floor,
            "cross_block_block_initialization_include_future_dependency": self.feature1_cross_block_block_initialization_include_future_dependency,
            "cross_block_block_initialization_protected_future_blocks": self.feature1_cross_block_block_initialization_protected_future_blocks,
            "cross_block_block_initialization_deep_bits": self.feature1_cross_block_block_initialization_deep_bits,
            "cross_block_block_initialization_deep_clip_ratio": self.feature1_cross_block_block_initialization_deep_clip_ratio,
            "cross_block_block_initialization_dependency_tiebreak": self.feature1_cross_block_block_initialization_dependency_tiebreak,
            "cross_block_block_initialization_context_bits": 8,
            "cross_block_block_initialization_protect_question": self.feature1_cross_block_block_initialization_protect_question,
            "cross_block_block_initialization_protect_generation": self.feature1_cross_block_block_initialization_protect_generation,
            "cross_block_block_initialization_all_a8": self.feature1_cross_block_block_initialization_all_a8,
            "cross_block_block_initialization_keep_global_l0_cache": self.feature1_cross_block_block_initialization_keep_global_l0_cache,
            "cross_block_block_initialization_global_layers": 1,
            "cross_block_boundary_deep_a8_row_limit": self.feature1_cross_block_boundary_deep_a8_row_limit,
            "cross_block_boundary_context_row_bits": self.cross_block_boundary_context_row_bits,
            "cross_block_pending_confidence_mode": self.cross_block_pending_confidence_mode,
            "cross_block_pending_relation_mode": self.cross_block_pending_relation_mode,
            "cross_block_dependency_policy": self.cross_block_dependency_policy,
        }
        row["feature2_precision_parameters"] = {
            "context_a8_rows": self.feature2_context_a8_rows,
            "cache_initialization_activation_policy": self.feature2_cache_initialization_activation_policy,
            "block0_full_sequence_future_bits": 4
            if self.feature2_cache_initialization_activation_policy == "a4"
            else 8,
            "block0_full_sequence_prefix_bits": 4
            if self.feature2_cache_initialization_activation_policy == "a4"
            else 8,
        }
        if self.generation_mode == "feature1_packed_feature2_fused_dynamic_block":
            row["feature3_dynamic_block_parameters"] = {
                "logical_block_rows": 32,
                "target_joint_rows": self.feature3_dynamic_block_target_joint_rows,
                "max_next_rows": self.feature3_dynamic_block_max_next_rows,
                "source_b_max_attempts": self.feature3_dynamic_block_source_b_max_attempts,
                "source_b_retry_min_confidence": self.feature3_dynamic_block_source_b_retry_min_confidence,
                "source_b_a4_only": self.feature3_dynamic_block_source_b_a4_only,
                "source_b_dependency_tie_rank": self.feature3_dynamic_block_source_b_dependency_tie_rank,
                "source_a_confirm_at_handoff": self.feature3_dynamic_block_source_a_confirm_at_handoff,
                "target_prediction_rows": self.feature3_dynamic_block_target_prediction_rows,
                "next_admission_budget": self.feature3_dynamic_block_next_admission_budget,
                "min_reuse_score": self.feature3_dynamic_block_min_reuse_score,
                "next_candidate_policy": self.feature3_dynamic_block_next_candidate_policy,
                "next_relation_preference": self.feature3_dynamic_block_next_relation_preference,
                "max_current_unresolved": self.feature3_dynamic_block_max_current_unresolved,
                "max_handoff_verification_rows": self.feature3_dynamic_block_max_handoff_verification_rows,
                "canonical_future": self.feature3_dynamic_block_canonical_future,
                "future_horizon": self.feature3_dynamic_block_future_horizon,
                "canonical_direct_tau": self.feature3_dynamic_block_canonical_direct_tau,
                "allow_deferred_verification": self.feature3_dynamic_block_allow_deferred_verification,
                "next_min_confidence": self.feature3_dynamic_block_next_min_confidence,
                "activation_capacity_bytes": 393216,
                "k12288_activation_capacity_bytes": 393216,
            }
        if self.generation_mode == "baseline_fixed_k":
            row["feature1_parameters"] = {
                "enabled": False,
                "execution": "full_sequence_recompute",
                "proposal_scope": "current_block_unresolved_rows",
            }
        row["doc_hash"] = hash_string(
            json.dumps(
                request.doc,
                indent=2,
                default=handle_non_serializable,
                ensure_ascii=False,
            )
        )
        if feature2_stats is not None:
            row["generation_mode"] = self.generation_mode
            row["source_commit"] = self.source_commit
            row["checkpoint"] = dict(self.checkpoint_identity)
            row["feature2_parameters"] = dict(self.feature2_parameters)
            row["feature2_stats"] = feature2_stats.to_dict()
            state_bits = {
                "all_a8": (8, 8),
                "mature_only": (8, 4),
                "masked_only": (4, 8),
                "original": (4, 4),
            }["all_a8" if self.fixed_decoding else "original"]
            row["feature3_parameters"] = {
                "maturity_age": 3,
                "precision_policy": "all_a8" if self.fixed_decoding else "original",
                "masked_bits": state_bits[0],
                "tentative_bits": 8,
                "young_locked_bits": 8,
                "mature_locked_bits": state_bits[1],
            }
        with self.trace_output_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            )

    @torch.no_grad()
    def generate_until(self, requests: Any, disable_tqdm: bool = False) -> List[str]:
        outputs: List[str] = []
        completed_requests: Dict[int, str] = {}
        for request in requests:
            request_identity = id(request)
            if self.world_size > 1 and request_identity in completed_requests:
                outputs.append(completed_requests[request_identity])
                continue
            (context, generation_kwargs) = request.args
            requested_temperature = float(generation_kwargs.get("temperature", 0.0))
            if requested_temperature != 0.0 or generation_kwargs.get(
                "do_sample", False
            ):
                raise ValueError(
                    "generation requires temperature=0 and do_sample=false"
                )
            requested_max = int(generation_kwargs.get("max_gen_toks", self.gen_length))
            if requested_max != self.gen_length:
                raise ValueError(
                    f"Task requested max_gen_toks={requested_max}; configured generation requires {self.gen_length}"
                )
            if self.instruct_prompt_mode == "chat":
                messages = [{"role": "user", "content": context}]
                rendered_context = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                rendered_context = context
            encoded = self.tokenizer(
                rendered_context,
                return_tensors="pt",
                add_special_tokens=True,
                return_offsets_mapping=self.feature1_cross_block_block_initialization_protect_question,
            )
            input_ids = encoded.input_ids.to(self._device)
            question_start = (
                _question_start_token(
                    request, rendered_context, encoded.offset_mapping[0].tolist()
                )
                if self.feature1_cross_block_block_initialization_protect_question
                else -1
            )
            suppressed_candidate_token_ids = ()
            if self.confidence_eos_eot_inf:
                eot_token_id = int(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
                if eot_token_id < 0 or eot_token_id == int(
                    self.tokenizer.unk_token_id or -1
                ):
                    raise ValueError(
                        "Instruct confidence suppression requires a valid <|eot_id|> token"
                    )
                suppressed_candidate_token_ids = (
                    int(self.tokenizer.eos_token_id),
                    eot_token_id,
                )
            if int(input_ids.shape[1]) + self.gen_length > self._max_length:
                raise ValueError(
                    f"prompt length {int(input_ids.shape[1])} + gen_length {self.gen_length} exceeds max_length {self._max_length}"
                )
            state_capture_evidence: Optional[List[Dict[str, Any]]] = (
                []
                if os.environ.get("SUPRA_DIAGNOSTIC_STATE_CAPTURE", "0") == "1"
                else None
            )
            (tokens, nfe, trace, feature2_stats) = generate(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                decoding_mode=self.decoding_mode,
                decode_k=self.decode_k,
                decode_threshold=self.decode_threshold,
                feature3_precision_policy="all_a8" if self.fixed_decoding else "original",
                mask_id=self.mask_id,
                tau_high=self.feature2_parameters["tau_high"],
                tau_high_tail=self.feature2_parameters["tau_high_tail"],
                tau_high_tail_after_step=self.feature2_parameters[
                    "tau_high_tail_after_step"
                ],
                tau_low=self.feature2_parameters["tau_low"],
                confirm_tau=self.feature2_parameters["confirm_tau"],
                selective_a4_direct_tau=self.feature2_parameters["a4_direct_tau"],
                budget_scale=self.feature2_parameters["budget_scale"],
                stability_bonus=self.feature2_parameters["stability_bonus"],
                row_precision_context=self.row_precision_context,
                cache_initialization_activation_policy=self.feature2_cache_initialization_activation_policy,
                tail_confirmation_policy=self.feature2_parameters[
                    "tail_confirmation_policy"
                ],
                suppressed_candidate_token_ids=suppressed_candidate_token_ids,
                state_capture_callback=None
                if state_capture_evidence is None
                else lambda event: state_capture_evidence.append(
                    _compact_state_capture(event)
                ),
                packed_attention_refresh=self.generation_mode
                in {
                    "feature1_packed_feature2_fused_a4a8",
                    "feature1_packed_feature2_fused_dynamic_block",
                    "feature1_fixed_k",
                    "feature1_fixed_threshold",
                },
                packed_attention_layer_mode="all",
                packed_attention_target_active_rows=self.feature1_tri_target_active_rows,
                packed_attention_initial_burst_rows=0.0,
                packed_attention_context_a8_rows=self.feature2_context_a8_rows,
                full_sequence_recompute=self.generation_mode == "baseline_fixed_k",
                cross_block_cache_handoff=self.generation_mode != "baseline_fixed_k",
                cross_block_boundary_scope="tri" if self.generation_mode == "baseline_fixed_k" else "layer0_attention_two_stage",
                cross_block_boundary_target_rows=self.feature1_cross_block_boundary_target_rows,
                cross_block_full_prefix_oracle_block=-1 if self.generation_mode == "baseline_fixed_k" else 1,
                cross_block_block_initialization_deep_rows=self.feature1_cross_block_block_initialization_deep_rows,
                cross_block_block_initialization_dependency_only=self.feature1_cross_block_block_initialization_dependency_only,
                cross_block_block_initialization_relative_score_floor=self.feature1_cross_block_block_initialization_relative_score_floor,
                cross_block_block_initialization_include_future_dependency=self.feature1_cross_block_block_initialization_include_future_dependency,
                cross_block_block_initialization_protected_future_blocks=self.feature1_cross_block_block_initialization_protected_future_blocks,
                cross_block_block_initialization_deep_bits=self.feature1_cross_block_block_initialization_deep_bits,
                cross_block_block_initialization_deep_clip_ratio=self.feature1_cross_block_block_initialization_deep_clip_ratio,
                cross_block_block_initialization_dependency_tiebreak=self.feature1_cross_block_block_initialization_dependency_tiebreak,
                question_start_token=question_start,
                cross_block_block_initialization_protect_generation=self.feature1_cross_block_block_initialization_protect_generation,
                cross_block_block_initialization_all_a8=self.feature1_cross_block_block_initialization_all_a8,
                cross_block_block_initialization_keep_global_l0_cache=self.feature1_cross_block_block_initialization_keep_global_l0_cache,
                cross_block_boundary_deep_a8_row_limit=self.feature1_cross_block_boundary_deep_a8_row_limit,
                cross_block_boundary_context_row_bits=self.cross_block_boundary_context_row_bits,
                cross_block_pending_confidence_mode=self.cross_block_pending_confidence_mode,
                cross_block_pending_relation_mode="direct",
                cross_block_dependency_policy=self.cross_block_dependency_policy,
                dynamic_block_lookahead=self.generation_mode
                == "feature1_packed_feature2_fused_dynamic_block",
                dynamic_block_target_joint_rows=self.feature3_dynamic_block_target_joint_rows,
                dynamic_block_max_next_rows=self.feature3_dynamic_block_max_next_rows,
                dynamic_block_source_b_max_attempts=self.feature3_dynamic_block_source_b_max_attempts,
                dynamic_block_source_b_retry_min_confidence=self.feature3_dynamic_block_source_b_retry_min_confidence,
                dynamic_block_source_b_a4_only=self.feature3_dynamic_block_source_b_a4_only,
                dynamic_block_source_b_dependency_tie_rank=self.feature3_dynamic_block_source_b_dependency_tie_rank,
                dynamic_block_source_a_confirm_at_handoff=self.feature3_dynamic_block_source_a_confirm_at_handoff,
                dynamic_block_target_prediction_rows=self.feature3_dynamic_block_target_prediction_rows,
                dynamic_block_next_admission_budget=self.feature3_dynamic_block_next_admission_budget,
                dynamic_block_min_reuse_score=self.feature3_dynamic_block_min_reuse_score,
                dynamic_block_next_candidate_policy=self.feature3_dynamic_block_next_candidate_policy,
                dynamic_block_next_relation_preference=self.feature3_dynamic_block_next_relation_preference,
                dynamic_block_max_current_unresolved=self.feature3_dynamic_block_max_current_unresolved,
                dynamic_block_max_handoff_verification_rows=self.feature3_dynamic_block_max_handoff_verification_rows,
                dynamic_block_canonical_future=self.feature3_dynamic_block_canonical_future,
                dynamic_block_future_horizon=self.feature3_dynamic_block_future_horizon,
                dynamic_block_canonical_direct_tau=self.feature3_dynamic_block_canonical_direct_tau,
                dynamic_block_allow_deferred_verification=self.feature3_dynamic_block_allow_deferred_verification,
                dynamic_block_next_min_confidence=self.feature3_dynamic_block_next_min_confidence,
            )
            if self.fixed_decoding:
                if any(
                    (
                        (event.forward_kind == "local_block" and event.active_a4_rows)
                        or event.tentative_after
                        or event.remasked_positions
                        for event in trace
                    )
                ):
                    raise RuntimeError(
                        "fixed decoding contains regular A4 or reversible transitions"
                    )
            generated_ids = tokens[:, input_ids.shape[1] :][0]
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            is_humaneval = (
                str(request.doc.get("task_id", "")).lower().startswith("humaneval")
            )
            output = (
                decoded
                if self.humaneval_full_completion and is_humaneval
                else self._trim_stop_sequences(decoded, generation_kwargs.get("until"))
            )
            self._write_trace(
                request,
                input_ids,
                tokens,
                nfe,
                trace,
                feature2_stats,
                state_capture_evidence,
                question_start_token=question_start,
                generation_response=output,
            )
            self.cache_hook.add_partial(
                "generate_until", (context, generation_kwargs), output
            )
            outputs.append(output)
            if self.world_size > 1:
                completed_requests[request_identity] = output
        return outputs


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
