#!/usr/bin/env python3
"""Translate supported algorithm parameters to DDR scalar fields.

Binary memory preparation remains in prepare_head_testcase.py,
prepare_handoff_testcase.py and integration/prepare_layer_testcase.py.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
from pathlib import Path
import struct

TYPE_SIZE = {"u8": 1, "u16": 2, "u32": 4, "u48": 6, "u64": 8, "i16": 2}


def integer(value: object, name: str) -> int:
    if isinstance(value, bool) or (isinstance(value, float) and (not math.isfinite(value) or not value.is_integer())):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value, 0) if isinstance(value, str) else int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    return result


def pack_schema(schema: dict, values: dict, name: str) -> bytes:
    consume_current_begin = None
    if schema.get("schema") == "supra-execution-config/v4":
        flags = integer(values.get("flags", 0), "execution.flags")
        if flags & 0x60 == 0x60:
            raise ValueError("four-batch and six-batch FFN flags are mutually exclusive")
        if flags & 0x60 and not flags & 0x10:
            raise ValueError("four-batch or six-batch FFN requires repeat grouping")
        if flags & 0x80 and (not flags & 0x60 or not flags & 0x10):
            raise ValueError("fused FFN product requires repeated four-batch or six-batch grouping")
        if flags & 0x100 and not flags & 0x80:
            raise ValueError("paired Down requires fused FFN product")
        if flags & 0x1000 and flags & 0xe00:
            raise ValueError("Q/K/V/O grouping cannot be combined with Attention or KV pairing")
    if schema.get("schema") == "supra-draft-verify-block-config/v1":
        flags = integer(values.get("flags", 0), "block.flags")
        handoff = (flags >> 8) & 63
        if handoff > 32 or (handoff and not flags & 2):
            raise ValueError("handoff limit 1..32 requires canonical future")
        if flags & 0x80 and not flags & 2:
            raise ValueError("Source A handoff admission flag requires canonical future")
        if flags & 0x40 and flags & 0x3e:
            raise ValueError("stable-only tail cannot combine with future, transfer, closeout or all-tail")
        if flags & 0x20 and (flags != 0x20 or integer(values.get("observed_mask", 0), "observed_mask")):
            raise ValueError("transfer-only cannot combine with Feature2 flags or an observed mask")
    if schema.get("schema") == "supra-uaps-attempt-config/v1":
        control = integer(values.get("state_control", 0), "joint.state_control")
        current = integer(values.get("max_current_unresolved", 0), "joint.max_current_unresolved")
        handoff = integer(values.get("max_handoff_tokens", 0), "joint.max_handoff_tokens")
        admission = integer(values.get("admission_budget", 0), "joint.admission_budget")
        if (control & 1 and (not 1 <= current <= 32 or not 1 <= handoff <= 32)) or (
                not control & 1 and (control or current or handoff or admission)):
            raise ValueError("live-state control and current/handoff/admission limits disagree")
    if schema.get("schema") == "supra-refresh-score-config/v1":
        flags = integer(values.get("flags", 0), "refresh_score.flags")
        mode = (flags >> 3) & 3
        if (flags & 32) and (not flags & 1 or flags & 4):
            raise ValueError("block advance requires cross-block pending without regular token-table update")
        if mode == 3 or (mode and not flags & 1):
            raise ValueError("ordinary confidence mode requires cross-block refresh and mode 0..2")
        if flags & 0xc0 and not flags & 1:
            raise ValueError("live consumption requires cross-block refresh")
        if flags & 128:
            if flags & 96 or integer(values.get("regular_budget_base", 0), "regular_budget_base"):
                raise ValueError("initial-key boundary consumption excludes advance, KV-consume and regular budget")
            values = dict(values)
            consume_current_begin = integer(values.pop("consume_current_begin", None), "consume_current_begin")
            rows = integer(values.get("token_count", 0), "token_count")
            if not 0 <= consume_current_begin <= rows-32:
                raise ValueError("initial-key boundary current32-token range must fit token_count")
    if schema.get("schema") == "supra-token-refresh-config/v1":
        flags = integer(values.get("flags", 0), "refresh.flags")
        pending = integer(values.get("pending_configuration_offset", 0), "refresh.pending_configuration_offset")
        if flags & 128 and (not pending or flags & 1 or flags & 6 != 6):
            raise ValueError("paired pending requires two records, embedding sources and metadata output, without cache commit")
        context, deep = bool(flags & 8), bool(flags & 32)
        quota = (flags >> 8) & 127
        if flags & 64:
            base = integer(values.get("metadata_base", 0), "refresh.metadata_base")
            limit = integer(values.get("metadata_limit", 0), "refresh.metadata_limit")
            if not flags & 2 or limit % 16 or limit <= base or limit-base <= 16:
                raise ValueError("selection result requires metadata output and an aligned reserved final beat")
        if quota > 96 or (context and deep) or (context and not pending) or (
                deep and (pending or flags & 4 or not flags & 2)) or (
                not context and (flags & 16 or (not deep and quota))):
            raise ValueError("refresh context/deep precision flags or pending configuration conflict")
    image = bytearray(schema["size_bytes"])
    known = {field["name"] for field in schema["fields"]}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    for field in schema["fields"]:
        field_name = field["name"]
        if "constant" in field:
            value = integer(field["constant"], f"{name}.{field_name}.constant")
            if field_name in values and integer(values[field_name], field_name) != value:
                raise ValueError(f"{name}.{field_name} must equal {value}")
        elif field_name in values:
            value = integer(values[field_name], f"{name}.{field_name}")
        elif "default" in field:
            value = integer(field["default"], f"{name}.{field_name}.default")
        else:
            raise ValueError(f"{name} is missing field {field_name}")
        width = TYPE_SIZE[field["type"]]
        signed = field["type"].startswith("i")
        minimum = -(1 << (width * 8 - 1)) if signed else 0
        maximum = (1 << (width * 8 - 1)) - 1 if signed else\
            (1 << (width * 8)) - 1
        if value < minimum or value > maximum:
            raise ValueError(f"{name}.{field_name} does not fit {field['type']}")
        if "minimum" in field and value < integer(field["minimum"], field_name):
            raise ValueError(f"{name}.{field_name} is below its minimum")
        if "maximum" in field and value > integer(field["maximum"], field_name):
            raise ValueError(f"{name}.{field_name} is above its maximum")
        if "valid_mask" in field and value & ~integer(field["valid_mask"], field_name):
            raise ValueError(f"{name}.{field_name} has unsupported bits")
        if "allowed_values" in field and value not in {
                integer(item, field_name) for item in field["allowed_values"]}:
            raise ValueError(f"{name}.{field_name} has an unsupported value")
        if "alignment" in field and value % integer(field["alignment"], field_name):
            raise ValueError(f"{name}.{field_name} is not aligned")
        image[field["offset"]:field["offset"] + width] = value.to_bytes(
            width, "little", signed=signed)
    if consume_current_begin is not None:
        extension = schema["boundary_consumption"]
        if len(image) != extension["offset"]:
            raise ValueError("boundary consumption suffix does not follow its base descriptor")
        image.extend(bytes(extension["size_bytes"]))
        image[6:8] = len(image).to_bytes(2, "little")
        image[extension["offset"]:extension["offset"]+2] = consume_current_begin.to_bytes(2, "little")
    return bytes(image)


def patch_record(image, offset, schema, values):
    """Validate the completed record, then update its selected fields in place."""
    size = schema["size_bytes"]
    if offset < 0 or offset + size > len(image):
        raise ValueError("record extends beyond the memory image")
    fields = {field["name"]: field for field in schema["fields"]}
    current = {}
    for name, field in fields.items():
        start, width = offset + field["offset"], TYPE_SIZE[field["type"]]
        current[name] = int.from_bytes(image[start:start + width], "little",
                                      signed=field["type"].startswith("i"))
    current.update(values)
    packed = pack_schema(schema, current, schema.get("schema", "record"))
    for name in values:
        field = fields[name]
        start, width = field["offset"], TYPE_SIZE[field["type"]]
        image[offset + start:offset + start + width] = packed[start:start + width]


def encode_admission_budget(value: float) -> int:
    """Equivalent BF16 multiplier for block32 and the accepted scale [16,32].

    For base quota 1, ceil(scale) is exact. For base quota >=2 both
    multipliers saturate at the at-most-32 active positions. Rounding the
    original decimal to BF16 first would incorrectly admit 20 for 20.01.
    """
    value = float(value)
    if not math.isfinite(value) or not 16 <= value <= 32:
        raise ValueError("Feature2 budget scale must be in [16, 32] for block32")
    return struct.unpack("<I", struct.pack("<f", float(math.ceil(value))))[0] >> 16


def deployment_fields(config: dict, tier: str = "feature123", *, block_step_index: int | None = None) -> dict:
    """Encode algorithm settings and request-specific scalar fields."""
    if tier not in ("baseline", "feature1", "feature12", "feature123"):
        raise ValueError("unknown deployment tier")
    supplied = config.get("model_args", config)
    release = Path(__file__).resolve().parents[2]
    algorithm = release / "algorithm/llada"
    model = ast.parse((algorithm / "evaluation/model.py").read_text())
    evaluator = next(node for node in model.body if isinstance(node, ast.ClassDef)
                     and node.name == "QuantizedLLaDALM")
    constructor = next(node for node in evaluator.body if isinstance(node, ast.FunctionDef)
                       and node.name == "__init__")
    known = {arg.arg for arg in (*constructor.args.args, *constructor.args.kwonlyargs)} - {"self"}
    unknown = set(supplied) - known
    if unknown:
        raise ValueError(f"unknown model_args fields: {sorted(unknown)}")
    default_nodes = list(zip(constructor.args.args[-len(constructor.args.defaults):],
                             constructor.args.defaults))
    default_nodes.extend((arg, value) for arg, value in
                         zip(constructor.args.kwonlyargs, constructor.args.kw_defaults)
                         if value is not None)
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in default_nodes}
    args = {**defaults, **supplied}
    mode = args["generation_mode"]
    fixed_decoding = mode in ("baseline_fixed_k", "feature1_fixed_k")
    if mode not in ("feature1_packed_feature2_fused_dynamic_block", "feature1_packed_feature2_fused_a4a8", "feature1_fixed_k", "baseline_fixed_k"):
        raise ValueError(f"generation_mode has no DDR conversion: {mode!r}")
    if fixed_decoding and tier != ("baseline" if mode == "baseline_fixed_k" else "feature1"):
        raise ValueError("fixed-k deployment tier must match its generation_mode")
    if tier in ("baseline", "feature1") and not 1 <= integer(supplied.get("decode_k", 3), "decode_k") <= 32:
        raise ValueError("fixed-k deployment requires decode_k in 1..32")
    if mode == "feature1_packed_feature2_fused_a4a8" and tier == "feature123":
        raise ValueError("Feature12 input cannot enable Feature3")
    supported = dict(feature3_precision_policy="all_a8" if fixed_decoding else "original",
        feature1_tri_neighbor_scope="tri", feature1_dependency_layer_mode="all",
        feature1_cross_block_boundary_scope="tri" if mode == "baseline_fixed_k" else "layer0_attention_two_stage",
        feature3_maturity_age=3, cross_block_pending_relation_mode="direct")
    if tier == "feature123":
        supported.update(feature3_dynamic_block_canonical_future=True,
            feature3_dynamic_block_future_horizon=1,
            feature3_dynamic_block_next_candidate_policy="high_stable",
            feature3_dynamic_block_next_relation_preference="low_dependency")
    for key, value in supported.items():
        if args[key] != value:
            raise ValueError(f"{key}={args[key]!r}; DDR conversion expects {value!r}")

    def bf16(value):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("BF16 scalar must be finite and nonnegative")
        raw = struct.unpack("<I", struct.pack("<f", value))[0]
        return (raw + 0x7fff + ((raw >> 16) & 1)) >> 16

    for key, expected in (("feature2_tau_high", 0x3f66), ("feature2_a4_direct_tau", 0x3f66),
                          ("feature2_tau_low", 0x3f40), ("feature2_confirm_tau", 0x3f40)):
        if not fixed_decoding and bf16(args[key]) != expected:
            raise ValueError(f"hardware fixed threshold mismatch: {key}")
    low = bf16(args["feature2_tau_low"])
    if tier == "feature123" and any(bf16(args[key]) != low for key in (
            "feature3_dynamic_block_canonical_direct_tau", "feature3_dynamic_block_next_min_confidence")):
        raise ValueError("canonical future thresholds must equal low_confidence_threshold")
    target_token_count_x2 = float(args["feature1_tri_target_active_rows"]) * 2
    context = integer(args["feature2_context_a8_rows"], "context A8 tokens")
    deep = integer(args["feature1_cross_block_boundary_deep_a8_row_limit"], "deep A8 tokens")
    if not target_token_count_x2.is_integer() or not 48 <= target_token_count_x2 <= 88 or not 0 <= context <= 16 or not -1 <= deep <= 48:
        raise ValueError("token budget requires target_active_rows=24..44, context_a8_rows=0..16 and deep_a8_row_limit=-1..48")
    normal = args["a4_clip_ratio"]
    normal = 1.0 if normal is None else float(normal)
    output = args["a4_output_clip_ratio"]
    output = normal if output is None else float(output)
    if not 0.75 <= normal <= 1 or not 0.60 <= output <= 1:
        raise ValueError("clipping requires a4_clip_ratio=0.75..1 and a4_output_clip_ratio=0.60..1")
    normal_clip, output_clip = bf16(normal), bf16(output)
    clips = dict.fromkeys(("query", "key", "value", "ffn_gate", "ffn_up"), normal_clip)
    clips.update(attention_output=output_clip, ffn_down=output_clip)
    tail = float(args["feature2_tau_high_tail"])
    tail_step = integer(args["feature2_tau_high_tail_after_step"], "tail step")
    policy = args["feature2_tail_confirmation_policy"]
    if tail != -1 and (not 0.75 <= tail <= 0.90 or not 4 <= tail_step <= 16):
        raise ValueError("tail threshold/step is outside the supported range")
    if policy not in ("none", "stable_only", "all"):
        raise ValueError("unknown tail confirmation policy")
    bonus = float(args["feature2_stability_bonus"])
    if not 0 <= bonus <= 0.10:
        raise ValueError("stability bonus must be in [0, 0.10]")
    context_bits = integer(args["cross_block_boundary_context_row_bits"], "boundary context bits")
    if context_bits not in (4, 8):
        raise ValueError("boundary context bits must be 4 or 8")
    target = integer(args["feature1_cross_block_boundary_target_rows"], "deep target")
    if target < 48 or target > 96 or target % 8:
        raise ValueError("boundary target must be 48..96 with step 8")
    confidence_modes = {"all_changes": 0, "stable_unmask": 1, "remask_only": 2}
    if args["cross_block_pending_confidence_mode"] not in confidence_modes:
        raise ValueError("unknown cross-block pending confidence mode")
    dependency_policy = args["cross_block_dependency_policy"]
    if dependency_policy not in ("initial_keys_selected_rows", "current_keys_committed_rows"):
        raise ValueError("unknown cross-block dependency policy")
    if dependency_policy == "initial_keys_selected_rows" and args["feature1_cross_block_block_initialization_relative_score_floor"] != 0:
        raise ValueError("initial-key dependencies require zero block initialization score floor")
    cross_flags = 3 | (confidence_modes[args["cross_block_pending_confidence_mode"]] << 3)
    block_initialization = {key[len("feature1_cross_block_block_initialization_"):]: value
        for key, value in args.items() if key.startswith("feature1_cross_block_block_initialization_")}
    deep_tokens = integer(block_initialization["deep_rows"], "block initialization deep tokens")
    deep_bits = integer(block_initialization["deep_bits"], "block initialization deep bits")
    future_blocks = integer(block_initialization["protected_future_blocks"], "protected future blocks")
    floor, deep_clip = float(block_initialization["relative_score_floor"]), float(block_initialization["deep_clip_ratio"])
    if args["feature2_cache_initialization_activation_policy"] not in ("default", "a4"):
        raise ValueError("cache initialization activation policy must be default or a4")
    if block_initialization["global_layers"] != 1 or (deep_tokens and (not 32 <= deep_tokens <= 432)):
        raise ValueError("block initialization requires one full L0 and 32..432 deep rows, or zero for full sequence")
    if not 0 <= floor <= .25 or (floor and not block_initialization["dependency_only"]):
        raise ValueError("block initialization floor must be0..0.25 and requires dependency-only")
    if block_initialization["dependency_only"] and not block_initialization["dependency_tiebreak"]:
        raise ValueError("dependency-only block initialization requires dependency tiebreak")
    if block_initialization["include_future_dependency"] and not block_initialization["dependency_only"]:
        raise ValueError("future dependency requires dependency-only block initialization")
    if future_blocks not in (-1, 0, 1, 2) or (future_blocks >= 0 and not block_initialization["protect_generation"]):
        raise ValueError("future protection requires generation protection and -1/0/1/2 blocks")
    if deep_tokens:
        if not block_initialization["all_a8"] or deep_bits not in (4, 8):
            raise ValueError("sparse block initialization requires full A8 L0 and uniform A4/A8 deep execution")
    elif any(block_initialization[key] for key in ("dependency_only", "relative_score_floor", "include_future_dependency",
            "deep_bits", "deep_clip_ratio", "dependency_tiebreak", "protect_question", "protect_generation",
            "all_a8", "keep_global_l0_cache")) or future_blocks != -1:
        raise ValueError("full block initialization requires inactive sparse options")
    if (deep_clip != 0 and not .75 <= deep_clip <= 1) or (deep_clip and deep_bits != 4):
        raise ValueError("deep clipping requires A4 and zero or0.75..1 ratio")
    deep_clips = dict.fromkeys(clips, bf16(deep_clip)) if deep_clip else clips
    fields = dict(
        post_block=dict(flags=1, high_confidence_threshold_bf16=bf16(args["feature2_tau_high"]),
            tail_high_confidence_threshold_bf16=bf16(tail if tail >= 0 else args["feature2_tau_high"]),
            low_confidence_threshold_bf16=low, verify_threshold_bf16=bf16(args["feature2_confirm_tau"]),
            stability_bonus_bf16=bf16(bonus), budget_scale_bf16=encode_admission_budget(args["feature2_budget_scale"])),
        feature2_block=dict(flags=int(tail >= 0) | {"none": 0, "all": 4, "stable_only": 64}[policy],
            tail_after_step=tail_step if tail >= 0 else 0, maturity_age=3),
        refresh_regular=dict(flags=6 | 8 | (context << 8)),
        refresh_cross_block=dict(flags=cross_flags,
            boundary_flags=cross_flags | (32 if dependency_policy == "current_keys_committed_rows" else 0),
            completed_forward_flags=cross_flags | (64 if dependency_policy == "current_keys_committed_rows" else 0),
            selected_boundary_flags=cross_flags | (64 if dependency_policy == "current_keys_committed_rows" else 128)),
        regular_budget=dict(target_token_count_x2=int(target_token_count_x2), selection_control=0),
        refresh_boundary=dict(flags=2 | (32 | (deep << 8) if deep >= 0 else 0),
            target_token_count=target, a8_limit=deep, context_activation_bits=context_bits),
        joint=dict(target_token_count=48, max_next_tokens=0, priority_control=0xc0, scheduled_quota=0, future_block_flags=2),
        attempt=dict(enabled=False, max_attempts=0),
        cache_initialization=dict(all_a4=args["feature2_cache_initialization_activation_policy"] == "a4"),
        block_initialization=dict(target_token_count=deep_tokens,
            l0_activation_bits=8, deep_activation_bits=deep_bits if deep_tokens else 8,
            keep_global_l0_cache=bool(block_initialization["keep_global_l0_cache"]) if deep_tokens else True,
            protect_question=bool(block_initialization["protect_question"]),
            protect_generation=bool(block_initialization["protect_generation"]), protected_future_blocks=future_blocks,
            shortlist_flags=(4 | int(block_initialization["include_future_dependency"]) |
                (2 if block_initialization["dependency_only"] else 0) | (8 if block_initialization["dependency_tiebreak"] else 0)) if deep_tokens else 0,
            relative_score_floor_bf16=bf16(floor),
            layer=dict(flags=1, **{key + "_clip_ratio_bf16": value for key, value in deep_clips.items()})),
        layer=dict(flags=1, **{key + "_clip_ratio_bf16": value for key, value in clips.items()}))
    if tier == "feature123":
        attempts = integer(args["feature3_dynamic_block_source_b_max_attempts"], "new future-token admission attempts")
        retry = float(args["feature3_dynamic_block_source_b_retry_min_confidence"])
        reuse = float(args["feature3_dynamic_block_min_reuse_score"])
        if not -1 <= attempts <= 4 or not 0 <= retry <= 0.5 or not 0 <= reuse <= 1:
            raise ValueError("future attempt limit/retry threshold is out of range")
        fields["attempt"] = dict(enabled=attempts >= 0 or retry > 0,
            max_attempts=attempts if attempts >= 0 else (0x7fffffff if retry > 0 else 0))
        fields["joint"].update(target_token_count=integer(args["feature3_dynamic_block_target_joint_rows"], "joint tokens"),
            max_next_tokens=integer(args["feature3_dynamic_block_max_next_rows"], "next tokens"),
            scheduled_quota=integer(args["feature3_dynamic_block_next_admission_budget"], "future admission budget"),
            future_block_flags=2 | (128 if args["feature3_dynamic_block_source_a_confirm_at_handoff"] else 0))
        max_current = integer(args["feature3_dynamic_block_max_current_unresolved"], "current unresolved limit")
        max_handoff = integer(args["feature3_dynamic_block_max_handoff_verification_rows"], "future handoff limit")
        quota = fields["joint"]["scheduled_quota"]
        if not 32 <= fields["joint"]["target_token_count"] <= 48 or not 1 <= fields["joint"]["max_next_tokens"] <= 32 or not 1 <= quota <= 16:
            raise ValueError("joint target32..48, next tokens1..32, and admission budget1..16 are required")
        if not 1 <= max_current <= 32 or not 1 <= max_handoff <= 32 or not 0 <= quota <= 32:
            raise ValueError("current/handoff/admission limits exceed one block")
        fields["joint"]["future_block_flags"] |= max_handoff << 8
        prediction_target = integer(args["feature3_dynamic_block_target_prediction_rows"], "prediction target")
        if prediction_target != -1 and not 16 <= prediction_target <= 48:
            raise ValueError("prediction target must be -1 or16..48")
        b_rank = bool(args.get("feature3_dynamic_block_source_b_dependency_tie_rank", False))
        b_a4 = bool(args.get("feature3_dynamic_block_source_b_a4_only", False))
        if b_rank and block_step_index is None:
            raise ValueError("Source B dependency tie rank requires actual block_step_index")
        step = 0 if block_step_index is None else integer(block_step_index, "block step index")
        if not 0 <= step <= 65535:
            raise ValueError("block step index must fit uint16")
        fields["joint_extension"] = dict(source_b_flags=int(b_rank) | (int(b_a4) << 1),
            block_step_index=step,retry_min_confidence_bf16=bf16(retry), min_reuse_score_bf16=bf16(reuse),
            prediction_target=0 if prediction_target == -1 else prediction_target,
            max_current_unresolved=max_current, max_handoff_tokens=max_handoff, admission_budget=quota,
            state_control=1 | (2 if args["feature3_dynamic_block_allow_deferred_verification"] else 0))
    result = apply_execution_tier(fields, tier)
    if tier in ("baseline", "feature1"):
        result["execution"]["decode_k"] = int(supplied.get("decode_k", 3))
    return result


def apply_execution_tier(preencoded_fields: dict, tier: str) -> dict:
    """Select the active DDR scalar fields for a cumulative feature tier."""
    if tier not in ("baseline", "feature1", "feature12", "feature123"):
        raise ValueError("unknown deployment tier")
    source_tier = preencoded_fields.get("execution", {}).get("tier", "feature123")
    if source_tier != "feature123" and source_tier != tier:
        raise ValueError("tier translation requires the original Feature123 scalar configuration")
    fields = copy.deepcopy(preencoded_fields)
    if tier != "feature123":
        fields["joint"] = dict(target_token_count=48, max_next_tokens=0, priority_control=0xc0,
                               scheduled_quota=0, future_block_flags=0)
        fields["attempt"] = dict(enabled=False, max_attempts=0)
        fields.pop("joint_extension", None)
    if tier in ("baseline", "feature1"):
        fields["feature2_block"]["flags"] = 0x20
        fields["refresh_regular"]["flags"] = 0x1e
        fields["post_block"].update(flags=0, stability_bonus_bf16=0, budget_scale_bf16=0x3f80)
    if tier == "baseline":
        fields["refresh_boundary"].update(flags=2, a8_limit=-1)
        fields["layer"] = {key: 0 for key in fields["layer"]}
        if "cache_initialization" in fields:
            fields["cache_initialization"]["all_a4"] = False
        if "block_initialization" in fields:
            fields["block_initialization"].update(target_token_count=0, l0_activation_bits=8, deep_activation_bits=8,
                keep_global_l0_cache=True, protect_question=False, protect_generation=False,
                protected_future_blocks=-1, shortlist_flags=0, relative_score_floor_bf16=0,
                layer={key: 0 for key in fields["block_initialization"]["layer"]})
    fields["execution"] = dict(tier=tier, fixed_a8=tier == "baseline",
        regular_fixed_a8=tier in ("baseline", "feature1"),
        refresh_enabled=tier != "baseline", joint_enabled=tier == "feature123",
        full_sequence_each_step=tier == "baseline",
        boundary_kind="full_sequence_recompute" if tier == "baseline" else "layer0_scout_deep")
    if tier in ("baseline", "feature1"):
        fields["execution"]["decode_k"] = preencoded_fields.get("execution", {}).get("decode_k", 3)
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="algorithm JSON containing model_args, or a model_args object")
    parser.add_argument("--tier", choices=("baseline", "feature1", "feature12", "feature123"), default="feature123")
    parser.add_argument("--block-step-index", type=int, help="actual zero-based iteration within the current block")
    args = parser.parse_args()
    try:
        fields = deployment_fields(json.loads(args.config.read_text()), args.tier, block_step_index=args.block_step_index)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(fields, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
