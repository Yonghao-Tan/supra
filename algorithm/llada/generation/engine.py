"""LLaDA generation with dependency refresh, reversible admission and lookahead."""

from __future__ import annotations
from generation.records import (
    ForwardCaptureEvent,
    GenerationStats,
    GenerationTraceEvent,
)
from dataclasses import replace
import inspect
import math
from typing import Any, Callable, List, Sequence, Tuple
import torch
from quantization.model import using_target_block_a4_clip_ratio
from numerics.candidate import (
    candidate_action_confidence,
    candidate_token_probability_bf16,
    streaming_candidate_bf16,
)
from generation.refresh import (
    _bf16_state_mul,
    CrossBlockPrefixAttentionState,
    TriBlockAttentionState,
    begin_boundary_layer0_scout,
    begin_tri_block_attention_monitor,
    configure_tri_block_attention_monitor,
    end_boundary_layer0_scout,
    end_tri_block_attention_monitor,
    prepare_tri_block_attention_step,
    read_boundary_layer0_scout,
    read_tri_block_attention_profile,
)
from generation.common import (
    PastKeyValues,
    _cache_metadata,
    _require_cached_output,
    get_num_transfer_tokens,
)
from generation.state import (
    Feature2BlockState,
    LOCKED,
    MASKED,
    TENTATIVE,
    select_admissions,
)
from generation.lookahead import (
    ActivationBufferProfile,
    ActivationResidency,
    DynamicJointWindowScheduler,
)
from numerics.precision import RowPrecisionContext


def _locate_unique_positions(
    packed_positions: torch.Tensor, selected_positions: torch.Tensor
) -> torch.Tensor:
    if packed_positions.ndim != 1 or selected_positions.ndim != 1:
        raise ValueError("packed and selected positions must be rank one")
    if selected_positions.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=packed_positions.device)
    matches = selected_positions[:, None] == packed_positions[None, :]
    if bool((matches.sum(dim=1) != 1).any()):
        raise RuntimeError("selected positions are not unique in the packed input")
    return matches.to(torch.int64).argmax(dim=1)


def _snapshot_cache_rows(
    past_key_values: PastKeyValues, positions: torch.Tensor
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Copy selected K/V cache rows before an already-scheduled refresh."""
    if positions.ndim != 1 or positions.dtype != torch.long:
        raise ValueError("cache error positions must be a rank-1 long tensor")
    return tuple(
        (
            tuple((field.index_select(-2, positions).clone() for field in layer))
            for layer in past_key_values
        )
    )


def _restore_cache_rows(
    before: tuple[tuple[torch.Tensor, ...], ...],
    after: PastKeyValues,
    positions: torch.Tensor,
) -> int:
    """Restore every persistent K/V field for selected query rows."""
    if len(before) != len(after):
        raise ValueError("cache snapshot layer count changed")
    restored = 0
    for old_layer, new_layer in zip(before, after):
        if len(old_layer) != len(new_layer) or len(new_layer) not in {2, 3}:
            raise ValueError("cache snapshot field layout changed")
        for old, new in zip(old_layer, new_layer):
            new.index_copy_(-2, positions, old)
        restored += int(positions.numel())
    return restored


def _select_boundary_layer0_deep_positions(
    score_q8: torch.Tensor,
    *,
    current_positions: torch.Tensor,
    target_rows: int,
    eligible_positions: torch.Tensor | None = None,
    mandatory_candidate_positions: torch.Tensor | None = None,
    required_candidate_positions: torch.Tensor | None = None,
    required_candidate_count: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the current block plus attention-ranked optional rows."""
    if score_q8.ndim != 1 or score_q8.dtype != torch.uint8:
        raise ValueError("boundary Layer0 score must be a rank-1 P8 vector")
    if (
        current_positions.ndim != 1
        or current_positions.dtype != torch.long
        or current_positions.device != score_q8.device
        or (current_positions.numel() == 0)
        or (current_positions.unique().numel() != current_positions.numel())
    ):
        raise ValueError("boundary Layer0 current positions must be unique rows")
    if not int(current_positions.numel()) <= target_rows <= score_q8.numel():
        raise ValueError("boundary Layer0 target rows are outside the sequence")
    if eligible_positions is None:
        all_positions = torch.arange(
            score_q8.numel(), dtype=torch.long, device=score_q8.device
        )
        candidates = all_positions[~torch.isin(all_positions, current_positions)]
    else:
        if (
            eligible_positions.ndim != 1
            or eligible_positions.dtype != torch.long
            or eligible_positions.device != score_q8.device
            or (eligible_positions.unique().numel() != eligible_positions.numel())
            or (
                eligible_positions.numel()
                and bool(
                    (
                        (eligible_positions < 0)
                        | (eligible_positions >= score_q8.numel())
                        | torch.isin(eligible_positions, current_positions)
                    ).any()
                )
            )
        ):
            raise ValueError("boundary Layer0 eligible positions are invalid")
        candidates = eligible_positions
    optional_rows = target_rows - int(current_positions.numel())
    if candidates.numel() < optional_rows:
        raise ValueError("boundary Layer0 candidates cannot fill the row target")
    if mandatory_candidate_positions is None:
        mandatory_candidate_positions = candidates[:0]
    if (
        mandatory_candidate_positions.ndim != 1
        or mandatory_candidate_positions.dtype != torch.long
        or mandatory_candidate_positions.device != score_q8.device
        or (
            mandatory_candidate_positions.unique().numel()
            != mandatory_candidate_positions.numel()
        )
        or (
            mandatory_candidate_positions.numel()
            and (not bool(torch.isin(mandatory_candidate_positions, candidates).all()))
        )
        or (mandatory_candidate_positions.numel() > optional_rows)
    ):
        raise ValueError("boundary Layer0 mandatory candidates are invalid")
    if required_candidate_positions is None:
        required_candidate_positions = candidates[:0]
    if (
        required_candidate_positions.ndim != 1
        or required_candidate_positions.dtype != torch.long
        or required_candidate_positions.device != score_q8.device
        or (
            required_candidate_positions.unique().numel()
            != required_candidate_positions.numel()
        )
        or (
            required_candidate_positions.numel()
            and (not bool(torch.isin(required_candidate_positions, candidates).all()))
        )
        or (
            not 0
            <= required_candidate_count
            <= min(optional_rows, int(required_candidate_positions.numel()))
        )
    ):
        raise ValueError("boundary Layer0 required candidates are invalid")
    mandatory_required_count = int(
        torch.isin(mandatory_candidate_positions, required_candidate_positions)
        .sum()
        .item()
    )
    required_candidate_positions = required_candidate_positions[
        ~torch.isin(required_candidate_positions, mandatory_candidate_positions)
    ]
    required_candidate_count = max(
        0, required_candidate_count - mandatory_required_count
    )
    remaining_optional_rows = optional_rows - int(mandatory_candidate_positions.numel())
    if (
        not 0
        <= required_candidate_count
        <= min(remaining_optional_rows, int(required_candidate_positions.numel()))
    ):
        raise ValueError("boundary Layer0 required candidates are invalid")
    order = torch.argsort(
        score_q8.index_select(0, candidates), descending=True, stable=True
    )
    ranked = candidates.index_select(0, order)
    required_order = torch.argsort(
        score_q8.index_select(0, required_candidate_positions),
        descending=True,
        stable=True,
    )
    required_selected = required_candidate_positions.index_select(
        0, required_order[:required_candidate_count]
    )
    remaining = ranked[
        ~torch.isin(
            ranked, torch.cat((mandatory_candidate_positions, required_selected))
        )
    ]
    selected_optional = torch.cat(
        (
            mandatory_candidate_positions,
            required_selected,
            remaining[
                : optional_rows
                - int(mandatory_candidate_positions.numel())
                - required_candidate_count
            ],
        )
    )
    deep_positions = torch.sort(
        torch.cat((current_positions, selected_optional))
    ).values
    if deep_positions.numel() != target_rows:
        raise RuntimeError("boundary Layer0 selection did not fill the row target")
    return (deep_positions, selected_optional, ranked)


def _layer0_global_then_attention_guided_deep(
    model: Any,
    row_precision_context: RowPrecisionContext | None,
    *,
    tokens: torch.Tensor,
    stale_cache: PastKeyValues,
    current_positions: torch.Tensor,
    current_row_bits: torch.Tensor | None,
    target_rows: int,
    deep_candidate_positions: torch.Tensor | None = None,
    mandatory_candidate_positions: torch.Tensor | None = None,
    required_candidate_positions: torch.Tensor | None = None,
    required_candidate_count: int = 0,
    transition_positions: torch.Tensor | None = None,
    global_context_bits: int = 4,
    global_prefix_bits: int = 4,
    deep_a8_row_limit: int = -1,
    deep_uniform_bits: int = 0,
    deep_clip_ratio: float = 0.0,
    replay_deep_row_bits: torch.Tensor | None = None,
    keep_global_layer0_cache: bool = False,
) -> tuple[
    torch.Tensor,
    PastKeyValues,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Run full Layer0, select deep rows from fresh attention, then run Layers1-31."""
    if tokens.shape[0] != 1 or current_positions.ndim != 1:
        raise ValueError(
            "Layer0 scout boundary requires batch one and rank-1 positions"
        )
    if (
        current_row_bits is not None
        and current_row_bits.shape != current_positions.shape
    ):
        raise ValueError("Layer0 scout current row bits do not align")
    if global_context_bits not in {4, 8} or global_prefix_bits not in {4, 8}:
        raise ValueError("Layer0 scout context and prefix bits must be 4 or 8")
    if deep_uniform_bits not in (0, 4, 8) or (
        deep_uniform_bits and row_precision_context is None
    ):
        raise ValueError("uniform deep precision requires quantization and bits 4 or 8")
    if not 0 <= deep_clip_ratio <= 1 or (deep_clip_ratio and deep_uniform_bits != 4):
        raise ValueError("deep clipping override requires uniform A4 and ratio in (0,1]")
    if deep_a8_row_limit < -1 or (
        deep_a8_row_limit >= 0 and replay_deep_row_bits is not None
    ):
        raise ValueError("invalid deep A8 limit or simultaneous replay override")
    backbone = model.model
    total_length = int(tokens.shape[1])
    global_positions = torch.arange(
        total_length, dtype=torch.long, device=tokens.device
    )
    global_bits = None
    if row_precision_context is not None:
        if current_row_bits is None:
            raise ValueError("quantized Layer0 scout requires current row bits")
        global_bits = torch.full(
            (total_length,), global_context_bits, dtype=torch.int8, device=tokens.device
        )
        global_bits[: int(current_positions.min().item())].fill_(global_prefix_bits)
        global_bits.index_copy_(0, current_positions, current_row_bits)
    x = backbone.transformer.wte(tokens)
    if backbone.config.input_emb_norm:
        x = x * backbone.config.d_model**0.5
    x = backbone.transformer.emb_drop(x)
    cache_input = tuple(
        (tuple((field.clone() for field in layer)) for layer in stale_cache)
    )
    present: list[tuple[torch.Tensor, ...]] = []
    if row_precision_context is not None:
        row_precision_context.activate(global_bits)
    begin_boundary_layer0_scout(
        backbone,
        current_positions=current_positions,
        total_length=total_length,
        transition_positions=transition_positions,
    )
    try:
        (x, layer_cache) = backbone.transformer.blocks[0](
            x,
            attention_bias=None,
            layer_past=cache_input[0],
            use_cache=True,
            query_position_ids=global_positions.unsqueeze(0),
            kv_write_position_ids=global_positions.unsqueeze(0),
        )
        (score_q8, probability_entries) = read_boundary_layer0_scout(backbone)
    finally:
        end_boundary_layer0_scout(backbone)
    if layer_cache is None:
        raise RuntimeError("Layer0 scout boundary produced no Layer0 cache")
    (
        deep_positions,
        selected_optional,
        ranked,
    ) = _select_boundary_layer0_deep_positions(
        score_q8,
        current_positions=current_positions,
        target_rows=target_rows,
        eligible_positions=deep_candidate_positions,
        mandatory_candidate_positions=mandatory_candidate_positions,
        required_candidate_positions=required_candidate_positions,
        required_candidate_count=required_candidate_count,
    )
    rejected = global_positions[~torch.isin(global_positions, deep_positions)]
    if rejected.numel() and not keep_global_layer0_cache:
        before = _snapshot_cache_rows((stale_cache[0],), rejected)
        _restore_cache_rows(before, (layer_cache,), rejected)
    present.append(layer_cache)
    x = x.index_select(1, deep_positions)
    deep_bits = (
        None if global_bits is None else global_bits.index_select(0, deep_positions)
    )
    if deep_uniform_bits:
        deep_bits = torch.full_like(deep_bits, deep_uniform_bits)
    if deep_bits is not None and replay_deep_row_bits is not None:
        if (
            replay_deep_row_bits.shape != deep_bits.shape
            or replay_deep_row_bits.dtype != torch.int8
            or replay_deep_row_bits.device != deep_bits.device
            or bool(((replay_deep_row_bits != 4) & (replay_deep_row_bits != 8)).any())
        ):
            raise ValueError("captured deep precision does not match selected rows")
        in_current = torch.isin(deep_positions, current_positions)
        if not torch.equal(replay_deep_row_bits[in_current], deep_bits[in_current]):
            raise ValueError("captured deep precision changes current-row bits")
        deep_bits = replay_deep_row_bits
    elif deep_bits is not None and deep_a8_row_limit >= 0:
        protected_positions = (
            current_positions
            if mandatory_candidate_positions is None
            else torch.cat((current_positions, mandatory_candidate_positions))
        )
        deep_bits = boundary_deep_precision_bits(
            deep_bits,
            score_q8.index_select(0, deep_positions),
            torch.isin(deep_positions, protected_positions),
            deep_a8_row_limit,
        )
    if row_precision_context is not None:
        row_precision_context.activate(deep_bits)
    for layer_index, block in enumerate(backbone.transformer.blocks[1:], start=1):
        with using_target_block_a4_clip_ratio(block, deep_clip_ratio):
            (x, layer_cache) = block(
                x,
                attention_bias=None,
                layer_past=cache_input[layer_index],
                use_cache=True,
                query_position_ids=deep_positions.unsqueeze(0),
                kv_write_position_ids=deep_positions.unsqueeze(0),
            )
        if layer_cache is None:
            raise RuntimeError("Layer0 scout boundary produced no deep-layer cache")
        present.append(layer_cache)
    x = backbone.transformer.ln_f(x)
    logits = backbone.transformer.ff_out(x)
    if backbone.config.scale_logits:
        logits.mul_(1 / math.sqrt(backbone.config.d_model))
    return (
        logits,
        tuple(present),
        deep_positions,
        deep_bits,
        selected_optional,
        score_q8.index_select(0, selected_optional),
        probability_entries,
    )


def boundary_deep_precision_bits(row_bits, scores, protected, a8_limit):
    """Keep required precision, then spend remaining A8 rows by scout rank."""
    if (
        row_bits.ndim != 1
        or row_bits.dtype != torch.int8
        or scores.shape != row_bits.shape
        or (protected.shape != row_bits.shape)
        or (protected.dtype != torch.bool)
        or (a8_limit < 0)
        or (scores.device != row_bits.device)
        or (protected.device != row_bits.device)
        or bool(((row_bits != 4) & (row_bits != 8)).any())
    ):
        raise ValueError("invalid boundary deep precision inputs")
    result = row_bits.clone()
    optional_a8 = torch.nonzero((row_bits == 8) & ~protected, as_tuple=False).flatten()
    available = max(0, a8_limit - int(((row_bits == 8) & protected).sum().item()))
    if optional_a8.numel() > available:
        order = torch.argsort(
            scores.index_select(0, optional_a8), descending=True, stable=True
        )
        result[optional_a8.index_select(0, order[available:])] = 4
    return result


def block_initialization_refresh_positions(
    total_length, block_end, block_length, future_blocks, mandatory, start=0
):
    """Refresh the selected interval plus every outstanding required row."""
    if (
        future_blocks < -1
        or not 0 <= start < block_end <= total_length
        or block_length <= 0
    ):
        raise ValueError("invalid block initialization position bounds")
    if (
        mandatory.ndim != 1
        or mandatory.dtype != torch.long
        or bool(((mandatory < 0) | (mandatory >= total_length)).any())
    ):
        raise ValueError("invalid mandatory block initialization positions")
    stop = (
        total_length
        if future_blocks == -1
        else min(total_length, block_end + future_blocks * block_length)
    )
    prefix = torch.arange(start, stop, dtype=torch.long, device=mandatory.device)
    if start == 0 and stop == total_length:
        return prefix
    return torch.unique(torch.cat((prefix, mandatory)), sorted=True)


def block_initialization_relative_selection(ranked_positions, ranked_scores, mandatory, current_rows, row_cap, relative_floor):
    """Keep required rows and truncate the existing optional dependency ranking."""
    if (ranked_scores.dtype != torch.bfloat16 or ranked_scores.shape != ranked_positions.shape
            or not 0 < relative_floor <= 1 or current_rows + mandatory.numel() > row_cap):
        raise ValueError("relative block initialization selection requires BF16 scores and a sufficient row cap")
    eligible = ~torch.isin(ranked_positions, mandatory)
    optional = ranked_positions[eligible]
    values = ranked_scores[eligible]
    capacity = row_cap - current_rows - mandatory.numel()
    maximum = values.max() if values.numel() else ranked_scores.new_zeros(())
    cutoff = _bf16_state_mul(maximum, relative_floor)
    # All-zero observations do not establish that the cache needs no refresh.
    count = min(capacity, int((values >= cutoff).sum())) if float(maximum) else min(capacity, optional.numel())
    selected = torch.cat((mandatory, optional[:count]))
    return selected, current_rows + selected.numel(), float(cutoff), optional.numel()


def block_initialization_precision_bits(
    positions, prompt_length, block_end, mandatory, context_bits, prefix_a8_start=-1
):
    """Retain A8 for generated history/current and required cache refreshes."""
    if context_bits not in (4, 8) or not 0 <= prompt_length < block_end:
        raise ValueError("invalid block initialization precision configuration")
    if (
        positions.ndim != 1
        or positions.dtype != torch.long
        or mandatory.dtype != torch.long
        or (mandatory.ndim != 1)
    ):
        raise ValueError("block initialization precision requires rank-one logical positions")
    if prefix_a8_start < -1 or prefix_a8_start >= prompt_length:
        raise ValueError("block initialization protected prefix starts outside the prompt")
    protected = (positions >= prompt_length) & (positions < block_end) | torch.isin(
        positions, mandatory
    )
    if prefix_a8_start >= 0:
        protected |= (positions >= prefix_a8_start) & (positions < prompt_length)
    return torch.where(protected, 8, context_bits).to(torch.int8)


def _positions(selection: torch.Tensor, offset: int) -> Tuple[int, ...]:
    if selection.shape[0] != 1:
        return ()
    return tuple(
        (
            offset + int(index)
            for index in torch.nonzero(selection[0]).flatten().tolist()
        )
    )


def _restrict_cross_block_profile_queries(
    profile: dict[str, Any], *, query_end: int
) -> dict[str, Any]:
    positions = profile.get("prefix_dependency_query_positions")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
        raise RuntimeError("cross-block relation query positions are invalid")
    keep = torch.nonzero(positions < query_end, as_tuple=False).flatten()
    if keep.numel() == positions.numel():
        return profile
    result = dict(profile)
    result["prefix_dependency_query_positions"] = positions.index_select(0, keep)
    for name in (
        "prefix_dependency_mean",
        "prefix_dependency_last",
        "prefix_future_dependency_mean",
    ):
        value = profile.get(name)
        if isinstance(value, torch.Tensor):
            if value.ndim < 2 or value.shape[1] != positions.numel():
                raise RuntimeError(
                    f"cross-block relation field {name} has invalid query rows"
                )
            result[name] = value.index_select(1, keep)
    return result


def _local_positions(selection: torch.Tensor, offset: int) -> Tuple[int, ...]:
    return tuple(
        (offset + int(index) for index in torch.nonzero(selection).flatten().tolist())
    )


def context_a8_upgrade_count(
    row_bits: torch.Tensor,
    context_candidate_count: int,
    *,
    fixed_context_a8_rows: int,
) -> int:
    """Return the number of context rows to upgrade to A8."""
    if min(context_candidate_count, fixed_context_a8_rows) < 0:
        raise ValueError("context A8 row counts must be nonnegative")
    if row_bits.ndim != 1 or bool(((row_bits != 4) & (row_bits != 8)).any()):
        raise ValueError("row_bits must be a rank-1 tensor containing only 4 or 8")
    return min(int(context_candidate_count), int(fixed_context_a8_rows))


def select_irreversible_transfers(masked, confidence, *, mode, k=3, threshold=0.9):
    """Select masked positions without stability bonuses or tentative admission."""
    if (
        masked.ndim != 2
        or masked.dtype != torch.bool
        or confidence.shape != masked.shape
    ):
        raise ValueError("expected matching [batch, positions] mask and confidence")
    if mode not in {"fixed_k", "fixed_threshold"}:
        raise ValueError("unsupported irreversible decoding mode")
    if k < 1 or not 0.0 <= threshold <= 1.0:
        raise ValueError("k must be positive and threshold in [0,1]")
    score = confidence.float().masked_fill(~masked, -float("inf"))
    order = torch.argsort(score, dim=-1, descending=True, stable=True)
    selected = torch.zeros_like(masked)
    if mode == "fixed_k":
        selected.scatter_(1, order[:, : min(k, masked.shape[1])], True)
    else:
        selected = masked & (score >= threshold)
        selected.scatter_(1, order[:, :1], True)
    return selected & masked


@torch.no_grad()
def generate(
    model: Any,
    prompt: torch.Tensor,
    *,
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    decoding_mode: str = "feature2",
    decode_k: int = 3,
    decode_threshold: float = 0.9,
    mask_id: int = 126336,
    tau_high: float = 0.86,
    tau_high_tail: float | None = None,
    tau_high_tail_after_step: int = 0,
    tau_low: float = 0.68,
    confirm_tau: float = 0.68,
    selective_a4_direct_tau: float = -1.0,
    budget_scale: float = 16.0,
    stability_bonus: float = 0.05,
    row_precision_context: RowPrecisionContext | None = None,
    cache_initialization_activation_policy: str = "default",
    feature3_maturity_age: int = 3,
    feature3_precision_policy: str = "original",
    tail_confirmation_policy: str = "none",
    candidate_numeric_mode: str = "bf16_lut",
    suppressed_candidate_token_ids: Sequence[int] = (),
    full_sequence_recompute: bool = False,
    packed_attention_refresh: bool = False,
    packed_attention_neighbor_scope: str = "tri",
    packed_attention_layer_mode: str = "last",
    packed_attention_target_active_rows: float = 39.5,
    packed_attention_initial_burst_rows: float = 8.0,
    packed_attention_force_full_current: bool = False,
    packed_attention_full_current_context_rows: int = 8,
    packed_attention_context_a8_rows: int = 10,
    cross_block_cache_handoff: bool = False,
    cross_block_boundary_scope: str = "tri",
    cross_block_boundary_target_rows: int = 64,
    cross_block_full_prefix_oracle_block: int = -1,
    cross_block_block_initialization_deep_rows: int = 0,
    cross_block_block_initialization_dependency_only: bool = False,
    cross_block_block_initialization_relative_score_floor: float = 0.0,
    cross_block_block_initialization_include_future_dependency: bool = False,
    cross_block_block_initialization_protected_future_blocks: int = -1,
    cross_block_block_initialization_deep_bits: int = 0,
    cross_block_block_initialization_deep_clip_ratio: float = 0.0,
    cross_block_block_initialization_dependency_tiebreak: bool = False,
    question_start_token: int = -1,
    cross_block_block_initialization_protect_generation: bool = False,
    cross_block_block_initialization_all_a8: bool = False,
    cross_block_block_initialization_keep_global_l0_cache: bool = False,
    cross_block_block_initialization_global_layers: int = 1,
    cross_block_boundary_deep_a8_row_limit: int = -1,
    cross_block_boundary_context_row_bits: int = 8,
    cross_block_pending_confidence_mode: str = "remask_only",
    cross_block_pending_relation_mode: str = "direct",
    cross_block_dependency_policy: str = "current_keys_committed_rows",
    dynamic_block_lookahead: bool = False,
    dynamic_block_target_joint_rows: int = 48,
    dynamic_block_max_next_rows: int = 4,
    dynamic_block_source_b_max_attempts: int = -1,
    dynamic_block_source_b_retry_min_confidence: float = 0.0,
    dynamic_block_source_b_a4_only: bool = False,
    dynamic_block_source_b_dependency_tie_rank: bool = False,
    dynamic_block_source_a_confirm_at_handoff: bool = False,
    dynamic_block_target_prediction_rows: int = -1,
    dynamic_block_next_admission_budget: int = 1,
    dynamic_block_min_reuse_score: float = 0.0,
    dynamic_block_next_candidate_policy: str = "high_only",
    dynamic_block_next_relation_preference: str = "high_dependency",
    dynamic_block_max_current_unresolved: int = 8,
    dynamic_block_max_handoff_verification_rows: int = 0,
    dynamic_block_canonical_future: bool = False,
    dynamic_block_future_horizon: int = 1,
    dynamic_block_canonical_direct_tau: float = -1.0,
    dynamic_block_allow_deferred_verification: bool = False,
    dynamic_block_next_min_confidence: float = 0.92,
    state_capture_callback: Callable[[ForwardCaptureEvent], None] | None = None,
    attention_profile_capture_callback: Callable[[int, dict[str, Any]], None]
    | None = None,
    attention_profile_capture_scope: str = "local",
    forward_end_callback: Callable[[dict[str, Any]], None] | None = None,
    forward_start_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[torch.Tensor, int, List[GenerationTraceEvent], GenerationStats]:
    if cross_block_dependency_policy not in {"initial_keys_selected_rows", "current_keys_committed_rows"}:
        raise ValueError("unsupported cross-block dependency policy")
    if cache_initialization_activation_policy not in {"default", "a4"}:
        raise ValueError("cache initialization activation policy must be default or a4")
    if cache_initialization_activation_policy != "default" and row_precision_context is None:
        raise ValueError("cache initialization precision override requires quantized execution")
    if cross_block_block_initialization_global_layers != 1:
        raise ValueError("sparse block initialization supports one full global layer")
    if not 0 <= cross_block_block_initialization_relative_score_floor <= 1 or (
        cross_block_block_initialization_relative_score_floor and (
            not cross_block_block_initialization_deep_rows or not cross_block_block_initialization_dependency_only
            or cross_block_dependency_policy != "current_keys_committed_rows"
        )
    ):
        raise ValueError("relative block initialization floor requires current-key dependency-ranked uniform deep rows")
    if cross_block_block_initialization_dependency_only and not cross_block_block_initialization_dependency_tiebreak:
        raise ValueError("dependency-only block initialization requires dependency ordering")
    if cross_block_block_initialization_include_future_dependency and not cross_block_block_initialization_dependency_only:
        raise ValueError("joint block initialization risk requires dependency-only selection")
    if cross_block_block_initialization_protected_future_blocks < -1 or (
        cross_block_block_initialization_protected_future_blocks >= 0
        and not cross_block_block_initialization_protect_generation
    ):
        raise ValueError("near-future protection requires generation protection")
    if cross_block_block_initialization_deep_bits not in (0, 4, 8) or (
        cross_block_block_initialization_deep_bits and not cross_block_block_initialization_all_a8
    ):
        raise ValueError("separate deep bits require an A8 sparse block initialization scout")
    cross_block_block_initialization_deep_clip_ratio = float(
        torch.tensor(cross_block_block_initialization_deep_clip_ratio, dtype=torch.bfloat16)
    )
    if not 0 <= cross_block_block_initialization_deep_clip_ratio <= 1 or (
        cross_block_block_initialization_deep_clip_ratio and cross_block_block_initialization_deep_bits != 4
    ):
        raise ValueError("block initialization clipping override requires uniform deep A4")
    if cross_block_block_initialization_deep_rows < 0 or (
        cross_block_block_initialization_deep_rows
        and (
            cross_block_block_initialization_deep_rows < block_length
            or cross_block_full_prefix_oracle_block != 1
            or cross_block_boundary_scope != "layer0_attention_two_stage"
            or row_precision_context is None
        )
    ):
        raise ValueError(
            "sparse block initialization requires quantized Block1 and a valid row budget"
        )
    if (
        cross_block_block_initialization_dependency_tiebreak
        or cross_block_block_initialization_protect_generation
        or cross_block_block_initialization_all_a8
        or cross_block_block_initialization_keep_global_l0_cache
        or question_start_token >= 0
    ) and not cross_block_block_initialization_deep_rows:
        raise ValueError("block initialization selection options require sparse block initialization")
    if decoding_mode not in {"feature2", "fixed_k", "fixed_threshold"}:
        raise ValueError("unsupported decoding_mode")
    if decode_k < 1 or not 0.0 <= decode_threshold <= 1.0:
        raise ValueError("invalid fixed decoding parameter")
    irreversible = decoding_mode != "feature2"
    if irreversible:
        if dynamic_block_lookahead or dynamic_block_canonical_future:
            raise ValueError("fixed decoding excludes Feature3")
        if feature3_precision_policy != "all_a8":
            raise ValueError(
                "fixed decoding requires A8 regular execution"
            )
        if cross_block_boundary_context_row_bits != 8:
            raise ValueError("fixed decoding requires A8 boundary context")
        tail_confirmation_policy = "none"
    if cross_block_boundary_scope not in {"tri", "layer0_attention_two_stage"}:
        raise ValueError(
            "supported boundary scopes are tri and layer0_attention_two_stage"
        )
    if prompt.ndim != 2 or prompt.dtype != torch.long:
        raise ValueError("prompt must be a rank-2 torch.long tensor")
    if question_start_token < -1 or question_start_token >= prompt.shape[1]:
        raise ValueError("block initialization question position must be inside the prompt")
    if attention_profile_capture_scope not in {"local", "cross_block"}:
        raise ValueError("attention_profile_capture_scope must be local or cross_block")
    if (
        attention_profile_capture_scope == "cross_block"
        and attention_profile_capture_callback is None
    ):
        raise ValueError("cross-block profile capture requires a callback")
    if temperature != 0.0 or remasking != "low_confidence":
        raise ValueError("Feature 2 requires deterministic low_confidence decoding")
    if full_sequence_recompute and (
        packed_attention_refresh or cross_block_cache_handoff
        or decoding_mode != "fixed_k" or feature3_precision_policy != "all_a8"
        or cache_initialization_activation_policy != "default"
    ):
        raise ValueError("full-sequence recomputation requires fixed-k, all A8 and no cached refresh")
    if gen_length <= 0 or block_length <= 0 or gen_length % block_length:
        raise ValueError("gen_length must be positive and divisible by block_length")
    num_blocks = gen_length // block_length
    if steps <= 0 or steps % num_blocks:
        raise ValueError(
            "steps must be positive and divisible by the generation block count"
        )
    if not 0.0 <= tau_low <= tau_high <= 1.0 or not 0.0 <= confirm_tau <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= tau_low <= tau_high <= 1")
    if selective_a4_direct_tau != -1.0 and (not 0.0 <= selective_a4_direct_tau <= 1.0):
        raise ValueError("selective_a4_direct_tau must be -1 or in [0, 1]")
    if selective_a4_direct_tau >= 0.0 and (
        row_precision_context is None or not packed_attention_refresh
    ):
        raise ValueError(
            "selective A4 direct commit requires packed refresh and row precision"
        )
    if tau_high_tail is not None and (not tau_low <= tau_high_tail <= tau_high):
        raise ValueError("tau_high_tail must satisfy tau_low <= value <= tau_high")
    if tau_high_tail_after_step < 0:
        raise ValueError("tau_high_tail_after_step must be nonnegative")
    if budget_scale < 1.0:
        raise ValueError("budget_scale must be at least one")
    if tail_confirmation_policy not in {"none", "stable_only", "all"}:
        raise ValueError("unsupported tail_confirmation_policy")
    if row_precision_context is not None and feature3_maturity_age <= 0:
        raise ValueError("feature3_maturity_age must be positive")
    if row_precision_context is not None and feature3_precision_policy not in {
        "all_a8",
        "mature_only",
        "masked_only",
        "original",
    }:
        raise ValueError("unsupported Feature 3 precision policy")
    if candidate_numeric_mode not in {"bf16_lut", "native_fp64"}:
        raise ValueError("candidate_numeric_mode must be bf16_lut or native_fp64")
    if packed_attention_neighbor_scope not in {
        "current",
        "previous_current",
        "current_next",
        "tri",
        "tri_budgeted",
    }:
        raise ValueError("unsupported packed_attention_neighbor_scope")
    max_full_current_context_rows = max(0, 40 - block_length)
    if (
        not 0
        <= packed_attention_full_current_context_rows
        <= max_full_current_context_rows
    ):
        raise ValueError(
            "full-current context rows must keep current plus context within 40 base rows"
        )
    if packed_attention_force_full_current and (not packed_attention_refresh):
        raise ValueError("full-current refresh requires packed Feature 1")
    if packed_attention_layer_mode not in {"last", "all"}:
        raise ValueError("packed_attention_layer_mode must be last or all")
    if not 1.0 <= packed_attention_target_active_rows <= 96.0:
        raise ValueError("packed_attention_target_active_rows must be in [1, 96]")
    if packed_attention_initial_burst_rows < 0.0:
        raise ValueError("packed_attention_initial_burst_rows must be nonnegative")
    if packed_attention_context_a8_rows < 0:
        raise ValueError("packed_attention_context_a8_rows must be nonnegative")
    if dynamic_block_lookahead and (not packed_attention_refresh):
        raise ValueError("dynamic block lookahead requires packed Feature 1")
    if dynamic_block_lookahead and row_precision_context is None:
        raise ValueError("dynamic block lookahead requires A4/A8 row precision")
    if dynamic_block_lookahead and packed_attention_neighbor_scope not in {
        "current_next",
        "tri",
    }:
        raise ValueError(
            "dynamic block lookahead requires the next block in the packed region"
        )
    if not 1 <= dynamic_block_target_joint_rows <= 48:
        raise ValueError("dynamic_block_target_joint_rows must be in [1, 48]")
    if dynamic_block_max_next_rows <= 0:
        raise ValueError("dynamic_block_max_next_rows must be positive")
    if dynamic_block_source_b_max_attempts < -1:
        raise ValueError("Source B attempt limit must be -1 or nonnegative")
    if not 0.0 <= dynamic_block_source_b_retry_min_confidence <= 1.0:
        raise ValueError("Source B retry confidence must be in [0,1]")
    if (
        dynamic_block_target_prediction_rows != -1
        and not 1 <= dynamic_block_target_prediction_rows <= 64
    ):
        raise ValueError("joint prediction target must be -1 or in [1,64]")
    if dynamic_block_target_prediction_rows >= 0 and (
        not dynamic_block_lookahead
        or not dynamic_block_canonical_future
        or dynamic_block_future_horizon != 1
    ):
        raise ValueError(
            "joint prediction target requires canonical one-block future Feature3"
        )
    retry_filter = dynamic_block_source_b_retry_min_confidence > 0
    if retry_filter and (
        not dynamic_block_lookahead
        or not dynamic_block_canonical_future
        or dynamic_block_future_horizon != 1
    ):
        raise ValueError(
            "Source B retry rules require canonical one-block future Feature3"
        )
    if dynamic_block_source_a_confirm_at_handoff and (
        not dynamic_block_lookahead
        or not dynamic_block_canonical_future
        or dynamic_block_future_horizon != 1
        or not packed_attention_refresh
    ):
        raise ValueError(
            "Source A handoff confirmation requires packed canonical one-block future Feature3"
        )
    if (
        dynamic_block_source_b_a4_only
    ) and (
        not dynamic_block_lookahead
        or not dynamic_block_canonical_future
        or dynamic_block_future_horizon != 1
    ):
        raise ValueError(
            "Source B admission requires canonical one-block future Feature3"
        )
    if dynamic_block_source_b_dependency_tie_rank and (
        not dynamic_block_lookahead
        or not dynamic_block_canonical_future
        or dynamic_block_future_horizon != 1
    ):
        raise ValueError(
            "Source B dependency tie rank requires canonical one-block future Feature3"
        )
    if dynamic_block_source_b_max_attempts >= 0 and (
        not dynamic_block_lookahead
        or not dynamic_block_canonical_future
        or dynamic_block_future_horizon != 1
    ):
        raise ValueError(
            "Source B attempt limit requires canonical one-block future Feature3"
        )
    if dynamic_block_next_admission_budget <= 0:
        raise ValueError("dynamic_block_next_admission_budget must be positive")
    if not 0.0 <= dynamic_block_min_reuse_score <= 1.0:
        raise ValueError("dynamic_block_min_reuse_score must be in [0, 1]")
    if dynamic_block_max_current_unresolved <= 0:
        raise ValueError("dynamic_block_max_current_unresolved must be positive")
    if dynamic_block_lookahead and dynamic_block_max_handoff_verification_rows == 0:
        dynamic_block_max_handoff_verification_rows = min(
            dynamic_block_max_next_rows, block_length
        )
    if dynamic_block_lookahead and (
        not 1 <= dynamic_block_max_handoff_verification_rows <= block_length
    ):
        raise ValueError(
            "dynamic_block_max_handoff_verification_rows must be in [1, block_length]"
        )
    if dynamic_block_canonical_future and (not dynamic_block_lookahead):
        raise ValueError(
            "canonical future requires dynamic lookahead, immediate K/V writes, and no shadow handoff"
        )
    if dynamic_block_future_horizon not in {1, 2}:
        raise ValueError("dynamic_block_future_horizon must be 1 or 2")
    if dynamic_block_future_horizon > 1 and (not dynamic_block_canonical_future):
        raise ValueError("multi-block future horizon requires canonical future")
    if not (
        dynamic_block_canonical_direct_tau == -1.0
        or 0.0 <= dynamic_block_canonical_direct_tau <= 1.0
    ):
        raise ValueError("canonical future direct tau must be -1 or in [0, 1]")
    if dynamic_block_canonical_direct_tau >= 0.0 and (
        not dynamic_block_canonical_future
    ):
        raise ValueError("canonical future direct tau requires canonical future")
    if dynamic_block_next_candidate_policy not in {
        "high_only",
        "high_stable",
        "stable_high",
        "stable_low_only",
        "stable_low_seen",
        "stable_any",
        "proposal_verify",
        "observe_only",
    }:
        raise ValueError("unsupported dynamic_block_next_candidate_policy")
    if dynamic_block_next_relation_preference not in {
        "high_dependency",
        "low_dependency",
    }:
        raise ValueError("unsupported dynamic_block_next_relation_preference")
    if not 0.0 <= dynamic_block_next_min_confidence <= 1.0:
        raise ValueError("dynamic_block_next_min_confidence must be in [0, 1]")
    if cross_block_cache_handoff and (not packed_attention_refresh):
        raise ValueError("cross-block cache handoff requires packed Feature 1")
    if not cross_block_cache_handoff and cross_block_boundary_scope != "tri":
        raise ValueError("cross-block boundary scope requires cache handoff")
    boundary_row_limit = (
        256 if cross_block_boundary_scope == "layer0_attention_two_stage" else 96
    )
    if not block_length <= cross_block_boundary_target_rows <= boundary_row_limit:
        raise ValueError(
            "cross_block_boundary_target_rows exceeds the scope-specific range"
        )
    block_count = gen_length // block_length
    if cross_block_boundary_deep_a8_row_limit < -1 or (
        cross_block_boundary_deep_a8_row_limit >= 0
        and (
            cross_block_boundary_scope not in {"layer0_attention_two_stage"}
            or row_precision_context is None
        )
    ):
        raise ValueError("deep A8 limit requires quantized Layer0 scout boundaries")
    # A one-block functional run never reaches the configured block-1 block_initialization.
    if cross_block_full_prefix_oracle_block >= block_count and not (
        block_count == 1 and cross_block_full_prefix_oracle_block == 1
    ):
        raise ValueError("full-prefix oracle block is outside generation")
    if cross_block_full_prefix_oracle_block == 0:
        raise ValueError("block 0 uses the initial full-sequence forward")
    if cross_block_full_prefix_oracle_block > 0 and cross_block_boundary_scope not in {
        "layer0_attention_two_stage"
    }:
        raise ValueError("full-prefix oracle requires an L1-global scope")
    if cross_block_boundary_context_row_bits not in {4, 8}:
        raise ValueError("cross_block_boundary_context_row_bits must be 4 or 8")
    if cross_block_pending_confidence_mode not in {
        "remask_only",
        "all_changes",
        "stable_unmask",
    }:
        raise ValueError(
            "cross_block_pending_confidence_mode must be remask_only, all_changes, or stable_unmask"
        )
    if cross_block_pending_relation_mode != "direct":
        raise ValueError("cross_block_pending_relation_mode must be direct")
    if (
        cross_block_pending_confidence_mode != "remask_only"
        or cross_block_pending_relation_mode != "direct"
    ) and cross_block_boundary_scope not in {"layer0_attention_two_stage"}:
        raise ValueError(
            "cross-block pending confidence requires layer0_attention_two_stage scope"
        )
    if packed_attention_refresh:
        configure_tri_block_attention_monitor(
            model, layer_mode=packed_attention_layer_mode
        )
    (batch_size, prompt_length) = prompt.shape
    if dynamic_block_lookahead and batch_size != 1:
        raise ValueError("dynamic block lookahead currently requires batch size one")
    total_length = prompt_length + gen_length
    steps_per_block = steps // num_blocks
    if irreversible:
        steps_per_block = (
            math.ceil(block_length / decode_k)
            if decoding_mode == "fixed_k"
            else block_length
        )
    tokens = torch.full(
        (batch_size, total_length), mask_id, dtype=torch.long, device=prompt.device
    )
    tokens[:, :prompt_length] = prompt
    trace: List[GenerationTraceEvent] = []
    stats = GenerationStats()
    capture_index = 0
    model_callable = model.forward if hasattr(model, "forward") else model.__call__
    parameters = inspect.signature(model_callable).parameters
    supports_local_idx = "local_idx" in parameters
    activation_buffer_profile = ActivationBufferProfile()
    joint_scheduler = (
        DynamicJointWindowScheduler(
            target_joint_rows=dynamic_block_target_joint_rows,
            max_next_rows=dynamic_block_max_next_rows,
            min_reuse_score=dynamic_block_min_reuse_score,
            source_b_a4_only=dynamic_block_source_b_a4_only,
            source_b_dependency_tie_rank=dynamic_block_source_b_dependency_tie_rank,
            profile=activation_buffer_profile,
        )
        if dynamic_block_lookahead
        else None
    )

    persistent_block_states = (
        [
            Feature2BlockState((batch_size, block_length), device=tokens.device)
            for _ in range(num_blocks)
        ]
        if dynamic_block_lookahead
        else []
    )
    persistent_next_preconfirmed = (
        [
            torch.zeros(
                (batch_size, block_length),
                dtype=torch.bool,
                device=tokens.device,
            )
            for _ in range(num_blocks)
        ]
        if dynamic_block_lookahead
        else []
    )
    past_key_values: PastKeyValues | None = None
    cross_block_prefix_state = (
        CrossBlockPrefixAttentionState(
            total_length=total_length,
            block_length=block_length,
            boundary_target_rows=max(
                block_length,
                int(packed_attention_target_active_rows),
            )
            if cross_block_boundary_scope in {"layer0_attention_two_stage"}
            else cross_block_boundary_target_rows,
            pending_confidence_mode=cross_block_pending_confidence_mode,
            pending_relation_mode=cross_block_pending_relation_mode,
            track_future_actual_remask=cross_block_boundary_scope
            in {"layer0_attention_two_stage"},
            device=tokens.device,
        )
        if cross_block_boundary_scope in {"layer0_attention_two_stage"}
        else None
    )
    cross_block_carry_changed_positions = torch.empty(
        0, dtype=torch.long, device=tokens.device
    )
    cache_refresh_due = torch.zeros_like(tokens, dtype=torch.bool)
    source_a_handoff_pending = torch.zeros_like(tokens, dtype=torch.bool)
    cross_block_carry_refresh_due_positions = torch.empty(
        0, dtype=torch.long, device=tokens.device
    )
    for block_index in range(num_blocks):
        sparse_block_initialization = block_index == cross_block_full_prefix_oracle_block and bool(
            cross_block_block_initialization_deep_rows
        )
        full_block_initialization = (
            block_index == cross_block_full_prefix_oracle_block and not sparse_block_initialization
        )
        boundary_target_rows = (
            min(cross_block_block_initialization_deep_rows, total_length)
            if sparse_block_initialization
            else cross_block_boundary_target_rows
        )
        block_initialization_all_a8 = sparse_block_initialization and cross_block_block_initialization_all_a8
        boundary_deep_bits = cross_block_block_initialization_deep_bits if sparse_block_initialization else 0
        boundary_deep_clip_ratio = cross_block_block_initialization_deep_clip_ratio if sparse_block_initialization else 0.0
        keep_boundary_l0 = (
            sparse_block_initialization and cross_block_block_initialization_keep_global_l0_cache
        )
        boundary_context_bits = (
            8 if block_initialization_all_a8 else cross_block_boundary_context_row_bits
        )
        block_nfe_start = stats.nfe
        block_target_active_rows = float(packed_attention_target_active_rows)
        block_start = prompt_length + block_index * block_length
        block_end = block_start + block_length
        if dynamic_block_source_a_confirm_at_handoff:
            source_a_handoff_pending[:, block_start:block_end] = False
        cache_refresh_due_at_forward_start: Tuple[int, ...] = ()
        block_positions = torch.arange(block_start, block_end, device=tokens.device)
        region_start = block_start
        region_end = block_end
        if packed_attention_refresh:
            capture_cross_block_profile = (
                attention_profile_capture_callback is not None
                and attention_profile_capture_scope == "cross_block"
            )
            if packed_attention_neighbor_scope in {
                "previous_current",
                "tri",
                "tri_budgeted",
            }:
                region_start = max(prompt_length, block_start - block_length)
            if packed_attention_neighbor_scope in {
                "current_next",
                "tri",
                "tri_budgeted",
            }:
                region_end = min(prompt_length + gen_length, block_end + block_length)
            begin_tri_block_attention_monitor(
                model,
                region_start=region_start,
                region_end=region_end,
                prediction_start=block_start,
                prediction_end=block_end,
                prompt_end=prompt_length if capture_cross_block_profile else None,
                track_prefix_dependency=capture_cross_block_profile
                or cross_block_boundary_scope == "layer0_attention_two_stage",
                track_prefix_reverse_dependency=capture_cross_block_profile,
                track_prefix_future_dependency=capture_cross_block_profile,
                track_future_query_rows=capture_cross_block_profile
                or cross_block_boundary_scope in {"layer0_attention_two_stage"},
                trace_attention_relation_diagnostics=capture_cross_block_profile,
                capture_qk_probe=capture_cross_block_profile,
                capture_deployment_p8_relation=capture_cross_block_profile,
            )
        region_length = region_end - region_start
        block_offset = block_start - region_start
        block_state = (
            persistent_block_states[block_index]
            if dynamic_block_lookahead
            else Feature2BlockState((batch_size, block_length), device=tokens.device)
        )
        block_preconfirmed = (
            persistent_next_preconfirmed[block_index]
            if dynamic_block_lookahead
            else None
        )
        if dynamic_block_canonical_future and block_index > 0:
            stats.dynamic_block_canonical_handoff_locked_tokens += int(
                (block_state.state == LOCKED).sum().item()
            )
        handoff_verification_due = torch.zeros_like(block_state.state, dtype=torch.bool)
        if block_preconfirmed is not None:
            invalid_preconfirmed = block_preconfirmed & (block_state.state != TENTATIVE)
            if bool(invalid_preconfirmed.any()):
                raise RuntimeError(
                    "preconfirmed next-block rows must remain tentative at handoff"
                )
            handoff_verification_due |= block_preconfirmed
            if (
                int(handoff_verification_due.sum().item())
                > dynamic_block_max_handoff_verification_rows
            ):
                raise RuntimeError(
                    "handoff verification rows exceed the configured A8 capacity"
                )
            stats.dynamic_block_handoff_verification_tokens += int(
                handoff_verification_due.sum().item()
            )
            block_preconfirmed[handoff_verification_due] = False
        if dynamic_block_canonical_future and block_index > 0:
            canonical_due = (block_state.state == TENTATIVE) & ~handoff_verification_due
            handoff_verification_due |= canonical_due
            if (
                int(handoff_verification_due.sum().item())
                > dynamic_block_max_handoff_verification_rows
            ):
                raise RuntimeError(
                    "canonical handoff verification exceeds the configured A8 capacity"
                )
            stats.dynamic_block_handoff_verification_tokens += int(
                canonical_due.sum().item()
            )
        if not dynamic_block_lookahead:
            future_block_count = 0
        else:
            future_block_count = min(
                dynamic_block_future_horizon, num_blocks - block_index - 1
            )
        next_block_rows = future_block_count * block_length
        next_state_blocks = persistent_block_states[
            block_index + 1 : block_index + 1 + future_block_count
        ]
        if future_block_count == 1:
            next_block_state = next_state_blocks[0]
            next_preconfirmed = persistent_next_preconfirmed[block_index + 1]
            next_frontier_service_count = None
        elif future_block_count > 1:
            next_block_state = Feature2BlockState(
                (batch_size, next_block_rows), device=tokens.device
            )
            for field_name in ("state", "last_top1", "commit_origin", "precision_age"):
                setattr(
                    next_block_state,
                    field_name,
                    torch.cat(
                        [getattr(state, field_name) for state in next_state_blocks],
                        dim=1,
                    ).clone(),
                )
            next_preconfirmed = torch.cat(
                persistent_next_preconfirmed[
                    block_index + 1 : block_index + 1 + future_block_count
                ],
                dim=1,
            ).clone()
            next_frontier_service_count = None
        else:
            next_block_state = None
            next_preconfirmed = None
            next_frontier_service_count = None
        next_source_b_attempts = (
            torch.zeros(next_block_rows, dtype=torch.int32, device=tokens.device)
            if dynamic_block_source_b_max_attempts >= 0 or retry_filter
            else None
        )
        next_last_confidence = (
            torch.full(
                (next_block_rows,), -1.0, dtype=torch.bfloat16, device=tokens.device
            )
            if dynamic_block_source_b_retry_min_confidence > 0
            else None
        )

        def sync_future_block_states() -> None:
            if future_block_count <= 1 or next_block_state is None:
                return
            for future_index, state in enumerate(next_state_blocks):
                start = future_index * block_length
                end = start + block_length
                for field_name in (
                    "state",
                    "last_top1",
                    "commit_origin",
                    "precision_age",
                ):
                    getattr(state, field_name).copy_(
                        getattr(next_block_state, field_name)[:, start:end]
                    )
                persistent_next_preconfirmed[block_index + 1 + future_index].copy_(
                    next_preconfirmed[:, start:end]
                )

        schedule = get_num_transfer_tokens(block_state.state == MASKED, steps_per_block)
        if not cross_block_cache_handoff:
            past_key_values = None
        replace_position = torch.zeros_like(tokens, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True
        changed_rows = torch.zeros(
            (batch_size, block_length), dtype=torch.bool, device=tokens.device
        )
        block_transition_union = torch.zeros_like(changed_rows)
        changed_confidence = torch.ones(
            (batch_size, block_length), dtype=torch.bfloat16, device=tokens.device
        )
        cross_block_changed_confidence = changed_confidence.clone()
        feature2_local_tail_forward = False
        packed_state: TriBlockAttentionState | None = None
        packed_profile: dict[str, torch.Tensor] | None = None
        cross_block_profile: dict[str, torch.Tensor] | None = None
        packed_input_positions = torch.empty(0, dtype=torch.long, device=tokens.device)
        packed_refresh_positions = torch.empty(
            0, dtype=torch.long, device=tokens.device
        )
        packed_row_bits: torch.Tensor | None = None
        next_progress_positions = torch.empty(0, dtype=torch.long, device=tokens.device)
        next_added_positions = torch.empty(0, dtype=torch.long, device=tokens.device)
        next_source_b_positions = torch.empty(
            0, dtype=torch.long, device=tokens.device
        )
        next_source_a_positions = torch.empty(
            0, dtype=torch.long, device=tokens.device
        )
        next_progress_row_bits = torch.empty(0, dtype=torch.int8, device=tokens.device)
        next_relation_score = torch.empty(0, dtype=torch.bfloat16, device=tokens.device)
        next_observation: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        joint_residency: ActivationResidency | None = None
        base_residency: ActivationResidency | None = None
        next_step_verification_residency: ActivationResidency | None = None
        current_required_rows = 0
        next_rejected_for_buffer = 0
        next_rejected_for_current_step_capacity = 0
        next_rejected_for_verification_capacity = 0
        next_rejected_for_budget = 0
        source_b_optional_proposed_rows = 0
        source_b_optional_proposed_a8_rows = 0
        source_b_optional_reject_reason = "none"
        next_changed_rows = torch.zeros(
            (batch_size, max(block_length, next_block_rows)),
            dtype=torch.bool,
            device=tokens.device,
        )
        next_admission_quota = 0

        def account_tail_bypass(bypassed: torch.Tensor) -> None:
            if not bool(bypassed.any()):
                return
            stats.tail_bypass_tokens += int(bypassed.sum().item())
            stats.tail_bypass_forwards_saved += 1
            if not trace or trace[-1].block_index != block_index:
                if not dynamic_block_lookahead:
                    raise RuntimeError(
                        "tail bypass requires a preceding forward event in the same block"
                    )
                stats.dynamic_block_zero_forward_blocks += 1
                return
            trace[-1] = replace(
                trace[-1],
                tentative_after=0,
                tail_bypassed_positions=_positions(bypassed, block_start),
            )

        def current_row_bits(step_index: int) -> torch.Tensor | None:
            if row_precision_context is None:
                return None
            bits = block_state.row_bits(
                maturity_age=feature3_maturity_age, policy=feature3_precision_policy
            )
            if block_index == 0 and step_index < 0:
                bits[block_state.state == MASKED] = 8
            return bits

        def activate_row_bits(
            block_row_bits: torch.Tensor | None, *, local: bool
        ) -> torch.Tensor | None:
            if row_precision_context is None or block_row_bits is None:
                return None
            if local:
                row_precision_context.activate(block_row_bits)
                return block_row_bits
            full_row_bits = torch.full(
                (batch_size, total_length), 8, dtype=torch.int8, device=tokens.device
            )
            full_row_bits[:, block_start:block_end] = block_row_bits
            if block_index == 0 and past_key_values is None:
                full_row_bits[:, block_end:] = 8
                full_row_bits[:, :prompt_length] = 8
            row_precision_context.activate(full_row_bits)
            return full_row_bits

        def account_row_bits(block_row_bits: torch.Tensor | None) -> dict[str, Any]:
            row_budget_counts: dict[str, Any] = {
                "regular_row_budget_target_active_rows": block_target_active_rows,
                "regular_row_budget_allowed_active_rows": 0
                if packed_state is None
                else packed_state.allowed_active_rows,
                "regular_row_budget_regular_steps": 0
                if packed_state is None
                else packed_state.regular_steps,
                "regular_row_budget_cumulative_active_rows": 0
                if packed_state is None
                else packed_state.cumulative_active_rows,
            }
            if block_row_bits is None:
                return {
                    "a4_rows": 0,
                    "a8_rows": 0,
                    "masked_a4_rows": 0,
                    "tentative_a8_rows": 0,
                    "young_locked_a8_rows": 0,
                    "mature_locked_a4_rows": 0,
                    **row_budget_counts,
                }
            masked = block_state.state == MASKED
            tentative = block_state.state == TENTATIVE
            locked = block_state.state == LOCKED
            mature = locked & (block_state.precision_age >= feature3_maturity_age)
            count_values = torch.stack(
                (
                    (block_row_bits == 4).sum(),
                    (block_row_bits == 8).sum(),
                    (masked & (block_row_bits == 4)).sum(),
                    (tentative & (block_row_bits == 8)).sum(),
                    (locked & ~mature & (block_row_bits == 8)).sum(),
                    (mature & (block_row_bits == 4)).sum(),
                )
            ).tolist()
            counts = {
                "a4_rows": int(count_values[0]),
                "a8_rows": int(count_values[1]),
                "masked_a4_rows": int(count_values[2]),
                "tentative_a8_rows": int(count_values[3]),
                "young_locked_a8_rows": int(count_values[4]),
                "mature_locked_a4_rows": int(count_values[5]),
                **row_budget_counts,
            }
            stats.feature3_a4_rows += counts["a4_rows"]
            stats.feature3_a8_rows += counts["a8_rows"]
            stats.feature3_masked_a4_rows += counts["masked_a4_rows"]
            stats.feature3_tentative_a8_rows += counts["tentative_a8_rows"]
            stats.feature3_young_locked_a8_rows += counts["young_locked_a8_rows"]
            stats.feature3_mature_locked_a4_rows += counts["mature_locked_a4_rows"]
            stats.feature3_activation_bit_sum += (
                4 * counts["a4_rows"] + 8 * counts["a8_rows"]
            )
            return counts

        def reduce_candidates(
            logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if candidate_numeric_mode == "bf16_lut":
                return streaming_candidate_bf16(logits)
            probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
            (confidence, proposal) = probabilities.max(dim=-1)
            top_logit = torch.gather(logits, -1, proposal.unsqueeze(-1)).squeeze(-1)
            return (proposal, top_logit, confidence)

        def selected_candidate_probability(
            logits: torch.Tensor,
            token_ids: torch.Tensor,
            top_logit: torch.Tensor,
            top_confidence: torch.Tensor,
        ) -> torch.Tensor:
            if candidate_numeric_mode == "bf16_lut":
                return candidate_token_probability_bf16(
                    logits, token_ids, top_logit, top_confidence
                )
            return torch.gather(
                torch.softmax(logits.to(torch.float64), dim=-1),
                -1,
                token_ids.unsqueeze(-1),
            ).squeeze(-1)

        def emit_forward_capture(
            *,
            step_index: int,
            forward_kind: str,
            nfe_before: int,
            cache_initialized_before: bool,
            model_input_ids: torch.Tensor,
            input_positions: torch.Tensor,
            refresh_positions: torch.Tensor,
            row_bits: torch.Tensor | None,
            prediction_logits: torch.Tensor | None,
            prediction_positions: torch.Tensor,
            prediction_mask: torch.Tensor,
            teacher_proposal: torch.Tensor | None,
            teacher_top_logit: torch.Tensor | None,
            teacher_confidence: torch.Tensor | None,
            teacher_action_confidence: torch.Tensor | None,
            layer0_global_selected_deep: bool = False,
            layer0_keep_global_cache: bool = False,
            boundary_deep_bits: int = 0,
            layer0_current_row_bits: Tuple[int, ...] = (),
            boundary_deep_clip_ratio: float = 0.0,
            layer1_global_context_bits: int = 4,
            layer1_global_prefix_bits: int = 4,
        ) -> None:
            nonlocal capture_index
            if state_capture_callback is None:
                capture_index += 1
                return
            empty_ids = torch.empty(
                (batch_size, 0), dtype=torch.int32, device=tokens.device
            )
            empty_values = torch.empty(
                (batch_size, 0), dtype=torch.float32, device=tokens.device
            )
            if prediction_logits is None or prediction_logits.shape[1] == 0:
                top1_ids = empty_ids
                top2_ids = empty_ids.clone()
                top1_margin = empty_values
                top1_confidence = empty_values.clone()
                top1_action_confidence = empty_values.clone()
                current_probability = empty_values.clone()
            else:
                if teacher_proposal is None or teacher_top_logit is None:
                    raise RuntimeError(
                        "captured prediction logits require teacher candidates"
                    )
                if teacher_confidence is None:
                    raise RuntimeError(
                        "captured prediction logits require teacher confidence"
                    )
                if teacher_action_confidence is None:
                    raise RuntimeError(
                        "captured prediction logits require teacher action confidence"
                    )
                if prediction_logits.shape[:2] != teacher_proposal.shape:
                    raise RuntimeError(
                        "captured candidate shape does not match prediction logits"
                    )
                top2 = torch.topk(prediction_logits.float(), k=2, dim=-1)
                top1_ids = teacher_proposal.to(torch.int32)
                top2_ids = torch.where(
                    top2.indices[..., 0] == teacher_proposal,
                    top2.indices[..., 1],
                    top2.indices[..., 0],
                ).to(torch.int32)
                second_logit = torch.gather(
                    prediction_logits.float(), -1, top2_ids.to(torch.long).unsqueeze(-1)
                ).squeeze(-1)
                top1_margin = teacher_top_logit.float() - second_logit
                top1_confidence = teacher_confidence.float()
                top1_action_confidence = teacher_action_confidence.float()
                current_tokens = tokens.index_select(1, prediction_positions)
                current_probability = selected_candidate_probability(
                    prediction_logits,
                    current_tokens,
                    teacher_top_logit,
                    teacher_confidence,
                ).float()

            def cpu_copy(
                value: torch.Tensor, dtype: torch.dtype | None = None
            ) -> torch.Tensor:
                detached = value.detach()
                if dtype is not None:
                    detached = detached.to(dtype)
                return detached.cpu().clone()

            future_capture: dict[str, torch.Tensor] = {}
            if next_observation is not None:
                (future_local, _, future_logits, _) = next_observation
                if future_local.numel():
                    if next_block_state is None:
                        raise RuntimeError("future capture requires next-block state")
                    (future_top1, _, future_confidence) = reduce_candidates(
                        future_logits
                    )
                    future_capture = {
                        "future_prediction_positions": cpu_copy(
                            block_end + future_local, torch.int32
                        ),
                        "future_state_before": cpu_copy(
                            next_block_state.state.index_select(1, future_local),
                            torch.int8,
                        ),
                        "future_top1_token_ids": cpu_copy(future_top1, torch.int32),
                        "future_action_confidence": cpu_copy(
                            candidate_action_confidence(
                                future_top1,
                                future_confidence,
                                suppressed_candidate_token_ids,
                            )
                        ),
                    }
            event = ForwardCaptureEvent(
                capture_index=capture_index,
                block_index=block_index,
                step_index=step_index,
                forward_kind=forward_kind,
                nfe_before=nfe_before,
                cache_initialized_before=cache_initialized_before,
                full_sequence_recompute=full_sequence_recompute,
                tokens_before=cpu_copy(tokens, torch.int32),
                block_state_before=cpu_copy(block_state.state, torch.int8),
                **future_capture,
                model_input_ids=cpu_copy(model_input_ids, torch.int32),
                input_positions=cpu_copy(input_positions, torch.int32),
                refresh_positions=cpu_copy(refresh_positions, torch.int32),
                row_bits=cpu_copy(
                    row_bits
                    if row_bits is not None
                    else torch.empty(0, dtype=torch.int8, device=tokens.device),
                    torch.int8,
                ),
                prediction_positions=cpu_copy(prediction_positions, torch.int32),
                prediction_mask=cpu_copy(prediction_mask, torch.bool),
                teacher_top1_token_ids=cpu_copy(top1_ids, torch.int32),
                teacher_top2_token_ids=cpu_copy(top2_ids, torch.int32),
                teacher_top1_margin=cpu_copy(top1_margin, torch.float32),
                teacher_top1_confidence=cpu_copy(top1_confidence, torch.float32),
                teacher_top1_action_confidence=cpu_copy(
                    top1_action_confidence, torch.float32
                ),
                teacher_current_token_probability=cpu_copy(
                    current_probability, torch.float32
                ),
                layer0_global_selected_deep=layer0_global_selected_deep,
                layer0_keep_global_cache=layer0_keep_global_cache,
                boundary_deep_bits=boundary_deep_bits,
                layer0_current_row_bits=layer0_current_row_bits,
                boundary_deep_clip_ratio=boundary_deep_clip_ratio,
                layer1_global_context_bits=layer1_global_context_bits,
                layer1_global_prefix_bits=layer1_global_prefix_bits,
            )
            state_capture_callback(event)
            capture_index += 1

        def advance_next_block() -> dict[str, torch.Tensor]:
            empty = torch.zeros(
                (batch_size, max(block_length, next_block_rows)),
                dtype=torch.bool,
                device=tokens.device,
            )
            result = {
                "admitted": empty.clone(),
                "direct_locked": empty.clone(),
                "tentative": empty.clone(),
                "confirmed": empty.clone(),
                "remasked": empty.clone(),
                "admission_eligible": empty.clone(),
            }
            if (
                next_block_state is None
                or next_preconfirmed is None
                or next_observation is None
            ):
                return result
            (
                local_positions,
                _observed_row_bits,
                next_logits,
                relation_selected,
            ) = next_observation
            if local_positions.numel() == 0:
                return result
            (
                proposal_selected,
                top_logit_selected,
                confidence_selected,
            ) = reduce_candidates(next_logits)
            action_confidence_selected = candidate_action_confidence(
                proposal_selected, confidence_selected, suppressed_candidate_token_ids
            )
            if next_last_confidence is not None:
                next_last_confidence[local_positions] = action_confidence_selected[
                    0
                ].to(torch.bfloat16)
            observed = torch.zeros_like(next_block_state.state, dtype=torch.bool)
            observed[:, local_positions] = True
            proposal = torch.full_like(next_block_state.last_top1, -1)
            confidence = torch.full(
                next_block_state.state.shape,
                -1e309,
                dtype=confidence_selected.dtype,
                device=tokens.device,
            )
            proposal[:, local_positions] = proposal_selected
            confidence[:, local_positions] = action_confidence_selected
            relation_ok = torch.ones_like(observed, dtype=torch.bool)
            canonical_next_tokens = tokens[:, block_end : block_end + next_block_rows]
            next_tokens = canonical_next_tokens
            masked_at_start = next_block_state.state == MASKED
            tentative_at_start = next_block_state.state == TENTATIVE
            locked_at_start = next_block_state.state == LOCKED
            due_selected = tentative_at_start[:, local_positions]
            if bool(due_selected.any()):
                keep_probability = selected_candidate_probability(
                    next_logits,
                    next_tokens[:, local_positions],
                    top_logit_selected,
                    confidence_selected,
                )
                keep_probability = candidate_action_confidence(
                    proposal_selected, keep_probability, suppressed_candidate_token_ids
                )
                supported_selected = (
                    due_selected
                    & (proposal_selected == next_tokens[:, local_positions])
                    & (keep_probability >= float(confirm_tau))
                )
                failed_selected = due_selected & ~supported_selected
                supported = torch.zeros_like(tentative_at_start)
                failed = torch.zeros_like(tentative_at_start)
                supported[:, local_positions] = supported_selected
                failed[:, local_positions] = failed_selected
                next_block_state.state[failed] = MASKED
                next_block_state.precision_age[failed] = -1
                next_block_state.commit_origin[failed] = 0
                next_block_state.last_top1[failed] = -1
                next_tokens[failed] = int(mask_id)
                if dynamic_block_canonical_future:
                    next_block_state.state[supported] = LOCKED
                    next_block_state.precision_age[supported] = 0
                next_preconfirmed[supported] = not dynamic_block_canonical_future
                next_preconfirmed[failed] = False
                result["confirmed"] = supported
                result["remasked"] = failed
            admission_mask = masked_at_start & observed
            stable = admission_mask & (next_block_state.last_top1 == proposal)
            high = admission_mask & (confidence >= tau_high)
            canonical_direct = (
                admission_mask & (confidence >= dynamic_block_canonical_direct_tau)
                if dynamic_block_canonical_future
                and dynamic_block_canonical_direct_tau >= 0.0
                else high
            )
            selection_canonical_direct = canonical_direct.clone()
            if dynamic_block_source_a_confirm_at_handoff:
                source_a_observed = torch.zeros_like(admission_mask)
                source_a_local = local_positions[
                    torch.isin(
                        block_end + local_positions, next_source_a_positions
                    )
                ]
                source_a_observed[:, source_a_local] = True
                canonical_direct &= ~source_a_observed
            stable_low = (
                admission_mask
                & stable
                & relation_ok
                & (confidence >= tau_low)
                & (confidence < tau_high)
            )
            next_high = admission_mask & (
                confidence
                >= (
                    min(
                        dynamic_block_next_min_confidence,
                        dynamic_block_canonical_direct_tau,
                    )
                    if dynamic_block_canonical_future
                    and dynamic_block_canonical_direct_tau >= 0.0
                    else dynamic_block_next_min_confidence
                )
            )
            if dynamic_block_next_candidate_policy == "observe_only":
                next_candidate = torch.zeros_like(admission_mask)
            elif dynamic_block_next_candidate_policy == "high_only":
                next_candidate = next_high
            elif dynamic_block_next_candidate_policy == "high_stable":
                next_candidate = next_high | stable_low
            elif dynamic_block_next_candidate_policy in {
                "stable_low_only",
                "stable_low_seen",
            }:
                next_candidate = stable_low
            elif dynamic_block_next_candidate_policy == "stable_any":
                next_candidate = admission_mask & stable & relation_ok
            elif dynamic_block_next_candidate_policy == "proposal_verify":
                next_candidate = admission_mask & relation_ok
            else:
                next_candidate = next_high & stable
            next_admission_mask = admission_mask & next_candidate
            # Observe the existing candidate mask before quota selection.
            result["admission_eligible"] = next_admission_mask
            selected = torch.zeros_like(next_admission_mask)
            if next_admission_quota > 0:
                selected = select_admissions(
                    next_admission_mask,
                    selection_canonical_direct,
                    stable_low,
                    stable,
                    confidence,
                    torch.full(
                        (batch_size,),
                        int(next_admission_quota),
                        dtype=torch.long,
                        device=tokens.device,
                    ),
                    remaining_forwards=max(1, steps_per_block),
                    budget_scale=1.0,
                    stability_bonus=stability_bonus,
                )
            admission = next_block_state.admit(
                next_tokens,
                proposal,
                selected,
                canonical_direct
                if dynamic_block_canonical_future
                else torch.zeros_like(high),
                stable_low,
            )
            next_block_state.update_masked_history(proposal, admission_mask)
            next_block_state.advance_locked_age(locked_at_start & observed)
            result["admitted"] = (
                admission.direct_locked
                | admission.stable_tentative
                | admission.fallback_tentative
            )
            result["direct_locked"] = admission.direct_locked
            result["tentative"] = admission.stable_tentative | admission.fallback_tentative
            next_preconfirmed[result["admitted"]] = False
            if dynamic_block_canonical_future:
                stats.dynamic_block_canonical_direct_tokens += int(
                    admission.direct_locked.sum().item()
                )
            if (
                int((next_block_state.state == TENTATIVE).sum().item())
                > dynamic_block_max_handoff_verification_rows
            ):
                raise RuntimeError(
                    "next-block tentative rows exceed handoff verification capacity"
                )
            stats.dynamic_block_next_admitted_tokens += int(
                result["admitted"].sum().item()
            )
            stats.dynamic_block_next_confirmed_tokens += int(
                result["confirmed"].sum().item()
            )
            stats.dynamic_block_next_remasked_tokens += int(
                result["remasked"].sum().item()
            )
            sync_future_block_states()
            return result

        def mark_handoff_future_cache_due(
            next_update: dict[str, torch.Tensor],
        ) -> None:
            if not dynamic_block_source_a_confirm_at_handoff:
                return
            admitted_local = torch.nonzero(
                next_update["admitted"][0], as_tuple=False
            ).flatten().to(torch.long)
            if admitted_local.numel() == 0:
                return
            all_admitted_positions = block_end + admitted_local
            admitted_tentative = block_end + torch.nonzero(
                next_update["tentative"][0], as_tuple=False
            ).flatten().to(torch.long)
            if dynamic_block_source_a_confirm_at_handoff:
                admitted_source_a = all_admitted_positions[
                    torch.isin(all_admitted_positions, next_source_a_positions)
                ]
                if not bool(torch.isin(admitted_source_a, admitted_tentative).all()):
                    raise RuntimeError("Source A handoff proposals must remain tentative")
                source_a_handoff_pending[:, admitted_source_a] = True
                cache_refresh_due[:, admitted_source_a] = True

        def dynamic_trace_fields(
            next_update: dict[str, torch.Tensor],
        ) -> dict[str, Any]:
            moving_fields: dict[str, Any] = {}
            if feature2_local_tail_forward:
                moving_fields["feature2_local_tail_forward"] = True
            if not dynamic_block_lookahead or joint_residency is None:
                return moving_fields
            qkv = joint_residency.operator("qkv_k4096_all_a8")
            mixed = joint_residency.operator("mixed_linear_k4096")
            down = joint_residency.operator("ffn_down_k12288")
            if base_residency is None:
                raise RuntimeError("dynamic block base residency is unavailable")
            base_down = base_residency.operator("ffn_down_k12288")
            next_down = (
                next_step_verification_residency.operator("ffn_down_k12288")
                if next_step_verification_residency is not None
                else down
            )
            return {
                **moving_fields,
                "next_progress_positions": tuple(
                    (int(value) for value in next_progress_positions.tolist())
                ),
                "next_added_positions": tuple(
                    (int(value) for value in next_added_positions.tolist())
                ),
                "next_admission_eligible_a_tokens": int(
                    next_update["admission_eligible"][:, next_source_a_positions - block_end]
                    .sum().item()
                ),
                "next_admission_eligible_b_tokens": int(
                    next_update["admission_eligible"][:, next_source_b_positions - block_end]
                    .sum().item()
                ),
                "next_admission_quota": int(next_admission_quota),
                "next_admitted_positions": _positions(
                    next_update["admitted"], block_end
                ),
                "next_direct_locked_positions": _positions(next_update["direct_locked"], block_end),
                "next_tentative_positions": _positions(next_update["tentative"], block_end),
                "next_confirmed_positions": _positions(
                    next_update["confirmed"], block_end
                ),
                "next_remasked_positions": _positions(
                    next_update["remasked"], block_end
                ),
                "current_required_rows": int(current_required_rows),
                "next_progress_rows": int(next_progress_positions.numel()),
                "next_added_rows": int(next_added_positions.numel()),
                "joint_rows": int(joint_residency.active_rows),
                "joint_issued_rows": int(joint_residency.issued_slice_units),
                "joint_issued_slice_units": int(joint_residency.issued_slice_units),
                "joint_pe_issue_groups": int(joint_residency.pe_issue_groups),
                "joint_a4_rows": int(joint_residency.a4_rows),
                "joint_a8_rows": int(joint_residency.a8_rows),
                "qkv_activation_bytes": int(qkv.activation_bytes),
                "mixed_k4096_activation_bytes": int(mixed.activation_bytes),
                "ffn_down_activation_bytes": int(down.activation_bytes),
                "next_step_ffn_down_activation_bytes": int(next_down.activation_bytes),
                "base_weight_read_multiplier": int(base_down.weight_read_multiplier),
                "max_weight_read_multiplier": int(down.weight_read_multiplier),
                "feature3_extra_weight_read_multiplier": max(
                    0,
                    int(down.weight_read_multiplier)
                    - int(base_down.weight_read_multiplier),
                ),
                "buffer_valid": bool(down.fits_one_weight_pass),
                "next_rejected_for_buffer": int(next_rejected_for_buffer),
                "next_rejected_for_current_step_capacity": int(
                    next_rejected_for_current_step_capacity
                ),
                "next_rejected_for_verification_capacity": int(
                    next_rejected_for_verification_capacity
                ),
                "next_rejected_for_budget": int(next_rejected_for_budget),
                "source_b_optional_proposed_rows": source_b_optional_proposed_rows,
                "source_b_optional_proposed_a8_rows": source_b_optional_proposed_a8_rows,
                "source_b_optional_reject_reason": source_b_optional_reject_reason,
            }

        def run_forward(
            step_index: int,
            forward_kind: str,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, PastKeyValues, dict[str, Any]
        ]:
            nonlocal past_key_values, packed_state, packed_profile
            nonlocal cross_block_profile
            nonlocal packed_input_positions, packed_refresh_positions, packed_row_bits
            nonlocal next_progress_positions, next_added_positions
            nonlocal next_source_b_positions, next_source_a_positions
            nonlocal next_progress_row_bits, next_relation_score
            nonlocal next_observation, joint_residency, base_residency
            nonlocal next_admission_quota
            nonlocal next_step_verification_residency
            nonlocal current_required_rows, next_rejected_for_buffer
            nonlocal next_rejected_for_current_step_capacity
            nonlocal next_rejected_for_verification_capacity
            nonlocal next_rejected_for_budget
            nonlocal source_b_optional_proposed_rows
            nonlocal source_b_optional_proposed_a8_rows
            nonlocal source_b_optional_reject_reason
            nonlocal cache_refresh_due_at_forward_start
            if forward_start_callback is not None:
                forward_start_callback(dict(capture_index=capture_index, block_index=block_index,
                                            step_index=step_index, forward_kind=forward_kind))
            next_progress_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            capture_attention_profile: dict[str, Any] | None = None
            captured_layer0_current_bits = ()
            next_added_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            next_source_b_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            next_source_a_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            next_progress_row_bits = torch.empty(
                0, dtype=torch.int8, device=tokens.device
            )
            next_relation_score = torch.empty(
                0, dtype=torch.bfloat16, device=tokens.device
            )
            next_admission_quota = 0
            next_observation = None
            joint_residency = None
            base_residency = None
            next_step_verification_residency = None
            current_required_rows = 0
            next_rejected_for_buffer = 0
            next_rejected_for_current_step_capacity = 0
            next_rejected_for_verification_capacity = 0
            next_rejected_for_budget = 0
            source_b_optional_proposed_rows = 0
            source_b_optional_proposed_a8_rows = 0
            source_b_optional_reject_reason = "none"
            cache_refresh_due_at_forward_start = tuple(
                (
                    int(value)
                    for value in torch.nonzero(cache_refresh_due[0], as_tuple=False)
                    .flatten()
                    .tolist()
                )
            )
            block_row_bits = current_row_bits(step_index)
            unresolved_at_start = block_state.state != LOCKED
            selection_changed_positions: Tuple[int, ...] = ()
            selection_changed_confidence: Tuple[float, ...] = ()
            selection_cross_block_changed_confidence: Tuple[float, ...] = ()
            selection_mandatory_positions: Tuple[int, ...] = ()
            selection_optional_positions: Tuple[int, ...] = ()
            selection_cross_block_positions: Tuple[int, ...] = ()
            selection_cross_block_ranked_positions: Tuple[int, ...] = ()
            selection_cross_block_handoff_changed_positions: Tuple[int, ...] = ()
            boundary_changed_mandatory_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            selection_boundary_context_positions: Tuple[int, ...] = ()
            selection_boundary_context_scores: Tuple[float, ...] = ()
            block_initialization_score_floor = 0.0
            block_initialization_score_scan_rows = 0
            selection_boundary_context_candidate_rows = 0
            selection_boundary_context_score_entries_read = 0
            selection_boundary_context_topk_comparisons = 0
            selection_dependency_score: Tuple[float, ...] = ()
            selection_entries_read = 0
            selection_direct_reads = 0
            selection_confidence_reads = 0
            selection_risk_subtractions = 0
            selection_risk_multiplications = 0
            selection_causal_union_additions = 0
            selection_remask_status_reads = 0
            selection_joint_multiplications = 0
            selection_candidate_rows = 0
            selection_comparisons = 0
            dependency_rows_written = 0
            executed_row_bits: torch.Tensor | None = None
            cache_initialized_before = past_key_values is not None and not full_sequence_recompute
            capture_nfe_before = stats.nfe
            capture_model_input_ids: torch.Tensor | None = None
            capture_input_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            capture_refresh_positions = torch.empty(
                0, dtype=torch.long, device=tokens.device
            )
            if packed_state is not None and past_key_values is not None:
                selection_changed_positions = _positions(changed_rows, block_start)
                selection_changed_confidence = tuple(
                    (
                        float(value)
                        for value in changed_confidence[changed_rows]
                        .detach()
                        .to(torch.float32)
                        .cpu()
                        .tolist()
                    )
                )
                selection_cross_block_changed_confidence = tuple(
                    (
                        float(value)
                        for value in cross_block_changed_confidence[changed_rows]
                        .detach()
                        .to(torch.float32)
                        .cpu()
                        .tolist()
                    )
                )
                if next_block_state is not None:
                    selection_changed_positions += _positions(
                        next_changed_rows, block_end
                    )
                    selection_changed_confidence += (1.0,) * int(
                        next_changed_rows.sum().item()
                    )
                    selection_cross_block_changed_confidence += (1.0,) * int(
                        next_changed_rows.sum().item()
                    )
                selection_mandatory_positions = _local_positions(
                    packed_state.mandatory, region_start
                )
                selection_optional_positions = _local_positions(
                    packed_state.optional_selected, region_start
                )
                selection_dependency_score = tuple(
                    (
                        float(value)
                        for value in packed_state.dependency_score.detach()
                        .to(torch.float32)
                        .cpu()
                        .tolist()
                    )
                )
                selection_entries_read = packed_state.dependency_entries_read
                selection_direct_reads = packed_state.direct_dependency_entries_read
                selection_confidence_reads = packed_state.change_confidence_entries_read
                selection_risk_subtractions = packed_state.change_risk_subtractions
                selection_risk_multiplications = (
                    packed_state.change_risk_multiplications
                )
                selection_causal_union_additions = packed_state.causal_union_additions
                selection_remask_status_reads = packed_state.remask_status_entries_read
                selection_joint_multiplications = (
                    packed_state.joint_relation_multiplications
                )
                selection_candidate_rows = packed_state.selector_candidate_rows
                selection_comparisons = packed_state.top_budget_comparisons
            if full_sequence_recompute:
                executed_row_bits = activate_row_bits(block_row_bits, local=False)
                prediction_indices = torch.nonzero(
                    (block_state.state != LOCKED).any(dim=0), as_tuple=False
                ).flatten()
                output = model(tokens, use_cache=True,
                               logits_positions=block_start + prediction_indices)
                selected_logits, past_key_values = _require_cached_output(output)
                logits = selected_logits.new_zeros((batch_size, block_length, selected_logits.shape[-1]))
                logits.index_copy_(1, prediction_indices, selected_logits)
                capture_model_input_ids = tokens
                capture_input_positions = torch.arange(total_length, device=tokens.device)
                capture_refresh_positions = capture_input_positions
            elif past_key_values is None:
                executed_row_bits = activate_row_bits(block_row_bits, local=False)
                arguments: dict[str, Any] = {"use_cache": True}
                if cache_initialization_activation_policy == "default":
                    output = model(tokens, **arguments)
                else:
                    executed_row_bits = torch.full_like(executed_row_bits, 4)
                    with row_precision_context.using(executed_row_bits):
                        output = model(tokens, **arguments)
                capture_model_input_ids = tokens
                capture_input_positions = torch.arange(
                    total_length, dtype=torch.long, device=tokens.device
                )
                capture_refresh_positions = capture_input_positions
                (logits, past_key_values) = _require_cached_output(output)
                logits = logits[:, block_start:block_end]
                if packed_attention_refresh:
                    initial_attention_profile = read_tri_block_attention_profile(model)
                    capture_attention_profile = initial_attention_profile
                    packed_state = TriBlockAttentionState(
                        region_start=region_start,
                        region_end=region_end,
                        total_length=total_length,
                        initial_attention_profile=initial_attention_profile,
                        target_active_rows=block_target_active_rows,
                        initial_burst_rows=packed_attention_initial_burst_rows,
                    )
                    packed_profile = None
                    if cross_block_prefix_state is not None:
                        if block_index != 0:
                            raise RuntimeError(
                                "carried-prefix initialization requires block 0"
                            )
                        cross_block_prefix_state.start_initial_block(
                            block_start=block_start, block_end=block_end
                        )
                        cross_block_profile = initial_attention_profile
                    packed_input_positions = torch.arange(
                        total_length, dtype=torch.long, device=tokens.device
                    )
                    packed_refresh_positions = packed_input_positions
                    packed_row_bits = None
            elif packed_attention_refresh and packed_state is None:
                if not cross_block_cache_handoff or block_index == 0:
                    raise RuntimeError(
                        "packed Feature 1 state was not initialized by the full-sequence forward"
                    )
                if batch_size != 1:
                    raise ValueError(
                        "cross-block cache handoff currently requires batch size one"
                    )
                if cross_block_boundary_scope in {"layer0_attention_two_stage"}:
                    packed_input_positions = block_positions
                    (boundary_start, boundary_end) = (block_start, block_end)
                    selection_mandatory_positions = tuple(
                        (int(value) for value in block_positions.tolist())
                    )
                    selection_optional_positions = ()
                elif cross_block_boundary_scope == "tri":
                    (boundary_start, boundary_end) = (region_start, region_end)
                else:
                    raise RuntimeError(
                        "block 0 did not publish prompt Attention importance"
                    )
                if cross_block_boundary_scope not in {"layer0_attention_two_stage"}:
                    packed_input_positions = torch.arange(
                        boundary_start,
                        boundary_end,
                        dtype=torch.long,
                        device=tokens.device,
                    )
                packed_refresh_positions = packed_input_positions
                local_tokens = tokens.index_select(
                    1, packed_input_positions
                )
                current_packed = torch.searchsorted(
                    packed_input_positions, block_positions
                )
                if not torch.equal(
                    packed_input_positions.index_select(0, current_packed),
                    block_positions,
                ):
                    raise RuntimeError(
                        "cross-block boundary omitted a current-block position"
                    )
                if block_row_bits is not None:
                    boundary_row_bits = torch.full(
                        (batch_size, packed_input_positions.numel()),
                        cross_block_boundary_context_row_bits,
                        dtype=torch.int8,
                        device=tokens.device,
                    )
                    if boundary_changed_mandatory_positions.numel():
                        boundary_row_bits[
                            :,
                            torch.isin(
                                packed_input_positions,
                                boundary_changed_mandatory_positions,
                            ),
                        ] = 4
                    boundary_row_bits[:, current_packed] = block_row_bits
                    row_precision_context.activate(boundary_row_bits)
                    executed_row_bits = boundary_row_bits
                    packed_row_bits = boundary_row_bits[0]
                else:
                    packed_row_bits = None
                if (
                    cross_block_boundary_scope == "layer0_attention_two_stage"
                    and full_block_initialization
                ):
                    full_positions = block_initialization_refresh_positions(
                        total_length,
                        block_end,
                        block_length,
                        -1,
                        cross_block_carry_refresh_due_positions,
                        0,
                    )
                    required_block_initialization_positions = full_positions
                    ranked_prefix = full_positions[:0]
                    block_initialization_tokens = tokens.index_select(
                        1, full_positions
                    )
                    full_row_bits = None
                    if row_precision_context is not None:
                        full_row_bits = block_initialization_precision_bits(
                            full_positions,
                            prompt_length,
                            block_end,
                            cross_block_carry_refresh_due_positions,
                            8,
                            -1,
                        )
                        row_precision_context.activate(full_row_bits)
                    full_output = model(
                        block_initialization_tokens,
                        past_key_values=past_key_values,
                        use_cache=True,
                        query_position_ids=full_positions.unsqueeze(0),
                        kv_write_position_ids=full_positions.unsqueeze(0),
                    )
                    (local_logits, past_key_values) = _require_cached_output(
                        full_output
                    )
                    packed_input_positions = full_positions
                    packed_refresh_positions = full_positions
                    packed_row_bits = full_row_bits
                    executed_row_bits = full_row_bits
                    local_tokens = block_initialization_tokens
                    current_packed = torch.searchsorted(full_positions, block_positions)
                    selection_mandatory_positions = tuple(
                        (int(value) for value in required_block_initialization_positions.tolist())
                    )
                    selection_optional_positions = tuple(
                        (int(x) for x in torch.sort(ranked_prefix).values.tolist())
                    )
                    selection_cross_block_positions = selection_optional_positions
                    selection_cross_block_ranked_positions = tuple(
                        (int(x) for x in ranked_prefix.tolist())
                    )
                    capture_model_input_ids = block_initialization_tokens
                    capture_input_positions = full_positions
                    capture_refresh_positions = packed_refresh_positions
                elif cross_block_boundary_scope in {"layer0_attention_two_stage"}:
                    mandatory_deep_positions = cross_block_carry_refresh_due_positions
                    mandatory_deep_positions = mandatory_deep_positions[
                        ~torch.isin(mandatory_deep_positions, block_positions)
                    ]
                    effective_boundary_target = boundary_target_rows
                    if sparse_block_initialization and (
                        question_start_token >= 0
                        or cross_block_block_initialization_protect_generation
                    ):
                        protected_start = (
                            question_start_token
                            if question_start_token >= 0
                            else prompt_length
                        )
                        protected_end = (
                            total_length
                            if cross_block_block_initialization_protect_generation
                            else prompt_length
                        )
                        if cross_block_block_initialization_protected_future_blocks >= 0:
                            protected_end = min(
                                total_length,
                                block_end + block_length * cross_block_block_initialization_protected_future_blocks,
                            )
                        protected_positions = torch.arange(
                            protected_start, protected_end, device=tokens.device
                        )
                        mandatory_deep_positions = torch.unique(
                            torch.cat((mandatory_deep_positions, protected_positions)),
                            sorted=True,
                        )
                        mandatory_deep_positions = mandatory_deep_positions[
                            ~torch.isin(mandatory_deep_positions, block_positions)
                        ]
                        effective_boundary_target = max(
                            boundary_target_rows,
                            block_length + int(mandatory_deep_positions.numel()),
                        )
                    scout_candidates = None
                    scout_context_candidates = torch.cat(
                        (
                            torch.arange(
                                region_start,
                                block_start,
                                dtype=torch.long,
                                device=tokens.device,
                            ),
                            torch.arange(
                                block_end,
                                region_end,
                                dtype=torch.long,
                                device=tokens.device,
                            ),
                        )
                    )
                    required_context_rows = (
                        min(
                            packed_attention_full_current_context_rows,
                            int(scout_context_candidates.numel()),
                        )
                        if packed_attention_force_full_current
                        else 0
                    )
                    if cross_block_boundary_scope == "layer0_attention_two_stage":
                        if cross_block_prefix_state is None:
                            raise RuntimeError(
                                "two-stage Layer0 boundary lacks previous attention state"
                            )
                        all_optional = torch.arange(
                            total_length, dtype=torch.long, device=tokens.device
                        )
                        all_optional = all_optional[
                            ~torch.isin(all_optional, block_positions)
                        ]
                        previous_scores = cross_block_prefix_state.score_positions(
                            all_optional
                        )
                        if sparse_block_initialization and cross_block_block_initialization_include_future_dependency:
                            future_pending = cross_block_prefix_state.future_pending
                            if future_pending is None:
                                raise ValueError("joint block initialization risk requires observed future pending")
                            previous_scores = torch.maximum(
                                previous_scores, future_pending.index_select(0, all_optional)
                            )
                        previous_order = torch.argsort(
                            previous_scores, descending=True, stable=True
                        )
                        if sparse_block_initialization and cross_block_block_initialization_relative_score_floor:
                            _, effective_boundary_target, block_initialization_score_floor, block_initialization_score_scan_rows = block_initialization_relative_selection(
                                all_optional[previous_order], previous_scores[previous_order], mandatory_deep_positions,
                                block_length, cross_block_block_initialization_deep_rows, cross_block_block_initialization_relative_score_floor,
                            )
                        shortlist_rows = min(
                            int(all_optional.numel()),
                            2 * (effective_boundary_target - block_length),
                        )
                        scout_candidates = all_optional.index_select(
                            0, previous_order[:shortlist_rows]
                        )
                        scout_candidates = torch.sort(
                            torch.unique(
                                torch.cat(
                                    (
                                        scout_candidates,
                                        scout_context_candidates,
                                        mandatory_deep_positions,
                                    )
                                )
                            )
                        ).values
                        if (
                            sparse_block_initialization
                            and cross_block_block_initialization_dependency_tiebreak
                        ):
                            dependency_order = all_optional.index_select(
                                0, previous_order
                            )
                            scout_candidates = dependency_order[
                                torch.isin(dependency_order, scout_candidates)
                            ]
                        if sparse_block_initialization and cross_block_block_initialization_dependency_only:
                            dependency_order = all_optional.index_select(0, previous_order)
                            remaining = dependency_order[
                                ~torch.isin(dependency_order, mandatory_deep_positions)
                            ]
                            capacity = effective_boundary_target - block_length - int(mandatory_deep_positions.numel())
                            scout_candidates = torch.cat((mandatory_deep_positions, remaining[:capacity]))
                    scout_current_row_bits = (
                        None if block_row_bits is None
                        else torch.full_like(block_row_bits[0], 8)
                        if block_initialization_all_a8 else block_row_bits[0]
                    )
                    if scout_current_row_bits is not None:
                        captured_layer0_current_bits = tuple(
                            int(bit) for bit in scout_current_row_bits.tolist()
                        )
                    configure_tri_block_attention_monitor(model, layer_mode="last")
                    try:
                        (
                            local_logits,
                            past_key_values,
                            packed_input_positions,
                            packed_row_bits,
                            selected_optional,
                            selected_scores_q8,
                            scout_probability_entries,
                        ) = _layer0_global_then_attention_guided_deep(
                            model,
                            row_precision_context,
                            tokens=tokens,
                            stale_cache=past_key_values,
                            current_positions=block_positions,
                            current_row_bits=scout_current_row_bits,
                            target_rows=effective_boundary_target,
                            deep_candidate_positions=scout_candidates,
                            mandatory_candidate_positions=mandatory_deep_positions,
                            required_candidate_positions=(
                                scout_context_candidates if required_context_rows
                                else scout_context_candidates[:0]
                            ),
                            required_candidate_count=required_context_rows,
                            transition_positions=cross_block_carry_changed_positions
                            if cross_block_boundary_scope
                            == "layer0_attention_two_stage"
                            and cross_block_carry_changed_positions.numel()
                            else None,
                            global_context_bits=boundary_context_bits,
                            global_prefix_bits=boundary_context_bits,
                            deep_a8_row_limit=-1
                            if block_initialization_all_a8
                            else cross_block_boundary_deep_a8_row_limit,
                            keep_global_layer0_cache=keep_boundary_l0,
                            deep_uniform_bits=boundary_deep_bits,
                            deep_clip_ratio=boundary_deep_clip_ratio,
                        )
                    finally:
                        configure_tri_block_attention_monitor(
                            model, layer_mode=packed_attention_layer_mode
                        )
                    packed_refresh_positions = packed_input_positions
                    local_tokens = tokens.index_select(
                        1, packed_input_positions
                    )
                    current_packed = torch.searchsorted(
                        packed_input_positions, block_positions
                    )
                    if not torch.equal(
                        packed_input_positions.index_select(0, current_packed),
                        block_positions,
                    ):
                        raise RuntimeError(
                            "Layer0 attention boundary omitted a current row"
                        )
                    executed_row_bits = packed_row_bits
                    selection_mandatory_positions = tuple(
                        (int(value) for value in block_positions.tolist())
                    )
                    selection_optional_positions = tuple(
                        (
                            int(value)
                            for value in torch.sort(selected_optional).values.tolist()
                        )
                    )
                    selection_cross_block_positions = selection_optional_positions
                    selection_cross_block_handoff_changed_positions = tuple(
                        (
                            int(value)
                            for value in cross_block_carry_refresh_due_positions.tolist()
                        )
                    )
                    selection_cross_block_ranked_positions = tuple(
                        (int(value) for value in selected_optional.tolist())
                    )
                    if cross_block_prefix_state is not None and cross_block_dependency_policy == "initial_keys_selected_rows":
                        cross_block_prefix_state.selected_regular = (
                            selected_optional.detach().clone()
                        )
                    selection_boundary_context_positions = selection_optional_positions
                    selection_boundary_context_scores = tuple(
                        float(value) / 255.0
                        for value in selected_scores_q8.index_select(
                            0, torch.argsort(selected_optional)
                        ).tolist()
                    )
                    selection_boundary_context_candidate_rows = int(
                        scout_candidates.numel()
                        if scout_candidates is not None
                        else total_length - block_length
                    )
                    selection_boundary_context_score_entries_read = int(
                        scout_probability_entries
                    )
                    selection_boundary_context_topk_comparisons = max(
                        0, total_length - block_length - 1
                    )
                    capture_model_input_ids = local_tokens
                    capture_input_positions = packed_input_positions
                    capture_refresh_positions = packed_refresh_positions
                else:
                    output = model(
                        local_tokens,
                        past_key_values=past_key_values,
                        use_cache=True,
                        query_position_ids=packed_input_positions.unsqueeze(0),
                        kv_write_position_ids=packed_refresh_positions.unsqueeze(0),
                    )
                    capture_model_input_ids = local_tokens
                    capture_input_positions = packed_input_positions
                    capture_refresh_positions = packed_refresh_positions
                    (local_logits, past_key_values) = _require_cached_output(output)
                if local_logits.shape[:2] != local_tokens.shape:
                    raise ValueError(
                        "cross-block boundary refresh returned the wrong logits shape"
                    )
                logits = local_logits.index_select(1, current_packed)
                if cross_block_prefix_state is not None and cross_block_dependency_policy == "current_keys_committed_rows":
                    cross_block_prefix_state.advance_block(
                        block_start=block_start, block_end=block_end
                    )
                boundary_profile = read_tri_block_attention_profile(model)
                capture_attention_profile = boundary_profile
                if cross_block_prefix_state is not None:
                    cross_block_profile = boundary_profile
                packed_state = TriBlockAttentionState(
                    region_start=region_start,
                    region_end=region_end,
                    total_length=total_length,
                    initial_attention_profile=boundary_profile,
                    target_active_rows=block_target_active_rows,
                    initial_burst_rows=packed_attention_initial_burst_rows,
                )
                packed_profile = None
                dependency_rows_written = int(
                    boundary_profile["query_positions"].numel()
                )
            elif packed_attention_refresh:
                if packed_state is None:
                    raise RuntimeError(
                        "packed Feature 1 state was not initialized by the full-sequence forward"
                    )
                refresh_local = packed_state.refresh.detach().clone()
                if packed_attention_force_full_current:
                    old_refresh_rows = int(refresh_local.sum().item())
                    current_slice = slice(block_offset, block_offset + block_length)
                    full_current = torch.zeros_like(refresh_local)
                    full_current[current_slice] = True
                    required_refresh = full_current | packed_state.mandatory
                    context_candidates = torch.nonzero(
                        ~required_refresh, as_tuple=False
                    ).flatten()
                    context_scores = packed_state.dependency_score.index_select(
                        0, context_candidates
                    )
                    order = torch.argsort(context_scores, descending=True, stable=True)
                    positive = context_scores.index_select(0, order) > 0
                    context_limit = packed_attention_full_current_context_rows
                    if dynamic_block_lookahead:
                        context_limit = min(
                            context_limit,
                            max(
                                0,
                                dynamic_block_target_joint_rows
                                - int(required_refresh.sum().item()),
                            ),
                        )
                    selected_context = context_candidates.index_select(
                        0, order[positive][:context_limit]
                    )
                    optional = torch.zeros_like(refresh_local)
                    optional[selected_context] = True
                    packed_state.mandatory = required_refresh
                    packed_state.optional_selected = optional
                    packed_state.refresh = required_refresh | optional
                    packed_state.allowed_active_rows = int(
                        packed_state.refresh.sum().item()
                    )
                    packed_state.cumulative_active_rows += (
                        int(packed_state.refresh.sum().item()) - old_refresh_rows
                    )
                    refresh_local = packed_state.refresh.detach().clone()
                    selection_mandatory_positions = _local_positions(
                        required_refresh, region_start
                    )
                    selection_optional_positions = _local_positions(
                        optional, region_start
                    )
                active_local = refresh_local.clone()
                active_local[
                    block_offset : block_offset + block_length
                ] |= unresolved_at_start[0]
                packed_input_positions = region_start + torch.nonzero(
                    active_local, as_tuple=False
                ).flatten().to(torch.long)
                packed_refresh_positions = (
                    region_start + packed_state.refresh_local_indices()
                )
                prediction_positions = block_start + torch.nonzero(
                    unresolved_at_start[0], as_tuple=False
                ).flatten().to(torch.long)
                if prediction_positions.numel() == 0:
                    raise RuntimeError(
                        "packed forward requires at least one unresolved prediction row"
                    )
                if block_row_bits is not None:
                    packed_row_bits = torch.full(
                        (packed_input_positions.numel(),),
                        4,
                        dtype=torch.int8,
                        device=tokens.device,
                    )
                    in_current = (packed_input_positions >= block_start) & (
                        packed_input_positions < block_end
                    )
                    current_offsets = packed_input_positions[in_current] - block_start
                    current_unresolved = unresolved_at_start[0].index_select(
                        0, current_offsets
                    )
                    current_bits = block_row_bits[0].index_select(0, current_offsets)
                    packed_indices = torch.nonzero(in_current, as_tuple=False).flatten()
                    packed_row_bits[packed_indices[current_unresolved]] = current_bits[
                        current_unresolved
                    ]
                    if feature3_precision_policy == "all_a8":
                        packed_row_bits.fill_(8)
                    else:
                        mandatory_current = torch.zeros_like(
                            packed_row_bits, dtype=torch.bool
                        )
                        mandatory_current[packed_indices[current_unresolved]] = True
                        context_candidates = torch.nonzero(
                            ~mandatory_current, as_tuple=False
                        ).flatten()
                        context_count = context_a8_upgrade_count(
                            packed_row_bits,
                            int(context_candidates.numel()),
                            fixed_context_a8_rows=packed_attention_context_a8_rows,
                        )
                        if context_count:
                            context_positions = packed_input_positions.index_select(
                                0, context_candidates
                            )
                            in_region = (context_positions >= region_start) & (
                                context_positions < region_end
                            )
                            attention_values = torch.zeros(
                                context_positions.numel(),
                                dtype=torch.bfloat16,
                                device=tokens.device,
                            )
                            if bool(in_region.any()):
                                attention_values[
                                    in_region
                                ] = packed_state.dependency_score.index_select(
                                    0, context_positions[in_region] - region_start
                                )
                            if bool((~in_region).any()):
                                if cross_block_prefix_state is None:
                                    raise RuntimeError(
                                        "out-of-region packed rows lack cross-block scores"
                                    )
                                attention_values[
                                    ~in_region
                                ] = cross_block_prefix_state.score_positions(
                                    context_positions[~in_region]
                                )
                            order = torch.argsort(
                                attention_values, descending=True, stable=True
                            )[:context_count]
                            packed_row_bits[
                                context_candidates.index_select(0, order)
                            ] = 8
                else:
                    packed_row_bits = None
                current_required_rows = int(packed_input_positions.numel())
                current_unresolved_count = int(unresolved_at_start.sum().item())
                if next_block_state is not None and current_unresolved_count <= min(
                    dynamic_block_max_current_unresolved, block_length
                ):
                    if packed_row_bits is None:
                        raise RuntimeError(
                            "dynamic joint window requires packed row bits"
                        )
                    if joint_scheduler is None:
                        raise RuntimeError("dynamic joint scheduler is unavailable")
                    in_next = (packed_input_positions >= block_end) & (
                        packed_input_positions < block_end + next_block_rows
                    )
                    existing_packed = torch.nonzero(in_next, as_tuple=False).flatten()
                    existing_local = packed_input_positions[in_next] - block_end
                    next_region_start = block_end - region_start
                    available = min(
                        block_length, max(0, region_length - next_region_start)
                    )
                    next_bits = next_block_state.row_bits(
                        maturity_age=feature3_maturity_age, policy=feature3_precision_policy
                    )
                    next_priority = torch.zeros(
                        next_block_rows, dtype=torch.bfloat16, device=tokens.device
                    )
                    next_dependency_score = torch.zeros_like(next_priority)
                    next_relation = torch.zeros_like(next_priority)
                    if available:
                        attention = packed_state.dependency_score[
                            next_region_start : next_region_start + available
                        ].to(torch.bfloat16)
                        next_dependency_score[:available] = attention
                        if attention.numel() > 1:
                            order = torch.argsort(attention, stable=True)
                            ranks = torch.empty_like(order, dtype=torch.bfloat16)
                            ranks[order] = torch.arange(
                                attention.numel(),
                                device=tokens.device,
                                dtype=torch.bfloat16,
                            )
                            normalized_rank = (
                                ranks.float() / float(attention.numel() - 1)
                            ).to(torch.bfloat16)
                            next_relation[:available] = normalized_rank
                            next_priority[:available] = (
                                1.0 - normalized_rank
                                if dynamic_block_next_relation_preference
                                == "low_dependency"
                                else normalized_rank
                            )
                    if future_block_count > 1:
                        next_priority[block_length:] = 0.75
                    next_priority[next_block_state.state[0] == TENTATIVE] = 1.0
                    if existing_local.numel():
                        next_priority[existing_local] = 1.0
                    next_eligible = next_block_state.state[0] != LOCKED
                    next_eligible &= ~next_preconfirmed[0]
                    pending_handoff_count = int(
                        (next_block_state.state[0] == TENTATIVE).sum().item()
                    )
                    next_admission_quota = min(
                        int(dynamic_block_next_admission_budget),
                        max(
                            0,
                            int(dynamic_block_max_handoff_verification_rows)
                            - pending_handoff_count,
                        ),
                    )
                    if next_admission_quota == 0:
                        next_eligible &= next_block_state.state[0] == TENTATIVE
                    if existing_local.numel():
                        existing_precision_sufficient = packed_row_bits.index_select(
                            0, existing_packed
                        ) >= next_bits[0].index_select(0, existing_local)
                        next_eligible[existing_local] = (
                            next_block_state.state[0].index_select(0, existing_local)
                            != LOCKED
                        ) & existing_precision_sufficient
                    if dynamic_block_source_a_confirm_at_handoff:
                        next_eligible &= ~source_a_handoff_pending[
                            0, block_end : block_end + next_block_rows
                        ]
                    next_step_base_row_bits = packed_row_bits.clone()
                    in_current_base = (packed_input_positions >= block_start) & (
                        packed_input_positions < block_end
                    )
                    if bool(in_current_base.any()) and (
                        not dynamic_block_allow_deferred_verification
                    ):
                        current_base_indices = torch.nonzero(
                            in_current_base, as_tuple=False
                        ).flatten()
                        current_base_local = (
                            packed_input_positions[in_current_base] - block_start
                        )
                        current_may_be_tentative = (
                            block_state.state[0].index_select(0, current_base_local)
                            == MASKED
                        )
                        next_step_base_row_bits[
                            current_base_indices[current_may_be_tentative]
                        ] = 8
                    selection = joint_scheduler.select(
                        base_positions=packed_input_positions,
                        base_row_bits=packed_row_bits,
                        next_step_base_row_bits=next_step_base_row_bits,
                        next_block_start=block_end,
                        next_row_bits=next_bits[0],
                        next_unresolved=next_eligible,
                        next_tentative=next_block_state.state[0] == TENTATIVE,
                        next_priority=next_priority,
                        next_dependency_score=next_dependency_score,
                        source_b_attempts=next_source_b_attempts,
                        max_source_b_attempts=dynamic_block_source_b_max_attempts,
                        source_b_retry_min_confidence=dynamic_block_source_b_retry_min_confidence,
                        next_last_confidence=next_last_confidence,
                        target_prediction_rows=dynamic_block_target_prediction_rows,
                        current_prediction_rows=int(prediction_positions.numel()),
                        step_index=step_index,
                        next_service_count=next_frontier_service_count[0]
                        if next_frontier_service_count is not None
                        else None,
                    )
                    next_progress_positions = (
                        block_end + selection.progress_local_positions
                    )
                    next_added_positions = block_end + selection.added_local_positions
                    added_was_masked = next_block_state.state[0].index_select(
                        0, selection.added_local_positions
                    ) == MASKED
                    next_source_b_positions = block_end + selection.added_local_positions[
                        added_was_masked
                    ]
                    progress_was_masked = next_block_state.state[0].index_select(
                        0, selection.progress_local_positions
                    ) == MASKED
                    progress_positions = block_end + selection.progress_local_positions
                    next_source_a_positions = progress_positions[
                        progress_was_masked
                        & ~torch.isin(progress_positions, next_source_b_positions)
                    ]
                    if next_source_b_attempts is not None:
                        next_source_b_attempts[selection.added_local_positions] += 1
                    next_progress_row_bits = next_bits[0].index_select(
                        0, selection.progress_local_positions
                    )
                    next_relation_score = next_relation.index_select(
                        0, selection.progress_local_positions
                    )
                    joint_residency = selection.joint_residency
                    base_residency = selection.base_residency
                    next_step_verification_residency = (
                        selection.next_step_verification_residency
                    )
                    next_rejected_for_buffer = selection.rejected_for_buffer
                    next_rejected_for_current_step_capacity = (
                        selection.rejected_for_current_step_capacity
                    )
                    next_rejected_for_verification_capacity = (
                        selection.rejected_for_next_step_verification_capacity
                    )
                    next_rejected_for_budget = selection.rejected_for_budget
                    source_b_optional_proposed_rows = (
                        selection.source_b_optional_proposed_rows
                    )
                    source_b_optional_proposed_a8_rows = (
                        selection.source_b_optional_proposed_a8_rows
                    )
                    source_b_optional_reject_reason = (
                        selection.source_b_optional_reject_reason
                    )
                    if selection.added_local_positions.numel():
                        packed_input_positions = torch.cat(
                            (packed_input_positions, next_added_positions)
                        )
                        packed_row_bits = torch.cat(
                            (packed_row_bits, selection.added_row_bits)
                        )
                        order = torch.argsort(packed_input_positions, stable=True)
                        packed_input_positions = packed_input_positions.index_select(
                            0, order
                        )
                        packed_row_bits = packed_row_bits.index_select(0, order)
                    if next_progress_positions.numel():
                        packed_refresh_positions = torch.sort(
                            torch.unique(
                                torch.cat(
                                    (packed_refresh_positions, next_progress_positions)
                                )
                            )
                        ).values
                    if dynamic_block_source_a_confirm_at_handoff:
                        packed_refresh_positions = packed_refresh_positions[
                            ~source_a_handoff_pending[0, packed_refresh_positions]
                        ]
                elif packed_row_bits is not None:
                    joint_residency = activation_buffer_profile.analyze(packed_row_bits)
                    base_residency = joint_residency
                model_input_positions = packed_input_positions
                model_row_bits = packed_row_bits
                model_refresh_positions = packed_refresh_positions
                local_tokens = tokens.index_select(1, model_input_positions)
                if model_row_bits is not None:
                    row_precision_context.activate(model_row_bits)
                    executed_row_bits = model_row_bits
                monitor_prediction_positions = (
                    prediction_positions
                    if next_progress_positions.numel() == 0
                    else torch.cat((prediction_positions, next_progress_positions))
                )
                prepare_tri_block_attention_step(
                    model,
                    prediction_positions=monitor_prediction_positions,
                    excluded_query_positions=None,
                )
                output = model(
                    local_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    query_position_ids=model_input_positions.unsqueeze(0),
                    kv_write_position_ids=model_refresh_positions.unsqueeze(0),
                )
                capture_model_input_ids = local_tokens
                capture_input_positions = model_input_positions
                capture_refresh_positions = packed_refresh_positions
                (local_logits, past_key_values) = _require_cached_output(output)
                if local_logits.shape[:2] != local_tokens.shape:
                    raise ValueError(
                        "packed Feature 1 forward returned the wrong logits shape"
                    )
                prediction_packed = _locate_unique_positions(
                    model_input_positions, prediction_positions
                )
                logits = local_logits.new_zeros(
                    (batch_size, block_length, local_logits.shape[-1])
                )
                logits[
                    :, prediction_positions - block_start, :
                ] = local_logits.index_select(1, prediction_packed)
                if next_block_state is not None and next_progress_positions.numel():
                    next_packed = _locate_unique_positions(
                        model_input_positions, next_progress_positions
                    )
                    next_observation = (
                        next_progress_positions - block_end,
                        next_progress_row_bits,
                        local_logits.index_select(1, next_packed),
                        next_relation_score,
                    )
                packed_profile = read_tri_block_attention_profile(model)
                capture_attention_profile = packed_profile
                if cross_block_prefix_state is not None:
                    cross_block_profile = packed_profile
                dependency_rows_written = int(
                    packed_profile["query_positions"].numel()
                )
            else:
                local_tokens = tokens[:, block_start:block_end]
                arguments = {
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "replace_position": replace_position,
                }
                if supports_local_idx:
                    arguments["local_idx"] = block_positions
                executed_row_bits = activate_row_bits(block_row_bits, local=True)
                output = model(local_tokens, **arguments)
                capture_model_input_ids = local_tokens
                capture_input_positions = block_positions
                capture_refresh_positions = block_positions
                (logits, past_key_values) = _require_cached_output(output)
            if logits.shape[:2] != (batch_size, block_length):
                raise ValueError(f"{forward_kind} must return current-block logits")
            (_, cache_length, _, _, _) = _cache_metadata(past_key_values)
            if cache_length != total_length:
                raise ValueError(
                    f"KV cache length must remain {total_length}, got {cache_length}"
                )
            stats.nfe += 1
            if feature2_local_tail_forward:
                stats.feature2_local_tail_forwards += 1
            if dynamic_block_lookahead and joint_residency is not None:
                stats.dynamic_block_regular_forwards += 1
                stats.dynamic_block_next_progress_rows += int(
                    next_progress_positions.numel()
                )
                stats.dynamic_block_next_added_rows += int(next_added_positions.numel())
                stats.dynamic_block_current_required_rows += int(current_required_rows)
                stats.dynamic_block_joint_rows += joint_residency.active_rows
                stats.dynamic_block_joint_issued_rows += (
                    joint_residency.issued_slice_units
                )
                stats.dynamic_block_joint_issued_slice_units += (
                    joint_residency.issued_slice_units
                )
                stats.dynamic_block_joint_pe_issue_groups += (
                    joint_residency.pe_issue_groups
                )
                stats.dynamic_block_joint_a4_rows += joint_residency.a4_rows
                stats.dynamic_block_joint_a8_rows += joint_residency.a8_rows
                down_residency = joint_residency.operator("ffn_down_k12288")
                if base_residency is None:
                    raise RuntimeError("dynamic block base residency is unavailable")
                base_down_residency = base_residency.operator("ffn_down_k12288")
                stats.dynamic_block_base_weight_read_traversals += int(
                    base_down_residency.weight_read_multiplier
                )
                stats.dynamic_block_joint_weight_read_traversals += int(
                    down_residency.weight_read_multiplier
                )
                stats.dynamic_block_extra_weight_read_traversals += max(
                    0,
                    int(down_residency.weight_read_multiplier)
                    - int(base_down_residency.weight_read_multiplier),
                )
                stats.dynamic_block_max_weight_read_multiplier = max(
                    stats.dynamic_block_max_weight_read_multiplier,
                    down_residency.weight_read_multiplier,
                )
                if not down_residency.fits_one_weight_pass:
                    stats.dynamic_block_buffer_invalid_forwards += 1
                stats.dynamic_block_next_rejected_for_buffer += int(
                    next_rejected_for_buffer
                )
                stats.dynamic_block_next_rejected_for_current_step_capacity += int(
                    next_rejected_for_current_step_capacity
                )
                stats.dynamic_block_next_rejected_for_verification_capacity += int(
                    next_rejected_for_verification_capacity
                )
                stats.dynamic_block_next_rejected_for_budget += int(
                    next_rejected_for_budget
                )
            if (
                packed_attention_refresh
                and past_key_values is not None
                and (packed_state is not None)
            ):
                is_full_sequence = forward_kind == "full_sequence"
                is_boundary = forward_kind == "boundary_refresh"
                is_initial = is_full_sequence or is_boundary
                active_rows = (
                    total_length if is_full_sequence else int(packed_input_positions.numel())
                )
                refresh_rows = (
                    total_length if is_full_sequence else int(packed_refresh_positions.numel())
                )
                layer0_rows = active_rows
                layer0_a4_rows = (
                    int((executed_row_bits == 4).sum().item())
                    if executed_row_bits is not None
                    else 0
                )
                layer0_a8_rows = (
                    int((executed_row_bits == 8).sum().item())
                    if executed_row_bits is not None
                    else 0
                )
                if (
                    is_boundary
                    and not full_block_initialization
                    and (cross_block_boundary_scope in {"layer0_attention_two_stage"})
                ):
                    layer0_rows = total_length
                    if block_row_bits is not None:
                        layer0_a4_rows = captured_layer0_current_bits.count(4)
                        if boundary_context_bits == 4:
                            layer0_a4_rows += total_length - block_length
                        layer0_a8_rows = total_length - layer0_a4_rows
                dynamic_counts = {
                    "layer0_rows": layer0_rows,
                    "layer0_a4_rows": layer0_a4_rows,
                    "layer0_a8_rows": layer0_a8_rows,
                    "layer0_keep_global_cache": is_boundary and keep_boundary_l0,
                    "boundary_global_layers": 1,
                    "boundary_deep_bits": boundary_deep_bits if is_boundary else 0,
                    "layer0_current_row_bits": captured_layer0_current_bits,
                    "boundary_deep_clip_ratio": boundary_deep_clip_ratio if is_boundary else 0.0,
                    "prediction_rows": int(unresolved_at_start.sum().item()),
                    "refresh_rows": refresh_rows,
                    "active_rows": active_rows,
                    "early_exit_rows": 0 if is_initial else region_length - active_rows,
                    "mandatory_refresh_rows": active_rows
                    if is_initial
                    else int(packed_state.mandatory.sum().item()),
                    "optional_refresh_rows": 0
                    if is_initial
                    else int(packed_state.optional_selected.sum().item()),
                    "dependency_rows_updated": total_length
                    if is_full_sequence
                    else dependency_rows_written,
                    "dependency_table_bytes": 0
                    if is_initial
                    else int(
                        packed_state.dependency.numel()
                        * packed_state.dependency.element_size()
                    ),
                    "pending_dependency_rows": 0
                    if is_initial
                    else int((packed_state.dependency_score > 0).sum().item()),
                    "cross_block_handoff_changed_positions": selection_cross_block_handoff_changed_positions,
                    "cross_block_selected_positions": selection_cross_block_positions,
                    "cross_block_ranked_positions": selection_cross_block_ranked_positions,
                    "boundary_context_selected_positions": selection_boundary_context_positions,
                    "boundary_context_selected_scores": selection_boundary_context_scores,
                    "block_initialization_score_floor_bf16": block_initialization_score_floor,
                    "block_initialization_score_scan_rows": block_initialization_score_scan_rows,
                    "boundary_context_candidate_rows": selection_boundary_context_candidate_rows,
                    "boundary_context_score_entries_read": selection_boundary_context_score_entries_read,
                    "boundary_context_topk_comparisons": selection_boundary_context_topk_comparisons,
                    "refresh_start": region_start,
                    "refresh_end": region_end,
                    "changed_positions": selection_changed_positions,
                    "mandatory_refresh_positions": selection_mandatory_positions,
                    "optional_refresh_positions": selection_optional_positions,
                    "dependency_score_by_row": selection_dependency_score,
                    "changed_confidence_by_position": selection_changed_confidence,
                    "cross_block_changed_confidence_by_position": selection_cross_block_changed_confidence,
                    "dependency_shape": ()
                    if is_initial
                    else (region_length, region_length),
                    "dependency_dtype": ""
                    if is_initial
                    else str(packed_state.dependency.dtype).split(".", 1)[-1],
                    "dependency_entries_read": (
                        0 if is_initial else selection_entries_read
                    )
                    + selection_boundary_context_score_entries_read,
                    "direct_dependency_entries_read": 0
                    if is_initial
                    else selection_direct_reads,
                    "change_confidence_entries_read": 0
                    if is_initial
                    else selection_confidence_reads,
                    "change_risk_subtractions": 0
                    if is_initial
                    else selection_risk_subtractions,
                    "change_risk_multiplications": 0
                    if is_initial
                    else selection_risk_multiplications,
                    "causal_union_additions": 0
                    if is_initial
                    else selection_causal_union_additions,
                    "remask_status_entries_read": 0
                    if is_initial
                    else selection_remask_status_reads,
                    "joint_relation_multiplications": 0
                    if is_initial
                    else selection_joint_multiplications,
                    "dependency_entries_written": 0
                    if is_initial
                    else dependency_rows_written * region_length,
                    "selector_candidate_rows": (
                        0 if is_initial else selection_candidate_rows
                    ),
                    "top_budget_comparisons": (
                        0 if is_initial else selection_comparisons
                    ) + selection_boundary_context_topk_comparisons,
                }
                if is_full_sequence:
                    stats.packed_full_sequence_forwards += 1
                    stats.packed_full_sequence_rows += active_rows
                elif is_boundary:
                    stats.packed_boundary_forwards += 1
                    stats.packed_boundary_rows += active_rows
                    if executed_row_bits is not None:
                        boundary_a4_rows = int((executed_row_bits == 4).sum().item())
                        boundary_a8_rows = int((executed_row_bits == 8).sum().item())
                        stats.packed_boundary_a4_rows += boundary_a4_rows
                        stats.packed_boundary_a8_rows += boundary_a8_rows
                        stats.packed_boundary_activation_bit_sum += (
                            4 * boundary_a4_rows + 8 * boundary_a8_rows
                        )
                else:
                    stats.packed_regular_forwards += 1
            else:
                dynamic_counts = {
                }
            precision_counts = account_row_bits(block_row_bits)
            precision_counts.update(dynamic_counts)
            if full_sequence_recompute:
                precision_counts.update(
                    active_a4_rows=0, active_a8_rows=total_length,
                    layer0_a4_rows=0, layer0_a8_rows=total_length,
                    active_rows=total_length, layer0_rows=total_length,
                    refresh_rows=total_length,
                    prediction_rows=int(prediction_indices.numel()),
                    linear_activation_rows={name: {"a4": 0, "a8": total_length}
                        for name in ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")},
                )
            if forward_kind == "full_sequence" and cache_initialization_activation_policy == "a4":
                precision_counts["cache_initialization_activation_policy"] = "a4"
                precision_counts["linear_activation_rows"] = {
                    name: {"a4": total_length, "a8": 0}
                    for name in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "attn_out",
                        "ff_proj",
                        "up_proj",
                        "ff_out",
                    )
                }
            if packed_attention_refresh:
                stats.feature1_layer0_rows += dynamic_counts["layer0_rows"]
                stats.feature1_prediction_rows += dynamic_counts["prediction_rows"]
                stats.feature1_refresh_rows += dynamic_counts["refresh_rows"]
                if forward_kind == "boundary_refresh" and executed_row_bits is not None:
                    precision_counts["active_a4_rows"] = int(
                        (executed_row_bits == 4).sum().item()
                    )
                    precision_counts["active_a8_rows"] = int(
                        (executed_row_bits == 8).sum().item()
                    )
                if forward_kind not in {"full_sequence", "boundary_refresh"}:
                    stats.feature1_active_rows += dynamic_counts["active_rows"]
                    stats.feature1_early_exit_rows += dynamic_counts["early_exit_rows"]
                    stats.feature1_mandatory_refresh_rows += dynamic_counts[
                        "mandatory_refresh_rows"
                    ]
                    stats.feature1_dependency_rows_updated += dynamic_counts[
                        "dependency_rows_updated"
                    ]
                    stats.feature1_max_dependency_table_bytes = max(
                        stats.feature1_max_dependency_table_bytes,
                        dynamic_counts["dependency_table_bytes"],
                    )
                    if executed_row_bits is not None:
                        active_a4_rows = int((executed_row_bits == 4).sum().item())
                        active_a8_rows = int((executed_row_bits == 8).sum().item())
                        precision_counts["active_a4_rows"] = active_a4_rows
                        precision_counts["active_a8_rows"] = active_a8_rows
                        stats.feature1_active_a4_rows += active_a4_rows
                        stats.feature1_active_a8_rows += active_a8_rows
                        stats.feature1_active_activation_bit_sum += (
                            4 * active_a4_rows + 8 * active_a8_rows
                        )
            (proposal, top_logit, confidence) = reduce_candidates(logits)
            action_confidence = candidate_action_confidence(
                proposal, confidence, suppressed_candidate_token_ids
            )
            if capture_model_input_ids is None:
                raise RuntimeError(
                    "executed forward did not publish capture input identity"
                )
            if capture_refresh_positions.numel():
                completed_refresh = capture_refresh_positions
                cache_refresh_due[:, completed_refresh] = False
                if cross_block_prefix_state is not None and cross_block_dependency_policy == "current_keys_committed_rows":
                    cross_block_prefix_state.selected_regular = completed_refresh.detach().clone()
            if attention_profile_capture_callback is not None:
                if capture_attention_profile is None:
                    raise RuntimeError(
                        "attention profile capture requires a published monitor profile"
                    )
                attention_profile_capture_callback(
                    capture_index, capture_attention_profile
                )
            emit_forward_capture(
                step_index=step_index,
                forward_kind=forward_kind,
                nfe_before=capture_nfe_before,
                cache_initialized_before=cache_initialized_before,
                model_input_ids=capture_model_input_ids,
                input_positions=capture_input_positions,
                refresh_positions=capture_refresh_positions,
                row_bits=executed_row_bits,
                prediction_logits=logits,
                prediction_positions=block_positions,
                prediction_mask=block_state.state != LOCKED,
                teacher_proposal=proposal,
                teacher_top_logit=top_logit,
                teacher_confidence=confidence,
                teacher_action_confidence=action_confidence,
                layer0_global_selected_deep=forward_kind == "boundary_refresh"
                and not full_block_initialization
                and (cross_block_boundary_scope in {"layer0_attention_two_stage"}),
                layer0_keep_global_cache=forward_kind == "boundary_refresh"
                and keep_boundary_l0,
                boundary_deep_bits=boundary_deep_bits if forward_kind == "boundary_refresh" else 0,
                layer0_current_row_bits=captured_layer0_current_bits,
                boundary_deep_clip_ratio=boundary_deep_clip_ratio if forward_kind == "boundary_refresh" else 0.0,
                layer1_global_context_bits=boundary_context_bits
                if cross_block_boundary_scope in {"layer0_attention_two_stage"}
                else 4,
                layer1_global_prefix_bits=boundary_context_bits
                if cross_block_boundary_scope in {"layer0_attention_two_stage"}
                else 4,
            )
            return (logits, proposal, top_logit, confidence, precision_counts)

        def schedule_next_packed_forward() -> None:
            nonlocal packed_profile
            if not packed_attention_refresh:
                return
            if packed_state is None:
                raise RuntimeError("packed Feature 1 state is unavailable")
            predicted = torch.zeros(
                region_length, dtype=torch.bool, device=tokens.device
            )
            predicted[block_offset : block_offset + block_length] = (
                block_state.state[0] != LOCKED
            )
            changed_global = torch.zeros_like(tokens, dtype=torch.bool)
            changed_global[:, block_start:block_end] = changed_rows
            changed_confidence_global = torch.ones(
                changed_global.shape, dtype=torch.bfloat16, device=changed_global.device
            )
            changed_confidence_global[:, block_start:block_end] = changed_confidence
            changed_remask_global = torch.zeros_like(changed_global)
            changed_remask_global[:, block_start:block_end] = changed_rows & (
                changed_confidence < 1.0
            )
            if next_block_state is not None:
                changed_global[
                    :, block_end : block_end + next_block_rows
                ] = next_changed_rows
            if packed_profile is None:
                packed_state.plan_initial(
                    predicted=predicted,
                    changed_global=changed_global,
                    changed_confidence_global=changed_confidence_global,
                    changed_remask_global=changed_remask_global,
                )
            else:
                packed_state.observe(
                    packed_profile,
                    predicted=predicted,
                    changed_global=changed_global,
                    changed_confidence_global=changed_confidence_global,
                    changed_remask_global=changed_remask_global,
                )
                packed_profile = None

        def accumulate_cross_block_dependency(
            *,
            remasked: torch.Tensor,
            ordinary_confidence: torch.Tensor | None = None,
            relation_update: bool = True,
            current_transition_masks: Sequence[torch.Tensor] = (),
        ) -> None:
            nonlocal cross_block_profile, cross_block_changed_confidence
            nonlocal block_transition_union
            for mask in current_transition_masks:
                if mask.shape != changed_rows.shape or mask.dtype != torch.bool:
                    raise ValueError("cross-block current transition mask is invalid")
                block_transition_union |= mask
            if cross_block_prefix_state is None:
                return
            if (
                cross_block_prefix_state is not None
                and relation_update
                and (cross_block_profile is None)
            ):
                raise RuntimeError(
                    "carried-prefix dependency profile is unavailable after a forward"
                )
            changed_global = torch.zeros_like(tokens, dtype=torch.bool)
            changed_global[:, block_start:block_end] = changed_rows
            changed_confidence_global = torch.ones(
                changed_global.shape, dtype=torch.bfloat16, device=tokens.device
            )
            cross_block_confidence = changed_confidence
            if cross_block_prefix_state is not None:
                if cross_block_pending_confidence_mode in {
                    "all_changes",
                    "stable_unmask",
                }:
                    if ordinary_confidence is None:
                        raise RuntimeError(
                            "ordinary pending confidence requires proposal confidence"
                        )
                    cross_block_confidence = changed_confidence.clone()
                    ordinary_changed = changed_rows & ~remasked
                    cross_block_confidence[ordinary_changed] = ordinary_confidence[
                        ordinary_changed
                    ].to(torch.bfloat16)
                cross_block_changed_confidence = cross_block_confidence
                changed_confidence_global[
                    :, block_start:block_end
                ] = cross_block_confidence
            changed_remask_global = torch.zeros_like(changed_global)
            changed_remask_global[:, block_start:block_end] = remasked
            if cross_block_prefix_state is not None and relation_update:
                relation_profile = cross_block_profile
                if not cross_block_prefix_state.track_future_actual_remask:
                    relation_profile = _restrict_cross_block_profile_queries(
                        relation_profile, query_end=block_end
                    )
                cross_block_prefix_state.observe(
                    relation_profile,
                    changed_global=changed_global,
                    changed_confidence_global=changed_confidence_global,
                    changed_remask_global=changed_remask_global,
                )
            elif cross_block_prefix_state is not None:
                cross_block_prefix_state.accumulate_changes(
                    changed_global=changed_global,
                    changed_confidence_global=changed_confidence_global,
                    changed_remask_global=changed_remask_global,
                )
            cross_block_profile = None

        def emit_forward_end(step_index: int, forward_kind: str) -> None:
            if forward_end_callback is None:
                return
            next_bits = current_row_bits(step_index + 1)
            event = {
                "block_index": block_index,
                "step_index": step_index,
                "forward_kind": forward_kind,
                "decision_positions": torch.arange(
                    block_start, block_end, dtype=torch.int32
                ),
                "state": block_state.state.detach().cpu().clone(),
                "tokens": block_tokens.detach().cpu().clone(),
                "last_top1": block_state.last_top1.detach().cpu().clone(),
                "precision_age": block_state.precision_age.detach()
                .cpu()
                .clone(),
                "commit_origin": block_state.commit_origin.detach()
                .cpu()
                .clone(),
                "next_current_row_bits": None
                if next_bits is None
                else next_bits.detach().cpu().clone(),
                "cache_refresh_due": cache_refresh_due.detach().cpu().clone(),
                "trace": trace[-1],
            }
            forward_end_callback(event)

        for step_index in range(steps_per_block):
            unresolved_for_transfer = int((block_state.state != LOCKED).sum().item())
            feature2_local_tail_forward = bool(
                packed_state is not None and 0 < unresolved_for_transfer <= 8
            )
            bypassed = (
                torch.zeros_like(block_state.state, dtype=torch.bool)
                if bool(handoff_verification_due.any())
                else block_state.bypass_tail_confirmation(tail_confirmation_policy)
            )
            if bool(bypassed.any()):
                account_tail_bypass(bypassed)
                break
            masked_at_start = block_state.state == MASKED
            tentative_at_start = block_state.state == TENTATIVE
            locked_at_start = block_state.state == LOCKED
            handoff_verification_due_at_start = handoff_verification_due.clone()
            masked_before = int(masked_at_start.sum().item())
            tentative_before = int(tentative_at_start.sum().item())
            forward_kind = (
                "full_sequence"
                if past_key_values is None or full_sequence_recompute
                else "boundary_refresh"
                if packed_attention_refresh and packed_state is None
                else "local_block"
            )
            (
                logits,
                proposal,
                top_logit,
                confidence,
                precision_counts,
            ) = run_forward(
                step_index,
                forward_kind,
            )
            forward_row_bits = current_row_bits(step_index)
            decision_row_bits = forward_row_bits
            block_tokens = tokens[:, block_start:block_end]
            block_tokens_before = block_tokens.clone()
            action_confidence = candidate_action_confidence(
                proposal, confidence, suppressed_candidate_token_ids
            )
            next_update = advance_next_block()
            mark_handoff_future_cache_due(next_update)
            next_changed_rows = next_update["admitted"] | next_update["remasked"]
            keep_probability = action_confidence
            # Uniform-A4 refresh cannot verify inherited drafts. Keep them
            # tentative until a regular forward executes their A8 rows.
            a4_confirmation_input = (
                forward_kind == "boundary_refresh" and boundary_deep_bits == 4
            ) or (
                forward_kind == "full_sequence" and cache_initialization_activation_policy == "a4"
            )
            if bool(tentative_at_start.any()) and not a4_confirmation_input:
                keep_probability = selected_candidate_probability(
                    logits, block_tokens, top_logit, confidence
                )
                keep_probability = candidate_action_confidence(
                    proposal, keep_probability, suppressed_candidate_token_ids
                )
                confirmation = block_state.confirm(
                    block_tokens,
                    proposal,
                    keep_probability,
                    confirm_tau=confirm_tau,
                    mask_id=mask_id,
                )
                confirmed = confirmation.locked
                remasked = confirmation.remasked
                handoff_verification_due[confirmed | remasked] = False
            else:
                confirmed = torch.zeros_like(masked_at_start)
                remasked = torch.zeros_like(masked_at_start)
            admission_mask = masked_at_start
            admission_mask &= block_state.state == MASKED
            stable = admission_mask & (block_state.last_top1 == proposal)
            active_tau_high = (
                tau_high_tail
                if tau_high_tail is not None and step_index >= tau_high_tail_after_step
                else tau_high
            )
            high = admission_mask & (action_confidence >= active_tau_high)
            stable_low = (
                admission_mask
                & stable
                & (action_confidence >= tau_low)
                & (action_confidence < tau_high)
            )
            direct_high = high
            selective_a4_direct = torch.zeros_like(admission_mask)
            if packed_attention_refresh and decision_row_bits is not None:
                a4_high = high & (decision_row_bits == 4)
                a8_high = high & (decision_row_bits == 8)
                if selective_a4_direct_tau >= 0.0:
                    selective_a4_direct = (
                        admission_mask
                        & (decision_row_bits == 4)
                        & (action_confidence >= selective_a4_direct_tau)
                    )
                high = a8_high | selective_a4_direct
                direct_high = (
                    a8_high & ((action_confidence >= tau_high) | stable)
                    | selective_a4_direct
                )
                stable_low = (stable_low | a4_high) & ~selective_a4_direct
            admission_priority = action_confidence
            if irreversible:
                if bool(tentative_at_start.any()):
                    raise RuntimeError(
                        "irreversible decoding reached a tentative state"
                    )
                selected = select_irreversible_transfers(
                    admission_mask,
                    action_confidence,
                    mode=decoding_mode,
                    k=decode_k,
                    threshold=decode_threshold,
                )
                direct_high = selected
                stable_low = torch.zeros_like(selected)
            else:
                selected = select_admissions(
                    admission_mask,
                    high,
                    stable_low,
                    stable,
                    admission_priority,
                    schedule[:, step_index],
                    remaining_forwards=steps_per_block - step_index,
                    budget_scale=budget_scale,
                    stability_bonus=stability_bonus,
                )
            admission = block_state.admit(
                block_tokens, proposal, selected, direct_high, stable_low
            )
            block_state.update_masked_history(proposal, admission_mask)
            block_state.advance_locked_age(locked_at_start)
            changed_rows = block_tokens != block_tokens_before
            cache_refresh_due[:, block_start:block_end] |= changed_rows
            changed_confidence = torch.ones_like(
                action_confidence, dtype=torch.bfloat16
            )
            changed_confidence[remasked] = keep_probability[remasked].to(torch.bfloat16)
            accumulate_cross_block_dependency(
                remasked=remasked & changed_rows,
                ordinary_confidence=action_confidence,
                relation_update=True,
                current_transition_masks=(
                    changed_rows,
                    admission.direct_locked,
                    admission.stable_tentative,
                    admission.fallback_tentative,
                    confirmed,
                    remasked,
                ),
            )
            if bool((block_state.state != LOCKED).any()):
                schedule_next_packed_forward()
            stats.direct_lock_tokens += int(admission.direct_locked.sum().item())
            stats.selective_a4_direct_tokens += int(
                (admission.direct_locked & selective_a4_direct).sum().item()
            )
            stats.stable_tentative_tokens += int(
                admission.stable_tentative.sum().item()
            )
            stats.fallback_tentative_tokens += int(
                admission.fallback_tentative.sum().item()
            )
            stats.one_step_confirm_lock_tokens += int(confirmed.sum().item())
            stats.one_step_remask_tokens += int(remasked.sum().item())
            trace.append(
                GenerationTraceEvent(
                    block_index,
                    step_index,
                    forward_kind,
                    0
                    if forward_kind == "full_sequence"
                    else int(packed_input_positions[0].item())
                    if forward_kind == "boundary_refresh"
                    else block_start,
                    total_length
                    if forward_kind == "full_sequence"
                    else int(packed_input_positions.numel())
                    if forward_kind == "boundary_refresh"
                    else block_length,
                    total_length,
                    _positions(selected, block_start),
                    _positions(admission.direct_locked, block_start),
                    _positions(admission.stable_tentative, block_start),
                    _positions(admission.fallback_tentative, block_start),
                    _positions(confirmed, block_start),
                    _positions(remasked, block_start),
                    (),
                    masked_before,
                    int((block_state.state == MASKED).sum().item()),
                    tentative_before,
                    int((block_state.state == TENTATIVE).sum().item()),
                    stats.nfe,
                    selective_a4_direct_positions=_positions(
                        admission.direct_locked & selective_a4_direct, block_start
                    ),
                    cache_refresh_due_positions_before=cache_refresh_due_at_forward_start,
                    cache_refresh_due_positions_after=tuple(
                        (
                            int(value)
                            for value in torch.nonzero(
                                cache_refresh_due[0], as_tuple=False
                            )
                            .flatten()
                            .tolist()
                        )
                    ),
                    input_positions=tuple(
                        (int(value) for value in packed_input_positions.tolist())
                    )
                    if packed_attention_refresh
                    else (),
                    region_start=region_start if packed_attention_refresh else -1,
                    region_end=region_end if packed_attention_refresh else -1,
                    refresh_positions=tuple(
                        (int(value) for value in packed_refresh_positions.tolist())
                    )
                    if packed_attention_refresh
                    else (),
                    prediction_positions=_positions(
                        masked_at_start | tentative_at_start, block_start
                    )
                    if packed_attention_refresh
                    else (),
                    handoff_verification_positions=_positions(
                        handoff_verification_due_at_start, block_start
                    ),
                    **dynamic_trace_fields(next_update),
                    **precision_counts,
                )
            )
            emit_forward_end(step_index, forward_kind)
            if not bool((block_state.state != LOCKED).any()):
                break
        if irreversible and bool((block_state.state != LOCKED).any()):
            raise RuntimeError(
                "fixed decoding did not finish within its guaranteed step bound"
            )
        extra_step = steps_per_block
        bypassed = block_state.bypass_tail_confirmation(tail_confirmation_policy)
        account_tail_bypass(bypassed)
        if bool((block_state.state == TENTATIVE).any()):
            tentative_at_start = block_state.state == TENTATIVE
            locked_at_start = block_state.state == LOCKED
            masked_before = int((block_state.state == MASKED).sum().item())
            (logits, proposal, top_logit, confidence, precision_counts) = run_forward(
                extra_step, "local_confirmation"
            )
            block_tokens = tokens[:, block_start:block_end]
            block_tokens_before = block_tokens.clone()
            next_update = advance_next_block()
            mark_handoff_future_cache_due(next_update)
            next_changed_rows = next_update["admitted"] | next_update["remasked"]
            keep_probability = selected_candidate_probability(
                logits, block_tokens, top_logit, confidence
            )
            keep_probability = candidate_action_confidence(
                proposal, keep_probability, suppressed_candidate_token_ids
            )
            confirmation = block_state.confirm(
                block_tokens,
                proposal,
                keep_probability,
                confirm_tau=confirm_tau,
                mask_id=mask_id,
            )
            stats.one_step_confirm_lock_tokens += int(confirmation.locked.sum().item())
            stats.one_step_remask_tokens += int(confirmation.remasked.sum().item())
            block_state.advance_locked_age(locked_at_start)
            changed_rows = block_tokens != block_tokens_before
            cache_refresh_due[:, block_start:block_end] |= changed_rows
            changed_confidence = torch.ones_like(confidence, dtype=torch.bfloat16)
            changed_confidence[confirmation.remasked] = keep_probability[
                confirmation.remasked
            ].to(torch.bfloat16)
            accumulate_cross_block_dependency(
                remasked=confirmation.remasked & changed_rows,
                ordinary_confidence=candidate_action_confidence(
                    proposal, confidence, suppressed_candidate_token_ids
                ),
                current_transition_masks=(
                    changed_rows,
                    confirmation.locked,
                    confirmation.remasked,
                ),
            )
            if bool((block_state.state != LOCKED).any()):
                schedule_next_packed_forward()
            trace.append(
                GenerationTraceEvent(
                    block_index,
                    extra_step,
                    "local_confirmation",
                    block_start,
                    block_length,
                    total_length,
                    (),
                    (),
                    (),
                    (),
                    _positions(confirmation.locked, block_start),
                    _positions(confirmation.remasked, block_start),
                    (),
                    masked_before,
                    int((block_state.state == MASKED).sum().item()),
                    int(tentative_at_start.sum().item()),
                    0,
                    stats.nfe,
                    cache_refresh_due_positions_before=cache_refresh_due_at_forward_start,
                    cache_refresh_due_positions_after=tuple(
                        (
                            int(value)
                            for value in torch.nonzero(
                                cache_refresh_due[0], as_tuple=False
                            )
                            .flatten()
                            .tolist()
                        )
                    ),
                    input_positions=tuple(
                        (int(value) for value in packed_input_positions.tolist())
                    )
                    if packed_attention_refresh
                    else (),
                    region_start=region_start if packed_attention_refresh else -1,
                    region_end=region_end if packed_attention_refresh else -1,
                    refresh_positions=tuple(
                        (int(value) for value in packed_refresh_positions.tolist())
                    )
                    if packed_attention_refresh
                    else (),
                    prediction_positions=_positions(tentative_at_start, block_start)
                    if packed_attention_refresh
                    else (),
                    **dynamic_trace_fields(next_update),
                    **precision_counts,
                )
            )
            emit_forward_end(extra_step, "local_confirmation")
            extra_step += 1
        remaining = block_state.state == MASKED
        if bool(remaining.any()):
            masked_before = int(remaining.sum().item())
            locked_at_start = block_state.state == LOCKED
            (_, proposal, _, forced_confidence, precision_counts) = run_forward(
                extra_step, "local_forced_finish"
            )
            forced_action_confidence = candidate_action_confidence(
                proposal, forced_confidence, suppressed_candidate_token_ids
            )
            next_update = advance_next_block()
            mark_handoff_future_cache_due(next_update)
            next_changed_rows = next_update["admitted"] | next_update["remasked"]
            block_tokens = tokens[:, block_start:block_end]
            block_tokens_before = block_tokens.clone()
            block_tokens[remaining] = proposal[remaining]
            forced = int(remaining.sum().item())
            block_state.state[remaining] = LOCKED
            block_state.precision_age[remaining] = 0
            block_state.advance_locked_age(locked_at_start)
            changed_rows = block_tokens != block_tokens_before
            cache_refresh_due[:, block_start:block_end] |= changed_rows
            changed_confidence = torch.ones(
                changed_rows.shape, dtype=torch.bfloat16, device=tokens.device
            )
            accumulate_cross_block_dependency(
                remasked=torch.zeros_like(changed_rows),
                ordinary_confidence=forced_action_confidence,
                current_transition_masks=(changed_rows, remaining),
            )
            stats.forced_finish_tokens += forced
            trace.append(
                GenerationTraceEvent(
                    block_index,
                    extra_step,
                    "local_forced_finish",
                    block_start,
                    block_length,
                    total_length,
                    _positions(remaining, block_start),
                    (),
                    (),
                    (),
                    (),
                    (),
                    _positions(remaining, block_start),
                    masked_before,
                    0,
                    0,
                    0,
                    stats.nfe,
                    cache_refresh_due_positions_before=cache_refresh_due_at_forward_start,
                    cache_refresh_due_positions_after=tuple(
                        (
                            int(value)
                            for value in torch.nonzero(
                                cache_refresh_due[0], as_tuple=False
                            )
                            .flatten()
                            .tolist()
                        )
                    ),
                    input_positions=tuple(
                        (int(value) for value in packed_input_positions.tolist())
                    )
                    if packed_attention_refresh
                    else (),
                    region_start=region_start if packed_attention_refresh else -1,
                    region_end=region_end if packed_attention_refresh else -1,
                    refresh_positions=tuple(
                        (int(value) for value in packed_refresh_positions.tolist())
                    )
                    if packed_attention_refresh
                    else (),
                    prediction_positions=_positions(remaining, block_start)
                    if packed_attention_refresh
                    else (),
                    **dynamic_trace_fields(next_update),
                    **precision_counts,
                )
            )
            emit_forward_end(extra_step, "local_forced_finish")
        stats.per_block_nfe.append(stats.nfe - block_nfe_start)
        carry_mask = block_transition_union
        cross_block_carry_changed_positions = block_start + torch.nonzero(
            carry_mask[0], as_tuple=False
        ).flatten().to(torch.long)
        cross_block_carry_refresh_due_positions = (
            torch.nonzero(cache_refresh_due[0], as_tuple=False).flatten().to(torch.long)
        )
        if packed_attention_refresh:
            end_tri_block_attention_monitor(model)
    return (tokens, stats.nfe, trace, stats)
