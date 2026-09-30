"""Cache initialization, selected refresh rows and persistent cache replay."""

from dataclasses import asdict
from types import SimpleNamespace
import ast
import inspect
import textwrap

import pytest
import torch

from capture.replay import ForwardReplayer
from generation.engine import generate
from numerics.precision import RowPrecisionContext


@pytest.mark.parametrize("scores,expected", [
    ([1., .5, .25, .125, .0625], [4, 0, 1, 2]),
    ([.5, .5, .5, .5, .5], [4, 0, 1, 2, 3]),
    ([0., 0., 0., 0., 0.], [4, 0, 1, 2, 3]),
])
def test_relative_block_initialization_preserves_required_and_cutoff_ties(scores, expected):
    from generation.engine import block_initialization_relative_selection

    selected, rows, cutoff, scanned = block_initialization_relative_selection(
        torch.arange(5), torch.tensor(scores, dtype=torch.bfloat16), torch.tensor([4]),
        current_rows=2, row_cap=7, relative_floor=.25,
    )
    assert selected.tolist() == expected
    assert rows == len(expected) + 2 and scanned == 4
    assert cutoff == float(torch.tensor(scores[0] * .25, dtype=torch.bfloat16))


def test_relative_block_initialization_cap_and_empty_optional():
    from generation.engine import block_initialization_relative_selection

    scores = torch.tensor([1., .5, .25], dtype=torch.bfloat16)
    selected, rows, _, _ = block_initialization_relative_selection(
        torch.arange(3), scores, torch.tensor([2]), 2, 4, .125,
    )
    assert selected.tolist() == [2, 0] and rows == 4
    selected, rows, cutoff, scanned = block_initialization_relative_selection(
        torch.arange(3), scores, torch.arange(3), 2, 5, .125,
    )
    assert selected.tolist() == [0, 1, 2] and rows == 5
    assert cutoff == 0 and scanned == 0
    with pytest.raises(ValueError, match="sufficient row cap"):
        block_initialization_relative_selection(torch.arange(3), scores, torch.arange(3), 2, 4, .125)


@torch.no_grad()
def test_relative_block_initialization_changes_only_optional_rows_and_replays(monkeypatch):
    from generation import engine

    monkeypatch.setattr(engine.CrossBlockPrefixAttentionState, "score_positions",
                        lambda self, positions: torch.where(positions < 8, 1., .125).to(torch.bfloat16))
    captured = []
    options = dict(
        steps=96, gen_length=96, block_length=32, mask_id=7, suppressed_candidate_token_ids=(7,),
        tau_high=0., tau_low=0., confirm_tau=0., selective_a4_direct_tau=0.,
        budget_scale=2., cache_initialization_activation_policy="a4", packed_attention_refresh=True,
        packed_attention_layer_mode="all", packed_attention_target_active_rows=27.5,
        cross_block_cache_handoff=True, cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=64, cross_block_dependency_policy="current_keys_committed_rows",
        cross_block_full_prefix_oracle_block=1, cross_block_block_initialization_deep_rows=160,
        cross_block_block_initialization_dependency_only=True, cross_block_block_initialization_dependency_tiebreak=True,
        cross_block_block_initialization_protect_generation=True, cross_block_block_initialization_relative_score_floor=.25,
        cross_block_block_initialization_deep_bits=4, cross_block_block_initialization_all_a8=True,
        cross_block_block_initialization_keep_global_l0_cache=True,
    )
    observed = generate(model(), torch.full((1, 128), 2), **options,
                        row_precision_context=RowPrecisionContext(), state_capture_callback=captured.append)
    plain = generate(model(), torch.full((1, 128), 2), **options,
                     row_precision_context=RowPrecisionContext())
    assert torch.equal(observed[0], plain[0]) and observed[1:] == plain[1:]
    boundary = next(e for e in observed[2] if e.forward_kind == "boundary_refresh")
    assert boundary.active_rows == 104 and boundary.active_a4_rows == 104
    assert boundary.block_initialization_score_floor_bf16 == .25 and boundary.block_initialization_score_scan_rows == 128
    assert set(boundary.input_positions) == set(range(8)) | set(range(128, 224))
    later = next(e for e in observed[2] if e.block_index == 2 and e.forward_kind == "boundary_refresh")
    assert later.active_rows == 64 and later.block_initialization_score_scan_rows == 0
    replayer = ForwardReplayer(model(), RowPrecisionContext(), device=torch.device("cpu"))
    for event in captured:
        actual = replayer.step(event).prediction_logits.argmax(-1).to(torch.int32)
        assert torch.equal(actual, event.teacher_top1_token_ids[:, event.prediction_mask[0]])


@pytest.mark.parametrize("deep_bits,reset_cache", [(4, False), (8, False), (4, True), (8, True)])
@torch.no_grad()
def test_inherited_drafts_wait_for_an_a8_forward(deep_bits, reset_cache):
    captured = []
    _, _, trace, _ = generate(
        model(), torch.full((1, 128), 2), steps=96, gen_length=96,
        block_length=32, mask_id=7, suppressed_candidate_token_ids=(7,),
        tau_high=0., tau_low=0., confirm_tau=0., selective_a4_direct_tau=0.,
        budget_scale=26., cache_initialization_activation_policy="default" if reset_cache and deep_bits == 8 else "a4",
        packed_attention_refresh=True,
        packed_attention_layer_mode="all", packed_attention_target_active_rows=27.5,
        cross_block_cache_handoff=not reset_cache,
        cross_block_boundary_scope="tri" if reset_cache else "layer0_attention_two_stage",
        cross_block_boundary_target_rows=64,
        cross_block_dependency_policy="current_keys_committed_rows",
        cross_block_full_prefix_oracle_block=-1 if reset_cache else 1,
        cross_block_block_initialization_deep_rows=0 if reset_cache else 96,
        cross_block_block_initialization_deep_bits=0 if reset_cache else deep_bits,
        cross_block_block_initialization_all_a8=not reset_cache,
        cross_block_block_initialization_keep_global_l0_cache=not reset_cache,
        dynamic_block_lookahead=True, dynamic_block_target_joint_rows=48,
        dynamic_block_max_next_rows=32, dynamic_block_canonical_future=True,
        dynamic_block_canonical_direct_tau=1.,
        dynamic_block_next_candidate_policy="proposal_verify",
        dynamic_block_allow_deferred_verification=True,
        row_precision_context=RowPrecisionContext(), state_capture_callback=captured.append,
    )
    refresh_kind = "full_sequence" if reset_cache else "boundary_refresh"
    index = next(i for i, e in enumerate(trace)
                 if e.block_index == 1 and e.forward_kind == refresh_kind)
    block_initialization = trace[index]
    assert block_initialization.tentative_before > 0
    if deep_bits == 4:
        assert (block_initialization.layer0_a8_rows if reset_cache else block_initialization.active_a8_rows) == 0
        assert not block_initialization.confirmed_positions and not block_initialization.remasked_positions
        assert block_initialization.tentative_after >= block_initialization.tentative_before
        confirmation = trace[index + 1]
        assert confirmation.block_index == 1 and confirmation.forward_kind == "local_block"
    else:
        confirmation = block_initialization
    assert confirmation.confirmed_positions
    execution = captured[index + (deep_bits == 4)]
    for position in confirmation.confirmed_positions:
        row = (execution.input_positions == position).nonzero().item()
        assert execution.row_bits.reshape(-1)[row] == 8
    replay = ForwardReplayer(model(), RowPrecisionContext(), device=torch.device("cpu"))
    for event in captured:
        actual = replay.step(event).prediction_logits.argmax(-1).to(torch.int32)
        assert torch.equal(actual, event.teacher_top1_token_ids[:, event.prediction_mask[0]])


@torch.no_grad()
def test_source_a_handoff_confirmation_keeps_source_b_direct():
    captured = []
    _, _, trace, _ = generate(
        model(),
        torch.full((1, 32), 2),
        steps=64,
        gen_length=64,
        block_length=32,
        mask_id=7,
        suppressed_candidate_token_ids=(7,),
        tau_high=0.0,
        tau_low=0.0,
        confirm_tau=0.0,
        selective_a4_direct_tau=0.0,
        budget_scale=2.0,
        packed_attention_refresh=True,
        packed_attention_layer_mode="all",
        packed_attention_target_active_rows=27.5,
        cross_block_cache_handoff=True,
        cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=64,
        cross_block_full_prefix_oracle_block=-1,
        cross_block_dependency_policy="current_keys_committed_rows",
        dynamic_block_lookahead=True,
        dynamic_block_target_joint_rows=48,
        dynamic_block_max_next_rows=32,
        dynamic_block_max_current_unresolved=32,
        dynamic_block_canonical_future=True,
        dynamic_block_canonical_direct_tau=0.0,
        dynamic_block_next_min_confidence=0.0,
        dynamic_block_next_candidate_policy="proposal_verify",
        dynamic_block_allow_deferred_verification=True,
        dynamic_block_source_a_confirm_at_handoff=True,
        row_precision_context=RowPrecisionContext(),
        state_capture_callback=captured.append,
    )
    source_a_pending = []
    source_b_direct = []
    for index, event in enumerate(trace):
        source_a = set(event.next_admitted_positions) - set(
            event.next_added_positions
        )
        source_b = set(event.next_admitted_positions) & set(
            event.next_added_positions
        )
        source_b_direct.extend(source_b & set(event.next_direct_locked_positions))
        for position in source_a:
            assert position in event.next_tentative_positions
            assert position not in event.next_direct_locked_positions
            assert position in event.refresh_positions
            same_block = [
                later
                for later in trace[index + 1 :]
                if later.block_index == event.block_index
            ]
            assert all(position not in later.next_progress_positions for later in same_block)
            assert all(position not in later.refresh_positions for later in same_block)
            resolution = next(
                later
                for later in trace[index + 1 :]
                if later.block_index == event.block_index + 1
                and position
                in set(later.confirmed_positions) | set(later.remasked_positions)
            )
            assert position in resolution.refresh_positions
            source_a_pending.append(position)
    assert source_a_pending
    assert source_b_direct

    replay = ForwardReplayer(
        model(), RowPrecisionContext(), device=torch.device("cpu")
    )
    for event in captured:
        actual = replay.step(event)
        expected = event.teacher_top1_token_ids[:, event.prediction_mask[0]]
        assert torch.equal(actual.prediction_logits.argmax(-1).to(torch.int32), expected)


def model():
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import (
        ActivationType,
        BlockType,
        LLaDAModel,
        LLaDAModelLM,
        ModelConfig,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        config = ModelConfig(
            d_model=16,
            n_heads=2,
            n_kv_heads=2,
            n_layers=2,
            mlp_hidden_size=32,
            activation_type=ActivationType.silu,
            block_type=BlockType.llama,
            max_sequence_length=640,
            vocab_size=8,
            embedding_size=8,
            rope=True,
            weight_tying=False,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        return (
            LLaDAModelLM(LLaDAConfig(**asdict(config)), model=LLaDAModel(config))
            .eval()
            .to(torch.bfloat16)
        )


def test_dependency_handoff_retains_pending_and_updates_key_block():
    from generation.refresh import CrossBlockPrefixAttentionState

    state = CrossBlockPrefixAttentionState(
        total_length=12, block_length=2, boundary_target_rows=4,
        device=torch.device("cpu"), pending_confidence_mode="all_changes",
        track_future_actual_remask=True,
    )
    state.start_initial_block(block_start=4, block_end=6)
    state.pending[0] = 0.25
    state.future_pending[6] = 0.5
    state.future_pending[10] = 0.75
    state.relation.fill_(1)
    state.advance_block(block_start=6, block_end=8)
    assert not bool(state.relation.any())
    assert state.pending[0] == 0.25 and state.pending[6] == 0.5
    assert state.future_pending[6] == 0 and state.future_pending[10] == 0.75
    changed = torch.zeros((1, 12), dtype=torch.bool)
    changed[0, 6] = True
    state.selected_regular = torch.tensor([0])
    state.observe(
        {"prefix_dependency_query_positions": torch.tensor([0, 1]),
         "prefix_dependency_mean": torch.tensor([[[0.125, 0.], [0.75, 0.]]])},
        changed_global=changed, changed_confidence_global=torch.ones((1, 12)),
        changed_remask_global=torch.zeros_like(changed),
    )
    assert state.pending[0] == 0.125 and state.pending[1] == 0.75
    assert state.pending[6] == 0.5 and state.future_pending[10] == 0.75


@pytest.mark.parametrize("policy", ["initial_keys_selected_rows", "current_keys_committed_rows"])
@torch.no_grad()
def test_generation_dependency_policy_tracks_keys_and_completed_rows(monkeypatch, policy):
    from generation import engine

    states, observations, scores = [], [], []
    original = engine.CrossBlockPrefixAttentionState
    boundary = engine._layer0_global_then_attention_guided_deep

    class ObservedState(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            states.append(self)

    def observe_boundary(*args, **kwargs):
        result = boundary(*args, **kwargs)
        scores.append(dict(zip(result[4].tolist(), result[5].tolist())))
        return result

    monkeypatch.setattr(engine, "CrossBlockPrefixAttentionState", ObservedState)
    monkeypatch.setattr(engine, "_layer0_global_then_attention_guided_deep", observe_boundary)

    def capture(event):
        state = states[0]
        current_start = 128 + event.block_index * 32
        expected_start = current_start if policy == "current_keys_committed_rows" else 128
        assert (state.block_start, state.block_end) == (expected_start, expected_start + 32)
        if policy == "current_keys_committed_rows":
            assert torch.equal(state.selected_regular.cpu(), event.refresh_positions.to(torch.long))
        elif event.forward_kind == "boundary_refresh":
            expected = set(event.input_positions.tolist()) - set(range(current_start, current_start + 32))
            assert set(state.selected_regular.tolist()) == expected
        if event.forward_kind == "boundary_refresh":
            observations.append(event.block_index)

    _, _, trace, _ = generate(
        model(), torch.full((1, 128), 2), steps=96, gen_length=96,
        block_length=32, mask_id=7, suppressed_candidate_token_ids=(7,),
        tau_high=0., tau_low=0., confirm_tau=0., selective_a4_direct_tau=0.,
        budget_scale=2., cache_initialization_activation_policy="a4", packed_attention_refresh=True,
        packed_attention_layer_mode="all", packed_attention_target_active_rows=27.5,
        cross_block_cache_handoff=True, cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=64, row_precision_context=RowPrecisionContext(),
        cross_block_dependency_policy=policy, state_capture_callback=capture,
    )
    assert observations == [1, 2]
    boundaries = [event for event in trace if event.forward_kind == "boundary_refresh"]
    for event, expected in zip(boundaries, scores):
        assert dict(zip(event.boundary_context_selected_positions,
                        event.boundary_context_selected_scores)) == {
            position: score / 255.0 for position, score in expected.items()
        }


def test_replay_requires_one_global_boundary_layer_before_changing_cache():
    replay = ForwardReplayer(None, None, device=torch.device("cpu"))
    cache = ((torch.tensor([123]),),)
    replay.past_key_values = cache
    with pytest.raises(ValueError, match="one full global boundary layer"):
        replay.step(SimpleNamespace(boundary_global_layers=2))
    assert replay.past_key_values is cache and replay.next_capture_index == 0


def test_full_sequence_precision_scope_restores_after_failure():
    context = RowPrecisionContext()
    before = torch.tensor([8, 4], dtype=torch.int8)
    context.activate(before)
    with pytest.raises(RuntimeError, match="injected"):
        with context.using(torch.tensor([4, 4], dtype=torch.int8)):
            raise RuntimeError("injected")
    assert torch.equal(context.row_bits, before)


@pytest.mark.parametrize("budget,later_rows", [(352, 128), (352, 256), (512, 64), (768, 64)])
@torch.no_grad()
def test_sparse_boundary_rows_precision_and_replay(budget, later_rows):
    original, replay_model = model(), model()
    deep_bits = 4 if budget == 352 else 8
    context = RowPrecisionContext()
    observed, captures = [], []
    for name, module in original.named_modules():
        if ".blocks." in name and name.rsplit(".", 1)[-1] in {
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_out",
            "ff_proj",
            "up_proj",
            "ff_out",
        }:

            def observe(_module, args):
                rows = args[0].numel() // args[0].shape[-1]
                observed.append(context.require(rows, args[0].device).clone())

            module.register_forward_pre_hook(observe)
    _, _, trace, _ = generate(
        original,
        torch.full((1, 512), 2),
        steps=96,
        gen_length=96,
        block_length=32,
        mask_id=7,
        suppressed_candidate_token_ids=(7,),
        tau_high=0.0,
        tau_low=0.0,
        confirm_tau=0.0,
        selective_a4_direct_tau=0.0,
        row_precision_context=context,
        cache_initialization_activation_policy="a4",
        packed_attention_refresh=True,
        packed_attention_layer_mode="all",
        packed_attention_target_active_rows=27.5,
        packed_attention_initial_burst_rows=0,
        cross_block_cache_handoff=True,
        cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=later_rows,
        cross_block_full_prefix_oracle_block=1,
        cross_block_block_initialization_deep_rows=budget,
        cross_block_block_initialization_deep_bits=deep_bits,
        cross_block_block_initialization_dependency_only=budget == 352,
        cross_block_block_initialization_include_future_dependency=budget == 352,
        cross_block_block_initialization_protected_future_blocks=1 if budget == 352 else -1,
        cross_block_block_initialization_dependency_tiebreak=True,
        question_start_token=480,
        cross_block_block_initialization_protect_generation=True,
        cross_block_block_initialization_all_a8=True,
        cross_block_block_initialization_keep_global_l0_cache=True,
        state_capture_callback=captures.append,
    )
    assert all(
        bits.numel() == 608 and bool((bits == 4).all()) for bits in observed[:14]
    )
    index = next(i for i, e in enumerate(trace) if e.forward_kind == "boundary_refresh")
    boundary = trace[index]
    assert boundary.block_index == 1
    assert boundary.layer0_keep_global_cache
    assert boundary.layer0_rows == 608
    assert boundary.active_rows == min(budget, 608)
    assert boundary.active_a8_rows == (boundary.active_rows if deep_bits == 8 else 0)
    assert boundary.active_a4_rows == (boundary.active_rows if deep_bits == 4 else 0)
    assert set(range(480, 608)) <= set(captures[index].input_positions.tolist())
    for i, bits in enumerate(observed[index * 14 : (index + 1) * 14]):
        assert bits.numel() == (608 if i < 7 else min(budget, 608))
        assert bool((bits == (8 if i < 7 else deep_bits)).all())
    later = next(
        e for e in trace if e.block_index == 2 and e.forward_kind == "boundary_refresh"
    )
    assert later.active_rows == later_rows
    assert not later.layer0_keep_global_cache
    assert all(
        e.cache_initialization_activation_policy == "default"
        for e in trace
        if e.forward_kind != "full_sequence"
    )
    replayer = ForwardReplayer(
        replay_model, RowPrecisionContext(), device=torch.device("cpu")
    )
    for event in captures:
        result = replayer.step(event)
        expected = event.teacher_top1_token_ids[:, event.prediction_mask[0]]
        assert torch.equal(
            result.prediction_logits.argmax(-1).to(torch.int32), expected
        )


@pytest.mark.parametrize("rows,scope", [(257, "layer0_attention_two_stage"), (128, "tri")])
def test_uniform_boundary_budget_rejects_unsupported_scope(rows, scope):
    with pytest.raises(ValueError, match="scope-specific range"):
        generate(
            model(), torch.full((1, 128), 2), steps=32, gen_length=32,
            block_length=32, mask_id=7, packed_attention_refresh=True,
            cross_block_cache_handoff=True, cross_block_boundary_scope=scope,
            cross_block_boundary_target_rows=rows,
        )


@pytest.mark.parametrize("scout_bits", [(8, 8), (4, 8)])
@torch.no_grad()
def test_quantized_boundary_clipping_replay_and_restore(scout_bits):
    from copy import deepcopy
    from capture.layers import raw_equal
    from generation.engine import _layer0_global_then_attention_guided_deep
    from generation.records import ForwardCaptureEvent
    from numerics.linear_kernels import LinearNumericWorkspace
    from numerics.precision import clip_a4_rows_bf16
    from quantization.model import SpinQuantW4A8Linear, TARGET_BLOCK_LINEARS
    from quantization.numeric import SpinQuantW4Tensor

    original = model()
    context = RowPrecisionContext()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        for block in original.model.transformer.blocks:
            for name in TARGET_BLOCK_LINEARS:
                linear = getattr(block, name)
                replacement = SpinQuantW4A8Linear(
                    SpinQuantW4Tensor(
                        torch.randint(-7, 8, linear.weight.shape, dtype=torch.int8),
                        torch.full((linear.out_features,), 0.015625, dtype=torch.bfloat16),
                    ),
                    LinearNumericWorkspace(), context, module_name=name,
                )
                replacement._target_a4_clip_ratio = 0.5

                def clip(module, args):
                    values = args[0]
                    bits = module.row_precision_context.require(
                        values.numel() // values.shape[-1], values.device
                    )
                    return (clip_a4_rows_bf16(values, bits, module._target_a4_clip_ratio),)

                replacement.register_forward_pre_hook(clip)
                setattr(block, name, replacement)
    # Small W4 layers isolate replay; test_clipping_capture covers production R4 ordering.
    replay_model = deepcopy(original)
    tokens = torch.tensor([[2, 3, 4, 2, 3, 4, 2, 3]])
    context.activate(torch.full((8,), 4, dtype=torch.int8))
    full_sequence = original(input_ids=tokens, use_cache=True).past_key_values
    current = torch.tensor([4, 5])
    result = _layer0_global_then_attention_guided_deep(
        original, context, stale_cache=deepcopy(full_sequence), tokens=tokens,
        current_positions=current, current_row_bits=torch.tensor(scout_bits, dtype=torch.int8),
        target_rows=4, deep_candidate_positions=torch.tensor([0, 1]),
        global_context_bits=8, global_prefix_bits=8,
        keep_global_layer0_cache=True, deep_uniform_bits=4, deep_clip_ratio=0.80078125,
    )
    event = ForwardCaptureEvent(
        capture_index=0, block_index=1, step_index=0, forward_kind="boundary_refresh",
        nfe_before=1, cache_initialized_before=True, tokens_before=tokens,
        block_state_before=torch.zeros((1, 2), dtype=torch.int8),
        model_input_ids=tokens.index_select(1, result[2]), input_positions=result[2],
        refresh_positions=result[2], row_bits=result[3], prediction_positions=current,
        prediction_mask=torch.ones((1, 2), dtype=torch.bool),
        teacher_top1_token_ids=torch.empty(0, dtype=torch.long),
        teacher_top2_token_ids=torch.empty(0, dtype=torch.long),
        teacher_top1_margin=torch.empty(0), teacher_top1_confidence=torch.empty(0),
        teacher_top1_action_confidence=torch.empty(0), teacher_current_token_probability=torch.empty(0),
        layer0_global_selected_deep=True, layer0_keep_global_cache=True,
        boundary_deep_bits=4, boundary_deep_clip_ratio=0.80078125,
        layer0_current_row_bits=scout_bits,
        layer1_global_context_bits=8, layer1_global_prefix_bits=8,
    )
    replay_context = replay_model.model.transformer.blocks[0].q_proj.row_precision_context
    replayer = ForwardReplayer(replay_model, replay_context, device=torch.device("cpu"))
    replayer.past_key_values = deepcopy(full_sequence)
    actual = replayer.step(event)
    assert raw_equal(
        result[0].index_select(1, torch.searchsorted(result[2], current)), actual.prediction_logits
    )
    for expected_layer, actual_layer in zip(result[1], replayer.past_key_values):
        assert all(raw_equal(x, y) for x, y in zip(expected_layer, actual_layer))
    assert all(
        getattr(block, name)._target_a4_clip_ratio == 0.5
        for block in original.model.transformer.blocks for name in TARGET_BLOCK_LINEARS
    )


def test_block_specific_boundary_budgets_are_not_a_public_argument():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        generate(
            None, torch.full((1, 512), 2),
            cross_block_block_initialization_deep_rows=352,
            cross_block_boundary_early_rows="160:160",
            cross_block_boundary_scope="layer0_attention_two_stage",
            cross_block_full_prefix_oracle_block=1,
        )


def test_temporary_clipping_restores_after_failure():
    from quantization.model import TARGET_BLOCK_LINEARS, using_target_block_a4_clip_ratio

    block = SimpleNamespace(**{
        name: SimpleNamespace(_target_a4_clip_ratio=0.5) for name in TARGET_BLOCK_LINEARS
    })
    with pytest.raises(RuntimeError, match="injected"):
        with using_target_block_a4_clip_ratio(block, 0.8):
            assert all(getattr(block, name)._target_a4_clip_ratio == 0.80078125
                       for name in TARGET_BLOCK_LINEARS)
            raise RuntimeError("injected")
    assert all(getattr(block, name)._target_a4_clip_ratio == 0.5 for name in TARGET_BLOCK_LINEARS)


def test_question_boundary_uses_input_metadata():
    from evaluation.model import _question_start_token

    text = "prefix\nQuestion: What is 2+2?\nAnswer:"
    offsets = [(i, i + 1) for i in range(len(text))]
    request = SimpleNamespace(doc={"question": "What is 2+2?", "answer": "unused"})
    assert _question_start_token(request, text, offsets) == 7
    with pytest.raises(ValueError, match="question metadata"):
        _question_start_token(
            SimpleNamespace(doc={"question": "other"}), text, offsets
        )


def test_generation_trace_call_matches_writer_signature():
    from evaluation.model import QuantizedLLaDALM

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(QuantizedLLaDALM.generate_until))
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_write_trace"
    ]
    assert calls
    for call in calls:
        inspect.signature(QuantizedLLaDALM._write_trace).bind(
            object(),
            *[object() for _ in call.args],
            **{keyword.arg: object() for keyword in call.keywords},
        )


def test_serialized_trace_retains_actual_scout_precision():
    from evaluation.model import QuantizedLLaDALM
    from generation.records import GenerationTraceEvent

    event = GenerationTraceEvent(
        block_index=1, step_index=0, forward_kind="boundary_refresh",
        input_start=0, input_length=2, cache_sequence_length=8,
        transferred_positions=(), direct_locked_positions=(),
        stable_tentative_positions=(), fallback_tentative_positions=(),
        confirmed_positions=(), remasked_positions=(), forced_finish_positions=(),
        masked_before=2, masked_after=2, tentative_before=0, tentative_after=0, nfe=2,
        layer0_current_row_bits=(4, 8), boundary_deep_bits=4,
    )
    tree = ast.parse(textwrap.dedent(inspect.getsource(QuantizedLLaDALM._write_trace)))
    assignment = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "compact_trace"
                for target in node.targets)
    )
    rows = eval(compile(ast.Expression(assignment.value), "<trace serialization>", "eval"),
                {"trace": [event], "self": SimpleNamespace(
                    generation_mode="feature1_packed_feature2_fused_dynamic_block")})
    assert rows[0]["layer0_current_row_bits"] == (4, 8)
    assert rows[0]["boundary_deep_bits"] == 4


@torch.no_grad()
def test_final_confirmation_keeps_unchanged_remasks_out_of_dependency_updates():
    result = generate(
        model(), torch.full((1,128),2), steps=96, gen_length=96, block_length=32,
        mask_id=7, suppressed_candidate_token_ids=(7,),
        tau_high=.99, tau_low=.99, confirm_tau=.99,
        row_precision_context=RowPrecisionContext(), cache_initialization_activation_policy="a4",
        packed_attention_refresh=True, packed_attention_layer_mode="all",
        packed_attention_target_active_rows=27.5, cross_block_cache_handoff=True,
        cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=64, cross_block_full_prefix_oracle_block=1,
        cross_block_block_initialization_deep_rows=160, cross_block_block_initialization_deep_bits=4,
        cross_block_block_initialization_dependency_only=True,
        cross_block_block_initialization_dependency_tiebreak=True,
        cross_block_block_initialization_protect_generation=True,
        cross_block_block_initialization_all_a8=True,
        cross_block_block_initialization_keep_global_l0_cache=True,
    )
    assert any(event.forced_finish_positions for event in result[2])
    assert result[3].one_step_remask_tokens > 0
