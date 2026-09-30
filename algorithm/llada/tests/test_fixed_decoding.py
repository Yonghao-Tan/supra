"""Irreversible decoding keeps Feature1 refresh independently of Feature2 states."""

from dataclasses import asdict
import json
from pathlib import Path
import pytest
import torch
from generation.engine import generate, select_irreversible_transfers


def test_fixed_k_masks_ties_and_short_tail():
    mask = torch.tensor([[True, False, True, True], [False, True, False, False]])
    confidence = torch.tensor([[0.5, 1.0, 0.5, 0.4], [1.0, 0.2, 1.0, 1.0]])
    chosen = select_irreversible_transfers(mask, confidence, mode="fixed_k", k=2)
    assert chosen.tolist() == [[True, False, True, False], [False, True, False, False]]


def test_threshold_fp32_comparison_and_progress_fallback():
    confidence = torch.tensor(
        [[0.8984375, 0.90234375, 0.5], [0.5, 0.5, 0.25], [1.0, 1.0, 1.0]],
        dtype=torch.bfloat16,
    )
    mask = torch.tensor([[True, True, True], [True, True, True], [False, False, False]])
    selected = select_irreversible_transfers(
        mask, confidence, mode="fixed_threshold", threshold=0.9
    )
    assert selected.tolist() == [
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ]


@pytest.mark.parametrize(
    "mode,threshold",
    [("fixed_k", 0.9), ("fixed_threshold", 1.0), ("fixed_threshold", 0.0)],
)
def test_real_generator_has_no_reversible_transition_or_a4(mode, threshold):
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import (
        ActivationType,
        BlockType,
        LLaDAModel,
        LLaDAModelLM,
        ModelConfig,
    )
    from numerics.precision import RowPrecisionContext

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
            max_sequence_length=128,
            vocab_size=8,
            embedding_size=8,
            rope=True,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            use_manual_attention=True,
        )
        model = LLaDAModelLM(
            LLaDAConfig(**asdict(config)), model=LLaDAModel(config)
        ).eval()
    (captures, ends) = ([], [])
    (_, _, trace, stats) = generate(
        model,
        torch.tensor([[2]]),
        steps=64,
        gen_length=64,
        block_length=32,
        mask_id=7,
        decoding_mode=mode,
        decode_k=3,
        decode_threshold=threshold,
        row_precision_context=RowPrecisionContext(),
        feature3_precision_policy="all_a8",
        packed_attention_refresh=True,
        packed_attention_layer_mode="all",
        packed_attention_target_active_rows=27.5,
        packed_attention_initial_burst_rows=0,
        cross_block_cache_handoff=True,
        cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=64,
        cross_block_full_prefix_oracle_block=1,
        cross_block_boundary_context_row_bits=8,
        cross_block_boundary_deep_a8_row_limit=-1,
        state_capture_callback=captures.append,
        forward_end_callback=ends.append,
    )
    assert captures and all((bool((event.row_bits == 8).all()) for event in captures))
    assert all(
        (not event.active_a4_rows and (not event.layer0_a4_rows) for event in trace)
    )
    assert all(
        (
            not event.tentative_after and (not event.remasked_positions)
            for event in trace
        )
    )
    assert stats.one_step_confirm_lock_tokens == stats.one_step_remask_tokens == 0
    assert stats.stable_tentative_tokens == stats.fallback_tentative_tokens == 0
    for block in range(2):
        decisions = [event["trace"] for event in ends if event["block_index"] == block]
        counts = [len(event.direct_locked_positions) for event in decisions]
        expected = (
            [3] * 10 + [2]
            if mode == "fixed_k"
            else [1] * 32
            if threshold == 1
            else [32]
        )
        assert counts == expected


@pytest.mark.parametrize("mode", ["fixed_k", "fixed_threshold"])
@pytest.mark.parametrize("optimized_refresh", [False, True])
def test_fixed_evaluator_reaches_loading(monkeypatch, mode, optimized_refresh):
    from evaluation import model as evaluator

    settings = json.loads(
        (Path(__file__).parents[1] / "configs/gsm8k.json").read_text()
    )["model_args"]
    settings = {
        key: value
        for (key, value) in settings.items()
        if (not key.startswith(("feature2_", "feature3_"))
            or key == "feature2_cache_initialization_activation_policy")
        and (optimized_refresh or not key.startswith("feature1_cross_block_block_initialization_"))
    }
    settings.update(
        generation_mode="feature1_" + mode,
        feature3_precision_policy="all_a8",
    )
    if not optimized_refresh:
        settings.update(feature2_cache_initialization_activation_policy="default", a4_clip_ratio=1.0,
                        a4_output_clip_ratio=1.0,
                        feature1_cross_block_boundary_deep_a8_row_limit=-1)
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator, "checkpoint_identity", lambda _: {})

    def stop(*args, **kwargs):
        raise RuntimeError("reached tokenizer")

    monkeypatch.setattr(evaluator.AutoTokenizer, "from_pretrained", stop)
    with pytest.raises(RuntimeError, match="reached tokenizer"):
        evaluator.QuantizedLLaDALM(
            model_path="/unused", spinquant_artifact_dir="/unused", **settings
        )


@pytest.mark.parametrize("mode", ["fixed_k", "fixed_threshold"])
def test_fixed_decoding_preserves_a4_full_sequence_sparse_block_initialization_and_boundary(mode):
    from test_cache_initialization import model
    from numerics.precision import RowPrecisionContext

    captures = []
    _, _, trace, stats = generate(
        model(), torch.full((1, 128), 2), steps=96, gen_length=96,
        block_length=32, mask_id=7, suppressed_candidate_token_ids=(7,),
        decoding_mode=mode, decode_k=1, decode_threshold=.95,
        row_precision_context=RowPrecisionContext(), feature3_precision_policy="all_a8",
        cache_initialization_activation_policy="a4", packed_attention_refresh=True,
        packed_attention_layer_mode="all", packed_attention_target_active_rows=27.5,
        cross_block_cache_handoff=True,
        cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=88, cross_block_boundary_deep_a8_row_limit=40,
        cross_block_full_prefix_oracle_block=1, cross_block_block_initialization_deep_rows=160,
        cross_block_block_initialization_dependency_only=True,
        cross_block_block_initialization_dependency_tiebreak=True,
        cross_block_block_initialization_include_future_dependency=True,
        cross_block_block_initialization_relative_score_floor=.125,
        cross_block_block_initialization_protect_generation=True,
        cross_block_block_initialization_protected_future_blocks=1,
        cross_block_block_initialization_all_a8=True, cross_block_block_initialization_deep_bits=4,
        cross_block_block_initialization_keep_global_l0_cache=True,
        cross_block_dependency_policy="current_keys_committed_rows",
        cross_block_pending_confidence_mode="all_changes",
        state_capture_callback=captures.append,
    )
    assert trace[0].layer0_a4_rows == 224 and trace[0].layer0_a8_rows == 0
    block_initialization = next(e for e in trace if e.block_index == 1)
    assert block_initialization.forward_kind == "boundary_refresh"
    assert block_initialization.layer0_a8_rows == 224 and block_initialization.layer0_a4_rows == 0
    assert 32 <= block_initialization.active_rows <= 160
    assert block_initialization.active_a4_rows == block_initialization.active_rows and block_initialization.active_a8_rows == 0
    boundary = next(e for e in trace if e.block_index == 2)
    assert boundary.active_rows == 88 and boundary.active_a8_rows == 40
    regular = [e for e in trace if e.forward_kind == "local_block"]
    assert regular and all(e.active_a4_rows == 0 for e in regular)
    assert all(not e.tentative_after and not e.remasked_positions and
               not e.next_admitted_positions for e in trace)
    assert stats.one_step_confirm_lock_tokens == stats.one_step_remask_tokens == 0
    assert stats.stable_tentative_tokens == stats.fallback_tentative_tokens == 0
    assert all(len(e.direct_locked_positions) == 1 for e in trace)
    assert stats.nfe == 96
    assert len(captures) == len(trace)
    assert bool((captures[0].row_bits == 4).all())
    for event, capture in zip(trace, captures):
        if event is block_initialization:
            assert bool((capture.row_bits == 4).all())
        elif event.forward_kind == "local_block":
            assert bool((capture.row_bits == 8).all())


def test_baseline_recomputes_full_sequence_with_selected_head():
    from test_cache_initialization import model
    from numerics.precision import RowPrecisionContext

    network = model()
    inputs, head_rows, captures, starts = [], [], [], []
    def before_model(_, args, kwargs):
        assert starts[-1]["capture_index"] == len(inputs)
        inputs.append((args[0].shape[1], kwargs.get("past_key_values")))
    network.register_forward_pre_hook(
        before_model,
        with_kwargs=True,
    )
    network.model.transformer.ln_f.register_forward_pre_hook(
        lambda _, args: head_rows.append(args[0].shape[1])
    )
    tokens, nfe, trace, stats = generate(
        network, torch.tensor([[2, 2]]), steps=64, gen_length=64, block_length=32,
        mask_id=7, suppressed_candidate_token_ids=(7,), decoding_mode="fixed_k", decode_k=3,
        feature3_precision_policy="all_a8", row_precision_context=RowPrecisionContext(),
        full_sequence_recompute=True, state_capture_callback=captures.append,
        forward_start_callback=starts.append,
    )
    assert nfe == 22 and tokens.shape == (1, 66)
    assert inputs == [(66, None)] * nfe
    assert head_rows == list(range(32, 0, -3)) * 2
    assert all(not c.cache_initialized_before and c.input_positions.numel() == 66 for c in captures)
    assert all(bool((c.row_bits == 8).all()) for c in captures)
    assert all(e.input_length == 66 and e.active_a8_rows == 66 and not e.active_a4_rows for e in trace)
    assert not stats.one_step_remask_tokens
    assert sum(len(e.direct_locked_positions) for e in trace) == 64
    assert starts == [dict(capture_index=i, block_index=c.block_index, step_index=c.step_index,
                           forward_kind=c.forward_kind) for i, c in enumerate(captures)]
    alone = []
    plain = generate(model(), torch.tensor([[2, 2]]), steps=64, gen_length=64, block_length=32,
        mask_id=7, suppressed_candidate_token_ids=(7,), decoding_mode="fixed_k", decode_k=3,
        feature3_precision_policy="all_a8", row_precision_context=RowPrecisionContext(),
        full_sequence_recompute=True, forward_start_callback=alone.append)
    assert alone == starts and torch.equal(plain[0], tokens) and plain[1:] == (nfe, trace, stats)
    from capture.replay import ForwardReplayer
    replay = ForwardReplayer(model(), RowPrecisionContext(), device=torch.device("cpu"))
    for captured in captures:
        result = replay.step(captured)
        expected = captured.teacher_top1_token_ids[captured.prediction_mask].reshape(1, -1)
        assert torch.equal(result.prediction_logits.argmax(dim=-1).cpu(), expected)


def test_logits_positions_match_full_logits_and_preserve_cache():
    from test_cache_initialization import model

    network = model()
    tokens = torch.tensor([[2, 3, 4, 2, 3, 4]])
    positions = torch.tensor([4, 1, 3])
    with torch.no_grad():
        full = network(tokens, use_cache=True)
        selected = network(tokens, use_cache=True, logits_positions=positions)
        repeated = network(tokens, use_cache=True)
    torch.testing.assert_close(full.logits, repeated.logits, rtol=0, atol=0)
    torch.testing.assert_close(selected.logits, full.logits.index_select(1, positions), rtol=0, atol=0)
    for full_layer, selected_layer in zip(full.past_key_values, selected.past_key_values):
        for a, b in zip(full_layer, selected_layer):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="combined"):
        network(tokens, logits_positions=positions, logits_start=0, logits_end=2)


@pytest.mark.parametrize("tier", ["baseline", "feature1", "feature12", "feature123"])
def test_tier_evaluator_reaches_loading(monkeypatch, tier):
    from evaluation import model as evaluator
    from evaluation.generate import resolve_model_args

    config = json.loads((Path(__file__).parents[1] / "configs/gsm8k.json").read_text())
    settings = resolve_model_args(config, tier=tier)
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator, "checkpoint_identity", lambda _: {})
    def stop(*args, **kwargs):
        raise RuntimeError("reached tokenizer")
    monkeypatch.setattr(evaluator.AutoTokenizer, "from_pretrained", stop)
    with pytest.raises(RuntimeError, match="reached tokenizer"):
        evaluator.QuantizedLLaDALM(model_path="/unused", spinquant_artifact_dir="/unused", **settings)


@pytest.mark.parametrize("tier", ["baseline", "feature1", "feature12", "feature123"])
@pytest.mark.parametrize("length", [32, 64, 256])
def test_short_generation_flows_through_evaluator(monkeypatch, tmp_path, tier, length):
    import inspect
    from types import SimpleNamespace
    from lm_eval.api.model import LM
    from evaluation.model import QuantizedLLaDALM
    from evaluation.generate import resolve_model_args
    from numerics.precision import RowPrecisionContext
    from test_cache_initialization import model

    evaluator = QuantizedLLaDALM.__new__(QuantizedLLaDALM)
    LM.__init__(evaluator)
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(QuantizedLLaDALM.__init__).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    config = json.loads((Path(__file__).parents[1] / "configs/gsm8k.json").read_text())
    defaults.update(resolve_model_args(config, tier=tier, overrides={"steps": length, "gen_length": length}))
    for name, value in defaults.items():
        if not isinstance(getattr(QuantizedLLaDALM, name, None), property):
            setattr(evaluator, name, value)
    evaluator._world_size, evaluator._device, evaluator._max_length = 1, torch.device("cpu"), length + 2
    evaluator.model, evaluator.row_precision_context = model(), RowPrecisionContext()
    # This small BF16 model exercises generation length and state progression.
    # Clipping hooks are tested on quantized Linears in test_clipping_capture.py.
    evaluator.feature1_cross_block_block_initialization_deep_clip_ratio = 0.0
    evaluator.fixed_decoding = tier in {"baseline", "feature1"}
    evaluator.decoding_mode = "fixed_k" if evaluator.fixed_decoding else "feature2"
    evaluator.confidence_eos_eot_inf = False
    evaluator.mask_id = 7
    evaluator.feature2_parameters = {key.removeprefix("feature2_"): value for key, value in defaults.items() if key.startswith("feature2_")}
    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[2, 2]]), offset_mapping=torch.tensor([[[0, 10], [10, 25]]]))
        def decode(self, ids, **kwargs):
            return str(len(ids))
    evaluator.tokenizer = Tokenizer()
    evaluator.numeric_coverage = None
    evaluator.numeric_coverage_sha256 = None
    evaluator.artifact_manifest_sha256 = None
    evaluator.deployment_diagnostic = {}
    evaluator.checkpoint_identity = {}
    evaluator.trace_output_path = tmp_path / "trace.jsonl"
    recorded = []
    def write_trace(*args, **kwargs):
        recorded.append(args)
        QuantizedLLaDALM._write_trace(evaluator, *args, **kwargs)
    evaluator._write_trace = write_trace
    request = SimpleNamespace(args=("Question: prompt\nAnswer:", {"max_gen_toks": length}), doc={"question": "prompt"}, task_name="gsm8k", doc_id=0, idx=0)
    assert evaluator.max_gen_toks == length
    assert evaluator.generate_until([request]) == [str(length)]
    assert recorded[0][2].shape == (1, length + 2)
    assert recorded[0][4][0].input_length == length + 2
    serialized = json.loads(evaluator.trace_output_path.read_text())
    assert serialized["generation_protocol"]["gen_length"] == length
    assert serialized["generation_protocol"]["steps"] == length
    assert serialized["nfe"] == len(serialized["trace"])
    assert len(serialized["generated_token_ids"][0]) == length
    if tier == "baseline":
        assert all(event.input_length == length + 2 and event.active_a8_rows == length + 2 for event in recorded[0][4])
    else:
        assert recorded[0][4][0].cache_initialization_activation_policy == "a4"
    evaluator._max_length = length + 1
    with pytest.raises(ValueError, match="exceeds max_length"):
        evaluator.generate_until([request])
    request.args = ("prompt", {"max_gen_toks": length + 32})
    with pytest.raises(ValueError, match=f"configured generation requires {length}"):
        evaluator.generate_until([request])
