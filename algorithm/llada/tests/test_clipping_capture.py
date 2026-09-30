"""Clipping observations retain deployed arithmetic and installation state."""

import pytest
import torch
from numerics.linear_kernels import LinearNumericWorkspace
from capture.layers import LayerObserver, clone_tree, raw_equal
from numerics.precision import RowPrecisionContext, clip_a4_rows_bf16
from quantization.model import (
    TARGET_BLOCK_LINEARS,
    SpinQuantW4A8Linear,
    _install_target_r4,
    describe_target_block_a4_clipping,
    install_target_block_a4_clipping,
)
from quantization.numeric import SpinQuantW4Tensor


def small_block(*, r4=True):
    block = torch.nn.Module()
    context = RowPrecisionContext()
    context.activate(torch.tensor([4, 8], dtype=torch.int8))
    for name in TARGET_BLOCK_LINEARS:
        setattr(
            block,
            name,
            SpinQuantW4A8Linear(
                SpinQuantW4Tensor(
                    torch.tensor([[1, -1]], dtype=torch.int8),
                    torch.ones(1, dtype=torch.bfloat16),
                ),
                LinearNumericWorkspace(),
                context,
                module_name=name,
            ),
        )
    if r4:
        _install_target_r4(block)
    return block


def test_evaluator_installs_initialization_clipping_with_unclipped_regular_steps(monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from evaluation import model as evaluator
    from quantization.model import using_target_block_a4_clip_ratio, _target_a4_clip_input

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.transformer = transformer = torch.nn.Module()
    transformer.wte = torch.nn.Embedding(2, 2)
    transformer.ln_f = torch.nn.Linear(1, 1, bias=False)
    transformer.blocks = torch.nn.ModuleList([small_block() for _ in range(32)])
    for block in transformer.blocks:
        block.attn_norm = torch.nn.Linear(1, 1, bias=False)
        block.ff_norm = torch.nn.Linear(1, 1, bias=False)
    monkeypatch.setattr(evaluator.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(evaluator.LLaDAConfig, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(evaluator.LLaDAModelLM, "_from_config", lambda *a, **k: model)
    monkeypatch.setattr(evaluator, "_sha256_file", lambda *a: None)
    monkeypatch.setattr(evaluator, "checkpoint_identity", lambda *a: {})
    monkeypatch.setattr(evaluator, "SpinQuantArtifactReader", lambda *a, **k: SimpleNamespace(
        manifest_sha256="test", transformer_weight_arithmetic="gminus1"))
    monkeypatch.setattr(evaluator, "replace_with_spinquant_joint_full_w4a8_v8",
                        lambda *a, **k: {"dynamic_a4a8_linear": 224})
    monkeypatch.setattr(evaluator, "build_target_numeric_coverage",
                        lambda *a, **k: {"valid": True, "transformer_activation": {}})
    settings = json.loads((Path(evaluator.__file__).parents[1] / "configs/gsm8k.json").read_text())["model_args"]
    settings.update(a4_clip_ratio=1.0, a4_output_clip_ratio=1.0)
    evaluator.QuantizedLLaDALM(model_path="/unused", spinquant_artifact_dir="/unused",
                             device="cpu", **settings)
    values = torch.tensor([[[4.0, -2.0], [-0.0, -8.0]]], dtype=torch.bfloat16)
    for block in transformer.blocks:
        assert all(getattr(block, name)._target_a4_clip_ratio == 1 for name in TARGET_BLOCK_LINEARS)
        assert raw_equal(_target_a4_clip_input(block.q_proj, (values,))[0], values)
        with using_target_block_a4_clip_ratio(block, 0.80078125):
            actual = _target_a4_clip_input(block.q_proj, (values,))[0]
            expected = clip_a4_rows_bf16(values, torch.tensor([4, 8], dtype=torch.int8), 0.80078125)
            assert raw_equal(actual, expected)
            assert not raw_equal(actual[0, 0], values[0, 0])
            assert raw_equal(actual[0, 1], values[0, 1])
        assert all(getattr(block, name)._target_a4_clip_ratio == 1 for name in TARGET_BLOCK_LINEARS)


def test_actual_clipping_observations_and_a8_identity():
    values = torch.tensor([[[4.0, -2.0], [-0.0, -8.0]]], dtype=torch.bfloat16)
    bits = torch.tensor([4, 8], dtype=torch.int8)
    ratio = float(torch.tensor(0.8, dtype=torch.bfloat16))
    baseline = clip_a4_rows_bf16(values, bits, ratio)
    captured = []
    observed = clip_a4_rows_bf16(
        values, bits, ratio, observer=lambda fields: captured.append(clone_tree(fields))
    )
    fields = captured[0]
    assert raw_equal(baseline, observed)
    assert raw_equal(observed[0, 1], values[0, 1])
    assert not raw_equal(observed[0, 0], values[0, 0])
    assert fields["row_max"].dtype == fields["limit_fp32"].dtype == torch.float32
    assert fields["row_max"].shape == (2, 1)
    assert fields["row_max"].flatten().tolist() == [4.0, 8.0]
    assert raw_equal(fields["limit"], fields["limit_fp32"].bfloat16())
    original = fields["input"].clone()
    values.zero_()
    assert raw_equal(fields["input"], original)


@pytest.mark.parametrize("raw,ratio,limit", [(5, 0.80078125, 4), (1, 0.5, 0)])
def test_clipping_rounding_and_zero_limit_are_observed(raw, ratio, limit):
    values = (
        torch.tensor([raw], dtype=torch.int16).view(torch.bfloat16).reshape(1, 1, 1)
    )
    captured = []
    output = clip_a4_rows_bf16(
        values,
        torch.tensor([4], dtype=torch.int8),
        ratio,
        observer=lambda fields: captured.append(clone_tree(fields)),
    )
    assert captured[0]["limit"].view(torch.int16).item() == limit
    assert output.view(torch.int16).item() == limit


def test_single_block_mixed_installation_and_down_order(monkeypatch):
    transform = torch.tensor([2.0, 0.5], dtype=torch.bfloat16)
    monkeypatch.setattr(
        "quantization.model.structured_hadamard_12288_bf16",
        lambda values: values * transform,
    )
    block = small_block()
    ratios = dict.fromkeys(TARGET_BLOCK_LINEARS, None)
    ratios.update(q_proj=1.0, ff_out=0.8)
    install_target_block_a4_clipping(block, ratios)
    config = describe_target_block_a4_clipping(block)
    assert config["q_proj"]["installed"] and config["q_proj"]["ratio_bf16"].item() == 1
    assert not config["k_proj"]["installed"] and config["k_proj"]["ratio_bf16"] is None
    events = []
    block.ff_out._target_r4_observer = lambda value: events.append(
        ("r4", clone_tree(value))
    )
    block.ff_out._target_a4_clip_observer = lambda fields: events.append(
        ("clip", clone_tree(fields))
    )
    values = torch.tensor([[4.0, -2.0], [8.0, -1.0]], dtype=torch.bfloat16)
    actual = block.ff_out(values)
    assert [event[0] for event in events] == ["r4", "clip"]
    assert raw_equal(events[0][1], events[1][1]["input"])
    expected = block.ff_out.forward(events[1][1]["output"])
    assert raw_equal(actual, expected)
    assert raw_equal(events[1][1]["output"][1], (values * transform)[1])


def test_installation_validates_before_mutation_and_rejects_missing_r4():
    block = small_block(r4=False)
    with pytest.raises(ValueError, match="target R4"):
        install_target_block_a4_clipping(
            block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8)
        )
    assert not any(
        (
            hasattr(getattr(block, n), "_target_a4_clip_ratio")
            for n in TARGET_BLOCK_LINEARS
        )
    )
    _install_target_r4(block)
    invalid = dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8)
    invalid["ff_out"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        install_target_block_a4_clipping(block, invalid)
    assert not hasattr(block.q_proj, "_target_a4_clip_ratio")
    with pytest.raises(ValueError, match="seven"):
        install_target_block_a4_clipping(block, {"q_proj": 0.8})
    install_target_block_a4_clipping(block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8))
    with pytest.raises(ValueError, match="already installed"):
        install_target_block_a4_clipping(
            block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8)
        )
    with pytest.raises(ValueError, match="once"):
        _install_target_r4(block)


def test_installation_failure_removes_only_new_hooks(monkeypatch):
    block = small_block()
    r4_id = block.ff_out._target_r4_hook_handle.id

    def fail(*args, **kwargs):
        raise RuntimeError("injected registration failure")

    monkeypatch.setattr(block.k_proj, "register_forward_pre_hook", fail)
    with pytest.raises(RuntimeError, match="injected"):
        install_target_block_a4_clipping(
            block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8)
        )
    assert not block.q_proj._forward_pre_hooks
    assert not hasattr(block.q_proj, "_target_a4_clip_ratio")
    assert list(block.ff_out._forward_pre_hooks) == [r4_id]


def test_descriptor_detects_removed_or_reordered_hooks():
    block = small_block()
    install_target_block_a4_clipping(block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8))
    hooks = block.ff_out._forward_pre_hooks
    hooks.move_to_end(block.ff_out._target_r4_hook_handle.id)
    with pytest.raises(ValueError, match="follow target R4"):
        describe_target_block_a4_clipping(block)
    hooks.move_to_end(block.ff_out._target_a4_clip_hook_handle.id)
    block.q_proj._target_a4_clip_hook_handle.remove()
    with pytest.raises(ValueError, match="unsupported clipping"):
        describe_target_block_a4_clipping(block)


def test_observer_entry_failure_restores_hooks_and_callbacks():
    block = small_block()
    install_target_block_a4_clipping(block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8))
    original = {
        name: tuple(getattr(block, name)._forward_pre_hooks)
        for name in TARGET_BLOCK_LINEARS
    }
    with pytest.raises(AttributeError):
        with LayerObserver(block):
            pass
    for name in TARGET_BLOCK_LINEARS:
        module = getattr(block, name)
        assert tuple(module._forward_pre_hooks) == original[name]
        assert not hasattr(module, "_target_a4_clip_observer")
    assert not hasattr(block.ff_out, "_target_r4_observer")
    describe_target_block_a4_clipping(block)


def test_observer_does_not_replace_existing_callback():
    block = small_block()
    install_target_block_a4_clipping(block, dict.fromkeys(TARGET_BLOCK_LINEARS, 0.8))
    existing = lambda fields: None
    block.k_proj._target_a4_clip_observer = existing
    with pytest.raises(RuntimeError, match="another observer"):
        with LayerObserver(block):
            pass
    assert block.k_proj._target_a4_clip_observer is existing
    assert not hasattr(block.q_proj, "_target_a4_clip_observer")
