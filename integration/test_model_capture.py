"""Raw encoding and capture lifetime checks; real CUDA replay is in the index."""

import importlib.util
import ast
import inspect
from dataclasses import make_dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load("real_exporter_test", HERE / "model_capture.py")
observer = load("real_observer_test", exporter.OBSERVER)


def test_head_replay_calls_bind_to_numeric_kernel_signatures(monkeypatch):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    from numerics.linear_kernels import accelerated_linear_bf16
    from numerics.operator_kernels import rms_norm_bf16_cuda
    functions = {f.__name__: f for f in (accelerated_linear_bf16, rms_norm_bf16_cuda)}
    checked = set()
    for node in ast.walk(ast.parse(inspect.getsource(exporter.replay_head.__wrapped__))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            inspect.signature(functions[node.func.id]).bind(
                *[None for _ in node.args], **{item.arg: None for item in node.keywords})
            checked.add(node.func.id)
    assert checked == set(functions)














@pytest.mark.parametrize("keyword_prompt", [False, True])
@pytest.mark.parametrize("maturity_override", [None, 5])
def test_public_generation_callback_and_binding_restore(keyword_prompt, maturity_override):
    events = []
    prompt = torch.tensor([[1, 2]])
    event = object()

    def generate(*args, feature3_maturity_age=3, cross_block_boundary_context_row_bits=8, **kwargs):
        kwargs["state_capture_callback"](event)
        raise observer.CaptureComplete()

    evaluation = SimpleNamespace(generate=generate)
    capture = SimpleNamespace(on_event=lambda item: events.append(("capture", item)))
    with pytest.raises(observer.CaptureComplete):
        with exporter.generation_capture(evaluation, capture, observer):
            arguments = {"steps": 2, "state_capture_callback": lambda item: events.append(("previous", item))}
            if maturity_override is not None:
                arguments["feature3_maturity_age"] = maturity_override
            if keyword_prompt:
                evaluation.generate(model=None, prompt=prompt, **arguments)
            else:
                evaluation.generate(None, prompt, **arguments)
    assert evaluation.generate is generate
    assert events == [("previous", event), ("capture", event)]
    prompt[0, 0] = 9
    assert capture.prompt_ids.tolist() == [[1, 2]]
    assert capture.generation_config["steps"] == 2
    assert capture.generation_config["feature3_maturity_age"] == (3 if maturity_override is None else maturity_override)
    assert capture.generation_config["cross_block_boundary_context_row_bits"] == 8


def test_public_algorithm_capture_default():
    assert exporter.SOURCE.parts[-2:] == ("algorithm", "llada")
    assert exporter.OBSERVER == exporter.SOURCE / "capture/layers.py"


def test_baseline_control_observes_implicit_full_sequence_without_dependency(monkeypatch):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = SimpleNamespace(module_name="q", row_precision_context=SimpleNamespace(
                require=lambda count, device, **_: torch.full((count,), 8, device=device)))
            self.config = make_dataclass("Config", [])()

        def forward(self, x, layer_past=None, attention_bias=None):
            raise RuntimeError("stop after observed layer input")

    blocks = torch.nn.ModuleList([Block() for _ in range(32)])
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks)))
    monkeypatch.setattr(exporter, "capture_control_state", lambda **_: dict(
        full_sequence_recompute=True, total_length=34))
    capture = exporter.IndexedForwardCapture(model, observer, 0, control_state=True,
        first_layer=31, layer_count=1, on_complete=lambda _: None)
    watched = SimpleNamespace(__enter__=lambda: None, __exit__=lambda *_: None)
    monkeypatch.setattr(observer, "LayerObserver", lambda *_: watched)
    monkeypatch.setattr(exporter, "capture_layer_dependency_input",
                        lambda *_: pytest.fail("baseline queried packed dependency state"))
    hidden = torch.zeros((1, 34, 4), dtype=torch.bfloat16)
    with capture:
        capture.control["before_forward"] = dict(full_sequence_recompute=True)
        with pytest.raises(RuntimeError, match="stop after observed"):
            blocks[31](hidden)
        capture.next_index = 1
        capture.event = object()
        with pytest.raises(RuntimeError, match="stop after observed"):
            blocks[0](hidden)
    following = capture.control["next_layer_inputs"]
    assert following["query_position_ids"].tolist() == [list(range(34))]
    assert following["activation_bits"].tolist() == [8] * 34
    assert torch.equal(following["hidden"], hidden) and capture.completed


def test_multi_forward_capture_uses_one_run_and_releases_hooks():
    norm = torch.nn.LayerNorm(4)
    head = torch.nn.Identity()
    head.weight_bits = 8
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(ln_f=norm, ff_out=head)))
    event_type = make_dataclass("HeadEvent", [("capture_index", int), ("forward_kind", str)])
    completed, outputs = [], []
    calls = []

    def generate(model, prompt, **kwargs):
        calls.append(1)
        for index in range(4):
            value = head(norm(prompt.float() + index * torch.tensor([1., 2., 3., 4.])))
            outputs.append(value.detach().clone())
            if "state_capture_callback" in kwargs:
                kwargs["state_capture_callback"](event_type(index, "local_block"))

    prompt = torch.tensor([[[2., 1., -3., 4.]]])
    generate(model, prompt)
    baseline = outputs[:]
    outputs.clear(); calls.clear()
    evaluation = SimpleNamespace(generate=generate)
    selections = [dict(name="later", capture_index=3, head_only=True),
                  dict(name="earlier", capture_index=1, head_only=True)]
    capture = exporter.ForwardCaptureSet(model, observer, selections,
        lambda name, item: completed.append((name, item.target, item.head["output"].clone())))
    with pytest.raises(observer.CaptureComplete):
        with exporter.generation_capture(evaluation, capture, observer), capture:
            evaluation.generate(model, prompt)
    assert calls == [1]
    assert [name for name, _, _ in completed] == ["earlier", "later"]
    assert all(exporter.raw_equal(actual, baseline[index]) for _, index, actual in completed)
    assert all(exporter.raw_equal(a, b) for a, b in zip(outputs, baseline))
    assert evaluation.generate is generate and sys.getprofile() is None
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in (norm, head))
    assert all(item.completed and not item.head and not item.layers for item in capture.observations)


@pytest.mark.parametrize("selections", [[], [dict(name="../bad", capture_index=0)],
    [dict(name="a", capture_index=0), dict(name="b", capture_index=0)],
    [dict(name="a", capture_index=True)], [dict(name="a", capture_index=0, unexpected=True)]])
def test_multi_capture_rejects_ambiguous_or_unsafe_selections(selections):
    with pytest.raises(ValueError):
        exporter.ForwardCaptureSet(None, observer, selections, lambda *args: None)


def cpu_selection_model(monkeypatch, layers=2):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    from dataclasses import asdict
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import ActivationType, BlockType, LLaDAModel, LLaDAModelLM, ModelConfig
    from numerics.precision import RowPrecisionContext
    context = RowPrecisionContext()
    config = ModelConfig(d_model=16, n_heads=2, n_kv_heads=2, n_layers=layers,
        mlp_hidden_size=32, activation_type=ActivationType.silu, block_type=BlockType.llama,
        max_sequence_length=128, vocab_size=8, embedding_size=8, rope=True, weight_tying=False,
        attention_dropout=0., residual_dropout=0., embedding_dropout=0., use_manual_attention=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        model = LLaDAModelLM(LLaDAConfig(**asdict(config)), model=LLaDAModel(config)).eval().bfloat16()
    model.model.transformer.ff_out.weight_bits = 8
    for block in model.model.transformer.blocks:
        block.q_proj.row_precision_context = context
        block.q_proj.module_name = "q_proj"
    return model, dict(steps=64, gen_length=64, block_length=32, mask_id=7,
        suppressed_candidate_token_ids=(7,), decoding_mode="fixed_k", decode_k=3,
        full_sequence_recompute=True, feature3_precision_policy="all_a8", row_precision_context=context)


@torch.no_grad()
def test_identity_selection_observes_real_head_and_control_before_next_forward(monkeypatch):
    model, options = cpu_selection_model(monkeypatch)
    from generation.engine import generate
    evaluation = SimpleNamespace(generate=generate)
    captured = []
    capture = exporter.ForwardCaptureSet(model, observer, [dict(name="second_block",
        block_index=1, step_index=1, phase="full_sequence", head_only=True, control_state=True)],
        lambda name, item: captured.append((item.target, observer.clone_tree(item.event),
                                            observer.clone_tree(item.control), item.head["output"].clone(),
                                            json.loads(json.dumps(item.generation_config)))))
    with pytest.raises(observer.CaptureComplete):
        with exporter.generation_capture(evaluation, capture, observer), capture:
            evaluation.generate(model, torch.tensor([[2, 2]]), **options)
    assert len(captured) == 1 and captured[0][0] == 12
    _, event, control, head, generated = captured[0]
    assert not {"forward_start_callback", "state_capture_callback", "forward_end_callback"} & generated.keys()
    assert (event["block_index"], event["step_index"], event["forward_kind"]) == (1,1,"full_sequence")
    assert head.shape[1] == 29 and control["next_forward"]["capture_index"] == 13
    assert control["transition"]["step_index"] == 1
    assert torch.equal(control["after_postprocess"]["tokens"], control["next_forward"]["tokens"])
    assert evaluation.generate is generate and sys.getprofile() is None


@torch.no_grad()
def test_actual_tail_forward_callbacks_capture_confirmation_and_forced_finish(monkeypatch):
    model, options = cpu_selection_model(monkeypatch)
    from generation.engine import generate
    from hardware_adapter import control_reference_data as references
    options.update(steps=1, gen_length=32, decoding_mode="feature2", full_sequence_recompute=False,
                   feature3_precision_policy="original", tail_confirmation_policy="none",
                   tau_high=1., tau_low=1., confirm_tau=1., packed_attention_refresh=True,
                   packed_attention_layer_mode="all", packed_attention_target_active_rows=27.5,
                   packed_attention_initial_burst_rows=0.)
    baseline = generate(model, torch.tensor([[2, 2]]), **options)
    assert [event.forward_kind for event in baseline[2]] == ["full_sequence", "local_confirmation", "local_forced_finish"]
    captured = {}
    capture = exporter.ForwardCaptureSet(model, observer, [dict(name=phase,
        block_index=0, step_index=step, phase=phase, head_only=True, control_state=True)
        for step, phase in ((1, "local_confirmation"), (2, "local_forced_finish"))],
        lambda name, item: captured.update({name: (observer.clone_tree(item.control),
            observer.clone_tree(item.event), dict(item.generation_config))}))
    evaluation = SimpleNamespace(generate=generate)
    with pytest.raises(observer.CaptureComplete):
        with exporter.generation_capture(evaluation, capture, observer), capture:
            actual = evaluation.generate(model, torch.tensor([[2, 2]]), **options)
            assert torch.equal(actual[0], baseline[0]) and actual[1:] == baseline[1:]
            capture.finish_request()
    assert set(captured) == {"local_confirmation", "local_forced_finish"}
    defaults = {name: value.default for name, value in inspect.signature(generate).parameters.items()
                if value.default is not inspect.Parameter.empty}
    for phase, (control, prediction, generated) in captured.items():
        assert control["transition"]["forward_kind"] == phase
        assert control["transition"]["step_index"] == (1 if phase == "local_confirmation" else 2)
        with patch.object(references, "read_captured_control", return_value=(
                {"generation_config": {**defaults, **generated}}, control, prediction)):
            reference = references.captured_feature2_reference(Path("actual-tail"))
            if phase == "local_confirmation":
                pending = references.captured_feature1_reference(Path("actual-tail"))
                assert pending["records"][0]["expected"]["next_refresh"]["raw"] == control["after_postprocess"]["packed_state"]["refresh"].tolist()
        assert reference["config"]["closeout_kind"] == (1 if phase == "local_confirmation" else 2)
        if phase == "local_confirmation":
            assert reference["expected"]["masks"]["remasked"] == 0xffffffff
            assert control["next_forward"]["capture_index"] == 2
        else:
            assert control["request_end"] and "next_forward" not in control
            assert reference["expected"]["state"] == [2]*32
            assert reference["expected"]["masks"]["tail_closed"] == 0xffffffff


@torch.no_grad()
def test_initial_completed_block_prepares_without_unused_joint_call(monkeypatch, tmp_path):
    model, options = cpu_selection_model(monkeypatch)
    from generation.engine import generate
    from hardware_adapter import control_reference_data as references
    import prepare_layer_testcase as layers
    import prepare_head_testcase as head
    from prepare_handoff_testcase import read_record
    options.update(steps=2, gen_length=64, decoding_mode="feature2", full_sequence_recompute=False,
        tau_high=.01, tau_low=0., confirm_tau=.01, selective_a4_direct_tau=.01,
        feature3_precision_policy="original",
        packed_attention_refresh=True, packed_attention_layer_mode="all",
        packed_attention_target_active_rows=27.5, packed_attention_initial_burst_rows=0.,
        dynamic_block_lookahead=True, dynamic_block_canonical_future=True,
        cross_block_cache_handoff=True, cross_block_boundary_scope="layer0_attention_two_stage",
        cross_block_boundary_target_rows=64)
    model.model.transformer.ff_out.weight.zero_()
    saved = []
    capture = exporter.ForwardCaptureSet(model, observer,
        [dict(name="initial", capture_index=0, head_only=True, control_state=True)],
        lambda name, item: saved.append((observer.clone_tree(item.control),
            observer.clone_tree(item.event), dict(item.generation_config))))
    evaluation = SimpleNamespace(generate=generate)
    with pytest.raises(observer.CaptureComplete):
        with exporter.generation_capture(evaluation, capture, observer), capture:
            evaluation.generate(model, torch.full((1,64),2), **options)
    states, prediction, generated = saved[0]
    assert prediction["forward_kind"] == "full_sequence"
    assert states["next_forward"]["block_index"] == 1
    assert "current_joint_selection" not in states and "next_joint_selection" not in states
    assert states["before_forward"]["next_block_state"] is not None
    defaults = {name:value.default for name,value in inspect.signature(generate).parameters.items()
                if value.default is not inspect.Parameter.empty}
    observed = dict(provenance={}, generation_config={**defaults, **generated})
    monkeypatch.setattr(references,"read_captured_control",lambda _, **kwargs: (observed,states,prediction))
    reference = references.captured_feature2_reference(Path("initial-completion"))
    assert reference["future"] and reference["future"]["prediction_positions"] == []
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(provenance={})))
    reference.update(source_index=str(source))
    control = tmp_path / "control.json"
    control.write_text(json.dumps(reference))
    image = bytearray((head.HARDWARE / "cases/head_state_update_32tokens/initial.bin").read_bytes())
    head.patch_record(image,head.POST-head.BASE,head.load_schema("forward_postprocess_config"),{"flags":1})
    (tmp_path / "initial.bin").write_bytes(image)
    first = read_record("execution_config",image[:320])
    first.update(start_layer=0,layer_count=32,total_token_count=128,sequence_length=128,
                 flags=3,refresh_configuration_offset=0,joint_configuration_offset=0)
    schema = json.loads((layers.HARDWARE / "config/execution_config.json").read_text())
    (tmp_path / "layer.bin").write_bytes(layers.pack_record(schema,first))
    case = dict(ddr_image="initial.bin",memory_map="map.json",expected=[],head_checkpoints={},
        executions=[dict(config_address=0x60000000),dict(config_address=head.BASE, expected_post_block_completions=[True])],
        initial_segments=[dict(path="layer.bin",address=0x60000000,bytes=320)],
        provenance=dict(control_reference=str(control)),
        attention_checkpoints=dict(token_positions=list(range(128)),
            physical_to_reference_token=[list(range(start,min(start+48,128))) for start in range(0,128,48)]))
    (tmp_path / "map.json").write_text(json.dumps(dict(memory_map=[
        dict(name="head",base=head.BASE,limit=head.BASE+len(image)),
        dict(name="layer",base=0x60000000,limit=0x60001000)])))
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    layers.attach_regular_control(path,source)
    prepared = json.loads(path.read_text())
    data = (tmp_path / "regular_control.bin").read_bytes()
    assert read_record("execution_config",data[:320])["joint_configuration_offset"] == 0
    assert prepared["provenance"]["regular_control"]["closeout"]
    assert prepared["executions"][-1]["expected_post_block_completions"] == [True]
    assert len(data) % 32 == 0


@torch.no_grad()
@pytest.mark.parametrize("end_step", [2, 99])
def test_range_keeps_real_intermediate_forwards_and_endpoint_layer_hooks(monkeypatch, end_step):
    model, options = cpu_selection_model(monkeypatch, layers=32)
    from generation.engine import generate
    # CPU layers use ordinary BF16 arithmetic. Retain the real head/control
    # hooks; observe layer outputs directly instead of requiring CUDA W4 kernels.
    original = exporter.IndexedForwardCapture
    class CPULayers(original):
        def __enter__(self):
            original_head, include = self.head_only, self.include_head
            self.head_only = self.include_head = True
            super().__enter__()
            self.head_only, self.include_head = original_head, include
            for layer in range(self.first_layer, self.first_layer+self.layer_count):
                def after(module, args, output, layer=layer):
                    if self.next_index == self.target:
                        self.layers.append(dict(model_layer=layer, output=output[0].detach().clone()))
                self.handles.append(self.model.model.transformer.blocks[layer].register_forward_hook(after))
            return self

        def on_event(self, event):
            self.head_only = True
            try:
                return super().on_event(event)
            finally:
                self.head_only = False
    monkeypatch.setattr(exporter, "IndexedForwardCapture", CPULayers)
    selections = dict(start=dict(block_index=0,step_index=0,phase="full_sequence",layer=30),
        end=dict(block_index=0,step_index=end_step,phase="full_sequence",layer=2))
    captured=[]
    capture=exporter.ForwardCaptureSet(model,observer,selections,lambda name,item: captured.append(dict(
        index=item.target, layers=[layer["model_layer"] for layer in item.layers],
        event=observer.clone_tree(item.event), control=observer.clone_tree(item.control),
        head=item.include_head)))
    evaluation=SimpleNamespace(generate=generate)
    if end_step == 99:
        # A short real request reaches its actual end without inventing step 99.
        options.update(gen_length=32,steps=32,decode_k=32)
    expected = observer.CaptureComplete if end_step == 2 else ValueError
    with pytest.raises(expected, match=None if end_step == 2 else "range end"):
        with exporter.generation_capture(evaluation,capture,observer),capture:
            evaluation.generate(model,torch.tensor([[2,2]]),**options)
            capture.finish_request()
    if end_step == 2:
        assert [item["index"] for item in captured] == [0,1,2]
        assert [item["layers"] for item in captured] == [[30,31],list(range(32)),[0,1,2]]
        assert [item["head"] for item in captured] == [True,True,False]
        for index,item in enumerate(captured[:2]):
            assert item["control"]["next_forward"]["capture_index"] == index+1
            assert torch.equal(item["control"]["after_postprocess"]["tokens"],
                               captured[index+1]["event"]["tokens_before"].long())
        assert not captured[-1]["control"]
    assert sys.getprofile() is None


@torch.no_grad()
def test_actual_boundary_l0_endpoint_keeps_selector_inputs_without_next_forward(monkeypatch):
    model, options = cpu_selection_model(monkeypatch)
    from generation.engine import generate
    from hardware_adapter import control_reference_data as references
    options.update(steps=2, gen_length=64, decode_k=32, full_sequence_recompute=False,
        packed_attention_refresh=True, packed_attention_layer_mode="all",
        packed_attention_target_active_rows=27.5, cross_block_cache_handoff=True,
        cross_block_boundary_scope="layer0_attention_two_stage", cross_block_boundary_target_rows=64)
    original = exporter.IndexedForwardCapture
    class CPUL0(original):
        def __enter__(self):
            self.head_only = self.include_head = True  # Bypass CUDA layer arithmetic observation.
            super().__enter__()
            self.head_only = self.include_head = False
            block = self.model.model.transformer.blocks[0]
            def before(module, args, kwargs):
                if self.next_index == self.target:
                    self.input_bits = module.q_proj.row_precision_context.require(args[0].shape[1], args[0].device).clone()
            def after(module, args, result):
                if self.next_index == self.target:
                    self.control["after_l0_scout"] = observer.clone_tree(module._LLaDABlock__cache["boundary_layer0_scout"])
            self.handles.extend((block.register_forward_pre_hook(before, with_kwargs=True), block.register_forward_hook(after)))
            return self
        def on_event(self, event):
            self.head_only = True
            try:
                return super().on_event(event)
            finally:
                self.head_only = False
    monkeypatch.setattr(exporter, "IndexedForwardCapture", CPUL0)
    saved = []
    endpoint = dict(block_index=1, step_index=0, phase="boundary_refresh", layer=0)
    capture = exporter.ForwardCaptureSet(model, observer, dict(start=endpoint,end=endpoint),
        lambda name,item: saved.append((observer.clone_tree(item.control), observer.clone_tree(item.event),
                                        dict(item.generation_config), item.input_bits.clone())))
    evaluation = SimpleNamespace(generate=generate)
    with pytest.raises(observer.CaptureComplete):
        with exporter.generation_capture(evaluation,capture,observer),capture:
            evaluation.generate(model,torch.full((1,64),2),**options)
    control, prediction, generated, bits = saved[0]
    assert set(control) == {"before_forward", "before_postprocess", "after_l0_scout"}
    assert prediction["capture_index"] == 1 and prediction["layer0_global_selected_deep"]
    defaults = {name:value.default for name,value in inspect.signature(generate).parameters.items()
                if value.default is not inspect.Parameter.empty}
    metadata = dict(generation_config={**defaults,**generated},
                    tensors=[dict(role="input",name="activation_bits")])
    with patch.object(references,"read_captured_control",return_value=(metadata,control,prediction)),\
            patch.object(exporter,"read_tensor",return_value=bits),\
            patch("model_capture.read_tensor",return_value=bits):
        reference = references.captured_boundary_reference(Path("actual-l0-only"))
    assert reference["records"][0]["expected"]["deep_positions"]["raw"] == prediction["input_positions"].reshape(-1).tolist()


def test_control_completion_preserves_queued_kwargs_hooks(monkeypatch):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = SimpleNamespace(module_name="q", row_precision_context=SimpleNamespace(
                require=lambda count, device, **kw: torch.full((count,), 8, device=device)))

        def forward(self, x, *, query_position_ids=None):
            return x + 1

    block = Block()
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
    completed = []
    capture = exporter.IndexedForwardCapture(model, observer, 0, layer_count=1,
        control_state=True, on_complete=lambda item: completed.append(item.control.copy()))
    # The selected event has ended; completing it inside the next pre-hook must
    # not change how PyTorch calls the already-queued layer observer hook.
    capture.event = {"capture_index": 0}
    capture.next_index = 1
    monkeypatch.setattr(exporter, "capture_control_state", lambda: {"block_index": 0})
    value = torch.ones((1, 2, 3))
    with capture:
        assert torch.equal(block(value, query_position_ids=torch.tensor([[4, 5]])), value + 1)
        assert torch.equal(block(value, query_position_ids=torch.tensor([[4, 5]])), value + 1)
    assert len(completed) == 1
    assert completed[0]["next_layer_inputs"]["query_position_ids"].tolist() == [[4, 5]]
    assert not block._forward_pre_hooks and not block._forward_pre_hooks_with_kwargs


def test_request_end_control_has_no_invented_next_forward(tmp_path):
    from types import SimpleNamespace
    capture = exporter.IndexedForwardCapture(None, observer, 2, head_only=True, control_state=True)
    capture.event = {"capture_index": 2}
    capture.control = dict(before_forward={}, before_postprocess={}, after_postprocess={"state": 2})
    state = torch.tensor([[1, 2]], dtype=torch.int64)
    capture.current_state_references = dict(block_start=1, block_end=3,
        block_state=SimpleNamespace(state=state), tokens=torch.tensor([[0, 42, 43]]),
        cache_refresh_due=torch.zeros((1, 3), dtype=torch.bool))
    state[0, 0] = 2  # The generator closes the block after its forward callback.
    with pytest.raises(observer.CaptureComplete):
        capture.finish_request()
    final = capture.control["current_state_at_request_end"]
    assert final["block_state"]["state"].tolist() == [[2, 2]]
    assert final["tokens"].tolist() == [[42, 43]]
    state.zero_()
    assert final["block_state"]["state"].tolist() == [[2, 2]]
    metadata = {"tensors": []}
    exporter.export_control_state(capture, metadata, tmp_path)
    assert metadata["control_observation"]["checkpoints"]["request_end"] is True
    assert "next_forward" not in metadata["control_observation"]["checkpoints"]
    missing = exporter.IndexedForwardCapture(None, observer, 5)
    with pytest.raises(ValueError, match="ended before capture index 5"):
        missing.finish_request()


def test_shared_capture_payload_uses_actual_layer_and_raw_bits(tmp_path):
    pool = {}
    def write(folder, name, bits, first_layer=0, role="input"):
        output = tmp_path / folder
        output.mkdir(exist_ok=True)
        value = torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        raw, encoding, dtype = exporter.encode_tensor(value)
        return exporter.write_capture_tensor(raw, role, name, output, tmp_path / (folder + ".json"),
                                             encoding, dtype, pool, first_layer)
    original = write("first", "layer0.weight.q_proj.codes", [0, 32768], 30)
    reused = write("middle", "layer30.weight.q_proj.codes", [0, 32768])
    assert reused["path"] == original["path"]
    assert not list((tmp_path / "middle").iterdir())
    # A different physical layer and a signed-zero change both need their own input.
    other = write("other", "layer0.weight.q_proj.codes", [0, 32768])
    changed = write("changed", "layer30.weight.q_proj.codes", [0, 0])
    assert other["path"] != original["path"] != changed["path"]
    expected = write("expected", "layer30.weight.q_proj.codes", [0, 32768], role="expected")
    assert expected["path"] != original["path"]
    moved = tmp_path.with_name(tmp_path.name + "_moved")
    tmp_path.rename(moved)
    actual = exporter.read_tensor(reused, index=moved / "middle.json")
    assert actual.view(torch.int16).tolist() == [0, -32768]


@pytest.mark.parametrize("keep_global", [None, False, True])
@pytest.mark.parametrize("layer_count", [1, 2])
def test_boundary_capture_preserves_actual_l0_commit_scope(keep_global, layer_count):
    values = dict(capture_index=0,
                  layer0_global_selected_deep=True, input_positions=torch.tensor([0, 2]))
    if keep_global is not None:
        values["layer0_keep_global_cache"] = keep_global
    event = make_dataclass("Event", [(key, type(value)) for key, value in values.items()])(**values)
    capture = exporter.IndexedForwardCapture(None, observer, 0, layer_count=layer_count)
    for layer, positions in enumerate((torch.arange(4), torch.tensor([0, 2]))[:layer_count]):
        expected = {}
        cache = tuple(torch.full((1, 1, 4, 1), i + layer, dtype=torch.int8)
                      for i in range(len(observer.CACHE_NAMES)))
        capture.layers.append(dict(layer=layer, positions=positions, write_positions=positions,
                                   cache_refs=cache, expected=expected,
                                   observer=SimpleNamespace(save=lambda name, tensor, store=expected:
                                                            store.setdefault(name, tensor.clone()))))
    with pytest.raises(observer.CaptureComplete):
        capture.on_event(event)
    assert capture.layers[0]["commit_positions"].tolist() == ([0, 1, 2, 3] if keep_global else [0, 2])
    if layer_count == 2:
        assert capture.layers[1]["commit_positions"].tolist() == [0, 2]
    assert capture.layers[0]["write_positions"].tolist() == [0, 1, 2, 3]
    assert capture.layers[0]["expected"]["cache_" + observer.CACHE_NAMES[0]].shape == (1, 1, 4, 1)


def test_capture_registers_public_metadata_path_before_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter.os, "environ", {"SUPRA_RELOCATED_INSTRUCT_CHECKPOINT": "/stale/model"})
    artifact = tmp_path / "artifact_w8_head"
    exporter.configure_capture_environment(dict(model_path=str(tmp_path / "model_metadata"),
                                                spinquant_artifact_dir=str(artifact)))
    assert exporter.os.environ["SUPRA_RELOCATED_INSTRUCT_CHECKPOINT"] == str(tmp_path / "model_metadata")
    assert exporter.os.environ["SUPRA_SPINQUANT_PARENT_ARTIFACT_DIR"] == str(tmp_path / "artifact_w4")
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    rotation = load("capture_rotation_consumer", exporter.SOURCE / "quantization/rotation.py")
    assert (tmp_path / "model_metadata") in rotation.CHECKPOINT_PROFILES
    assert exporter.os.environ["SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC"] == "1"


@pytest.mark.parametrize("request_config,sample,result,expected", [
    ({"dataset": "custom/train"}, {}, {}, "custom/train"),
    ({}, {"dataset": "ignored"}, {}, None),
    (None, {"doc": {"task_id": "custom/1"}}, {}, None),
    (None, {"doc": {"question": "example"}}, {}, None),
    (None, {"dataset": "custom"}, {}, "custom"),
    (None, {}, {"configs": {"a": {"dataset_path": "gsm8k", "dataset_name": "main"},
                            "b": {"dataset_path": "gsm8k", "dataset_name": "main"}}}, "gsm8k/main"),
    (None, {}, {"configs": {"a": {"dataset_path": "one"}, "b": {"dataset_path": "two"}}}, None),
    (None, {}, {"configs": {"a": {"dataset_path": "gsm8k"}, "b": {}}}, None),
    (None, {"task_name": "b"}, {"configs": {"a": {"dataset_path": "one"},
                                           "b": {"dataset_path": "two"}}}, "two"),
    (None, {"task_name": "missing"}, {"configs": {"a": {"dataset_path": "one"}}}, None),
])
def test_capture_dataset_uses_explicit_sources(request_config, sample, result, expected):
    assert exporter.capture_dataset(request_config, sample, result) == expected


def test_request_uses_task_settings_and_explicit_overrides(tmp_path):
    task = {"task": "gsm8k", "model_args": {"spinquant_execution": "joint-full-w4a8-v8", "seed": 9,
            "a4_clip_ratio": 0.80078125}, "gen_kwargs": {"max_gen_toks": 256, "temperature": 0}}
    (tmp_path / "task.json").write_text(json.dumps(task))
    request = {"task_config": "task.json", "prompt": "test", "model_args": {
        "model_path": "model", "spinquant_artifact_dir": "weights", "seed": 12},
        "generation_args": {"max_gen_toks": 32}}
    (tmp_path / "request.json").write_text(json.dumps(request))
    loaded = exporter.load_request_config(tmp_path / "request.json")
    assert loaded["model_args"]["a4_clip_ratio"] == 0.80078125
    assert loaded["model_args"]["seed"] == 12
    assert loaded["model_args"]["gen_length"] == 32
    assert loaded["generation_args"] == {"max_gen_toks": 32, "temperature": 0}
    assert exporter.capture_dataset(loaded, {}, {}) is None


@pytest.mark.parametrize("tier,mode,fixed", [
    ("baseline", "baseline_fixed_k", True),
    ("feature1", "feature1_fixed_k", True),
    ("feature12", "feature1_packed_feature2_fused_a4a8", False),
    ("feature123", "feature1_packed_feature2_fused_dynamic_block", False),
])
def test_capture_uses_progressive_tier_and_preserves_overrides(tmp_path, tier, mode, fixed):
    task = json.loads((exporter.SOURCE / "configs/gsm8k.json").read_text())
    task["tier"] = "feature12"
    (tmp_path / "task.json").write_text(json.dumps(task))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(dict(task_config="task.json", prompt="Question: test\nAnswer:",
        tier=tier, model_args=dict(model_path="model", spinquant_artifact_dir="weights", decode_k=5,
                                   a4_clip_ratio=0.75), generation_args={})))
    model = exporter.load_request_config(path)["model_args"]
    assert model["generation_mode"] == mode
    assert model["feature3_precision_policy"] == ("all_a8" if fixed else "original")
    assert model["decode_k"] == 5 and model["a4_clip_ratio"] == 0.75
    assert model["feature2_cache_initialization_activation_policy"] == ("default" if tier == "baseline" else "a4")
    if tier != "baseline":
        assert model["feature1_cross_block_block_initialization_deep_rows"] == 384
    if tier != "feature123":
        assert not any(key.startswith("feature3_dynamic_block_") for key in model)


def test_capture_inherits_task_tier_and_rejects_unknown_tier(tmp_path):
    task = dict(model_args=dict(model_path="model", spinquant_artifact_dir="weights",
                               spinquant_execution="joint-full-w4a8-v8"), tier="feature1")
    (tmp_path / "task.json").write_text(json.dumps(task))
    path = tmp_path / "request.json"
    request = dict(task_config="task.json", prompt="test", generation_args={})
    path.write_text(json.dumps(request))
    assert exporter.load_request_config(path)["model_args"]["generation_mode"] == "feature1_fixed_k"
    request["tier"] = "unknown"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="unknown tier"):
        exporter.load_request_config(path)


@pytest.mark.parametrize("overrides", [
    {"model_args": {"gen_length": 64}},
    {"generation_args": {"max_gen_toks": 64}},
])
def test_request_length_override_reaches_model_and_generation(tmp_path, overrides):
    task = {"model_args": {"model_path": "model", "spinquant_artifact_dir": "weights",
            "spinquant_execution": "joint-full-w4a8-v8", "gen_length": 256},
            "gen_kwargs": {"max_gen_toks": 256}}
    (tmp_path / "task.json").write_text(json.dumps(task))
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"task_config": "task.json", "prompt": "test", **overrides}))
    request = exporter.load_request_config(path)
    assert request["model_args"]["gen_length"] == 64
    assert request["generation_args"]["max_gen_toks"] == 64


def test_request_reports_conflicting_explicit_lengths(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"prompt": "test", "model_args": {"gen_length": 32},
                               "generation_args": {"max_gen_toks": 64}}))
    with pytest.raises(ValueError, match="gen_length=32.*max_gen_toks=64"):
        exporter.load_request_config(path)


@pytest.mark.parametrize("explicit,sample,result,expected", [
    (None, {"doc_id": 174, "task_name": "gsm8k"}, {}, "gsm8k/174"),
    (None, {"doc_id": 174}, {"configs": {"gsm8k": {}}}, "gsm8k/174"),
    (None, {"doc_id": 174}, {"configs": {"a": {}, "b": {}}}, "174"),
    ("user-label", {"doc_id": 174}, {}, "user-label"),
    (None, {"sample_id": "saved/174", "doc_id": 174}, {}, "saved/174"),
    (None, {}, {}, None),
])
def test_capture_sample_identity_uses_recorded_values(explicit, sample, result, expected):
    assert exporter.capture_sample_id(explicit, sample, result) == expected


@pytest.mark.parametrize("explicit_start", [False, True])
def test_capture_request_preserves_question_boundary(explicit_start):
    prompt = "Question: example\nAnswer: 1\n\nQuestion: actual\nAnswer:"
    start = prompt.index("Question: actual")
    sample = dict(doc_id=174, doc={"sample_id": "gsm8k/train/174"})
    if not explicit_start:
        sample["doc"]["question"] = "actual"
    request = exporter.evaluation_request(prompt, {"max_gen_toks": 256}, sample,
        task_name="gsm8k_train", request_index=3,
        question_char_start=start if explicit_start else None)
    assert request.question_char_start == (start if explicit_start else None)
    assert request.doc == sample["doc"]
    assert (request.doc_id, request.idx, request.task_name) == (174, 3, "gsm8k_train")
    assert request.args == (prompt, {"max_gen_toks": 256})


def test_shipped_request_has_protected_question_metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    from evaluation.model import _question_start_token
    example = json.loads((HERE / "request_example.json").read_text())
    request = exporter.evaluation_request(example["prompt"], example["generation_args"],
        {"doc": example.get("sample", {})}, task_name=example["dataset"],
        question_char_start=example.get("question_char_start"))
    assert _question_start_token(request, example["prompt"], [(0, len(example["prompt"]))]) == 0


@pytest.mark.parametrize("start", [-1, 4, True, 1.5])
def test_capture_request_rejects_invalid_character_offset(start):
    with pytest.raises(ValueError, match="question_char_start"):
        exporter.evaluation_request("abc", {}, {}, task_name="test", question_char_start=start)


@pytest.mark.parametrize("value,encoding", [
    (torch.tensor([0., -0., 1.5], dtype=torch.bfloat16), "bf16_raw"),
    (torch.tensor([0., -0., 1.5], dtype=torch.float32), "bf16_raw"),
    (torch.tensor([0., -0., 1.0000001192092896], dtype=torch.float32), "float32_raw"),
    (torch.tensor([0x7fc01234, -2147483648], dtype=torch.int32).view(torch.float32), "float32_raw"),
    (torch.tensor([-8, -1, 0, 7], dtype=torch.int8), "integer"),
    (torch.tensor([-2147483648, 2147483647], dtype=torch.int32), "integer"),
])
def test_raw_roundtrip_preserves_bits_and_runtime_dtype(tmp_path, value, encoding):
    raw, actual_encoding, runtime_dtype = exporter.encode_tensor(value)
    assert actual_encoding == encoding
    path = tmp_path / "tensor.bin"
    raw.tofile(path)
    restored = exporter.read_tensor(dict(path=str(path), byte_count=raw.nbytes,
        shape=list(raw.shape), dtype=raw.dtype.str, encoding=encoding, runtime_dtype=runtime_dtype))
    assert observer.raw_equal(value, restored)


@pytest.mark.parametrize("field", ["row_max", "limit_fp32"])
def test_clipping_fp32_semantics_never_compress_to_bf16(field):
    value = torch.tensor([[1.0], [0.0], [-0.0]], dtype=torch.float32)
    raw, encoding, runtime_dtype = exporter.encode_tensor(value, name="layer0.q_proj.clip." + field)
    assert encoding == "float32_raw"
    assert raw.dtype.str == "<u4"
    assert runtime_dtype == "float32"
    assert raw.reshape(-1).tolist() == [0x3f800000, 0, 0x80000000]


def test_initial_cache_snapshot_does_not_alias_mutable_cache():
    live = (torch.tensor([1, 2], dtype=torch.int8), torch.ones(2, dtype=torch.bfloat16))
    before = observer.clone_tree({"cache": live})
    live[0][0] = 7
    live[1][0] = 2
    assert before["cache"][0].tolist() == [1, 2]
    assert before["cache"][1].tolist() == [1, 1]


def test_raw_comparison_distinguishes_signed_zero():
    assert not observer.raw_equal(torch.tensor([0.]), torch.tensor([-0.]))


def test_full_cache_write_has_empty_retained_set():
    values = torch.ones(1, 32, 413, 1)
    empty = values[:, :, torch.zeros(413, dtype=torch.bool)]
    assert exporter.raw_equal(empty, empty.clone())
    assert not exporter.raw_equal(empty, empty.bfloat16())
    assert not exporter.raw_equal(empty, torch.empty(1, 32, 0, 2))
    assert not exporter.raw_equal(torch.tensor([0.]), torch.tensor([-0.]))


def test_payload_cannot_be_placed_in_source_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "ALGO", tmp_path)
    with pytest.raises(ValueError, match="payload must"):
        exporter.validate_paths(HERE / "payload", HERE / "reference_data" / "unused.json")


def test_payload_requires_configured_algorithm_root(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "ALGO", None)
    with pytest.raises(ValueError, match="set SUPRA_ALGORITHM_ROOT"):
        exporter.validate_paths(tmp_path / "payload", HERE / "reference_data" / "unused.json")


def test_reader_rejects_truncated_raw(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="wrong payload size"):
        exporter.read_tensor(dict(path=str(path), byte_count=2))


def test_existing_profiler_is_not_replaced():
    block = type("Block", (), {})()
    # Avoid model construction: the conflict must fail before accessing a block.
    capture = object.__new__(observer.LayerObserver)
    capture.block = block
    previous = sys.getprofile()
    sentinel = lambda *args: None
    sys.setprofile(sentinel)
    try:
        with pytest.raises(RuntimeError, match="no other Python profiler"):
            capture.__enter__()
        assert sys.getprofile() is sentinel
    finally:
        sys.setprofile(previous)


def test_head_cleanup_preserves_profiler_it_did_not_start():
    capture = exporter.IndexedForwardCapture(None, observer, 0, head_only=True)
    previous = sys.getprofile()
    sentinel = lambda *args: None
    sys.setprofile(sentinel)
    try:
        capture.__exit__(None, None, None)
        assert sys.getprofile() is sentinel
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("future", [False, True])
@pytest.mark.parametrize("preselected", [False, True])
@pytest.mark.parametrize("locked_outside_query", [False, True])
@pytest.mark.parametrize("full_sequence", [False, True])
def test_non_full_sequence_head_export_uses_recorded_prediction_order(tmp_path, monkeypatch, preselected, locked_outside_query, future, full_sequence):
    monkeypatch.setattr(exporter, "validate_paths", lambda *args: None)
    tokens = 1 if full_sequence else 2 if preselected else 5
    head = dict(
        hidden=torch.arange(tokens, dtype=torch.bfloat16).view(1, tokens, 1).expand(-1, -1, 4).clone(),
        norm_weight=torch.ones(4, dtype=torch.bfloat16),
        weight_codes=torch.ones(3, 4, dtype=torch.int8),
        weight_scale=torch.ones(3, dtype=torch.bfloat16),
        epsilon=1e-5,
        norm_output=torch.arange(tokens, dtype=torch.bfloat16).view(1, tokens, 1).expand(-1, -1, 4).clone(),
        activation_codes=torch.arange(tokens, dtype=torch.int8).view(tokens, 1).expand(-1, 4).clone(),
        activation_scale=torch.arange(tokens, dtype=torch.bfloat16),
        accumulator=torch.arange(tokens, dtype=torch.int32).view(tokens, 1).expand(-1, 3).clone(),
        output=torch.arange(tokens, dtype=torch.bfloat16).view(1, tokens, 1).expand(-1, -1, 3).clone(),
    )
    event = dict(capture_index=15, block_index=1, step_index=0,
                 forward_kind="boundary_refresh",
                 prediction_positions=torch.tensor([7, 13 if locked_outside_query else 9]),
                 input_positions=torch.tensor([1, 3, 7, 9, 11]),
                 prediction_mask=torch.tensor([[True, False]]))
    event["full_sequence_recompute"] = full_sequence
    if future:
        event["future_prediction_positions"] = torch.tensor([11])
    capture = SimpleNamespace(head=head, event=event)
    index = tmp_path / "index.json"
    if (preselected or full_sequence) and future:
        with pytest.raises(ValueError, match="head input must cover|full-sequence head rows"):
            exporter.export_head(capture, {"seed": 7}, tmp_path / "payload", index)
        assert not index.exists()
        return
    exporter.export_head(capture, {"seed": 7}, tmp_path / "payload", index)
    saved = json.loads(index.read_text())
    assert saved["tokens"] == (2 if future else 1)
    assert saved["current_prediction_positions"] == [7]
    assert saved["future_prediction_positions"] == ([11] if future else [])
    assert saved["forward"] == {"capture_index": 15, "block_index": 1,
                                 "step_index": 0, "forward_kind": "boundary_refresh"}
    assert saved["validation"]["status"] == "EXPORTED_NOT_REPLAYED"
    hidden = next(entry for entry in saved["tensors"]
                  if entry["role"] == "input" and entry["name"] == "hidden")
    assert hidden["shape"] == [1, 2 if future else 1, 4]
    restored = exporter.read_tensor(hidden, index=index)
    assert restored.reshape(-1, 4)[:, 0].tolist() == ([2, 4] if future else [0] if preselected or full_sequence else [2])


@pytest.mark.parametrize("last_layer_only, selected", [(False, (0, 1)), (True, (31,))])
def test_selected_layer_hooks_are_removed_after_exception(last_layer_only, selected):
    blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks)))
    capture = exporter.IndexedForwardCapture(model, observer, 0, last_layer_only=last_layer_only)
    with pytest.raises(RuntimeError, match="interrupted"):
        with capture:
            assert tuple(i for i, block in enumerate(blocks) if block._forward_pre_hooks) == selected
            assert tuple(i for i, block in enumerate(blocks) if block._forward_hooks) == selected
            raise RuntimeError("interrupted")
    assert all(not block._forward_pre_hooks and not block._forward_hooks for block in blocks)


def test_last_layer_capture_rejects_head_only():
    with pytest.raises(ValueError, match="different observations"):
        exporter.IndexedForwardCapture(None, observer, 0, head_only=True, last_layer_only=True)


@pytest.mark.parametrize("first,count", [(0, 32), (5, 3), (31, 1)])
def test_explicit_consecutive_layer_hooks(first, count):
    blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks)))
    with exporter.IndexedForwardCapture(model, observer, 0, first_layer=first, layer_count=count):
        assert tuple(i for i, block in enumerate(blocks) if block._forward_pre_hooks) == tuple(range(first, first+count))
    assert all(not block._forward_pre_hooks and not block._forward_hooks for block in blocks)


@pytest.mark.parametrize("first,count", [(-1, 1), (0, 0), (0, 33), (31, 2), (32, 1)])
def test_capture_rejects_out_of_model_layer_range(first, count):
    with pytest.raises(ValueError, match="within L0..L31"):
        exporter.IndexedForwardCapture(None, observer, 0, first_layer=first, layer_count=count)


@pytest.mark.parametrize("layer", [30, 31])
def test_dependency_input_is_a_snapshot_before_selected_layer_update(layer):
    cache = {
        "attn_monitor_dependency_layer_count": layer,
        "attn_monitor_dependency_query_positions": torch.tensor([3, 11]),
        "attn_monitor_dependency_max": torch.tensor([[[0.125, 0.25]]]),
        "attn_monitor_prefix_dependency_max": torch.tensor([[[0.5]]]),
        "unrelated_large_tensor": torch.ones(32),
    }
    block = SimpleNamespace(layer_id=layer, _LLaDABlock__cache=cache)
    saved = exporter.capture_layer_dependency_input(block, observer)
    assert "unrelated_large_tensor" not in saved
    cache["attn_monitor_dependency_max"].fill_(0.75)
    cache["attn_monitor_dependency_layer_count"] = layer + 1
    assert saved["attn_monitor_dependency_layer_count"] == layer
    assert saved["attn_monitor_dependency_max"].tolist() == [[[0.125, 0.25]]]
    with pytest.raises(RuntimeError, match="layer count"):
        exporter.capture_layer_dependency_input(block, observer)


def test_last_layer_export_keeps_physical_slot_and_model_layer_distinct(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    monkeypatch.setattr(exporter, "validate_paths", lambda *args: None)
    positions = torch.tensor([7], dtype=torch.long)
    inputs = dict(hidden=torch.ones(1, 1, 1, dtype=torch.bfloat16), activation_bits=torch.tensor([4], dtype=torch.int8),
                  retained_key_codes=torch.zeros(1, 1, 8, 1, dtype=torch.int8),
                  softmax_exp_lut=torch.ones(2, dtype=torch.bfloat16),
                  softmax_reciprocal_lut=torch.ones(2, dtype=torch.bfloat16))
    record = dict(layer=0, model_layer=31, inputs=inputs,
                  expected={"block_output": inputs["hidden"].clone()},
                  kwargs={}, positions=positions, write_positions=positions,
                  config={"rope_theta": 500000}, r4=True, norm_implementation="test")
    keys = ("capture_index", "block_index", "step_index", "forward_kind", "nfe_before",
            "input_positions", "prediction_positions", "refresh_positions", "row_bits",
            "prediction_mask", "layer0_global_selected_deep")
    capture = SimpleNamespace(first_layer=31, layers=[record], event=dict.fromkeys(keys),
                              prompt_ids=torch.tensor([[1]]), generation_config={})
    result = exporter.export_capture(capture, {"seed": 7}, tmp_path / "payload", tmp_path / "index.json")
    assert result["model_layer_index"] == 31
    assert result["initial_layer_state"] == "hidden_and_retained_cache"
    assert result["layer_count"] == 1
    assert result["layers"][0]["layer"] == 0
    assert result["layers"][0]["model_layer"] == 31
    assert "connection" not in result
    entry = next(e for e in result["tensors"] if e["name"] == "layer0.block_output")
    assert observer.raw_equal(exporter.read_tensor(entry, index=tmp_path / "index.json"), inputs["hidden"])
    assert result["validation"]["status"] == "EXPORTED_NOT_REPLAYED"


def test_single_l0_export_has_no_layer_connection(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    monkeypatch.setattr(exporter, "validate_paths", lambda *args: None)
    positions = torch.tensor([7], dtype=torch.long)
    inputs = dict(hidden=torch.ones(1, 1, 1, dtype=torch.bfloat16), activation_bits=torch.tensor([4], dtype=torch.int8),
                  retained_key_codes=torch.zeros(1, 1, 8, 1, dtype=torch.int8),
                  softmax_exp_lut=torch.ones(2, dtype=torch.bfloat16),
                  softmax_reciprocal_lut=torch.ones(2, dtype=torch.bfloat16))
    record = dict(layer=0, inputs=inputs, expected={"block_output": inputs["hidden"].clone()},
                  kwargs={}, positions=positions, write_positions=positions,
                  config={"rope_theta": 500000}, r4=True, norm_implementation="test")
    keys = ("capture_index", "block_index", "step_index", "forward_kind", "nfe_before",
            "input_positions", "prediction_positions", "refresh_positions", "row_bits",
            "prediction_mask", "layer0_global_selected_deep")
    capture = SimpleNamespace(first_layer=0, layers=[record], event=dict.fromkeys(keys),
                              prompt_ids=torch.tensor([[1]]), generation_config={})
    result = exporter.export_capture(capture, {"seed": 7}, tmp_path / "payload", tmp_path / "index.json")
    assert result["model_layer_index"] == 0
    assert result["initial_layer_state"] == "hidden_and_retained_cache"
    assert result["layer_count"] == 1
    assert result["layers"][0]["layer"] == 0
    assert "model_layer" not in result["layers"][0]
    assert "connection" not in result
    entry = next(e for e in result["tensors"] if e["name"] == "layer0.block_output")
    assert observer.raw_equal(exporter.read_tensor(entry, index=tmp_path / "index.json"), inputs["hidden"])
    assert result["validation"]["status"] == "EXPORTED_NOT_REPLAYED"


def test_explicit_request_preserves_prompt_and_generation_parameters(tmp_path):
    request = {"prompt": "Exact prompt\nwith examples", "generation_args": {"max_gen_toks": 256},
               "model_args": {"model_path": "model", "spinquant_artifact_dir": "artifact",
                              "spinquant_execution": "joint-full-w4a8-v8", "seed": 7, "gen_length": 256, "steps": 64}}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    loaded = exporter.load_request_config(path)
    assert loaded["prompt"] == request["prompt"]
    assert loaded["generation_args"] == request["generation_args"]
    assert all(loaded["model_args"][key] == value for key, value in request["model_args"].items())
    del request["model_args"]["seed"]
    path.write_text(json.dumps(request))
    assert exporter.load_request_config(path)["model_args"]["seed"] == 1234






def test_capture_sample_row():
    args = exporter.argument_parser().parse_args(['capture', '--index', 'index.json', '--sample-row', '134'])
    assert args.sample_row == 134


def test_control_export_rejects_incomplete_transition_and_preserves_raw(tmp_path):
    capture = SimpleNamespace(capture_control=True, control={}, event={})
    metadata = {'tensors': []}
    with pytest.raises(ValueError, match='next forward'):
        exporter.export_control_state(capture, metadata, tmp_path)
    raw = torch.tensor([0., -0.], dtype=torch.bfloat16)
    capture.control = dict(before_forward={'pending': raw}, before_postprocess={},
                           next_forward={}, next_layer_inputs={})
    exporter.export_control_state(capture, metadata, tmp_path)
    entry = metadata['tensors'][0]
    assert not Path(entry['path']).is_absolute()
    assert exporter.raw_equal(raw, exporter.read_tensor(entry, payload_root=tmp_path))
    assert metadata['control_observation']['status'] == 'OBSERVED'



def test_public_capture_provenance_keeps_config_without_machine_paths():
    original = dict(
        model_arguments={"model_path": "/private/model", "seed": 1234,
                         "silu_table": "/private/silu.json",
                         "extra_file": "/private/data.bin",
                         "spinquant_artifact_dir": "/private/weights", "a4_clip_ratio": .8},
        dataset="gsm8k")
    result = exporter.capture_provenance(original)
    assert result == dict(model_arguments={"seed": 1234, "a4_clip_ratio": .8},
                          dataset="gsm8k")
    assert original["model_arguments"]["model_path"] == "/private/model"


@pytest.mark.parametrize("command", ["capture", "replay"])
def test_capture_cli_external_index_boundary(monkeypatch, tmp_path, command):
    monkeypatch.setattr(exporter, "PROJECT", tmp_path / "source")
    monkeypatch.setattr(sys, "argv", ["model_capture.py", command,
                                     "--index", str(tmp_path / "external/index.json")])
    with patch.object(exporter, "load_observer", side_effect=RuntimeError("observer reached")) as load:
        with pytest.raises(RuntimeError, match="observer reached"):
            exporter.main()
        load.assert_called_once()

@pytest.mark.parametrize("separate_index", [False, True])
def test_control_capture_moves_with_index_and_relative_payload(tmp_path, separate_index):
    from hardware_adapter.control_reference_data import read_captured_control
    original = tmp_path / "original"
    payload = original / "payload"
    payload.mkdir(parents=True)
    index = original / "index.json"
    raw = torch.tensor([0., -0.], dtype=torch.bfloat16)
    capture = SimpleNamespace(capture_control=True, event={}, control=dict(
        before_forward={"pending": raw}, before_postprocess={},
        next_forward={}, next_layer_inputs={}))
    metadata = {"tensors": [], "validation": {"status": "PASS"}}
    exporter.export_control_state(capture, metadata, payload, index)
    index.write_text(json.dumps(metadata))
    moved = tmp_path / "moved"
    original.rename(moved)
    relocated = tmp_path / "separate.json"
    if separate_index:
        (moved / "index.json").rename(relocated)
    _, states, _ = read_captured_control(relocated if separate_index else moved / "index.json",
                                        payload_root=moved if separate_index else None)
    assert exporter.raw_equal(states["before_forward"]["pending"], raw)
    assert not any(Path(entry["path"]).is_absolute() for entry in metadata["tensors"])


@torch.no_grad()
@pytest.mark.parametrize("feature3,terminal,fixed", [(False, False, False), (True, False, False),
    (False, True, False), (True, True, False), (False, True, True)])
def test_live_control_snapshot_preserves_generator_results(monkeypatch, feature3, terminal, fixed):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    from dataclasses import asdict
    from generation.engine import generate
    from model.configuration_llada import LLaDAConfig
    from model.modeling_llada import ActivationType, BlockType, LLaDAModel, LLaDAModelLM, ModelConfig
    from numerics.precision import RowPrecisionContext
    config = ModelConfig(d_model=16, n_heads=2, n_kv_heads=2, n_layers=2,
        mlp_hidden_size=32, activation_type=ActivationType.silu, block_type=BlockType.llama,
        max_sequence_length=128, vocab_size=8, embedding_size=8, rope=True, weight_tying=False,
        attention_dropout=0., residual_dropout=0., embedding_dropout=0., use_manual_attention=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        model = LLaDAModelLM(LLaDAConfig(**asdict(config)), model=LLaDAModel(config)).eval().bfloat16()
    options = dict(steps=32, gen_length=32, block_length=32, mask_id=7,
        suppressed_candidate_token_ids=(7,), budget_scale=26., tail_confirmation_policy='all',
        packed_attention_refresh=True, packed_attention_layer_mode='all',
        packed_attention_target_active_rows=27.5, cross_block_cache_handoff=True,
        cross_block_boundary_scope='layer0_attention_two_stage', cross_block_boundary_target_rows=64)
    if feature3:
        options.update(steps=64, gen_length=64, dynamic_block_lookahead=True,
                       dynamic_block_canonical_future=True,
                       dynamic_block_source_a_confirm_at_handoff=True)
    if fixed:
        options.update(decoding_mode="fixed_k", decode_k=1, feature3_precision_policy="all_a8",
                       cache_initialization_activation_policy="a4")
    if not feature3 and not terminal:
        options.update(steps=64, gen_length=64, packed_attention_initial_burst_rows=0.)
    if terminal:
        model.model.transformer.ff_out.weight.zero_()
        options.update(tau_high=0.01, tau_low=0.0, confirm_tau=0.01)
    snapshots, observations, completions = [], [], {}
    retained = {}
    def observe(event):
        snapshot = exporter.capture_control_state()
        assert torch.equal(snapshot['block_state']['state'], event.block_state_before)
        assert torch.equal(snapshot['tokens'].to(torch.int32), event.tokens_before)
        snapshots.append(snapshot)
        observations.append((asdict(event), snapshot))
    def finish(event):
        retained.clear()
        completions[(event['block_index'], event['step_index'], event['forward_kind'])] = (
            {**event, 'trace': asdict(event['trace'])}, exporter.capture_control_state(retain_current=retained))
    prompt = torch.full((1, 64), 2)
    actual = generate(model, prompt, **options, row_precision_context=RowPrecisionContext(),
                      state_capture_callback=observe, forward_end_callback=finish)
    plain = generate(model, prompt, **options, row_precision_context=RowPrecisionContext())
    assert torch.equal(actual[0], plain[0]) and actual[1:] == plain[1:]
    assert snapshots and not bool(snapshots[0]['block_state']['state'].any())
    assert snapshots[0]['block_state']['last_top1'].eq(-1).all()
    from hardware_adapter import control_reference_data as references
    generated = {name: value.default for name, value in inspect.signature(generate).parameters.items()
                 if value.default is not inspect.Parameter.empty}
    generated.update(options)
    if not feature3 and not terminal:
        phases = set()
        for (prediction, before), (following, _) in zip(observations, observations[1:]):
            if (prediction['block_index'] != following['block_index'] or
                    prediction['forward_kind'] not in {'full_sequence', 'boundary_refresh', 'local_block'} or
                    following['forward_kind'] != 'local_block'):
                continue
            transition, after = completions[(prediction['block_index'], prediction['step_index'],
                                             prediction['forward_kind'])]
            checkpoints = dict(before_postprocess=before, after_postprocess=after, transition=transition,
                next_layer_inputs=dict(query_position_ids=following['input_positions'],
                                       activation_bits=following['row_bits']))
            with patch.object(references, 'read_captured_control', return_value=(
                    {'generation_config': generated}, checkpoints, prediction)):
                pending = references.captured_feature1_reference(Path('cpu-selection'))
            assert pending['records'][0]['expected']['next_refresh']['raw'] == after['packed_state']['refresh'].tolist()
            phases.add(prediction['forward_kind'])
        assert phases == {'full_sequence', 'boundary_refresh', 'local_block'}
    if terminal:
        prediction, before = observations[-1]
        transition, after = completions[(prediction['block_index'], prediction['step_index'],
                                         prediction['forward_kind'])]
        terminal = exporter.IndexedForwardCapture(None, observer, prediction['capture_index'],
            head_only=True, control_state=True)
        terminal.event = prediction
        terminal.current_state_references = retained
        terminal.control = dict(before_postprocess=before, after_postprocess=after, transition=transition)
        with pytest.raises(observer.CaptureComplete):
            terminal.finish_request()
        with patch.object(references, 'read_captured_control', return_value=(
                {'generation_config': generated}, terminal.control, prediction)):
            closed = references.captured_feature2_reference(Path('request-end-cpu-model'))
        assert closed['expected']['state'] == [2] * 32
        assert closed['expected']['tokens'] == actual[0][0, -32:].tolist()
        assert 'next_forward' not in terminal.control
        if fixed:
            assert closed['config']['transfer_only'] and closed['config']['scheduled'] == 1
            assert closed['expected']['bits'] == [8] * 32
    if feature3 and not terminal:
        checked = 0
        for prediction, before in observations:
            future = prediction.get('future_prediction_positions')
            if (prediction['forward_kind'] != 'local_block' or before['next_block_state'] is None or
                    (future is not None and future.numel())):
                continue
            transition, after = completions[(prediction['block_index'], prediction['step_index'],
                                             prediction['forward_kind'])]
            checkpoints = dict(before_postprocess=before, after_postprocess=after, transition=transition,
                               next_forward=dict(block_index=before['block_index']))
            with patch.object(references, 'read_captured_control', return_value=(
                    {'generation_config': generated}, checkpoints, prediction)):
                reference = references.captured_feature2_reference(Path('observed-cpu-model'))
            assert reference['future']['prediction_positions'] == []
            assert len(reference['future']['rows']) == 32
            for field in ('state', 'precision_age', 'last_top1', 'commit_origin'):
                assert reference['future']['expected'][field] == before['next_block_state'][field].reshape(-1).tolist()
            assert not any(reference['future']['expected']['masks'].values())
            checked += 1
            break
        assert checked, 'CPU model must exercise a live, unobserved future block'


def test_capture_metadata_uses_generator_event_schema(monkeypatch):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    from dataclasses import fields
    from generation.records import ForwardCaptureEvent
    values = {field.name: None for field in fields(ForwardCaptureEvent)}
    values.update(nfe_before=33, row_bits=torch.tensor([4, 8], dtype=torch.int8))
    actual = exporter.forward_event_metadata(values)
    assert actual['execution_before'] == 33 and actual['activation_bits'] == [4, 8]
    inputs = {'row_bits': values['row_bits'], 'q_proj.row_bits': values['row_bits']}
    hardware = exporter.activation_field_names(inputs)
    algorithm = exporter.activation_field_names(hardware, to_algorithm=True)
    assert set(algorithm) == set(inputs)
    assert all(exporter.raw_equal(inputs[k], algorithm[k]) for k in inputs)


def test_layer_export_accepts_actual_clipping_rule(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(exporter.SOURCE))
    monkeypatch.setattr(exporter, 'validate_paths', lambda *args: None)
    from dataclasses import fields
    from generation.records import ForwardCaptureEvent
    from quantization.model import (_install_target_r4, install_target_block_a4_clipping,
                                    describe_target_block_a4_clipping)
    from numerics.precision import clip_a4_rows_bf16
    from quantization.model import SpinQuantW4A8Linear
    from quantization.numeric import SpinQuantW4Tensor
    from numerics.linear_kernels import LinearNumericWorkspace
    from numerics.precision import RowPrecisionContext
    block = torch.nn.Module()
    context = RowPrecisionContext()
    context.activate(torch.tensor([4, 8], dtype=torch.int8))
    for name in observer.LINEARS:
        setattr(block, name, SpinQuantW4A8Linear(
            SpinQuantW4Tensor(torch.tensor([[1, -1]], dtype=torch.int8), torch.ones(1, dtype=torch.bfloat16)),
            LinearNumericWorkspace(), context, module_name=name))
    _install_target_r4(block)
    install_target_block_a4_clipping(block, {name: .8 for name in observer.LINEARS})
    descriptions = describe_target_block_a4_clipping(block)
    hidden = torch.tensor([[[4., -2.], [1., 3.]]], dtype=torch.bfloat16)
    bits = torch.tensor([4, 8], dtype=torch.int8)
    inputs = dict(hidden=hidden, row_bits=bits,
        retained_key_codes=torch.zeros(1, 1, 8, 1, dtype=torch.int8),
        softmax_exp_lut=torch.ones(2, dtype=torch.bfloat16),
        softmax_reciprocal_lut=torch.ones(2, dtype=torch.bfloat16))
    expected = {'block_output': hidden, 'ff_out.r4_output': hidden}
    for name, desc in descriptions.items():
        inputs[name+'.clip.installed'] = torch.tensor(desc['installed'])
        inputs[name+'.clip.ratio'] = desc['ratio_bf16']
        inputs[name+'.row_bits'] = bits
        observed = []
        value = clip_a4_rows_bf16(hidden, bits, float(desc['ratio_bf16']), observer=observed.append)
        expected[name+'.input'] = value
        expected.update({name+'.clip.'+key: value for key, value in observed[0].items()})
    positions = torch.tensor([3, 7])
    record = dict(layer=0, inputs=inputs, expected=expected, kwargs={}, positions=positions,
        write_positions=positions, config={'rope_theta': 500000}, r4=True, norm_implementation='test',
        numerical_observations=dict(clipping=descriptions,
                                    clipping_observations={name:'complete' for name in observer.LINEARS}))
    event = {field.name: None for field in fields(ForwardCaptureEvent)}
    event.update(row_bits=bits, future_prediction_positions=torch.empty(0, dtype=torch.int32))
    capture = SimpleNamespace(first_layer=0, layers=[record], event=event, prompt_ids=torch.tensor([[1]]),
                              generation_config={})
    result = exporter.export_capture(capture, {'seed': 7}, tmp_path/'payload', tmp_path/'index.json')
    assert result['numeric_capabilities']['a4_clipping'] == descriptions['q_proj']['rule']
    assert any(e['name'] == 'layer0.q_proj.activation_bits' for e in result['tensors'])


def test_capture_entry_failure_removes_partially_registered_hooks():
    blocks = torch.nn.ModuleList([torch.nn.Identity()])
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks)))
    capture = exporter.IndexedForwardCapture(model, observer, 0, control_state=True)
    with pytest.raises(ValueError, match="selected layer range"):
        with capture:
            pass
    assert not blocks[0]._forward_pre_hooks
    assert not blocks[0]._forward_hooks


@torch.no_grad()
def test_head_only_confirmation_saves_selection_without_control(monkeypatch):
    from generation.engine import generate
    masks = []
    for control_state in (False, True):
        model, options = cpu_selection_model(monkeypatch)
        options.update(steps=1, gen_length=32, decoding_mode="feature2", full_sequence_recompute=False,
            feature3_precision_policy="original", tail_confirmation_policy="none", tau_high=1.,
            tau_low=1., confirm_tau=1., packed_attention_refresh=True,
            packed_attention_layer_mode="all", packed_attention_target_active_rows=27.5,
            packed_attention_initial_burst_rows=0.)
        capture = exporter.ForwardCaptureSet(model, observer, [dict(name="confirmation",
            block_index=0, step_index=1, phase="local_confirmation", head_only=True,
            control_state=control_state)], lambda name, item: masks.append(item.head["prediction_mask"].clone()))
        evaluation = SimpleNamespace(generate=generate)
        with pytest.raises(observer.CaptureComplete):
            with exporter.generation_capture(evaluation, capture, observer), capture:
                evaluation.generate(model, torch.tensor([[2, 2]]), **options)
                capture.finish_request()
    assert masks[0].any() and torch.equal(masks[0], masks[1])
