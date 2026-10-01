"""Capture and replay model layers and control events from a generation request."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from dataclasses import asdict

import numpy as np
import torch

RELEASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RELEASE / "hardware/scripts"))
from artifact_paths import PROJECT_ROOT as PROJECT
_algorithm_data_root = os.environ.get("SUPRA_ALGORITHM_ROOT")
ALGO = Path(_algorithm_data_root).expanduser().resolve() if _algorithm_data_root else None
SOURCE = RELEASE / "algorithm/llada"
OBSERVER = SOURCE / "capture/layers.py"
sys.path.insert(0, str(SOURCE))
SCHEMA = "supra-layer-reference-data/v1"


def load_observer(path):
    spec = importlib.util.spec_from_file_location("real_layer_capture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_capture_environment(model_args):
    """Use the same relocated checkpoint interface as algorithm/run.py."""
    artifact = Path(model_args["spinquant_artifact_dir"])
    os.environ["SUPRA_RELOCATED_INSTRUCT_CHECKPOINT"] = str(Path(model_args["model_path"]).resolve())
    os.environ["SUPRA_SPINQUANT_PARENT_ARTIFACT_DIR"] = str(artifact.parent / "artifact_w4")
    os.environ["SUPRA_REQUIRE_TRITON_LINEAR_NUMERIC"] = "1"


@contextmanager
def generation_capture(evaluation, capture, observer):
    """Attach observations to the generator binding used by evaluation.model."""
    original_generate = evaluation.generate
    scheduler_type = None
    if getattr(capture, "capture_control", False):
        from generation.lookahead import DynamicJointWindowScheduler
        scheduler_type = DynamicJointWindowScheduler
    original_select = scheduler_type.select if scheduler_type is not None else None

    def observations():
        return getattr(capture, "observations", (capture,))

    def select(scheduler, **inputs):
        targets = []
        for item in observations():
            if not getattr(item, "capture_control", False) or getattr(item, "completed", False):
                continue
            if item.next_index == item.target:
                targets.append((item, "current_joint_selection"))
            elif item.event is not None and item.next_index == item.target + 1:
                targets.append((item, "next_joint_selection"))
        saved = observer.clone_tree(inputs) if targets else None
        result = original_select(scheduler, **inputs)
        for item, key in targets:
            item.control[key] = dict(inputs=saved, result=observer.clone_tree(asdict(result)))
        return result

    def generate(*positional, **kwargs):
        prompt = positional[1] if len(positional) > 1 else kwargs["prompt"]
        parameters = inspect.signature(original_generate).parameters
        configuration = {
            name: parameters[name].default
            for name in ("feature3_maturity_age", "cross_block_boundary_context_row_bits")
            if name in parameters and parameters[name].default is not inspect.Parameter.empty
        }
        configuration.update(kwargs)
        configuration = json_value(configuration)
        for item in observations():
            item.prompt_ids = observer.clone_tree(prompt)
            item.generation_config = configuration
        previous_callback = kwargs.get("state_capture_callback")

        def on_event(event):
            if previous_callback is not None:
                previous_callback(event)
            capture.on_event(event)

        kwargs["state_capture_callback"] = on_event
        if hasattr(capture, "on_forward_start"):
            previous_start = kwargs.get("forward_start_callback")
            def on_start(event):
                if previous_start is not None:
                    previous_start(event)
                capture.on_forward_start(event)
                for item in observations():
                    if item.prompt_ids is None:
                        item.prompt_ids = observer.clone_tree(prompt)
                        item.generation_config = configuration
            kwargs["forward_start_callback"] = on_start
        if getattr(capture, "capture_control", False):
            previous_end = kwargs.get("forward_end_callback")
            def on_end(event):
                if previous_end is not None:
                    previous_end(event)
                for item in observations():
                    selected = item.event
                    if (not getattr(item, "completed", False) and item.capture_control
                            and selected is not None and all(event[name] == selected[name]
                            for name in ("block_index", "step_index", "forward_kind"))):
                        item.current_state_references = {}
                        item.control["after_postprocess"] = capture_control_state(
                            retain_current=item.current_state_references)
                        item.control["transition"] = observer.clone_tree({**event, "trace": asdict(event["trace"])})
            kwargs["forward_end_callback"] = on_end
        return original_generate(*positional, **kwargs)

    evaluation.generate = generate
    if scheduler_type is not None:
        scheduler_type.select = select
    try:
        yield
    finally:
        evaluation.generate = original_generate
        if scheduler_type is not None:
            scheduler_type.select = original_select


def encode_tensor(value, *, name=""):
    tensor = value.detach().cpu().contiguous()
    dtype = str(tensor.dtype).split(".", 1)[1]
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view("<u2"), "bf16_raw", dtype
    if tensor.dtype == torch.float32:
        if name.endswith((".clip.row_max", ".clip.limit_fp32")):
            return tensor.view(torch.int32).numpy().view("<u4"), "float32_raw", dtype
        # Preserve non-BF16 intermediates, signed zeros and NaN payloads.
        if torch.equal(tensor.view(torch.int32), tensor.bfloat16().float().view(torch.int32)):
            return tensor.bfloat16().view(torch.int16).numpy().view("<u2"), "bf16_raw", dtype
        return tensor.view(torch.int32).numpy().view("<u4"), "float32_raw", dtype
    if tensor.is_floating_point():
        raise ValueError(f"unsupported actual floating dtype {tensor.dtype}")
    array = tensor.numpy()
    return array.astype(array.dtype.newbyteorder("<"), copy=False), "integer", dtype


def read_tensor(entry, *, index=None, payload_root=None):
    path = Path(entry["path"])
    if not path.is_absolute():
        if payload_root is None and index is None:
            raise ValueError("relative payload path requires index or payload_root")
        path = (Path(payload_root) if payload_root is not None else Path(index).parent) / path
    if path.stat().st_size != entry["byte_count"]:
        raise ValueError(f"wrong payload size: {path}")
    raw = np.fromfile(path, dtype=entry["dtype"]).reshape(entry["shape"])
    tensor = torch.from_numpy(raw.copy())
    if entry["encoding"] == "bf16_raw":
        tensor = tensor.view(torch.bfloat16)
    elif entry["encoding"] == "float32_raw":
        tensor = tensor.view(torch.float32)
    return tensor.to(getattr(torch, entry["runtime_dtype"]))


def validate_paths(output, index):
    output, index = output.resolve(), index.resolve()
    if ALGO is None:
        raise ValueError("set SUPRA_ALGORITHM_ROOT before generating reference_data payload")
    if ALGO not in output.parents or PROJECT in output.parents:
        raise ValueError(f"payload must be below {ALGO}")
    if output.exists() or index.exists():
        raise FileExistsError("use fresh payload and index paths")


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items() if isinstance(v, (str, int, float, bool, list, tuple, dict, type(None), torch.Tensor))}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def capture_provenance(provenance):
    """Keep numerical configuration without local source or artifact locations."""
    fields = {"dataset", "split", "sample_id", "logged_doc_id", "sample_doc",
              "sample_doc_hash", "fewshot", "request_task_name", "question_char_start",
              "seed", "model_arguments", "backend", "torch_version", "generation_request"}
    result = {key: value for key, value in provenance.items() if key in fields}
    if "model_arguments" in result:
        result["model_arguments"] = {key: value for key, value in result["model_arguments"].items()
            if not key.endswith(("_path", "_dir", "_root")) and key != "silu_table"
            and not (isinstance(value, str) and Path(value).is_absolute())}
    return result


def raw_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.numel() == 0:
        return True
    return torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))




# Capture generator state at forward boundaries.
_CONTROL_OBJECTS = ("block_state", "packed_state", "cross_block_prefix_state", "next_block_state")
_CONTROL_VALUES = (
    "decoding_mode", "decode_k", "decode_threshold", "full_sequence_recompute",
    "tokens", "capture_index", "block_index", "block_start", "block_end", "step_index",
    "prompt_length", "total_length", "region_start", "region_end", "block_offset",
    "schedule", "steps_per_block", "cache_refresh_due", "changed_rows", "changed_confidence",
    "block_transition_union", "packed_input_positions", "packed_refresh_positions", "packed_row_bits",
    "packed_profile", "cross_block_profile", "block_tokens_before", "locked_at_start",
    "handoff_verification_due", "source_a_handoff_pending", "mask_id", "budget_scale",
    "stability_bonus", "tau_high", "tau_low", "confirm_tau", "selective_a4_direct_tau",
    "tau_high_tail", "tau_high_tail_after_step", "tail_confirmation_policy",
    "cross_block_dependency_policy", "cross_block_pending_confidence_mode",
    "cross_block_carry_changed_positions", "cross_block_carry_refresh_due_positions",
    "dynamic_block_lookahead", "future_block_count", "next_block_rows",
    "next_preconfirmed", "next_frontier_service_count", "next_source_b_attempts",
    "next_last_confidence", "next_progress_positions", "next_added_positions",
    "next_source_a_positions", "next_source_b_positions", "next_progress_row_bits",
    "next_relation_score", "next_admission_quota", "next_changed_rows",
    "next_changed_confidence",
)


def capture_control_state(*, retain_current=None):
    """Copy current and future state at the generator callback."""
    from generation import engine
    frame = inspect.currentframe()
    try:
        while frame is not None and not (
                frame.f_code.co_name == "generate" and
                Path(frame.f_code.co_filename).resolve() == Path(engine.__file__).resolve()):
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("control observation requires the generation.engine.generate callback")
        local = frame.f_locals
        for name in ("tokens", "block_state", "schedule", "capture_index"):
            if name not in local:
                raise RuntimeError(f"control observation lacks required {name}")
        if retain_current is not None:
            # Keep the actual block object until the next forward so that a
            # tail transition between forwards is observed, not inferred.
            retain_current.update(block_state=local["block_state"],
                block_start=local["block_start"], block_end=local["block_end"],
                tokens=local["tokens"], cache_refresh_due=local["cache_refresh_due"])
        def clone(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: clone(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clone(v) for v in value]
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            raise TypeError(f"unsupported control state type {type(value).__name__}")
        result = {name: clone(local[name]) for name in _CONTROL_VALUES if name in local}
        for name in _CONTROL_OBJECTS:
            if name not in local:
                raise RuntimeError(f"control observation lacks required {name}")
            value = local[name]
            result[name] = None if value is None else clone(vars(value))
        trace = local.get("trace")
        result["previous_trace_event"] = clone(asdict(trace[-1])) if trace else None
        future = local.get("next_observation")
        if future is not None and future[0].numel():
            positions, _, logits, _ = future
            proposal, top_logit, confidence = local["reduce_candidates"](logits)
            probability = local["selected_candidate_probability"](
                logits, local["tokens"].index_select(1, local["block_end"] + positions),
                top_logit, confidence)
            result["future_candidates"] = clone(dict(
                positions=local["block_end"] + positions, top1=proposal,
                top_logit=top_logit, raw_confidence=confidence, probability=probability,
                action_confidence=engine.candidate_action_confidence(
                    proposal, confidence, local["suppressed_candidate_token_ids"])))
        return result
    finally:
        del frame


def capture_layer_dependency_input(module, observer):
    """Copy the live all-layer reduction before the selected layer updates it."""
    cache = getattr(module, "_LLaDABlock__cache", None)
    if not isinstance(cache, dict):
        raise RuntimeError("layer dependency observation lacks the shared cache")
    names = (
        "attn_monitor_dependency_layer_count", "attn_monitor_dependency_query_positions",
        "attn_monitor_dependency_max", "attn_monitor_prefix_dependency_query_positions",
        "attn_monitor_prefix_dependency_max", "attn_monitor_prefix_reverse_dependency_max",
        "attn_monitor_prefix_future_dependency_max",
    )
    required = names[:3]
    if any(name not in cache for name in required):
        raise RuntimeError("layer dependency observation lacks the preceding layer reduction")
    if int(cache[required[0]]) != int(module.layer_id):
        raise RuntimeError("dependency layer count differs from the selected layer input")
    return observer.clone_tree({name: cache[name] for name in names if name in cache})


def export_control_state(capture, metadata, output, index=None):
    if not getattr(capture, "capture_control", False) and not getattr(capture, "capture_boundary", False):
        return
    required = {"before_forward", "before_postprocess"}
    if capture.capture_control:
        required |= ({"after_postprocess", "request_end", "current_state_at_request_end"} if capture.control.get("request_end") is True
                     else {"next_forward", "next_layer_inputs"})
    if not required.issubset(capture.control):
        raise ValueError("control capture did not reach the next forward; no complete transition exported")
    def encode(value, name):
        if isinstance(value, torch.Tensor):
            raw, encoding, runtime_dtype = encode_tensor(value, name=name)
            path = output / (name + ".bin")
            raw.tofile(path)
            # Relative to the explicit payload root, independent of local capture paths.
            entry = dict(role="control", name=name, path=(os.path.relpath(path, index.parent) if index else path.name), shape=list(raw.shape),
                         dtype=raw.dtype.str, encoding=encoding, runtime_dtype=runtime_dtype,
                         byte_count=raw.nbytes)
            if raw.nbytes >= 16 * 1024**2:
                entry["sha256"] = hashlib.sha256(memoryview(raw).cast("B")).hexdigest()
            metadata["tensors"].append(entry)
            return {"tensor": name}
        if isinstance(value, dict):
            return {key: encode(item, name + "." + key) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item, name + "." + str(i)) for i, item in enumerate(value)]
        return value
    metadata["control_observation"] = dict(
        schema="supra-control-observation/v1",
        status="OBSERVED", payload_root="." if index else str(output),
        source="generation.engine.generate",
        checkpoints=encode(capture.control, "control"),
        prediction=encode(capture.event, "control.prediction"))


class IndexedForwardCapture:
    """Observe one generator capture index, including uncached full_sequence and scout."""

    def __init__(self, model, observer, capture_index, head_only=False, last_layer_only=False,
                 control_state=False, first_layer=None, layer_count=None, include_head=False,
                 on_complete=None):
        if head_only and last_layer_only:
            raise ValueError("head-only and last-layer-only select different observations")
        if (head_only or last_layer_only) and (first_layer is not None or layer_count is not None):
            raise ValueError("explicit layer range cannot be combined with head-only or last-layer-only")
        self.model, self.observer, self.target = model, observer, capture_index
        self.first_layer = 31 if last_layer_only else (0 if first_layer is None else first_layer)
        self.layer_count = 1 if last_layer_only else (2 if layer_count is None else layer_count)
        if (type(self.first_layer) is not int or type(self.layer_count) is not int or
                not 0 <= self.first_layer < 32 or not 1 <= self.layer_count <= 32 - self.first_layer):
            raise ValueError("capture requires 1..32 consecutive layers within L0..L31")
        self.layers, self.handles = [], []
        self.active = None
        self.next_index = 0
        self.event = self.prompt_ids = self.generation_config = None
        self.head_only, self.head = head_only, {}
        self.include_head = include_head or head_only
        self.on_complete = on_complete
        self.completed = False
        self.head_profiler_active = False
        self.capture_control = control_state
        self.observe_boundary = not control_state and not head_only and self.first_layer == 0 and self.layer_count == 1
        self.capture_boundary = False
        self.control = {}
        self.current_state_references = {}

    def on_forward_start(self, event):
        if event["capture_index"] == self.target:
            self.capture_boundary = self.observe_boundary and event["forward_kind"] == "boundary_refresh"

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def start(self):
        if self.capture_control or self.observe_boundary:
            def control_before(module, args, kwargs):
                if self.completed or not (self.capture_control or self.capture_boundary):
                    return
                if self.next_index == self.target:
                    self.control["before_forward"] = capture_control_state()
                elif self.event is not None:
                    retained = self.current_state_references
                    if retained:
                        begin, end = retained["block_start"], retained["block_end"]
                        self.control["current_state_at_next_forward"] = self.observer.clone_tree(dict(
                            block_state=vars(retained["block_state"]), block_start=begin, block_end=end,
                            tokens=retained["tokens"][:, begin:end],
                            cache_refresh_due=retained["cache_refresh_due"][:, begin:end]))
                    self.control["next_forward"] = capture_control_state()
                    hidden = args[0] if args else kwargs["x"]
                    positions = kwargs.get("query_position_ids")
                    if positions is None:
                        if hidden.shape[1] != self.control["next_forward"]["total_length"]:
                            raise ValueError("implicit next positions require the full sequence")
                        positions = torch.arange(hidden.shape[1], device=hidden.device).reshape(1, -1)
                    self.control["next_layer_inputs"] = self.observer.clone_tree(dict(
                        hidden=hidden, query_position_ids=positions,
                        activation_bits=module.q_proj.row_precision_context.require(
                            hidden.shape[1], hidden.device)))
                    self.complete()
            self.handles.append(self.model.model.transformer.blocks[0].register_forward_pre_hook(
                control_before, with_kwargs=True))
        if self.include_head:
            def norm_before(module, args):
                if self.next_index == self.target:
                    self.head["hidden"] = self.observer.clone_tree(args[0])
                    self.head["norm_weight"] = self.observer.clone_tree(module.weight)
                    self.head["epsilon"] = float(module.eps)

            def norm_after(module, args, result):
                if self.next_index == self.target:
                    self.head["norm_output"] = self.observer.clone_tree(result)

            def profile(frame, event, result):
                if event == "return" and frame.f_code.co_name == "accelerated_linear_bf16":
                    local = frame.f_locals
                    self.head.update({key: self.observer.clone_tree(local[name]) for key, name in (
                        ("weight_codes", "weight_codes"), ("weight_scale", "weight_scales_bf16"),
                        ("activation_scale", "activation_scales"), ("accumulator", "sums"))})
                    self.head["activation_codes"] = self.observer.clone_tree(local["codes"][:local["rows"]])

            def head_before(module, args):
                if self.next_index == self.target:
                    if sys.getprofile() is not None or module.weight_bits != 8:
                        raise ValueError("head capture requires W8 and no active profiler")
                    sys.setprofile(profile)
                    self.head_profiler_active = True

            def head_after(module, args, result):
                if self.next_index == self.target:
                    sys.setprofile(None)
                    self.head_profiler_active = False
                    self.head["output"] = self.observer.clone_tree(result)

            norm, head = self.model.model.transformer.ln_f, self.model.model.transformer.ff_out
            self.handles.extend((norm.register_forward_pre_hook(norm_before),
                norm.register_forward_hook(norm_after), head.register_forward_pre_hook(head_before),
                head.register_forward_hook(head_after)))
            if self.head_only:
                return self
        blocks = self.model.model.transformer.blocks
        if len(blocks) < self.first_layer + self.layer_count:
            raise ValueError("model does not contain the selected layer range")
        for layer_id, block in enumerate(blocks[self.first_layer:self.first_layer + self.layer_count]):
            def before(module, args, kwargs, layer=layer_id):
                if self.next_index != self.target:
                    return
                if len(self.layers) != layer:
                    raise RuntimeError("selected forward repeats or skips a captured layer")
                before_forward = self.control.get("before_forward", {})
                boundary = before_forward.get("block_index", 0) > 0 and before_forward.get("packed_state") is None
                dependency = self.capture_control and not before_forward.get("full_sequence_recompute", False) and not boundary
                if dependency and layer == 0 and self.first_layer > 0:
                    self.control["before_first_layer_dependency"] = capture_layer_dependency_input(module, self.observer)
                if dependency and self.first_layer + layer == 31:
                    self.control["before_l31_dependency"] = capture_layer_dependency_input(module, self.observer)
                call = inspect.signature(module.forward).bind(*args, **kwargs)
                call.apply_defaults()
                call = dict(call.arguments)
                hidden, past = call.pop("x"), call.pop("layer_past")
                if call.get("attention_bias") is not None:
                    raise ValueError("explicit attention mask capture is not implemented")
                inputs = dict(hidden=self.observer.clone_tree(hidden),
                    activation_bits=self.observer.clone_tree(module.q_proj.row_precision_context.require(
                        hidden.shape[1], hidden.device)))
                if past is not None:
                    if len(past) != 3:
                        raise ValueError("capture requires K8/V8 cache")
                    inputs.update({"retained_" + n: self.observer.clone_tree(t)
                        for n, t in zip(self.observer.CACHE_NAMES, past)})
                watched = self.observer.LayerObserver(module, inputs)
                self.layers.append(dict(layer=layer, model_layer=self.first_layer + layer,
                    inputs=inputs, kwargs=self.observer.clone_tree(call),
                    config=asdict(module.config), observer=watched, cache_initialized=past is not None))
                self.active = watched
                watched.__enter__()

            def after(module, args, kwargs, output, layer=layer_id):
                if self.active is None:
                    return
                watched = self.active
                watched.__exit__(None, None, None)
                self.active = None
                record = self.layers[layer]
                record["expected"] = watched.expected
                record["numerical_observations"] = watched.numerical_observations()
                record["cache_refs"] = output[1]
                if (self.capture_control or self.capture_boundary) and int(module.layer_id) == 0:
                    scout = getattr(module, "_LLaDABlock__cache", {}).get("boundary_layer0_scout")
                    if scout is not None:
                        if "score_q8" not in scout:
                            raise RuntimeError("completed L0 scout is missing score_q8")
                        self.control["after_l0_scout"] = self.observer.clone_tree(scout)
                for name, value in zip(self.observer.CACHE_NAMES, output[1]):
                    watched.save("cache_write_" + name, value)
                    if not record["cache_initialized"]:
                        # No retained cache is present. These bytes initialize the RTL
                        # arena only; all positions are overwritten before Attention.
                        record["inputs"]["retained_" + name] = torch.zeros_like(value, device="cpu")
                if not self.observer.raw_equal(watched.expected["block_output"], output[0]):
                    raise RuntimeError("layer output differs from residual checkpoint")
                sequence = output[1][0].shape[-2]
                pos = record["kwargs"].get("query_position_ids")
                if pos is None:
                    if output[0].shape[1] != sequence:
                        raise ValueError("implicit positions require a complete sequence")
                    pos = torch.arange(sequence).reshape(1, -1)
                record["positions"] = pos.flatten().long()
                writes = record["kwargs"].get("kv_write_position_ids")
                if writes is None:
                    mask = record["kwargs"].get("replace_position_kv")
                    if mask is None:
                        mask = record["kwargs"].get("replace_position")
                    writes = pos if mask is None else mask[0].nonzero().flatten()
                record["write_positions"] = writes.flatten().long()
                sine, cosine = module.rotary_emb.get_rotary_embedding(sequence, output[0].device)
                record["inputs"].update(rope_sine=self.observer.clone_tree(sine),
                    rope_cosine=self.observer.clone_tree(cosine),
                    v_scale=self.observer.clone_tree(module._spinquant_v_cache_codec.expanded_scales(
                        module.config.n_heads, output[0].device)))
                coeff = module._target_numeric_silu_coefficients
                for i, value in enumerate(coeff):
                    record["inputs"]["silu_coefficients." + str(i)] = torch.tensor(value, dtype=torch.bfloat16)
                record["r4"] = module._target_numeric_r4
                record["norm_implementation"] = module.attn_norm._target_numeric_rms_norm
                if layer:
                    source = self.layers[layer - 1]
                    matches = pos.flatten()[:, None] == source["positions"][None, :]
                    if not bool((matches.sum(1) == 1).all()):
                        raise ValueError(f"layer {self.first_layer + layer} input does not map uniquely to preceding tokens")
                    source_token_indices = matches.to(torch.int64).argmax(1)
                    record["source_token_indices"] = source_token_indices
                    if not self.observer.raw_equal(source["expected"]["block_output"].index_select(1, source_token_indices),
                                                   record["inputs"]["hidden"]):
                        raise RuntimeError(f"layer {self.first_layer + layer} input differs from actual preceding output")

            self.handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(block.register_forward_hook(after, with_kwargs=True))
        return self

    def on_event(self, event):
        if self.completed:
            return
        if event.capture_index != self.next_index:
            raise RuntimeError("generator capture indices are not consecutive")
        self.next_index += 1
        if event.capture_index != self.target:
            return
        if self.capture_control or self.capture_boundary:
            self.control["before_postprocess"] = capture_control_state()
        if self.include_head and event.forward_kind == "local_confirmation":
            state = self.control.get("before_postprocess")
            if state is None:
                state = capture_control_state()
            self.head["prediction_mask"] = state["block_state"]["state"].reshape(-1) == 1
        if self.head_only:
            self.event = self.observer.clone_tree(asdict(event))
            if not self.capture_control:
                self.complete()
            return
        if len(self.layers) != self.layer_count:
            raise RuntimeError("selected event lacks the requested model layers")
        self.event = self.observer.clone_tree(asdict(event))
        for record in self.layers:
            commits = record["write_positions"]
            if (event.layer0_global_selected_deep and self.first_layer == 0 and record["layer"] == 0
                    and not getattr(event, "layer0_keep_global_cache", False)):
                # The generator publishes its actual deep query set even when
                # this capture only observes L0. Writes already exclude any
                # deferred KV positions; intersecting preserves that policy.
                deep = event.input_positions.reshape(-1).to(commits.device)
                if len(self.layers) > 1 and not torch.equal(
                        deep, self.layers[1]["positions"].to(deep.device)):
                    raise ValueError("boundary event deep positions differ from captured L1")
                commits = commits[torch.isin(commits, deep)]
            record["commit_positions"] = commits
            for name, tensor in zip(self.observer.CACHE_NAMES, record.pop("cache_refs")):
                record["observer"].save("cache_" + name, tensor)
            record.pop("observer")
        if not self.capture_control:
            self.complete()

    def complete(self):
        self.completed = True
        if self.on_complete is None:
            raise self.observer.CaptureComplete()
        # Keep hook registrations until the surrounding capture context exits.
        # PyTorch may already be iterating a snapshot of this module's hooks;
        # removing with_kwargs metadata here changes a queued hook's signature.
        self.on_complete(self)

    def finish_request(self):
        if self.completed:
            return
        if self.event is None:
            raise ValueError(f"request ended before capture index {self.target}")
        if not self.capture_control or "after_postprocess" not in self.control:
            raise ValueError(f"capture index {self.target} lacks its final control transition")
        retained = self.current_state_references
        if not retained:
            raise ValueError("request-end capture lacks the actual final block state")
        begin, end = retained["block_start"], retained["block_end"]
        self.control["current_state_at_request_end"] = self.observer.clone_tree(dict(
            block_state=vars(retained["block_state"]), block_start=begin, block_end=end,
            tokens=retained["tokens"][:, begin:end],
            cache_refresh_due=retained["cache_refresh_due"][:, begin:end]))
        self.control["request_end"] = True
        self.complete()

    def __exit__(self, *exc):
        if self.head_profiler_active:
            sys.setprofile(None)
            self.head_profiler_active = False
        if self.active is not None:
            self.active.__exit__(*exc)
            self.active = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture_event_selector(value):
    """Use a real capture index or the generator's block/step/phase identity."""
    if set(value) == {"capture_index"}:
        if type(value["capture_index"]) is not int or value["capture_index"] < 0:
            raise ValueError("capture indices must be unique nonnegative integers")
        return dict(value)
    if set(value) != {"block_index", "step_index", "phase"}:
        raise ValueError("event selector needs capture_index or block_index, step_index and phase")
    if (type(value["block_index"]) is not int or value["block_index"] < 0 or
            type(value["step_index"]) is not int or value["step_index"] < 0 or
            value["phase"] not in ("full_sequence", "boundary_refresh", "local_block",
                                   "local_confirmation", "local_forced_finish")):
        raise ValueError("invalid block/step/phase event selector")
    return dict(block_index=value["block_index"], step_index=value["step_index"], forward_kind=value["phase"])


class ForwardCaptureSet:
    """Share one generator run across selected forwards; export each before releasing it."""

    def __init__(self, model, observer, selections, on_complete):
        self.observations = []
        self.model = model
        self.observer = observer
        self.on_complete = on_complete
        self.range = None
        self.range_started = self.range_ended = False
        self.started_indices = set()
        if isinstance(selections, dict):
            if set(selections) != {"start", "end"}:
                raise ValueError("capture range needs start and end endpoints")
            self.range = {}
            for name in ("start", "end"):
                endpoint = dict(selections[name])
                layer = endpoint.pop("layer", None)
                if type(layer) is not int or not 0 <= layer < 32:
                    raise ValueError("range endpoint layer must be in 0..31")
                self.range[name] = dict(selector=capture_event_selector(endpoint), layer=layer)
            if (self.range["start"]["selector"] == self.range["end"]["selector"] and
                    self.range["start"]["layer"] > self.range["end"]["layer"]):
                raise ValueError("range end layer precedes start layer")
            self.capture_control = True
            return
        if not isinstance(selections, list) or not selections:
            raise ValueError("selections must be a nonempty list or a start/end range")
        names, indices = set(), set()
        for selection in selections:
            if not isinstance(selection, dict):
                raise ValueError("each selection must be an object")
            selection = dict(selection)
            name = selection.pop("name", None)
            if (not isinstance(name, str) or not name or
                    any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
                    or name in names):
                raise ValueError("selection names must be unique simple file names")
            allowed = {"capture_index", "block_index", "step_index", "phase", "head_only", "first_layer", "layer_count", "control_state", "include_head"}
            if set(selection) - allowed:
                raise ValueError("unknown capture selection fields")
            event = capture_event_selector({key: selection.pop(key) for key in
                ("capture_index", "block_index", "step_index", "phase") if key in selection})
            identity = tuple(event.items())
            if identity in indices:
                raise ValueError("capture indices/event identities must be unique; use include_head for the same event")
            names.add(name); indices.add(identity)
            selection["capture_index"] = event.get("capture_index")
            item = IndexedForwardCapture(model, observer, **selection,
                on_complete=lambda item, name=name: self.export_completed(name, item))
            item.name, item.selector = name, event
            self.observations.append(item)
        self.observations.sort(key=lambda item: item.target if item.target is not None else float("inf"))
        self.capture_control = any(item.capture_control for item in self.observations)

    def complete(self):
        return bool(self.observations) and (self.range is None or self.range_ended) and all(
            item.completed for item in self.observations)

    def on_forward_start(self, event):
        index = event["capture_index"]
        if index in self.started_indices:
            raise ValueError("generator repeated a forward start index")
        self.started_indices.add(index)
        matches = lambda selector: all(event.get(key) == value for key, value in selector.items())
        if self.range is not None:
            start, end = self.range["start"], self.range["end"]
            ending = matches(end["selector"])
            if not self.range_started:
                self.range_started = matches(start["selector"])
                if ending and not self.range_started:
                    raise ValueError("range end event precedes start event")
                if not self.range_started:
                    return
                first = start["layer"]
            elif self.range_ended:
                return
            else:
                first = 0
            last = end["layer"] if ending else 31
            if self.observations and index != self.observations[-1].target + 1:
                raise ValueError("range skipped an actual forward")
            self.range_ended = ending
            name = f"forward_{index}"
            item = IndexedForwardCapture(self.model, self.observer, index,
                first_layer=first, layer_count=last-first+1,
                include_head=not ending, control_state=not ending,
                on_complete=lambda item: self.export_completed(name, item))
            item.name, item.selector, item.next_index = name, dict(event), index
            item.on_forward_start(event)
            self.observations.append(item)
            self.stack.enter_context(item)
        else:
            selected = [item for item in self.observations if matches(item.selector)]
            if len(selected) > 1:
                raise ValueError("capture index and event identity select the same forward; use include_head")
            for item in selected:
                item.target = index
                item.on_forward_start(event)

    def export_completed(self, name, item):
        self.on_complete(name, item)
        item.layers.clear()
        item.head.clear()
        item.control.clear()
        item.current_state_references.clear()
        if self.complete():
            raise self.observer.CaptureComplete()

    def __enter__(self):
        self.stack = ExitStack()
        try:
            for item in self.observations:
                self.stack.enter_context(item)
        except BaseException:
            self.stack.close()
            raise
        return self

    def on_event(self, event):
        for item in self.observations:
            item.on_event(event)
        if self.complete():
            raise self.observer.CaptureComplete()

    def finish_request(self):
        if self.range is not None and not self.range_ended:
            raise ValueError("request ended without capture range " + ("end" if self.range_started else "start") + " event")
        if any(item.target is None for item in self.observations):
            raise ValueError("request ended without a selected block/step/phase event")
        for item in self.observations:
            item.finish_request()

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def write_capture_tensor(raw, role, name, output, index, encoding, runtime_dtype,
                         shared=None, first_layer=0, head=False):
    """Reuse equal immutable inputs within one capture request."""
    path = output / f"{role}.{name}.bin"
    key = None
    if shared is not None and role == "input":
        if head and name in {"norm_weight", "weight_codes", "weight_scale", "epsilon"}:
            key = "head." + name
        elif not head:
            parts = name.split(".", 1)
            suffix = parts[-1]
            constant = (suffix.startswith("weight.") or suffix.endswith(".weight") or
                        suffix in {"v_scale", "softmax_exp_lut", "softmax_reciprocal_lut",
                                   "silu_breakpoints", "rope_sine", "rope_cosine"})
            if constant:
                key = name
                if len(parts) == 2 and parts[0].startswith("layer"):
                    key = f"layer{int(parts[0][5:]) + first_layer}.{suffix}"
    entry = dict(role=role, name=name, shape=list(raw.shape), dtype=raw.dtype.str,
                 encoding=encoding, runtime_dtype=runtime_dtype, byte_count=raw.nbytes)
    previous = shared.get(key) if key is not None else None
    if previous and all(previous[1][field] == entry[field] for field in
                        ("shape", "dtype", "encoding", "runtime_dtype", "byte_count")):
        saved = np.memmap(previous[0], dtype=raw.dtype, mode="r", shape=raw.shape)
        equal = np.array_equal(saved, raw)
        del saved
        if equal:
            entry["path"] = os.path.relpath(previous[0], index.parent)
            if "sha256" in previous[1]:
                entry["sha256"] = previous[1]["sha256"]
            return entry
    raw.tofile(path)
    entry["path"] = os.path.relpath(path, index.parent)
    if raw.nbytes >= 16 * 1024**2:
        entry["sha256"] = hashlib.sha256(memoryview(raw).cast("B")).hexdigest()
    if key is not None:
        shared[key] = (path, entry)
    return entry


def export_head(capture, provenance, output, index, *, shared=None):
    validate_paths(output, index)
    head, event = capture.head, capture.event
    candidate_positions = event["prediction_positions"].flatten().long()
    mask = event["prediction_mask"].reshape(-1)
    if event.get("forward_kind") == "local_confirmation":
        mask = head["prediction_mask"]
    if mask.dtype != torch.bool or mask.numel() != candidate_positions.numel():
        raise ValueError("head prediction mask must match the recorded candidate positions")
    current_positions = candidate_positions[mask]
    future_positions = event.get("future_prediction_positions")
    future_positions = (torch.empty(0, dtype=torch.long) if future_positions is None
                        else future_positions.flatten().long())
    positions = torch.cat((current_positions, future_positions))
    if positions.unique().numel() != positions.numel():
        raise ValueError("current/future head positions must be unique and disjoint")
    if positions.numel() == 0:
        raise ValueError("head capture has no active prediction rows")
    tokens = head["hidden"].shape[1]
    input_positions = event["input_positions"].flatten().long()
    if event.get("full_sequence_recompute", False):
        if tokens != current_positions.numel() or future_positions.numel():
            raise ValueError("full-sequence head rows must match the active current positions")
        selected = torch.arange(tokens, device=positions.device)
    elif tokens == input_positions.numel():
        matches = positions[:, None] == input_positions[None, :]
        if not bool((matches.sum(1) == 1).all()):
            raise ValueError("active head prediction positions do not map uniquely to executed tokens")
        selected = matches.long().argmax(1)
    elif tokens == candidate_positions.numel() and not future_positions.numel():
        selected = torch.arange(tokens, device=positions.device)[mask]
    else:
        raise ValueError("head input must cover executed queries or the recorded candidate positions")
    # Selecting independent output rows changes no numerical expected value.
    inputs = dict(hidden=head["hidden"].index_select(1, selected),
        norm_weight=head["norm_weight"], weight_codes=head["weight_codes"],
        weight_scale=head["weight_scale"], positions=positions,
        prediction_mask=torch.ones(1, positions.numel(), dtype=torch.bool),
        epsilon=torch.tensor(head["epsilon"], dtype=torch.float32))
    expected = {key: head[key].index_select(1 if key in ("norm_output", "output") else 0, selected)
        for key in ("norm_output", "activation_codes", "activation_scale", "accumulator", "output")}
    output.mkdir(parents=True)
    tensors = []
    for role, values in (("input", inputs), ("expected", expected)):
        for name, value in values.items():
            raw, encoding, runtime_dtype = encode_tensor(value, name=name)
            entry = write_capture_tensor(raw, role, name, output, index, encoding, runtime_dtype,
                                         shared, head=True)
            tensors.append(entry)
    metadata = dict(schema="supra-final-output-reference-data/v1", reference_data_kind="real_cuda_final_output",
        tokens=positions.numel(), hidden=4096, vocab=head["weight_codes"].shape[0], weight_bits=8,
        activation_bits=8, provenance=capture_provenance(provenance), tensors=tensors,
        generation_config=getattr(capture, "generation_config", None),
        forward={k: json_value(event[k]) for k in ("capture_index", "block_index", "step_index", "forward_kind")},
        current_prediction_positions=current_positions.tolist(),
        future_prediction_positions=future_positions.tolist(),
        validation=dict(status="EXPORTED_NOT_REPLAYED"))
    export_control_state(capture, metadata, output, index)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(metadata, indent=2) + "\n")


@torch.inference_mode()
def replay_head(index, device="cuda", *, payload_root=None):
    from numerics.operator_kernels import rms_norm_bf16_cuda, RMSNormNumericWorkspace
    from numerics.linear_kernels import accelerated_linear_bf16, LinearNumericWorkspace
    metadata = json.loads(index.read_text())
    inputs = {e["name"]: read_tensor(e, index=index, payload_root=payload_root).to(device) for e in metadata["tensors"] if e["role"] == "input"}
    expected = {e["name"]: read_tensor(e, index=index, payload_root=payload_root) for e in metadata["tensors"] if e["role"] == "expected"}
    if not bool((inputs["norm_weight"] == 1).all()):
        raise ValueError("final norm requires the deployed folded gamma")
    norm = rms_norm_bf16_cuda(inputs["hidden"], None, float(inputs["epsilon"]), RMSNormNumericWorkspace())
    actual = dict(norm_output=norm)
    def profile(frame, event, result):
        if event == "return" and frame.f_code.co_name == "accelerated_linear_bf16":
            local = frame.f_locals
            actual.update(activation_codes=local["codes"][:local["rows"]],
                activation_scale=local["activation_scales"], accumulator=local["sums"])
    if sys.getprofile() is not None:
        raise RuntimeError("head replay requires no active profiler")
    sys.setprofile(profile)
    try:
        actual["output"] = accelerated_linear_bf16(norm.reshape(-1, 4096), inputs["weight_codes"],
            inputs["weight_scale"], LinearNumericWorkspace(), activation_bits=8).reshape_as(expected["output"])
    finally:
        sys.setprofile(None)
    checks = [dict(name=name, match=raw_equal(value, actual[name])) for name, value in expected.items()]
    metadata["validation"] = dict(status="PASS" if all(c["match"] for c in checks) else "FAIL",
        backend=device, checks=checks)
    index.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["validation"]), flush=True)
    if metadata["validation"]["status"] != "PASS":
        raise RuntimeError("actual final norm/head replay differs")


def activation_field_names(values, *, to_algorithm=False):
    """Translate precision names once at the algorithm/hardware data boundary."""
    source, target = (("activation_bits", "row_bits") if to_algorithm else ("row_bits", "activation_bits"))
    result = {}
    for name, value in values.items():
        parts = name.split(".")
        if parts[-1] == source:
            parts[-1] = target
        key = ".".join(parts)
        if key in result and not raw_equal(result[key], value):
            raise ValueError(f"conflicting precision fields for {key}")
        result[key] = value
    return result


def forward_event_metadata(event):
    fields = ("capture_index", "block_index", "step_index", "forward_kind", "nfe_before",
        "input_positions", "prediction_positions", "refresh_positions", "row_bits",
        "prediction_mask", "layer0_global_selected_deep")
    rename = {"nfe_before": "execution_before", "row_bits": "activation_bits"}
    return {rename.get(key, key): json_value(event[key]) for key in fields}


def export_capture(capture, provenance, output, index, *, shared=None):
    validate_paths(output, index)
    if any(record["kwargs"].get("attention_bias") is not None for record in capture.layers):
        raise ValueError("attention_bias has no corresponding input in the prepared Attention descriptor")
    first_layer = getattr(capture, "first_layer", 0)
    if not 0 <= first_layer < 32 or not 1 <= len(capture.layers) <= 32 - first_layer:
        raise ValueError("layer export requires 1..32 consecutive layers within L0..L31")
    forward_fields = forward_event_metadata(capture.event)
    output.mkdir(parents=True)
    inputs, expected, layers = {}, {}, []
    event = capture.event
    for record in capture.layers:
        record["inputs"] = activation_field_names(record["inputs"])
        prefix = "layer" + str(record["layer"]) + "."
        inputs.update({prefix + k: v for k, v in record["inputs"].items()})
        expected.update({prefix + k: v for k, v in record["expected"].items()})
        kwargs = {}
        for name, value in record["kwargs"].items():
            if isinstance(value, torch.Tensor):
                key = prefix + "call." + name
                inputs[key] = value
                kwargs[name] = {"tensor": key}
            else:
                kwargs[name] = value
        positions = record.get("positions", record["kwargs"].get("query_position_ids"))
        if positions is None:
            positions = event["input_positions"].reshape(1, -1).long()
        writes = record.get("write_positions", record["kwargs"].get("kv_write_position_ids"))
        if writes is None:
            mask = record["kwargs"].get("replace_position_kv")
            if mask is None:
                mask = record["kwargs"].get("replace_position")
            if mask is None:
                raise ValueError("cannot identify actual cached KV-write positions")
            writes = mask[0].nonzero().flatten()[None, :]
        inputs[prefix + "positions"] = positions.reshape(-1)
        inputs[prefix + "kv_write_positions"] = writes.reshape(-1)
        commits = record.get("commit_positions", writes).reshape(-1)
        inputs[prefix + "cache_commit_positions"] = commits
        bits = record["inputs"]["activation_bits"]
        layers.append(dict(layer=record["layer"], config=record["config"], kwargs=kwargs,
            tokens=bits.numel(), a4_token_count=int((bits == 4).sum()), a8_token_count=int((bits == 8).sum()),
            positions=positions.reshape(-1).tolist(), kv_write_positions=writes.reshape(-1).tolist(),
            cache_commit_positions=commits.tolist(), cache_initialized=record.get("cache_initialized", True),
            r4=record["r4"], norm_implementation=record["norm_implementation"], gamma_bypass=True))
        if first_layer:
            layers[-1]["model_layer"] = first_layer + record["layer"]
        observations = record.get("numerical_observations")
        if observations is None and record.get("observer") is not None:
            observations = record["observer"].numerical_observations()
        clipping = observations.get("clipping") if observations else None
        if clipping is not None and observations["clipping_observations"]:
            layers[-1]["clipping"] = {
                name: dict(installed=description["installed"],
                    observation_status=observations["clipping_observations"][name],
                    rule=description["rule"],
                    ratio_bf16_raw=(int(description["ratio_bf16"].view(torch.int16)) & 0xffff)
                        if description["installed"] else None)
                for name, description in clipping.items()}
    from numerics.bf16 import _SILU_BREAKPOINTS
    for table in ("softmax_exp_lut", "softmax_reciprocal_lut"):
        if any(not torch.equal(inputs["layer0." + table], inputs[f"layer{i}." + table])
               for i in range(1, len(layers))):
            raise ValueError("layers use different softmax LUTs; shared table export is unavailable")
    inputs.update(prompt_token_ids=capture.prompt_ids, activation_bits=capture.layers[0]["inputs"]["activation_bits"],
        positions=inputs["layer0.positions"], hidden=capture.layers[0]["inputs"]["hidden"],
        softmax_exp_lut=inputs["layer0.softmax_exp_lut"],
        softmax_reciprocal_lut=inputs["layer0.softmax_reciprocal_lut"],
        silu_breakpoints=torch.tensor(_SILU_BREAKPOINTS, dtype=torch.bfloat16))
    total = sum(v.numel() * v.element_size() for group in (inputs, expected) for v in group.values())
    if shutil.disk_usage(output).free < total + 64 * 1024**2:
        raise OSError("insufficient space for selected layer payloads")
    tensors = []
    for role, values in (("input", inputs), ("expected", expected)):
        for name, value in values.items():
            raw, encoding, runtime_dtype = encode_tensor(value, name=name)
            entry = write_capture_tensor(raw, role, name, output, index, encoding, runtime_dtype,
                                         shared, first_layer)
            entry["source"] = "actual_cuda_forward" if role == "expected" else "actual_deployed_input_or_constant"
            if role == "input" and any(name.startswith(f"layer{r['layer']}.retained_")
                    and not r.get("cache_initialized", True) for r in capture.layers):
                entry["source"] = "cache_initialization"
            tensors.append(entry)
    first = capture.layers[0]["inputs"]
    connections = []
    for i in range(1, len(layers)):
        indices = capture.layers[i].get("source_token_indices", torch.arange(layers[i]["tokens"]))
        identity = indices.tolist() == list(range(layers[i-1]["tokens"]))
        connections.append(dict(source=f"layer{i-1}.block_output", destination=f"layer{i}.hidden",
            mapping="identity" if identity else "index_select", source_token_indices=indices.tolist()))
    metadata = dict(schema=SCHEMA, reference_data_kind="real_cuda_consecutive_layers", seed=provenance["seed"],
        tokens=layers[0]["tokens"], sequence=first["retained_key_codes"].shape[-2], hidden=4096, ffn=12288,
        heads=32, head_dim=128, layer_count=len(layers), standard_layer=True, weight_group=-1,
        provenance=capture_provenance(provenance), generation_config=json_value(capture.generation_config),
        forward=forward_fields,
        layers=layers, token_order="logical_input_order", gamma_bypass=True,
        weight_layout="output_channel,input_channel; unpacked signed int8 W4 codes",
        validation=dict(status="EXPORTED_NOT_REPLAYED"), tensors=tensors)
    if connections:
        metadata["connections"] = connections
        if len(connections) == 1:
            metadata["connection"] = connections[0]
        metadata["model_layer_index"] = first_layer
    else:
        metadata.update(model_layer_index=first_layer, initial_layer_state="hidden_and_retained_cache")
    if "future_prediction_positions" in event:
        metadata["forward"]["future_prediction_positions"] = json_value(event["future_prediction_positions"])
    if any("clipping" in layer for layer in layers):
        if not all("clipping" in layer for layer in layers):
            raise ValueError("clipping observation capability must cover every selected layer")
        rules = {state["rule"] for layer in layers for state in layer["clipping"].values()}
        if len(rules) != 1:
            raise ValueError("all observed Linears must use the same clipping rule")
        metadata["numeric_capabilities"] = {"a4_clipping": rules.pop()}
    from layer_reference_data import validate_numeric_capabilities
    validate_numeric_capabilities(metadata)
    export_control_state(capture, metadata, output, index)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@torch.inference_mode()
def replay(index, observer, device="cuda", *, payload_root=None):
    metadata = json.loads(index.read_text())
    from layer_reference_data import validate_numeric_capabilities
    validate_numeric_capabilities(metadata)
    inputs = {e["name"]: read_tensor(e, index=index, payload_root=payload_root)
              for e in metadata["tensors"] if e["role"] == "input"}
    expected = {e["name"]: read_tensor(e, index=index, payload_root=payload_root)
                for e in metadata["tensors"] if e["role"] == "expected"}
    checks = []
    previous = None
    for ordinal, record in enumerate(metadata["layers"]):
        prefix = "layer" + str(record["layer"]) + "."
        local = {k[len(prefix):]: v for k, v in inputs.items() if k.startswith(prefix)}
        block = observer.replay_block(record["config"], activation_field_names(local, to_algorithm=True), device=device,
            layer_id=record.get("model_layer", record["layer"]))
        hidden = local["hidden"].to(device) if previous is None else previous
        if previous is not None:
            connections = metadata.get("connections", [metadata.get("connection")])
            connection = connections[ordinal - 1]
            if connection["mapping"] == "index_select":
                hidden = previous.index_select(1, torch.tensor(connection["source_token_indices"], device=device))
        if previous is not None:
            checks.append(dict(name=prefix + "input_from_previous_layer", match=observer.raw_equal(hidden, local["hidden"])))
        def call_kwargs():
            kw = {k: inputs[v["tensor"]].to(device) if isinstance(v, dict) else v
                  for k, v in record["kwargs"].items()}
            kw["layer_past"] = (tuple(local["retained_" + n].to(device).clone() for n in observer.CACHE_NAMES)
                if record.get("cache_initialized", True) else None)
            return kw
        baseline, baseline_cache = block(hidden, **call_kwargs())
        with observer.LayerObserver(block) as watched:
            observed, observed_cache = block(hidden, **call_kwargs())
        checks.append(dict(name=prefix + "observer_output_unchanged", match=observer.raw_equal(baseline, observed)))
        for n, a, b in zip(observer.CACHE_NAMES, baseline_cache, observed_cache):
            checks.append(dict(name=prefix + "observer_cache_unchanged." + n, match=observer.raw_equal(a, b)))
            watched.save("cache_write_" + n, b)
            persistent = b.clone()
            keep = torch.ones(b.shape[-2], dtype=torch.bool, device=device)
            keep[local["cache_commit_positions"].to(device).long()] = False
            persistent[:, :, keep] = local["retained_" + n].to(device)[:, :, keep]
            watched.save("cache_" + n, persistent)
        positions = local["positions"].long()
        for label, full in (("sin", "sine"), ("cos", "cosine")):
            checks.append(dict(name=prefix + "rope_operand." + label, match=observer.raw_equal(
                local["rope_selected_" + label], local["rope_" + full].bfloat16().index_select(2, positions))))
        writes = local["cache_commit_positions"].long()
        keep = torch.ones(local["retained_key_codes"].shape[-2], dtype=torch.bool)
        keep[writes] = False
        for n in observer.CACHE_NAMES:
            checks.append(dict(name=prefix + "unwritten_cache_unchanged." + n,
                match=raw_equal(local["retained_" + n][:, :, keep],
                    expected[prefix + "cache_" + n][:, :, keep])))
        for operand, cache in (("qk_key_codes", "key_codes"), ("qk_key_scale", "key_scale"),
                               ("pv_value_codes", "value_codes")):
            a, b = expected[prefix + operand], expected[prefix + "cache_write_" + cache]
            checks.append(dict(name=prefix + "attention_sees_written_cache." + operand,
                               match=observer.raw_equal(a, b.to(a.dtype))))
        for name, actual in watched.expected.items():
            reference = expected[prefix + name]
            check = dict(name=prefix + name, match=observer.raw_equal(reference, actual))
            if not check["match"] and actual.shape == reference.shape and actual.dtype == reference.dtype:
                a = actual.contiguous().view(torch.uint8).flatten()
                b = reference.contiguous().view(torch.uint8).flatten()
                first = int((a != b).nonzero()[0])
                check.update(first_byte=first, expected_byte=int(b[first]), actual_byte=int(a[first]))
            checks.append(check)
        if set(watched.expected) != {k[len(prefix):] for k in expected if k.startswith(prefix)}:
            raise RuntimeError("replay checkpoint coverage differs from capture")
        previous = baseline
        del block
    report = dict(status="PASS" if all(c["match"] for c in checks) else "FAIL", backend=str(device),
        numerical_source="algorithm/llada",
        chained_layers=len(metadata["layers"]) > 1, expected_checkpoints=len(expected), checks=checks)
    metadata["validation"] = report
    index.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}), flush=True)
    for check in checks:
        if not check["match"]:
            print(json.dumps(check), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("real capture replay differs; original expected preserved")


def load_request_config(path):
    from evaluation.generate import resolve_model_args

    path = Path(path)
    request = json.loads(path.read_text())
    task = {}
    explicit_length = request.get("model_args", {}).get("gen_length")
    explicit_max = request.get("generation_args", {}).get("max_gen_toks")
    if "task_config" in request:
        task_path = Path(request["task_config"])
        if not task_path.is_absolute():
            task_path = path.parent / task_path
        task = json.loads(task_path.read_text())
        request["model_args"] = {**task["model_args"], **request.get("model_args", {})}
        request["generation_args"] = {**task.get("gen_kwargs", {}), **request.get("generation_args", {})}
    if not isinstance(request.get("prompt"), str) or not request["prompt"]:
        raise ValueError("request prompt must be nonempty text")
    for name in ("model_args", "generation_args"):
        if not isinstance(request.get(name), dict):
            raise ValueError(f"request {name} must be an object")
    request["model_args"] = resolve_model_args(
        dict(task, model_args=request["model_args"]), tier=request.get("tier"))
    if explicit_length is not None and explicit_max is not None and explicit_length != explicit_max:
        raise ValueError(f"model_args.gen_length={explicit_length} differs from generation_args.max_gen_toks={explicit_max}")
    length = explicit_length if explicit_length is not None else explicit_max
    if length is None:
        length = request["model_args"].get("gen_length", request["generation_args"].get("max_gen_toks", 256))
    request["model_args"]["gen_length"] = length
    request["generation_args"]["max_gen_toks"] = length
    request["model_args"].setdefault("seed", 1234)
    for name in ("model_path", "spinquant_artifact_dir", "spinquant_execution", "seed"):
        if name not in request["model_args"]:
            raise ValueError(f"request model_args lacks {name}")
    return request


def evaluation_request(prompt, generation_args, sample, *, task_name, request_index=0,
                       question_char_start=None):
    """Pass input metadata to the evaluation entry."""
    if not isinstance(sample.get("doc", {}), dict):
        raise ValueError("capture sample must be an object")
    if question_char_start is not None and (
            type(question_char_start) is not int or not 0 <= question_char_start < len(prompt)):
        raise ValueError("question_char_start must index the supplied prompt")
    return SimpleNamespace(args=(prompt, generation_args), doc=sample.get("doc", {}),
        doc_id=sample.get("doc_id"), task_name=task_name, idx=request_index,
        question_char_start=question_char_start)


def capture_dataset(request_config, sample, result):
    """Read explicit provenance; document fields do not identify a dataset."""
    if request_config is not None:
        return request_config.get("dataset")
    if sample.get("dataset"):
        return sample["dataset"]
    configs = result.get("configs", {})
    task = sample.get("task_name")
    if task is not None:
        configs = {task: configs[task]} if task in configs else {}
    datasets = set()
    for config in configs.values():
        path = config.get("dataset_path")
        if not path:
            return None
        name = config.get("dataset_name")
        datasets.add(str(path) + ("/" + str(name) if name else ""))
    return next(iter(datasets)) if len(datasets) == 1 else None


def capture_sample_id(explicit, sample, result):
    """Use the supplied label or the sample's recorded task and document identity."""
    if explicit is not None:
        return explicit
    if sample.get("sample_id") is not None:
        return sample["sample_id"]
    doc_id = sample.get("doc_id")
    if doc_id is None:
        return None
    tasks = result.get("configs", {})
    task = sample.get("task_name")
    if task is None and len(tasks) == 1:
        task = next(iter(tasks))
    return f"{task}/{doc_id}" if task is not None else str(doc_id)




def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "replay"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--observer", type=Path, help="defaults to SOURCE/capture/layers.py")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--evaluation-result", type=Path)
    parser.add_argument("--evaluation-samples", type=Path)
    parser.add_argument("--request-config", type=Path,
                        help="JSON with model_args, prompt and generation_args")
    parser.add_argument("--sample-row", type=int, default=0,
                        help="zero-based sample JSONL row")
    parser.add_argument("--sample-id")
    parser.add_argument("--capture-index", type=int,
                        help="exact generator forward index, default 0")
    parser.add_argument("--selections", type=Path,
                        help="JSON named index/block-step-phase selections or start/end event-and-layer range")
    parser.add_argument("--head-only", action="store_true", help="capture full-vocabulary head prediction tokens at the selected forward")
    parser.add_argument("--last-layer-only", action="store_true",
                        help="capture actual L31 block input/cache and outputs for a single-layer RTL replay")
    parser.add_argument("--first-layer", type=int, help="first captured model layer, default 0")
    parser.add_argument("--layer-count", type=int, help="number of consecutive captured layers, 1..32")
    parser.add_argument("--model-path", type=Path, help="local path for the same saved model")
    parser.add_argument("--artifact-dir", type=Path, help="local path for the same saved quantized artifact")
    parser.add_argument("--silu-table", type=Path, help="local path for the same saved SiLU table")
    parser.add_argument("--control-state", action="store_true",
                        help="capture Feature12 state before execution/postprocess and before the next forward")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.selections and (args.command != "capture" or args.capture_index is not None
            or args.head_only or args.last_layer_only or args.control_state
            or args.first_layer is not None or args.layer_count is not None):
        parser.error("selections supplies all event/layer/head/control selectors")
    if args.observer is None:
        args.observer = args.source / "capture/layers.py"
    if not args.selections and args.capture_index is None:
        args.capture_index = 0
    if args.capture_index is not None and args.capture_index < 0:
        parser.error("capture-index must be nonnegative")
    if args.last_layer_only and args.head_only:
        parser.error("last-layer-only cannot be combined with head-only")
    if args.sample_row < 0:
        parser.error("sample-row must be nonnegative")
    sys.path.insert(0, str(args.source.resolve()))
    observer = load_observer(args.observer)
    if args.command == "replay":
        if json.loads(args.index.read_text()).get("reference_data_kind") == "real_cuda_final_output":
            replay_head(args.index, payload_root=args.payload_root)
        else:
            replay(args.index, observer, payload_root=args.payload_root)
        return
    if args.output is None:
        parser.error("capture requires output")
    if args.request_config and (args.evaluation_result or args.evaluation_samples):
        parser.error("request-config and evaluation logs are alternative request sources")
    if not args.request_config and (args.evaluation_result is None or args.evaluation_samples is None):
        parser.error("capture requires request-config or both evaluation-result and evaluation-samples")
    validate_paths(args.output, args.index)
    if args.request_config:
        request_config = load_request_config(args.request_config)
        model_args = request_config["model_args"]
        request = dict(arg_0=request_config["prompt"], arg_1=request_config["generation_args"])
        sample = dict(doc_id=request_config.get("sample_id", args.sample_id),
                      doc=request_config.get("sample", {}))
        result = {"n-shot": request_config.get("fewshot")}
        request_task = request_config.get("task_name", request_config.get("dataset", "user_request"))
        question_char_start = request_config.get("question_char_start")
    else:
        result = json.loads(args.evaluation_result.read_text())
        with args.evaluation_samples.open() as stream:
            sample = next((json.loads(line) for i, line in enumerate(stream) if i == args.sample_row), None)
        if sample is None:
            raise ValueError("sample-row absent from completed evaluation")
        request = sample["arguments"]["gen_args_0"]
        from lm_eval.utils import simple_parse_args_string
        model_args = simple_parse_args_string(result["config"]["model_args"])
        request_task = sample.get("task_name", "saved_evaluation")
        question_char_start = sample.get("question_char_start")
    actual_request = evaluation_request(request["arg_0"], request["arg_1"], sample,
        task_name=request_task, request_index=args.sample_row, question_char_start=question_char_start)
    for name, value in (("model_path", args.model_path), ("spinquant_artifact_dir", args.artifact_dir),
                        ("silu_table", args.silu_table)):
        if value is not None:
            model_args[name] = str(value.resolve(strict=True))
    model_args["trace_output_path"] = ""
    if model_args["spinquant_execution"] != "joint-full-w4a8-v8":
        raise ValueError("spinquant_execution must be joint-full-w4a8-v8")
    configure_capture_environment(model_args)
    from evaluation import model as evaluation
    if model_args.get("feature1_cross_block_block_initialization_protect_question", False):
        # Validate prompt metadata before model loading. The evaluator resolves
        # the actual token boundary with tokenizer offsets during inference.
        evaluation._question_start_token(actual_request, request["arg_0"], [(0, len(request["arg_0"]))])
    print(json.dumps(dict(stage="load_model", source=str(args.source), sample=sample["doc_id"])), flush=True)
    model = evaluation.QuantizedLLaDALM(**model_args)
    provenance = dict(dataset=capture_dataset(request_config if args.request_config else None, sample, result),
        split=request_config.get("split") if args.request_config else "test",
        sample_id=sample["doc_id"] if args.request_config else capture_sample_id(args.sample_id, sample, result),
        logged_doc_id=sample["doc_id"], sample_doc=sample["doc"], sample_doc_hash=sample.get("doc_hash"),
        fewshot=result["n-shot"],
        request_task_name=actual_request.task_name, question_char_start=actual_request.question_char_start,
        seed=model_args["seed"], model_arguments=model_args,
        backend="CUDA/Triton", torch_version=torch.__version__,
        generation_request=request["arg_1"])
    exported, shared = [], {}
    def export_selected(name, selected):
        for kind in (("layers", "head") if selected.include_head and not selected.head_only
                     else ("head",) if selected.head_only else ("layers",)):
            target = args.index.with_name(args.index.stem + "-" + name + "-" + kind + ".json")
            payload = args.output / name / kind
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = export_head if kind == "head" else export_capture
            writer(selected, provenance, payload.resolve(), target.resolve(), shared=shared)
            exported.append(dict(name=name, kind=kind, capture_index=selected.target,
                                 index=os.path.relpath(target, args.index.parent),
                                 forward={key: selected.event[key] for key in
                                          ("block_index", "step_index", "forward_kind")},
                                 **({"layer_range": [selected.first_layer, selected.first_layer+selected.layer_count-1]}
                                    if kind == "layers" else {})))
            print(json.dumps(dict(stage="export", selection=name, kind=kind,
                                  capture_index=selected.target)), flush=True)
    if args.selections:
        capture = ForwardCaptureSet(model.model, observer,
            json.loads(args.selections.read_text()), export_selected)
        # Reject collisions before entering model inference, including child indices.
        for item in capture.observations:
            for kind in (("head",) if item.head_only else
                         ("layers", "head") if item.include_head else ("layers",)):
                validate_paths(args.output / item.name / kind,
                    args.index.with_name(args.index.stem + "-" + item.name + "-" + kind + ".json"))
    else:
        capture = IndexedForwardCapture(model.model, observer, args.capture_index,
            head_only=args.head_only, last_layer_only=args.last_layer_only,
            control_state=args.control_state, first_layer=args.first_layer, layer_count=args.layer_count)
    print(json.dumps(dict(stage="generate_to_selected_forward")), flush=True)
    try:
        with generation_capture(evaluation, capture, observer), torch.inference_mode(), capture:
            model.generate_until([actual_request])
            capture.finish_request()
    except observer.CaptureComplete:
        pass
    if args.selections:
        if not capture.complete():
            raise RuntimeError("capture set ended without all selected forwards")
    else:
        if capture.event is None:
            raise RuntimeError("request ended without a forward matching the capture selectors")
        writer = export_head if args.head_only else export_capture
        writer(capture, provenance, args.output.resolve(), args.index.resolve())
    del model, capture
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if args.selections:
        for entry in exported:
            target = args.index.parent / entry["index"]
            if entry["kind"] == "head":
                replay_head(target)
            else:
                replay(target, observer)
        args.index.parent.mkdir(parents=True, exist_ok=True)
        args.index.write_text(json.dumps(dict(schema="supra-capture-set/v1", captures=exported), indent=2) + "\n")
    elif args.head_only:
        replay_head(args.index)
    else:
        replay(args.index, observer)


if __name__ == "__main__":
    main()
