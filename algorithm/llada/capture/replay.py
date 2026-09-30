"""Replay captured generation forwards through the quantized model."""

from __future__ import annotations
from typing import Any
import torch
from generation.common import PastKeyValues, _require_cached_output
from generation.records import ForwardCaptureEvent, ForwardReplayResult
from generation.engine import (
    _locate_unique_positions,
    _layer0_global_then_attention_guided_deep,
)
from numerics.precision import RowPrecisionContext
from generation.refresh import _unwrap_runtime_owner


class ForwardReplayer:
    """Rebuild a persistent student cache from an authoritative forward sequence."""

    def __init__(
        self,
        model: torch.nn.Module,
        row_precision_context: RowPrecisionContext | None,
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        if model is not None:
            for config in (model.config, _unwrap_runtime_owner(model).config):
                config.noncausal_cached_attention = True
        self.row_precision_context = row_precision_context
        self.device = device
        self.past_key_values: PastKeyValues | None = None
        self.next_capture_index = 0

    @staticmethod
    def _detach_cache(past_key_values: PastKeyValues) -> PastKeyValues:
        return tuple(
            (tuple((field.detach() for field in layer)) for layer in past_key_values)
        )

    def detach_cache(self) -> None:
        if self.past_key_values is not None:
            self.past_key_values = self._detach_cache(self.past_key_values)

    def step(self, event: ForwardCaptureEvent) -> ForwardReplayResult:
        if event.boundary_global_layers != 1:
            raise ValueError("replay supports one full global boundary layer")
        if event.capture_index != self.next_capture_index:
            raise ValueError(
                f"capture sequence is not contiguous: expected {self.next_capture_index}, got {event.capture_index}"
            )
        if not event.cache_initialized_before and self.past_key_values is not None:
            if event.forward_kind != "full_sequence" or (event.block_index <= 0 and not event.full_sequence_recompute):
                raise ValueError(
                    "only a later per-block full_sequence may reset the replay cache"
                )
            self.past_key_values = None
        cache_initialized = self.past_key_values is not None
        if cache_initialized != event.cache_initialized_before:
            raise ValueError("replayed cache initialization does not match the capture")
        input_ids = event.model_input_ids.to(self.device, dtype=torch.long)
        input_positions = event.input_positions.to(self.device, dtype=torch.long)
        refresh_positions = event.refresh_positions.to(self.device, dtype=torch.long)
        row_bits = event.row_bits.to(self.device, dtype=torch.int8)
        if self.row_precision_context is None:
            if row_bits.numel():
                raise ValueError("BF16 capture replay must not carry row bits")
        else:
            self.row_precision_context.activate(row_bits)
        arguments: dict[str, Any] = {"use_cache": True}
        if self.past_key_values is not None:
            arguments.update(
                {
                    "past_key_values": self.past_key_values,
                    "query_position_ids": input_positions.unsqueeze(0),
                    "kv_write_position_ids": refresh_positions.unsqueeze(0),
                }
            )
        if event.layer0_global_selected_deep:
            if (
                self.past_key_values is None
                or event.forward_kind != "boundary_refresh"
                or (not torch.equal(refresh_positions, input_positions))
            ):
                raise ValueError(
                    "Layer0-global replay requires an ordinary cached boundary event"
                )
            current_positions = event.prediction_positions.to(
                self.device, dtype=torch.long
            )
            current_local = torch.searchsorted(input_positions, current_positions)
            if (
                current_local.numel()
                and int(current_local.max().item()) >= int(input_positions.numel())
                or not torch.equal(
                    input_positions.index_select(0, current_local), current_positions
                )
            ):
                raise ValueError("Layer0-global replay input omits a current row")
            optional_positions = input_positions[
                ~torch.isin(input_positions, current_positions)
            ]
            captured_scout_bits = getattr(event, "layer0_current_row_bits", ())
            scout_bits = None
            if self.row_precision_context is not None:
                if captured_scout_bits:
                    if len(captured_scout_bits) != current_positions.numel() or any(
                        bit not in (4, 8) for bit in captured_scout_bits
                    ):
                        raise ValueError("captured Layer0 current bits do not align with current positions")
                    scout_bits = torch.tensor(captured_scout_bits, dtype=torch.int8, device=self.device)
                elif event.boundary_deep_bits:
                    if event.layer1_global_context_bits != 8 or event.layer1_global_prefix_bits != 8:
                        raise ValueError("a non-A8 scout with uniform deep bits requires captured Layer0 current bits")
                    scout_bits = torch.full_like(current_positions, 8, dtype=torch.int8)
                else:
                    scout_bits = row_bits.reshape(-1).index_select(0, current_local)
            (
                logits,
                self.past_key_values,
                replay_positions,
                _,
                _,
                _,
                _,
            ) = _layer0_global_then_attention_guided_deep(
                self.model,
                self.row_precision_context,
                tokens=event.tokens_before.to(self.device, dtype=torch.long),
                stale_cache=self.past_key_values,
                current_positions=current_positions,
                current_row_bits=scout_bits,
                target_rows=int(input_positions.numel()),
                deep_candidate_positions=optional_positions,
                global_context_bits=event.layer1_global_context_bits,
                global_prefix_bits=event.layer1_global_prefix_bits,
                replay_deep_row_bits=None
                if self.row_precision_context is None
                else row_bits.reshape(-1),
                keep_global_layer0_cache=event.layer0_keep_global_cache,
                deep_uniform_bits=event.boundary_deep_bits,
                deep_clip_ratio=event.boundary_deep_clip_ratio,
            )
            if not torch.equal(replay_positions, input_positions):
                raise RuntimeError("Layer0-global replay changed captured deep rows")
        else:
            if event.full_sequence_recompute:
                if event.cache_initialized_before or event.future_prediction_positions.numel():
                    raise ValueError("full-sequence replay requires a fresh cache and current predictions")
                selected_positions = event.prediction_positions.to(self.device, dtype=torch.long)[
                    event.prediction_mask.to(self.device, dtype=torch.bool)[0]]
                arguments["logits_positions"] = _locate_unique_positions(input_positions, selected_positions)
            output = self.model(input_ids, **arguments)
            (logits, self.past_key_values) = _require_cached_output(output)
        prediction_positions = event.prediction_positions.to(
            self.device, dtype=torch.long
        )
        prediction_mask = event.prediction_mask.to(self.device, dtype=torch.bool)
        if prediction_mask.ndim != 2 or prediction_mask.shape[0] != 1:
            raise ValueError("capture replay currently requires batch size one")
        active_positions = prediction_positions[prediction_mask[0]]
        if active_positions.numel():
            position_matches = input_positions[:, None] == active_positions[None, :]
            if not bool((position_matches.sum(dim=0) == 1).all()):
                raise ValueError(
                    "captured active prediction row is absent or duplicated"
                )
            packed_indices = position_matches.to(torch.int64).argmax(dim=0)
            prediction_logits = logits if event.full_sequence_recompute else logits.index_select(1, packed_indices)
        else:
            prediction_logits = logits[:, :0]
        future_positions = event.future_prediction_positions.to(
            self.device, dtype=torch.long
        )
        if (
            future_positions.ndim != 1
            or future_positions.unique().numel() != future_positions.numel()
        ):
            raise ValueError(
                "captured future prediction positions must be a unique vector"
            )
        if bool(torch.isin(future_positions, prediction_positions).any()):
            raise ValueError("future and current prediction positions must be disjoint")
        if future_positions.numel():
            future_indices = _locate_unique_positions(input_positions, future_positions)
            future_logits = logits.index_select(1, future_indices)
        else:
            future_logits = logits[:, :0]
        result = ForwardReplayResult(
            capture_index=event.capture_index,
            prediction_positions=active_positions,
            prediction_logits=prediction_logits,
            future_prediction_positions=future_positions,
            future_prediction_logits=future_logits,
        )
        self.next_capture_index += 1
        return result
