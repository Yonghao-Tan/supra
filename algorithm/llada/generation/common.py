# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Cache validation and per-step token quotas."""

from __future__ import annotations
from typing import Any, Sequence, Tuple
import torch

PastKeyValues = Sequence[Tuple[torch.Tensor, ...]]



def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Split a block's remaining masked tokens as evenly as possible across steps."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if block_mask_index.ndim != 2 or block_mask_index.dtype != torch.bool:
        raise ValueError("block_mask_index must be a rank-2 bool tensor")
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    remainder = total - base * steps
    schedule = base.unsqueeze(1).expand(-1, steps).to(torch.long)
    step_columns = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    return schedule + (step_columns < remainder.unsqueeze(1)).to(torch.long)



def _cache_metadata(
    past_key_values: PastKeyValues,
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], str]:
    if not past_key_values:
        raise ValueError("model returned an empty past_key_values sequence")
    first_cache = past_key_values[0]
    if len(first_cache) == 2:
        (first_key, first_value) = first_cache
        cache_dtype = str(first_key.dtype)
    elif len(first_cache) == 3:
        (first_key, first_key_scale, first_value) = first_cache
        if first_key.dtype != torch.int8 or first_key_scale.dtype != torch.float32:
            raise ValueError(
                "three-field cache must contain INT8 K codes and BF16-materialized FP32 K scales"
            )
        if first_value.dtype == torch.int8:
            cache_dtype = "int8_kv_codes+bfloat16_k_scale+static_bfloat16_v_scale"
        elif first_value.dtype == torch.bfloat16:
            cache_dtype = "int8_k_codes+bfloat16_k_scale+bfloat16_v"
        else:
            raise ValueError(
                f"unsupported three-field cache V dtype {first_value.dtype}"
            )
    else:
        raise ValueError(
            f"cache entry must contain 2 or 3 tensors, got {len(first_cache)}"
        )
    if first_key.ndim < 3:
        raise ValueError(
            f"cache key must have a token axis, got shape {tuple(first_key.shape)}"
        )
    return (
        len(past_key_values),
        int(first_key.shape[-2]),
        tuple((int(dimension) for dimension in first_key.shape)),
        tuple((int(dimension) for dimension in first_value.shape)),
        cache_dtype,
    )


def _require_cached_output(output: Any) -> Tuple[torch.Tensor, PastKeyValues]:
    logits = getattr(output, "logits", None)
    past_key_values = getattr(output, "past_key_values", None)
    if not isinstance(logits, torch.Tensor):
        raise ValueError("model output must provide a tensor logits attribute")
    if past_key_values is None:
        raise ValueError(
            "model output must provide past_key_values when use_cache=True"
        )
    return (logits, past_key_values)
