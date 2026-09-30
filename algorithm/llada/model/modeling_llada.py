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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from __future__ import annotations
import logging
import math
import sys
from abc import abstractmethod
from functools import partial
from typing import (
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    cast,
)
from dataclasses import MISSING, fields
from typing import Union
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")
__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
]
log = logging.getLogger(__name__)


@torch.compile()
def scaled_dot_product_attention(
    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


def _explicit_write_local_mask(
    query_position_ids: torch.Tensor, write_position_ids: torch.Tensor
) -> torch.Tensor:
    """Map arbitrary global K/V write positions onto packed query rows."""
    write_local = torch.isin(query_position_ids, write_position_ids)
    if int(write_local.sum().item()) != int(write_position_ids.numel()):
        raise ValueError("explicit K/V writes must be a subset of packed queries")
    return write_local


def _packed_subset_indices(
    positions: torch.Tensor, packed_positions: torch.Tensor, *, name: str
) -> torch.Tensor:
    """Map sorted absolute positions onto one sorted packed query vector."""
    if positions.ndim != 1 or positions.dtype != torch.long:
        raise RuntimeError(f"{name} positions must be a rank-1 int64 tensor")
    if packed_positions.ndim != 1 or packed_positions.dtype != torch.long:
        raise RuntimeError("packed query positions must be a rank-1 int64 tensor")
    if packed_positions.numel() > 1 and bool(
        (packed_positions[1:] <= packed_positions[:-1]).any()
    ):
        raise RuntimeError("packed query positions must be sorted and unique")
    indices = torch.searchsorted(packed_positions, positions)
    if indices.numel() and (
        bool((indices >= packed_positions.numel()).any())
        or not torch.equal(packed_positions.index_select(0, indices), positions)
    ):
        raise RuntimeError(f"{name} positions are not packed query rows")
    return indices


def _current_to_prefix_attention_risk_q8(
    *,
    probabilities: torch.Tensor,
    query_positions: torch.Tensor,
    current_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    total_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Take max current32-to-prefix P8 attention after head reduction."""
    if probabilities.ndim != 4 or probabilities.shape[0] != 1:
        raise RuntimeError("layerwise attention probabilities must be [1,H,Q,K]")
    if query_positions.numel() != probabilities.shape[2]:
        raise RuntimeError("layerwise query positions do not match probabilities")
    current_local = _packed_subset_indices(
        current_positions, query_positions, name="layerwise current"
    )
    key_length = int(probabilities.shape[3])
    if bool(
        ((current_positions < 0) | (current_positions >= key_length)).any()
    ) or bool(((candidate_positions < 0) | (candidate_positions >= key_length)).any()):
        raise RuntimeError("layerwise attention positions are outside K/V")
    risk_q8 = torch.zeros(total_length, dtype=torch.uint8, device=probabilities.device)
    valid = torch.zeros_like(risk_q8, dtype=torch.bool)
    if candidate_positions.numel():
        current_to_candidate = (
            probabilities.index_select(2, current_local)
            .index_select(3, candidate_positions)
            .mean(dim=1)
            .amax(dim=1)[0]
        )
        candidate_risk_q8 = torch.clamp(
            torch.round(current_to_candidate * 255.0), 0, 255
        ).to(torch.uint8)
        risk_q8.index_copy_(0, candidate_positions, candidate_risk_q8)
        valid.index_fill_(0, candidate_positions, True)
    probability_entries = (
        int(current_positions.numel())
        * int(candidate_positions.numel())
        * int(probabilities.shape[1])
    )
    return (risk_q8, valid, probability_entries)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value
            )
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(
                f"When using the {InitFnType.full_megatron} init, every module must have a type."
            )
        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3
        if type_of_module == ModuleType.in_module:
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            std = config.d_model ** (-0.5)
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if config.init_fn == InitFnType.normal and getattr(
            module, "_is_residual", False
        ):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(
    x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False
):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == -1e309, torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == 1e309, torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        cfg.attention_dropout == 0.0
        and cfg.embedding_dropout == 0.0
        and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint, preserve_rng_state=preserve_rng_state, use_reentrant=False
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (
            elementwise_affine is None and self.config.layer_norm_with_affine
        ):
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, device=config.init_device)
            )
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, device=config.init_device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(
        cls, config: ModelConfig, size: Optional[int] = None, **kwargs
    ) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(
                f"Unknown LayerNorm type: '{config.layer_norm_type}'"
            )

    def _cast_if_autocast_enabled(
        self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype()
            )
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(
                dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype()
            )
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(
            config, size=size, elementwise_affine=elementwise_affine, eps=eps
        )
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight)
                if self.weight is not None
                else self.weight
            )
            downcast_bias = (
                self._cast_if_autocast_enabled(self.bias)
                if self.bias is not None
                else self.bias
            )
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x,
                    self.normalized_shape,
                    weight=downcast_weight,
                    bias=downcast_bias,
                    eps=self.eps,
                )
        else:
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(
            config,
            size=size,
            elementwise_affine=elementwise_affine,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)
        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(
            config,
            size=size,
            elementwise_affine=elementwise_affine,
            eps=config.rms_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)
        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(
            config.max_sequence_length, _non_meta_init_device(config)
        )

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and (pos_sin.shape[-2] >= seq_len)
            and (pos_cos.shape[-2] >= seq_len)
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return (pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :])
        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / self.rope_theta ** (
                torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            (pos_sin, pos_cos) = (
                positions.sin()[None, None, :, :],
                positions.cos()[None, None, :, :],
            )
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return (pos_sin, pos_cos)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        (B, nh, T, hs) = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        (x1, x2) = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return (t * pos_cos + self.rotate_half(t) * pos_sin).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_end_index: Optional[torch.Tensor] = None,
        query_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            (q_, k_) = (q.float(), k.float())
        else:
            (q_, k_) = (q, k)
        with torch.autocast(q.device.type, enabled=False):
            (query_len, key_len) = (q_.shape[-2], k_.shape[-2])
            (pos_sin, pos_cos) = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if query_position_ids is not None:
                if (
                    query_position_ids.ndim != 1
                    or query_position_ids.numel() != query_len
                ):
                    raise ValueError(
                        "query_position_ids must contain one position per query row"
                    )
                idx = query_position_ids.to(device=q_.device, dtype=torch.long)
                if bool(((idx < 0) | (idx >= key_len)).any()):
                    raise ValueError("query_position_ids are outside the K/V sequence")
            elif block_end_index is None:
                start = key_len - query_len
                end = key_len
                idx = torch.arange(start, end, device=q_.device, dtype=torch.long)
            else:
                start = block_end_index - query_len
                end = block_end_index
                idx = torch.arange(start, end, device=q_.device, dtype=torch.long)
            pos_sin_slice = pos_sin.index_select(2, idx)
            pos_cos_slice = pos_cos.index_select(2, idx)
            q_ = self.apply_rotary_pos_emb(pos_sin_slice, pos_cos_slice, q_)
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return (q_.type_as(q), k_.type_as(k))


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x, gate) = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float), diagonal=1
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)


def get_causal_attention_bias(
    cache: BufferCache, seq_len: int, device: torch.device
) -> torch.Tensor:
    if (
        causal_bias := cache.get("causal_attention_bias")
    ) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(
    seq_len: int, config: ModelConfig, device: torch.device
) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(
        1, 1, 1, seq_len
    )
    alibi_bias = alibi_bias - torch.arange(
        1 - seq_len, 1, dtype=torch.float, device=device
    ).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)
    return alibi_bias * (1.0 / 2 ** m.view(1, config.n_heads, 1, 1))


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size
            if config.mlp_hidden_size is not None
            else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0
        self._activation_checkpoint_fn = None
        self.dropout = Dropout(config.residual_dropout)
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=config.d_model // config.n_heads * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(
                config, elementwise_affine=config.attention_layer_norm_with_affine
            )
        self.act = Activation.build(config)
        assert self.act.output_multiplier * self.hidden_size % 1 == 0
        self.attn_out = nn.Linear(
            config.d_model,
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)
        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(
        cls, bias: torch.Tensor, input_dtype: torch.dtype
    ) -> torch.Tensor:
        target_dtype = input_dtype
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=dropout_p,
                causal=False,
            )
            return r.transpose(1, 2)
        else:
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(
                    num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
                )
                v = v.repeat_interleave(
                    num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads
                )
            return scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        replace_position_kv: Optional[torch.Tensor] = None,
        local_idx: Optional[torch.Tensor] = None,
        query_position_ids: Optional[torch.Tensor] = None,
        kv_write_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        if kv_write_position_ids is None and replace_position_kv is None:
            replace_position_kv = replace_position
        (B, T, C) = q.size()
        dtype = k.dtype
        if query_position_ids is not None:
            if (
                query_position_ids.shape != (B, T)
                or query_position_ids.dtype != torch.long
            ):
                raise ValueError(
                    "query_position_ids must be int64 with shape [batch, query_rows]"
                )
        if kv_write_position_ids is not None:
            if query_position_ids is None:
                raise ValueError("explicit K/V writes require query_position_ids")
            if kv_write_position_ids.ndim != 2 or kv_write_position_ids.shape[0] != B:
                raise ValueError(
                    "kv_write_position_ids must have shape [batch, write_rows]"
                )
            if kv_write_position_ids.dtype != torch.long:
                raise ValueError("kv_write_position_ids must be int64")
        replacement_requested = (
            replace_position_kv is not None or kv_write_position_ids is not None
        )
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)
        v = v.view(
            B, T, self.config.effective_n_kv_heads, C // self.config.n_heads
        ).transpose(1, 2)
        target_rope_before_k8 = bool(
            getattr(self, "_target_numeric_rope_before_k8", False)
        )
        if target_rope_before_k8:
            if not self.config.rope or (
                B != 1
                and (
                    use_cache
                    or layer_past is not None
                    or query_position_ids is not None
                    or replacement_requested
                    or (local_idx is not None)
                )
            ):
                raise ValueError(
                    "batched target RoPE requires dense, uncached forwards with shared implicit positions"
                )
            past_length = 0 if layer_past is None else int(layer_past[0].shape[-2])
            if query_position_ids is not None:
                local_positions = query_position_ids[0]
            elif layer_past is None or not replacement_requested:
                local_positions = torch.arange(
                    past_length, past_length + T, device=q.device, dtype=torch.long
                )
            elif local_idx is not None:
                local_positions = local_idx.to(device=q.device, dtype=torch.long)
            elif replace_position is not None:
                local_positions = torch.nonzero(
                    replace_position[0], as_tuple=False
                ).flatten()
            else:
                raise ValueError(
                    "target cache replacement requires explicit query positions"
                )
            if local_positions.numel() != T:
                raise ValueError(
                    "target RoPE positions must match the local query row count"
                )
            sequence_length = int(local_positions.max().item()) + 1
            (position_sin, position_cos) = self.rotary_emb.get_rotary_embedding(
                sequence_length, q.device
            )
            position_sin = position_sin.to(torch.bfloat16).index_select(
                2, local_positions
            )
            position_cos = position_cos.to(torch.bfloat16).index_select(
                2, local_positions
            )
            from numerics.operator_kernels import (
                rope_apply_bf16_cuda,
                triton_operator_numeric_available,
            )
            from numerics.bf16 import rope_apply_bf16

            apply_rope = (
                rope_apply_bf16_cuda
                if q.is_cuda and triton_operator_numeric_available()
                else rope_apply_bf16
            )
            (q_input, k_input) = (q, k)
            q_exact = apply_rope(q_input, position_sin, position_cos).to(dtype)
            k_exact = apply_rope(k_input, position_sin, position_cos).to(dtype)
            (q, k) = (q_exact, k_exact)
        k_cache_codec = getattr(self, "_spinquant_k_cache_codec", None)
        v_cache_codec = getattr(self, "_spinquant_v_cache_codec", None)
        v_storage = v_cache_codec.encode(v) if v_cache_codec is not None else v
        if layer_past is not None:
            if k_cache_codec is None:
                if len(layer_past) != 2:
                    raise ValueError("native BF16 cache requires (K, V)")
                (past_key, past_value) = layer_past
                past_key_scale = None
            else:
                if len(layer_past) != 3:
                    raise ValueError(
                        "SpinQuant K8 cache requires (K codes, K scales, V)"
                    )
                (past_key, past_key_scale, past_value) = layer_past
                if past_key.dtype != torch.int8:
                    raise ValueError("SpinQuant K cache codes must be INT8")
                if v_cache_codec is not None and past_value.dtype != torch.int8:
                    raise ValueError("SpinQuant V cache codes must be INT8")
                (k, k_scale) = k_cache_codec.encode(k)
            if not replacement_requested:
                if k_cache_codec is None:
                    past_key = torch.cat((past_key, k), dim=-2)
                else:
                    past_key = torch.cat((past_key, k), dim=-2)
                    past_key_scale = torch.cat((past_key_scale, k_scale), dim=-2)
                v_storage = torch.cat((past_value, v_storage), dim=-2)
            else:
                B = q.shape[0]
                for batch_idx in range(B):
                    batch_replace_indices = (
                        kv_write_position_ids[batch_idx]
                        if kv_write_position_ids is not None
                        else replace_position_kv[batch_idx].nonzero(as_tuple=True)[0]
                    )
                    if len(batch_replace_indices) > 0:
                        if query_position_ids is not None:
                            global_idx = batch_replace_indices
                            write_local = _explicit_write_local_mask(
                                query_position_ids[batch_idx], global_idx
                            )
                            past_key[batch_idx, :, global_idx] = k[
                                batch_idx, :, write_local
                            ]
                            if past_key_scale is not None:
                                past_key_scale[batch_idx, :, global_idx] = k_scale[
                                    batch_idx, :, write_local
                                ]
                            past_value[batch_idx, :, global_idx] = v_storage[
                                batch_idx, :, write_local
                            ]
                        else:
                            past_key[batch_idx, :, batch_replace_indices] = k[
                                batch_idx, :, : len(batch_replace_indices)
                            ]
                            if past_key_scale is not None:
                                past_key_scale[batch_idx, :, batch_replace_indices] = (
                                    k_scale[batch_idx, :, : len(batch_replace_indices)]
                                )
                            past_value[batch_idx, :, batch_replace_indices] = v_storage[
                                batch_idx, :, : len(batch_replace_indices)
                            ]
            k = (
                past_key
                if k_cache_codec is None
                else k_cache_codec.decode(past_key, past_key_scale)
            )
            v = v_storage if not replacement_requested else past_value
            present = (
                (past_key, past_key_scale, v)
                if use_cache and k_cache_codec is not None
                else (k, v)
                if use_cache
                else None
            )
        elif k_cache_codec is not None:
            (k_codes, k_scale) = k_cache_codec.encode(k)
            k = k_cache_codec.decode(k_codes, k_scale)
            v = v_storage
            present = (k_codes, k_scale, v_storage) if use_cache else None
        else:
            v = v_storage
            present = (k, v_storage) if use_cache else None
        (query_len, key_len) = (q.shape[-2], k.shape[-2])
        if self.config.rope and (not target_rope_before_k8):
            if query_position_ids is not None:
                if B != 1:
                    raise ValueError(
                        "packed arbitrary-position RoPE currently requires batch size one"
                    )
                (q, k) = self.rotary_emb(q, k, query_position_ids=query_position_ids[0])
            elif replace_position is None:
                (q, k) = self.rotary_emb(q, k)
            else:
                max_replace_pos = (
                    replace_position.nonzero(as_tuple=True)[1].max() + 1
                    if replace_position.any()
                    else key_len
                )
                (q, k) = self.rotary_emb(q, k, max_replace_pos)
        if attention_bias is not None:
            if query_position_ids is not None:
                attention_bias = attention_bias.index_select(2, query_position_ids[0])
                attention_bias = self._cast_attn_bias(
                    attention_bias[:, :, :, :key_len], dtype
                )
            else:
                attention_bias = self._cast_attn_bias(
                    attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
                )
        score_override = getattr(self, "_spinquant_score_override", None)
        probability_override = getattr(self, "_spinquant_probability_override", None)
        context_override = getattr(self, "_spinquant_context_override", None)
        use_manual = (
            getattr(self.config, "use_manual_attention", False)
            or score_override is not None
            or probability_override is not None
            or (context_override is not None)
        )
        monitor_range_active = self.__cache.get("attn_monitor_query_range") is not None
        monitor_all_layers = bool(
            getattr(self.config, "attn_monitor_all_layers", False)
        )
        monitor_publish_layer = int(getattr(self.config, "attn_monitor_layer", -1))
        monitor_enabled = monitor_range_active and (
            monitor_all_layers
            or monitor_publish_layer == int(getattr(self, "layer_id", -2))
        )
        boundary_layer0_scout_active = bool(
            isinstance(self.__cache.get("boundary_layer0_scout"), dict)
            and int(self.layer_id) == 0
        )
        use_manual_eff = use_manual
        compute_probs_eff = (
            use_manual_eff or monitor_enabled or boundary_layer0_scout_active
        )
        probs_fp32 = None
        (k_for, v_for) = (k, v)
        if compute_probs_eff:
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                rep = num_q_heads // num_kv_heads
                k_for = k.repeat_interleave(rep, dim=1, output_size=num_q_heads)
                v_for = v.repeat_interleave(rep, dim=1, output_size=num_q_heads)
            if bool(self.__cache.get("attn_monitor_capture_qk_probe", False)) or bool(
                self.__cache.get("attn_monitor_capture_deployment_p8_relation", False)
            ):
                if layer_past is None:
                    if q.shape[-2] != k_for.shape[-2]:
                        raise RuntimeError(
                            "dense Q8/K8 probe requires equal query and key lengths"
                        )
                    probe_query_positions = torch.arange(
                        q.shape[-2], device=q.device, dtype=torch.long
                    ).unsqueeze(0)
                else:
                    if query_position_ids is None or query_position_ids.shape != (B, T):
                        raise RuntimeError(
                            "cached Q8/K8 probe requires explicit query positions"
                        )
                    probe_query_positions = query_position_ids
                if kv_write_position_ids is not None:
                    probe_write_positions = kv_write_position_ids
                elif replace_position_kv is not None:
                    write_rows = [
                        replace_position_kv[batch_index]
                        .nonzero(as_tuple=False)
                        .flatten()
                        for batch_index in range(B)
                    ]
                    if any(
                        (row.numel() != write_rows[0].numel() for row in write_rows[1:])
                    ):
                        raise RuntimeError(
                            "Q8/K8 probe requires equal K/V write counts per batch"
                        )
                    probe_write_positions = torch.stack(write_rows)
                else:
                    probe_write_positions = torch.empty(
                        (B, 0), device=q.device, dtype=torch.long
                    )
                self.__cache["attn_monitor_numeric_position_context"] = {
                    "staged_group": -1,
                    "query_positions": probe_query_positions.detach().clone(),
                    "key_positions": torch.arange(
                        k_for.shape[-2], device=q.device, dtype=torch.long
                    )
                    .unsqueeze(0)
                    .expand(B, -1)
                    .clone(),
                    "kv_write_positions": probe_write_positions.detach().clone(),
                }
            if (
                score_override is not None
                and target_rope_before_k8
                and (k_cache_codec is not None)
            ):
                (score_key_codes, score_key_scales) = (
                    (past_key, past_key_scale)
                    if layer_past is not None
                    else (k_codes, k_scale)
                )
                if num_q_heads != num_kv_heads:
                    score_key_codes = score_key_codes.repeat_interleave(rep, dim=1)
                    score_key_scales = score_key_scales.repeat_interleave(rep, dim=1)
                scores = score_override(
                    q, k_for, key_codes=score_key_codes, key_scales=score_key_scales
                )
            else:
                scores = (
                    score_override(q, k_for)
                    if score_override is not None
                    else torch.matmul(q, k_for.transpose(-2, -1))
                    / math.sqrt(q.shape[-1])
                )
            if attention_bias is not None:
                scores = scores + attention_bias
            probs_fp32 = (
                probability_override(scores)
                if probability_override is not None
                else torch.softmax(scores.to(torch.float32), dim=-1)
            )
            boundary_scout = self.__cache.get("boundary_layer0_scout")
            if isinstance(boundary_scout, dict) and int(self.layer_id) == 0:
                if query_position_ids is None or B != 1:
                    raise RuntimeError(
                        "boundary Layer0 scout requires one batch and explicit query positions"
                    )
                current_positions = boundary_scout.get("current_positions")
                total_length = int(boundary_scout.get("total_length", 0))
                if (
                    not isinstance(current_positions, torch.Tensor)
                    or current_positions.ndim != 1
                    or current_positions.dtype != torch.long
                    or (current_positions.numel() == 0)
                    or (total_length != int(probs_fp32.shape[3]))
                ):
                    raise RuntimeError("boundary Layer0 scout state is invalid")
                candidates = torch.arange(
                    total_length, dtype=torch.long, device=probs_fp32.device
                )
                candidates = candidates[~torch.isin(candidates, current_positions)]
                transition_positions = boundary_scout.get("transition_positions")
                if isinstance(transition_positions, torch.Tensor):
                    if (
                        transition_positions.ndim != 1
                        or transition_positions.dtype != torch.long
                        or transition_positions.device != probs_fp32.device
                        or (
                            transition_positions.unique().numel()
                            != transition_positions.numel()
                        )
                        or (transition_positions.numel() == 0)
                        or bool(
                            (
                                (transition_positions < 0)
                                | (transition_positions >= total_length)
                            ).any()
                        )
                    ):
                        raise RuntimeError(
                            "boundary Layer0 transition positions are invalid"
                        )
                    candidate_local = _packed_subset_indices(
                        candidates,
                        query_position_ids[0],
                        name="boundary Layer0 candidate queries",
                    )
                    candidate_to_transition = (
                        probs_fp32.detach()
                        .index_select(2, candidate_local)
                        .index_select(3, transition_positions)
                        .mean(dim=1)
                        .amax(dim=2)[0]
                    )
                    risk_q8 = torch.zeros(
                        total_length, dtype=torch.uint8, device=probs_fp32.device
                    )
                    risk_q8.index_copy_(
                        0,
                        candidates,
                        torch.clamp(
                            torch.round(candidate_to_transition * 255.0), 0, 255
                        ).to(torch.uint8),
                    )
                    probability_entries = int(
                        candidates.numel()
                        * transition_positions.numel()
                        * probs_fp32.shape[1]
                    )
                else:
                    (risk_q8, valid, probability_entries) = (
                        _current_to_prefix_attention_risk_q8(
                            probabilities=probs_fp32.detach(),
                            query_positions=query_position_ids[0],
                            current_positions=current_positions,
                            candidate_positions=candidates,
                            total_length=total_length,
                        )
                    )
                    if candidates.numel() and (
                        not bool(valid.index_select(0, candidates).all())
                    ):
                        raise RuntimeError("boundary Layer0 scout score is incomplete")
                boundary_scout["score_q8"] = risk_q8
                boundary_scout["probability_entries"] = probability_entries
        if use_manual_eff:
            if probs_fp32 is None:
                raise RuntimeError(
                    "use_manual_attention=True but failed to compute attention probabilities."
                )
            probs = probs_fp32.to(q.dtype)
            if self.training and self.config.attention_dropout > 0:
                probs = F.dropout(probs, p=self.config.attention_dropout, training=True)
            att = (
                context_override(probs, v_for)
                if context_override is not None
                else torch.matmul(probs, v_for)
            )
        else:
            att = self._scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_bias,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=False,
            )
        if monitor_enabled and probs_fp32 is not None:
            monitor = probs_fp32.detach()
            query_range = self.__cache.get("attn_monitor_query_range")
            if not isinstance(query_range, torch.Tensor) or query_range.numel() != 2:
                raise RuntimeError(
                    "attn_monitor_query_range must be Tensor([start,end])"
                )
            start = int(query_range[0].item())
            end = int(query_range[1].item())
            if not 0 <= start < end <= monitor.shape[-1]:
                raise RuntimeError(
                    "attention dependency region is outside the K/V sequence"
                )
            if layer_past is None:
                global_queries = torch.arange(
                    monitor.shape[2], device=monitor.device, dtype=torch.long
                )
            else:
                if query_position_ids is None or query_position_ids.shape[0] != 1:
                    raise RuntimeError(
                        "cached attention monitoring requires packed query positions"
                    )
                global_queries = query_position_ids[0]
            excluded_queries = self.__cache.get("attn_monitor_excluded_query_positions")
            if excluded_queries is None:
                excluded_query_mask = torch.zeros_like(global_queries, dtype=torch.bool)
            else:
                if (
                    not isinstance(excluded_queries, torch.Tensor)
                    or excluded_queries.ndim != 1
                    or excluded_queries.dtype != torch.long
                    or (excluded_queries.device != global_queries.device)
                ):
                    raise RuntimeError(
                        "attention monitor excluded query positions are invalid"
                    )
                excluded_query_mask = torch.isin(global_queries, excluded_queries)
            dependency_local = torch.nonzero(
                (global_queries >= start)
                & (global_queries < end)
                & ~excluded_query_mask,
                as_tuple=False,
            ).flatten()
            if dependency_local.numel() == 0:
                raise RuntimeError(
                    "attention monitor found no query rows in the dependency region"
                )
            query_positions = global_queries.index_select(0, dependency_local)
            dependency = (
                monitor.index_select(2, dependency_local)[:, :, :, start:end]
                .mean(dim=1)
                .to(torch.float32)
            )
            segment_head_mass = None
            if bool(
                self.__cache.get("attn_monitor_capture_segment_head_mass", False)
            ) and int(self.layer_id) in {0, int(self.config.n_layers) - 1}:
                segment_groups = self.__cache.get("attn_monitor_segment_key_groups")
                if not isinstance(segment_groups, tuple) or not segment_groups:
                    raise RuntimeError(
                        "segment head-mass capture requires configured key groups"
                    )
                selected_queries = monitor.index_select(2, dependency_local)
                masses = []
                for group in segment_groups:
                    if (
                        not isinstance(group, torch.Tensor)
                        or group.dtype != torch.long
                        or group.ndim != 1
                        or (group.device != monitor.device)
                        or (
                            not bool(((group >= 0) & (group < monitor.shape[-1])).all())
                        )
                    ):
                        raise RuntimeError(
                            "segment head-mass key group is outside the attention key range"
                        )
                    masses.append(
                        selected_queries.index_select(3, group)
                        .sum(dim=-1)
                        .to(torch.bfloat16)
                    )
                segment_head_mass = torch.stack(masses, dim=-1)
            numeric_probe = None
            if (
                bool(self.__cache.get("attn_monitor_capture_qk_probe", False))
                or bool(
                    self.__cache.get(
                        "attn_monitor_capture_deployment_p8_relation", False
                    )
                )
            ) and int(self.layer_id) == monitor_publish_layer:
                groups = self.__cache.get("attn_monitor_numeric_probe_groups")
                group_key = -1
                if not isinstance(groups, dict) or not isinstance(
                    groups.get(group_key), dict
                ):
                    raise RuntimeError(
                        "target Q8/K8/P8 path did not publish a numeric probe"
                    )
                live_numeric_probe = groups[group_key]
                expected_probe_layers = (
                    int(self.config.n_layers)
                    if monitor_all_layers
                    or monitor_publish_layer == int(self.config.n_layers) - 1
                    else monitor_publish_layer + 1
                )
                if int(live_numeric_probe.get("p8_relation_layer_count", 0)) != int(
                    expected_probe_layers
                ):
                    raise RuntimeError(
                        "deployment P8 probe did not reach the selected transformer layer"
                    )
                if int(live_numeric_probe.get("selected_layer_id", -1)) != int(
                    monitor_publish_layer
                ):
                    raise RuntimeError(
                        "deployment P8 probe selected-layer identity is invalid"
                    )
                numeric_probe = {
                    key: value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for (key, value) in live_numeric_probe.items()
                }
                probe_queries = numeric_probe.get("query_positions")
                probe_keys = numeric_probe.get("key_positions")
                if (
                    not isinstance(probe_queries, torch.Tensor)
                    or not isinstance(probe_keys, torch.Tensor)
                    or probe_queries.shape != (1, global_queries.numel())
                    or (not torch.equal(probe_queries[0], global_queries))
                    or (probe_keys.shape != (1, monitor.shape[-1]))
                ):
                    raise RuntimeError(
                        "target Q8/K8/P8 probe position identity is invalid"
                    )
            prompt_importance = None
            prompt_change_dependency = None
            prefix_dependency = None
            prefix_dependency_positions = None
            prefix_reverse_dependency = None
            prefix_future_dependency = None
            future_prefix_dependency = None
            future_prefix_dependency_positions = None
            future_prefix_key_end = None
            prompt_end = self.__cache.get("attn_monitor_prompt_end")
            if prompt_end is not None:
                prompt_end = int(prompt_end)
                prediction_range = self.__cache.get("attn_monitor_prediction_range")
                if (
                    not isinstance(prediction_range, torch.Tensor)
                    or prediction_range.numel() != 2
                ):
                    raise RuntimeError(
                        "attn_monitor_prediction_range must be Tensor([start,end])"
                    )
                prediction_start = int(prediction_range[0].item())
                prediction_end = int(prediction_range[1].item())
                if not 0 < prompt_end <= monitor.shape[-1]:
                    raise RuntimeError(
                        "attention prompt range is outside the K/V sequence"
                    )
                prediction_local = torch.nonzero(
                    (global_queries >= prediction_start)
                    & (global_queries < prediction_end),
                    as_tuple=False,
                ).flatten()
                if prediction_local.numel():
                    prompt_importance = (
                        monitor.index_select(2, prediction_local)[:, :, :, :prompt_end]
                        .mean(dim=1)
                        .amax(dim=1)
                        .to(torch.float16)
                    )
                prompt_query_local = torch.nonzero(
                    global_queries < prompt_end, as_tuple=False
                ).flatten()
                if prompt_query_local.numel() == prompt_end:
                    prompt_change_dependency = (
                        monitor.index_select(2, prompt_query_local)[
                            :, :, :, prediction_start:prediction_end
                        ]
                        .mean(dim=1)
                        .amax(dim=-1)
                        .to(torch.float16)
                    )
            if bool(self.__cache.get("attn_monitor_track_prefix_dependency", False)):
                prediction_range = self.__cache.get("attn_monitor_prediction_range")
                if (
                    not isinstance(prediction_range, torch.Tensor)
                    or prediction_range.numel() != 2
                ):
                    raise RuntimeError("prefix dependency requires a prediction range")
                prediction_start = int(prediction_range[0].item())
                prediction_end = int(prediction_range[1].item())
                query_end = (
                    monitor.shape[-1]
                    if bool(
                        self.__cache.get("attn_monitor_track_future_query_rows", False)
                    )
                    else prediction_end
                )
                prefix_local = torch.nonzero(
                    (global_queries < query_end) & ~excluded_query_mask, as_tuple=False
                ).flatten()
                prefix_dependency_positions = global_queries.index_select(
                    0, prefix_local
                )
                prefix_dependency = (
                    monitor.index_select(2, prefix_local)[
                        :, :, :, prediction_start:prediction_end
                    ]
                    .mean(dim=1)
                    .to(torch.float32)
                )
            if bool(
                self.__cache.get("attn_monitor_track_prefix_reverse_dependency", False)
            ):
                prediction_range = self.__cache.get("attn_monitor_prediction_range")
                if (
                    not isinstance(prediction_range, torch.Tensor)
                    or prediction_range.numel() != 2
                ):
                    raise RuntimeError(
                        "prefix reverse dependency requires a prediction range"
                    )
                prediction_start = int(prediction_range[0].item())
                prediction_end = int(prediction_range[1].item())
                current_local = torch.nonzero(
                    (global_queries >= prediction_start)
                    & (global_queries < prediction_end),
                    as_tuple=False,
                ).flatten()
                prefix_reverse_dependency = torch.zeros(
                    (
                        monitor.shape[0],
                        prediction_end,
                        prediction_end - prediction_start,
                    ),
                    dtype=torch.float32,
                    device=monitor.device,
                )
                if current_local.numel():
                    current_positions = global_queries.index_select(0, current_local)
                    current_offsets = current_positions - prediction_start
                    observed_reverse = (
                        monitor.index_select(2, current_local)[:, :, :, :prediction_end]
                        .mean(dim=1)
                        .transpose(1, 2)
                        .to(torch.float32)
                    )
                    prefix_reverse_dependency.index_copy_(
                        2, current_offsets, observed_reverse
                    )
            if bool(
                self.__cache.get("attn_monitor_track_prefix_future_dependency", False)
            ):
                if prompt_end is None:
                    raise RuntimeError(
                        "future prefix dependency requires the prompt boundary"
                    )
                prediction_range = self.__cache.get("attn_monitor_prediction_range")
                if (
                    not isinstance(prediction_range, torch.Tensor)
                    or prediction_range.numel() != 2
                ):
                    raise RuntimeError(
                        "future prefix dependency requires a prediction range"
                    )
                prediction_end = int(prediction_range[1].item())
                query_end = (
                    monitor.shape[-1]
                    if bool(
                        self.__cache.get("attn_monitor_track_future_query_rows", False)
                    )
                    else prediction_end
                )
                prefix_local = torch.nonzero(
                    (global_queries < query_end) & ~excluded_query_mask, as_tuple=False
                ).flatten()
                prefix_future_dependency = (
                    monitor.index_select(2, prefix_local)[:, :, :, prompt_end:]
                    .mean(dim=1)
                    .to(torch.float32)
                )
                if prefix_dependency_positions is None:
                    prefix_dependency_positions = global_queries.index_select(
                        0, prefix_local
                    )
            if bool(
                self.__cache.get(
                    "attn_monitor_trace_attention_relation_diagnostics", False
                )
            ):
                prediction_range = self.__cache.get("attn_monitor_prediction_range")
                if (
                    not isinstance(prediction_range, torch.Tensor)
                    or prediction_range.numel() != 2
                ):
                    raise RuntimeError(
                        "attention relation diagnostics require a prediction range"
                    )
                prediction_end = int(prediction_range[1].item())
                future_local = torch.nonzero(
                    (global_queries >= prediction_end) & ~excluded_query_mask,
                    as_tuple=False,
                ).flatten()
                future_prefix_dependency_positions = global_queries.index_select(
                    0, future_local
                )
                future_prefix_dependency = (
                    monitor.index_select(2, future_local)[:, :, :, :prediction_end]
                    .mean(dim=1)
                    .to(torch.float32)
                )
                future_prefix_key_end = prediction_end
            if monitor_all_layers:
                monitor_state = self.__cache
                prior_positions = monitor_state.get(
                    "attn_monitor_dependency_query_positions"
                )
                if prior_positions is None:
                    monitor_state["attn_monitor_dependency_query_positions"] = (
                        query_positions.detach().clone()
                    )
                    monitor_state["attn_monitor_dependency_max"] = dependency
                    monitor_state["attn_monitor_dependency_layer_count"] = 1
                    if segment_head_mass is not None:
                        monitor_state["attn_monitor_l0_segment_head_mass_bf16"] = (
                            segment_head_mass
                        )
                    if prompt_importance is not None:
                        monitor_state["attn_monitor_prompt_key_max"] = prompt_importance
                    if prompt_change_dependency is not None:
                        monitor_state["attn_monitor_prompt_change_max"] = (
                            prompt_change_dependency
                        )
                    if (
                        prefix_dependency is not None
                        or prefix_future_dependency is not None
                    ):
                        monitor_state[
                            "attn_monitor_prefix_dependency_query_positions"
                        ] = prefix_dependency_positions
                    if prefix_dependency is not None:
                        monitor_state["attn_monitor_prefix_dependency_max"] = (
                            prefix_dependency
                        )
                    if prefix_reverse_dependency is not None:
                        monitor_state["attn_monitor_prefix_reverse_dependency_max"] = (
                            prefix_reverse_dependency
                        )
                    if prefix_future_dependency is not None:
                        monitor_state["attn_monitor_prefix_future_dependency_max"] = (
                            prefix_future_dependency
                        )
                    if future_prefix_dependency is not None:
                        monitor_state[
                            "attn_monitor_future_prefix_dependency_query_positions"
                        ] = future_prefix_dependency_positions
                        monitor_state["attn_monitor_future_prefix_dependency_max"] = (
                            future_prefix_dependency
                        )
                        monitor_state["attn_monitor_future_prefix_key_end"] = (
                            future_prefix_key_end
                        )
                else:
                    if not torch.equal(prior_positions, query_positions):
                        raise RuntimeError(
                            "all-layer dependency monitor changed query positions across layers"
                        )
                    monitor_state["attn_monitor_dependency_max"] = torch.maximum(
                        monitor_state["attn_monitor_dependency_max"], dependency
                    )
                    monitor_state["attn_monitor_dependency_layer_count"] = (
                        int(monitor_state["attn_monitor_dependency_layer_count"]) + 1
                    )
                    if prompt_importance is not None:
                        monitor_state["attn_monitor_prompt_key_max"] = torch.maximum(
                            monitor_state["attn_monitor_prompt_key_max"],
                            prompt_importance,
                        )
                    if prompt_change_dependency is not None:
                        monitor_state["attn_monitor_prompt_change_max"] = torch.maximum(
                            monitor_state["attn_monitor_prompt_change_max"],
                            prompt_change_dependency,
                        )
                    if prefix_dependency is not None:
                        prior_prefix_positions = monitor_state[
                            "attn_monitor_prefix_dependency_query_positions"
                        ]
                        if not torch.equal(
                            prior_prefix_positions, prefix_dependency_positions
                        ):
                            raise RuntimeError(
                                "all-layer prefix dependency changed query positions"
                            )
                        monitor_state["attn_monitor_prefix_dependency_max"] = (
                            torch.maximum(
                                monitor_state["attn_monitor_prefix_dependency_max"],
                                prefix_dependency,
                            )
                        )
                    if prefix_reverse_dependency is not None:
                        monitor_state["attn_monitor_prefix_reverse_dependency_max"] = (
                            torch.maximum(
                                monitor_state[
                                    "attn_monitor_prefix_reverse_dependency_max"
                                ],
                                prefix_reverse_dependency,
                            )
                        )
                    if prefix_future_dependency is not None:
                        prior_prefix_positions = monitor_state[
                            "attn_monitor_prefix_dependency_query_positions"
                        ]
                        if not torch.equal(
                            prior_prefix_positions, prefix_dependency_positions
                        ):
                            raise RuntimeError(
                                "all-layer future prefix dependency changed query positions"
                            )
                        monitor_state["attn_monitor_prefix_future_dependency_max"] = (
                            torch.maximum(
                                monitor_state[
                                    "attn_monitor_prefix_future_dependency_max"
                                ],
                                prefix_future_dependency,
                            )
                        )
                    if future_prefix_dependency is not None:
                        prior_future_positions = monitor_state[
                            "attn_monitor_future_prefix_dependency_query_positions"
                        ]
                        if not torch.equal(
                            prior_future_positions, future_prefix_dependency_positions
                        ):
                            raise RuntimeError(
                                "attention relation diagnostic query positions changed"
                            )
                        if (
                            monitor_state["attn_monitor_future_prefix_key_end"]
                            != future_prefix_key_end
                        ):
                            raise RuntimeError(
                                "attention relation diagnostic key range changed"
                            )
                        monitor_state["attn_monitor_future_prefix_dependency_max"] = (
                            torch.maximum(
                                monitor_state[
                                    "attn_monitor_future_prefix_dependency_max"
                                ],
                                future_prefix_dependency,
                            )
                        )
                if int(self.layer_id) == monitor_publish_layer:
                    dependency_last = dependency
                    layer_count = int(
                        monitor_state["attn_monitor_dependency_layer_count"]
                    )
                    if layer_count != int(self.config.n_layers):
                        raise RuntimeError(
                            "all-layer dependency monitor did not observe every transformer layer"
                        )
                    dependency = monitor_state["attn_monitor_dependency_max"]
                    current_profile = {
                        "query_positions": query_positions.detach().clone(),
                        "dependency_mean": dependency,
                        "dependency_last": dependency_last,
                        "prompt_key_importance": monitor_state.get(
                            "attn_monitor_prompt_key_max"
                        ),
                        "prompt_change_dependency": monitor_state.get(
                            "attn_monitor_prompt_change_max"
                        ),
                        "prefix_dependency_query_positions": monitor_state.get(
                            "attn_monitor_prefix_dependency_query_positions"
                        ),
                        "prefix_dependency_mean": monitor_state.get(
                            "attn_monitor_prefix_dependency_max"
                        ),
                        "prefix_reverse_dependency_mean": monitor_state.get(
                            "attn_monitor_prefix_reverse_dependency_max"
                        ),
                        "prefix_future_dependency_mean": monitor_state.get(
                            "attn_monitor_prefix_future_dependency_max"
                        ),
                        "future_prefix_dependency_query_positions": monitor_state.get(
                            "attn_monitor_future_prefix_dependency_query_positions"
                        ),
                        "future_prefix_dependency_mean": monitor_state.get(
                            "attn_monitor_future_prefix_dependency_max"
                        ),
                        "future_prefix_key_end": monitor_state.get(
                            "attn_monitor_future_prefix_key_end"
                        ),
                        "prefix_dependency_last": prefix_dependency,
                        "prefix_reverse_dependency_last": prefix_reverse_dependency,
                        "numeric_probe": numeric_probe,
                        "l0_segment_head_mass_bf16": monitor_state.get(
                            "attn_monitor_l0_segment_head_mass_bf16"
                        ),
                        "l31_segment_head_mass_bf16": segment_head_mass,
                        "layer_reduction": "max",
                        "layer_count": torch.tensor(
                            layer_count, device=dependency.device, dtype=torch.int32
                        ),
                    }
                    self.__cache["attn_monitor_current"] = current_profile
            else:
                self.__cache["attn_monitor_current"] = {
                    "query_positions": query_positions.detach().clone(),
                    "dependency_mean": dependency,
                    "prompt_key_importance": prompt_importance,
                    "prompt_change_dependency": prompt_change_dependency,
                    "prefix_dependency_query_positions": prefix_dependency_positions,
                    "prefix_dependency_mean": prefix_dependency,
                    "prefix_reverse_dependency_mean": prefix_reverse_dependency,
                    "prefix_future_dependency_mean": prefix_future_dependency,
                    "future_prefix_dependency_query_positions": future_prefix_dependency_positions,
                    "future_prefix_dependency_mean": future_prefix_dependency,
                    "future_prefix_key_end": future_prefix_key_end,
                    "prefix_dependency_last": prefix_dependency,
                    "prefix_reverse_dependency_last": prefix_reverse_dependency,
                    "numeric_probe": numeric_probe,
                    "selected_layer_segment_head_mass_bf16": segment_head_mass,
                    "layer_reduction": str(
                        getattr(self.config, "attn_monitor_layer_mode", "last")
                    ),
                    "layer_count": torch.tensor(
                        1, device=dependency.device, dtype=torch.int32
                    ),
                }
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return (self.attn_out(att), present)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(
        cls, layer_id: int, config: ModelConfig, cache: BufferCache
    ) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model,
            sum(self.fused_dims),
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(
            self.config,
            self.att_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )
        init_weights(
            self.config,
            self.ff_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self._activation_checkpoint_fn is not None:
            (q, k, v) = self.att_proj(
                self._activation_checkpoint_fn(self.attn_norm, x)
            ).split(self.fused_dims, dim=-1)
        else:
            (q, k, v) = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)
        if self._activation_checkpoint_fn is not None:
            (att, cache) = self._activation_checkpoint_fn(
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )
        else:
            (att, cache) = self.attention(
                q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        x = x + self.dropout(att)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x
        return (x, cache)


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model,
            q_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            k_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            v_proj_out_dim,
            bias=config.include_bias | config.include_qkv_bias,
            device=config.init_device,
        )
        self.ff_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.up_proj = nn.Linear(
            config.d_model,
            self.hidden_size,
            bias=config.include_bias,
            device=config.init_device,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)

    def _residual_add(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_target_numeric_bf16_residual", False):
            return lhs + rhs
        from numerics.bf16 import bf16_add

        return bf16_add(lhs, rhs).to(lhs.dtype)

    def _silu_multiply(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_target_numeric_silu_pwl16", False):
            return self.act(gate) * up
        from numerics.operator_kernels import (
            silu_pwl_bf16_cuda,
            triton_operator_numeric_available,
        )
        from numerics.bf16 import bf16_mul, silu_pwl_bf16, _silu_pwl_bf16

        coefficients = getattr(self, "_target_numeric_silu_coefficients", None)
        activated = (
            silu_pwl_bf16_cuda(gate.contiguous(), coefficients=coefficients)
            if gate.is_cuda and triton_operator_numeric_available()
            else silu_pwl_bf16(gate)
            if coefficients is None
            else _silu_pwl_bf16(gate, *coefficients)
        )
        exact = bf16_mul(activated, up).to(gate.dtype)
        return exact

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        replace_position: Optional[torch.Tensor] = None,
        replace_position_kv: Optional[torch.Tensor] = None,
        local_idx: Optional[torch.Tensor] = None,
        query_position_ids: Optional[torch.Tensor] = None,
        kv_write_position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        if self._activation_checkpoint_fn is not None:
            (att, cache) = self._activation_checkpoint_fn(
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                replace_position=replace_position,
                replace_position_kv=replace_position_kv,
                local_idx=local_idx,
                query_position_ids=query_position_ids,
                kv_write_position_ids=kv_write_position_ids,
            )
        else:
            (att, cache) = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                replace_position=replace_position,
                replace_position_kv=replace_position_kv,
                local_idx=local_idx,
                query_position_ids=query_position_ids,
                kv_write_position_ids=kv_write_position_ids,
            )
        x = self._residual_add(x, self.dropout(att))
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        (x, x_up) = (self.ff_proj(x), self.up_proj(x))
        if self._activation_checkpoint_fn is not None and getattr(
            self, "_target_numeric_silu_pwl16", False
        ):
            raise RuntimeError(
                "target numeric SiLU does not support activation checkpointing"
            )
        x = self._silu_multiply(x, x_up)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = self._residual_add(og_x, x)
        return (x, cache)


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden_states: Optional[Tuple[torch.Tensor]]


class LLaDABlockGroup(nn.ModuleList):
    def __init__(
        self,
        config: ModelConfig,
        layer_offset: int,
        modules: Optional[Iterable[nn.Module]] = None,
    ):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[
            ActivationCheckpointingStrategy
        ] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                self.activation_checkpointing_strategy
                == ActivationCheckpointingStrategy.whole_layer
                or (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                (x, cache) = self._activation_checkpoint_fn(
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            else:
                (x, cache) = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return (x, attn_key_values)

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")
        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")
        if (
            self.config.embedding_size is not None
            and self.config.embedding_size != self.config.vocab_size
        ):
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception(
                    "embedding size should be at least as big as vocab size"
                )
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.",
                    UserWarning,
                )
        self.activation_checkpointing_strategy: Optional[
            ActivationCheckpointingStrategy
        ] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(
            self.config
        )
        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                    device=config.init_device,
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )
        blocks = [
            LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)
        ]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})
        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {
                    "wpe": nn.Embedding(
                        config.max_sequence_length,
                        config.d_model,
                        device=config.init_device,
                    )
                }
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        if self.config.alibi:
            get_causal_attention_bias(
                self.__cache, config.max_sequence_length, _non_meta_init_device(config)
            )
            self.get_alibi_attention_bias(
                config.max_sequence_length, _non_meta_init_device(config)
            )

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy]
    ):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        init_weights(
            self.config,
            self.transformer.wte,
            std_factor=0.5 * math.sqrt(self.config.d_model)
            if self.config.scale_logits
            else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(
                self.config, self.transformer.wpe, type_of_module=ModuleType.emb
            )
        self.transformer.ln_f.reset_parameters()
        if hasattr(self.transformer, "ff_out"):
            init_weights(
                self.config,
                self.transformer.ff_out,
                type_of_module=ModuleType.final_out,
            )
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        if (
            alibi_bias := self.__cache.get("alibi_attention_bias")
        ) is not None and alibi_bias.shape[-1] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        logits_start: Optional[int] = None,
        logits_end: Optional[int] = None,
        logits_positions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
        replace_position_kv: Optional[torch.Tensor] = None,
        local_idx: Optional[torch.Tensor] = None,
        query_position_ids: Optional[torch.Tensor] = None,
        kv_write_position_ids: Optional[torch.Tensor] = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            Dense diffusion forwards use bidirectional attention. Cached forwards use
            bidirectional attention when noncausal_cached_attention is enabled.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        assert (
            not self.config.alibi
        ), "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        if past_key_values:
            assert len(past_key_values) == self.config.n_layers
        (batch_size, seq_len) = (
            input_ids.size()
            if input_embeddings is None
            else input_embeddings.size()[:2]
        )
        if query_position_ids is not None:
            if query_position_ids.shape != (batch_size, seq_len):
                raise ValueError("query_position_ids must match the packed input shape")
            if query_position_ids.dtype != torch.long:
                raise ValueError("query_position_ids must be int64")
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
        x = (
            self.transformer.wte(input_ids)
            if input_embeddings is None
            else input_embeddings
        )
        if self.config.input_emb_norm:
            x = x * self.config.d_model**0.5
        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=x.device
            ).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x
        x = self.transformer.emb_drop(x)
        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[
                :, None, None, :
            ]
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                attention_mask.dtype
            ).min
        else:
            attention_mask = None
        noncausal_cached_attention = bool(
            past_key_values is not None
            and getattr(self.config, "noncausal_cached_attention", False)
        )
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            or (past_key_values is not None and (not noncausal_cached_attention))
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                if past_key_values is not None and not noncausal_cached_attention:
                    attention_bias = get_causal_attention_bias(
                        self.__cache, past_length + seq_len, x.device
                    )
                else:
                    attention_bias = torch.zeros(
                        (1, 1, past_length + seq_len, past_length + seq_len),
                        dtype=torch.float, device=x.device,
                    )
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(
                    attention_bias == 0.0, torch.finfo(attention_bias.dtype).min
                )
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(
                dtype=torch.float
            )
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
            [] if use_cache else None
        )
        all_hidden_states = []
        if self.config.block_group_size == 1:
            replace_position_kv_eff = replace_position_kv
            local_idx_eff = local_idx
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x)
                layer_past = (
                    None if past_key_values is None else past_key_values[block_idx]
                )
                if (
                    self.activation_checkpointing_strategy
                    == ActivationCheckpointingStrategy.whole_layer
                    or (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy
                        == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    (x, cache) = self._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        replace_position=replace_position,
                        replace_position_kv=replace_position_kv_eff,
                        local_idx=local_idx_eff,
                        query_position_ids=query_position_ids,
                        kv_write_position_ids=kv_write_position_ids,
                    )
                else:
                    (x, cache) = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        replace_position=replace_position,
                        replace_position_kv=replace_position_kv_eff,
                        local_idx=local_idx_eff,
                        query_position_ids=query_position_ids,
                        kv_write_position_ids=kv_write_position_ids,
                    )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    all_hidden_states.append(x)
                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1)
                        * self.config.block_group_size
                    ]
                )
                (x, cache) = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)
        if logits_positions is not None:
            if last_logits_only or logits_start is not None or logits_end is not None:
                raise ValueError("logits_positions cannot be combined with other logits selection")
            if logits_positions.ndim != 1 or logits_positions.dtype != torch.long:
                raise ValueError("logits_positions must be a rank-1 long tensor")
            if bool(((logits_positions < 0) | (logits_positions >= seq_len)).any()):
                raise ValueError("logits_positions is outside the input sequence")
            x = x.index_select(1, logits_positions.to(x.device))
        if last_logits_only and (logits_start is not None or logits_end is not None):
            raise ValueError(
                "last_logits_only cannot be combined with logits selection"
            )
        if (logits_start is None) != (logits_end is None):
            raise ValueError("logits_start and logits_end must be provided together")
        if logits_start is not None and logits_end is not None:
            if logits_start < 0 or logits_end <= logits_start or logits_end > seq_len:
                raise ValueError(
                    f"invalid logits range [{logits_start}, {logits_end}) for sequence length {seq_len}"
                )
            x = x[:, logits_start:logits_end, :]
        elif last_logits_only:
            x = x[:, -1, :].unsqueeze(1)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))
        return LLaDAOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    """
    Utility function
    """
    kwargs = {}
    for field in fields(ModelConfig):
        if hasattr(config, field.name):
            kwargs[field.name] = getattr(config, field.name)
        elif field.default is not MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not MISSING:
            kwargs[field.name] = field.default_factory()
        else:
            raise AttributeError(
                f"{type(config).__name__} object has no attribute {field.name!r}"
            )
    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(PreTrainedModel):
    """
    Hugging Face model wrapper for LLaDA.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]
    _tp_plan = []

    def __init__(
        self,
        config: LLaDAConfig,
        model: Optional[LLaDAModel] = None,
        init_params: bool = False,
    ):
        super().__init__(config)
        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        replace_position: Optional[torch.Tensor] = None,
        replace_position_kv: Optional[torch.Tensor] = None,
        local_idx: Optional[torch.Tensor] = None,
        logits_start: Optional[int] = None,
        logits_end: Optional[int] = None,
        logits_positions: Optional[torch.Tensor] = None,
        query_position_ids: Optional[torch.Tensor] = None,
        kv_write_position_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            replace_position=replace_position,
            replace_position_kv=replace_position_kv,
            local_idx=local_idx,
            logits_positions=logits_positions,
            logits_start=logits_start,
            logits_end=logits_end,
            query_position_ids=query_position_ids,
            kv_write_position_ids=kv_write_position_ids,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        loss = None
        if labels is not None:
            import warnings

            warnings.warn(
                "Note that for LLaDA, you cannot calculate the loss here.", UserWarning
            )
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple]] = None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}
        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte


AutoModel.register(LLaDAConfig, LLaDAModelLM)
