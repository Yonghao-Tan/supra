"""
LLaDA configuration
"""

from transformers import AutoConfig, PretrainedConfig
from enum import Enum
from os import PathLike
from typing import Union
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "LayerNormType",
    "InitFnType",
    "ModelConfig",
]
PathOrStr = Union[str, PathLike]


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class LayerNormType(StrEnum):
    default = "default"
    "\n    The default LayerNorm implementation, equivalent to PyTorch's built-in version.\n    "
    low_precision = "low_precision"
    "\n    A low-precision version of the default LayerNorm.\n    "
    rms = "rms"
    "\n    An RMSNorm implementation. When using ``torch.compile`` this is\n    probably the fastest implementation.\n    "
    gemma_rms = "gemma_rms"
    "\n    An RMSNorm implementation by Gemma. When using ``torch.compile`` this is\n    probably the fastest implementation.\n    "
    amd_compatible = "amd_compatible"
    "\n    LayerNorm implemented manually to work around an issue with ROCm.\n    "


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"
    llama = "llama"
    "\n    A block similar to the sequential block with slightly different\n    implementations of operations like attention to imitate the behavior of Llama.\n    "


class InitFnType(StrEnum):
    mitchell = "mitchell"
    "\n    The strategy suggested to us by Mitchell Wortsman from UW.\n    This uses a truncated normal distribution with an adaptive standard deviation that depends\n    on the size of the weights as well as the depth of the layer.\n    "
    normal = "normal"
    "\n    All weights are initialized from the same normal distribution.\n    "
    kaiming_normal = "kaiming_normal"
    "\n    All weights are initialized with the Kaiming method from a normal distribution.\n    Note this currently won't work with FSDP.\n    "
    fan_in = "fan_in"
    '\n    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``\n    is the input dimensionality of the kernel.\n    '
    full_megatron = "full_megatron"
    '\n    This is what metaseq calls "full megatron init". It is the init used for Llama 2.\n    '


@dataclass
class ModelConfig:
    """
    LLaDA (model) configuration.
    """

    d_model: int = 768
    "\n    The hidden size of the model.\n    "
    n_heads: int = 12
    "\n    The number of self-attention heads.\n    "
    n_kv_heads: Optional[int] = None
    "\n    The number of heads to use for keys and values. Defaults to `n_heads`.\n    Set this to ``None`` or ``n_heads`` for normal multi-head attention.\n    Set this to 1 for multi-query attention.\n    Set it to some in-between value for Llama2-style grouped query attention.\n    "
    n_layers: int = 12
    "\n    The number of layers/blocks.\n    "
    mlp_ratio: int = 4
    "\n    The ratio of the inner MLP dimensionality to ``d_model``.\n    This is only used when ``mlp_hidden_size`` is not set.\n    "
    mlp_hidden_size: Optional[int] = None
    "\n    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.\n    "
    activation_type: ActivationType = ActivationType.swiglu
    "\n    The activation function to use within the MLP layers.\n    "
    block_type: BlockType = BlockType.sequential
    "\n    The transformer block implementation.\n    "
    block_group_size: int = 1
    "\n    The number of blocks to group together into a single parent block.\n    This has no effect on the number of parameters in the model and is only used to wrap groups\n    of blocks together with a single FSDP wrapper during training.\n    "
    alibi: bool = False
    "\n    If ``True``, use ALiBi embeddings. Mutually exclusive with ``rope``.\n    "
    alibi_bias_max: float = 8.0
    "\n    Maximum absolute value of ALiBi bias.\n    "
    rope: bool = False
    "\n    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.\n    "
    rope_full_precision: bool = True
    "\n    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,\n    apply RoPE at the precision of the input.\n    "
    flash_attention: bool = False
    "\n    If ``True``, use ``FlashAttention``.\n    "
    attention_dropout: float = 0.1
    "\n    The dropout probability within the attention modules.\n    "
    multi_query_attention: Optional[bool] = None
    "\n    Use the Multi-Query formulation of attention used in PaLM. This reduces the number of parameters\n    and is more efficient during inference.\n    "
    attention_layer_norm: bool = False
    "\n    Apply layer norm to the keys and queries within the attention mechanism.\n    This can help stabilize training.\n    "
    residual_dropout: float = 0.1
    "\n    The dropout probability for the MLP and attention output within each block.\n    "
    embedding_dropout: float = 0.1
    "\n    The dropout probability for embeddings.\n    "
    input_emb_norm: bool = False
    "\n    An input hidden_states norm implementation by Gemma.\n    "
    layer_norm_type: LayerNormType = LayerNormType.default
    "\n    The layernorm implementation to use.\n    "
    layer_norm_with_affine: bool = True
    "\n    Whether to include bias and weight parameters for the layer norms.\n    This only affects layer norms that are immediately followed by a linear layer in the forward pass,\n    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`\n    to ``False``.\n    "
    rms_norm_eps: float = 1e-05
    "\n    The rms layernorm eps param.\n    "
    attention_layer_norm_with_affine: bool = True
    "\n    Toggle affine transform for the QK norms.\n    "
    max_sequence_length: int = 1024
    "\n    The maximum input sequence length supported by the model.\n    "
    train_max_sequence_length: int = 1024
    "\n    The maximum input sequence length supported by the model during training.\n    "
    attn_monitor_layer: int = -1
    attn_monitor_all_layers: bool = False
    noncausal_cached_attention: bool = False
    "\n    Select the transformer layer used by the packed-row Attention monitor.\n    "
    use_manual_attention: bool = False
    "\n    If ``True``, compute attention as manual softmax(QK^T)V instead of calling PyTorch SDPA / FlashAttention.\n    Explicit attention implementation; requires more memory than fused attention.\n    "
    rope_theta: float = 10000.0
    "\n    The rope base param.\n    "
    include_qkv_bias: Optional[bool] = False
    "\n    Whether or not to include bias parameters in qkv linear layers.\n    "
    include_bias: bool = False
    "\n    Whether or not to include bias parameters in linear layers.\n    In PaLM, they got rid of all bias terms because they found that large\n    models tend to have near 0 bias terms anyway.\n    "
    bias_for_layer_norm: Optional[bool] = None
    "\n    Whether or not to include bias parameters in layer norm.\n    This is separate from the include_bias parameter, because of a ROCm crash when biases are disabled in\n    layer norm.\n    When this is None (the default), it inherits the setting from include_bias.\n    "
    scale_logits: bool = False
    "\n    If ``True``, scale the output logits by ``1 / sqrt(d_model)``.\n    "
    vocab_size: int = 50257
    "\n    Vocabulary size of the model.\n    "
    embedding_size: Optional[int] = 50304
    "\n    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default\n    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the\n    next multiple of 128 that's greater than ``vocab_size`` can improve throughput\n    substantially.\n    "
    weight_tying: bool = True
    "\n    Whether to tie output linear weights to the input embedding.\n    "
    eos_token_id: int = 50256
    "\n    The ID of the end-of-sentence special token.\n    "
    pad_token_id: int = 50256
    "\n    The ID of the token to use for padding. Defaults to the ID of the EOS token.\n    "
    mask_token_id: Optional[int] = 50256
    "\n    The ID of the token to use for mask token. Defaults to the ID of the EOS token.\n    "
    init_device: Optional[str] = None
    '\n    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".\n    '
    init_fn: InitFnType = InitFnType.normal
    "\n    The weight initialization strategy.\n    "
    init_std: float = 0.02
    '\n    The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such\n    as "normal".\n    '
    init_cutoff_factor: Optional[float] = None
    '\n    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such\n    as "normal". Setting this to None means values are not cutoff.\n    '
    precision: Optional[str] = None
    "\n    Precision metadata for training and evaluation.\n    "

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise Exception(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    "\n    Checkpoint every transformer layer.\n    "
    one_in_two = "one_in_two"
    "\n    Checkpoint one in two transformer layers.\n    "
    one_in_three = "one_in_three"
    "\n    Checkpoint one in three transformer layers.\n    "
    one_in_four = "one_in_four"
    "\n    Checkpoint one in four transformer layers.\n    "
    two_in_three = "two_in_three"
    "\n    Checkpoint two out of every three transformer layers.\n    "
    three_in_four = "three_in_four"
    "\n    Checkpoint three out of four of every transformer layers.\n    "
    four_in_five = "four_in_five"
    "\n    Checkpoint four out of five of every transformer layers.\n    "
    nine_in_ten = "nine_in_ten"
    "\n    Checkpoint nine out of ten of every transformer layers.\n    "
    fine_grained = "fine_grained"
    "\n    Focus checkpointing on where it is cheap to recompute and saves most memory.\n    "


class LLaDAConfig(PretrainedConfig):
    model_type = "llada"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.__dict__
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {"architectures": all_kwargs.get("architectures", ["LLaDAModelLM"])}
        )
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model


AutoConfig.register("llada", LLaDAConfig)
