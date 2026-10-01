"""Fixed SpinQuant rotations and BF16 transform boundaries for LLaDA."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import torch
from numerics.bf16 import bf16, bf16_add, bf16_mul
from quantization.checkpoint import sha256_file

INSTRUCT_CHECKPOINT = Path(
    os.environ.get("SUPRA_RELOCATED_INSTRUCT_CHECKPOINT", "models/LLaDA-8B-Instruct")
).resolve()
BASE_CHECKPOINT = Path(
    os.environ.get("SUPRA_RELOCATED_BASE_CHECKPOINT", "models/LLaDA-8B-Base")
).resolve()
INSTRUCT_CONFIG_SHA256 = (
    "5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a"
)
INSTRUCT_INDEX_SHA256 = (
    "28b4ec27206e42e7ade630450e6ce618bd197acf34d35120e8e86d2bb910a408"
)
INSTRUCT_TOKENIZER_SHA256 = (
    "071eb20fe7bd601550b1b7838ff696d6c93b88f51257f753303d3df23163c381"
)
BASE_TOKENIZER_SHA256 = (
    "ee1ef8e5f6d9493ac25480b7b7337ff5d2c1b946190afff18d12c86ca738ae00"
)
ALGO_ROOT = Path(os.environ.get("SUPRA_ALGORITHM_ROOT", "artifacts")).resolve()
ROTATION_SEED = 20260806
CHECKPOINT_PROFILES = {
    INSTRUCT_CHECKPOINT: {
        "config_sha256": INSTRUCT_CONFIG_SHA256,
        "index_sha256": INSTRUCT_INDEX_SHA256,
        "tokenizer_sha256": INSTRUCT_TOKENIZER_SHA256,
        "model_identity": "LLaDA-8B-Instruct-original-bf16",
    },
    BASE_CHECKPOINT: {
        "config_sha256": INSTRUCT_CONFIG_SHA256,
        "index_sha256": INSTRUCT_INDEX_SHA256,
        "tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "model_identity": "LLaDA-8B-Base-original-bf16",
    },
}


def _validate_llada_config(config_path: Path, model_identity: str) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "architectures": ["LLaDAModelLM"],
        "d_model": 4096,
        "n_heads": 32,
        "n_kv_heads": 32,
        "n_layers": 32,
        "mlp_hidden_size": 12288,
        "include_bias": False,
        "weight_tying": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"unexpected {model_identity} config {key}={config.get(key)!r}, expected {value!r}"
            )


def checkpoint_identity(path: Path) -> dict[str, str]:
    """Return the identity of a supported original LLaDA checkpoint."""
    resolved = Path(path).resolve()
    config_path = resolved / "config.json"
    index_path = resolved / "model.safetensors.index.json"
    tokenizer_path = resolved / "tokenizer.json"
    config_hash = sha256_file(config_path)
    index_hash = sha256_file(index_path)
    tokenizer_hash = sha256_file(tokenizer_path)
    profile = next((entry for entry in CHECKPOINT_PROFILES.values()
                    if config_hash == entry["config_sha256"]
                    and index_hash == entry["index_sha256"]
                    and tokenizer_hash == entry["tokenizer_sha256"]), None)
    if profile is None:
        raise ValueError(f"LLaDA config/index/tokenizer identity mismatch: {resolved}")
    _validate_llada_config(config_path, profile["model_identity"])
    return {
        "path": str(resolved),
        "config_sha256": config_hash,
        "index_sha256": index_hash,
        "model_identity": profile["model_identity"],
    }


def supported_checkpoint_identities() -> tuple[dict[str, str], ...]:
    return tuple(
        (
            {
                "path": str(path),
                "config_sha256": profile["config_sha256"],
                "index_sha256": profile["index_sha256"],
                "model_identity": profile["model_identity"],
            }
            for (path, profile) in CHECKPOINT_PROFILES.items()
        )
    )


def require_algo_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    source = Path(__file__).resolve()
    algorithm_root = source.parents[2]
    release_root = (
        algorithm_root.parent
        if algorithm_root.name == "algorithm" and (algorithm_root.parent / "hardware").is_dir()
        else algorithm_root
    )
    project = next(
        (p for p in source.parents if (p / ".git").exists()), release_root
    )
    if resolved == project or project in resolved.parents:
        raise ValueError("algorithm output must be outside the source repository")
    try:
        resolved.relative_to(ALGO_ROOT)
    except ValueError as error:
        raise ValueError(
            f"algorithm output must be below {ALGO_ROOT}, got {resolved}"
        ) from error
    if resolved == ALGO_ROOT:
        raise ValueError(
            "algorithm output must be a child of SUPRA_ALGORITHM_ROOT, not the root itself"
        )
    return resolved


def fwht_last_dim(values: torch.Tensor) -> torch.Tensor:
    width = int(values.shape[-1])
    if width <= 0 or width & width - 1:
        raise ValueError(f"FWHT width must be a positive power of two, got {width}")
    result = values
    stride = 1
    while stride < width:
        stage = result.reshape(*result.shape[:-1], width // (2 * stride), 2, stride)
        (left, right) = stage.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        stride *= 2
    return result


def normalized_hadamard(values: torch.Tensor) -> torch.Tensor:
    return fwht_last_dim(values.to(torch.float32)) / math.sqrt(values.shape[-1])


def fixed_rotation_signs(
    seed: int = ROTATION_SEED,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    r1 = (
        torch.randint(0, 2, (4096,), generator=generator, dtype=torch.int8)
        .mul(2)
        .sub(1)
    )
    r2 = [
        torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8).mul(2).sub(1)
        for _ in range(32)
    ]
    return (r1, r2)


def random_hadamard_dense(sign: torch.Tensor) -> torch.Tensor:
    size = int(sign.numel())
    identity = torch.eye(size, device=sign.device, dtype=torch.float32)
    return normalized_hadamard(identity * sign.to(torch.float32).unsqueeze(1))


def rotate_input_weight_fixed(weight: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or sign.shape != weight.shape[1:]:
        raise ValueError("fixed R input rotation geometry mismatch")
    return normalized_hadamard(
        weight.to(torch.float32) * sign.to(weight.device, torch.float32).unsqueeze(0)
    )


def rotate_residual_output_weight_fixed(
    weight: torch.Tensor, sign: torch.Tensor
) -> torch.Tensor:
    if weight.ndim != 2 or sign.shape != weight.shape[:1]:
        raise ValueError("fixed R output rotation geometry mismatch")
    signed = weight.to(torch.float32) * sign.to(weight.device, torch.float32).unsqueeze(
        1
    )
    return normalized_hadamard(signed.T).T


def rotate_v_weight_fixed(
    weight: torch.Tensor, r1_sign: torch.Tensor, r2_sign: torch.Tensor, heads: int
) -> torch.Tensor:
    head_dim = int(r2_sign.numel())
    rotated = rotate_input_weight_fixed(weight, r1_sign).reshape(
        heads, head_dim, weight.shape[1]
    )
    signed = rotated * r2_sign.to(weight.device, torch.float32).reshape(1, head_dim, 1)
    return (
        normalized_hadamard(signed.transpose(1, 2)).transpose(1, 2).reshape_as(weight)
    )


def rotate_o_weight_fixed(
    weight: torch.Tensor, r1_sign: torch.Tensor, r2_sign: torch.Tensor, heads: int
) -> torch.Tensor:
    head_dim = int(r2_sign.numel())
    result = rotate_residual_output_weight_fixed(weight, r1_sign)
    blocks = result.reshape(weight.shape[0], heads, head_dim)
    return normalized_hadamard(
        blocks * r2_sign.to(weight.device, torch.float32).reshape(1, 1, head_dim)
    ).reshape_as(weight)


def rotate_ff_out_weight_fixed(
    weight: torch.Tensor, r1_sign: torch.Tensor, *, had: bool
) -> torch.Tensor:
    result = rotate_residual_output_weight_fixed(weight, r1_sign)
    return structured_hadamard_12288(result) if had else result


def paley_hadamard_12(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Construct a deterministic order-12 Paley Hadamard matrix."""
    q = 11
    quadratic_residues = {value * value % q for value in range(1, q)}
    matrix = torch.ones((q + 1, q + 1), dtype=torch.float64)
    for row in range(q):
        for column in range(q):
            if row == column:
                value = -1.0
            else:
                value = 1.0 if (row - column) % q in quadratic_residues else -1.0
            matrix[row + 1, column + 1] = value
    if not torch.equal(
        matrix @ matrix.T, torch.eye(q + 1, dtype=torch.float64) * (q + 1)
    ):
        raise RuntimeError("internal Paley H12 construction is not Hadamard")
    return matrix.to(device=device, dtype=dtype)


def structured_hadamard_12288(
    values: torch.Tensor, *, transpose: bool = False
) -> torch.Tensor:
    if values.shape[-1] != 12288:
        raise ValueError(f"structured H12288 expects K=12288, got {values.shape[-1]}")
    original_shape = values.shape
    blocks = values.to(torch.float32).reshape(-1, 12, 1024)
    blocks = fwht_last_dim(blocks)
    h12 = paley_hadamard_12(device=values.device, dtype=blocks.dtype)
    if transpose:
        h12 = h12.T
    blocks = torch.matmul(h12.unsqueeze(0), blocks)
    return (blocks / math.sqrt(12288.0)).reshape(original_shape)


def structured_hadamard_12288_bf16(values: torch.Tensor) -> torch.Tensor:
    """Target R4 with BF16 rounding at every H1024 and H12 reduction stage."""
    if values.shape[-1] != 12288:
        raise ValueError(f"structured H12288 expects K=12288, got {values.shape[-1]}")
    original_shape = values.shape
    blocks = bf16(values).reshape(-1, 12, 1024)
    stride = 1
    while stride < 1024:
        stage = blocks.reshape(-1, 12, 1024 // (2 * stride), 2, stride)
        (left, right) = stage.unbind(dim=-2)
        blocks = torch.stack(
            (bf16_add(left, right), bf16_add(left, -right)), dim=-2
        ).reshape_as(blocks)
        stride *= 2
    signs = paley_hadamard_12(device=values.device, dtype=torch.float32)
    output_blocks = []
    for output_start in range(0, 12, 3):
        output_signs = signs[output_start : output_start + 3].reshape(1, 3, 12, 1)
        terms = bf16_mul(blocks.unsqueeze(1), output_signs)
        terms = torch.cat(
            (
                terms,
                torch.zeros(
                    *terms.shape[:2],
                    4,
                    terms.shape[-1],
                    device=terms.device,
                    dtype=terms.dtype,
                ),
            ),
            dim=2,
        )
        while terms.shape[2] > 1:
            terms = bf16_add(terms[:, :, 0::2], terms[:, :, 1::2])
        output_blocks.append(terms[:, :, 0])
    blocks = torch.cat(output_blocks, dim=1)
    normalization = bf16(torch.tensor(12288.0 ** (-0.5), device=values.device))
    normalized = bf16_mul(blocks, normalization)
    return normalized.reshape(original_shape).to(values.dtype)


def rotate_input_weight(weight: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    return weight.to(torch.float32) @ r1.to(torch.float32)


def rotate_residual_output_weight(
    weight: torch.Tensor, r1: torch.Tensor
) -> torch.Tensor:
    return r1.to(torch.float32).T @ weight.to(torch.float32)


def rotate_v_weight(
    weight: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, heads: int
) -> torch.Tensor:
    head_dim = int(r2.shape[0])
    if weight.shape[0] != heads * head_dim or r2.shape != (head_dim, head_dim):
        raise ValueError("V weight/R2 geometry mismatch")
    rotated = rotate_input_weight(weight, r1).reshape(heads, head_dim, weight.shape[1])
    return torch.matmul(r2.to(torch.float32).T.unsqueeze(0), rotated).reshape_as(weight)


def rotate_o_weight(
    weight: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, heads: int
) -> torch.Tensor:
    head_dim = int(r2.shape[0])
    if weight.shape[1] != heads * head_dim or r2.shape != (head_dim, head_dim):
        raise ValueError("O weight/R2 geometry mismatch")
    residual_rotated = rotate_residual_output_weight(weight, r1)
    blocks = residual_rotated.reshape(weight.shape[0], heads, head_dim)
    return torch.matmul(blocks, r2.to(torch.float32)).reshape_as(weight)


def rotate_ff_out_weight(
    weight: torch.Tensor, r1: torch.Tensor, *, had: bool
) -> torch.Tensor:
    result = rotate_residual_output_weight(weight, r1)
    return structured_hadamard_12288(result) if had else result
