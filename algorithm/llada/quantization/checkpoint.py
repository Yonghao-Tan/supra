"""Read checkpoint shard indices and verify file identities."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _load_index(checkpoint_dir: Path) -> Mapping[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"index has no weight_map: {index_path}")
    return weight_map


def _shard_hashes(
    checkpoint_dir: Path, weight_map: Mapping[str, str]
) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = checkpoint_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"index references missing shard: {shard_path}")
        hashes[shard_name] = sha256_file(shard_path)
    return hashes
