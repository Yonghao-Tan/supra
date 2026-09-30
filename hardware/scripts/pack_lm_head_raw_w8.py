"""Pack signed-W8 head weights and check their BF16 scales."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def validate_scales(scale_path: Path, vocabulary_size: int) -> tuple[int, int]:
    if scale_path.stat().st_size != vocabulary_size * 2:
        raise ValueError("BF16 scale payload length does not match vocabulary size")
    values = np.memmap(scale_path, dtype="<u2", mode="r", shape=(vocabulary_size,))
    exponent = values & np.uint16(0x7F80)
    magnitude = values & np.uint16(0x7FFF)
    sign = values & np.uint16(0x8000)
    invalid = (exponent == np.uint16(0x7F80)) | (magnitude == 0) | (sign != 0)
    invalid_count = int(np.count_nonzero(invalid))
    del values
    if invalid_count:
        raise ValueError(f"weight scales contain {invalid_count} non-positive or non-finite values")
    return vocabulary_size, invalid_count


def pack_weights(
    source_path: Path,
    destination_path: Path,
    vocabulary_size: int,
    input_features: int,
    *,
    chunk_tiles: int = 64,
) -> tuple[int, int, int]:
    expected_bytes = vocabulary_size * input_features
    if source_path.stat().st_size != expected_bytes:
        raise ValueError("signed-W8 source length does not match [vocabulary,input_features]")
    source = np.memmap(
        source_path, dtype=np.int8, mode="r", shape=(vocabulary_size, input_features)
    )
    vocabulary_tiles = vocabulary_size // 8
    k_tiles = input_features // 8
    destination = np.memmap(
        destination_path,
        dtype=np.int8,
        mode="w+",
        shape=(vocabulary_tiles, k_tiles, 8, 8),
    )
    observed_min = 127
    observed_max = -127
    minus128_count = 0
    for first_tile in range(0, vocabulary_tiles, chunk_tiles):
        tile_count = min(chunk_tiles, vocabulary_tiles - first_tile)
        first_output = first_tile * 8
        rows = np.asarray(source[first_output:first_output + tile_count * 8])
        observed_min = min(observed_min, int(rows.min()))
        observed_max = max(observed_max, int(rows.max()))
        minus128_count += int(np.count_nonzero(rows == -128))
        if minus128_count:
            break
        packed = rows.reshape(tile_count, 8, k_tiles, 8).transpose(0, 2, 3, 1)
        destination[first_tile:first_tile + tile_count] = packed
    destination.flush()
    del destination
    del source
    if minus128_count:
        destination_path.unlink(missing_ok=True)
        raise ValueError(f"signed-W8 source contains {minus128_count} occurrences of -128")
    return observed_min, observed_max, minus128_count


def verify_panels(
    source_path: Path,
    packed_path: Path,
    vocabulary_size: int,
    input_features: int,
) -> list[int]:
    source = np.memmap(
        source_path, dtype=np.int8, mode="r", shape=(vocabulary_size, input_features)
    )
    packed = np.memmap(
        packed_path,
        dtype=np.int8,
        mode="r",
        shape=(vocabulary_size // 8, input_features // 8, 8, 8),
    )
    panels = sorted({0, vocabulary_size // 16, vocabulary_size // 8 - 1})
    for panel in panels:
        expected = np.asarray(source[panel * 8:panel * 8 + 8]).reshape(
            8, input_features // 8, 8
        ).transpose(1, 2, 0)
        if not np.array_equal(packed[panel], expected):
            raise ValueError(f"packed panel {panel} does not round-trip")
    del packed
    del source
    return panels
