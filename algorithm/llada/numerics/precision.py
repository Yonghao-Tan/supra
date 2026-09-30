"""Shared per-forward activation precision for state-driven W4 Linear rows."""

from __future__ import annotations
from contextlib import contextmanager
import torch


class RowPrecisionContext:
    """Expose one A4/A8 mode per logical input row to all block Linears."""

    def __init__(self) -> None:
        self._row_bits: torch.Tensor | None = None

    @property
    def row_bits(self) -> torch.Tensor:
        if self._row_bits is None:
            raise RuntimeError("row precision is not set for the current forward")
        return self._row_bits

    def activate(self, row_bits: torch.Tensor, *, validate_values: bool = True) -> None:
        if row_bits.dtype != torch.int8 or row_bits.ndim not in {1, 2}:
            raise ValueError("row_bits must be a rank-1 or rank-2 torch.int8 tensor")
        if validate_values and (not bool(torch.all((row_bits == 4) | (row_bits == 8)))):
            raise ValueError("row_bits values must be 4 or 8")
        self._row_bits = row_bits.reshape(-1)

    @contextmanager
    def using(self, row_bits: torch.Tensor):
        previous = self._row_bits
        try:
            self.activate(row_bits)
            yield
        finally:
            self._row_bits = previous

    def require(self, rows: int, device: torch.device) -> torch.Tensor:
        row_bits = self.row_bits
        if row_bits.numel() != rows:
            raise ValueError(
                f"row precision contains {row_bits.numel()} rows, expected {rows}"
            )
        if row_bits.device != device:
            raise ValueError(
                f"row precision is on {row_bits.device}, expected {device}"
            )
        return row_bits


def clip_a4_rows_bf16(
    values: torch.Tensor, row_bits: torch.Tensor, ratio: float, *, observer=None
) -> torch.Tensor:
    """Clamp A4 input rows with a BF16 limit; the caller materializes ratio as BF16."""
    rows = values.reshape(-1, values.shape[-1])
    if (
        values.dtype != torch.bfloat16
        or row_bits.dtype != torch.int8
        or row_bits.numel() != rows.shape[0]
        or (row_bits.device != values.device)
        or (not 0 < ratio <= 1)
    ):
        raise ValueError(
            "A4 clipping requires BF16 rows, matching int8 state bits and a ratio in (0,1]"
        )
    row_max = rows.float().abs().amax(1, keepdim=True)
    limit_fp32 = row_max * ratio
    limit = limit_fp32.to(torch.bfloat16)
    clipped = rows.clamp(-limit, limit)
    output = torch.where(
        (row_bits.reshape(-1) == 4)[:, None], clipped, rows
    ).reshape_as(values)
    if observer is not None:
        observer(
            dict(
                input=values,
                row_max=row_max,
                limit_fp32=limit_fp32,
                limit=limit,
                output=output,
            )
        )
    return output
