#!/usr/bin/env python3
"""Resolve simulation outputs under an external artifact directory."""

from __future__ import annotations

import os
from pathlib import Path
import re


RTL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RTL_ROOT.parent
ENVIRONMENT_VARIABLE = "SUPRA_ARTIFACT_ROOT"
COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlink_components(path: Path, root: Path) -> None:
    current = root
    if current.is_symlink():
        raise ValueError(f"artifact root must not be a symbolic link: {current}")
    for component in path.relative_to(root).parts:
        current = current / component
        if current.exists() and current.is_symlink():
            raise ValueError(f"artifact path contains a symbolic link: {current}")


def artifact_root(create: bool = True) -> Path:
    configured = os.environ.get(ENVIRONMENT_VARIABLE)
    configured_path = Path(configured) if configured else None
    if configured_path is None:
        raise ValueError(f"set {ENVIRONMENT_VARIABLE} to an external run directory")
    if not configured_path.is_absolute():
        raise ValueError(
            f"{ENVIRONMENT_VARIABLE} must be an absolute path: {configured_path}"
        )
    candidate = configured_path.resolve(strict=False)
    if _inside(candidate, PROJECT_ROOT.resolve()):
        raise ValueError(f"artifact root must be outside the project: {candidate}")
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    if not candidate.is_dir() or not os.access(candidate, os.W_OK | os.X_OK):
        raise ValueError(f"artifact root is not a writable directory: {candidate}")
    return candidate


def artifact_directory(stage: str, run_id: str) -> Path:
    for label, value in (("stage", stage), ("run_id", run_id)):
        if not COMPONENT_PATTERN.fullmatch(value):
            raise ValueError(f"invalid artifact {label}: {value!r}")
    root = artifact_root()
    path = root / stage / run_id
    _reject_symlink_components(path, root)
    path.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path, root)
    return path


def reference_source_path(reference: dict, reference_path: Path | None = None) -> Path:
    """Resolve a capture reference without depending on the caller's directory."""
    source = Path(reference["source_index"])
    if not source.is_absolute():
        if reference.get("source_index_base") != "reference_directory" or reference_path is None:
            raise ValueError("relative source_index requires source_index_base=reference_directory and the reference file path")
        source = Path(reference_path).resolve(strict=True).parent / source
    return source.resolve(strict=True)
