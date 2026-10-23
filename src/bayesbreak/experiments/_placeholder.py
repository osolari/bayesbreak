r"""Provenance + hashing helpers for the real-data pipelines.

Each real-data figure writes a sidecar JSON next to it
(``<figure>.json``) recording the dataset, the source
(``"downloaded"`` vs. the deterministic ``"simulated"`` analog), a
SHA-256 hash of the response array, and any extra metadata the figure
script wants to capture (fit hyperparameters, DP diagnostics, …).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def hash_array(arr: np.ndarray | None) -> str:
    """SHA-256 hex digest of an array's bytes; ``''`` if ``arr is None``."""

    if arr is None:
        return ""
    a = np.asarray(arr).astype(np.float64, copy=False)
    return hashlib.sha256(a.tobytes()).hexdigest()


def hash_file(path: Path | str) -> str:
    """SHA-256 of a file's bytes (returns ``''`` if missing)."""

    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_record(
    figure_path: Path,
    *,
    dataset: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a sidecar ``<figure>.json`` describing the run.

    Parameters
    ----------
    figure_path : Path
        Path to the figure file (e.g. ``fig6_welllog.pdf``); the JSON sidecar
        is written next to it as ``<stem>.json``.
    dataset : str
    source : str
        ``"downloaded"`` or ``"simulated"``.
    extra : dict, optional
        Additional fields to merge into the record (e.g. raw-data hash,
        preprocessing hash, fit hyperparameters, random seed).
    """

    # Store a repo-relative path so the sidecar is portable across machines.
    abs_path = Path(figure_path).resolve()
    # Walk up to find the repo root (contains pyproject.toml).
    candidate = abs_path.parent
    repo_root: Path | None = None
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            repo_root = candidate
            break
        candidate = candidate.parent
    figure_path_str = (
        str(abs_path.relative_to(repo_root)) if repo_root is not None else str(abs_path)
    )

    record = {
        "dataset": dataset,
        "source": source,
        "ran_at": time.time(),
        "figure_path": figure_path_str,
    }
    if extra:
        record.update(extra)
    out = figure_path.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    return out
