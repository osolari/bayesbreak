r"""Placeholder/verified-mode utilities for the real-data pipelines.

The report (§6 + handoff §10) requires real-data figures and tables to be
rendered as **placeholders** until the author has explicitly verified the
finalized pipeline output. This module supplies:

- :func:`is_verified_mode` — checks the ``--verified`` CLI flag or the
  ``BAYESBREAK_VERIFIED=1`` environment variable.
- :func:`overlay_placeholder_banner` — adds a translucent watermark to a
  matplotlib figure when the run is *not* verified.
- :func:`write_run_record` — writes a sidecar JSON next to the figure /
  table containing the verification status, raw-data hash, preprocessing
  hash, random seed, and timestamp.

The intent is that no real-data artifact ever leaves the pipeline without a
machine-readable record of whether it is a placeholder or a finalized run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_VERIFIED_ENV = "BAYESBREAK_VERIFIED"


def is_verified_mode(verified_flag: bool = False) -> bool:
    """Return True iff the caller explicitly opted into verified mode.

    Two activation paths: the CLI ``--verified`` flag or the
    ``BAYESBREAK_VERIFIED=1`` environment variable. Either disables the
    placeholder watermark and switches the sidecar JSON to ``verified=true``.
    """

    if verified_flag:
        return True
    return os.environ.get(_VERIFIED_ENV, "").strip().lower() in {"1", "true", "yes", "y"}


def overlay_placeholder_banner(fig, *, source: str, alpha: float = 0.35) -> None:
    """Add a diagonal "PLACEHOLDER" watermark to ``fig`` at the figure level.

    Used for unverified real-data figures so that downstream readers can
    visually distinguish placeholder pipeline output from finalized
    author-approved results. ``source`` is shown alongside the watermark
    (e.g., "downloaded", "simulated").
    """

    fig.text(
        0.5,
        0.5,
        f"PLACEHOLDER\n({source})",
        ha="center",
        va="center",
        rotation=30,
        fontsize=44,
        color="red",
        alpha=alpha,
        weight="bold",
        transform=fig.transFigure,
        zorder=100,
    )


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
    verified: bool,
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
    verified : bool
    extra : dict, optional
        Additional fields to merge into the record (e.g. raw-data hash,
        preprocessing hash, fit hyperparameters, random seed).
    """

    record = {
        "dataset": dataset,
        "source": source,
        "verified": bool(verified),
        "ran_at": time.time(),
        "figure_path": str(figure_path),
    }
    if extra:
        record.update(extra)
    out = figure_path.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str))
    return out
