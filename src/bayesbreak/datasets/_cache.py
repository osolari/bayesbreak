"""Pooch-based dataset cache.

``pooch`` is an optional dependency (it ships in the ``bayesbreak[datasets]``
extra). When it is not importable, loaders gracefully fall back to the
simulated analogs in :mod:`bayesbreak.datasets._simulate`.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any


def cache_dir() -> pathlib.Path:
    """Return the directory used to cache downloaded datasets.

    Defaults to ``~/.cache/bayesbreak``; override via the
    ``BAYESBREAK_DATA`` environment variable.
    """

    env = os.environ.get("BAYESBREAK_DATA")
    if env:
        path = pathlib.Path(env).expanduser()
    else:
        path = pathlib.Path.home() / ".cache" / "bayesbreak"
    path.mkdir(parents=True, exist_ok=True)
    return path


def try_fetch(*, url: str, known_hash: str | None, fname: str) -> pathlib.Path | None:
    """Try to fetch ``url`` with pooch; return ``None`` on any failure.

    Parameters
    ----------
    url : str
        Remote URL of the file.
    known_hash : str or None
        SHA256 hash (``sha256:...``) to verify against, or ``None`` to skip.
    fname : str
        Local file name inside the cache directory.
    """

    try:
        import pooch
    except ImportError:
        return None

    try:
        path = pooch.retrieve(
            url=url,
            known_hash=known_hash,
            path=str(cache_dir()),
            fname=fname,
            progressbar=False,
        )
        return pathlib.Path(path)
    except Exception:  # pragma: no cover - network / hash failures
        return None


def banner(msg: str) -> None:
    """Print a visible one-line banner announcing dataset provenance."""

    print(f"[bayesbreak.datasets] {msg}", flush=True)


def describe_fallback(name: str, reason: str) -> None:
    banner(
        f"{name}: falling back to simulated analog ({reason}). "
        "Install the 'datasets' extra and retry for the real dataset."
    )


def _ensure_loaded(obj: Any) -> Any:
    """Identity helper used by loaders that want a single place to hook tests."""

    return obj
