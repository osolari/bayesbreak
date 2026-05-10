r"""Pickle-based fit cache for the real-data figures.

Refitting CGH (2215 × 43), SPX (~2200 daily bars, k_max=50), or methylation
(1904 CpGs with quadrature) costs minutes; re-rendering the *figure* takes
seconds. The cache stores the fitted estimator under
``~/.cache/bayesbreak/fitcache/`` (override via ``BAYESBREAK_FITCACHE``)
so that tweaking only the plotting code hits the cache instead of refitting.

Cache key: SHA-256 of the response array + sample-weight array + sorted
estimator parameters + bayesbreak version. Mismatch → invalidate, refit,
overwrite. ``BAYESBREAK_REFIT=1`` (or any non-empty value) bypasses the
cache and forces a fresh fit.

Important: cache files live **outside** the repository tree. They contain
duplicated raw arrays and would balloon ``git`` history.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__


def _project_root() -> Path:
    """Return the BayesBreak project root (one level above ``src/bayesbreak``)."""

    return Path(__file__).resolve().parents[3]


def fitcache_dir() -> Path:
    """Return the directory used to cache fitted estimators.

    Defaults to ``<repo-root>/.cache/fitcache/`` (gitignored). Override via
    the ``BAYESBREAK_FITCACHE`` environment variable.
    """

    env = os.environ.get("BAYESBREAK_FITCACHE")
    if env:
        path = Path(env).expanduser()
    else:
        path = _project_root() / ".cache" / "fitcache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _stable_param_repr(params: dict[str, Any]) -> str:
    """Deterministic string representation of an sklearn-style params dict."""

    def _repr(v: Any) -> str:
        if isinstance(v, np.ndarray):
            return hashlib.sha256(v.astype(np.float64, copy=False).tobytes()).hexdigest()[:16]
        if callable(v):
            return f"<callable:{v.__module__}.{getattr(v, '__qualname__', v.__name__)}>"
        if isinstance(v, str | int | float | bool | type(None)):
            return repr(v)
        return repr(v)

    return json.dumps({k: _repr(v) for k, v in sorted(params.items())}, sort_keys=True)


def _hash_array(arr: np.ndarray | None) -> str:
    if arr is None:
        return ""
    return hashlib.sha256(np.asarray(arr).astype(np.float64, copy=False).tobytes()).hexdigest()


def cache_key(
    *,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    params: dict[str, Any],
    extra: str = "",
) -> str:
    """Compute a deterministic cache key for one fit."""

    h = hashlib.sha256()
    h.update(_hash_array(y).encode())
    h.update(b"|")
    h.update(_hash_array(sample_weight).encode())
    h.update(b"|")
    h.update(_stable_param_repr(params).encode())
    h.update(b"|")
    h.update(extra.encode())
    h.update(b"|")
    h.update(__version__.encode())
    return h.hexdigest()


def _refit_forced() -> bool:
    return bool(os.environ.get("BAYESBREAK_REFIT", "").strip())


_SLIM_DROP_ATTRS_REPLICATES = (
    # subject_states_ holds 43 subjects × full lA0/A1 tables (~80 MB each for
    # n=2215 → ~1.7 GB total). The figure renderer only reads pooled
    # quantities; the per-subject diagnostics can be regenerated cheaply if
    # ever needed.
    "subject_states_",
)


def _slim_replicates_for_cache(rep: Any) -> Any:
    """Drop heavyweight per-subject tables before pickling a replicates fit."""

    for attr in _SLIM_DROP_ATTRS_REPLICATES:
        if hasattr(rep, attr):
            try:
                setattr(rep, attr, None)
            except Exception:  # pragma: no cover
                pass
    return rep


def _slim_for_cache(obj: Any) -> Any:
    """Strip large incidental fields before pickling."""

    cls_name = type(obj).__name__
    if cls_name == "SharedBoundaryReplicatesSegmenter":
        return _slim_replicates_for_cache(obj)
    if isinstance(obj, dict):
        return {k: _slim_for_cache(v) for k, v in obj.items()}
    return obj


def _resolve_cache_path(name: str | Path) -> Path:
    """Resolve ``name`` to an absolute path under :func:`fitcache_dir`.

    Bare names (no separator) go under the cache dir; absolute paths are
    used as-is. Old call sites that passed ``outdir / "...fit.pkl"`` are
    redirected to the cache dir using only the file basename, so callers
    don't need updating.
    """

    p = Path(name)
    if p.is_absolute() or len(p.parts) > 1:
        # Redirect: use only the basename under the cache dir.
        return fitcache_dir() / p.name
    return fitcache_dir() / p


def fit_or_load(
    cache_path: str | Path,
    key: str,
    fit_fn: Callable[[], Any],
) -> Any:
    """Return a cached fit if its key matches, otherwise call ``fit_fn`` and persist.

    Cache is a single ``.pkl`` file containing ``{"key": <sha>, "fit": <obj>}``.
    Pickle is acceptable here because the figure pipeline runs only inside
    the user's own checkout — no untrusted input.
    """

    resolved = _resolve_cache_path(cache_path)
    name = resolved.name

    if not _refit_forced() and resolved.exists():
        try:
            with resolved.open("rb") as fh:
                payload = pickle.load(fh)
            if payload.get("key") == key:
                print(f"[fitcache] hit: {name}", flush=True)
                return payload["fit"]
            print(f"[fitcache] miss (stale): {name}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[fitcache] miss (load failed: {exc}): {name}", flush=True)
    elif _refit_forced():
        print(f"[fitcache] BAYESBREAK_REFIT set; forcing refit: {name}", flush=True)
    else:
        print(f"[fitcache] miss (no cache): {name}", flush=True)

    fit = fit_fn()
    slim = _slim_for_cache(fit)
    with resolved.open("wb") as fh:
        pickle.dump({"key": key, "fit": slim}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = resolved.stat().st_size / 1024 / 1024
    print(f"[fitcache] stored: {name} ({size_mb:.1f} MB)", flush=True)
    return fit
