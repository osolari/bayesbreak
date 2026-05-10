"""Array-CGH (copy-number) loader.

Real source: the ``ACGH`` dataset from CRAN package ``ecp``
(``https://github.com/cran/ecp/blob/master/data/ACGH.RData``), 2215 probes
across 43 individuals as floating-point log-2 copy-number ratios. We pull
the ``.RData`` blob via ``requests`` (no hash check — the upstream is a
mirror) and parse it with the ``rdata`` Python package.

The loader returns a multi-subject :class:`DatasetBundle` with ``y`` of
shape ``(n_probes, n_subjects)`` and ``sample_weight`` populated by
rolling-MAD-based per-probe inverse-variance precisions. Pass
``individuals=...`` to subset.

When the network or ``rdata`` is unavailable, the deterministic simulated
analog is returned. A single-subject CSV path is still accepted for users
with a local extract.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np

from . import DatasetBundle
from ._cache import banner, cache_dir, describe_fallback
from ._simulate import simulate_cgh

_ACGH_URL = "https://github.com/cran/ecp/raw/master/data/ACGH.RData"


def _rolling_mad(y: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling MAD-based per-position σ estimate, scaled to a Gaussian σ."""

    n = y.size
    out = np.zeros(n, dtype=float)
    half = max(1, window // 2)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = y[lo:hi]
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        out[i] = max(mad * 1.4826, 1e-3)
    return out


def _load_acgh_real(individuals: list[int] | None) -> DatasetBundle | None:
    """Try to fetch + parse ``ACGH.RData``; return None on any failure."""

    try:
        import rdata  # type: ignore[import-not-found]
        import requests  # type: ignore[import-not-found]
    except ImportError:
        return None

    cache = cache_dir() / "ACGH.RData"
    try:
        if cache.exists():
            blob = cache.read_bytes()
        else:
            r = requests.get(_ACGH_URL, timeout=30)
            if r.status_code != 200 or len(r.content) < 1000:
                return None
            blob = r.content
            cache.write_bytes(blob)
        parsed = rdata.parser.parse_file(io.BytesIO(blob))
        converted = rdata.conversion.convert(parsed)
        acgh = converted["ACGH"]
        data = np.asarray(acgh["data"].values, dtype=float)  # (n_probes, n_subj)
    except Exception:  # pragma: no cover - network / parsing edge cases
        return None

    n_probes, n_subj = data.shape
    if individuals is not None:
        idx = [int(i) for i in individuals if 0 <= int(i) < n_subj]
        if not idx:
            return None
        y = data[:, idx]
    else:
        y = data

    # Per-probe heteroscedastic precision: inverse rolling-MAD variance,
    # averaged across selected subjects (heteroscedastic Gaussian §replicates).
    sigma = np.stack([_rolling_mad(y[:, s]) for s in range(y.shape[1])], axis=1)
    sigma_avg = float(np.median(sigma))
    sample_weight = (sigma_avg / sigma) ** 2  # shape (n, S)

    banner(f"cgh: loaded n_probes={n_probes}, n_subj={y.shape[1]} from cran/ecp ACGH.RData.")
    return DatasetBundle(
        X=np.arange(n_probes, dtype=float).reshape(-1, 1),
        y=y if y.shape[1] > 1 else y[:, 0],
        sample_weight=sample_weight if y.shape[1] > 1 else sample_weight[:, 0],
        true_boundaries=[],
        name="cgh",
        source="downloaded",
        description=(
            f"Array-CGH log2-ratios from cran/ecp ACGH (n_probes={n_probes}, "
            f"n_subj={y.shape[1]})."
        ),
        metadata={"url": _ACGH_URL, "individuals": individuals},
    )


def load_cgh(
    *,
    simulated: bool = False,
    csv_path: str | pathlib.Path | None = None,
    individuals: list[int] | None = None,
) -> DatasetBundle:
    """Load an array-CGH log2-ratio sequence.

    Parameters
    ----------
    simulated : bool, default False
        Force the deterministic simulated analog regardless of network or
        ``csv_path`` availability.
    csv_path : str or Path or None
        Optional path to a single-column CSV of log2-ratio values. When
        provided and parseable, returns a single-subject bundle.
    individuals : list of int or None
        Subset of subject column indices to keep. ``None`` returns all 43.

    Returns
    -------
    DatasetBundle
        ``y`` is ``(n_probes,)`` for single-subject and ``(n_probes, n_subjects)``
        for multi-subject. ``sample_weight`` is the per-probe inverse-variance
        precision matching ``y.shape``.
    """

    if simulated:
        return DatasetBundle.from_dict(simulate_cgh())

    # Real ACGH first.
    bundle = _load_acgh_real(individuals)
    if bundle is not None:
        return bundle

    # CSV fallback path.
    if csv_path is not None:
        path = pathlib.Path(csv_path).expanduser()
        if path.exists():
            try:
                raw = np.loadtxt(path, delimiter=",", ndmin=1)
                y = np.asarray(raw, dtype=float).ravel()
                y = y[np.isfinite(y)]
                if y.size >= 50:
                    sigma = _rolling_mad(y)
                    sigma_avg = float(np.median(sigma))
                    banner(f"cgh: loaded {y.size} points from {path}")
                    return DatasetBundle(
                        X=np.arange(y.size, dtype=float).reshape(-1, 1),
                        y=y,
                        sample_weight=(sigma_avg / sigma) ** 2,
                        true_boundaries=[],
                        name="cgh",
                        source="downloaded",
                        description=f"User-provided array-CGH log2-ratios (n={y.size}) from {path}.",
                        metadata={"csv_path": str(path)},
                    )
            except Exception as exc:
                describe_fallback("cgh", f"csv parse failed: {exc}")
        else:
            describe_fallback("cgh", f"{path} not found")

    describe_fallback("cgh", "real download unavailable")
    return DatasetBundle.from_dict(simulate_cgh())
