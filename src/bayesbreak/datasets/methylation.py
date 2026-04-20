"""CpG methylation loader.

We do not ship a public real-data mirror for methylation: the tracks we'd
like to use (UCSC CpG atlas subsets, ENCODE) are heavy and have gated
licensing. The loader therefore returns the deterministic simulated analog by
default, with the option to pass a user-provided CSV via the ``csv_path``
parameter when real data is available locally.
"""

from __future__ import annotations

import pathlib

import numpy as np

from . import DatasetBundle
from ._cache import banner, describe_fallback
from ._simulate import simulate_methylation


def load_methylation(
    *,
    simulated: bool = False,
    csv_path: str | pathlib.Path | None = None,
) -> DatasetBundle:
    """Load a CpG methylation sequence (``y_i \u2208 (0, 1)``).

    Parameters
    ----------
    simulated : bool, default False
        Force the deterministic simulated analog and ignore ``csv_path``.
    csv_path : str or Path or None
        Optional path to a single-column CSV of methylation fractions. Values
        outside ``(0, 1)`` are dropped; if fewer than 50 valid rows remain
        we fall back to the simulation.
    """

    if simulated or csv_path is None:
        if not simulated:
            describe_fallback(
                "methylation",
                "no csv_path provided; pass csv_path=... to load real data",
            )
        return DatasetBundle.from_dict(simulate_methylation())

    path = pathlib.Path(csv_path).expanduser()
    if not path.exists():
        describe_fallback("methylation", f"{path} not found")
        return DatasetBundle.from_dict(simulate_methylation())

    try:
        raw = np.loadtxt(path, delimiter=",", ndmin=1)
    except Exception as exc:
        describe_fallback("methylation", f"parse failed: {exc}")
        return DatasetBundle.from_dict(simulate_methylation())

    y = np.asarray(raw, dtype=float).ravel()
    y = y[np.isfinite(y) & (y > 0.0) & (y < 1.0)]
    if y.size < 50:
        describe_fallback("methylation", f"only {y.size} valid rows after filtering")
        return DatasetBundle.from_dict(simulate_methylation())

    banner(f"methylation: loaded {y.size} points from {path}")
    return DatasetBundle(
        X=np.arange(y.size, dtype=float).reshape(-1, 1),
        y=y,
        sample_weight=None,
        true_boundaries=[],
        name="methylation",
        source="downloaded",
        description=f"User-provided methylation fractions (n={y.size}) from {path}.",
        metadata={"csv_path": str(path)},
    )
