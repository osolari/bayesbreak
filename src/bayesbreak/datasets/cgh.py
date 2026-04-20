"""Array-CGH (copy-number) loader.

There is no stable, small, license-clean public mirror of the Lai et al.
(2005) array-CGH profile that we can reliably redistribute through ``pooch``.
This loader therefore returns the deterministic simulated CGH analog by
default, and accepts a user-provided ``csv_path`` when real data is available
locally (e.g. extracted from the ``ecp`` R package or an institutional
archive).
"""

from __future__ import annotations

import pathlib

import numpy as np

from . import DatasetBundle
from ._cache import banner, describe_fallback
from ._simulate import simulate_cgh


def load_cgh(
    *,
    simulated: bool = False,
    csv_path: str | pathlib.Path | None = None,
) -> DatasetBundle:
    """Load an array-CGH log2-ratio sequence.

    Parameters
    ----------
    simulated : bool, default False
        Force the deterministic simulated analog regardless of ``csv_path``.
    csv_path : str or Path or None
        Optional path to a single-column CSV of log2-ratio values. When
        provided and parseable, the real data is returned; otherwise the
        simulated analog is used.
    """

    if simulated or csv_path is None:
        if not simulated:
            describe_fallback(
                "cgh",
                "no csv_path provided; pass csv_path=... to load real data",
            )
        return DatasetBundle.from_dict(simulate_cgh())

    path = pathlib.Path(csv_path).expanduser()
    if not path.exists():
        describe_fallback("cgh", f"{path} not found")
        return DatasetBundle.from_dict(simulate_cgh())

    try:
        raw = np.loadtxt(path, delimiter=",", ndmin=1)
    except Exception as exc:
        describe_fallback("cgh", f"parse failed: {exc}")
        return DatasetBundle.from_dict(simulate_cgh())

    y = np.asarray(raw, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 50:
        describe_fallback("cgh", f"only {y.size} valid rows after filtering")
        return DatasetBundle.from_dict(simulate_cgh())

    banner(f"cgh: loaded {y.size} points from {path}")
    return DatasetBundle(
        X=np.arange(y.size, dtype=float).reshape(-1, 1),
        y=y,
        sample_weight=None,
        true_boundaries=[],
        name="cgh",
        source="downloaded",
        description=f"User-provided array-CGH log2-ratios (n={y.size}) from {path}.",
        metadata={"csv_path": str(path)},
    )
