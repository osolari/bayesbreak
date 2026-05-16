"""SMUCE (SiMultaneous MUltiscale Changepoint Estimator; Frick, Munk & Sieling 2014)
via the R package ``stepR``, driven through ``rpy2``.

We do not re-implement SMUCE. The wrapper loads ``stepR`` lazily inside
the active R session if it is available, drives ``stepR::stepFit`` (or
``stepR::smuceR`` on newer releases) on a 1-D numeric sequence, and
returns the detected boundaries as a
:class:`~bayesbreak.baselines._types.BaselineResult`.

SMUCE is the planned multiscale baseline referenced in §5b limitations
and in the §6 planned external-comparator pass.

Setup
-----

Requires both ``rpy2`` (Python side) and the R package ``stepR``. Install
hints:

.. code-block:: bash

    pip install bayesbreak[baselines-r]
    R -e 'install.packages("stepR", repos="https://cloud.r-project.org")'

If either ``rpy2`` or ``stepR`` is missing, :func:`run_smuce` raises
``ImportError`` with a single readable message.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._types import BaselineResult

_STEPR_HINT = (
    "rpy2 + R package stepR are required for the SMUCE wrapper. "
    "Install with `pip install bayesbreak[baselines-r]` and "
    '`R -e \'install.packages("stepR", repos="https://cloud.r-project.org")\'`.'
)


def _load_rpy2():
    try:
        rpy2 = importlib.import_module("rpy2")
        robjects = importlib.import_module("rpy2.robjects")
        packages = importlib.import_module("rpy2.robjects.packages")
        numpy2ri = importlib.import_module("rpy2.robjects.numpy2ri")
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(_STEPR_HINT) from exc
    return rpy2, robjects, packages, numpy2ri


def run_smuce(
    y: ArrayLike,
    *,
    alpha: float = 0.05,
    family: str = "gauss",
    sd: float | None = None,
    seed: int = 0,
) -> BaselineResult:
    """Run SMUCE (Frick, Munk & Sieling 2014) on a 1-D numeric sequence.

    Parameters
    ----------
    y : 1-D array-like
        Observed sequence.
    alpha : float, default 0.05
        Confidence level passed to ``stepR::stepFit(alpha=)`` /
        ``stepR::smuceR(alpha=)``. SMUCE controls the probability of any
        spurious changepoint by this number.
    family : {"gauss", "gaussvar", "poisson", "binomial"}, default "gauss"
        Distributional family passed to ``stepR``.
    sd : float or None, default None
        Optional residual standard deviation for ``family="gauss"``.
        ``None`` lets ``stepR`` estimate it via its default MAD-based
        routine.
    seed : int, default 0
        R-side seed set before fitting (the SMUCE optimizer is deterministic
        once data and ``alpha`` are fixed, but multiple calls share an RNG
        state we want to pin).
    """
    rpy2, robjects, packages, numpy2ri = _load_rpy2()

    arr = np.asarray(y, dtype=float).ravel()
    n = int(arr.size)

    try:
        stepr = packages.importr("stepR")
    except Exception as exc:  # pragma: no cover - env-specific
        raise ImportError(_STEPR_HINT) from exc
    base = packages.importr("base")

    numpy2ri.activate()
    try:
        base.set_seed(int(seed))

        # ``stepR`` exposes ``stepFit`` in modern releases and ``smuceR`` in
        # older ones. Prefer ``stepFit``; fall back to ``smuceR``.
        if hasattr(stepr, "stepFit"):
            kwargs: dict[str, Any] = {"alpha": float(alpha), "family": family}
            if sd is not None and family == "gauss":
                kwargs["sd"] = float(sd)
            fit = stepr.stepFit(arr, **kwargs)
            engine = "stepFit"
        else:  # pragma: no cover - exercised only on older stepR releases
            fit = stepr.smuceR(arr, alpha=float(alpha))
            engine = "smuceR"

        # ``stepFit`` returns an ``stepfit`` object; ``$rightIndex`` lists
        # the rightmost index of each detected segment in 1-based R indexing.
        as_df = robjects.r["as.data.frame"]
        df = as_df(fit)
        right_index = np.asarray(df.rx2("rightIndex")).astype(int)
    finally:
        numpy2ri.deactivate()

    # Convert R's 1-based right-index endpoints to 0-based interior
    # boundaries on the Python index axis. Drop the final segment terminator
    # (which equals ``n`` and is not an interior boundary).
    interior = sorted({int(r) for r in right_index if 0 < int(r) < n})
    boundaries = np.asarray(interior, dtype=np.intp)

    return BaselineResult(
        algorithm="smuce",
        package="stepR",
        package_version=str(packages.utils_package_version("stepR"))
        if hasattr(packages, "utils_package_version")
        else "unknown",
        n=n,
        boundaries=boundaries,
        tuning={
            "alpha": float(alpha),
            "family": family,
            "sd": sd,
            "seed": int(seed),
            "engine": engine,
        },
    )
