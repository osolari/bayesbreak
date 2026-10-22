"""Bayesian-MCMC multiple-changepoint baseline via the R ``mcp`` package
(Lindeløv, JOSS 2020), driven through ``rpy2``.

We do not re-implement RJMCMC. The ``mcp`` package takes a formula-list
representation of linear segments and compiles JAGS code that samples
the posterior over changepoint locations and segment parameters via
MCMC. ``mcp`` is not strictly RJMCMC in the Green (1995) trans-dimensional
sense — it fixes the number of segments per formula — but it is the most
widely-used packaged Bayesian-MCMC baseline for multiple changepoints
and the closest practical comparator for the manuscript's §5b RJMCMC
slot. Selection of the number of changepoints across fits is done by
``mcp::loo`` (leave-one-out cross-validation); this wrapper exposes a
single fit at a user-specified segment count.

Setup
-----

Requires both ``rpy2`` (Python side) and the R packages ``mcp`` and
``rjags`` (which itself requires the JAGS binary). Install hints:

.. code-block:: bash

    pip install bayesbreak[baselines-r]
    # JAGS itself: macOS `brew install jags`; Ubuntu `apt install jags`.
    R -e 'install.packages(c("mcp","rjags"), repos="https://cloud.r-project.org")'

If ``rpy2`` or ``mcp`` is missing, :func:`run_rjmcmc` raises
``ImportError`` with a single readable hint.
"""

from __future__ import annotations

import importlib

import numpy as np
from numpy.typing import ArrayLike

from ._types import BaselineResult

_MCP_HINT = (
    "rpy2 + R packages 'mcp' and 'rjags' (with the JAGS binary) are required "
    "for the RJMCMC-style baseline. Install with `pip install bayesbreak[baselines-r]`, "
    "the system JAGS binary (`brew install jags` / `apt install jags`), and "
    '`R -e \'install.packages(c("mcp","rjags"), repos="https://cloud.r-project.org")\'`.'
)


def _load_rpy2():
    try:
        rpy2 = importlib.import_module("rpy2")
        robjects = importlib.import_module("rpy2.robjects")
        packages = importlib.import_module("rpy2.robjects.packages")
        numpy2ri = importlib.import_module("rpy2.robjects.numpy2ri")
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(_MCP_HINT) from exc
    return rpy2, robjects, packages, numpy2ri


def run_rjmcmc(
    y: ArrayLike,
    *,
    n_segments: int,
    n_iter: int = 3000,
    n_chains: int = 2,
    adapt: int = 1000,
    seed: int = 0,
) -> BaselineResult:
    """Bayesian MCMC multi-changepoint fit via ``mcp::mcp``.

    Builds the segment formula
    ``[y ~ 1, ~ 1, ..., ~ 1]`` (intercept-only per segment) with
    ``n_segments`` segments and runs MCMC. Posterior mean of each
    changepoint becomes the returned interior boundary.

    Parameters
    ----------
    y : 1-D array-like
        Observed sequence.
    n_segments : int
        Fixed number of segments (``n_segments - 1`` interior changepoints).
    n_iter : int, default 3000
        Posterior iterations per chain after adaptation.
    n_chains : int, default 2
        Number of MCMC chains.
    adapt : int, default 1000
        JAGS adaptation iterations.
    seed : int, default 0
        R-side seed.

    Notes
    -----
    ``mcp`` requires ``n_segments >= 1``; ``n_segments == 1`` returns an
    empty interior-boundary array. ``mcp`` is not RJMCMC: trans-dimensional
    selection across ``n_segments`` is done externally via ``mcp::loo``.
    For the §5b RJMCMC slot, run this wrapper at several candidate
    ``n_segments`` and compare by held-out predictive log-likelihood
    (the same scoring rule as
    :func:`bayesbreak.diagnostics.select_n_groups_by_holdout`).
    """
    if int(n_segments) < 1:
        raise ValueError("n_segments must be >= 1.")
    rpy2, robjects, packages, numpy2ri = _load_rpy2()
    arr = np.asarray(y, dtype=float).ravel()
    n = int(arr.size)

    try:
        mcp = packages.importr("mcp")
    except Exception as exc:  # pragma: no cover - env-specific
        raise ImportError(_MCP_HINT) from exc
    base = packages.importr("base")

    numpy2ri.activate()
    try:
        base.set_seed(int(seed))

        # Build a list of intercept-only formulas, one per segment.
        # The mcp DSL is: model = list(y ~ 1, ~ 1, ..., ~ 1), where the
        # first formula sets the response and each subsequent ~ 1 declares
        # a new segment with its own intercept.
        # We construct it in R because R's formula objects don't
        # round-trip cleanly through rpy2.
        formula_list = robjects.r(
            "list(" + ", ".join(["y ~ 1"] + ["~ 1"] * max(0, int(n_segments) - 1)) + ")"
        )
        # Build the dataframe.
        df = robjects.r["data.frame"](x=np.arange(n, dtype=float), y=arr)

        fit = mcp.mcp(
            model=formula_list,
            data=df,
            iter=int(n_iter),
            chains=int(n_chains),
            adapt=int(adapt),
        )

        # Posterior mean of each changepoint coordinate. mcp exposes them
        # as cp_1, cp_2, ... in the posterior; the public summary helper
        # returns a data.frame with rows per parameter.
        summary_df = base.as_data_frame(mcp.summary(fit))
        param_names = list(summary_df.rx2("name"))
        mean_col = np.asarray(summary_df.rx2("mean"))
        boundaries: list[int] = []
        for nm, mn in zip(param_names, mean_col, strict=False):
            if str(nm).startswith("cp_"):
                b = int(round(float(mn)))
                if 0 < b < n:
                    boundaries.append(b)
        boundaries = sorted(set(boundaries))
    finally:
        numpy2ri.deactivate()

    return BaselineResult(
        algorithm="rjmcmc",
        package="mcp",
        package_version=str(packages.utils_package_version("mcp"))
        if hasattr(packages, "utils_package_version")
        else "unknown",
        n=n,
        boundaries=np.asarray(boundaries, dtype=np.intp),
        tuning={
            "n_segments": int(n_segments),
            "n_iter": int(n_iter),
            "n_chains": int(n_chains),
            "adapt": int(adapt),
            "seed": int(seed),
        },
        extra={"engine": "mcp::mcp (JAGS backend)"},
    )
