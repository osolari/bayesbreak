"""Circular Binary Segmentation (CBS) via the Bioconductor ``DNAcopy``
package, driven through ``rpy2``.

We do not re-implement CBS. The wrapper installs ``DNAcopy`` lazily inside
the active R session if it is not already available, drives
``DNAcopy::segment`` on a single-sample log-2 ratio vector, and returns the
detected boundaries as a :class:`~bayesbreak.baselines._types.BaselineResult`.

CBS is the standard array-CGH baseline (the BayesBreak §6 ``fig:cgh`` /
``tab:real_cgh`` planned comparison) and is the algorithm run by the
``DNAcopy::coriell`` example referenced in §6 of the new manuscript.

Setup
-----

Requires both ``rpy2`` (Python side) and the R package ``DNAcopy``
(Bioconductor). Install hints:

.. code-block:: bash

    pip install bayesbreak[baselines-r]
    R -e 'if (!requireNamespace("BiocManager")) install.packages("BiocManager"); BiocManager::install("DNAcopy")'

If either ``rpy2`` or ``DNAcopy`` is missing, :func:`run_cbs` raises
``ImportError`` with a single readable message.
"""

from __future__ import annotations

import importlib

import numpy as np
from numpy.typing import ArrayLike

from ._types import BaselineResult

_RPY2_HINT = (
    "rpy2 + Bioconductor DNAcopy are required for the CBS wrapper. "
    "Install with `pip install bayesbreak[baselines-r]` and "
    "`R -e 'BiocManager::install(\"DNAcopy\")'`."
)


def _load_rpy2():
    try:
        rpy2 = importlib.import_module("rpy2")
        robjects = importlib.import_module("rpy2.robjects")
        packages = importlib.import_module("rpy2.robjects.packages")
        numpy2ri = importlib.import_module("rpy2.robjects.numpy2ri")
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(_RPY2_HINT) from exc
    return rpy2, robjects, packages, numpy2ri


def run_cbs(
    y: ArrayLike,
    *,
    chromosome: ArrayLike | None = None,
    position: ArrayLike | None = None,
    sample_id: str = "sample",
    alpha: float = 0.01,
    nperm: int = 10_000,
    undo_splits: str = "none",
    smooth: bool = False,
    seed: int = 0,
) -> BaselineResult:
    """Run CBS (Olshen et al. 2004) on a 1-D log-2 ratio profile.

    Parameters
    ----------
    y : 1-D array-like
        Log-2 ratio values along the genome.
    chromosome, position : array-like or None
        Optional per-probe chromosome and position columns. When ``None``
        the wrapper treats the whole sequence as one chromosome with
        positions ``0, 1, ..., n-1``.
    sample_id : str
        ``DNAcopy`` sample identifier (single sample).
    alpha : float, default 0.01
        Significance level for the CBS permutation test (``segment(alpha=)``).
    nperm : int, default 10_000
        Number of permutations.
    undo_splits : {"none", "prune", "sdundo"}, default "none"
        ``DNAcopy::segment(undo.splits=)`` argument.
    smooth : bool, default False
        Whether to call ``DNAcopy::smooth.CNA`` before ``segment``.
    seed : int, default 0
        R-side RNG seed for the permutation test (``set.seed`` is called
        before ``segment``).
    """
    rpy2, robjects, packages, numpy2ri = _load_rpy2()

    arr = np.asarray(y, dtype=float).ravel()
    n = int(arr.size)

    chrom = np.ones(n, dtype=int) if chromosome is None else np.asarray(chromosome).ravel()
    pos = np.arange(n, dtype=int) if position is None else np.asarray(position).ravel()
    if chrom.size != n or pos.size != n:
        raise ValueError("`chromosome` and `position` must match length of `y`.")

    try:
        dnacopy = packages.importr("DNAcopy")
    except Exception as exc:  # pragma: no cover - env-specific
        raise ImportError(_RPY2_HINT) from exc
    base = packages.importr("base")

    numpy2ri.activate()
    try:
        base.set_seed(int(seed))
        cna = dnacopy.CNA(
            arr,
            chrom,
            pos,
            **{"data.type": "logratio", "sampleid": sample_id},
        )
        if smooth:
            cna = dnacopy.smooth_CNA(cna)
        segres = dnacopy.segment(
            cna,
            alpha=float(alpha),
            nperm=int(nperm),
            **{"undo.splits": undo_splits, "verbose": 0},
        )
        output = robjects.r["as.data.frame"](segres.rx2("output"))
        end_col = np.asarray(output.rx2("loc.end")).astype(int)
    finally:
        numpy2ri.deactivate()

    # ``loc.end`` is the **position** of the last probe in each segment.
    # Map back to indices and drop the final endpoint to get interior
    # boundaries on the 1-D index axis.
    end_positions = sorted(int(v) for v in end_col)
    interior: list[int] = []
    for end_pos in end_positions[:-1]:  # drop final segment terminator
        # Find the array index whose ``position`` equals ``end_pos``.
        # ``pos`` is monotone, so ``searchsorted`` works.
        idx = int(np.searchsorted(pos, end_pos, side="right"))
        if 0 < idx < n:
            interior.append(idx)
    boundaries = np.asarray(sorted(set(interior)), dtype=np.intp)

    return BaselineResult(
        algorithm="cbs",
        package="DNAcopy",
        package_version=str(packages.utils_package_version("DNAcopy"))
        if hasattr(packages, "utils_package_version")
        else "unknown",
        n=n,
        boundaries=boundaries,
        tuning={
            "alpha": float(alpha),
            "nperm": int(nperm),
            "undo_splits": undo_splits,
            "smooth": bool(smooth),
            "seed": int(seed),
        },
        extra={"sample_id": sample_id, "n_segments": int(end_col.size)},
    )
