"""Fearnhead (2006) exact-DP reference comparator.

Unlike PELT (``ruptures``), CBS (``DNAcopy``), or SMUCE (``stepR``), there
is **no widely-distributed standalone implementation** of Fearnhead's 2006
exact offline DP. The §5b "RJMCMC, Fearnhead's exact DP" slot is in the
manuscript's planned external comparator list, and the closest
reproducible reference is the same exact DP that BayesBreak itself
implements — Fearnhead (2006) §3 is in fact the algorithmic ancestor of
the BayesBreak forward/backward recursion (``prop:fb-duality``,
``thm:dp-correctness``).

This wrapper therefore exposes BayesBreak's own DP at a configuration
that matches the Fearnhead 2006 prior choice (geometric on the segment
count plus a length-aware cohesion) and clearly labels the result as a
*reference comparator*, not a re-implementation. The returned
:class:`BaselineResult.package` is ``"bayesbreak (fearnhead2006 config)"``
so downstream tables can flag the provenance honestly.

If you need a genuinely third-party Fearnhead-2006 implementation, the
options are: (i) compile Paul Fearnhead's original Fortran/MATLAB code
(not packaged); (ii) use the ``cpts`` slot of R's ``changepoint`` with
``method="SegNeigh"``, which is a segment-neighbourhood DP closely
related to Fearnhead 2006 for Gaussian blocks; or (iii) drive the JSFdS
2015 pruned-DP code referenced in ``rigaill2010pruned``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._types import BaselineResult


def run_fearnhead_exact(
    y: ArrayLike,
    *,
    family: str = "gaussian",
    k_max: int = 25,
    geometric_rate: float = 0.5,
    length_alpha: float = 0.0,
    **family_kwargs: Any,
) -> BaselineResult:
    r"""Fearnhead (2006) exact-DP reference comparator.

    Drives BayesBreak's own DP at the Fearnhead-2006 prior choice:

    - Geometric segment-count prior: ``p(k) ∝ (1 − r)^{k−1} · r``
      with ``r = geometric_rate``.
    - Optional length-aware cohesion ``g(ℓ) ∝ ℓ^{length_alpha}`` at the
      partition prior. ``length_alpha = 0`` gives an index-uniform
      partition (the canonical Fearnhead setting for change-in-mean).

    Parameters
    ----------
    y : 1-D array-like
        Observed sequence.
    family : str, default "gaussian"
        Forwarded to :func:`bayesbreak.make_bayesbreak`.
    k_max : int, default 25
        Segment-count cap.
    geometric_rate : float, default 0.5
        ``r`` in the geometric ``p(k)`` prior.
    length_alpha : float, default 0.0
        Exponent in the length-aware cohesion ``g(ℓ) = ℓ^{length_alpha}``.
        ``0.0`` is index-uniform; ``1.0`` is length-proportional.
    **family_kwargs
        Forwarded to the block-family constructor (e.g. ``nu``,
        ``rho2``, ``sigma2`` for Gaussian).

    Notes
    -----
    This is the §5b "Fearnhead exact DP" baseline slot. There is no
    standalone third-party Fearnhead-2006 implementation packaged on
    PyPI or CRAN; the closest reproducible reference is BayesBreak's
    own DP at the matching prior configuration. The
    :class:`BaselineResult.package` field labels this provenance.
    """
    from .. import make_bayesbreak  # noqa: PLC0415 - local to avoid circular import

    arr = np.asarray(y, dtype=float).ravel()
    n = int(arr.size)

    # Geometric prior on the segment count.
    r = float(geometric_rate)
    if not 0.0 < r < 1.0:
        raise ValueError("geometric_rate must be in (0, 1).")

    def _prior_k(k: int) -> float:
        return ((1.0 - r) ** max(0, int(k) - 1)) * r

    # Length-aware cohesion.
    if length_alpha == 0.0:
        length_prior = None
    else:
        a = float(length_alpha)

        def length_prior(d: float) -> float:  # type: ignore[no-redef]
            return float(d) ** a

    estimator = make_bayesbreak(
        family,
        k_max=int(k_max),
        prior_k=_prior_k,
        length_prior=length_prior,
        **family_kwargs,
    )
    X = np.arange(n, dtype=float).reshape(-1, 1)
    estimator.fit(X, arr)

    interior = [int(b) for b in estimator.map_boundaries_[1:-1] if 0 < int(b) < n]
    boundaries = np.asarray(sorted(set(interior)), dtype=np.intp)

    return BaselineResult(
        algorithm="fearnhead_exact",
        package="bayesbreak (fearnhead2006 config)",
        package_version="n/a",
        n=n,
        boundaries=boundaries,
        tuning={
            "family": family,
            "k_max": int(k_max),
            "geometric_rate": r,
            "length_alpha": float(length_alpha),
            **{k: repr(v) for k, v in family_kwargs.items()},
        },
        extra={
            "k_hat": int(estimator.k_map_),
            "log_evidence": float(estimator.log_evidence_),
            "note": (
                "Reference comparator: BayesBreak's own exact DP under the "
                "Fearnhead-2006 prior configuration. No standalone third-party "
                "Fearnhead-2006 implementation is packaged on PyPI/CRAN; see "
                "the module docstring for alternative sources."
            ),
        },
    )
