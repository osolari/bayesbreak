"""Baseline changepoint algorithms (frequentist comparators).

We do **not** re-implement any of these algorithms in BayesBreak. Each
wrapper calls the canonical upstream library and normalizes its output
into a :class:`BaselineResult`. Algorithm coverage matches the new
manuscript's planned baseline list in §6 (paragraphs 5-A1, 6-E3) and the
``ruptures`` / ``changepoint`` positioning paragraph (1-D2/G-1):

- ``pelt``, ``optimal_partitioning``, ``binary_segmentation``,
  ``wild_binary_segmentation`` — via the ``ruptures`` Python package
  (Truong, Oudre & Vayatis 2018).
- ``cbs`` — via the Bioconductor ``DNAcopy`` package (Olshen et al. 2004),
  driven through ``rpy2``.
- ``smuce`` — via the CRAN ``stepR`` package
  (Frick, Munk & Sieling 2014), driven through ``rpy2``.
- ``rjmcmc`` (alias ``mcp``) — Bayesian-MCMC multi-changepoint baseline
  via the R ``mcp`` package (Lindeløv 2020), driven through ``rpy2``
  with a JAGS backend.
- ``fearnhead_exact`` (alias ``fearnhead``) — reference-comparator
  wrapper around BayesBreak's own exact DP at the Fearnhead-2006 prior
  configuration (geometric ``p(k)`` + optional length-aware cohesion).
  No standalone third-party Fearnhead-2006 implementation is packaged;
  see the module docstring for alternative sources.

Upstream dependencies are loaded lazily; missing packages raise
``ImportError`` with a clear hint pointing to the right ``pip``/``R``
install command.

Example
-------

>>> from bayesbreak.baselines import segment_with
>>> res = segment_with("pelt", y, penalty=10.0)            # doctest: +SKIP
>>> res.boundaries                                         # doctest: +SKIP
array([100, 220, ...])

The :class:`BaselineResult` records the upstream package name and version
in ``res.package`` / ``res.package_version`` so the run can be reproduced
without re-reading the calling code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._types import BaselineResult
from .cbs import run_cbs
from .fearnhead_exact import run_fearnhead_exact
from .rjmcmc import run_rjmcmc
from .ruptures_wrapper import run_binseg, run_dynp, run_pelt, run_wbs
from .smuce import run_smuce

_REGISTRY: dict[str, Callable[..., BaselineResult]] = {
    "pelt": run_pelt,
    "optimal_partitioning": run_dynp,
    "op": run_dynp,
    "dynp": run_dynp,
    "binary_segmentation": run_binseg,
    "binseg": run_binseg,
    "bs": run_binseg,
    "wild_binary_segmentation": run_wbs,
    "wbs": run_wbs,
    "cbs": run_cbs,
    "smuce": run_smuce,
    "rjmcmc": run_rjmcmc,
    "mcp": run_rjmcmc,
    "fearnhead_exact": run_fearnhead_exact,
    "fearnhead": run_fearnhead_exact,
}


def available_algorithms() -> list[str]:
    """Canonical algorithm names (drops short aliases)."""
    return [
        "pelt",
        "optimal_partitioning",
        "binary_segmentation",
        "wild_binary_segmentation",
        "cbs",
        "smuce",
        "rjmcmc",
        "fearnhead_exact",
    ]


def segment_with(algorithm: str, y: Any, **kwargs: Any) -> BaselineResult:
    """Dispatch to the named baseline.

    Parameters
    ----------
    algorithm : str
        One of :func:`available_algorithms` or a short alias (``"bs"``,
        ``"wbs"``, ``"dynp"``).
    y : array-like
        1-D or 2-D signal (CBS requires 1-D).
    **kwargs
        Forwarded to the underlying wrapper (see e.g. :func:`run_pelt`).
    """
    key = algorithm.lower().replace("-", "_")
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown baseline algorithm {algorithm!r}; available: {sorted(set(_REGISTRY))}"
        )
    return _REGISTRY[key](y, **kwargs)


__all__ = [
    "BaselineResult",
    "available_algorithms",
    "run_binseg",
    "run_cbs",
    "run_dynp",
    "run_fearnhead_exact",
    "run_pelt",
    "run_rjmcmc",
    "run_smuce",
    "run_wbs",
    "segment_with",
]
