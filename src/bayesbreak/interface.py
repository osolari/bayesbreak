"""Structural types for the block-evidence family interface.

The BayesBreak architecture cleanly separates:

- **Block evidence** — a family-specific integrated single-segment marginal
  likelihood plus optional moment numerators, indexed by a candidate block
  ``(i, j]``.
- **Partition DP** — a distribution-agnostic dynamic program over contiguous
  partitions that consumes the block evidences (:mod:`bayesbreak.dp`).

This module declares the protocols used across those two layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@runtime_checkable
class BlockEvidence(Protocol):
    """Protocol for a block-evidence primitive (§4 of the report).

    Implementations return the log integrated single-segment marginal
    likelihood and, optionally, moment numerators used by the Bayesian
    regression curve.
    """

    def log_evidence_table(self) -> FloatArray:
        """Return the ``(n+1, n+1)`` triangular log-block-evidence table."""

    def first_moment_table(self) -> FloatArray:
        """Return the ``(n+1, n+1)`` block first-moment numerator in linear scale."""

    def segment_posterior_mean(self, a: int, b: int) -> float:
        """Posterior mean of the segment parameter on block ``(a, b]``."""


@dataclass(frozen=True)
class SegmentationPosterior:
    """Dataclass capturing all posterior quantities produced by :mod:`dp`.

    Attributes
    ----------
    n : int
        Sequence length.
    k_max : int
        Maximum segment count considered.
    log_left, log_right : ndarray
        Sum-product forward / backward tables.
    log_posterior_k, posterior_k : ndarray
        Posterior over segment counts.
    log_evidence : float
        ``log P(y)`` (marginal likelihood).
    k_map : int
        ``argmax_k P(k | y)`` (posterior-mode segment count).
    boundary_marginals : ndarray
        ``P(b_i = 1 | y)`` for ``i = 1, ..., n-1``.
    boundary_location_posterior : ndarray
        ``P(t_p = h | y, k_map)`` for ``p = 1, ..., k_map - 1``.
    map_boundaries : list of int
        Joint MAP segmentation ``(0, t_1, ..., t_{k_map})``.
    bayes_curve_mean : ndarray or None
        Posterior mean of the latent piecewise-constant signal (optional).
    """

    n: int
    k_max: int
    log_left: FloatArray
    log_right: FloatArray
    log_posterior_k: FloatArray
    posterior_k: FloatArray
    log_evidence: float
    k_map: int
    boundary_marginals: FloatArray
    boundary_location_posterior: FloatArray
    map_boundaries: list[int]
    bayes_curve_mean: FloatArray | None = None
