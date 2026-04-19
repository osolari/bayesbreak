r"""Dynamic-programming recursions over contiguous partitions.

This module hosts the distribution-agnostic segmentation engine. Given a
triangular array of log block evidences ``log_block_evidence[i, j] = log A^0_{ij}``
(as defined in the report, eq. :eq:`problem-block-evidence`), it computes:

- the **sum-product** forward / backward tables (§4.3),
- the **segment-count posterior** :math:`P(k\mid y)` (Prop. ``posterior-k``),
- the **boundary-event marginal** :math:`P(b_i=1\mid y,k)` (§4.3),
- the **joint MAP segmentation** via max-sum DP with backtracking (§4.4),
- the **Bayesian regression curve** via the $O(n^2)$ difference-array trick.

All recursions run in log-space for numerical stability. Distribution-specific
code lives in :mod:`bayesbreak.families`; this module is deliberately unaware
of the block model.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .utils import log_binom, logsumexp

FloatArray = NDArray[np.floating]


# -----------------------------------------------------------------------------
# Sum-product recursions
# -----------------------------------------------------------------------------


def forward_backward(
    log_block_evidence: FloatArray, n: int, k_max: int
) -> tuple[FloatArray, FloatArray]:
    """Compute the forward (prefix) and backward (suffix) sum-product tables.

    Let ``la[i, j] = log A^0_{ij}`` for ``0 <= i < j <= n``. Then

    .. math::
        L[k, j] = \\log\\sum_{t \\in \\mathcal{T}_{k, j}}
                  \\prod_{q=1}^k A^0_{t_{q-1} t_q},
        \\quad
        R[k, i] = \\log\\sum_{t \\in \\mathcal{T}_{k, i}^{\\text{suf}}}
                  \\prod_{q=1}^k A^0_{t_{q-1} t_q}.

    Parameters
    ----------
    log_block_evidence : ndarray of shape (n+1, n+1)
        Upper-triangular log block evidences ``la[i, j]`` (``-inf`` when ``i >= j``).
    n : int
        Sequence length.
    k_max : int
        Maximum number of segments.

    Returns
    -------
    log_left : ndarray of shape (k_max+1, n+1)
        Forward prefix table.
    log_right : ndarray of shape (k_max+1, n+1)
        Backward suffix table.
    """

    L = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
    R = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
    L[0, 0] = 0.0
    R[0, n] = 0.0

    for k in range(0, k_max):
        for j in range(k + 1, n + 1):
            h = np.arange(k, j)
            terms = L[k, h] + log_block_evidence[h, j]
            L[k + 1, j] = float(logsumexp(terms)) if terms.size else -np.inf

    for k in range(0, k_max):
        for i in range(0, n):
            if i > n - 1 - k:
                continue
            h = np.arange(i + 1, n - k + 1)
            terms = log_block_evidence[i, h] + R[k, h]
            R[k + 1, i] = float(logsumexp(terms)) if terms.size else -np.inf

    return L, R


def posterior_over_k(
    log_left: FloatArray, n: int, k_max: int
) -> tuple[FloatArray, FloatArray, float]:
    """Compute ``log P(k|y)``, ``P(k|y)``, and ``log P(y)`` under a uniform prior.

    The combinatorial correction ``binom(n-1, k-1)`` is applied so that each
    boundary vector in :math:`\\mathcal{T}_k` receives equal prior mass
    (index-uniform partition prior).

    Parameters
    ----------
    log_left : ndarray of shape (k_max+1, n+1)
        Forward table from :func:`forward_backward`.
    n : int
        Sequence length.
    k_max : int
        Maximum segment count.

    Returns
    -------
    log_posterior_k : ndarray of shape (k_max,)
        ``log P(k|y)`` for ``k = 1, ..., k_max``.
    posterior_k : ndarray of shape (k_max,)
        ``P(k|y)``.
    log_evidence : float
        ``log P(y)``.
    """

    log_py_given_k = np.array(
        [log_left[k, n] - log_binom(n - 1, k - 1) for k in range(1, k_max + 1)],
        dtype=float,
    )
    log_prior = -math.log(float(k_max))
    log_unnorm = log_py_given_k + log_prior
    log_evidence = float(logsumexp(log_unnorm))
    log_posterior_k = log_unnorm - log_evidence
    posterior_k = np.exp(log_posterior_k)
    return log_posterior_k, posterior_k, log_evidence


def boundary_event_marginals(
    log_left: FloatArray,
    log_right: FloatArray,
    log_posterior_k: FloatArray,
    n: int,
    k_max: int,
) -> FloatArray:
    """Compute per-index boundary-event marginals ``P(b_i=1 | y)``.

    This is the calibration target ``d1`` in the report's §6.2: for each
    interior index ``i``, the posterior probability that some boundary lies at
    ``i``, marginalised over ``k`` with weights ``P(k|y)``.

    Returns
    -------
    ndarray of shape (n-1,)
        Element ``i-1`` is ``P(b_i = 1 | y)`` for ``i = 1, ..., n-1``.
    """

    w_k = np.exp(log_posterior_k)
    out = np.zeros(n - 1, dtype=float)
    for i in range(1, n):
        acc = 0.0
        for k in range(2, k_max + 1):
            p = np.arange(1, k)
            terms = log_left[p, i] + log_right[k - p, i] - log_left[k, n]
            acc += float(w_k[k - 1]) * float(np.exp(logsumexp(terms)))
        out[i - 1] = acc
    return out


def boundary_location_posterior(
    log_left: FloatArray,
    log_right: FloatArray,
    n: int,
    k: int,
) -> FloatArray:
    """Return ``P(t_p = h | y, k)`` for all boundaries ``p = 1, ..., k-1``.

    Parameters
    ----------
    log_left, log_right : ndarray
        Sum-product tables.
    n : int
        Sequence length.
    k : int
        Number of segments.

    Returns
    -------
    ndarray of shape (k-1, n+1)
        Row ``p-1`` gives ``P(t_p = h | y, k)`` for ``h = 0, ..., n``.
    """

    out = np.zeros((max(k - 1, 0), n + 1), dtype=float)
    if k <= 1:
        return out
    denom = log_left[k, n]
    for p in range(1, k):
        for h in range(p, n - (k - p) + 1):
            out[p - 1, h] = float(np.exp(log_left[p, h] + log_right[k - p, h] - denom))
    return out


# -----------------------------------------------------------------------------
# Max-sum recursion (joint MAP segmentation)
# -----------------------------------------------------------------------------


def max_sum_segmentation(
    log_block_evidence: FloatArray,
    k: int,
    *,
    log_length_prior: FloatArray | None = None,
) -> tuple[list[int], float]:
    """Exact joint MAP boundary vector via max-sum DP with backtracking.

    This is distinct from :func:`boundary_event_marginals`: ``argmax_t p(t|y,k)``
    is a *joint* optimum, whereas argmaxing the marginals independently may
    yield an infeasible or suboptimal point estimate (see §4.4, Remark).

    Parameters
    ----------
    log_block_evidence : ndarray of shape (n+1, n+1)
        ``la[i, j]`` from :func:`forward_backward`.
    k : int
        Target number of segments.
    log_length_prior : ndarray of shape (n+1, n+1), optional
        Optional additive log-prior ``log g(x_j - x_i)`` on segment lengths. If
        ``None``, the length prior is uniform.

    Returns
    -------
    boundaries : list of int
        MAP boundary vector ``(0, t_1, ..., t_{k-1}, n)``.
    log_joint : float
        ``log p(t_MAP | y, k)`` up to the partition-prior normaliser.

    Raises
    ------
    ValueError
        If ``k < 1`` or ``k > n``.
    """

    la = np.asarray(log_block_evidence, dtype=float)
    if la.ndim != 2 or la.shape[0] != la.shape[1]:
        raise ValueError("log_block_evidence must be a square (n+1, n+1) matrix.")
    n = la.shape[0] - 1
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}; got {k}.")

    score = la if log_length_prior is None else la + np.asarray(log_length_prior, dtype=float)

    M = np.full((k + 1, n + 1), -np.inf, dtype=float)
    back = np.full((k + 1, n + 1), -1, dtype=int)
    M[0, 0] = 0.0

    for q in range(1, k + 1):
        for j in range(q, n + 1):
            lower = q - 1
            upper = j - 1
            if upper < lower:
                continue
            h_range = np.arange(lower, upper + 1)
            prev = M[q - 1, h_range]
            step = score[h_range, j]
            candidate = prev + step
            if candidate.size == 0:
                continue
            best_idx = int(np.argmax(candidate))
            best_val = float(candidate[best_idx])
            if np.isfinite(best_val):
                M[q, j] = best_val
                back[q, j] = int(h_range[best_idx])

    log_joint = float(M[k, n])
    if not np.isfinite(log_joint):
        raise RuntimeError(f"No finite MAP segmentation exists for k={k}.")

    boundaries = [n]
    j = n
    for q in range(k, 0, -1):
        h = int(back[q, j])
        if h < 0:
            raise RuntimeError("Backtracking failed: missing back-pointer.")
        boundaries.append(h)
        j = h
    return sorted(boundaries), log_joint


def marginal_boundary_modes(d1: FloatArray, k_hat: int, n: int) -> list[int]:
    """Select interior boundaries by top-``k_hat - 1`` marginal scores.

    This returns the **marginal** boundary summary, *not* the joint MAP. Kept
    available as an explicit (and distinct) option; see
    :func:`max_sum_segmentation` for the exact joint optimum.
    """

    if k_hat <= 1:
        return [0, n]
    if d1.size != n - 1:
        raise ValueError("boundary score vector must have length n-1.")
    best = np.argsort(d1)[-(k_hat - 1) :]
    picks = np.sort(best + 1)
    return [0, *picks.tolist(), n]


# -----------------------------------------------------------------------------
# Bayesian regression curve (expectation of the latent signal)
# -----------------------------------------------------------------------------


def bayes_regression_curve_fixed_k(
    log_left: FloatArray,
    log_right: FloatArray,
    log_block_evidence: FloatArray,
    block_first_moment: FloatArray,
    n: int,
    k: int,
) -> FloatArray:
    r"""Posterior-expected latent signal conditional on ``k`` segments.

    Uses the difference-array trick to accumulate per-block contributions in
    :math:`O(n^2)`.
    """

    denom = float(log_left[k, n])
    if not np.isfinite(denom):
        return np.full(n, np.nan, dtype=float)

    diff = np.zeros(n + 1, dtype=float)
    for i in range(0, n):
        Li = log_left[0:k, i]
        for j in range(i + 1, n + 1):
            Rj = log_right[k - 1 :: -1, j]
            la = float(log_block_evidence[i, j])
            if not np.isfinite(la):
                continue
            log_pseg = float(logsumexp(Li + Rj) + la - denom)
            if log_pseg < -745.0:
                continue
            w = math.exp(log_pseg)

            a = float(block_first_moment[i, j])
            if a == 0.0 or not np.isfinite(a):
                continue
            log_abs_mu = math.log(abs(a)) - la
            if log_abs_mu < -745.0:
                continue
            if log_abs_mu > 709.0:
                mu_hat = math.copysign(float("inf"), a)
            else:
                mu_hat = math.copysign(math.exp(log_abs_mu), a)

            contrib = w * mu_hat
            if contrib != 0.0 and np.isfinite(contrib):
                diff[i] += contrib
                diff[j] -= contrib

    mu = np.cumsum(diff)
    return mu[:n]


def bayes_regression_curve_mixed_k(
    log_left: FloatArray,
    log_right: FloatArray,
    log_block_evidence: FloatArray,
    block_first_moment: FloatArray,
    n: int,
    k_max: int,
    posterior_k: FloatArray,
) -> FloatArray:
    """Posterior-expected latent signal mixed over ``k`` with weights ``posterior_k``."""

    out = np.zeros(n, dtype=float)
    for k in range(1, k_max + 1):
        w = float(posterior_k[k - 1])
        if w == 0.0 or not np.isfinite(log_left[k, n]):
            continue
        out += w * bayes_regression_curve_fixed_k(
            log_left, log_right, log_block_evidence, block_first_moment, n, k
        )
    return out
