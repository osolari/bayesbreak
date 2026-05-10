r"""Dynamic-programming recursions over contiguous partitions.

Given a triangular array of conjugate-block log evidences
``log_block_evidence[i, j] = log A^0_{ij}`` (eq. ``problem-block-evidence``)
this module computes:

- the **sum-product** forward / backward tables (eq. ``LR``),
- the **segment-count posterior** ``P(k|y)`` (eq. ``post-k``),
- the **boundary marginal** ``P(t_p=h|y,k)`` (eq. ``boundary-post``),
- the **boundary-event marginal** ``P(b_i=1|y,k)`` (eq. ``boundary-event``)
  *conditional on a fixed segment count ``k``* — the calibration target,
- the **joint MAP** segmentation via max-sum + backtracking (eq. ``joint-map-k``),
- the **Bayesian regression curve** via the difference-array trick on the
  per-block contributions ``F^{(r)}_{ij}(k)``.

A **design-aware partition prior** ``p(t|k) ∝ ∏_q g(Δ_x(t_{q-1}, t_q))`` is
threaded uniformly through every recursion: callers supply the optional
length-factor table ``log_g[i,j] = log g(Δ_x(i,j))``, the DP absorbs it into
``Ã^{(0)}_{ij} := A^{(0)}_{ij} g(Δ_x(i,j))`` (eq. ``Atilde``), and the
normalizer ``C_k = Σ_t ∏_q g(Δ_x(t_{q-1}, t_q))`` is computed by the same
recursion with ``A ≡ 1`` (eq. ``Ck-general``). The index-uniform default
(``g ≡ 1``) reduces to ``C_k = C(n-1, k-1)``.

Every recursion runs in log space.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .utils import log_binom, logsumexp

FloatArray = NDArray[np.floating]


# -----------------------------------------------------------------------------
# Length-prior helpers
# -----------------------------------------------------------------------------


def _resolve_log_g(log_g_table: FloatArray | None, n: int) -> FloatArray:
    """Return the ``(n+1, n+1)`` ``log g(Δ)`` table, defaulting to zeros."""

    if log_g_table is None:
        return np.zeros((n + 1, n + 1), dtype=float)
    arr = np.asarray(log_g_table, dtype=float)
    if arr.shape != (n + 1, n + 1):
        raise ValueError(f"log_g_table must have shape ({n+1}, {n+1}); got {arr.shape}.")
    return arr


def compute_log_C_k(log_g_table: FloatArray | None, n: int, k_max: int) -> FloatArray:
    r"""Compute ``log C_k`` for ``k = 0, ..., k_max`` under a length-factor prior.

    For ``g ≡ 1`` this returns ``[log binom(n-1, k-1)]``. For arbitrary ``log_g``
    the recursion ``L^{(g)}_{k+1, j} = Σ_h L^{(g)}_{k, h} · g(Δ_x(h, j))`` is run
    in log space; ``log C_k = log L^{(g)}_{k, n}`` (eq. ``Ck-general``).

    Parameters
    ----------
    log_g_table : ndarray of shape (n+1, n+1) or None
        Pairwise ``log g(Δ_x(i, j))``. ``None`` is the index-uniform shortcut.
    n, k_max : int

    Returns
    -------
    log_C_k : ndarray of shape (k_max + 1,)
        ``log_C_k[0]`` is set to ``-inf`` (no admissible 0-segment partition);
        ``log_C_k[k]`` is ``log C_k`` for ``k = 1, ..., k_max``.
    """

    log_C_k = np.full(k_max + 1, -np.inf, dtype=float)
    if k_max < 1:
        return log_C_k

    if log_g_table is None:
        for k in range(1, k_max + 1):
            log_C_k[k] = log_binom(n - 1, k - 1)
        return log_C_k

    Lg = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
    Lg[0, 0] = 0.0
    log_g = np.asarray(log_g_table, dtype=float)
    for k in range(0, k_max):
        for j in range(k + 1, n + 1):
            h = np.arange(k, j)
            terms = Lg[k, h] + log_g[h, j]
            if terms.size:
                Lg[k + 1, j] = float(logsumexp(terms))

    for k in range(1, k_max + 1):
        log_C_k[k] = float(Lg[k, n])
    return log_C_k


def _resolve_log_p_k(log_p_k: FloatArray | None, k_max: int) -> FloatArray:
    """Default to a uniform ``p(k)`` over ``k = 1, ..., k_max``."""

    if log_p_k is None:
        return np.full(k_max + 1, -math.log(float(k_max)), dtype=float)
    arr = np.asarray(log_p_k, dtype=float)
    if arr.shape != (k_max + 1,):
        # Tolerate a length-k_max array that omits the k=0 slot.
        if arr.shape == (k_max,):
            full = np.full(k_max + 1, -np.inf, dtype=float)
            full[1:] = arr
            return full
        raise ValueError(f"log_p_k must have shape ({k_max+1},) or ({k_max},); got {arr.shape}.")
    return arr


# -----------------------------------------------------------------------------
# Sum-product recursions
# -----------------------------------------------------------------------------


def forward_backward(
    log_block_evidence: FloatArray,
    n: int,
    k_max: int,
    *,
    log_g_table: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Forward (prefix) and backward (suffix) sum-product tables.

    With ``Ã^{(0)}_{ij} = A^{(0)}_{ij} g(Δ_x(i, j))`` (length factor absorbed),

    .. math::
        L̃[k, j] = \log\sum_{t \in \mathcal{T}_{k, j}} \prod_{q=1}^k Ã^{(0)}_{t_{q-1} t_q}, \qquad
        R̃[k, i] = \log\sum_{t} \prod_{q=1}^k Ã^{(0)}_{t_{q-1} t_q}.
    """

    L = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
    R = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
    L[0, 0] = 0.0
    R[0, n] = 0.0

    log_g = _resolve_log_g(log_g_table, n)
    score = np.asarray(log_block_evidence, dtype=float) + log_g

    for k in range(0, k_max):
        for j in range(k + 1, n + 1):
            h = np.arange(k, j)
            terms = L[k, h] + score[h, j]
            L[k + 1, j] = float(logsumexp(terms)) if terms.size else -np.inf

    for k in range(0, k_max):
        for i in range(0, n):
            if i > n - 1 - k:
                continue
            h = np.arange(i + 1, n - k + 1)
            terms = score[i, h] + R[k, h]
            R[k + 1, i] = float(logsumexp(terms)) if terms.size else -np.inf

    return L, R


def posterior_over_k(
    log_left: FloatArray,
    n: int,
    k_max: int,
    *,
    log_C_k: FloatArray | None = None,
    log_p_k: FloatArray | None = None,
    log_g_table: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, float]:
    r"""``log P(k|y)``, ``P(k|y)``, and ``log P(y)`` under an arbitrary ``p(k)``.

    Implements eq. ``post-k``: ``P(k|y) ∝ p(k) · L̃_{k, n} / C_k``. Either pass
    ``log_C_k`` directly (if pre-computed) or pass ``log_g_table`` and let this
    routine compute ``log C_k`` via :func:`compute_log_C_k`.

    Returns
    -------
    log_posterior_k : ndarray of shape (k_max,)
    posterior_k : ndarray of shape (k_max,)
    log_evidence : float
        ``log P(y) = logsumexp_k(log p(k) + log P(y|k))``.
    """

    if log_C_k is None:
        log_C_k = compute_log_C_k(log_g_table, n, k_max)
    log_C_k = np.asarray(log_C_k, dtype=float)
    if log_C_k.shape != (k_max + 1,):
        raise ValueError(f"log_C_k must have shape ({k_max+1},); got {log_C_k.shape}.")

    log_p_k_full = _resolve_log_p_k(log_p_k, k_max)

    log_py_given_k = np.array(
        [log_left[k, n] - log_C_k[k] for k in range(1, k_max + 1)],
        dtype=float,
    )
    log_unnorm = log_py_given_k + log_p_k_full[1:]
    log_evidence = float(logsumexp(log_unnorm))
    log_posterior_k = log_unnorm - log_evidence
    posterior_k = np.exp(log_posterior_k)
    return log_posterior_k, posterior_k, log_evidence


def boundary_event_marginals_fixed_k(
    log_left: FloatArray,
    log_right: FloatArray,
    n: int,
    k: int,
) -> FloatArray:
    r"""``P(b_i = 1 | y, k) = Σ_{p=1}^{k-1} L̃[p, i] R̃[k-p, i] / L̃[k, n]``.

    Eq. ``boundary-event``. Returns a length-``(n-1)`` vector indexed by
    interior ``i = 1, ..., n-1``. This is the calibration target referenced
    throughout §6.
    """

    out = np.zeros(n - 1, dtype=float)
    if k <= 1:
        return out
    denom = float(log_left[k, n])
    if not np.isfinite(denom):
        return out
    for i in range(1, n):
        p = np.arange(1, k)
        terms = log_left[p, i] + log_right[k - p, i] - denom
        out[i - 1] = float(np.exp(logsumexp(terms)))
    return out


def boundary_event_marginals_marginalised(
    log_left: FloatArray,
    log_right: FloatArray,
    log_posterior_k: FloatArray,
    n: int,
    k_max: int,
) -> FloatArray:
    """``P(b_i = 1 | y) = Σ_k P(k|y) · P(b_i = 1 | y, k)``.

    Provided for completeness; the calibration target in §6 is the
    fixed-``k`` version.
    """

    out = np.zeros(n - 1, dtype=float)
    p_k = np.exp(log_posterior_k)
    for k in range(2, k_max + 1):
        if p_k[k - 1] <= 0:
            continue
        out += float(p_k[k - 1]) * boundary_event_marginals_fixed_k(log_left, log_right, n, k)
    return out


def boundary_location_posterior(
    log_left: FloatArray,
    log_right: FloatArray,
    n: int,
    k: int,
) -> FloatArray:
    r"""``P(t_p = h | y, k)`` for all boundaries ``p = 1, ..., k - 1``."""

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
    log_g_table: FloatArray | None = None,
) -> tuple[list[int], float]:
    r"""Joint MAP boundary vector via max-sum DP + backtracking (§4.4).

    Distinct from :func:`boundary_event_marginals_fixed_k`: ``argmax_t p(t|y,k)``
    is a *joint* optimum and is generally not equal to the vector of marginal
    boundary modes.

    Parameters
    ----------
    log_block_evidence : ndarray of shape (n+1, n+1)
        ``la[i, j]`` from :func:`forward_backward`.
    k : int
        Target segment count.
    log_g_table : ndarray of shape (n+1, n+1), optional
        Optional length-prior table (eq. ``Atilde``). Added to the score.

    Returns
    -------
    boundaries : list of int
    log_joint : float
    """

    la = np.asarray(log_block_evidence, dtype=float)
    if la.ndim != 2 or la.shape[0] != la.shape[1]:
        raise ValueError("log_block_evidence must be square (n+1, n+1).")
    n = la.shape[0] - 1
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}; got {k}.")

    log_g = _resolve_log_g(log_g_table, n)
    score = la + log_g

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
            candidate = M[q - 1, h_range] + score[h_range, j]
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
    """Top-``k_hat - 1`` marginal boundary positions (diagnostic only).

    This is the *marginal* boundary summary; see :func:`max_sum_segmentation`
    for the joint MAP optimum.
    """

    if k_hat <= 1:
        return [0, n]
    if d1.size != n - 1:
        raise ValueError("boundary score vector must have length n-1.")
    best = np.argsort(d1)[-(k_hat - 1) :]
    picks = np.sort(best + 1)
    return [0, *picks.tolist(), n]


# -----------------------------------------------------------------------------
# Bayesian regression curve
# -----------------------------------------------------------------------------


def bayes_regression_curve_fixed_k(
    log_left: FloatArray,
    log_right: FloatArray,
    log_block_evidence: FloatArray,
    block_first_moment: FloatArray,
    n: int,
    k: int,
    *,
    log_g_table: FloatArray | None = None,
) -> FloatArray:
    r"""Posterior-expected latent signal conditional on ``k`` segments (eq. ``segmom``).

    The length factor ``g(Δ)`` cancels in segment-wise posterior moments
    (Eq. ``segmom-seg``) but does *not* cancel in the curve, where the weight
    of each candidate segmentation depends on ``g``. We therefore form
    ``Ã^{(0)} = A^{(0)} · g`` for the segmentation weights and use the *raw*
    ``A^{(1)}`` for the moment numerator (the ``g`` cancels segment-wise).
    """

    denom = float(log_left[k, n])
    if not np.isfinite(denom):
        return np.full(n, np.nan, dtype=float)

    log_g = _resolve_log_g(log_g_table, n)
    la = np.asarray(log_block_evidence, dtype=float)
    la_tilde = la + log_g

    diff = np.zeros(n + 1, dtype=float)
    for i in range(0, n):
        Li = log_left[0:k, i]
        for j in range(i + 1, n + 1):
            Rj = log_right[k - 1 :: -1, j]
            la_t = float(la_tilde[i, j])
            if not np.isfinite(la_t):
                continue
            log_pseg = float(logsumexp(Li + Rj) + la_t - denom)
            if log_pseg < -745.0:
                continue
            w = math.exp(log_pseg)

            a = float(block_first_moment[i, j])
            if a == 0.0 or not np.isfinite(a):
                continue
            la_raw = float(la[i, j])
            if not np.isfinite(la_raw):
                continue
            log_abs_mu = math.log(abs(a)) - la_raw
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
    *,
    log_g_table: FloatArray | None = None,
) -> FloatArray:
    """``Σ_k P(k|y) · E[μ_t | y, k]`` (eq. discussed after ``segmom``)."""

    out = np.zeros(n, dtype=float)
    for k in range(1, k_max + 1):
        w = float(posterior_k[k - 1])
        if w == 0.0 or not np.isfinite(log_left[k, n]):
            continue
        out += w * bayes_regression_curve_fixed_k(
            log_left,
            log_right,
            log_block_evidence,
            block_first_moment,
            n,
            k,
            log_g_table=log_g_table,
        )
    return out
