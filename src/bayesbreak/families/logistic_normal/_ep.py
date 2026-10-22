r"""Vectorised parallel expectation propagation for the Bernoulli-logistic block.

The site update schedule is Minka's parallel EP: every site is updated from
the same cavity constructed at the start of a pass. Damping keeps it
stable; the sequential schedule is not needed. Warm-starting along the
``j``-axis (:func:`ep_i_sweep`) makes the per-block cost near-constant
once the first block converges.
"""

from __future__ import annotations

import math

import numpy as np

from ...utils import logsumexp
from ._util import sigmoid


def _pass_parallel(
    y: np.ndarray,
    w: np.ndarray,
    a: np.ndarray,
    r: np.ndarray,
    log_c: np.ndarray,
    prior_prec: float,
    prior_mean: float,
    gh_x: np.ndarray,
    log_gh_w: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One synchronous parallel-EP pass with vectorised moment matching."""

    post_prec = prior_prec + float(a.sum())
    post_r = prior_mean * prior_prec + float(r.sum())

    cav_prec = post_prec - a
    cav_r = post_r - r
    valid = cav_prec > 1e-12
    if not np.any(valid):
        return a, r, log_c, 0.0

    v_cav = np.where(valid, 1.0 / np.maximum(cav_prec, 1e-30), 1.0)
    m_cav = np.where(valid, cav_r * v_cav, 0.0)

    theta = m_cav[:, None] + np.sqrt(2.0 * v_cav)[:, None] * gh_x[None, :]
    log_lik = w[:, None] * (y[:, None] * theta - np.logaddexp(0.0, theta))
    log_terms = log_gh_w[None, :] + log_lik

    logZ_cav = np.asarray(logsumexp(log_terms, axis=1))
    w_norm = np.exp(log_terms - logZ_cav[:, None])
    m_tilt = (w_norm * theta).sum(axis=1)
    m2_tilt = (w_norm * theta * theta).sum(axis=1)
    v_tilt = np.maximum(m2_tilt - m_tilt * m_tilt, 1e-12)

    new_a = 1.0 / v_tilt - cav_prec
    new_r = m_tilt / v_tilt - cav_r

    new_a = damping * new_a + (1.0 - damping) * a
    new_r = damping * new_r + (1.0 - damping) * r
    new_a = np.where(valid, new_a, a)
    new_r = np.where(valid, new_r, r)

    new_log_c = np.where(
        valid,
        logZ_cav + 0.5 * (np.log(v_cav) - np.log(v_tilt)),
        log_c,
    )

    step = float(np.max(np.abs(new_a - a)) + np.max(np.abs(new_r - r)))
    return new_a, new_r, new_log_c, step


def run_ep(
    y: np.ndarray,
    w: np.ndarray,
    nu: float,
    rho2: float,
    a_init: np.ndarray | None,
    r_init: np.ndarray | None,
    log_c_init: np.ndarray | None,
    *,
    max_iter: int,
    tol_kl: float,
    damping: float,
    n_gh: int,
) -> tuple[float, float, float, float, bool, float, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised parallel EP for one block.

    Convergence is declared when the largest natural-parameter update
    across sites falls below ``tol_kl`` (a proxy for Minka's site-KL
    residual). ``(a_init, r_init, log_c_init)`` supports warm-starting.
    """

    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    N = int(y.size)
    rho2 = max(float(rho2), 1e-12)
    prior_prec = 1.0 / rho2
    prior_mean = float(nu)

    if N == 0:
        empty = np.zeros(0, dtype=float)
        return (
            0.0,
            prior_mean,
            rho2,
            float(sigmoid(np.asarray(prior_mean))[()]),
            True,
            0.0,
            empty,
            empty,
            empty,
        )

    a = np.zeros(N, dtype=float) if a_init is None else np.array(a_init, dtype=float, copy=True)
    r = np.zeros(N, dtype=float) if r_init is None else np.array(r_init, dtype=float, copy=True)
    log_c = (
        np.zeros(N, dtype=float)
        if log_c_init is None
        else np.array(log_c_init, dtype=float, copy=True)
    )

    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh)
    log_gh_w = np.log(gh_w) - 0.5 * math.log(math.pi)

    converged = False
    step = math.inf
    for _ in range(max_iter):
        a, r, log_c, step = _pass_parallel(
            y, w, a, r, log_c, prior_prec, prior_mean, gh_x, log_gh_w, damping
        )
        if step < tol_kl:
            converged = True
            break

    post_prec = prior_prec + float(a.sum())
    if post_prec <= 1e-12:
        post_prec = prior_prec
        post_r = prior_mean * prior_prec
    else:
        post_r = prior_mean * prior_prec + float(r.sum())
    v_post = 1.0 / post_prec
    m_post = post_r * v_post

    log_post_norm = 0.5 * math.log(2.0 * math.pi * v_post) + 0.5 * (m_post * m_post / v_post)
    log_prior_norm = 0.5 * math.log(2.0 * math.pi * rho2) + 0.5 * (prior_mean * prior_mean / rho2)
    logA0 = float(log_c.sum()) + log_post_norm - log_prior_norm

    w_grid = gh_w / math.sqrt(math.pi)
    theta_grid = m_post + math.sqrt(2.0 * v_post) * gh_x
    p_mean = float((w_grid * sigmoid(theta_grid)).sum())

    return logA0, float(m_post), float(v_post), p_mean, converged, float(step), a, r, log_c


def ep_single_block(
    y_block: np.ndarray,
    w_block: np.ndarray,
    nu: float,
    rho2: float,
    *,
    max_iter: int = 20,
    tol_kl: float = 1e-4,
    damping: float = 0.7,
    n_gh: int = 25,
) -> tuple[float, float, float, float, bool]:
    """Parallel-EP for a single block; see :func:`run_ep`."""

    logA0, m, v, p_mean, ok, _step, _a, _r, _lc = run_ep(
        y_block,
        w_block,
        nu,
        rho2,
        a_init=None,
        r_init=None,
        log_c_init=None,
        max_iter=max_iter,
        tol_kl=tol_kl,
        damping=damping,
        n_gh=n_gh,
    )
    return logA0, m, v, p_mean, ok


def ep_i_sweep(
    y_full: np.ndarray,
    w_full: np.ndarray,
    i: int,
    j_indices: np.ndarray,
    nu: float,
    rho2: float,
    *,
    max_iter: int,
    tol_kl: float,
    damping: float,
    n_gh: int,
    warm_start_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warm-started parallel-EP for all blocks ``(i, j)``, ``j`` in ``j_indices``.

    Site vectors from block ``(i, j-1)`` are extended with a zero-initialised
    new site and refined for ``warm_start_iter`` passes; a full ``max_iter``
    pass runs whenever the warm start fails to converge. Terminal per-block
    site-update magnitudes are returned as the optimisation residual.
    """

    j_indices = np.asarray(j_indices, dtype=int)
    B = int(j_indices.size)
    logA0 = np.empty(B, dtype=float)
    m_out = np.empty(B, dtype=float)
    v_out = np.empty(B, dtype=float)
    p_out = np.empty(B, dtype=float)
    conv = np.empty(B, dtype=bool)
    step_out = np.empty(B, dtype=float)

    a_prev = np.zeros(0, dtype=float)
    r_prev = np.zeros(0, dtype=float)
    log_c_prev = np.zeros(0, dtype=float)

    for b, j in enumerate(j_indices):
        j = int(j)
        y_block = y_full[i:j]
        w_block = w_full[i:j]
        extra = j - i - a_prev.size
        if extra > 0:
            a_init = np.concatenate([a_prev, np.zeros(extra, dtype=float)])
            r_init = np.concatenate([r_prev, np.zeros(extra, dtype=float)])
            log_c_init = np.concatenate([log_c_prev, np.zeros(extra, dtype=float)])
        else:
            a_init = a_prev[: j - i]
            r_init = r_prev[: j - i]
            log_c_init = log_c_prev[: j - i]

        la0, m_b, v_b, p_b, ok, step, a_new, r_new, log_c_new = run_ep(
            y_block,
            w_block,
            nu=nu,
            rho2=rho2,
            a_init=a_init,
            r_init=r_init,
            log_c_init=log_c_init,
            max_iter=warm_start_iter,
            tol_kl=tol_kl,
            damping=damping,
            n_gh=n_gh,
        )
        if not ok:
            la0, m_b, v_b, p_b, ok, step, a_new, r_new, log_c_new = run_ep(
                y_block,
                w_block,
                nu=nu,
                rho2=rho2,
                a_init=a_new,
                r_init=r_new,
                log_c_init=log_c_new,
                max_iter=max_iter,
                tol_kl=tol_kl,
                damping=damping,
                n_gh=n_gh,
            )

        logA0[b] = la0
        m_out[b] = m_b
        v_out[b] = v_b
        p_out[b] = p_b
        conv[b] = ok
        step_out[b] = step

        a_prev = a_new
        r_prev = r_new
        log_c_prev = log_c_new

    return logA0, m_out, v_out, p_out, conv, step_out
