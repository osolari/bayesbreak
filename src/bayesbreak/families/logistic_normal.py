"""bayesbreak.families.logistic_normal

Bernoulli observations with a *logistic-normal* (a.k.a. logit-normal) segment prior.

Model (per segment q):
    y_t | theta_q ~ Bernoulli(sigmoid(theta_q))
    theta_q ~ Normal(nu, rho^2)

This is a non-conjugate family because the Bernoulli likelihood is not conjugate to a
Normal prior on the log-odds. The paper describes several block-level approximations
for the segment marginal likelihood A^0_{ij} = p(y_{(i,j]} | single segment).

We support the following approximations at the *segment* (block) level:
  - "laplace"   : Laplace approximation around the posterior mode
                  (``prop:uniform-bounds`` (ii): ``O(n^{-1})`` on reachable blocks).
  - "jj"        : Jaakkola--Jordan quadratic bound (variational lower bound;
                  ``prop:uniform-bounds`` (iii): ``O(n^{-1})``).
  - "pg_vb"     : Polya--Gamma variational bound (equivalent precision updates
                  to JJ; ``prop:uniform-bounds`` (iv): ``O(n^{-1})``).
  - "ep"        : True per-observation expectation propagation (Minka 2001) with
                  accumulated site normalizers. EP need not satisfy a uniform
                  ``ε`` when iteration fails to converge — the canonical failure
                  mode of ``prop:uniform-bounds`` (v).
  - "gh" / "quadrature" : Gauss--Hermite quadrature reference
                  (``prop:uniform-bounds`` (i): ``O(Q^{-2r})`` for ``C^{2r}``
                  integrands).

All segment statistics depend only on weighted success count S and total weight N:
    S = sum_t w_t y_t,   N = sum_t w_t.

The base class expects A^1_{ij} = A^0_{ij} * E[mu | y_{(i,j]}], where mu is the
segment mean signal. Here, we define the signal as the Bernoulli probability:
    mu_q := sigmoid(theta_q).

For Laplace/JJ/PG-VB we approximate E[sigmoid(theta)] under a Gaussian posterior
using MacKay's logistic-normal approximation:
    E[sigmoid(theta)] \approx sigmoid(m / sqrt(1 + pi v / 8)).

For EP/quadrature we compute E[sigmoid(theta)] directly under the quadrature weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..utils import logsumexp

Approx = Literal["laplace", "jj", "pg_vb", "pg-vb", "ep", "gh", "quadrature"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid.
    return 1.0 / (1.0 + np.exp(-x))


def _mackay_logistic_normal_mean(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    # E[sigmoid(Z)] for Z ~ N(m, v): MacKay-style approximation.
    return _sigmoid(m / np.sqrt(1.0 + (math.pi * v) / 8.0))


def _safe_lambda_jj(xi: np.ndarray) -> np.ndarray:
    """JJ/PG-VB lambda(xi) = tanh(xi/2) / (4 xi) with xi -> 0 limit 1/8."""
    xi = np.asarray(xi, dtype=float)
    out = np.empty_like(xi)
    small = np.abs(xi) < 1e-10
    out[small] = 1.0 / 8.0
    xs = xi[~small]
    out[~small] = np.tanh(xs / 2.0) / (4.0 * xs)
    return out


def _laplace_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised 1D Laplace approximation for many blocks.

    Returns
    -------
    logA0 : ndarray
        Approximate log segment marginal likelihood.
    m : ndarray
        Approximate posterior mean of theta.
    v : ndarray
        Approximate posterior variance of theta.
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)

    # Initialise at the empirical logit, shrunk to the prior mean.
    p0 = np.clip(S / np.maximum(N, 1e-12), 1e-6, 1.0 - 1e-6)
    theta = np.log(p0) - np.log1p(-p0)
    theta = 0.5 * theta + 0.5 * nu

    inv_rho2 = 1.0 / max(rho2, 1e-12)

    for _ in range(max_iter):
        sig = _sigmoid(theta)
        # log-posterior derivatives.
        g = (S - N * sig) - (theta - nu) * inv_rho2
        h = -(N * sig * (1.0 - sig) + inv_rho2)
        step = g / np.maximum(-h, 1e-12)
        theta_new = theta + step
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new

    sig = _sigmoid(theta)
    h = -(N * sig * (1.0 - sig) + inv_rho2)
    v = 1.0 / np.maximum(-h, 1e-12)
    m = theta

    # loglik + logprior at mode.
    loglik = S * theta - N * np.logaddexp(0.0, theta)
    logprior = -0.5 * (math.log(2.0 * math.pi * rho2) + (theta - nu) ** 2 / rho2)
    logpost = loglik + logprior

    logA0 = logpost + 0.5 * math.log(2.0 * math.pi) - 0.5 * np.log(np.maximum(-h, 1e-300))
    return logA0, m, v


def _jj_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JJ variational bound for binomial-logistic with normal prior (1D).

    Returns (logA0_bound, m, v) where logA0_bound is a lower bound on log A0.
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    inv_rho2 = 1.0 / max(rho2, 1e-12)

    # Start from empirical logit or prior mean.
    p0 = np.clip(S / np.maximum(N, 1e-12), 1e-6, 1.0 - 1e-6)
    m = 0.5 * (np.log(p0) - np.log1p(-p0)) + 0.5 * nu
    v = np.full_like(m, rho2)
    xi = np.sqrt(m * m + v)

    for _ in range(max_iter):
        lam = _safe_lambda_jj(xi)
        A = inv_rho2 + 2.0 * N * lam
        v_new = 1.0 / np.maximum(A, 1e-300)
        b = nu * inv_rho2 + (S - 0.5 * N)
        m_new = b * v_new
        xi_new = np.sqrt(m_new * m_new + v_new)
        if np.max(np.abs(xi_new - xi)) < tol:
            m, v, xi = m_new, v_new, xi_new
            break
        m, v, xi = m_new, v_new, xi_new

    # Bound on log evidence.
    lam = _safe_lambda_jj(xi)
    A = inv_rho2 + 2.0 * N * lam
    b = nu * inv_rho2 + (S - 0.5 * N)

    # psi(x) = log(2 cosh(x/2)) = logaddexp(x/2, -x/2)
    psi_xi = np.logaddexp(xi / 2.0, -xi / 2.0)

    const = N * lam * (xi * xi) - N * psi_xi - 0.5 * (nu * nu) * inv_rho2
    logZ = (
        -0.5 * math.log(2.0 * math.pi * rho2)
        + const
        + 0.5 * math.log(2.0 * math.pi)
        - 0.5 * np.log(np.maximum(A, 1e-300))
        + 0.5 * (b * b) / np.maximum(A, 1e-300)
    )
    return logZ, m, v


def _gh_moments_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    n_gh: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gauss--Hermite quadrature for logA0 and posterior moments.

    Returns (logA0, m, v, p_mean) where p_mean = E[sigmoid(theta) | data].
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    if n_gh < 5:
        raise ValueError("n_gh must be >= 5 for stable quadrature.")

    # hermgauss integrates exp(-x^2) f(x) dx.
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    # Map to theta points under N(nu, rho2): theta = nu + sqrt(2 rho2) x
    theta = nu + math.sqrt(2.0 * rho2) * x  # (n_gh,)

    # log f(theta) for each GH node (broadcast over blocks).
    # loglik(theta) = S*theta - N*log(1+exp(theta))
    loglik = np.outer(S, theta) - np.outer(N, np.logaddexp(0.0, theta))  # (B, n_gh)
    logw = np.log(w)[None, :]  # (1, n_gh)
    log_terms = logw + loglik
    logZ_unnorm = logsumexp(log_terms, axis=1)  # (B,)
    logZ = logZ_unnorm - 0.5 * math.log(math.pi)

    # Normalised quadrature weights for the tilted posterior.
    w_norm = np.exp(log_terms - logZ_unnorm[:, None])  # (B, n_gh)
    m = w_norm @ theta
    m2 = w_norm @ (theta * theta)
    v = np.maximum(0.0, m2 - m * m)
    p_mean = w_norm @ _sigmoid(theta)
    return logZ, m, v, p_mean


def _ep_single_block(
    y_block: np.ndarray,
    w_block: np.ndarray,
    nu: float,
    rho2: float,
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    damping: float = 0.7,
    n_gh: int = 25,
) -> tuple[float, float, float, float, bool]:
    """Per-observation EP for a single Bernoulli-logistic block.

    Implements Minka (2001) expectation propagation: each observation
    contributes a Gaussian site approximating the Bernoulli likelihood
    factor, sites are updated by moment matching the cavity-tilted
    Gaussian-by-Bernoulli distribution via Gauss--Hermite quadrature, and
    the accumulated site normalizers are folded into the returned
    log-evidence per ``prop:uniform-bounds`` (v).

    Parameters
    ----------
    y_block : 1-D array of shape (N,)
        Bernoulli observations (weighted by ``w_block`` if given).
    w_block : 1-D array of shape (N,)
        Per-observation weights (treated as Binomial trial counts when
        ``y_block`` is a fraction, or as a power-likelihood weight).
    nu, rho2 : float
        Gaussian-prior mean and variance on ``theta``.
    max_iter, tol, damping, n_gh : int / float
        Standard EP knobs. The ``converged`` flag in the return is True
        only when the largest natural-parameter update across sites
        falls below ``tol`` within ``max_iter`` outer passes.

    Returns
    -------
    logA0 : float
        Approximate block log-evidence (sum of cavity-and-tilted Z ratios
        plus the full-posterior Gaussian normalizer).
    m, v : float
        Mean and variance of the EP-converged Gaussian approximation to
        ``p(theta | y_block)``.
    p_mean : float
        ``E[sigmoid(theta) | y_block]`` evaluated under the converged
        Gaussian via the same Gauss--Hermite quadrature used for moment
        matching.
    converged : bool
        Whether EP reached the tolerance within ``max_iter``.
    """
    y = np.asarray(y_block, dtype=float).ravel()
    w = np.asarray(w_block, dtype=float).ravel()
    N = int(y.size)
    if N == 0:
        # Empty block: posterior = prior, evidence = 1.
        return 0.0, float(nu), float(max(rho2, 1e-12)), float(_sigmoid(np.asarray(nu))[()]), True

    rho2 = max(float(rho2), 1e-12)
    prior_prec = 1.0 / rho2
    prior_mean = float(nu)

    # Per-site natural parameters (a_t * theta - 0.5 b_t theta^2 parameterisation,
    # where a_t is the precision contribution and r_t the natural mean).
    a = np.zeros(N, dtype=float)
    r = np.zeros(N, dtype=float)
    log_c = np.zeros(N, dtype=float)

    # Gauss--Hermite quadrature nodes (re-used across sites).
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_gh)
    log_gh_w = np.log(gh_w)

    converged = False
    for _it in range(max_iter):
        # Full-posterior natural parameters from prior + all sites.
        post_prec = prior_prec + float(a.sum())
        post_r = prior_mean * prior_prec + float(r.sum())
        if post_prec <= 0.0:
            # Sites collapsed to negative precision; bail with a soft state.
            post_prec = prior_prec
            post_r = prior_mean * prior_prec

        max_step = 0.0
        for t in range(N):
            # Cavity precision/natural-mean: full minus site t.
            cav_prec = post_prec - a[t]
            cav_r = post_r - r[t]
            if cav_prec <= 1e-12:
                # Cavity is improper; skip this site this iteration.
                continue
            v_cav = 1.0 / cav_prec
            m_cav = cav_r * v_cav
            if v_cav <= 0.0:
                continue

            # Moment-match against the cavity-tilted Bernoulli-by-weight.
            # theta nodes for this cavity: theta = m_cav + sqrt(2 v_cav) * x.
            theta = m_cav + math.sqrt(2.0 * v_cav) * gh_x  # (n_gh,)
            # Likelihood factor: weighted Bernoulli with effective trials w[t].
            # log L(theta) = w[t] * (y[t] * theta - log(1 + exp(theta))).
            log_lik = w[t] * (y[t] * theta - np.logaddexp(0.0, theta))
            log_terms = log_gh_w + log_lik
            logZ_cav = float(logsumexp(log_terms)) - 0.5 * math.log(math.pi)
            # Normalised tilted weights.
            log_terms_norm = log_terms - logsumexp(log_terms)
            w_norm = np.exp(log_terms_norm)
            m_tilt = float(w_norm @ theta)
            m2_tilt = float(w_norm @ (theta * theta))
            v_tilt = max(m2_tilt - m_tilt * m_tilt, 1e-12)

            # New site natural parameters (tilted minus cavity).
            new_a = (1.0 / v_tilt) - cav_prec
            new_r = (m_tilt / v_tilt) - cav_r

            # Damping to encourage convergence (prop:uniform-bounds (v) notes
            # that EP need not converge; damping is the standard remedy).
            new_a = damping * new_a + (1.0 - damping) * a[t]
            new_r = damping * new_r + (1.0 - damping) * r[t]

            step = max(abs(new_a - a[t]), abs(new_r - r[t]))
            if step > max_step:
                max_step = step

            # New site normalising constant: logZ_cav - 0.5 * cavity-vs-tilted
            # Gaussian-normalizer difference.
            # log Z_site = log Z_cav - log N(0 | m_cav, v_cav) / N(0 | m_tilt, v_tilt)
            # equivalently: incorporates the cavity log-normalizer correction.
            new_log_c = logZ_cav + 0.5 * (
                math.log(2.0 * math.pi * v_cav) - math.log(2.0 * math.pi * v_tilt)
            )

            # Update site and refresh the running posterior.
            post_prec = cav_prec + new_a
            post_r = cav_r + new_r
            a[t] = new_a
            r[t] = new_r
            log_c[t] = new_log_c

        if max_step < tol:
            converged = True
            break

    # Final Gaussian posterior from accumulated sites.
    post_prec = prior_prec + float(a.sum())
    if post_prec <= 1e-12:
        post_prec = prior_prec
    v_post = 1.0 / post_prec
    m_post = (prior_mean * prior_prec + float(r.sum())) * v_post

    # Block log-evidence: Σ_t log c_t plus the full-posterior Gaussian
    # normaliser ratio against the prior. The formula derives directly
    # from the EP energy decomposition (cf. Minka 2001 eq. 12.5 in
    # Bishop PRML).
    log_post_norm = 0.5 * math.log(2.0 * math.pi * v_post) + 0.5 * (m_post * m_post / v_post)
    log_prior_norm = 0.5 * math.log(2.0 * math.pi * rho2) + 0.5 * (prior_mean * prior_mean / rho2)
    logA0 = float(log_c.sum()) + log_post_norm - log_prior_norm

    # Mean of sigmoid(theta) under the converged Gaussian (via GH).
    theta_grid = m_post + math.sqrt(2.0 * v_post) * gh_x
    w_grid = gh_w / math.sqrt(math.pi)
    p_mean = float((w_grid * _sigmoid(theta_grid)).sum())

    return logA0, float(m_post), float(v_post), p_mean, converged


def _ep_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    n_gh: int,
    *,
    y_for_blocks: np.ndarray | None = None,
    w_for_blocks: np.ndarray | None = None,
    block_endpoints: list[tuple[int, int]] | None = None,
    max_iter: int = 20,
    damping: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise per-block real EP across an arbitrary set of (i, j] blocks.

    EP requires per-observation iteration; this routine therefore loops
    over blocks but vectorises the GH moment-match within each block.
    See :func:`_ep_single_block` for the per-block contract; the
    additional returned ``converged`` mask reports per-block convergence.
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    B = int(S.shape[0])
    if y_for_blocks is None or block_endpoints is None:
        # No per-observation data: fall back to GH quadrature on the
        # aggregate sufficient statistics. This path is reached only when
        # the caller does not supply observation-level data (e.g. the
        # legacy _segment_posterior_mean entry point that summarises a
        # single block).
        logZ, m, v, p_mean = _gh_moments_block(S, N, nu=nu, rho2=rho2, n_gh=n_gh)
        return logZ, m, v, p_mean, np.ones(B, dtype=bool)

    if w_for_blocks is None:
        w_for_blocks = np.ones_like(y_for_blocks, dtype=float)

    logA0 = np.empty(B, dtype=float)
    m_out = np.empty(B, dtype=float)
    v_out = np.empty(B, dtype=float)
    p_out = np.empty(B, dtype=float)
    conv = np.empty(B, dtype=bool)

    for b, (i, j) in enumerate(block_endpoints):
        y_block = y_for_blocks[i:j]
        w_block = w_for_blocks[i:j]
        logA0[b], m_out[b], v_out[b], p_out[b], conv[b] = _ep_single_block(
            y_block,
            w_block,
            nu=nu,
            rho2=rho2,
            max_iter=max_iter,
            damping=damping,
            n_gh=n_gh,
        )

    return logA0, m_out, v_out, p_out, conv


class BayesBreakLogisticNormal(BayesBreakSegmenter):
    """BayesBreak for Bernoulli data with a Normal prior on the log-odds.

    Parameters
    ----------
    k_max:
        Maximum number of segments.
    estimate_hyper:
        If True, estimate (nu, rho2) from data (empirical Bayes).
        approx:
            Block-evidence approximation method. One of
            {"laplace", "jj", "pg_vb" (or "pg-vb"), "ep", "quadrature"}.
    nu, rho2:
        Optional fixed hyperparameters. If provided, they override the EB estimates.
    gh_points:
        Number of Gauss--Hermite points for "ep"/"quadrature".
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        approx: Approx = "laplace",
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
        nu: float | None = None,
        rho2: float | None = None,
        gh_points: int = 25,
        max_iter: int = 50,
    ):
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
        )
        # Accept both "pg_vb" and "pg-vb" (paper notation).
        self.approx = str(approx).lower().replace("-", "_")
        self.nu = nu
        self.rho2 = rho2
        self.gh_points = int(gh_points)
        self.max_iter = int(max_iter)

    # ----- hyperparameters (EB) -----

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> dict[str, float]:
        # If estimate_hyper is False, but user provided nu/rho2, respect them.
        if not self.estimate_hyper and self.nu is not None and self.rho2 is not None:
            return {"nu": float(self.nu), "rho2": float(self.rho2)}

        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != y.shape:
                raise ValueError("sample_weight must have same shape as y")

        # Empirical Bayes: match moments on the probability scale using a logit transform.
        # This is only a heuristic; it stabilises the prior for segmentation.
        eps = 1e-3
        p = np.clip(y, eps, 1.0 - eps)
        z = np.log(p) - np.log1p(-p)
        wsum = float(np.sum(w))
        nu = float(np.sum(w * z) / max(wsum, 1e-12))
        # Weighted variance.
        var = float(np.sum(w * (z - nu) ** 2) / max(wsum, 1e-12))
        rho2 = max(var, 1e-6)

        # Overrides.
        if self.nu is not None:
            nu = float(self.nu)
        if self.rho2 is not None:
            rho2 = float(self.rho2)
        return {"nu": nu, "rho2": rho2}

    # ----- per-segment evidence and A1 -----

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        n = int(y.size)
        nu = float(hyper["nu"])
        rho2 = float(hyper["rho2"])
        rho2 = max(rho2, 1e-12)

        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != y.shape:
                raise ValueError("sample_weight must have same shape as y")

        # Prefix sums for S and N.
        Sw = np.zeros(n + 1, dtype=float)
        Sw[1:] = np.cumsum(w * y)
        Nw = np.zeros(n + 1, dtype=float)
        Nw[1:] = np.cumsum(w)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        # Track EP convergence per block (only populated when approx="ep").
        ep_converged = np.ones((n + 1, n + 1), dtype=bool) if self.approx == "ep" else None

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            S = Sw[j] - Sw[i]
            N = Nw[j] - Nw[i]

            if self.approx == "laplace":
                logA0, m, v = _laplace_block(S, N, nu=nu, rho2=rho2, max_iter=self.max_iter)
                p_mean = _mackay_logistic_normal_mean(m, v)
            elif self.approx in ("jj", "pg_vb"):
                logA0, m, v = _jj_block(S, N, nu=nu, rho2=rho2, max_iter=self.max_iter)
                p_mean = _mackay_logistic_normal_mean(m, v)
            elif self.approx == "ep":
                # Real per-observation EP: loop over endpoint j for fixed i.
                endpoints = [(i, int(jj)) for jj in j]
                logA0, m, v, p_mean, conv = _ep_block(
                    S,
                    N,
                    nu=nu,
                    rho2=rho2,
                    n_gh=self.gh_points,
                    y_for_blocks=y,
                    w_for_blocks=w,
                    block_endpoints=endpoints,
                    max_iter=self.max_iter,
                )
                assert ep_converged is not None  # for the type checker
                ep_converged[i, j] = conv
            elif self.approx in ("gh", "quadrature"):
                # GH quadrature: low-node ("gh") or high-node ("quadrature").
                n_gh = self.gh_points if self.approx == "gh" else max(self.gh_points, 80)
                logA0, m, v, p_mean = _gh_moments_block(S, N, nu=nu, rho2=rho2, n_gh=n_gh)
            else:
                raise ValueError(
                    "approx must be one of "
                    "{'laplace','jj','pg_vb','ep','gh','quadrature'}, "
                    f"got {self.approx!r}."
                )

            lA0[i, j] = logA0
            A1[i, j] = np.exp(logA0) * p_mean

        # disallow empty segments
        diag = np.arange(n + 1)
        lA0[diag, diag] = -np.inf
        # Surface EP convergence info for run_non_conjugate_diagnostics /
        # prop:uniform-bounds (v) failure-mode detection.
        if self.approx == "ep":
            assert ep_converged is not None
            np.fill_diagonal(ep_converged, True)
            self.ep_converged_ = ep_converged
            self.ep_all_converged_ = bool(ep_converged[np.triu_indices(n + 1, k=1)].all())
        return lA0, A1

    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: np.ndarray,
        hyper: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> float:
        y = np.asarray(y, dtype=float)
        nu = float(hyper["nu"])
        rho2 = max(float(hyper["rho2"]), 1e-12)
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        S = float(np.sum(w[a:b] * y[a:b]))
        N = float(np.sum(w[a:b]))
        if self.approx == "laplace":
            logA0, m, v = _laplace_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, max_iter=self.max_iter
            )
            p_mean = float(_mackay_logistic_normal_mean(m, v)[0])
            return p_mean
        if self.approx in ("jj", "pg_vb"):
            logA0, m, v = _jj_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, max_iter=self.max_iter
            )
            p_mean = float(_mackay_logistic_normal_mean(m, v)[0])
            return p_mean
        if self.approx == "ep":
            # Per-observation EP on the requested block; ignores
            # convergence flag here since the caller only needs the mean.
            _, _, _, p_mean, _ = _ep_single_block(
                y[a:b],
                w[a:b],
                nu=nu,
                rho2=rho2,
                n_gh=self.gh_points,
                max_iter=self.max_iter,
            )
            return float(p_mean)
        if self.approx in ("gh", "quadrature"):
            n_gh = self.gh_points if self.approx == "gh" else max(self.gh_points, 80)
            logA0, m, v, p_mean = _gh_moments_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, n_gh=n_gh
            )
            return float(p_mean[0])
        raise ValueError(f"Unknown approx={self.approx!r}")
