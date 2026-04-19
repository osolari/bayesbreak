"""bayesbreak.families.logistic_normal

Bernoulli observations with a *logistic-normal* (a.k.a. logit-normal) segment prior.

Model (per segment q):
    y_t | theta_q ~ Bernoulli(sigmoid(theta_q))
    theta_q ~ Normal(nu, rho^2)

This is a non-conjugate family because the Bernoulli likelihood is not conjugate to a
Normal prior on the log-odds. The paper describes several block-level approximations
for the segment marginal likelihood A^0_{ij} = p(y_{(i,j]} | single segment).

We support the following approximations at the *segment* (block) level:
  - "laplace"   : Laplace approximation around the posterior mode.
  - "jj"        : Jaakkola--Jordan quadratic bound (variational lower bound).
  - "pg_vb"     : Polya--Gamma variational bound (equivalent precision updates to JJ).
  - "ep"        : Gaussian moment-matching via Gauss--Hermite quadrature (accurate).
  - "quadrature": Higher-accuracy Gauss--Hermite quadrature (reference / near-exact).

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
from typing import Literal

import numpy as np

from ..base import BayesBreakSegmenter
from ..utils import logsumexp

Approx = Literal["laplace", "jj", "pg_vb", "pg-vb", "ep", "quadrature"]


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
        nu: float | None = None,
        rho2: float | None = None,
        gh_points: int = 25,
        max_iter: int = 50,
    ):
        super().__init__(
            k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve
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
                logA0, m, v, p_mean = _gh_moments_block(S, N, nu=nu, rho2=rho2, n_gh=self.gh_points)
            elif self.approx == "quadrature":
                # Use more GH points by default for reference-quality numbers.
                n_gh = max(self.gh_points, 80)
                logA0, m, v, p_mean = _gh_moments_block(S, N, nu=nu, rho2=rho2, n_gh=n_gh)
            else:
                raise ValueError(
                    "approx must be one of {'laplace','jj','pg_vb','ep','quadrature'}, "
                    f"got {self.approx!r}."
                )

            lA0[i, j] = logA0
            A1[i, j] = np.exp(logA0) * p_mean

        # disallow empty segments
        diag = np.arange(n + 1)
        lA0[diag, diag] = -np.inf
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
            logA0, m, v, p_mean = _gh_moments_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, n_gh=self.gh_points
            )
            return float(p_mean[0])
        if self.approx == "quadrature":
            n_gh = max(self.gh_points, 80)
            logA0, m, v, p_mean = _gh_moments_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, n_gh=n_gh
            )
            return float(p_mean[0])
        raise ValueError(f"Unknown approx={self.approx!r}")
