r"""bayesbreak.families.beta_obs

Continuous observations y in (0, 1) with a fixed-precision Beta likelihood and a
Beta prior on the segment mean.

Model (per segment q):
    y_t | mu_q ~ Beta(phi * mu_q, phi * (1 - mu_q))
    mu_q ~ Beta(alpha, beta)

This is *not conjugate* in the mean parameterization because the Beta likelihood is
not conjugate to a Beta prior on mu. Following the paper's "BetaObsBlock" algorithm,
we compute per-segment evidences and posterior means for mu using 1D Gauss--Legendre
quadrature on mu \in (0, 1).

The resulting family plugs directly into the generic BayesBreak DP via the standard
segment-evidence interface.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..base import BayesBreakBase
from ..utils import gammaln, logsumexp


def _legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Legendre nodes/weights on (0, 1)."""
    if n < 2:
        raise ValueError("Need at least 2 quadrature points.")
    x, w = np.polynomial.legendre.leggauss(n)  # nodes, weights on [-1, 1]
    # Map to (0, 1): mu = (x+1)/2, dmu = 1/2 dx
    mu = 0.5 * (x + 1.0)
    w_mu = 0.5 * w
    return mu, w_mu


def _betaobs_block_quadrature(
    Slogy: np.ndarray,
    Slog1my: np.ndarray,
    Sw: np.ndarray,
    *,
    alpha0: float,
    beta0: float,
    phi: float,
    n_quad: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (logA0, mu_post_mean) for many blocks via Gauss--Legendre quadrature.

    Each block is summarized by:
      - Sw       = sum_t w_t
      - Slogy    = sum_t w_t * log(y_t)
      - Slog1my  = sum_t w_t * log(1 - y_t)

    Returns
    -------
    logA0 : (m,) array
        log marginal likelihood for each block.
    mu_mean : (m,) array
        posterior mean E[mu | block] computed from the quadrature weights.
    """
    Sw = np.asarray(Sw, dtype=float)
    Slogy = np.asarray(Slogy, dtype=float)
    Slog1my = np.asarray(Slog1my, dtype=float)
    _m = Sw.size  # noqa: F841

    mu_grid, w_grid = _legendre_nodes_weights(n_quad)
    # Precompute prior log-density (up to additive constant) on grid.
    # log prior = (a-1) log mu + (b-1) log(1-mu) - log B(a,b)
    logB_ab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)
    log_prior = (alpha0 - 1.0) * np.log(mu_grid) + (beta0 - 1.0) * np.log(1.0 - mu_grid) - logB_ab

    # Likelihood terms that depend on mu through gamma functions.
    # log BetaPDF(y; a=phi*mu, b=phi*(1-mu))
    # = log Γ(phi) - log Γ(phi*mu) - log Γ(phi*(1-mu)) + (phi*mu-1) log y + (phi*(1-mu)-1) log(1-y)
    logG_phi = math.lgamma(phi)
    # Grid-dependent gamma terms:
    logG1 = gammaln(phi * mu_grid)
    logG2 = gammaln(phi * (1.0 - mu_grid))
    # (phi*mu - 1) and (phi*(1-mu) - 1)
    a_minus_1 = phi * mu_grid - 1.0
    b_minus_1 = phi * (1.0 - mu_grid) - 1.0

    # Build block-wise log integrands on the grid:
    # log integrand = Sw*logG_phi - Sw*(logG1+logG2) + a_minus_1*Slogy + b_minus_1*Slog1my + log_prior + log w_grid
    # We vectorize by broadcasting (m,1) with (1,n_quad).
    Sw_col = Sw.reshape(-1, 1)
    Slogy_col = Slogy.reshape(-1, 1)
    Slog1my_col = Slog1my.reshape(-1, 1)

    log_w = np.log(w_grid).reshape(1, -1)
    # mu-grid terms shape (1, n_quad)
    grid_term = (log_prior + log_w + logG_phi - logG1 - logG2).reshape(1, -1)
    # block-varying linear terms in logs
    block_term = (a_minus_1.reshape(1, -1) * Slogy_col) + (b_minus_1.reshape(1, -1) * Slog1my_col)
    log_integrand = Sw_col * grid_term + block_term

    # logA0 via log-sum-exp across grid
    logA0 = logsumexp(log_integrand, axis=1)

    # posterior mean E[mu] from normalized weights on grid
    # weights proportional to exp(log_integrand - logA0)
    w_norm = np.exp(log_integrand - logA0.reshape(-1, 1))
    mu_mean = w_norm @ mu_grid
    return logA0, mu_mean


class BayesBreakBetaObs(BayesBreakBase):
    """Beta observations with fixed precision and Beta prior on the segment mean.

    Parameters
    ----------
    k_max:
        Maximum number of segments.
    estimate_hyper:
        If True, estimate (alpha, beta) by moment matching on y (with a crude
        variance correction). The precision ``phi`` is treated as fixed.
    phi:
        Fixed Beta precision (concentration). Larger values imply less observation
        noise around the segment mean.
    quadrature_points:
        Number of Gauss--Legendre nodes used for the 1D mu integral.
    alpha, beta:
        Optional fixed prior parameters.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: str = "none",
        *,
        phi: float = 50.0,
        quadrature_points: int = 32,
        quad_points: Optional[int] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        super().__init__(
            k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve
        )
        if phi <= 0:
            raise ValueError("phi must be > 0")
        self.phi = float(phi)

        # Backwards/ergonomic alias.
        if quad_points is not None:
            quadrature_points = int(quad_points)
        self.quadrature_points = int(quadrature_points)
        self.alpha = alpha
        self.beta = beta

    # ---- hyperparameters (EB) ----
    def _estimate_global_params(
        self, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        if (not self.estimate_hyper) and (self.alpha is not None) and (self.beta is not None):
            return {"alpha": float(self.alpha), "beta": float(self.beta), "phi": float(self.phi)}

        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        # Weighted mean in (0,1)
        mu = float(np.sum(w * y) / max(1e-12, np.sum(w)))
        # Weighted variance with a crude correction for Beta noise ~ mu(1-mu)/(phi+1)
        mu2 = float(np.sum(w * y * y) / max(1e-12, np.sum(w)))
        var_obs = max(0.0, mu2 - mu * mu)
        var_p = max(var_obs - mu * (1.0 - mu) / (self.phi + 1.0), 1e-12)

        tau = max(1e-8, mu * (1.0 - mu) / var_p - 1.0)
        alpha0 = mu * tau
        beta0 = (1.0 - mu) * tau

        if self.alpha is not None:
            alpha0 = float(self.alpha)
        if self.beta is not None:
            beta0 = float(self.beta)
        return {"alpha": float(alpha0), "beta": float(beta0), "phi": float(self.phi)}

    # ---- segment stats via quadrature ----
    def _compute_single_segment_stats(
        self,
        y: np.ndarray,
        hyper: Dict[str, float],
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if np.any((y <= 0.0) | (y >= 1.0)):
            raise ValueError("BetaObs requires y in (0,1) (open interval).")

        alpha0 = float(hyper["alpha"])
        beta0 = float(hyper["beta"])
        phi = float(hyper.get("phi", self.phi))
        n = y.size

        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        # Prefix sums for sufficient statistics
        logy = np.log(y)
        log1my = np.log(1.0 - y)
        Sw = np.zeros(n + 1)
        Slogy = np.zeros(n + 1)
        Slog1my = np.zeros(n + 1)
        Sw[1:] = np.cumsum(w)
        Slogy[1:] = np.cumsum(w * logy)
        Slog1my[1:] = np.cumsum(w * log1my)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Sw_ij = Sw[j] - Sw[i]
            Slogy_ij = Slogy[j] - Slogy[i]
            Slog1my_ij = Slog1my[j] - Slog1my[i]

            logA0_ij, mu_mean_ij = _betaobs_block_quadrature(
                Slogy_ij,
                Slog1my_ij,
                Sw_ij,
                alpha0=alpha0,
                beta0=beta0,
                phi=phi,
                n_quad=self.quadrature_points,
            )
            lA0[i, j] = logA0_ij
            A1[i, j] = np.exp(logA0_ij) * mu_mean_ij

        diag = np.arange(n + 1)
        lA0[diag, diag] = -np.inf
        return lA0, A1

    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: np.ndarray,
        hyper: Dict[str, float],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        y = np.asarray(y, dtype=float)
        alpha0 = float(hyper["alpha"])
        beta0 = float(hyper["beta"])
        phi = float(hyper.get("phi", self.phi))
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        Sw = float(np.sum(w[a:b]))
        Slogy = float(np.sum(w[a:b] * np.log(y[a:b])))
        Slog1my = float(np.sum(w[a:b] * np.log(1.0 - y[a:b])))
        _, mu_mean = _betaobs_block_quadrature(
            np.array([Slogy]),
            np.array([Slog1my]),
            np.array([Sw]),
            alpha0=alpha0,
            beta0=beta0,
            phi=phi,
            n_quad=self.quadrature_points,
        )
        return float(mu_mean[0])
