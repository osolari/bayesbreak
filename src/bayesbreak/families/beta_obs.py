r"""Continuous observations ``y in (0, 1)`` with a **known per-observation precision**
Beta likelihood and a Beta prior on the segment mean (§``sec:families``).

Model (per segment q):

.. math::
    y_t | \mu_q \sim \mathrm{Beta}(\phi_t \mu_q, \phi_t (1 - \mu_q)), \qquad
    \mu_q \sim \mathrm{Beta}(\alpha, \beta).

Per-observation precisions ``φ_t`` are the **primary** input; the constant-φ
case (``φ_t ≡ φ``) is the corresponding special case. This matches the
report's CpG-methylation pipeline where ``φ_t`` is the per-CpG read coverage.

The block evidence and posterior moments are computed via 1-D Gauss-Legendre
quadrature on ``μ ∈ (0, 1)``. The §``sec:families`` discussion does **not**
claim global log-concavity in ``μ`` — implementations should run a
node-refinement check (compare two node counts on a sample of blocks) when
``φ_t`` is small or the Beta posterior pushes against the endpoints.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..priors import PartitionPriorConfig
from ..utils import gammaln, logsumexp


def _legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Legendre nodes/weights on (0, 1)."""
    if n < 2:
        raise ValueError("Need at least 2 quadrature points.")
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


class BayesBreakBetaObs(BayesBreakSegmenter):
    r"""Beta-distributed observations with known precision and Beta prior on the mean.

    Parameters
    ----------
    k_max : int, default=50
    estimate_hyper : bool, default=True
    regression_curve : {"none", "fixed_k", "mix_k"}, default="none"
    phi : float or array-like of shape (n,), default=50.0
        Beta precision. **Per-observation `φ_t`** is the primary mode; pass
        a length-``n`` array for known per-CpG / per-position precisions
        (e.g. coverage). A scalar is the constant-precision special case.
    quadrature_points : int, default=32
        Number of Gauss-Legendre nodes for the 1-D integral over
        ``\mu \in (0, 1)``.
    alpha, beta : float, optional
        Optional fixed Beta prior parameters.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
        *,
        phi: float | ArrayLike = 50.0,
        quadrature_points: int = 32,
        alpha: float | None = None,
        beta: float | None = None,
        partition_prior: PartitionPriorConfig | None = None,
    ):
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
            partition_prior=partition_prior,
        )
        self.phi = phi
        self.quadrature_points = int(quadrature_points)
        self.alpha = alpha
        self.beta = beta

    def _phi_array(self, n: int) -> np.ndarray:
        if np.isscalar(self.phi):
            phi = float(self.phi)
            if phi <= 0:
                raise ValueError("phi must be > 0.")
            return np.full(n, phi, dtype=float)
        arr = np.asarray(self.phi, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"phi must be scalar or shape ({n},); got {arr.shape}.")
        if np.any(arr <= 0):
            raise ValueError("phi must be strictly positive.")
        return arr

    # ---- hyperparameters (EB) ----
    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> dict[str, float]:
        n = int(y.size)
        phi_arr = self._phi_array(n)
        self._phi_arr_ = phi_arr
        if (not self.estimate_hyper) and (self.alpha is not None) and (self.beta is not None):
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        wsum = float(np.sum(w))
        mu = float(np.sum(w * y) / max(1e-12, wsum))
        mu2 = float(np.sum(w * y * y) / max(1e-12, wsum))
        var_obs = max(0.0, mu2 - mu * mu)
        # Use mean phi for the prior shrinkage estimate.
        phi_mean = float(np.mean(phi_arr))
        var_p = max(var_obs - mu * (1.0 - mu) / (phi_mean + 1.0), 1e-12)

        tau = max(1e-8, mu * (1.0 - mu) / var_p - 1.0)
        alpha0 = mu * tau
        beta0 = (1.0 - mu) * tau
        if self.alpha is not None:
            alpha0 = float(self.alpha)
        if self.beta is not None:
            beta0 = float(self.beta)
        return {"alpha": float(alpha0), "beta": float(beta0)}

    def _compute_block_evidence(
        self,
        y: np.ndarray,
        hyper: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if np.any((y <= 0.0) | (y >= 1.0)):
            raise ValueError("BetaObs requires y in (0, 1) (open interval).")

        alpha0 = float(hyper["alpha"])
        beta0 = float(hyper["beta"])
        n = int(y.size)
        if not hasattr(self, "_phi_arr_"):
            self._phi_arr_ = self._phi_array(n)
        phi_arr = self._phi_arr_

        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != y.shape:
                raise ValueError("sample_weight must match y shape.")

        # Quadrature grid + prior log-density.
        n_quad = int(self.quadrature_points)
        mu_grid, w_grid = _legendre_nodes_weights(n_quad)
        log_Bab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)
        log_prior = (
            (alpha0 - 1.0) * np.log(mu_grid) + (beta0 - 1.0) * np.log(1.0 - mu_grid) - log_Bab
        )
        log_w_grid = np.log(w_grid)

        # Per-observation precomputations.
        log_y = np.log(y)
        log_1my = np.log(1.0 - y)
        # Prefix sums needed.
        Sw_logy = np.zeros(n + 1)  # Σ w_t log y_t
        Sw_log1my = np.zeros(n + 1)  # Σ w_t log(1 - y_t)
        Sw_phi_logy = np.zeros(n + 1)  # Σ w_t φ_t log y_t
        Sw_phi_log1my = np.zeros(n + 1)  # Σ w_t φ_t log(1 - y_t)
        Sw_logG_phi = np.zeros(n + 1)  # Σ w_t log Γ(φ_t)
        Sw_logy[1:] = np.cumsum(w * log_y)
        Sw_log1my[1:] = np.cumsum(w * log_1my)
        Sw_phi_logy[1:] = np.cumsum(w * phi_arr * log_y)
        Sw_phi_log1my[1:] = np.cumsum(w * phi_arr * log_1my)
        Sw_logG_phi[1:] = np.cumsum(w * gammaln(phi_arr))

        # The remaining terms ``Σ w_t log Γ(φ_t μ_g)`` and
        # ``Σ w_t log Γ(φ_t (1 - μ_g))`` are non-linear in μ_g and depend on
        # the *full per-observation* sequence. We precompute the (G, n+1)
        # prefix sums up front once, and then index into them per block.
        # Memory cost: 2 · G · (n+1) doubles.
        # log Γ(φ_t μ_g) for grid g, observation t: shape (n_quad, n).
        phi_mu = phi_arr[None, :] * mu_grid[:, None]  # (G, n)
        phi_1mu = phi_arr[None, :] * (1.0 - mu_grid)[:, None]  # (G, n)
        logG_phi_mu = gammaln(phi_mu)  # (G, n)
        logG_phi_1mu = gammaln(phi_1mu)  # (G, n)
        # Prefix sums weighted by w.
        S_logG_mu = np.zeros((n_quad, n + 1), dtype=float)
        S_logG_mu[:, 1:] = np.cumsum(w[None, :] * logG_phi_mu, axis=1)
        S_logG_1mu = np.zeros((n_quad, n + 1), dtype=float)
        S_logG_1mu[:, 1:] = np.cumsum(w[None, :] * logG_phi_1mu, axis=1)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        # The "once_term" added to every block at every grid node.
        once_term = log_prior + log_w_grid  # shape (G,)

        for i in range(n):
            j_arr = np.arange(i + 1, n + 1)
            # Block-aggregated linear-in-μ terms.
            d_logG_mu = S_logG_mu[:, j_arr] - S_logG_mu[:, i : i + 1]  # (G, J)
            d_logG_1mu = S_logG_1mu[:, j_arr] - S_logG_1mu[:, i : i + 1]  # (G, J)
            d_logG_phi = Sw_logG_phi[j_arr] - Sw_logG_phi[i]  # (J,)
            # Linear (μ, 1-μ) ·  φ_t log y_t  contributions.
            d_phi_logy = Sw_phi_logy[j_arr] - Sw_phi_logy[i]  # (J,)
            d_phi_log1my = Sw_phi_log1my[j_arr] - Sw_phi_log1my[i]  # (J,)
            d_logy = Sw_logy[j_arr] - Sw_logy[i]  # (J,)
            d_log1my = Sw_log1my[j_arr] - Sw_log1my[i]  # (J,)

            # log integrand on grid g for block (i, j]:
            # = once_term[g]
            #   + d_logG_phi[block]                           (no μ dep)
            #   - d_logG_mu[g, block] - d_logG_1mu[g, block]
            #   + μ_g · d_phi_logy[block]   + (1 - μ_g) · d_phi_log1my[block]
            #   - d_logy[block]   - d_log1my[block]
            mu_col = mu_grid[:, None]  # (G, 1)
            log_integrand = (
                once_term[:, None]
                + d_logG_phi[None, :]
                - d_logG_mu
                - d_logG_1mu
                + mu_col * d_phi_logy[None, :]
                + (1.0 - mu_col) * d_phi_log1my[None, :]
                - d_logy[None, :]
                - d_log1my[None, :]
            )  # (G, J)

            log_A0_ij = logsumexp(log_integrand, axis=0)  # (J,)
            lA0[i, j_arr] = log_A0_ij

            # Posterior mean E[μ | block] from normalised weights.
            w_norm = np.exp(log_integrand - log_A0_ij[None, :])  # (G, J)
            mu_mean = (w_norm * mu_col).sum(axis=0)  # (J,)
            A1[i, j_arr] = np.exp(log_A0_ij) * mu_mean

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: np.ndarray,
        hyper: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> float:
        n = int(y.size)
        if not hasattr(self, "_phi_arr_"):
            self._phi_arr_ = self._phi_array(n)
        phi_arr = self._phi_arr_

        alpha0 = float(hyper["alpha"])
        beta0 = float(hyper["beta"])
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        n_quad = int(self.quadrature_points)
        mu_grid, w_grid = _legendre_nodes_weights(n_quad)
        log_w = np.log(w_grid)
        log_Bab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)
        log_prior = (
            (alpha0 - 1.0) * np.log(mu_grid) + (beta0 - 1.0) * np.log(1.0 - mu_grid) - log_Bab
        )

        y_blk = y[a:b]
        w_blk = w[a:b]
        phi_blk = phi_arr[a:b]
        log_y = np.log(y_blk)
        log_1my = np.log(1.0 - y_blk)
        # log integrand at each μ_g (vectorised over g):
        # Σ w_t [ log Γ(φ_t) - log Γ(φ_t μ_g) - log Γ(φ_t (1-μ_g))
        #         + (φ_t μ_g - 1) log y_t + (φ_t (1-μ_g) - 1) log (1 - y_t) ]
        log_integrand = log_prior + log_w
        for g in range(n_quad):
            mu = mu_grid[g]
            ll = np.sum(
                w_blk
                * (
                    gammaln(phi_blk)
                    - gammaln(phi_blk * mu)
                    - gammaln(phi_blk * (1.0 - mu))
                    + (phi_blk * mu - 1.0) * log_y
                    + (phi_blk * (1.0 - mu) - 1.0) * log_1my
                )
            )
            log_integrand[g] += float(ll)
        log_norm = float(logsumexp(log_integrand))
        w_norm = np.exp(log_integrand - log_norm)
        return float(np.sum(w_norm * mu_grid))
