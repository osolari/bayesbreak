"""Gaussian BayesBreak family (Normal--Normal conjugate, weighted)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..priors import PartitionPriorConfig


class BayesBreakGaussian(BayesBreakSegmenter):
    r"""Piecewise-constant Gaussian segmentation with a Normal prior on the mean.

    Model
    -----
    For segment ``q`` with constant mean :math:`\mu_q`:

    .. math::
        y_i \mid \mu_q, w_i \sim \mathcal{N}(\mu_q,\; \sigma^2 / w_i), \quad
        \mu_q \sim \mathcal{N}(\nu, \rho^2).

    Here :math:`w_i > 0` is the **known observation precision** (inverse
    variance scale factor): the sampling variance of :math:`y_i` is
    :math:`\sigma^2 / w_i`.  This is a property of the sampling distribution,
    not a generalized-Bayes likelihood power.  For an optional power weight
    :math:`a_i \neq 1` that raises the entire log-likelihood contribution to a
    power, supply a separate mechanism; do not pass it as ``sample_weight``.

    ``sample_weight`` in :meth:`fit` carries the per-observation precisions
    :math:`w_i`.  When omitted, all precisions default to 1 (unit precision,
    variance :math:`\sigma^2`).

    Hyperparameters :math:`(\nu, \rho^2, \sigma^2)` are either estimated via a
    light-weight empirical-Bayes routine or fixed by the user.

    Parameters
    ----------
    k_max : int, default=50
        Maximum segment count.
    estimate_hyper : bool, default=True
        If False, user must supply ``nu``, ``rho2``, ``sigma2``.
    rho_estimation : {"cov", "var"}, default="cov"
        Moment estimator for ``rho2``.
    regression_curve : {"none", "fixed_k", "mix_k"}, default="none"
        Bayesian regression-curve mode.
    nu, rho2, sigma2 : float, optional
        Fixed hyperparameters (override any estimate).
    """

    # The Gaussian segment-mean target ``E[μ | (i, j]]`` is sign-changing;
    # ``block_first_moment_[i, j]`` stores that posterior mean directly in
    # linear space (§5 5-C1).
    MOMENT_SIGN_CONTRACT: str = "signed"

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        rho_estimation: Literal["cov", "var"] = "cov",
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
        nu: float | None = None,
        rho2: float | None = None,
        sigma2: float | None = None,
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
        self.rho_estimation = rho_estimation
        self.nu = nu
        self.rho2 = rho2
        self.sigma2 = sigma2

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> dict[str, float]:
        if not self.estimate_hyper:
            missing = [
                name
                for name, v in (("nu", self.nu), ("rho2", self.rho2), ("sigma2", self.sigma2))
                if v is None
            ]
            if missing:
                raise ValueError(
                    "estimate_hyper=False requires fixed hyperparameters; missing: "
                    + ", ".join(missing)
                )
            assert self.nu is not None and self.rho2 is not None and self.sigma2 is not None
            return {"nu": float(self.nu), "rho2": float(self.rho2), "sigma2": float(self.sigma2)}

        w = sample_weight
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            raise ValueError("sample_weight must have positive total weight.")

        nu_hat = float(np.sum(w * y) / w_sum)

        dy = np.diff(y)
        w_diff = 0.5 * (w[:-1] + w[1:])
        denom = 2.0 * max(float(np.sum(w_diff)), 1.0)
        sigma2_hat = float(np.sum(w_diff * (dy * dy)) / denom)

        if self.rho_estimation == "cov":
            y0 = y[:-1] - nu_hat
            y1 = y[1:] - nu_hat
            denom_cov = max(float(np.sum(w_diff)), 1.0)
            cov = float(np.sum(w_diff * y0 * y1) / denom_cov)
            rho2_hat = abs(cov)
        else:
            yc = y - nu_hat
            rho2_hat = float(np.sum(w * (yc * yc)) / w_sum)

        if self.nu is not None:
            nu_hat = float(self.nu)
        if self.rho2 is not None:
            rho2_hat = float(self.rho2)
        if self.sigma2 is not None:
            sigma2_hat = float(self.sigma2)

        rho2_hat = max(rho2_hat, 1e-12)
        sigma2_hat = max(sigma2_hat, 1e-12)
        return {"nu": nu_hat, "rho2": rho2_hat, "sigma2": sigma2_hat}

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nu, rho2, sigma2 = hyper["nu"], hyper["rho2"], hyper["sigma2"]
        n = y.size
        w = sample_weight
        # Generalized-Bayes power weights a_r (default ones = ordinary sampling).
        # Fallback to ones when the attribute is absent (wrappers that call
        # _compute_block_evidence directly without going through fit()).
        _lp = getattr(self, 'likelihood_power_', None)
        a = _lp if (_lp is not None and _lp.size == n) else np.ones(n, dtype=float)

        # Prefix tables for the general combined formula.
        # With a_r = 1 these reduce exactly to the Proposition 1 formula.
        SA = np.zeros(n + 1, dtype=float)
        SA[1:] = np.cumsum(a)                                           # Σa_r
        SAW = np.zeros(n + 1, dtype=float)
        SAW[1:] = np.cumsum(a * w)                                      # Σa_r·w_r
        SAWy = np.zeros(n + 1, dtype=float)
        SAWy[1:] = np.cumsum(a * w * y)                                 # Σa_r·w_r·y_r
        SAWy2 = np.zeros(n + 1, dtype=float)
        SAWy2[1:] = np.cumsum(a * w * (y * y))                         # Σa_r·w_r·y_r²
        SAlogW = np.zeros(n + 1, dtype=float)
        SAlogW[1:] = np.cumsum(a * np.log(np.maximum(w, 1e-300)))      # Σa_r·log(w_r)

        # Vectorised outer-difference construction of the (n+1, n+1) block
        # tables. Upper triangle j > i is admissible; the strict lower
        # triangle and the diagonal receive the sentinel -inf.
        upper = np.triu(np.ones((n + 1, n + 1), dtype=bool), k=1)

        SAWij = SAW[None, :] - SAW[:, None]
        SAWyij = SAWy[None, :] - SAWy[:, None]
        SAWy2ij = SAWy2[None, :] - SAWy2[:, None]
        SAlogWij = SAlogW[None, :] - SAlogW[:, None]
        SAij = SA[None, :] - SA[:, None]       # effective observation count Σa_r

        log2pi_sigma = math.log(2.0 * math.pi * sigma2)
        ysum_c = SAWyij - nu * SAWij
        y2sum = SAWy2ij - 2.0 * nu * SAWyij + (nu * nu) * SAWij
        # Guard log1p / division on inadmissible cells where SAWij <= 0.
        SAWij_safe = np.where(upper, np.maximum(SAWij, 1e-300), 1.0)
        # Combined normalization: Σa_r·(-½·log(2πσ²)) + Σa_r·½·log(w_r).
        # For a_r=1: reduces to −N_B/2·log(2πσ²) + ½·Σlog(w_r) (Proposition 1).
        # For w_r=1: reduces to −(Σa_r)/2·log(2πσ²) (generalized-Bayes).
        term1 = -0.5 * SAij * log2pi_sigma + 0.5 * SAlogWij
        term2 = -0.5 * np.log1p(SAWij_safe * (rho2 / sigma2))
        denom = SAWij_safe + (sigma2 / rho2)
        term3 = 0.5 / sigma2 * (ysum_c * ysum_c / denom - y2sum)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        lA0[upper] = (term1 + term2 + term3)[upper]

        mu_hat = (rho2 * SAWyij + sigma2 * nu) / (rho2 * SAWij_safe + sigma2)
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        A1[upper] = mu_hat[upper]

        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        nu, rho2, sigma2 = hyper["nu"], hyper["rho2"], hyper["sigma2"]
        w = sample_weight[a:b]
        _lp = getattr(self, 'likelihood_power_', None)
        lp = _lp[a:b] if _lp is not None else np.ones(b - a, dtype=float)
        Wseg = float(np.sum(lp * w))
        Syseg = float(np.sum(lp * w * y[a:b]))
        return (rho2 * Syseg + sigma2 * nu) / (rho2 * Wseg + sigma2)

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        """Gaussian posterior-predictive log-density for the block ``(a, b]``.

        Under the fitted Normal-Normal posterior, the predictive for a new
        observation is :math:`\\mathcal{N}(\\mu_B, \\sigma^2 / w_{\\text{new}} + \\rho^2_B)`,
        where :math:`\\rho^2_B` is the posterior variance of the segment mean.
        """

        assert self.hyper_ is not None
        nu, rho2, sigma2 = self.hyper_["nu"], self.hyper_["rho2"], self.hyper_["sigma2"]
        assert self.sample_weight_ is not None and self._y_train_ is not None
        w_train = self.sample_weight_[a:b]
        y_train = self._y_train_[a:b]
        _lp = getattr(self, 'likelihood_power_', None)
        lp = _lp[a:b] if _lp is not None else np.ones(b - a, dtype=float)
        Wseg = float(np.sum(lp * w_train))
        Syseg = float(np.sum(lp * w_train * y_train))
        mu_post = (rho2 * Syseg + sigma2 * nu) / (rho2 * Wseg + sigma2)
        rho2_post = (rho2 * sigma2) / (rho2 * Wseg + sigma2)

        w_new = np.asarray(w_new, dtype=float)
        var_new = sigma2 / np.maximum(w_new, 1e-12) + rho2_post
        return -0.5 * (np.log(2.0 * math.pi * var_new) + (y_new - mu_post) ** 2 / var_new)
