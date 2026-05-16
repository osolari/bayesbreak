"""Gaussian BayesBreak family (Normal--Normal conjugate, weighted)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter


class BayesBreakGaussian(BayesBreakSegmenter):
    r"""Piecewise-constant Gaussian segmentation with a Normal prior on the mean.

    Model
    -----
    For segment ``q`` with constant mean :math:`\mu_q`:

    .. math::
        y_i \mid \mu_q \sim \mathcal{N}(\mu_q, \sigma^2 / w_i), \quad
        \mu_q \sim \mathcal{N}(\nu, \rho^2).

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

    # The Gaussian segment-mean target ``E[μ | (i, j]]`` is sign-changing, so
    # ``block_first_moment_[i, j] = A^{(0)}_{ij} · μ̂_{ij}`` is stored as
    # signed-linear and must not be passed through ``log`` blindly (§5 5-C1).
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
    ):
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
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

        W = np.zeros(n + 1, dtype=float)
        W[1:] = np.cumsum(w)
        Sy = np.zeros(n + 1, dtype=float)
        Sy[1:] = np.cumsum(w * y)
        Sy2 = np.zeros(n + 1, dtype=float)
        Sy2[1:] = np.cumsum(w * (y * y))

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        log2pi_sigma = math.log(2.0 * math.pi * sigma2)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Wseg = W[j] - W[i]
            Syseg = Sy[j] - Sy[i]
            Sy2seg = Sy2[j] - Sy2[i]
            ysum_c = Syseg - nu * Wseg
            y2sum = Sy2seg - 2.0 * nu * Syseg + (nu * nu) * Wseg

            term1 = -0.5 * Wseg * log2pi_sigma
            term2 = -0.5 * np.log1p(Wseg * (rho2 / sigma2))
            denom = Wseg + (sigma2 / rho2)
            term3 = 0.5 / sigma2 * (ysum_c * ysum_c / denom - y2sum)
            logA0_ij = term1 + term2 + term3
            lA0[i, j] = logA0_ij

            mu_hat = (rho2 * Syseg + sigma2 * nu) / (rho2 * Wseg + sigma2)
            A1[i, j] = np.exp(logA0_ij) * mu_hat

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        nu, rho2, sigma2 = hyper["nu"], hyper["rho2"], hyper["sigma2"]
        w = sample_weight[a:b]
        Wseg = float(np.sum(w))
        Syseg = float(np.sum(w * y[a:b]))
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
        Wseg = float(np.sum(w_train))
        Syseg = float(np.sum(w_train * y_train))
        mu_post = (rho2 * Syseg + sigma2 * nu) / (rho2 * Wseg + sigma2)
        rho2_post = (rho2 * sigma2) / (rho2 * Wseg + sigma2)

        w_new = np.asarray(w_new, dtype=float)
        var_new = sigma2 / np.maximum(w_new, 1e-12) + rho2_post
        return -0.5 * (np.log(2.0 * math.pi * var_new) + (y_new - mu_post) ** 2 / var_new)
