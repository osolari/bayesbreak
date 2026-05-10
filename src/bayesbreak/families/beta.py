r"""Fractional Beta--Binomial for continuous observations in (0, 1)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..utils import gammaln


class BayesBreakBeta(BayesBreakSegmenter):
    r"""Piecewise-constant segmentation for :math:`y \in (0, 1)` via fractional Beta-Binomial.

    Each observation :math:`y_i` is mapped to pseudo-counts
    :math:`(s_i, f_i) = (\kappa y_i, \kappa (1 - y_i))`, preserving Beta-Binomial conjugacy.
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
        concentration: float = 50.0,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> None:
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
        )
        self.concentration = concentration
        self.alpha = alpha
        self.beta = beta

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> dict[str, float]:
        if float(self.concentration) <= 0:
            raise ValueError("concentration must be > 0")
        if np.any((y <= 0) | (y >= 1)):
            raise ValueError("BayesBreakBeta expects y strictly in (0, 1).")

        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError("When estimate_hyper=False, provide alpha and beta.")
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            w = np.ones_like(y, dtype=float)
            w_sum = float(y.size)

        kappa = float(self.concentration)
        mu = float(np.sum(w * y) / max(1e-12, w_sum))
        var_obs = float(np.sum(w * (y - mu) ** 2) / max(1e-12, w_sum)) if y.size > 1 else 1e-4
        var_p = max(var_obs - mu * (1 - mu) / kappa, 1e-12)
        tau = max(1e-8, mu * (1 - mu) / var_p - 1.0)
        alpha = mu * tau
        beta = (1 - mu) * tau

        if self.alpha is not None:
            alpha = float(self.alpha)
        if self.beta is not None:
            beta = float(self.beta)
        return {"alpha": float(alpha), "beta": float(beta)}

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha, beta = hyper["alpha"], hyper["beta"]
        kappa = float(self.concentration)
        n = y.size
        w = sample_weight

        s = kappa * y
        f = kappa * (1.0 - y)
        n_eff = np.full(n, kappa, dtype=float)

        S = np.zeros(n + 1)
        S[1:] = np.cumsum(w * s)
        F = np.zeros(n + 1)
        F[1:] = np.cumsum(w * f)
        N = np.zeros(n + 1)
        N[1:] = np.cumsum(w * n_eff)

        Lcomb = gammaln(n_eff + 1.0) - gammaln(s + 1.0) - gammaln(n_eff - s + 1.0)
        Csum = np.zeros(n + 1)
        Csum[1:] = np.cumsum(w * Lcomb)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        logB_ab = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Ssum = S[j] - S[i]
            Fsum = F[j] - F[i]
            Nsum = N[j] - N[i]
            const = Csum[j] - Csum[i]

            logB_post = gammaln(alpha + Ssum) + gammaln(beta + Fsum) - gammaln(alpha + beta + Nsum)
            lA0_ij = const + (logB_post - logB_ab)
            lA0[i, j] = lA0_ij

            mu_hat = (alpha + Ssum) / (alpha + beta + Nsum)
            A1[i, j] = np.exp(lA0_ij) * mu_hat

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        kappa = float(self.concentration)
        w = sample_weight[a:b]
        Ssum = float(np.sum(w * (kappa * y[a:b])))
        Nsum = float(np.sum(w) * kappa)
        return (alpha + Ssum) / (alpha + beta + Nsum)

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        """Beta posterior predictive evaluated at the posterior mode."""

        assert (
            self.hyper_ is not None
            and self.sample_weight_ is not None
            and self._y_train_ is not None
        )
        alpha, beta = self.hyper_["alpha"], self.hyper_["beta"]
        kappa = float(self.concentration)
        w_train = self.sample_weight_[a:b]
        y_train = self._y_train_[a:b]
        S_post = float(np.sum(w_train * kappa * y_train))
        N_post = float(np.sum(w_train) * kappa)
        alpha_post = alpha + S_post
        beta_post = alpha + beta + N_post - alpha_post
        # Beta density log p(y) = (a-1) log y + (b-1) log(1-y) - log B(a, b)
        y_new = np.clip(np.asarray(y_new, dtype=float), 1e-12, 1.0 - 1e-12)
        log_B = (
            math.lgamma(alpha_post) + math.lgamma(beta_post) - math.lgamma(alpha_post + beta_post)
        )
        return (alpha_post - 1.0) * np.log(y_new) + (beta_post - 1.0) * np.log(1.0 - y_new) - log_B
