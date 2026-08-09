"""Bernoulli BayesBreak family (Beta--Bernoulli conjugate, :math:`n_i \\equiv 1`)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..priors import PartitionPriorConfig
from ..utils import gammaln


class BayesBreakBernoulli(BayesBreakSegmenter):
    r"""Piecewise-constant Bernoulli segmentation with a Beta prior.

    .. math::
        y_i \mid p_q \sim \mathrm{Bernoulli}(p_q), \quad
        p_q \sim \mathrm{Beta}(\alpha, \beta).
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: str = "none",
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
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
        self.alpha = alpha
        self.beta = beta

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> dict[str, float]:
        if not self.estimate_hyper and self.alpha is not None and self.beta is not None:
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones_like(y, dtype=float)
            w_sum = float(y.size)

        mu = float(np.sum(w * y) / w_sum)
        var_obs = float(np.sum(w * (y - mu) ** 2) / w_sum) if y.size > 1 else 1e-4
        noise = mu * (1.0 - mu)
        var_p = max(var_obs - noise, 1e-12)

        tau = max(1e-8, mu * (1.0 - mu) / var_p - 1.0)
        alpha = mu * tau
        beta = (1.0 - mu) * tau

        if self.alpha is not None:
            alpha = float(self.alpha)
        if self.beta is not None:
            beta = float(self.beta)
        return {"alpha": float(alpha), "beta": float(beta)}

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha, beta = float(hyper["alpha"]), float(hyper["beta"])
        n = int(y.size)
        w = sample_weight

        S = np.zeros(n + 1, dtype=float)
        W = np.zeros(n + 1, dtype=float)
        S[1:] = np.cumsum(w * y)
        W[1:] = np.cumsum(w)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        logB_ab = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Ssum = S[j] - S[i]
            Wsum = W[j] - W[i]
            Fsum = Wsum - Ssum

            logB_post = gammaln(alpha + Ssum) + gammaln(beta + Fsum) - gammaln(alpha + beta + Wsum)
            lA0_ij = logB_post - logB_ab
            lA0[i, j] = lA0_ij

            mu_hat = (alpha + Ssum) / (alpha + beta + Wsum)
            A1[i, j] = np.exp(lA0_ij) * mu_hat

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: np.ndarray,
        hyper: dict[str, float],
        sample_weight: np.ndarray,
    ) -> float:
        alpha, beta = float(hyper["alpha"]), float(hyper["beta"])
        w = sample_weight[a:b]
        Ssum = float(np.sum(w * y[a:b]))
        Wsum = float(np.sum(w))
        return (alpha + Ssum) / (alpha + beta + Wsum)

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        assert (
            self.hyper_ is not None
            and self.sample_weight_ is not None
            and self._y_train_ is not None
        )
        alpha, beta = self.hyper_["alpha"], self.hyper_["beta"]
        w_train = self.sample_weight_[a:b]
        y_train = self._y_train_[a:b]
        S_post = float(np.sum(w_train * y_train))
        W_post = float(np.sum(w_train))
        alpha_post = alpha + S_post
        beta_post = beta + (W_post - S_post)
        p_hat = np.clip(alpha_post / (alpha_post + beta_post), 1e-12, 1.0 - 1e-12)
        y_new = np.asarray(y_new, dtype=float)
        return y_new * math.log(p_hat) + (1.0 - y_new) * math.log(1.0 - p_hat)
