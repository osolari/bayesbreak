"""Poisson BayesBreak family (Poisson--Gamma conjugate, exposure-weighted)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..utils import gammaln


class BayesBreakPoisson(BayesBreakSegmenter):
    r"""Piecewise-constant Poisson segmentation with a Gamma prior on the rate.

    Model
    -----

    .. math::
        y_i \mid \lambda_q \sim \mathrm{Poisson}(\lambda_q w_i), \quad
        \lambda_q \sim \mathrm{Gamma}(\alpha, \beta).
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
        alpha: float | None = None,
        beta: float | None = None,
    ):
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
        )
        self.alpha = alpha
        self.beta = beta

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> dict[str, float]:
        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError("estimate_hyper=False requires alpha and beta to be set.")
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones_like(y, dtype=float)
            w_sum = float(y.size)

        m = float(np.sum(w * y) / w_sum)
        v = float(np.sum(w * (y - m) ** 2) / max(w_sum, 1e-12)) if y.size > 1 else max(1.0, m)

        if v > m + 1e-12:
            alpha_hat = max(1e-8, m * m / (v - m))
        else:
            alpha_hat = 1e6
        beta_hat = max(1e-8, alpha_hat / max(m, 1e-12))

        if self.alpha is not None:
            alpha_hat = float(self.alpha)
        if self.beta is not None:
            beta_hat = float(self.beta)
        return {"alpha": alpha_hat, "beta": beta_hat}

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha, beta = hyper["alpha"], hyper["beta"]
        n = y.size
        w = sample_weight

        S = np.zeros(n + 1, dtype=float)
        S[1:] = np.cumsum(w * y)
        W = np.zeros(n + 1, dtype=float)
        W[1:] = np.cumsum(w)
        Lfac = np.zeros(n + 1, dtype=float)
        Lfac[1:] = np.cumsum(w * gammaln(y + 1.0))

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        log_beta_alpha = alpha * math.log(beta) - math.lgamma(alpha)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Wsum = W[j] - W[i]
            Ssum = S[j] - S[i]
            const = -(Lfac[j] - Lfac[i])

            logA0 = (
                const
                + log_beta_alpha
                + gammaln(alpha + Ssum)
                - (alpha + Ssum) * np.log(beta + Wsum)
            )
            lA0[i, j] = logA0
            logE = np.log(alpha + Ssum) - np.log(beta + Wsum)
            A1[i, j] = np.exp(logA0 + logE)

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        w = sample_weight
        Ssum = float(np.sum(w[a:b] * y[a:b]))
        Wsum = float(np.sum(w[a:b]))
        return (alpha + Ssum) / (beta + Wsum)

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        """Negative-Binomial posterior-predictive density on block ``(a, b]``."""

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
        beta_post = beta + W_post

        y_new = np.asarray(y_new, dtype=float)
        w_new = np.asarray(w_new, dtype=float)
        # Negative-Binomial(log-mean = log(w_new) + log(alpha_post/beta_post), r=alpha_post)
        # log p(y) = gammaln(alpha + y) - gammaln(y+1) - gammaln(alpha)
        #            + alpha*log(beta/(beta+w)) + y*log(w/(beta+w))
        r = alpha_post
        p = beta_post / (beta_post + w_new)
        return (
            gammaln(r + y_new)
            - gammaln(y_new + 1.0)
            - math.lgamma(r)
            + r * np.log(p)
            + y_new * np.log(1.0 - p)
        )
