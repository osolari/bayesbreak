"""Binomial BayesBreak family (Beta--Binomial conjugate)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..utils import gammaln


class BayesBreakBinomial(BayesBreakSegmenter):
    r"""Piecewise-constant Binomial segmentation with a Beta prior.

    .. math::
        y_i \mid p_q \sim \mathrm{Binomial}(n_i, p_q), \quad
        p_q \sim \mathrm{Beta}(\alpha, \beta).
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
        n_trials: int | float | ArrayLike = 1,
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
        self.n_trials = n_trials
        self.alpha = alpha
        self.beta = beta

    def _trials_array(self, n: int) -> np.ndarray:
        if np.isscalar(self.n_trials):
            return np.full(n, float(self.n_trials), dtype=float)
        arr = np.asarray(self.n_trials, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"n_trials must be scalar or shape ({n},), got {arr.shape}")
        return arr

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> dict[str, float]:
        n = y.size
        n_arr = self._trials_array(n)
        self._n_arr_ = n_arr  # cached fitted attribute for block evidence

        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError(
                    "estimate_hyper=False requires explicit alpha and beta for the Beta prior."
                )
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight
        S = float(np.sum(w * y))
        T = float(np.sum(w * n_arr))
        mu = S / max(T, 1e-12)

        p_i = np.where(n_arr > 0, y / n_arr, mu)
        if n > 1:
            w_sum = float(np.sum(w))
            var_p_obs = float(np.sum(w * (p_i - mu) ** 2) / w_sum) if w_sum > 0 else 1e-4
        else:
            var_p_obs = 1e-4

        denom_w = float(np.sum(w))
        noise = float(np.sum(w * (mu * (1.0 - mu)) / np.maximum(n_arr, 1.0)) / max(denom_w, 1e-12))
        var_p = max(var_p_obs - noise, 1e-12)
        tau = max(1e-8, mu * (1 - mu) / var_p - 1.0)
        alpha = mu * tau
        beta = (1 - mu) * tau

        if self.alpha is not None:
            alpha = float(self.alpha)
        if self.beta is not None:
            beta = float(self.beta)
        return {"alpha": float(max(alpha, 1e-12)), "beta": float(max(beta, 1e-12))}

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "_n_arr_"):
            self._n_arr_ = self._trials_array(y.size)
        n_arr = self._n_arr_
        alpha, beta = hyper["alpha"], hyper["beta"]
        n = y.size
        w = sample_weight

        S = np.zeros(n + 1, dtype=float)
        S[1:] = np.cumsum(w * y)
        N = np.zeros(n + 1, dtype=float)
        N[1:] = np.cumsum(w * n_arr)

        Lcomb = gammaln(n_arr + 1.0) - gammaln(y + 1.0) - gammaln(n_arr - y + 1.0)
        Csum = np.zeros(n + 1, dtype=float)
        Csum[1:] = np.cumsum(w * Lcomb)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        logB_ab = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Ssum = S[j] - S[i]
            Nsum = N[j] - N[i]
            Fsum = Nsum - Ssum
            const = Csum[j] - Csum[i]

            logB_post = gammaln(alpha + Ssum) + gammaln(beta + Fsum) - gammaln(alpha + beta + Nsum)
            lA0_ij = const + (logB_post - logB_ab)
            lA0[i, j] = lA0_ij

            log_E = np.log(alpha + Ssum) - np.log(alpha + beta + Nsum)
            A1[i, j] = np.exp(lA0_ij + log_E)

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        w = sample_weight
        Ssum = float(np.sum(w[a:b] * y[a:b]))
        Nsum = float(np.sum(w[a:b] * self._n_arr_[a:b]))
        return (alpha + Ssum) / (alpha + beta + Nsum)

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        """Beta-Binomial posterior-predictive log-density on block ``(a, b]``.

        For simplicity, we use ``n_trials = 1`` (Bernoulli-like) for new data
        unless the caller provides integer ``y_new`` with an implicit interpretation.
        """

        assert (
            self.hyper_ is not None
            and self.sample_weight_ is not None
            and self._y_train_ is not None
        )
        alpha, beta = self.hyper_["alpha"], self.hyper_["beta"]
        w_train = self.sample_weight_[a:b]
        y_train = self._y_train_[a:b]
        n_train = self._n_arr_[a:b]
        S_post = float(np.sum(w_train * y_train))
        N_post = float(np.sum(w_train * n_train))
        alpha_post = alpha + S_post
        beta_post = beta + (N_post - S_post)

        y_new = np.asarray(y_new, dtype=float)
        # Assume one trial per new point (commonest Bernoulli-test use case).
        p_hat = alpha_post / (alpha_post + beta_post)
        p_hat = np.clip(p_hat, 1e-12, 1.0 - 1e-12)
        return y_new * math.log(p_hat) + (1.0 - y_new) * math.log(1.0 - p_hat)
