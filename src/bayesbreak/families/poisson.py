"""Poisson BayesBreak family (Poisson--Gamma conjugate, exposure-weighted)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..priors import PartitionPriorConfig
from ..utils import gammaln


class BayesBreakPoisson(BayesBreakSegmenter):
    r"""Piecewise-constant Poisson segmentation with a Gamma prior on the rate.

    Model
    -----
    For segment ``q`` with constant rate :math:`\lambda_q`:

    .. math::
        y_i \mid \lambda_q, e_i \sim \mathrm{Poisson}(\lambda_q e_i), \quad
        \lambda_q \sim \mathrm{Gamma}(\alpha, \beta).

    Here :math:`e_i > 0` is the **known exposure** for observation :math:`i`:
    the mean of :math:`y_i` is :math:`\lambda_q e_i`.  This is a property of
    the sampling distribution, not a generalized-Bayes likelihood power.

    ``sample_weight`` in :meth:`fit` carries the per-observation exposures
    :math:`e_i`.  When omitted, all exposures default to 1 (unit exposure).
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
        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError("estimate_hyper=False requires alpha and beta to be set.")
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones_like(y, dtype=float)
            w_sum = float(y.size)

        m = float(np.sum(y) / w_sum)
        if y.size > 1:
            rates = y / np.maximum(w, 1e-300)
            v = float(np.sum(w * (rates - m) ** 2) / max(w_sum, 1e-12))
        else:
            v = max(1.0, m)

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
        # Generalized-Bayes power weights a_r (default ones = ordinary sampling).
        # Fallback to ones when the attribute is absent (wrappers that call
        # _compute_block_evidence directly without going through fit()).
        _lp = getattr(self, 'likelihood_power_', None)
        a = _lp if (_lp is not None and _lp.size == n) else np.ones(n, dtype=float)

        # Prefix tables for the general combined formula.
        # With a_r = 1 these reduce exactly to the Proposition 2 formula.
        SaY = np.zeros(n + 1, dtype=float)
        SaY[1:] = np.cumsum(a * y)                                      # Σa_r·y_r
        SaE = np.zeros(n + 1, dtype=float)
        SaE[1:] = np.cumsum(a * w)                                      # Σa_r·e_r
        aLfac = np.zeros(n + 1, dtype=float)
        aLfac[1:] = np.cumsum(a * gammaln(y + 1.0))                    # Σa_r·log Γ(y_r+1)
        # Observations with positive counts and zero exposure have probability
        # zero under any Poisson rate: P(y | e=0, λ) = 0 for y > 0.  Any such
        # observation makes the entire dataset incompatible with the model.
        # Zero-power (masked) observations are excluded from this check since
        # a_r = 0 eliminates their contribution entirely.
        impossible = (y > 0) & (w <= 0) & (a > 0)
        if np.any(impossible):
            bad_idx = np.where(impossible)[0].tolist()
            raise ValueError(
                f"BayesBreakPoisson: {len(bad_idx)} observation(s) have positive "
                f"counts with zero exposure at indices {bad_idx}. "
                "P(y | e=0, λ) = 0 for any y > 0 and any rate λ; these "
                "observations are incompatible with the Poisson model. "
                "Remove or correct them before fitting."
            )
        # For e_r = 0, y_r = 0: contribution to Σa_r·y_r·log(e_r) is 0 (unit
        # likelihood factor).  Use log(1)=0 as the safe replacement to avoid
        # 0·log(0) → nan in floating-point.
        log_e_safe = np.where(w > 0, np.log(np.maximum(w, 1e-300)), 0.0)
        SalogE = np.zeros(n + 1, dtype=float)
        SalogE[1:] = np.cumsum(a * y * log_e_safe)                    # Σa_r·y_r·log(e_r)

        upper = np.triu(np.ones((n + 1, n + 1), dtype=bool), k=1)
        SaEij = SaE[None, :] - SaE[:, None]
        SaYij = SaY[None, :] - SaY[:, None]
        aLij = aLfac[None, :] - aLfac[:, None]
        SalogEij = SalogE[None, :] - SalogE[:, None]

        log_beta_alpha = alpha * math.log(beta) - math.lgamma(alpha)
        # Restrict log/gammaln evaluations to admissible cells so inadmissible
        # (below-diagonal) entries never produce log(0) or invalid inputs.
        rate_num = np.where(upper, alpha + SaYij, 1.0)
        rate_den = np.where(upper, beta + SaEij, 1.0)
        logA0 = np.zeros((n + 1, n + 1), dtype=float)
        logA0[upper] = (
            SalogEij[upper]  # Σa_r·y_r·log(e_r) from Poisson base measure
            - aLij[upper]    # -Σa_r·log Γ(y_r+1) from Poisson base measure
            + log_beta_alpha
            + gammaln(rate_num[upper])
            - rate_num[upper] * np.log(rate_den[upper])
        )

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        lA0[upper] = logA0[upper]

        A1 = np.zeros((n + 1, n + 1), dtype=float)
        A1[upper] = rate_num[upper] / rate_den[upper]
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        _lp = getattr(self, 'likelihood_power_', None)
        lp = _lp[a:b] if _lp is not None else np.ones(b - a, dtype=float)
        Ysum = float(np.sum(lp * y[a:b]))
        Esum = float(np.sum(lp * sample_weight[a:b]))
        return (alpha + Ysum) / (beta + Esum)

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
        _lp = getattr(self, 'likelihood_power_', None)
        lp = _lp[a:b] if _lp is not None else np.ones(b - a, dtype=float)
        S_post = float(np.sum(lp * y_train))    # power-weighted count sum
        W_post = float(np.sum(lp * w_train))    # power-weighted exposure sum
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
