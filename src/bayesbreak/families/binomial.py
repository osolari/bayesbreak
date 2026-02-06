"""Binomial BayesBreak family.

Model
-----
Within each segment ``q`` the observations are independent Binomial draws with
segment-specific success probability ``p_q``:

.. math::

    y_i \\mid p_q \\sim \\mathrm{Binomial}(n_i, p_q),\\quad
    p_q \\sim \\mathrm{Beta}(\alpha,\beta).

The per-observation number of trials ``n_i`` can be a scalar or an array of
length ``n``.

The Beta prior is conjugate, providing closed-form segment evidence and first
moment.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from bayesbreak.base import BayesBreakBase
from bayesbreak.utils import gammaln


class BayesBreakBinomial(BayesBreakBase):
    """Bayesian piecewise-constant Binomial regression.

    Parameters
    ----------
    k_max:
        Maximum number of segments.
    estimate_hyper:
        If ``True`` (default), estimate ``alpha`` and ``beta`` via an empirical
        Bayes procedure on the global proportion with a variance correction for
        Binomial sampling noise. If ``False``, the user must provide ``alpha``
        and ``beta``.
    regression_curve:
        ``"none"`` (default), ``"fixed_k"`` or ``"mix_k"``.
    n_trials:
        Number of trials per observation. Can be:

        - A scalar (same number of trials for every observation), or
        - An array-like of shape ``(n,)``.

    alpha, beta:
        Beta prior parameters.

    Notes
    -----
    The empirical Bayes estimator uses the Beta prior mean/variance relations:

    .. math::

        \\mu = \frac{\alpha}{\alpha+\beta},\\qquad
        \\mathrm{Var}(p) = \frac{\\mu(1-\\mu)}{\alpha+\beta+1}.

    We estimate a de-noised variance of per-observation proportions by
    subtracting the average Binomial noise term ``mu(1-mu)/n_i``.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        *,
        n_trials: Union[int, float, ArrayLike] = 1,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        super().__init__(
            k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve
        )
        self.n_trials = n_trials
        self.alpha = alpha
        self.beta = beta

        # Cache set during fit(). This is required for segment statistics.
        self._n_arr: Optional[np.ndarray] = None

    def _trials_array(self, n: int) -> np.ndarray:
        """Materialize the per-observation trials array."""
        if np.isscalar(self.n_trials):
            return np.full(n, float(self.n_trials), dtype=float)
        arr = np.asarray(self.n_trials, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"n_trials must be scalar or shape ({n},), got {arr.shape}")
        return arr

    # ----- subclass hooks -----

    def _estimate_global_params(self, y: np.ndarray, sample_weight: np.ndarray) -> Dict[str, float]:
        n = y.size
        n_arr = self._trials_array(n)
        self._n_arr = n_arr

        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError(
                    "estimate_hyper=False requires explicit alpha and beta for the Beta prior."
                )
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight

        # aggregated mean proportion (replicate-weighted)
        S = float(np.sum(w * y))
        T = float(np.sum(w * n_arr))
        mu = S / max(T, 1e-12)

        # estimate Var[p] from per-observation proportions with noise correction
        p_i = np.where(n_arr > 0, y / n_arr, mu)
        # replicate-weighted variance of per-observation proportions
        if n > 1:
            w_sum = float(np.sum(w))
            if w_sum <= 0.0:
                var_p_obs = 1e-4
            else:
                var_p_obs = float(np.sum(w * (p_i - mu) ** 2) / w_sum)
        else:
            var_p_obs = 1e-4

        # Binomial sampling noise contribution, weighted by replicate counts.
        denom_w = float(np.sum(w))
        noise = float(np.sum(w * (mu * (1.0 - mu)) / np.maximum(n_arr, 1.0)) / max(denom_w, 1e-12))
        var_p = max(var_p_obs - noise, 1e-12)

        # Solve for tau = alpha+beta
        tau = max(1e-8, mu * (1 - mu) / var_p - 1.0)
        alpha = mu * tau
        beta = (1 - mu) * tau

        # user overrides
        if self.alpha is not None:
            alpha = float(self.alpha)
        if self.beta is not None:
            beta = float(self.beta)
        return {"alpha": float(max(alpha, 1e-12)), "beta": float(max(beta, 1e-12))}

    def _compute_single_segment_stats(
        self, y: np.ndarray, hyper: Dict[str, float], sample_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._n_arr is None:
            # In normal operation _estimate_global_params has already set this.
            self._n_arr = self._trials_array(y.size)
        n_arr = self._n_arr

        alpha, beta = hyper["alpha"], hyper["beta"]
        n = y.size

        w = sample_weight

        S = np.zeros(n + 1, dtype=float)
        S[1:] = np.cumsum(w * y)
        N = np.zeros(n + 1, dtype=float)
        N[1:] = np.cumsum(w * n_arr)

        # sum log comb(n_i, y_i)
        Lcomb = gammaln(n_arr + 1.0) - gammaln(y + 1.0) - gammaln(n_arr - y + 1.0)
        Csum = np.zeros(n + 1, dtype=float)
        Csum[1:] = np.cumsum(w * Lcomb)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        # log B(alpha,beta)
        logB_ab = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Ssum = S[j] - S[i]
            Nsum = N[j] - N[i]
            Fsum = Nsum - Ssum
            const = Csum[j] - Csum[i]

            # log A0 = const + log B(alpha+S, beta+F) - log B(alpha,beta)
            logB_post = gammaln(alpha + Ssum) + gammaln(beta + Fsum) - gammaln(alpha + beta + Nsum)
            lA0_ij = const + (logB_post - logB_ab)
            lA0[i, j] = lA0_ij

            # E[p | segment] = (alpha + S) / (alpha + beta + N)
            log_E = np.log(alpha + Ssum) - np.log(alpha + beta + Nsum)
            A1[i, j] = np.exp(lA0_ij + log_E)

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: Dict[str, float], sample_weight: np.ndarray
    ) -> float:
        if self._n_arr is None:
            raise RuntimeError("Internal error: trials array not initialized.")
        alpha, beta = hyper["alpha"], hyper["beta"]
        w = sample_weight
        Ssum = float(np.sum(w[a:b] * y[a:b]))
        Nsum = float(np.sum(w[a:b] * self._n_arr[a:b]))
        return (alpha + Ssum) / (alpha + beta + Nsum)
