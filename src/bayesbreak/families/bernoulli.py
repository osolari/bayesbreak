from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..base import BayesBreakBase
from ..utils import gammaln


class BayesBreakBernoulli(BayesBreakBase):
    """Bernoulli segments with a Beta prior (a.k.a. Beta–Bernoulli).

    This family is a special case of Beta–Binomial with n_i \\equiv 1:

    .. math::

        y_i \\mid p \\sim \\mathrm{Bernoulli}(p),\\qquad
        p \\sim \\mathrm{Beta}(\alpha, \beta).

    ``sample_weight`` is supported as a power-likelihood:
    each observation contributes multiplicatively as
    :math:`p(y_i\\mid p)^{w_i}`. When ``w_i`` are integers, this is exactly
    equivalent to repeating observation ``y_i`` ``w_i`` times.

    Hyperparameters
    --------------
    alpha, beta:
        Beta prior parameters. If ``estimate_hyper=True`` and ``alpha``/``beta``
        are not provided, a simple empirical-Bayes moment estimator is used.
        For Bernoulli data this tends to produce a concentrated prior when the
        global rate is stable; for diffuse priors, set ``alpha=beta=1``.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: str = "none",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        super().__init__(
            k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve
        )
        self.alpha = alpha
        self.beta = beta

    # ---- hyperparameters (EB) ----
    def _estimate_global_params(self, y: np.ndarray, sample_weight: np.ndarray) -> Dict[str, float]:
        if not self.estimate_hyper and self.alpha is not None and self.beta is not None:
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        w = sample_weight
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones_like(y, dtype=float)
            w_sum = float(y.size)

        # global mean of a Bernoulli rate
        mu = float(np.sum(w * y) / w_sum)

        # heuristic EB: treat per-observation proportions p_i in {0,1}
        # and subtract binomial noise mu(1-mu) (n_i=1) to estimate Var[p].
        var_obs = float(np.sum(w * (y - mu) ** 2) / w_sum) if y.size > 1 else 1e-4
        noise = mu * (1.0 - mu)
        var_p = max(var_obs - noise, 1e-12)

        tau = max(1e-8, mu * (1.0 - mu) / var_p - 1.0)  # alpha + beta
        alpha = mu * tau
        beta = (1.0 - mu) * tau

        # user overrides
        if self.alpha is not None:
            alpha = float(self.alpha)
        if self.beta is not None:
            beta = float(self.beta)
        return {"alpha": float(alpha), "beta": float(beta)}

    # ---- single-segment stats ----
    def _compute_single_segment_stats(
        self, y: np.ndarray, hyper: Dict[str, float], sample_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha, beta = float(hyper["alpha"]), float(hyper["beta"])
        n = int(y.size)
        w = sample_weight

        # weighted successes and weighted counts (n_i ≡ 1)
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

    # ---- posterior mean for PC fit ----
    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: np.ndarray,
        hyper: Dict[str, float],
        sample_weight: np.ndarray,
    ) -> float:
        alpha, beta = float(hyper["alpha"]), float(hyper["beta"])
        w = sample_weight[a:b]
        Ssum = float(np.sum(w * y[a:b]))
        Wsum = float(np.sum(w))
        return (alpha + Ssum) / (alpha + beta + Wsum)


# Backward-friendly alias: some users refer to this family as "logistic"
BayesBreakLogistic = BayesBreakBernoulli
