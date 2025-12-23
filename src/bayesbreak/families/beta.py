"""Continuous Beta-valued observations via a fractional Beta--Binomial.

This family targets real-valued observations in ``(0, 1)`` (e.g., methylation
rates, probabilities, proportions) by mapping each observation to pseudo-counts
and reusing Beta--Binomial conjugacy.

Model (fractional Beta--Binomial)
--------------------------------
Let ``kappa > 0`` denote a concentration parameter (pseudo-trials). For each
observation ``y_i \in (0,1)`` define

.. math::

    s_i = \kappa y_i,\qquad f_i = \kappa (1-y_i).

Treating ``(s_i, f_i)`` as fractional successes/failures, the segment-level
likelihood uses generalized binomial coefficients via the Gamma function.
A Beta prior on the segment probability ``p_q`` yields closed-form segment
marginal likelihoods.

Notes
-----
This is a pragmatic conjugate approximation. When the data truly arise from a
Beta distribution with varying concentration, consider a dedicated Beta
likelihood; however that breaks conjugacy and requires approximate inference.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Literal

import numpy as np

from bayesbreak.base import BayesBreakBase
from bayesbreak.utils import gammaln


class BayesBreakBeta(BayesBreakBase):
    """Bayesian piecewise-constant regression for ``y in (0,1)``.

    Parameters
    ----------
    k_max:
        Maximum number of segments.
    estimate_hyper:
        If ``True`` (default), estimate Beta prior parameters by empirical Bayes.
        If ``False``, the user must provide ``alpha`` and ``beta``.
    regression_curve:
        ``"none"`` (default), ``"fixed_k"`` or ``"mix_k"``.
    concentration:
        The pseudo-trial count :math:`\kappa`. Larger values correspond to less
        observation noise around the underlying segment mean.
    alpha, beta:
        Beta prior parameters.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        *,
        concentration: float = 50.0,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        if concentration <= 0:
            raise ValueError("concentration must be > 0")
        super().__init__(k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve)
        self.concentration = float(concentration)
        self.alpha = alpha
        self.beta = beta

    # ---- subclass hooks ----

    def _estimate_global_params(self, y: np.ndarray) -> Dict[str, float]:
        # validate domain early (helps catch data bugs before expensive DP)
        if np.any((y <= 0) | (y >= 1)):
            raise ValueError("BayesBreakBeta expects y strictly in (0,1).")

        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError("When estimate_hyper=False, provide alpha and beta.")
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        kappa = self.concentration
        mu = float(np.mean(y))
        var_obs = float(np.var(y, ddof=1)) if y.size > 1 else 1e-4

        # Var[y] ≈ Var[p] + mu(1-mu)/kappa
        var_p = max(var_obs - mu * (1 - mu) / kappa, 1e-12)
        tau = max(1e-8, mu * (1 - mu) / var_p - 1.0)  # tau = alpha+beta
        alpha = mu * tau
        beta = (1 - mu) * tau

        # user overrides
        if self.alpha is not None:
            alpha = float(self.alpha)
        if self.beta is not None:
            beta = float(self.beta)
        return {"alpha": float(alpha), "beta": float(beta)}

    def _compute_single_segment_stats(
        self, y: np.ndarray, hyper: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        alpha, beta = hyper["alpha"], hyper["beta"]
        kappa = self.concentration
        n = y.size

        s = kappa * y
        f = kappa * (1.0 - y)
        n_eff = np.full(n, kappa, dtype=float)

        S = np.zeros(n + 1)
        S[1:] = np.cumsum(s)
        F = np.zeros(n + 1)
        F[1:] = np.cumsum(f)
        N = np.zeros(n + 1)
        N[1:] = np.cumsum(n_eff)

        # generalized log comb(kappa, s) using Γ
        Lcomb = gammaln(n_eff + 1.0) - gammaln(s + 1.0) - gammaln(n_eff - s + 1.0)
        Csum = np.zeros(n + 1)
        Csum[1:] = np.cumsum(Lcomb)

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

            # E[p | y] = (α + S) / (α + β + N)
            mu_hat = (alpha + Ssum) / (alpha + beta + Nsum)
            A1[i, j] = np.exp(lA0_ij) * mu_hat

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: Dict[str, float]
    ) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        kappa = self.concentration
        Ssum = float(np.sum(kappa * y[a:b]))
        Nsum = float(kappa * (b - a))
        return (alpha + Ssum) / (alpha + beta + Nsum)
