"""Poisson BayesBreak family.

Model
-----
Within each segment ``q`` the observations are i.i.d. Poisson:

.. math::

    y_i \mid \lambda_q \sim \mathrm{Poisson}(\lambda_q),\quad
    \lambda_q \sim \mathrm{Gamma}(\alpha,\beta).

We use the (shape, rate) Gamma parameterization.

The segment marginal likelihood and first moment are available in closed form.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Literal

import numpy as np

from bayesbreak.base import BayesBreakBase
from bayesbreak.utils import gammaln


class BayesBreakPoisson(BayesBreakBase):
    """Bayesian piecewise-constant Poisson regression.

    Parameters
    ----------
    k_max:
        Maximum number of segments.
    estimate_hyper:
        If ``True`` (default), estimate ``alpha`` and ``beta`` using a simple
        method-of-moments empirical Bayes procedure. If ``False``, the user must
        provide both ``alpha`` and ``beta``.
    regression_curve:
        ``"none"`` (default), ``"fixed_k"`` or ``"mix_k"``.
    alpha, beta:
        Gamma prior parameters (shape, rate).

    Notes
    -----
    The empirical Bayes estimator uses the Gamma--Poisson relationship

    .. math::

        \mathbb{E}[Y] = m,\qquad \mathrm{Var}(Y) = m + m^2/\alpha.

    which implies ``alpha = m^2/(v - m)`` when ``v > m``.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        *,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        super().__init__(k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve)
        self.alpha = alpha
        self.beta = beta

    def _estimate_global_params(self, y: np.ndarray) -> Dict[str, float]:
        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError("estimate_hyper=False requires alpha and beta to be set.")
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        m = float(np.mean(y))
        v = float(np.var(y, ddof=1)) if y.size > 1 else max(1.0, m)

        if v > m + 1e-12:
            alpha_hat = max(1e-8, m * m / (v - m))
        else:
            alpha_hat = 1e6  # nearly fixed-rate prior when no overdispersion
        beta_hat = max(1e-8, alpha_hat / max(m, 1e-12))

        # User overrides take precedence.
        if self.alpha is not None:
            alpha_hat = float(self.alpha)
        if self.beta is not None:
            beta_hat = float(self.beta)

        return {"alpha": alpha_hat, "beta": beta_hat}

    def _compute_single_segment_stats(self, y: np.ndarray, hyper: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        alpha, beta = hyper["alpha"], hyper["beta"]
        n = y.size

        # Prefix sums: counts and log-factorials.
        S = np.zeros(n + 1, dtype=float)
        S[1:] = np.cumsum(y)

        Lfac = np.zeros(n + 1, dtype=float)
        Lfac[1:] = np.cumsum(gammaln(y + 1.0))

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        log_beta_alpha = alpha * math.log(beta) - math.lgamma(alpha)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            d = j - i
            Ssum = S[j] - S[i]

            const = -(Lfac[j] - Lfac[i])  # -sum log(y!)

            # log A^0 = -sum log y! + alpha log beta - log Gamma(alpha)
            #          + log Gamma(alpha + S) - (alpha + S) log(beta + d)
            logA0 = const + log_beta_alpha + gammaln(alpha + Ssum) - (alpha + Ssum) * np.log(beta + d)
            lA0[i, j] = logA0

            # E[lambda | segment] = (alpha + S) / (beta + d)
            logE = np.log(alpha + Ssum) - np.log(beta + d)
            A1[i, j] = np.exp(logA0 + logE)

        idx = np.arange(n + 1)
        lA0[idx, idx] = -np.inf
        return lA0, A1

    def _segment_posterior_mean(self, a: int, b: int, y: np.ndarray, hyper: Dict[str, float]) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        Ssum = float(np.sum(y[a:b]))
        d = b - a
        return (alpha + Ssum) / (beta + d)
