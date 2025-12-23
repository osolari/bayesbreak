"""Gaussian BayesBreak family.

Implements the conjugate Normal--Normal model used in the original BayesBreak
formulation.

Model
-----
For segment ``q`` with constant mean ``mu_q``:

.. math::

    y_i \mid \mu_q \sim \mathcal{N}(\mu_q,\sigma^2),\quad
    \mu_q \sim \mathcal{N}(\nu,\rho^2).

The hyperparameters \((\nu,\rho^2,\sigma^2)\) are either:

- Estimated from the whole series (empirical Bayes), or
- Provided by the user when ``estimate_hyper=False``.

The single-segment marginal likelihood and posterior mean have closed forms and
are used by the dynamic program in :class:`~bayesbreak.base.BayesBreakBase`.
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from ..base import BayesBreakBase


class BayesBreakGaussian(BayesBreakBase):
    """Bayesian piecewise-constant regression with Gaussian observations.

    Parameters
    ----------
    k_max:
        Maximum number of segments considered by the dynamic program.

    estimate_hyper:
        If ``True``, estimate global hyperparameters from the series using
        simple moment-based estimators. If ``False``, the user must provide
        ``nu``, ``rho2`` and ``sigma2``.

    rho_estimation:
        Strategy used when estimating ``rho2``. ``"cov"`` uses an adjacent
        covariance estimate; ``"var"`` uses the marginal variance.

    regression_curve:
        If ``"none"`` (default), compute only the MAP-like piecewise-constant
        fit. If ``"fixed_k"``, also compute the Bayesian regression curve
        conditioned on the selected ``k``. If ``"mix_k"``, compute the
        regression curve mixed over the posterior distribution of ``k``.

    nu, rho2, sigma2:
        Optional fixed hyperparameters. When provided, they override the
        corresponding estimated value.

    Notes
    -----
    The hyperparameter estimators are intentionally lightweight and intended as
    robust defaults. For downstream analyses that require calibrated posterior
    uncertainty, consider fixing hyperparameters based on domain knowledge.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        rho_estimation: Literal["cov", "var"] = "cov",
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        nu: Optional[float] = None,
        rho2: Optional[float] = None,
        sigma2: Optional[float] = None,
    ):
        super().__init__(k_max=k_max, estimate_hyper=estimate_hyper, regression_curve=regression_curve)
        self.rho_estimation = rho_estimation
        self.nu = nu
        self.rho2 = rho2
        self.sigma2 = sigma2

    # ---------------------------------------------------------------------
    # Subclass hooks
    # ---------------------------------------------------------------------

    def _estimate_global_params(self, y: np.ndarray) -> Dict[str, float]:
        # If hyperparameter estimation is disabled, require user input.
        if not self.estimate_hyper:
            missing = [name for name, v in (('nu', self.nu), ('rho2', self.rho2), ('sigma2', self.sigma2)) if v is None]
            if missing:
                raise ValueError(
                    "estimate_hyper=False requires fixed hyperparameters; missing: " + ", ".join(missing)
                )
            return {"nu": float(self.nu), "rho2": float(self.rho2), "sigma2": float(self.sigma2)}

        n = y.size
        nu_hat = float(np.mean(y))

        # Estimate sigma^2 from first differences:
        #   E[(y_{t+1}-y_t)^2] = 2 sigma^2  under iid Normal noise.
        dy = np.diff(y)
        denom = 2.0 * max(1, (n - 1))
        sigma2_hat = float(np.sum(dy * dy) / denom)

        if self.rho_estimation == "cov":
            y0 = y[:-1] - nu_hat
            y1 = y[1:] - nu_hat
            cov = float(np.sum(y0 * y1) / max(1, (n - 1)))
            rho2_hat = abs(cov)
        else:
            rho2_hat = float(np.var(y, ddof=0))

        # Allow user overrides even when estimate_hyper=True.
        if self.nu is not None:
            nu_hat = float(self.nu)
        if self.rho2 is not None:
            rho2_hat = float(self.rho2)
        if self.sigma2 is not None:
            sigma2_hat = float(self.sigma2)

        # Numerical floors for stability.
        rho2_hat = max(rho2_hat, 1e-12)
        sigma2_hat = max(sigma2_hat, 1e-12)

        return {"nu": nu_hat, "rho2": rho2_hat, "sigma2": sigma2_hat}

    def _compute_single_segment_stats(
        self, y: np.ndarray, hyper: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        nu, rho2, sigma2 = hyper["nu"], hyper["rho2"], hyper["sigma2"]
        n = y.size

        # Prefix sums for raw and centered observations.
        S_raw = np.zeros(n + 1, dtype=float)
        S_raw[1:] = np.cumsum(y)

        yc = y - nu
        S1 = np.zeros(n + 1, dtype=float)
        S1[1:] = np.cumsum(yc)

        S2 = np.zeros(n + 1, dtype=float)
        S2[1:] = np.cumsum(yc * yc)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)

        log2pi_sigma = math.log(2.0 * math.pi * sigma2)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            d = j - i

            ysum_c = S1[j] - S1[i]
            y2sum = S2[j] - S2[i]

            # Closed-form log segment evidence under Normal--Normal.
            # See docs/theory.md for a derivation.
            term1 = -0.5 * d * log2pi_sigma
            term2 = -0.5 * np.log1p(d * (rho2 / sigma2))
            denom = d + (sigma2 / rho2)
            term3 = 0.5 / sigma2 * (ysum_c * ysum_c / denom - y2sum)
            logA0_ij = term1 + term2 + term3
            lA0[i, j] = logA0_ij

            # Posterior mean of mu on the segment.
            ysum_raw = S_raw[j] - S_raw[i]
            mu_hat = (rho2 * ysum_raw + sigma2 * nu) / (d * rho2 + sigma2)

            # A^1_{ij} = A^0_{ij} * E[mu | segment]
            A1[i, j] = np.exp(logA0_ij) * mu_hat

        # Disallow empty segments on the diagonal.
        idx = np.arange(n + 1)
        lA0[idx, idx] = -np.inf
        return lA0, A1

    def _segment_posterior_mean(self, a: int, b: int, y: np.ndarray, hyper: Dict[str, float]) -> float:
        nu, rho2, sigma2 = hyper["nu"], hyper["rho2"], hyper["sigma2"]
        d = b - a
        ysum = float(np.sum(y[a:b]))
        return (rho2 * ysum + sigma2 * nu) / (d * rho2 + sigma2)


# Backward-compatible alias (historically BayesBreak == Gaussian model)
BayesBreak = BayesBreakGaussian
