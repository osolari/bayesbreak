r"""Negative-Binomial BayesBreak family with fixed dispersion (§``sec:nb-block``).

Model (per segment q):

.. math::
    y_i \mid p_q \sim \mathrm{NegBin}(r, p_q), \qquad
    p_q \sim \mathrm{Beta}(\alpha, \beta).

The pmf used here is ``binom(y + r - 1, y) (1 - p)^y p^r`` so that the count
mean is ``r (1 - p) / p``. Beta-NegBin conjugacy gives
``p | (i, j] ~ Beta(a_B, b_B)`` with ``a_B = α + N_{ij}``, ``b_B = β + C_{ij}``,
``N_{ij} = Σ a_t w_t r_t``, ``C_{ij} = Σ a_t w_t y_t``. The block evidence is

.. math::
    A^{(0)}_{ij} = \exp\{H_{ij}\}\, \frac{B(a_B, b_B)}{B(\alpha, \beta)}.

The moment numerator ``A^{(1)}_{ij}`` targets the **observation-mean** scale
``m_*(p) = r_* (1 - p) / p`` (count mean), *not* the parameter-scale Beta
moments of ``p``. Per §``sec:nb-block`` this distinction is essential:
parameter-scale and observation-scale moments of a NegBin block are not
interchangeable. The predict-time dispersion ``r_*`` defaults to the
segment's mean training dispersion; pass ``r_predict`` to override it.

Masking semantics
-----------------
``observation_mask`` and ``likelihood_power_`` are distinct from
``sample_weight``:

* ``sample_weight`` (``w_i``): per-observation frequency or reliability
  weight, independent of masking.
* ``likelihood_power_`` (``a_i``): set by ``base.fit`` to 0 for masked
  observations and to the caller-supplied ``likelihood_power`` otherwise
  (default 1).

The effective contribution of observation ``i`` is ``a_i * w_i * f(y_i, r_i)``.

``_r_arr_`` is always sized to the full sequence length ``n``.  It is set
inside ``_compute_block_evidence`` (which receives the full ``y``), not
inside ``_estimate_hyperparameters`` (which receives only observed data).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..priors import PartitionPriorConfig
from ..utils import gammaln


class BayesBreakNegBin(BayesBreakSegmenter):
    """Piecewise-constant Negative-Binomial segmentation with a Beta prior.

    Parameters
    ----------
    k_max : int, default=50
    estimate_hyper : bool, default=True
    regression_curve : {"none", "fixed_k", "mix_k"}, default="none"
    r : float or array-like of shape (n,), default=1.0
        Fixed training dispersion parameter. Scalar or per-observation vector.
    r_predict : float or None, default=None
        Predict-time dispersion ``r_*`` used for the observation-mean target
        ``m_*(p) = r_* (1 - p)/p``. ``None`` falls back to the segment's
        training-mean dispersion.
    alpha, beta : float or None
        Optional fixed Beta prior parameters. When ``estimate_hyper`` is
        ``False`` both must be supplied.
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
        r: float | ArrayLike = 1.0,
        r_predict: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        partition_prior: PartitionPriorConfig | None = None,
    ) -> None:
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
            partition_prior=partition_prior,
        )
        self.r = r
        self.r_predict = r_predict
        self.alpha = alpha
        self.beta = beta

    def _r_array(self, n: int) -> np.ndarray:
        if np.isscalar(self.r):
            return np.full(n, float(self.r), dtype=float)
        arr = np.asarray(self.r, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"r must be scalar or shape ({n},); got {arr.shape}.")
        if np.any(arr <= 0):
            raise ValueError("r must be strictly positive.")
        return arr

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray
    ) -> dict[str, float]:
        # Short-circuit for fixed hyperparameters: no descriptor initialization
        # needed here -- _compute_block_evidence initializes _r_arr_ from the
        # full-length y it receives, avoiding a size mismatch when called via
        # observation_mask (where y here is a shorter y_obs slice).
        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError("estimate_hyper=False requires alpha and beta.")
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        # base.fit() sets self.n_ and self.observation_mask_ before calling this
        # method. Use them to build _r_arr_ at the full sequence length and slice
        # to the observed subset for moment estimation.
        n_full = int(getattr(self, "n_", y.size))
        mask = getattr(self, "observation_mask_", None)
        self._r_arr_ = self._r_array(n_full)
        if mask is not None and mask.size == n_full:
            r_arr = self._r_arr_[mask]  # observed subset, length == y.size
        else:
            r_arr = self._r_arr_
        w = sample_weight
        # Method-of-moments on p_t = r_t / (r_t + y_t).
        eps = 1e-9
        p_t = r_arr / (r_arr + np.maximum(y, 0.0) + eps)
        mu = float(np.sum(w * p_t) / max(float(np.sum(w)), 1e-12))
        var_obs = (
            float(np.sum(w * (p_t - mu) ** 2) / max(float(np.sum(w)), 1e-12)) if y.size > 1 else 1e-4
        )
        var_p = max(var_obs, 1e-9)
        tau = max(1e-6, mu * (1 - mu) / var_p - 1.0)
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
        # Always initialize _r_arr_ from the full sequence length y.size.
        # Do not rely on any value cached by _estimate_hyperparameters, which
        # receives a shorter y_obs slice when observation_mask is active.
        n = y.size
        self._r_arr_ = self._r_array(n)
        r_arr = self._r_arr_
        alpha, beta = hyper["alpha"], hyper["beta"]
        w = sample_weight

        # Effective per-observation power: 0 for masked observations.
        # Falls back to all-ones when called outside fit() (e.g., direct tests).
        _lp = getattr(self, "likelihood_power_", None)
        a = _lp if (_lp is not None and _lp.size == n) else np.ones(n, dtype=float)

        # Prefix sums weighted by a_i * w_i.
        S_y = np.zeros(n + 1, dtype=float)
        S_y[1:] = np.cumsum(a * w * y)
        S_r = np.zeros(n + 1, dtype=float)
        S_r[1:] = np.cumsum(a * w * r_arr)
        # Base measure: log binom(y + r - 1, y).
        H = gammaln(y + r_arr) - gammaln(y + 1.0) - gammaln(r_arr)
        S_H = np.zeros(n + 1, dtype=float)
        S_H[1:] = np.cumsum(a * w * H)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        log_B_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            Csum = S_y[j] - S_y[i]
            Nsum = S_r[j] - S_r[i]
            Hsum = S_H[j] - S_H[i]

            a_post = alpha + Nsum
            b_post = beta + Csum
            log_B_post = gammaln(a_post) + gammaln(b_post) - gammaln(a_post + b_post)
            lA0_ij = Hsum + log_B_post - log_B_prior
            lA0[i, j] = lA0_ij

            # Observation-mean target m_*(p) = r_* (1-p)/p where r_* defaults
            # to the segment's mean training dispersion (per-block average of
            # r_t). E[m_*] = r_* · b_post / (a_post - 1) when a_post > 1.
            # E[m_*] diverges when a_post <= 1 (reported as nan).
            # The second moment E[m_*^2] = r_*^2 · b_post(b_post+1)/((a_post-1)(a_post-2))
            # requires a_post > 2 and is not currently computed or stored.
            if self.r_predict is not None:
                r_star = np.full(j.shape, float(self.r_predict), dtype=float)
            else:
                r_star = np.array([float(np.mean(r_arr[i:jj])) for jj in j], dtype=float)
            valid = a_post > 1.0
            # E[r(1-p)/p] = r_star * b_post / (a_post - 1) requires a_post > 1.
            # Report nan when a_post <= 1 rather than a silent 0 sentinel.
            E_m = np.where(valid, r_star * b_post / np.where(valid, a_post - 1.0, 1.0), np.nan)
            A1[i, j] = E_m

        np.fill_diagonal(lA0, -np.inf)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        if not hasattr(self, "_r_arr_"):
            self._r_arr_ = self._r_array(y.size)
        r_arr = self._r_arr_
        alpha, beta = hyper["alpha"], hyper["beta"]
        _lp = getattr(self, "likelihood_power_", None)
        lp = _lp[a:b] if (_lp is not None and _lp.size >= b) else np.ones(b - a, dtype=float)
        w = lp * sample_weight[a:b]
        Csum = float(np.sum(w * y[a:b]))
        Nsum = float(np.sum(w * r_arr[a:b]))
        a_post = alpha + Nsum
        b_post = beta + Csum
        # E[r(1-p)/p | y] = r_* * b_post / (a_post - 1) requires a_post > 1.
        if a_post <= 1.0:
            return float("nan")
        r_star = float(self.r_predict) if self.r_predict is not None else float(np.mean(r_arr[a:b]))
        return r_star * b_post / (a_post - 1.0)

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        """Beta-NegBin posterior-predictive log-density for new counts.

        Under the segment posterior ``p | train ~ Beta(α_B, β_B)``, the
        predictive for a new observation with dispersion ``r_*`` is

        .. math::
            p(y) = \\binom{y + r_* - 1}{y}\\,
                    \\frac{B(α_B + r_*, β_B + y)}{B(α_B, β_B)}.
        """

        assert (
            self.hyper_ is not None
            and self.sample_weight_ is not None
            and self._y_train_ is not None
        )
        alpha, beta = self.hyper_["alpha"], self.hyper_["beta"]
        r_arr = self._r_arr_
        _lp = getattr(self, "likelihood_power_", None)
        lp = _lp[a:b] if (_lp is not None and _lp.size >= b) else np.ones(b - a, dtype=float)
        w_train = lp * self.sample_weight_[a:b]
        y_train = self._y_train_[a:b]
        Csum = float(np.sum(w_train * y_train))
        Nsum = float(np.sum(w_train * r_arr[a:b]))
        a_post = alpha + Nsum
        b_post = beta + Csum

        # Predict-time dispersion: explicit r_predict if supplied, else the
        # segment's mean training dispersion.
        r_star = float(self.r_predict) if self.r_predict is not None else float(np.mean(r_arr[a:b]))
        y_new = np.asarray(y_new, dtype=float)
        log_binom_term = gammaln(y_new + r_star) - gammaln(y_new + 1.0) - math.lgamma(r_star)
        log_B_post_new = (
            gammaln(a_post + r_star)
            + gammaln(b_post + y_new)
            - gammaln(a_post + b_post + r_star + y_new)
        )
        log_B_post = math.lgamma(a_post) + math.lgamma(b_post) - math.lgamma(a_post + b_post)
        return log_binom_term + log_B_post_new - log_B_post
