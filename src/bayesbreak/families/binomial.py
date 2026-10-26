"""Binomial BayesBreak family (Beta--Binomial conjugate)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..base import BayesBreakSegmenter
from ..priors import PartitionPriorConfig
from ..utils import gammaln


class BayesBreakBinomial(BayesBreakSegmenter):
    r"""Piecewise-constant Binomial segmentation with a Beta prior.

    .. math::
        y_i \mid p_q \sim \mathrm{Binomial}(n_i, p_q), \quad
        p_q \sim \mathrm{Beta}(\alpha, \beta).

    Masking semantics
    -----------------
    ``observation_mask`` and ``likelihood_power_`` are distinct from
    ``sample_weight``:

    * ``sample_weight`` (``w_i``): per-observation frequency or reliability
      weight, independent of masking.  Passed directly by the caller.
    * ``likelihood_power_`` (``a_i``): set by ``base.fit`` to 0 for masked
      observations and to the caller-supplied ``likelihood_power`` otherwise
      (default 1).  Zeroing ``a_i`` excludes an observation from block evidence
      and posterior parameters without requiring the caller to zero ``w_i``.

    The effective contribution of observation ``i`` to any prefix sum is
    ``a_i * w_i * f(y_i, n_i)``.  For ordinary unmasked data both equal 1
    and the formula reduces to the standard weighted Beta-Binomial.

    ``_n_arr_`` is always sized to the full sequence length ``n``.  It is set
    inside ``_compute_block_evidence`` (which receives the full ``y``), not
    inside ``_estimate_hyperparameters`` (which receives only observed data).
    This separation prevents the shape mismatch that arises when
    ``observation_mask`` causes ``_estimate_hyperparameters`` to be called with
    a shorter ``y_obs`` slice.
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
        # base.fit() sets self.n_ and self.observation_mask_ before calling this
        # method. Use them to initialize _n_arr_ at the full sequence length and
        # slice to the observed subset for moment estimation.
        #
        # For direct calls (without fit()), self.n_ is absent; fall back to
        # y.size.  This path only works correctly for scalar n_trials.
        n_full = int(getattr(self, "n_", y.size))
        mask = getattr(self, "observation_mask_", None)

        # Build full-length descriptor now so _compute_block_evidence can use
        # self._n_arr_ directly without re-initialising.
        self._n_arr_ = self._trials_array(n_full)

        if not self.estimate_hyper:
            if self.alpha is None or self.beta is None:
                raise ValueError(
                    "estimate_hyper=False requires explicit alpha and beta for the Beta prior."
                )
            return {"alpha": float(self.alpha), "beta": float(self.beta)}

        # Get the observed subset of n_arr for moment estimation.
        # y contains only observed data (sliced by base.fit).
        if mask is not None and mask.size == n_full:
            n_arr = self._n_arr_[mask]   # observed subset, length == y.size
        else:
            n_arr = self._n_arr_         # no mask; full array equals observed
        w = sample_weight
        S = float(np.sum(w * y))
        T = float(np.sum(w * n_arr))
        mu = S / max(T, 1e-12)

        p_i = np.where(n_arr > 0, y / n_arr, mu)
        if y.size > 1:
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
        # Always initialize _n_arr_ from the full sequence length y.size.
        # Do not rely on any value cached by _estimate_hyperparameters, which
        # receives a shorter y_obs slice when observation_mask is active.
        n = y.size
        self._n_arr_ = self._trials_array(n)
        n_arr = self._n_arr_
        alpha, beta = hyper["alpha"], hyper["beta"]
        w = sample_weight

        # Effective per-observation power: 0 for masked observations.
        # Falls back to all-ones when called outside fit() (e.g., direct tests).
        _lp = getattr(self, "likelihood_power_", None)
        a = _lp if (_lp is not None and _lp.size == n) else np.ones(n, dtype=float)

        # Prefix sums weighted by a_i * w_i.  When a_i = 0 (masked) the
        # observation contributes nothing to posterior parameters or log evidence.
        S = np.zeros(n + 1, dtype=float)
        S[1:] = np.cumsum(a * w * y)
        N = np.zeros(n + 1, dtype=float)
        N[1:] = np.cumsum(a * w * n_arr)

        Lcomb = gammaln(n_arr + 1.0) - gammaln(y + 1.0) - gammaln(n_arr - y + 1.0)
        Csum = np.zeros(n + 1, dtype=float)
        Csum[1:] = np.cumsum(a * w * Lcomb)

        logB_ab = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        upper = np.triu(np.ones((n + 1, n + 1), dtype=bool), k=1)
        Sij = S[None, :] - S[:, None]
        Nij = N[None, :] - N[:, None]
        Fij = Nij - Sij
        Cij = Csum[None, :] - Csum[:, None]

        S_up = Sij[upper]
        N_up = Nij[upper]
        F_up = Fij[upper]

        logB_post = (
            gammaln(alpha + S_up)
            + gammaln(beta + F_up)
            - gammaln(alpha + beta + N_up)
        )
        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        lA0[upper] = Cij[upper] + (logB_post - logB_ab)

        # A1[i,j] = E[p | y_{i:j}] = posterior mean of the success probability.
        # This always exists (both posterior parameters are positive).
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        A1[upper] = (alpha + S_up) / (alpha + beta + N_up)
        return lA0, A1

    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray
    ) -> float:
        alpha, beta = hyper["alpha"], hyper["beta"]
        _lp = getattr(self, "likelihood_power_", None)
        lp = _lp[a:b] if (_lp is not None and _lp.size >= b) else np.ones(b - a, dtype=float)
        w = lp * sample_weight[a:b]
        Ssum = float(np.sum(w * y[a:b]))
        Nsum = float(np.sum(w * self._n_arr_[a:b]))
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

        ``w_new`` supplies the known number of trials for each new observation.
        The prediction router uses one trial per observation when no descriptor
        is supplied, recovering the Bernoulli special case.
        """

        assert (
            self.hyper_ is not None
            and self.sample_weight_ is not None
            and self._y_train_ is not None
        )
        alpha, beta = self.hyper_["alpha"], self.hyper_["beta"]
        _lp = getattr(self, "likelihood_power_", None)
        lp = _lp[a:b] if (_lp is not None and _lp.size >= b) else np.ones(b - a, dtype=float)
        w_train = lp * self.sample_weight_[a:b]
        y_train = self._y_train_[a:b]
        n_train = self._n_arr_[a:b]
        S_post = float(np.sum(w_train * y_train))
        N_post = float(np.sum(w_train * n_train))
        alpha_post = alpha + S_post
        beta_post = beta + (N_post - S_post)

        successes = np.asarray(y_new, dtype=float)
        trials = np.asarray(w_new, dtype=float)
        if successes.shape != trials.shape:
            raise ValueError("Binomial y_new and n_trials must have the same shape")
        if np.any(~np.isfinite(successes)) or np.any(successes != np.floor(successes)):
            raise ValueError("Binomial y_new must contain finite integer counts")
        if np.any(~np.isfinite(trials)) or np.any(trials < 1) or np.any(trials != np.floor(trials)):
            raise ValueError("Binomial prediction n_trials must be positive integers")
        if np.any(successes < 0) or np.any(successes > trials):
            raise ValueError("Binomial y_new must satisfy 0 <= y_new <= n_trials")

        log_combination = (
            gammaln(trials + 1.0) - gammaln(successes + 1.0) - gammaln(trials - successes + 1.0)
        )
        log_beta_posterior = (
            gammaln(alpha_post + successes)
            + gammaln(beta_post + trials - successes)
            - gammaln(alpha_post + beta_post + trials)
        )
        log_beta_training = (
            math.lgamma(alpha_post) + math.lgamma(beta_post) - math.lgamma(alpha_post + beta_post)
        )
        return log_combination + log_beta_posterior - log_beta_training
