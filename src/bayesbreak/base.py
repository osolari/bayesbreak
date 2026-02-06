"""Core estimator and dynamic-programming routines.

The BayesBreak algorithm performs Bayesian piecewise-constant regression using
closed-form segment marginal likelihoods. The model family (Gaussian,
Poisson/Gamma, Binomial/Beta, etc.) is implemented by subclasses via three
hooks:

- :meth:`~bayesbreak.base.BayesBreakBase._estimate_global_params`
- :meth:`~bayesbreak.base.BayesBreakBase._compute_single_segment_stats`
- :meth:`~bayesbreak.base.BayesBreakBase._segment_posterior_mean`

The base class is responsible for:

- Dynamic programming recursions to compute segment evidence tables.
- Posterior distribution over the number of segments ``k``.
- Marginal posterior probabilities of change points.
- A MAP-like piecewise-constant regression (via marginal change-point scores).
- Optional Bayesian regression curves that average over segmentations.

This module is intentionally distribution-agnostic; all distribution-specific
closed forms live in :mod:`bayesbreak.families`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin

from .utils import as_1d_float_array, check_sample_weight, log_binom, logsumexp, require_fitted


class BayesBreakBase(BaseEstimator, RegressorMixin, ABC):
    r"""Abstract base class for BayesBreak estimators.

    The estimator fits a piecewise-constant latent signal with an unknown number
    of segments. The latent segment parameter (e.g., mean, rate, probability)
    has a conjugate prior, allowing closed-form integration of the segment
    likelihood.

    Parameters
    ----------
    k_max:
        Maximum number of segments considered. Internally we cap this at ``n``
        (the number of observations).
    estimate_hyper:
        If ``True``, subclasses may estimate hyperparameters using an
        empirical-Bayes routine. If ``False``, required hyperparameters must be
        provided by the user; subclasses should raise a ``ValueError`` when
        they are missing.
    regression_curve:
        If ``"none"``, only the MAP-like piecewise-constant fit is computed.

        If ``"fixed_k"``, compute a Bayesian regression curve conditional on the
        selected ``k``.

        If ``"mix_k"``, compute a Bayesian regression curve that mixes over
        ``k`` with weights :math:`P(k\mid y)`.

    Notes
    -----
    - The dynamic program runs in :math:`O(k_{\max} n^2)` time and
      :math:`O(k_{\max} n)` memory.
    - The returned boundaries are *not* the exact MAP segmentation. Instead we
      compute marginal posterior probabilities for each boundary location and
      select the top ``k-1``. This matches the original BayesBreak reference
      implementation and scales well.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: str = "none",
    ):
        if k_max < 1:
            raise ValueError("k_max must be >= 1")
        if regression_curve not in {"none", "fixed_k", "mix_k"}:
            raise ValueError("regression_curve must be one of: 'none', 'fixed_k', 'mix_k'")

        self.k_max = int(k_max)
        self.estimate_hyper = bool(estimate_hyper)
        self.regression_curve = regression_curve

        # Fitted attributes (populated by fit)
        self.n_: Optional[int] = None
        self.hyper_: Optional[Dict[str, float]] = None
        self.lA0_: Optional[np.ndarray] = None  # log A^0 block evidence
        self.A1_: Optional[np.ndarray] = None  # A^1 block first moment (linear domain)
        self.L_: Optional[np.ndarray] = None  # left DP table (log)
        self.R_: Optional[np.ndarray] = None  # right DP table (log)
        self.logC_: Optional[np.ndarray] = None  # log P(k|y) for k=1..k_max
        self.C_: Optional[np.ndarray] = None  # P(k|y)
        # Selected number of segments.
        #
        # Historical note: early versions of this code used the attribute name
        # ``k_ml_`` (following the original R implementation). Internally we
        # still compute a heuristic point estimate close to E[k | y]. To remain
        # backwards-compatible, we expose both ``k_ml_`` and ``k_hat_`` and keep
        # them equal.
        self.k_ml_: Optional[int] = None
        self.k_hat_: Optional[int] = None  # alias of ``k_ml_``
        self.boundaries_: Optional[List[int]] = None
        self.boundary_post_: Optional[np.ndarray] = None
        self.pc_fit_: Optional[np.ndarray] = None  # MAP-like piecewise-constant fit
        self.brc_: Optional[np.ndarray] = None  # Bayesian regression curve
        self.log_evidence_: Optional[float] = None  # log P(y)
        self.sample_weight_: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Subclass hooks
    # ---------------------------------------------------------------------

    @abstractmethod
    def _estimate_global_params(self, y: np.ndarray, sample_weight: np.ndarray) -> Dict[str, float]:
        """Return model hyperparameters.

        Subclasses must implement this method and should honor
        ``self.estimate_hyper``.

        - If ``self.estimate_hyper`` is ``True``, the method may estimate missing
          hyperparameters from the full series ``y``.
        - If ``self.estimate_hyper`` is ``False``, the method should validate
          that the user provided all required hyperparameters.

        Returns
        -------
        dict
            Dictionary of hyperparameters. Keys are family-specific.
        """

    @abstractmethod
    def _compute_single_segment_stats(
        self, y: np.ndarray, hyper: Dict[str, float], sample_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-block evidences and first moments.

        For indices ``0 <= i < j <= n``, define the block as
        :math:`y_{i+1},\\ldots,y_j`, corresponding to Python slice ``y[i:j]``.

        Subclasses must return two upper-triangular ``(n+1) x (n+1)`` arrays:

        - ``lA0[i, j] = log A^0_{ij}``, the integrated (marginal) likelihood of
          the block under a single segment.
        - ``A1[i, j] = A^1_{ij} = A^0_{ij} * E[\theta\\mid y_{i:j}]``.

        ``A1`` is stored in the linear domain because it is used as an additive
        quantity in the regression-curve computation.
        """

    @abstractmethod
    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: Dict[str, float], sample_weight: np.ndarray
    ) -> float:
        """Posterior mean of the segment parameter for block ``y[a:b]``."""

    # ---------------------------------------------------------------------
    # Public sklearn-style API
    # ---------------------------------------------------------------------

    def fit(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
    ):
        """Fit the BayesBreak model.

        Parameters
        ----------
        X:
            Optional design matrix. BayesBreak primarily operates on ordered
            sequences; for compatibility with scikit-learn you may pass the
            observations as ``X`` and leave ``y=None``.
        y:
            Optional target vector. If provided, must be 1D.

        Returns
        -------
        self
            Fitted estimator.
        """

        # scikit-learn compatibility: allow fit(y) by passing y as X.
        if y is None:
            if X is None:
                raise ValueError("Provide y (preferred) or X as a 1D sequence.")
            y_arr = as_1d_float_array(X, name="y")
        else:
            y_arr = as_1d_float_array(y, name="y")

        n = int(y_arr.size)
        self.n_ = n

        w_arr = check_sample_weight(sample_weight, n)
        self.sample_weight_ = w_arr

        k_max = min(self.k_max, n)

        # -----------------------------------------------------------------
        # 1) Hyperparameters
        # -----------------------------------------------------------------
        hyper = self._estimate_global_params(y_arr, w_arr)
        if not isinstance(hyper, dict):
            raise TypeError("_estimate_global_params must return a dict")
        self.hyper_ = {str(k): float(v) for k, v in hyper.items()}

        # -----------------------------------------------------------------
        # 2) Block evidences
        # -----------------------------------------------------------------
        lA0, A1 = self._compute_single_segment_stats(y_arr, self.hyper_, w_arr)
        if lA0.shape != (n + 1, n + 1) or A1.shape != (n + 1, n + 1):
            raise ValueError(
                "_compute_single_segment_stats must return arrays of shape "
                f"({n+1},{n+1}); got {lA0.shape} and {A1.shape}."
            )
        self.lA0_, self.A1_ = lA0, A1

        # -----------------------------------------------------------------
        # 3) Dynamic programming
        # -----------------------------------------------------------------
        L, R = self._compute_left_right_recursions(lA0, n, k_max)
        self.L_, self.R_ = L, R

        # -----------------------------------------------------------------
        # 4) Posterior over k (and evidence)
        # -----------------------------------------------------------------
        logC, C, logE = self._posterior_over_k(L, n, k_max)
        self.logC_, self.C_, self.log_evidence_ = logC, C, logE

        # Select k (point estimate): choose the integer nearest E[k | y].
        ek = float(np.sum((np.arange(1, k_max + 1, dtype=float)) * C))
        valid_k = np.arange(1, k_max + 1, dtype=int)[np.isfinite(logC)]
        if valid_k.size == 0:
            raise RuntimeError("No valid segment counts k produced finite evidence.")
        k_sel = int(valid_k[np.argmin((valid_k.astype(float) - ek) ** 2)])
        self.k_ml_ = k_sel
        self.k_ml_ = k_sel
        self.k_hat_ = k_sel

        # -----------------------------------------------------------------
        # 5) Boundary posteriors (marginal over k)
        # -----------------------------------------------------------------
        d1 = self._boundary_posteriors_marginal(L, R, logC, n, k_max)
        self.boundary_post_ = d1

        # MAP-like boundaries: take top (k-1) candidates.
        boundaries = self._select_boundaries_from_scores(d1, k_sel, n)
        self.boundaries_ = boundaries

        # -----------------------------------------------------------------
        # 6) Piecewise-constant posterior-mean fit for the selected boundaries
        # -----------------------------------------------------------------
        self.pc_fit_ = self._compute_pc_fit(y_arr, w_arr, boundaries, self.hyper_)

        # -----------------------------------------------------------------
        # 7) Optional Bayesian regression curve
        # -----------------------------------------------------------------
        self.brc_ = None
        if self.regression_curve == "fixed_k":
            self.brc_ = self._bayes_regression_curve_fixed_k(L, R, lA0, A1, n, k_sel)
        elif self.regression_curve == "mix_k":
            self.brc_ = self._bayes_regression_curve_mixed_k(L, R, lA0, A1, n, k_max, C)

        return self

    def predict(self, X: Optional[ArrayLike] = None) -> np.ndarray:
        """Return the fitted piecewise-constant regression values.

        Notes
        -----
        BayesBreak is primarily a *sequence* model. The returned prediction is
        defined on the training indices. For scikit-learn compatibility, we
        accept an unused ``X`` parameter.
        """

        require_fitted(self, ["pc_fit_"])
        assert self.pc_fit_ is not None
        return self.pc_fit_.copy()

    # ---------------------------------------------------------------------
    # Convenience accessors
    # ---------------------------------------------------------------------

    def score(self, X: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None) -> float:
        """Return the log marginal likelihood ``log P(y)``.

        This is a proper Bayesian model score: larger values indicate a better
        fit (subject to the model family assumptions).
        """

        require_fitted(self, ["log_evidence_"])
        assert self.log_evidence_ is not None
        return float(self.log_evidence_)

    def get_segment_count(self) -> int:
        """Return the selected number of segments.

        The returned value is stored in both ``k_ml_`` (legacy name) and
        ``k_hat_``.
        """

        require_fitted(self, ["k_ml_"])
        assert self.k_ml_ is not None
        return int(self.k_ml_)

    def get_boundaries(self) -> List[int]:
        """Return the selected boundary indices, including ``0`` and ``n``."""

        require_fitted(self, ["boundaries_"])
        assert self.boundaries_ is not None
        return list(self.boundaries_)

    def get_boundary_posteriors(self) -> np.ndarray:
        """Return marginal posterior probability of a boundary after each index."""

        require_fitted(self, ["boundary_post_"])
        assert self.boundary_post_ is not None
        return self.boundary_post_.copy()

    def get_regression_curve(self) -> Optional[np.ndarray]:
        """Return Bayesian regression curve if enabled, else ``None``."""

        require_fitted(self, ["pc_fit_"])
        return None if self.brc_ is None else self.brc_.copy()

    # ---------------------------------------------------------------------
    # Dynamic programming (generic)
    # ---------------------------------------------------------------------

    @staticmethod
    def _compute_left_right_recursions(
        lA0: np.ndarray, n: int, k_max: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute DP tables L and R in log-space.

        L[k, j] stores the (log) evidence for the prefix ``y[:j]`` segmented into
        exactly ``k`` segments.

        R[k, i] stores the (log) evidence for the suffix ``y[i:]`` (i.e., block
        starting at index ``i``) segmented into exactly ``k`` segments.
        """

        L = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
        R = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
        L[0, 0] = 0.0
        R[0, n] = 0.0

        # Left recursion:
        #   L[k+1, j] = log \sum_{h=k..j-1} exp( L[k, h] + lA0[h, j] ).
        for k in range(0, k_max):
            for j in range(1, n + 1):
                if j < k + 1:
                    continue
                h = np.arange(k, j)
                terms = L[k, h] + lA0[h, j]
                L[k + 1, j] = logsumexp(terms) if terms.size else -np.inf

        # Right recursion:
        #   R[k+1, i] = log \sum_{h=i+1..n-k} exp( lA0[i, h] + R[k, h] ).
        for k in range(0, k_max):
            for i in range(0, n):
                if i > n - 1 - k:
                    continue
                h = np.arange(i + 1, n - k + 1)
                terms = lA0[i, h] + R[k, h]
                R[k + 1, i] = logsumexp(terms) if terms.size else -np.inf

        return L, R

    @staticmethod
    def _posterior_over_k(
        L: np.ndarray, n: int, k_max: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute ``log P(k|y)``, ``P(k|y)``, and ``log P(y)``.

        We use a uniform prior over ``k`` on ``{1,\\ldots,k_max}``.

        Additionally, we include the BayesBreak combinatorial correction
        :math:`1/\binom{n-1}{k-1}` to account for the number of distinct boundary
        placements.
        """

        log_py_given_k = np.array(
            [L[k, n] - log_binom(n - 1, k - 1) for k in range(1, k_max + 1)],
            dtype=float,
        )
        log_prior = -math.log(float(k_max))
        logC_unnorm = log_py_given_k + log_prior
        logE = float(logsumexp(logC_unnorm))
        logC = logC_unnorm - logE
        C = np.exp(logC)
        return logC, C, logE

    @staticmethod
    def _select_boundaries_from_scores(d1: np.ndarray, k_hat: int, n: int) -> List[int]:
        """Select interior boundaries by taking the top ``k_hat-1`` scores."""

        if k_hat <= 1:
            return [0, n]
        if d1.size != n - 1:
            raise ValueError("boundary score vector must have length n-1")

        # Indices in d1 correspond to boundary after i (1..n-1).
        best = np.argsort(d1)[-(k_hat - 1) :]
        picks = np.sort(best + 1)  # map back to index positions
        return [0, *picks.tolist(), n]

    def _compute_pc_fit(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray,
        boundaries: List[int],
        hyper: Dict[str, float],
    ) -> np.ndarray:
        pc = np.empty_like(y, dtype=float)
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=False):
            mu = self._segment_posterior_mean(a, b, y, hyper, sample_weight)
            pc[a:b] = mu
        return pc

    @staticmethod
    def _boundary_posteriors_marginal(
        L: np.ndarray, R: np.ndarray, logC: np.ndarray, n: int, k_max: int
    ) -> np.ndarray:
        """Marginal posterior probability that each position is a boundary.

        Returns
        -------
        ndarray
            Length ``n-1`` vector, where element ``i-1`` corresponds to the
            posterior probability of a boundary after index ``i``.
        """

        wk = np.exp(logC)  # P(k|y), for k=1..k_max
        d1 = np.zeros(n - 1, dtype=float)

        for i in range(1, n):
            acc = 0.0
            for k in range(2, k_max + 1):
                # p is the boundary number (1..k-1). Both sides must be non-empty.
                p = np.arange(1, k)
                terms = L[p, i] + R[k - p, i] - L[k, n]
                acc += wk[k - 1] * float(np.exp(logsumexp(terms)))
            d1[i - 1] = acc

        return d1

    # ---------------------------------------------------------------------
    # Bayesian regression curve
    # ---------------------------------------------------------------------

    @staticmethod
    def _bayes_regression_curve_fixed_k(
        L: np.ndarray,
        R: np.ndarray,
        lA0: np.ndarray,
        A1: np.ndarray,
        n: int,
        k: int,
    ) -> np.ndarray:
        """Compute Bayesian regression curve for a fixed number of segments ``k``.

        This is the posterior expectation of the latent segment parameter at each
        index, averaged over all segmentations with exactly ``k`` segments.

        Implementation detail: we use a difference-array trick to accumulate
        interval contributions in :math:`O(n^2)` time.
        """

        denom = float(L[k, n])
        if not np.isfinite(denom):
            return np.full(n, np.nan, dtype=float)

        diff = np.zeros(n + 1, dtype=float)
        for i in range(0, n):
            Li = L[0:k, i]  # p=0..k-1
            for j in range(i + 1, n + 1):
                Rj = R[k - 1 :: -1, j]  # R[k-1, j], ..., R[0, j]
                la0 = float(lA0[i, j])
                if not np.isfinite(la0):
                    continue

                # Segment posterior probability weight (should be <= 1).
                log_pseg = float(logsumexp(Li + Rj) + la0 - denom)
                if log_pseg < -745.0:
                    continue
                w = math.exp(log_pseg)

                # Segment posterior mean for the latent parameter.
                # We recover it from A1 / A0 in log domain to avoid overflow.
                a = float(A1[i, j])
                if a == 0.0 or not np.isfinite(a):
                    continue
                log_abs_mu = math.log(abs(a)) - la0
                if log_abs_mu < -745.0:
                    continue
                # Guard against pathological cases.
                if log_abs_mu > 709.0:
                    mu_hat = math.copysign(float("inf"), a)
                else:
                    mu_hat = math.copysign(math.exp(log_abs_mu), a)

                contrib = w * mu_hat
                if contrib != 0.0 and np.isfinite(contrib):
                    diff[i] += contrib
                    diff[j] -= contrib

        mu = np.cumsum(diff)
        return mu[:n]

    @staticmethod
    def _bayes_regression_curve_mixed_k(
        L: np.ndarray,
        R: np.ndarray,
        lA0: np.ndarray,
        A1: np.ndarray,
        n: int,
        k_max: int,
        C: np.ndarray,
    ) -> np.ndarray:
        """Compute Bayesian regression curve mixing over ``k`` with weights ``C``."""

        out = np.zeros(n, dtype=float)
        for k in range(1, k_max + 1):
            w = float(C[k - 1])
            if w == 0.0 or not np.isfinite(L[k, n]):
                continue
            out += w * BayesBreakBase._bayes_regression_curve_fixed_k(L, R, lA0, A1, n, k)
        return out
