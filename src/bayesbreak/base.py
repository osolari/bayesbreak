# base.py (or keep everything in one file while iterating)
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Literal, List

from bayesbreak.utils import log_binom
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.naive_bayes import logsumexp



# ============================================================
# BayesBreakBase: generic DP + posteriors (distribution-agnostic)
# ============================================================

class BayesBreakBase(BaseEstimator, RegressorMixin, ABC):
    """
    Abstract base for Bayesian Piecewise-Constant Regression (BPCR).
    Subclasses provide:
      - _estimate_global_params(y) -> dict of hyperparameters
      - _compute_single_segment_stats(y, hyper) -> (lA0, A1)
      - _segment_posterior_mean(a, b, y, hyper) -> float

    This base implements:
      - dynamic programming (L, R)
      - P(k|y), evidence, boundary posteriors
      - MAP-like k selection and MAP boundaries
      - piecewise-constant fit from per-segment posterior mean
      - Bayesian regression curve (fixed-k / mix-k) using A^1 and L/R
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
    ):
        self.k_max = k_max
        self.estimate_hyper = estimate_hyper
        self.regression_curve = regression_curve

        # fitted attrs (populated by fit)
        self.n_: Optional[int] = None
        self.hyper_: Optional[Dict[str, float]] = None
        self.lA0_: Optional[np.ndarray] = None   # log A^0
        self.A1_: Optional[np.ndarray] = None    # A^1 in linear domain
        self.L_: Optional[np.ndarray] = None     # left evidences (log)
        self.R_: Optional[np.ndarray] = None     # right evidences (log)
        self.logC_: Optional[np.ndarray] = None  # log P(k|y)
        self.C_: Optional[np.ndarray] = None     # P(k|y)
        self.k_ml_: Optional[int] = None
        self.boundaries_: Optional[List[int]] = None
        self.boundary_post_: Optional[np.ndarray] = None
        self.pc_fit_: Optional[np.ndarray] = None
        self.brc_: Optional[np.ndarray] = None
        self.log_evidence_: Optional[float] = None

    # ----- abstract hooks to implement in subclasses -----

    @abstractmethod
    def _estimate_global_params(self, y: np.ndarray) -> Dict[str, float]:
        """Return hyperparameters dict (keys are subclass-defined)."""
        raise NotImplementedError

    @abstractmethod
    def _compute_single_segment_stats(
        self, y: np.ndarray, hyper: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lA0, A1) with shape (n+1, n+1) on the (i<j) upper triangle.
        lA0[i,j] = log A^0_{ij} = log P(y_{(i,j]} | single segment)
        A1[i,j]  = A^1_{ij}     = A^0_{ij} * E[mu | y_{(i,j]}]  (linear domain)
        """
        raise NotImplementedError

    @abstractmethod
    def _segment_posterior_mean(
        self, a: int, b: int, y: np.ndarray, hyper: Dict[str, float]
    ) -> float:
        """Return E[mu | y_{(a,b]}] used for the MAP piecewise-constant fit."""
        raise NotImplementedError

    # --------------------- public API ---------------------

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        # sklearn compatibility: accept y or use X when y is None
        if y is None:
            if X is None:
                raise ValueError("Provide y (preferred) or X as a 1D sequence.")
            y = np.asarray(X, dtype=float)
        else:
            y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D.")

        n = y.size
        self.n_ = n
        k_max = min(max(1, n), self.k_max)

        # hyperparameters
        hyper = self._estimate_global_params(y) if self.estimate_hyper else {}
        self.hyper_ = hyper

        # per-segment integrals
        lA0, A1 = self._compute_single_segment_stats(y, hyper)
        self.lA0_, self.A1_ = lA0, A1

        # dynamic programming
        L, R = self._compute_left_right_recursions(lA0, n, k_max)
        self.L_, self.R_ = L, R

        # posterior over k and evidence
        logC, C, logE = self._posterior_over_k(L, n, k_max)
        self.logC_, self.C_, self.log_evidence_ = logC, C, logE

        # choose k_ml around E[k] (same criterion as the R code)
        ek = np.sum((np.arange(1, k_max + 1)) * C)
        valid = np.where(np.isfinite(logC))[0] + 1  # 1..k_max
        k_ml = int(valid[np.argmin((valid - ek) ** 2)])
        self.k_ml_ = k_ml

        # boundary posteriors averaged over k
        d1 = self._boundary_posteriors_marginal(L, R, logC, n, k_max)
        self.boundary_post_ = d1

        # MAP-like boundaries = top (k_ml-1) by d1
        boundaries = self._map_boundaries_from_scores(d1, k_ml, n)
        self.boundaries_ = boundaries

        # piecewise-constant fit (posterior mean per segment)
        pc = self._compute_pc_fit(y, boundaries, hyper)
        self.pc_fit_ = pc

        # optional Bayesian regression curve
        self.brc_ = None
        if self.regression_curve == "fixed_k":
            self.brc_ = self._bayes_regression_curve_fixed_k(L, R, A1, n, k_ml)
        elif self.regression_curve == "mix_k":
            self.brc_ = self._bayes_regression_curve_mixed_k(L, R, A1, n, k_max, self.C_)

        return self

    def predict(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        if self.pc_fit_ is None:
            raise RuntimeError("Call fit() first.")
        return self.pc_fit_.copy()

    # lightweight extras

    def get_segment_count(self) -> int:
        if self.k_ml_ is None:
            raise RuntimeError("Call fit() first.")
        return self.k_ml_

    def get_boundaries(self) -> List[int]:
        if self.boundaries_ is None:
            raise RuntimeError("Call fit() first.")
        return list(self.boundaries_)

    def get_boundary_posteriors(self) -> np.ndarray:
        if self.boundary_post_ is None:
            raise RuntimeError("Call fit() first.")
        return self.boundary_post_.copy()

    def get_regression_curve(self) -> Optional[np.ndarray]:
        return None if self.brc_ is None else self.brc_.copy()

    def score(self, X=None, y=None) -> float:
        """Return log evidence log P(y) (higher is better)."""
        if self.log_evidence_ is None:
            raise RuntimeError("Call fit() first.")
        return float(self.log_evidence_)

    # --------------------- internals (generic) ---------------------

    @staticmethod
    def _compute_left_right_recursions(
        lA0: np.ndarray, n: int, k_max: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        L = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
        R = np.full((k_max + 1, n + 1), -np.inf, dtype=float)
        L[0, 0] = 0.0
        R[0, n] = 0.0

        # left recursion: L[k+1, j] = logsum_{h=k..j-1} L[k, h] + lA0[h, j]
        for k in range(0, k_max):
            for j in range(0, n + 1):
                if j >= k and j > 0:
                    h = np.arange(k, j)
                    terms = L[k, h] + lA0[h, j]
                    L[k + 1, j] = logsumexp(terms) if terms.size else -np.inf

        # right recursion: R[k+1, i] = logsum_{h=i+1..n-k} lA0[i, h] + R[k, h]
        for k in range(0, k_max):
            for i in range(0, n + 1):
                if i <= n - 1 - k:
                    h = np.arange(i + 1, n - k + 1)
                    terms = lA0[i, h] + R[k, h]
                    R[k + 1, i] = logsumexp(terms) if terms.size else -np.inf
        return L, R

    @staticmethod
    def _posterior_over_k(
        L: np.ndarray, n: int, k_max: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        log_py_given_k = np.array(
            [L[k, n] - log_binom(n - 1, k - 1) for k in range(1, k_max + 1)],
            dtype=float,
        )
        log_prior = -math.log(k_max)   # uniform prior on k
        logC = log_py_given_k + log_prior
        logE = float(logsumexp(logC))
        logC_norm = logC - logE
        C = np.exp(logC_norm)
        return logC_norm, C, logE

    @staticmethod
    def _map_boundaries_from_scores(d1: np.ndarray, k_ml: int, n: int) -> List[int]:
        if k_ml <= 1:
            return [0, n]
        sidx = np.argsort(d1)              # ascending
        picks = np.sort(sidx[-(k_ml - 1):] + 1)  # interior indices 1..n-1
        return [0, *picks.tolist(), n]

    def _compute_pc_fit(
        self, y: np.ndarray, boundaries: List[int], hyper: Dict[str, float]
    ) -> np.ndarray:
        pc = np.empty_like(y, dtype=float)
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            mu = self._segment_posterior_mean(a, b, y, hyper)
            pc[a:b] = mu
        return pc

    @staticmethod
    def _boundary_posteriors_marginal(
        L: np.ndarray, R: np.ndarray, logC: np.ndarray, n: int, k_max: int
    ) -> np.ndarray:
        wk = np.exp(logC)  # P(k|y), k=1..k_max
        d1 = np.zeros(n - 1, dtype=float)
        for i in range(1, n):  # interior candidate break after i
            acc = 0.0
            for kk in range(2, k_max + 1):
                p = np.arange(1, kk)  # ensure both sides non-empty
                terms = L[p, i] + R[kk - p, i] - L[kk, n]
                acc += wk[kk - 1] * float(np.exp(logsumexp(terms)))
            d1[i - 1] = acc
        return d1

    def _bayes_regression_curve_fixed_k(
        self, L: np.ndarray, R: np.ndarray, A1: np.ndarray, n: int, k: int
    ) -> np.ndarray:
        """
        mu'_t = sum_{i< t <= j} [ exp( logsum_{p=0..k-1} (L[p,i]+R[k-1-p, j]) - L[k,n] ) * A1[i,j] ]
        Use a difference-array trick to accumulate interval contributions.
        """
        denom = L[k, n]
        diff = np.zeros(n + 1, dtype=float)
        for i in range(0, n):
            Li = L[0:k, i]            # p=0..k-1
            for j in range(i + 1, n + 1):
                Rj = R[k - 1 :: -1, j]  # k-1, ..., 0  (length k)
                log_w_ij = logsumexp(Li + Rj) - denom
                F1 = math.exp(log_w_ij) * A1[i, j]
                if F1 != 0.0:
                    diff[i] += F1
                    diff[j] -= F1
        mu = np.cumsum(diff)
        return mu[:n]

    def _bayes_regression_curve_mixed_k(
        self, L: np.ndarray, R: np.ndarray, A1: np.ndarray, n: int, k_max: int, C: np.ndarray
    ) -> np.ndarray:
        out = np.zeros(n, dtype=float)
        for k in range(1, k_max + 1):
            if C[k - 1] == 0.0 or not np.isfinite(L[k, n]):
                continue
            out += C[k - 1] * self._bayes_regression_curve_fixed_k(L, R, A1, n, k)
        return out

