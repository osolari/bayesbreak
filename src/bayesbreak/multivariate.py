r"""Multivariate (vector-valued) BayesBreak segmentation.

Two distinct modes share boundaries or fit independently per channel:

- :class:`SharedBoundaryMultivariateSegmenter`: a single segmentation is
  inferred from the joint block evidence
  :math:`\log \mathcal{L}_{ij} = \sum_c \log \mathcal{L}^{(c)}_{ij}` under
  conditional independence across channels given segment parameters.
- :class:`IndependentMultivariateSegmenter`: each channel is segmented
  independently by cloning the base estimator.

Both expose a strict sklearn ``fit(X, y)`` API where ``y`` has shape ``(n, d)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone

from . import dp as _dp
from .base import BayesBreakSegmenter
from .prediction import ExtrapolationPolicy, _record_prediction_policy, assign_to_partition
from .validation import check_sample_weight, check_segmentation_input, require_fitted

FloatArray = NDArray[np.floating]


def _normalize_multivariate_weights(sample_weight: ArrayLike | None, n: int, d: int) -> FloatArray:
    if sample_weight is None or np.isscalar(sample_weight):
        w1 = check_sample_weight(sample_weight, n)
        return np.repeat(w1[:, None], d, axis=1)
    w = np.asarray(sample_weight, dtype=float)
    if w.ndim == 1:
        if w.shape[0] != n:
            raise ValueError(f"sample_weight has length {w.shape[0]}, expected {n}.")
        return np.repeat(w[:, None], d, axis=1)
    if w.ndim == 2:
        if w.shape != (n, d):
            raise ValueError(f"sample_weight has shape {w.shape}, expected {(n, d)}.")
        return w
    raise ValueError("sample_weight must be None, scalar, 1-D (n,) or 2-D (n, d).")


@dataclass
class _ChannelState:
    est: BayesBreakSegmenter
    hyper: dict
    lA0: FloatArray
    A1: FloatArray


class SharedBoundaryMultivariateSegmenter(BaseEstimator, RegressorMixin):
    """Multivariate segmenter with a single shared segmentation across channels.

    Parameters
    ----------
    base_estimator : BayesBreakSegmenter
        Univariate block family template (cloned per channel).
    k_max : int, optional
        Maximum segment count. Defaults to ``base_estimator.k_max``.

    Attributes
    ----------
    n_, d_ : int
        Sample / channel counts.
    k_map_ : int
        Posterior-mode segment count.
    map_boundaries_ : list of int
        Joint MAP boundary vector (shared across channels).
    boundary_marginals_ : ndarray of shape (n-1,)
        Marginal boundary-event probabilities.
    map_segment_means_ : ndarray of shape (k_map, d)
        Per-channel posterior means on MAP segments.
    map_curve_ : ndarray of shape (n, d)
        Piecewise-constant fit.
    bayes_curve_mean_ : ndarray of shape (n, d) or None
        Optional posterior-mean curve.
    log_evidence_ : float
        Joint log marginal likelihood.
    """

    def __init__(
        self,
        base_estimator: BayesBreakSegmenter,
        *,
        k_max: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.k_max = k_max

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> SharedBoundaryMultivariateSegmenter:
        x_design, y_arr, _ = check_segmentation_input(X, y, multivariate=True)
        n, d = y_arr.shape
        self.n_, self.d_ = int(n), int(d)
        self.x_design_ = x_design

        w_mat = _normalize_multivariate_weights(sample_weight, n, d)
        k_max = int(self.k_max) if self.k_max is not None else int(self.base_estimator.k_max)
        k_max = min(max(1, n), k_max)

        channel_states: list[_ChannelState] = []
        lA0_joint: FloatArray | None = None
        for c in range(d):
            est_c = clone(self.base_estimator)
            hyper_c = est_c._estimate_hyperparameters(y_arr[:, c], w_mat[:, c])
            lA0_c, A1_c = est_c._compute_block_evidence(y_arr[:, c], hyper_c, w_mat[:, c])
            channel_states.append(_ChannelState(est=est_c, hyper=hyper_c, lA0=lA0_c, A1=A1_c))
            lA0_joint = lA0_c.copy() if lA0_joint is None else lA0_joint + lA0_c
        assert lA0_joint is not None

        log_left, log_right = _dp.forward_backward(lA0_joint, n, k_max)
        log_post_k, post_k, log_evidence = _dp.posterior_over_k(log_left, n, k_max)
        self.log_evidence_ = float(log_evidence)

        valid = np.arange(1, k_max + 1)[np.isfinite(log_post_k)]
        self.k_map_ = int(valid[int(np.argmax(log_post_k[valid - 1]))])
        self.boundary_marginals_ = _dp.boundary_event_marginals_fixed_k(
            log_left, log_right, n, self.k_map_
        )
        map_boundaries, _log_joint = _dp.max_sum_segmentation(lA0_joint, self.k_map_)
        self.map_boundaries_ = list(map_boundaries)
        self.boundaries_internal_ = np.asarray(self.map_boundaries_, dtype=int)

        means = np.zeros((self.k_map_, d), dtype=float)
        pc = np.zeros((n, d), dtype=float)
        for s, (a, b) in enumerate(
            zip(self.map_boundaries_[:-1], self.map_boundaries_[1:], strict=False)
        ):
            for c, st in enumerate(channel_states):
                mu = st.est._segment_posterior_mean(
                    int(a), int(b), y_arr[:, c], st.hyper, w_mat[:, c]
                )
                means[s, c] = float(mu)
                pc[int(a) : int(b), c] = mu
        self.map_segment_means_ = means
        self.map_curve_ = pc

        self.bayes_curve_mean_ = None
        rc = getattr(self.base_estimator, "regression_curve", "none")
        if rc in {"fixed_k", "mix_k"}:
            brc = np.zeros((n, d), dtype=float)
            for c, st in enumerate(channel_states):
                A1_joint_c = self._make_channel_A1_joint(st, lA0_joint)
                if rc == "fixed_k":
                    brc[:, c] = _dp.bayes_regression_curve_fixed_k(
                        log_left, log_right, lA0_joint, A1_joint_c, n, self.k_map_
                    )
                else:
                    brc[:, c] = _dp.bayes_regression_curve_mixed_k(
                        log_left, log_right, lA0_joint, A1_joint_c, n, k_max, post_k
                    )
            self.bayes_curve_mean_ = brc

        self.channel_estimators_ = [st.est for st in channel_states]
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        extrapolation: str | ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> FloatArray:
        """Piecewise-constant multivariate fit at query points ``X``."""

        require_fitted(self, ["map_curve_", "x_design_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
        segments = assign_to_partition(
            x_new,
            self.x_design_,
            self.map_boundaries_,
            extrapolation,
        )
        _record_prediction_policy(self, extrapolation)
        return self.map_segment_means_[segments]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean log marginal-likelihood evaluation on training data (scalar)."""

        require_fitted(self, ["log_evidence_"])
        return float(self.log_evidence_) / max(1, int(self.n_))

    @staticmethod
    def _make_channel_A1_joint(st: _ChannelState, lA0_joint: FloatArray) -> FloatArray:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            mu_hat = st.A1 * np.exp(-st.lA0)
            A1_joint = np.exp(lA0_joint) * mu_hat
        A1_joint[~np.isfinite(A1_joint)] = 0.0
        return A1_joint


class IndependentMultivariateSegmenter(BaseEstimator, RegressorMixin):
    """Multivariate segmenter that fits each channel independently.

    Each channel gets its own boundary vector.

    Attributes
    ----------
    channel_estimators_ : list of BayesBreakSegmenter
        Fitted per-channel segmenters.
    map_curve_ : ndarray of shape (n, d)
        Concatenated piecewise-constant fits.
    log_evidence_ : float
        Sum of per-channel ``log_evidence_`` values.
    """

    def __init__(
        self,
        base_estimator: BayesBreakSegmenter,
        *,
        k_max: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.k_max = k_max

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> IndependentMultivariateSegmenter:
        x_design, y_arr, _ = check_segmentation_input(X, y, multivariate=True)
        n, d = y_arr.shape
        self.n_, self.d_ = int(n), int(d)
        self.x_design_ = x_design
        w_mat = _normalize_multivariate_weights(sample_weight, n, d)

        X_col = x_design.reshape(-1, 1)
        self.channel_estimators_ = []
        pc = np.zeros((n, d), dtype=float)
        brcs: list[FloatArray] = []
        loge = 0.0
        for c in range(d):
            est_c = clone(self.base_estimator)
            if self.k_max is not None:
                est_c.k_max = int(self.k_max)
            est_c.fit(X_col, y_arr[:, c], sample_weight=w_mat[:, c])
            self.channel_estimators_.append(est_c)
            pc[:, c] = est_c.predict(X_col)
            if est_c.bayes_curve_mean_ is not None:
                brcs.append(est_c.bayes_curve_mean_)
            loge += float(est_c.log_evidence_)
        self.map_curve_ = pc
        self.bayes_curve_mean_ = np.column_stack(brcs) if brcs else None
        self.log_evidence_ = float(loge)
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        extrapolation: str | ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> FloatArray:
        require_fitted(self, ["channel_estimators_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
        X_col = x_new.reshape(-1, 1)
        cols = [est.predict(X_col, extrapolation=extrapolation) for est in self.channel_estimators_]
        _record_prediction_policy(self, extrapolation)
        return np.column_stack(cols)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        require_fitted(self, ["channel_estimators_"])
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        total = 0.0
        for c, est in enumerate(self.channel_estimators_):
            total += float(est.score(X, y_arr[:, c]))
        return total / max(1, self.d_)
