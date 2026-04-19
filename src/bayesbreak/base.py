"""Core estimator: :class:`BayesBreakSegmenter`.

The segmenter implements the report's §4 pipeline end-to-end:

1. Validate ``(X, y, sample_weight)`` (:mod:`bayesbreak.validation`).
2. Estimate family hyperparameters (family-specific hook).
3. Compute the triangular log block-evidence table ``log A^0_{ij}`` and linear
   first-moment table ``A^1_{ij}`` (family-specific hook).
4. Run the sum-product DP for ``log P(y)``, ``P(k|y)``, and boundary-event
   marginals (:mod:`bayesbreak.dp`).
5. Run the **max-sum DP with backtracking** for the joint MAP segmentation
   (:func:`bayesbreak.dp.max_sum_segmentation`).
6. Optionally compute the Bayesian regression curve (expected latent signal).

The public API follows strict scikit-learn conventions: ``fit(X, y)``,
``predict(X)``, ``score(X, y)``, ``transform(X)``. Constructor arguments are
stored untouched; validation happens inside ``fit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from . import dp as _dp
from .prediction import posterior_predictive_logpdf
from .validation import check_segmentation_input, require_fitted

FloatArray = NDArray[np.floating]


class BayesBreakSegmenter(BaseEstimator, RegressorMixin, TransformerMixin, ABC):
    r"""Abstract scikit-learn compatible Bayesian segmenter.

    Subclasses implement three family-specific hooks:

    - :meth:`_estimate_hyperparameters`
    - :meth:`_compute_block_evidence`
    - :meth:`_segment_posterior_mean`

    and (for the prediction layer) :meth:`posterior_predictive_logpdf_block`.

    Parameters
    ----------
    k_max : int, default=50
        Maximum number of segments considered. Internally capped at ``n``.
    estimate_hyper : bool, default=True
        If ``True``, subclasses may estimate hyperparameters empirically. If
        ``False``, the user must provide them via the constructor.
    regression_curve : {"none", "fixed_k", "mix_k"}, default="none"
        Whether to compute the Bayesian regression curve (posterior mean of
        the latent signal).

    Attributes
    ----------
    n_ : int
        Number of training observations.
    x_design_ : ndarray of shape (n,)
        Stored design points.
    hyper_ : dict
        Family-specific hyperparameters used at fit time.
    log_block_evidence_ : ndarray of shape (n+1, n+1)
        Triangular ``log A^0_{ij}`` table.
    block_first_moment_ : ndarray of shape (n+1, n+1)
        Linear ``A^1_{ij}`` table.
    log_left_, log_right_ : ndarray
        Sum-product DP tables.
    log_evidence_ : float
        ``log P(y)``.
    k_posterior_ : ndarray of shape (k_max,)
        ``P(k | y)``.
    k_map_ : int
        ``argmax_k P(k | y)``.
    boundary_marginals_ : ndarray of shape (n-1,)
        Per-index boundary-event marginals ``P(b_i = 1 | y)``.
    boundary_location_posterior_ : ndarray of shape (k_map-1, n+1)
        ``P(t_p = h | y, k_map)`` per boundary.
    map_boundaries_ : list of int
        Joint MAP boundary vector, including endpoints ``0`` and ``n``.
    map_segment_means_ : ndarray of shape (k_map,)
        Posterior-mean segment parameters under the MAP segmentation.
    bayes_curve_mean_ : ndarray of shape (n,) or None
        Posterior mean of the latent signal (if ``regression_curve != "none"``).
    sample_weight_ : ndarray of shape (n,)
        Weights actually used at fit time.

    Notes
    -----
    The joint MAP boundary vector is the argmax of ``p(t | y, k_map)`` via
    max-sum DP with backtracking — distinct from (and generally not equal to)
    the top-:math:`k-1` marginal boundary modes.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: str = "none",
    ):
        # Store constructor args untouched (sklearn contract).
        self.k_max = k_max
        self.estimate_hyper = estimate_hyper
        self.regression_curve = regression_curve

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _estimate_hyperparameters(
        self, y: FloatArray, sample_weight: FloatArray
    ) -> dict[str, float]:
        """Return block-family hyperparameters; honor ``estimate_hyper``."""

    @abstractmethod
    def _compute_block_evidence(
        self, y: FloatArray, hyper: dict[str, float], sample_weight: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(log_block_evidence, block_first_moment)`` tables."""

    @abstractmethod
    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: FloatArray,
        hyper: dict[str, float],
        sample_weight: FloatArray,
    ) -> float:
        """Posterior mean of the segment parameter on block ``(a, b]``."""

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: FloatArray,
        w_new: FloatArray,
    ) -> FloatArray:
        """Per-sample log posterior-predictive density on MAP block ``(a, b]``.

        Default implementation evaluates a Gaussian approximation around the
        segment posterior mean, matching the stability-bound claims for
        non-conjugate families. Conjugate families override this with the
        closed-form ratio ``Z(α_B + S_new, β_B + W_new) / Z(α_B, β_B)``.
        """

        require_fitted(self, ["map_segment_means_", "hyper_"])
        assert self.map_segment_means_ is not None and self.boundaries_internal_ is not None

        seg_idx = int(np.searchsorted(self.boundaries_internal_, a, side="right"))
        seg_idx = min(max(seg_idx - 1, 0), len(self.map_segment_means_) - 1)
        mu = float(self.map_segment_means_[seg_idx])

        sigma2 = float(self.hyper_.get("sigma2", 1.0)) if self.hyper_ is not None else 1.0
        denom = 2.0 * sigma2
        return -0.5 * np.log(2.0 * np.pi * sigma2) - w_new * (y_new - mu) ** 2 / denom

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> BayesBreakSegmenter:
        """Fit the segmenter to ``(X, y)``.

        Parameters
        ----------
        X : array-like of shape (n,) or (n, 1)
            Design points (locations). The first column is used if 2-D.
        y : array-like of shape (n,)
            Ordered response sequence.
        sample_weight : array-like of shape (n,), scalar, or None
            Per-observation exposure / precision (see
            :func:`bayesbreak.validation.check_sample_weight`).

        Returns
        -------
        self
        """

        # ---- 1. Validate inputs -------------------------------------------------
        if not isinstance(self.k_max, int | np.integer) or int(self.k_max) < 1:
            raise ValueError("k_max must be a positive integer.")
        if self.regression_curve not in {"none", "fixed_k", "mix_k"}:
            raise ValueError("regression_curve must be one of: 'none', 'fixed_k', 'mix_k'.")

        x_design, y_arr, w_arr = check_segmentation_input(
            X, y, sample_weight=sample_weight, multivariate=False
        )
        n = int(y_arr.size)
        k_max = min(int(self.k_max), n)

        self.n_ = n
        self.x_design_ = x_design
        self.sample_weight_ = w_arr
        # Cached for use by family-specific posterior_predictive_logpdf_block.
        self._y_train_ = y_arr.copy()

        # ---- 2. Hyperparameters ------------------------------------------------
        hyper = self._estimate_hyperparameters(y_arr, w_arr)
        if not isinstance(hyper, dict):
            raise TypeError("_estimate_hyperparameters must return a dict")
        self.hyper_ = {str(k): float(v) for k, v in hyper.items()}

        # ---- 3. Block evidences ------------------------------------------------
        lA0, A1 = self._compute_block_evidence(y_arr, self.hyper_, w_arr)
        if lA0.shape != (n + 1, n + 1) or A1.shape != (n + 1, n + 1):
            raise ValueError(
                "_compute_block_evidence must return arrays of shape "
                f"({n+1},{n+1}); got {lA0.shape} and {A1.shape}."
            )
        self.log_block_evidence_ = lA0
        self.block_first_moment_ = A1

        # ---- 4. Sum-product DP -------------------------------------------------
        log_left, log_right = _dp.forward_backward(lA0, n, k_max)
        self.log_left_ = log_left
        self.log_right_ = log_right

        log_post_k, post_k, log_evidence = _dp.posterior_over_k(log_left, n, k_max)
        self.log_posterior_k_ = log_post_k
        self.k_posterior_ = post_k
        self.log_evidence_ = float(log_evidence)

        valid_k = np.arange(1, k_max + 1)[np.isfinite(log_post_k)]
        if valid_k.size == 0:
            raise RuntimeError("No valid segment counts produced finite evidence.")
        # Use posterior *mode*, not mean, to match the report's k_hat = argmax.
        k_map = int(valid_k[int(np.argmax(log_post_k[valid_k - 1]))])
        self.k_map_ = k_map

        # ---- 5. Boundary posteriors -------------------------------------------
        d1 = _dp.boundary_event_marginals(log_left, log_right, log_post_k, n, k_max)
        self.boundary_marginals_ = d1
        self.boundary_location_posterior_ = _dp.boundary_location_posterior(
            log_left, log_right, n, k_map
        )

        # ---- 6. Joint MAP segmentation (max-sum DP + backtracking) ------------
        map_boundaries, log_joint = _dp.max_sum_segmentation(lA0, k_map)
        self.map_boundaries_ = list(map_boundaries)
        self.log_joint_map_ = float(log_joint)

        # Per-segment posterior means on the MAP partition.
        means = np.zeros(k_map, dtype=float)
        for s, (a, b) in enumerate(
            zip(self.map_boundaries_[:-1], self.map_boundaries_[1:], strict=False)
        ):
            means[s] = float(
                self._segment_posterior_mean(int(a), int(b), y_arr, self.hyper_, w_arr)
            )
        self.map_segment_means_ = means
        # Internal array version for searchsorted lookups.
        self.boundaries_internal_ = np.asarray(self.map_boundaries_, dtype=int)

        # Piecewise-constant fit in the observation space (used by predict).
        pc = np.empty(n, dtype=float)
        for s, (a, b) in enumerate(
            zip(self.map_boundaries_[:-1], self.map_boundaries_[1:], strict=False)
        ):
            pc[int(a) : int(b)] = means[s]
        self.map_curve_ = pc

        # ---- 7. Optional Bayesian regression curve ----------------------------
        self.bayes_curve_mean_ = None
        if self.regression_curve == "fixed_k":
            self.bayes_curve_mean_ = _dp.bayes_regression_curve_fixed_k(
                log_left, log_right, lA0, A1, n, k_map
            )
        elif self.regression_curve == "mix_k":
            self.bayes_curve_mean_ = _dp.bayes_regression_curve_mixed_k(
                log_left, log_right, lA0, A1, n, k_max, post_k
            )

        return self

    def predict(self, X: ArrayLike, *, mode: str = "map") -> FloatArray:
        """Return the piecewise-constant fit at query points.

        Parameters
        ----------
        X : array-like of shape (m,) or (m, 1)
            Query design points.
        mode : {"map", "bayes"}, default="map"
            ``"map"``: piecewise-constant using the MAP segment means.
            ``"bayes"``: posterior-mean latent signal (requires
            ``regression_curve != "none"`` at fit time).

        Returns
        -------
        ndarray of shape (m,)
        """

        require_fitted(self, ["map_segment_means_", "x_design_", "map_boundaries_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()

        if mode == "map":
            return self._evaluate_piecewise_constant(x_new, self.map_segment_means_)
        if mode == "bayes":
            if self.bayes_curve_mean_ is None:
                raise RuntimeError("mode='bayes' requires regression_curve != 'none' at fit time.")
            # Nearest-neighbor lookup on the training design.
            idx = self._nearest_training_index(x_new)
            return self.bayes_curve_mean_[idx]
        raise ValueError(f"Unknown mode={mode!r}; expected 'map' or 'bayes'.")

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        """Mean posterior-predictive log-density of ``(X, y)`` (higher is better).

        Follows §8 of the report. This is out-of-sample compatible: pass a
        held-out ``(X_test, y_test)``. The total log-density is normalised by
        the number of samples so that the scale is comparable across splits.
        """

        require_fitted(self, ["map_boundaries_"])
        y_arr = np.asarray(y, dtype=float)
        total = posterior_predictive_logpdf(
            self, X, y_arr, sample_weight=sample_weight, per_sample=False
        )
        assert isinstance(total, float)
        return float(total) / max(1, y_arr.shape[0])

    def transform(self, X: ArrayLike) -> NDArray[np.intp]:
        """Return the segment index assigned to each query point.

        The segmenter acts as a featurizer: ``transform(X)`` returns an integer
        vector in ``{0, ..., k_map - 1}`` suitable for downstream pipeline
        stages.
        """

        require_fitted(self, ["boundaries_internal_", "x_design_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
        idx = self._nearest_training_index(x_new)
        seg = np.searchsorted(self.boundaries_internal_, idx, side="right") - 1
        return np.clip(seg, 0, len(self.boundaries_internal_) - 2)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_map_segmentation(self) -> tuple[int, list[int], FloatArray]:
        """Return ``(k_map, map_boundaries, map_segment_means)``."""

        require_fitted(self, ["map_boundaries_", "map_segment_means_", "k_map_"])
        assert (
            self.k_map_ is not None
            and self.map_boundaries_ is not None
            and self.map_segment_means_ is not None
        )
        return int(self.k_map_), list(self.map_boundaries_), self.map_segment_means_.copy()

    def get_log_evidence(self) -> float:
        """Return ``log P(y)`` under the fitted model (training-sequence evidence)."""

        require_fitted(self, ["log_evidence_"])
        assert self.log_evidence_ is not None
        return float(self.log_evidence_)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_piecewise_constant(
        self, x_new: FloatArray, segment_means: FloatArray
    ) -> FloatArray:
        """Look up segment means for new design points."""

        idx = self._nearest_training_index(x_new)
        seg = np.searchsorted(self.boundaries_internal_, idx, side="right") - 1
        seg = np.clip(seg, 0, len(segment_means) - 1)
        return segment_means[seg]

    def _nearest_training_index(self, x_new: FloatArray) -> NDArray[np.intp]:
        """Return, for each new ``x``, the training index of the nearest design point."""

        order = np.argsort(self.x_design_)
        sorted_x = self.x_design_[order]
        pos = np.searchsorted(sorted_x, x_new, side="right") - 1
        pos = np.clip(pos, 0, len(sorted_x) - 1)
        return order[pos]

    def __sklearn_tags__(self):  # pragma: no cover - minor tag plumbing
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:
            return {}
        try:
            tags.target_tags.required = True
        except AttributeError:
            pass
        try:
            tags.input_tags.allow_nan = False
        except AttributeError:
            pass
        return tags
