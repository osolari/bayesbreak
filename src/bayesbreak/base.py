"""Core estimator: :class:`BayesBreakSegmenter`.

Implements the report's §4 pipeline end-to-end:

1. Validate ``(X, y, sample_weight)`` (:mod:`bayesbreak.validation`).
2. Estimate family hyperparameters (family-specific hook).
3. Compute the triangular log block-evidence table ``log A^0_{ij}`` and the
   linear first-moment table ``A^1_{ij}`` (family-specific hook).
4. Build the design-aware length-prior table ``log g(Δ_x(i, j))`` from
   ``boundary_coordinates`` (defaults to index-uniform ``g ≡ 1``).
5. Run the sum-product DP for ``log P(y)``, ``P(k|y)``, and the conditional
   boundary-event marginal ``P(b_i = 1 | y, k_map)`` (the §6 calibration target).
6. Run the **max-sum DP with backtracking** for the joint MAP segmentation
   (:func:`bayesbreak.dp.max_sum_segmentation`).
7. Optionally compute the Bayesian regression curve.

scikit-learn API: ``fit(X, y)``, ``predict(X)``, ``score(X, y)``,
``transform(X)``. Constructor arguments are stored untouched; validation
happens inside ``fit``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

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

    Subclasses also declare a ``MOMENT_SIGN_CONTRACT`` class attribute
    (§5 paragraph 5-C1): ``"nonneg"`` when the order-1 block moment
    ``block_first_moment_[i, j]`` targets a strictly-nonnegative
    observation-scale quantity (probability, rate, count mean, Beta mean),
    ``"signed"`` when it can change sign (centered Gaussian mean,
    non-conjugate Laplace test function). The base class default is
    ``"nonneg"``; the Gaussian family overrides to ``"signed"``.

    Parameters
    ----------
    k_max : int, default=50
        Maximum number of segments considered. Internally capped at ``n``.
    estimate_hyper : bool, default=True
        If ``True``, subclasses may estimate hyperparameters empirically.
    regression_curve : {"none", "fixed_k", "mix_k"}, default="none"
        Whether to compute the Bayesian regression curve.
    length_prior : callable or None, default=None
        Optional length-cohesion function ``g(Δ) -> float >= 0`` that defines
        the design-aware partition prior ``p(t|k) ∝ ∏_q g(Δ_x(t_{q-1}, t_q))``
        (eq. ``lengthprior``). ``None`` means ``g ≡ 1`` (index-uniform).
    boundary_coordinates : array-like of shape (n+1,) or None, default=None
        Strictly-increasing candidate boundary coordinates ``u_0 < ... < u_n``
        used to define ``Δ_x(i, j) = u_j - u_i`` (§``sec:notation``). When
        ``None``, default to a midpoint construction from the design ``X``
        (sentinel endpoints below ``x_0`` and above ``x_{n-1}``).

        Per §``sec:notation``: on a regular index grid set ``u_i = i`` (an
        explicit ``np.arange(n + 1)`` reproduces the index-uniform case exactly).
        For interval data the ``u_i`` are the observed interval endpoints.
        Point observations without meaningful physical interval endpoints
        should use the index-uniform prior or supply an external ``u_0``.
    prior_k : callable or None, default=None
        Optional prior on the segment count, ``p(k) -> float`` for
        ``k = 1, ..., k_max``. Returned values are normalized internally.
        ``None`` means a uniform ``p(k)``.

    Notes
    -----
    The joint MAP boundary vector is ``argmax_t p(t|y, k_map)`` via max-sum DP
    with backtracking; it is generally distinct from the vector of marginal
    boundary modes.
    """

    # §5 paragraph 5-C1: nonneg observation-scale targets store directly in
    # log; signed targets store via signed-linear or signed-log accumulators.
    # Families override as needed.
    MOMENT_SIGN_CONTRACT: str = "nonneg"

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        regression_curve: str = "none",
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
    ):
        # Store constructor args untouched (sklearn contract).
        self.k_max = k_max
        self.estimate_hyper = estimate_hyper
        self.regression_curve = regression_curve
        self.length_prior = length_prior
        self.boundary_coordinates = boundary_coordinates
        self.prior_k = prior_k

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
        """Return ``(log_block_evidence, block_first_moment)`` tables.

        Implementations follow the §``sec:setup`` admissibility contract:
        ``log_block_evidence[i, j]`` is finite on every admissible block
        ``(i, j]`` and ``-inf`` on every inadmissible block (minimum-length
        violations, zero usable weight, family-specific domain failures).
        ``block_first_moment[i, j]`` is queried only on admissible blocks;
        for families that target a strictly-nonnegative observation-scale
        mean it is stored directly in linear space, and for sign-changing
        targets (centered Gaussian mean, non-conjugate Laplace test
        functions) it uses signed-linear or signed-log accumulators per
        §5 (signed-moment storage guidance).
        """

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

        Default: a Gaussian fallback around the segment posterior mean.
        Conjugate families override with the exact closed form
        ``Z(α_B + S_new, β_B + W_new) / Z(α_B, β_B)``.
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
    # Length-prior helpers
    # ------------------------------------------------------------------

    def _build_boundary_coordinates(self, x_design: FloatArray) -> FloatArray:
        """Build ``u_0 < ... < u_n`` from the design or the user override.

        Default: midpoints between consecutive design points, with end caps so
        that ``u_0 < x_0`` and ``u_n > x_{n-1}``.
        """

        if self.boundary_coordinates is not None:
            u = np.asarray(self.boundary_coordinates, dtype=float).ravel()
            n = int(x_design.size)
            if u.size != n + 1:
                raise ValueError(f"boundary_coordinates must have length n+1={n+1}; got {u.size}.")
            if not np.all(np.diff(u) > 0):
                raise ValueError("boundary_coordinates must be strictly increasing.")
            return np.ascontiguousarray(u)

        x = np.asarray(x_design, dtype=float).ravel()
        n = int(x.size)
        if n == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
        mids = 0.5 * (x[:-1] + x[1:])
        first_gap = x[1] - x[0]
        last_gap = x[-1] - x[-2]
        u0 = float(x[0] - 0.5 * first_gap)
        un = float(x[-1] + 0.5 * last_gap)
        u = np.concatenate(([u0], mids, [un])).astype(float)
        # Guarantee strict monotonicity (numerical safety).
        eps = 1e-12 * max(1.0, float(np.ptp(x)))
        for i in range(1, u.size):
            if u[i] <= u[i - 1]:
                u[i] = u[i - 1] + eps
        return np.ascontiguousarray(u)

    def _build_log_g_table(self, u: FloatArray, n: int) -> FloatArray | None:
        """Materialise ``log g(Δ_x(i, j))`` from ``self.length_prior``.

        Returns ``None`` (the index-uniform shortcut) when no length prior is set.
        """

        if self.length_prior is None:
            return None
        log_g = np.full((n + 1, n + 1), -np.inf, dtype=float)
        # Vectorise over j for fixed i.
        for i in range(n):
            for j in range(i + 1, n + 1):
                d = float(u[j] - u[i])
                if d <= 0:
                    continue
                gv = float(self.length_prior(d))
                if gv > 0 and np.isfinite(gv):
                    log_g[i, j] = math.log(gv)
        return log_g

    def _build_log_p_k(self, k_max: int) -> FloatArray | None:
        if self.prior_k is None:
            return None
        vals = np.array([float(self.prior_k(k)) for k in range(1, k_max + 1)], dtype=float)
        if np.any(vals < 0):
            raise ValueError("prior_k(k) must return non-negative values.")
        total = float(np.sum(vals))
        if total <= 0:
            raise ValueError("prior_k must put positive mass on at least one k.")
        log_p = np.log(np.maximum(vals / total, 1e-300))
        full = np.full(k_max + 1, -np.inf, dtype=float)
        full[1:] = log_p
        return full

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
        # ---- 1. Validate inputs -------------------------------------------------
        if not isinstance(self.k_max, int | np.integer) or int(self.k_max) < 1:
            raise ValueError("k_max must be a positive integer.")
        if self.regression_curve not in {"none", "fixed_k", "mix_k"}:
            raise ValueError("regression_curve must be 'none', 'fixed_k', or 'mix_k'.")

        x_design, y_arr, w_arr = check_segmentation_input(
            X, y, sample_weight=sample_weight, multivariate=False
        )
        n = int(y_arr.size)
        k_max = min(int(self.k_max), n)

        self.n_ = n
        self.x_design_ = x_design
        self.sample_weight_ = w_arr
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
                f"_compute_block_evidence must return ({n+1}, {n+1}) arrays; "
                f"got {lA0.shape} and {A1.shape}."
            )
        self.log_block_evidence_ = lA0
        self.block_first_moment_ = A1
        # Admissibility mask: True wherever the block routine produced a
        # finite log evidence. The DP layer and ``compute_log_C_k`` operate
        # on the same mask; callers can inspect it for diagnostics or audit
        # traces.
        self.admissibility_mask_ = np.isfinite(lA0)

        # ---- 4. Length-prior + p(k) plumbing ------------------------------------
        u = self._build_boundary_coordinates(x_design)
        self.boundary_coordinates_ = u
        log_g = self._build_log_g_table(u, n)
        self.log_g_table_ = log_g
        log_C_k = _dp.compute_log_C_k(log_g, n, k_max)
        self.log_C_k_ = log_C_k
        log_p_k = self._build_log_p_k(k_max)

        # ---- 5. Sum-product DP -------------------------------------------------
        log_left, log_right = _dp.forward_backward(lA0, n, k_max, log_g_table=log_g)
        self.log_left_ = log_left
        self.log_right_ = log_right

        log_post_k, post_k, log_evidence = _dp.posterior_over_k(
            log_left, n, k_max, log_C_k=log_C_k, log_p_k=log_p_k
        )
        self.log_posterior_k_ = log_post_k
        self.k_posterior_ = post_k
        self.log_evidence_ = float(log_evidence)

        valid_k = np.arange(1, k_max + 1)[np.isfinite(log_post_k)]
        if valid_k.size == 0:
            raise RuntimeError("No valid segment counts produced finite evidence.")
        k_map = int(valid_k[int(np.argmax(log_post_k[valid_k - 1]))])
        self.k_map_ = k_map

        # ---- 6. Boundary marginals (calibration target = conditional on k_map) -
        self.boundary_marginals_ = _dp.boundary_event_marginals_fixed_k(
            log_left, log_right, n, k_map
        )
        self.boundary_location_posterior_ = _dp.boundary_location_posterior(
            log_left, log_right, n, k_map
        )

        # ---- 7. Joint MAP segmentation (max-sum DP + backtracking) -------------
        map_boundaries, log_joint = _dp.max_sum_segmentation(lA0, k_map, log_g_table=log_g)
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
        self.boundaries_internal_ = np.asarray(self.map_boundaries_, dtype=int)

        # Piecewise-constant fit in observation space.
        pc = np.empty(n, dtype=float)
        for s, (a, b) in enumerate(
            zip(self.map_boundaries_[:-1], self.map_boundaries_[1:], strict=False)
        ):
            pc[int(a) : int(b)] = means[s]
        self.map_curve_ = pc

        # ---- 8. Optional Bayesian regression curve -----------------------------
        self.bayes_curve_mean_ = None
        if self.regression_curve == "fixed_k":
            self.bayes_curve_mean_ = _dp.bayes_regression_curve_fixed_k(
                log_left, log_right, lA0, A1, n, k_map, log_g_table=log_g
            )
        elif self.regression_curve == "mix_k":
            self.bayes_curve_mean_ = _dp.bayes_regression_curve_mixed_k(
                log_left, log_right, lA0, A1, n, k_max, post_k, log_g_table=log_g
            )

        return self

    def predict(self, X: ArrayLike, *, mode: str = "map") -> FloatArray:
        require_fitted(self, ["map_segment_means_", "x_design_", "map_boundaries_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()

        if mode == "map":
            return self._evaluate_piecewise_constant(x_new, self.map_segment_means_)
        if mode == "bayes":
            if self.bayes_curve_mean_ is None:
                raise RuntimeError("mode='bayes' requires regression_curve != 'none' at fit time.")
            idx = self._nearest_training_index(x_new)
            return self.bayes_curve_mean_[idx]
        raise ValueError(f"Unknown mode={mode!r}; expected 'map' or 'bayes'.")

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        """Mean posterior-predictive log-density of ``(X, y)`` (higher is better)."""

        require_fitted(self, ["map_boundaries_"])
        y_arr = np.asarray(y, dtype=float)
        total = posterior_predictive_logpdf(
            self, X, y_arr, sample_weight=sample_weight, per_sample=False
        )
        assert isinstance(total, float)
        return float(total) / max(1, y_arr.shape[0])

    def transform(self, X: ArrayLike) -> NDArray[np.intp]:
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
        require_fitted(self, ["map_boundaries_", "map_segment_means_", "k_map_"])
        assert (
            self.k_map_ is not None
            and self.map_boundaries_ is not None
            and self.map_segment_means_ is not None
        )
        return int(self.k_map_), list(self.map_boundaries_), self.map_segment_means_.copy()

    def get_log_evidence(self) -> float:
        require_fitted(self, ["log_evidence_"])
        assert self.log_evidence_ is not None
        return float(self.log_evidence_)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_piecewise_constant(
        self, x_new: FloatArray, segment_means: FloatArray
    ) -> FloatArray:
        idx = self._nearest_training_index(x_new)
        seg = np.searchsorted(self.boundaries_internal_, idx, side="right") - 1
        seg = np.clip(seg, 0, len(segment_means) - 1)
        return segment_means[seg]

    def _nearest_training_index(self, x_new: FloatArray) -> NDArray[np.intp]:
        order = np.argsort(self.x_design_)
        sorted_x = self.x_design_[order]
        pos = np.searchsorted(sorted_x, x_new, side="right") - 1
        pos = np.clip(pos, 0, len(sorted_x) - 1)
        return order[pos]

    def __sklearn_tags__(self):  # pragma: no cover - sklearn tag plumbing
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
