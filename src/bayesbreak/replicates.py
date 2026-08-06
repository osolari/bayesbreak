r"""Exact **boundary-posterior pooling** for shared-boundary replicates
(Theorem ``multisubject``).

Given subject sequences :math:`\{y^{(s)}\}_{s=1}^S` on a *common* index grid
and subject-specific weights, the shared-boundary model has subject-specific
segment parameters :math:`\theta^{(s)}_q` integrated out per subject under
their own conjugate priors; subject-level block evidences then multiply, so

.. math::
    \log \tilde A^{(0,\mathrm{pool})}_{ij}
    = \sum_{s=1}^S \log \tilde A^{(0,s)}_{ij}.

The DP layer of :mod:`bayesbreak.dp` consumes the pooled table unchanged
and yields the **exact posterior over the shared boundary vector**
(:math:`P(k|y)`, boundary marginals, MAP segmentation). Subject-specific
segment-parameter posteriors are *not* a single tractable joint object —
they are recovered **conditionally** on a chosen or averaged boundary
configuration via per-subject conjugate updates. This estimator stores
per-subject MAP-segment posterior means under the pooled MAP partition; for
other boundary configurations (e.g. averaging over :math:`P(t \mid y, k_{\mathrm{map}})`)
recompute conditionally from the per-subject ``BayesBreakSegmenter``.

The estimator wraps any :class:`~bayesbreak.base.BayesBreakSegmenter` family.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone

from . import dp as _dp
from .base import BayesBreakSegmenter
from .prediction import ExtrapolationPolicy, _record_prediction_policy, assign_to_partition
from .validation import check_sample_weight, require_fitted

FloatArray = NDArray[np.floating]


@dataclass
class _SubjectState:
    est: BayesBreakSegmenter
    hyper: dict[str, float]
    lA0: FloatArray
    A1: FloatArray
    weights: FloatArray
    y: FloatArray


@dataclass(frozen=True)
class SharedBoundaryInput:
    """Aligned subject-level log-evidence tables on one boundary axis."""

    coordinate_axis: Sequence[float]
    sequence_tables: Sequence[FloatArray]


def aggregate_block_log_evidence(data: SharedBoundaryInput) -> FloatArray:
    """Accurately sum aligned finite log evidence and preserve zero support."""

    tables = [np.asarray(table, dtype=float) for table in data.sequence_tables]
    if not tables:
        raise ValueError("At least one sequence log-evidence table is required")
    shape = tables[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Sequence log-evidence tables must be square")
    if any(table.shape != shape for table in tables[1:]):
        raise ValueError("All sequence log-evidence tables must have the same shape")

    axis = np.asarray(data.coordinate_axis, dtype=float)
    if axis.ndim != 1 or axis.size != shape[0]:
        raise ValueError(f"coordinate_axis must have length {shape[0]}")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0):
        raise ValueError("coordinate_axis must be finite and strictly increasing")
    if any(np.any(np.isnan(table) | np.isposinf(table)) for table in tables):
        raise FloatingPointError("Log-evidence tables may contain finite values or -inf only")

    supported = np.logical_and.reduce([np.isfinite(table) for table in tables])
    pooled = np.full(shape, -np.inf, dtype=float)
    for start, stop in zip(*np.nonzero(supported), strict=True):
        value = math.fsum(float(table[start, stop]) for table in tables)
        if not math.isfinite(value):
            raise FloatingPointError(
                f"Pooled log evidence is not representable at block ({start}, {stop})"
            )
        pooled[start, stop] = value
    return pooled


class SharedBoundaryReplicatesSegmenter(BaseEstimator, RegressorMixin):
    """Shared-boundary multi-subject segmentation via exact boundary-posterior pooling
    (Theorem ``multisubject``).

    Identifiability of the pooled boundary structure is the content of
    Proposition ``prop:shared-boundary-identifiability``: under the
    common-grid + conditional-independence assumption
    ``ass:cond-indep-subjects``, the shared boundary vector is identifiable
    from the pooled subject-level evidences whenever the identifying-block
    hypothesis (Remark ``rem:identifying-block``) holds.

    Parameters
    ----------
    base_estimator : BayesBreakSegmenter
        Univariate block family template; cloned per subject.
    k_max : int or None
        Optional override for the maximum segment count. Defaults to
        ``base_estimator.k_max``.
    length_prior : callable or None
        Optional length-cohesion ``g(Δ) -> float >= 0``.
    boundary_coordinates : array-like of shape (n+1,) or None
        Candidate boundary coordinates ``u_0 < ... < u_n``. Defaults to the
        midpoint construction in :class:`~bayesbreak.base.BayesBreakSegmenter`.
    prior_k : callable or None
        Prior on the segment count.

    Attributes
    ----------
    n_, S_ : int
    k_map_ : int
    map_boundaries_ : list[int]
    map_segment_means_ : ndarray of shape (S, k_map)
        Subject-specific posterior means on the pooled MAP partition.
    boundary_marginals_ : ndarray of shape (n-1,)
        Conditional ``P(b_i = 1 | y, k_map)``.
    log_evidence_ : float
    log_C_k_ : ndarray
    """

    def __init__(
        self,
        base_estimator: BayesBreakSegmenter,
        *,
        k_max: int | None = None,
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
    ):
        self.base_estimator = base_estimator
        self.k_max = k_max
        self.length_prior = length_prior
        self.boundary_coordinates = boundary_coordinates
        self.prior_k = prior_k

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _normalize_inputs(
        Y: Sequence[ArrayLike] | ArrayLike,
        sample_weight: Sequence[ArrayLike] | ArrayLike | None,
    ) -> tuple[list[FloatArray], list[FloatArray]]:
        """Coerce ``Y`` into a list of 1-D float arrays of identical length."""

        if isinstance(Y, np.ndarray) and Y.ndim == 2:
            ys = [np.asarray(row, dtype=float) for row in Y]
        elif isinstance(Y, list | tuple):
            ys = [np.asarray(arr, dtype=float).ravel() for arr in Y]
        else:
            arr = np.asarray(Y, dtype=float)
            if arr.ndim == 1:
                ys = [arr]
            elif arr.ndim == 2:
                ys = [np.asarray(row, dtype=float) for row in arr]
            else:
                raise ValueError(f"Y must be 1-D, 2-D, or list of 1-D arrays; got {arr.shape}.")
        if not ys:
            raise ValueError("At least one subject sequence is required.")
        n = ys[0].shape[0]
        for s, y in enumerate(ys):
            if y.ndim != 1 or y.shape[0] != n:
                raise ValueError(
                    f"All subjects must have length {n}; subject {s} has shape {y.shape}."
                )
            if not np.all(np.isfinite(y)):
                raise ValueError(f"Subject {s} contains non-finite y.")

        S = len(ys)
        if sample_weight is None:
            ws = [np.ones(n, dtype=float) for _ in range(S)]
        elif isinstance(sample_weight, list | tuple):
            if len(sample_weight) != S:
                raise ValueError(f"sample_weight must have length {S}.")
            ws = [check_sample_weight(w, n) for w in sample_weight]
        else:
            arr = np.asarray(sample_weight)
            if arr.ndim == 1:
                w = check_sample_weight(arr, n)
                ws = [w for _ in range(S)]
            elif arr.ndim == 2:
                if arr.shape != (S, n):
                    raise ValueError(
                        f"2-D sample_weight must have shape ({S}, {n}); got {arr.shape}."
                    )
                ws = [check_sample_weight(arr[s], n) for s in range(S)]
            else:
                raise ValueError("sample_weight must be None, list, 1-D, or 2-D.")
        return ys, ws

    # ---- sklearn API --------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Sequence[ArrayLike] | ArrayLike,
        *,
        sample_weight: Sequence[ArrayLike] | ArrayLike | None = None,
    ) -> SharedBoundaryReplicatesSegmenter:
        ys, ws = self._normalize_inputs(y, sample_weight)
        n = ys[0].shape[0]
        S = len(ys)
        self.n_, self.S_ = int(n), int(S)

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 2:
            x_design = X_arr[:, 0]
        else:
            x_design = X_arr.ravel()
        if x_design.shape[0] != n:
            raise ValueError(f"X must have length {n}; got {x_design.shape[0]}.")
        self.x_design_ = np.ascontiguousarray(x_design)

        k_cap = int(self.k_max) if self.k_max is not None else int(self.base_estimator.k_max)
        k_max = min(max(1, n), k_cap)

        # Per-subject hyperparameter estimation + block evidence tables.
        states: list[_SubjectState] = []
        for s in range(S):
            est_s = clone(self.base_estimator)
            hyper_s = est_s._estimate_hyperparameters(ys[s], ws[s])
            lA0_s, A1_s = est_s._compute_block_evidence(ys[s], hyper_s, ws[s])
            states.append(
                _SubjectState(est=est_s, hyper=hyper_s, lA0=lA0_s, A1=A1_s, weights=ws[s], y=ys[s])
            )

        u = self._build_boundary_coordinates(self.x_design_)
        self.boundary_coordinates_ = u
        lA0_pool = aggregate_block_log_evidence(
            SharedBoundaryInput(
                coordinate_axis=u,
                sequence_tables=[state.lA0 for state in states],
            )
        )

        # A shared-boundary model has subject-specific segment parameters, not
        # one pooled A^(1). Expose the arithmetic mean of subject posterior
        # means as a bounded diagnostic without exponentiating block evidence.
        block_mean = np.zeros((n + 1, n + 1), dtype=float)
        for start, stop in zip(*np.nonzero(np.isfinite(lA0_pool)), strict=True):
            means = [
                state.est._segment_posterior_mean(
                    int(start),
                    int(stop),
                    state.y,
                    state.hyper,
                    state.weights,
                )
                for state in states
            ]
            if not all(math.isfinite(mean) for mean in means):
                raise FloatingPointError(
                    f"Nonfinite subject posterior mean at block ({start}, {stop})"
                )
            block_mean[start, stop] = math.fsum(means) / S

        # Length prior + p(k).
        log_g = self._build_log_g_table(u, n)
        self.log_g_table_ = log_g
        log_C_k = _dp.compute_log_C_k(log_g, n, k_max)
        self.log_C_k_ = log_C_k
        log_p_k = self._build_log_p_k(k_max)

        log_left, log_right = _dp.forward_backward(lA0_pool, n, k_max, log_g_table=log_g)
        log_post_k, post_k, log_E = _dp.posterior_over_k(
            log_left, n, k_max, log_C_k=log_C_k, log_p_k=log_p_k
        )
        self.log_left_ = log_left
        self.log_right_ = log_right
        self.log_posterior_k_ = log_post_k
        self.k_posterior_ = post_k
        self.log_evidence_ = float(log_E)

        valid = np.arange(1, k_max + 1)[np.isfinite(log_post_k)]
        if valid.size == 0:
            raise RuntimeError("No valid segment counts produced finite evidence.")
        k_map = int(valid[int(np.argmax(log_post_k[valid - 1]))])
        self.k_map_ = k_map

        self.boundary_marginals_ = _dp.boundary_event_marginals_fixed_k(
            log_left, log_right, n, k_map
        )
        self.boundary_location_posterior_ = _dp.boundary_location_posterior(
            log_left, log_right, n, k_map
        )

        map_boundaries, log_joint = _dp.max_sum_segmentation(lA0_pool, k_map, log_g_table=log_g)
        self.map_boundaries_ = list(map_boundaries)
        self.boundaries_internal_ = np.asarray(self.map_boundaries_, dtype=int)
        self.log_joint_map_ = float(log_joint)

        # Per-subject segment posterior means on the pooled MAP partition.
        means = np.zeros((S, k_map), dtype=float)
        per_subject_curves = np.zeros((S, n), dtype=float)
        for s, st in enumerate(states):
            for q, (a, b) in enumerate(
                zip(self.map_boundaries_[:-1], self.map_boundaries_[1:], strict=False)
            ):
                mu_q = float(
                    st.est._segment_posterior_mean(int(a), int(b), st.y, st.hyper, st.weights)
                )
                means[s, q] = mu_q
                per_subject_curves[s, int(a) : int(b)] = mu_q
        self.map_segment_means_ = means
        self.map_curve_ = per_subject_curves

        self.subject_states_ = states
        self.log_block_evidence_ = lA0_pool
        self.block_posterior_mean_ = block_mean
        return self

    # ---- prediction / scoring ----------------------------------------

    def predict(
        self,
        X: ArrayLike,
        *,
        extrapolation: str | ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> FloatArray:
        """Return the per-subject piecewise-constant fits at query points.

        Returns an array of shape ``(S, m)``.
        """

        require_fitted(self, ["map_segment_means_", "x_design_", "map_boundaries_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()

        segments = assign_to_partition(
            x_new,
            self.x_design_,
            self.map_boundaries_,
            extrapolation,
        )
        _record_prediction_policy(self, extrapolation)
        return self.map_segment_means_[:, segments]

    def score(self, X: ArrayLike, y: Sequence[ArrayLike] | ArrayLike) -> float:
        """Mean per-subject log evidence on training data (scalar)."""

        require_fitted(self, ["log_evidence_"])
        return float(self.log_evidence_) / max(1, int(self.n_) * int(self.S_))

    # ---- internals: reuse the segmenter's plumbing -------------------

    def _build_boundary_coordinates(self, x_design: FloatArray) -> FloatArray:
        if self.boundary_coordinates is not None:
            u = np.asarray(self.boundary_coordinates, dtype=float).ravel()
            if u.size != x_design.size + 1:
                raise ValueError("boundary_coordinates must have length n+1.")
            if not np.all(np.diff(u) > 0):
                raise ValueError("boundary_coordinates must be strictly increasing.")
            return np.ascontiguousarray(u)
        # Reuse the segmenter's default rule.
        x = x_design
        n = int(x.size)
        if n == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
        mids = 0.5 * (x[:-1] + x[1:])
        first_gap = x[1] - x[0]
        last_gap = x[-1] - x[-2]
        u0 = float(x[0] - 0.5 * first_gap)
        un = float(x[-1] + 0.5 * last_gap)
        return np.concatenate(([u0], mids, [un])).astype(float)

    def _build_log_g_table(self, u: FloatArray, n: int) -> FloatArray | None:
        if self.length_prior is None:
            return None
        log_g = np.full((n + 1, n + 1), -np.inf, dtype=float)
        for i in range(n):
            for j in range(i + 1, n + 1):
                d = float(u[j] - u[i])
                if d <= 0:
                    continue
                gv = float(self.length_prior(d))
                if gv > 0 and np.isfinite(gv):
                    log_g[i, j] = float(np.log(gv))
        return log_g

    def _build_log_p_k(self, k_max: int) -> FloatArray | None:
        if self.prior_k is None:
            return None
        vals = np.array([float(self.prior_k(k)) for k in range(1, k_max + 1)], dtype=float)
        total = float(np.sum(vals))
        if total <= 0:
            raise ValueError("prior_k must put positive mass on at least one k.")
        log_p = np.log(np.maximum(vals / total, 1e-300))
        full = np.full(k_max + 1, -np.inf, dtype=float)
        full[1:] = log_p
        return full
