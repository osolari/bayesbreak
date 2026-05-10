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

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone

from . import dp as _dp
from .base import BayesBreakSegmenter
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


class SharedBoundaryReplicatesSegmenter(BaseEstimator, RegressorMixin):
    """Shared-boundary multi-subject segmentation via exact boundary-posterior pooling
    (Theorem ``multisubject``).

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

        # Pool by summing log evidences over subjects (Theorem multisubject).
        lA0_pool = np.zeros_like(states[0].lA0)
        mask = np.isfinite(states[0].lA0)
        for st in states:
            mask &= np.isfinite(st.lA0)
        lA0_pool[~mask] = -np.inf
        for st in states:
            np.add(lA0_pool, st.lA0, out=lA0_pool, where=mask)
        # Pool first-moments by summing per-subject contributions; we only use
        # this for the optional Bayes curve, where the "pooled mean" is the
        # arithmetic mean of subject means under the pooled model. We compute
        # the canonical "mean of subject means" via A1_pool[i,j] = exp(lA0_pool)
        # · ((1/S) Σ_s A1_s/exp(lA0_s)).
        # In the report's regression-curve construction the curve is per-subject
        # anyway (see ``map_segment_means_``). We expose A1_pool only for the
        # average curve diagnostic.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mu_per_subject = np.zeros((S, n + 1, n + 1), dtype=float)
            for s, st in enumerate(states):
                ratio = np.where(np.exp(st.lA0) > 0, st.A1 / np.exp(st.lA0), 0.0)
                ratio[~np.isfinite(ratio)] = 0.0
                mu_per_subject[s] = ratio
            mu_avg = mu_per_subject.mean(axis=0)
        A1_pool = np.exp(lA0_pool) * mu_avg
        A1_pool[~np.isfinite(A1_pool)] = 0.0

        # Length prior + p(k).
        u = self._build_boundary_coordinates(self.x_design_)
        self.boundary_coordinates_ = u
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
        # Expose pooled tables under both names for diagnostics access.
        self.log_block_evidence_ = lA0_pool
        self.block_first_moment_ = A1_pool
        return self

    # ---- prediction / scoring ----------------------------------------

    def predict(self, X: ArrayLike) -> FloatArray:
        """Return the per-subject piecewise-constant fits at query points.

        Returns an array of shape ``(S, m)``.
        """

        require_fitted(self, ["map_segment_means_", "x_design_", "map_boundaries_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()

        order = np.argsort(self.x_design_)
        sorted_x = self.x_design_[order]
        pos = np.searchsorted(sorted_x, x_new, side="right") - 1
        pos = np.clip(pos, 0, len(sorted_x) - 1)
        idx = order[pos]
        seg = np.searchsorted(self.boundaries_internal_, idx, side="right") - 1
        seg = np.clip(seg, 0, self.map_segment_means_.shape[1] - 1)
        return self.map_segment_means_[:, seg]

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
