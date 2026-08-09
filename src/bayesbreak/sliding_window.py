"""Sliding-window decomposition for very large sequences (§5b
"Computational regime" planned approximation).

The exact BayesBreak DP is ``Θ(k_max n²)`` time and at least
``Θ(k_max n)`` working memory (``prop:bb-complexity``). At
``n ≳ 10^5`` the precomputation and DP become impractical. §5b notes that
the right substitution for very large ``n`` is a sliding-window
decomposition, which approximates the posterior over the full sequence
rather than computing it exactly: stitched results no longer inherit
``thm:dp-correctness`` and the boundary-event-sum identity is only
approximate near window seams. The decomposition is an engineering
device — it does not extend the framework's exact guarantees.

This module exposes :class:`SlidingWindowSegmenter`, a thin scikit-learn
compatible wrapper that:

1. Splits the sequence into overlapping windows of size ``window_size``
   with stride ``window_size - overlap``.
2. Fits an independent clone of the supplied ``base_estimator`` on each
   window.
3. Stitches per-window MAP boundaries and Bayes curves into global
   results, averaging boundary marginals across windows that share
   interior indices.

Limitations (documented inline because they shape what callers should
expect):

- The block-evidence array of any window covers only its own
  observations, so segments crossing a window seam are cut by the
  decomposition; choose ``overlap`` large enough to absorb the longest
  expected segment.
- The global ``log_evidence`` of the full sequence is **not** the sum
  of per-window log-evidences (independent priors collide at seams);
  ``log_evidence_`` is reported as the sum of per-window log-evidences
  and is labelled an *approximate* quantity in the docstring.
- Boundary-event marginals on indices covered by multiple windows are
  averaged across those windows; the sum-to-``k − 1`` identity holds
  per window but not globally.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .base import BayesBreakSegmenter
from .prediction import ExtrapolationPolicy, _assign_training_positions, _record_prediction_policy
from .validation import check_segmentation_input, require_fitted

FloatArray = NDArray[np.floating]


class SlidingWindowSegmenter(BaseEstimator, RegressorMixin):
    r"""Overlap-stitched sliding-window approximation of BayesBreak.

    Parameters
    ----------
    base_estimator : BayesBreakSegmenter
        Per-window block family + DP configuration. The estimator's
        ``k_max`` is interpreted as the *per-window* segment cap.
    window_size : int
        Number of observations per window. Must be greater than ``overlap``.
    overlap : int, default 0
        Number of observations shared between consecutive windows. A larger
        overlap improves boundary recovery at seams at the cost of more
        per-window DP runs.

    Attributes
    ----------
    n_ : int
    map_boundaries_ : list[int]
        Stitched global MAP boundaries: union of per-window MAP boundaries,
        mapped to the global index axis and de-duplicated.
    map_curve_ : ndarray of shape (n,)
        Piecewise-constant fit on the global axis. Where windows overlap,
        the curve averages the per-window segment-mean predictions.
    boundary_marginals_ : ndarray of shape (n-1,)
        Approximate global boundary marginals ``P(b_i = 1 | y)`` averaged
        across windows that contain index ``i``.
    log_evidence_ : float
        Sum of per-window ``log_evidence_`` values; an **approximate**
        sequence log-evidence that the decomposition does not equate to
        the exact full-sequence ``log p(y)``.
    windows_ : list[tuple[int, int, BayesBreakSegmenter]]
        Per-window ``(start, stop, fitted_estimator)`` triples for
        downstream inspection.
    k_hat_ : int
        Total interior boundaries plus one (``len(map_boundaries_) + 1``).

    Notes
    -----
    The decomposition does not inherit ``thm:dp-correctness`` exactly —
    boundary-event identities hold only per window — and the reported
    ``log_evidence_`` is approximate. Use ``SharedBoundaryReplicatesSegmenter``
    or the full DP whenever ``n`` is small enough that
    ``prop:bb-complexity`` is feasible.
    """

    def __init__(
        self,
        base_estimator: BayesBreakSegmenter,
        *,
        window_size: int,
        overlap: int = 0,
    ):
        self.base_estimator = base_estimator
        self.window_size = int(window_size)
        self.overlap = int(overlap)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> SlidingWindowSegmenter:
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.overlap < 0 or self.overlap >= self.window_size:
            raise ValueError("overlap must satisfy 0 <= overlap < window_size.")

        x_design, y_arr, w_arr = check_segmentation_input(
            X, y, sample_weight=sample_weight, multivariate=False
        )
        n = int(y_arr.size)
        self.n_ = n
        self.x_design_ = x_design

        if n <= self.window_size:
            # Fast path: single window. Behaviour matches the wrapped
            # base estimator exactly.
            est = clone(self.base_estimator).fit(X, y, sample_weight=sample_weight)
            self.map_boundaries_ = list(est.map_boundaries_)
            self.map_curve_ = np.asarray(est.map_curve_, dtype=float)
            self.boundary_marginals_ = np.asarray(est.boundary_marginals_, dtype=float)
            self.log_evidence_ = float(est.log_evidence_)
            self.windows_ = [(0, n, est)]
            self.k_hat_ = int(est.k_map_)
            return self

        stride = self.window_size - self.overlap
        windows: list[tuple[int, int, Any]] = []
        starts = list(range(0, max(1, n - self.window_size + 1), stride))
        if starts[-1] + self.window_size < n:
            # Anchor the final window to the right end so we cover the tail.
            starts.append(n - self.window_size)

        for s in starts:
            e = min(s + self.window_size, n)
            y_win = y_arr[s:e]
            w_win = w_arr[s:e] if w_arr is not None else None
            x_win = x_design[s:e].reshape(-1, 1)
            est = clone(self.base_estimator).fit(x_win, y_win, sample_weight=w_win)
            windows.append((s, e, est))
        self.windows_ = windows

        # Stitch boundaries (interior only; endpoints are 0 and n).
        global_boundaries: set[int] = set()
        for s, _e, est in windows:
            for b in est.map_boundaries_[1:-1]:
                gb = int(s + b)
                if 0 < gb < n:
                    global_boundaries.add(gb)
        self.map_boundaries_ = [0, *sorted(global_boundaries), n]
        self.k_hat_ = len(self.map_boundaries_) - 1

        # Build the global map_curve by overlap-averaging per-window
        # piecewise-constant predictions.
        accum = np.zeros(n, dtype=float)
        counts = np.zeros(n, dtype=float)
        for s, e, est in windows:
            curve = np.asarray(est.map_curve_, dtype=float)
            accum[s:e] += curve
            counts[s:e] += 1.0
        self.map_curve_ = accum / np.maximum(counts, 1.0)

        # Stitch boundary marginals: each interior index sits between two
        # candidate boundaries on the global axis. Per-window marginals
        # are length (window - 1), keyed by the interior index s+1, ...,
        # s+window-1. Average overlapping marginals.
        bm_accum = np.zeros(n - 1, dtype=float)
        bm_counts = np.zeros(n - 1, dtype=float)
        for s, e, est in windows:
            local_bm = np.asarray(est.boundary_marginals_, dtype=float)
            # local_bm has length (e - s - 1) and indexes interior
            # boundaries s+1, ..., e-1 on the global axis.
            for li, gi in enumerate(range(s + 1, e)):
                if gi - 1 < n - 1:
                    bm_accum[gi - 1] += float(local_bm[li])
                    bm_counts[gi - 1] += 1.0
        self.boundary_marginals_ = bm_accum / np.maximum(bm_counts, 1.0)

        # Approximate log-evidence (sum across windows).
        self.log_evidence_ = float(sum(float(est.log_evidence_) for _, _, est in windows))
        return self

    # -------------------------------------------------------------- predict
    def predict(
        self,
        X: ArrayLike,
        *,
        extrapolation: str | ExtrapolationPolicy = ExtrapolationPolicy.ERROR,
    ) -> FloatArray:
        require_fitted(self, ["map_curve_"])
        X_arr = np.asarray(X, dtype=float)
        x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
        positions = _assign_training_positions(self.x_design_, x_new, extrapolation)
        x_eval = self.x_design_[positions]
        _record_prediction_policy(self, extrapolation)
        # Reuse per-window predictions: assign each query x to the window
        # whose interior contains it. Where windows overlap, average their
        # predictions (matches map_curve semantics on the training axis).
        out = np.zeros(x_new.size, dtype=float)
        counts = np.zeros(x_new.size, dtype=float)
        for _s, _e, est in self.windows_:
            x_design_w = est.x_design_
            lo, hi = float(x_design_w[0]), float(x_design_w[-1])
            mask = (x_eval >= lo) & (x_eval <= hi)
            if not mask.any():
                continue
            preds = est.predict(x_eval[mask])
            out[mask] += preds
            counts[mask] += 1.0
        # Anywhere no window covers x_new, fall back to the nearest
        # window's prediction.
        uncov = counts == 0
        if uncov.any():
            for idx in np.flatnonzero(uncov):
                xv = float(x_eval[idx])
                _, _, est = min(
                    self.windows_,
                    key=lambda w: min(
                        abs(xv - float(w[2].x_design_[0])), abs(xv - float(w[2].x_design_[-1]))
                    ),
                )
                out[idx] = float(est.predict(np.array([xv]), extrapolation="clip")[0])
                counts[idx] = 1.0
        return out / np.maximum(counts, 1.0)

    def get_map_segmentation(self) -> tuple[int, list[int], FloatArray]:
        """Mirror ``BayesBreakSegmenter.get_map_segmentation`` on the
        stitched output. Segment means are read off ``map_curve_`` at the
        midpoint of each MAP segment.
        """
        require_fitted(self, ["map_boundaries_", "map_curve_"])
        means: list[float] = []
        for a, b in zip(self.map_boundaries_[:-1], self.map_boundaries_[1:], strict=False):
            mid = (int(a) + int(b)) // 2
            means.append(float(self.map_curve_[min(max(mid, 0), self.n_ - 1)]))
        return self.k_hat_, list(self.map_boundaries_), np.asarray(means, dtype=float)
