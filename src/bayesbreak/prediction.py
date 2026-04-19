r"""Posterior-predictive scoring for fitted BayesBreak estimators.

This module implements the prediction layer described in §8 of the report.
For a fitted segmenter :math:`\mathcal{M}` with MAP boundaries
:math:`(t_0, \dots, t_k)` and per-segment posterior hyperparameters
:math:`(\alpha_B, \beta_B)`, the posterior-predictive log-density of new data
is

.. math::
    \log p(y^{\mathrm{new}} \mid \mathcal{M}, t) =
      \sum_{B} \left[
        H^{\mathrm{new}}_B + \log Z(\alpha_B + S^{\mathrm{new}}_B,
                                     \beta_B + W^{\mathrm{new}}_B)
                           - \log Z(\alpha_B, \beta_B)
      \right],

where the sum runs over the MAP blocks :math:`B` and
:math:`(S^{\mathrm{new}}_B, W^{\mathrm{new}}_B, H^{\mathrm{new}}_B)` are the
new-data sufficient statistics on block :math:`B`.

Each family implements the block-level conjugate predictive as a
``posterior_predictive_block`` method; this module is responsible only for
routing new data through the fitted MAP segmentation and aggregating the
scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import check_sample_weight

if TYPE_CHECKING:
    from .base import BayesBreakSegmenter


FloatArray = NDArray[np.floating]


def _assign_to_map_blocks(x_design: FloatArray, x_new: FloatArray) -> NDArray[np.intp]:
    """Assign new design points to segments defined on ``x_design``.

    Points outside the training extent fall into the nearest endpoint segment.
    The returned vector has length ``x_new.size`` with values in ``{0, ..., n-1}``
    indexing into positions of ``x_design``.
    """

    order = np.argsort(x_design)
    sorted_x = x_design[order]
    positions = np.searchsorted(sorted_x, x_new, side="right") - 1
    positions = np.clip(positions, 0, len(sorted_x) - 1)
    return order[positions]


def posterior_predictive_logpdf(
    estimator: BayesBreakSegmenter,
    X: ArrayLike,
    y: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    per_sample: bool = False,
) -> float | FloatArray:
    """Log posterior-predictive density of ``y`` under the fitted MAP segmentation.

    Parameters
    ----------
    estimator : BayesBreakSegmenter
        Fitted estimator exposing ``map_boundaries_``, ``x_design_``, and the
        family-specific method ``posterior_predictive_logpdf_block``.
    X : array-like of shape (m,) or (m, p)
        New design points.
    y : array-like of shape (m,) or (m, d)
        New observations.
    sample_weight : array-like or None
        Optional exposure / precision weights.
    per_sample : bool, default False
        If ``True``, return per-observation log-densities
        (shape ``(m,)``) instead of the total.

    Returns
    -------
    float or ndarray of shape (m,)
        Sum (or per-sample vector) of log posterior-predictive densities.
    """

    from .base import BayesBreakSegmenter  # local import to avoid cycle

    if not isinstance(estimator, BayesBreakSegmenter):
        raise TypeError("posterior_predictive_logpdf requires a BayesBreakSegmenter.")

    X_arr = np.asarray(X, dtype=float)
    x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
    y_arr = np.asarray(y, dtype=float)
    m = int(x_new.size)
    if y_arr.shape[0] != m:
        raise ValueError("X and y must agree on the sample dimension.")
    w_arr = check_sample_weight(sample_weight, m)

    if estimator.map_boundaries_ is None or estimator.x_design_ is None:
        raise RuntimeError("Estimator must be fitted before scoring.")

    # Assign new points to MAP blocks via their position in the training design.
    training_pos = _assign_to_map_blocks(estimator.x_design_, x_new)
    boundaries = np.asarray(estimator.map_boundaries_, dtype=int)
    # Segment index for each new point: find which [boundaries[s], boundaries[s+1])
    # interval contains the training position.
    seg_index = np.searchsorted(boundaries, training_pos, side="right") - 1
    seg_index = np.clip(seg_index, 0, len(boundaries) - 2)

    per = np.zeros(m, dtype=float)
    for s in range(len(boundaries) - 1):
        mask = seg_index == s
        if not np.any(mask):
            continue
        a, b = int(boundaries[s]), int(boundaries[s + 1])
        per[mask] = estimator.posterior_predictive_logpdf_block(
            a=a,
            b=b,
            y_new=y_arr[mask],
            w_new=w_arr[mask],
        )

    return per if per_sample else float(np.sum(per))


def held_out_log_likelihood_trace(
    estimator: BayesBreakSegmenter,
    X_new: ArrayLike,
    y_new: ArrayLike,
    *,
    prefix_fractions: ArrayLike | None = None,
) -> FloatArray:
    """Running held-out log-likelihood for diagnostic plots (Table 0).

    Returns a vector of cumulative log posterior-predictive densities over
    the sorted new sequence at each prefix fraction.
    """

    X_arr = np.asarray(X_new, dtype=float)
    x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
    y_arr = np.asarray(y_new, dtype=float)
    order = np.argsort(x_new, kind="stable")
    x_sorted = x_new[order]
    y_sorted = y_arr[order]
    m = x_sorted.size

    per = posterior_predictive_logpdf(estimator, x_sorted, y_sorted, per_sample=True)
    assert isinstance(per, np.ndarray)
    cumulative = np.cumsum(per)

    if prefix_fractions is None:
        return cumulative

    fractions = np.asarray(prefix_fractions, dtype=float)
    idx = np.clip((fractions * m).astype(int) - 1, 0, m - 1)
    return cumulative[idx]
