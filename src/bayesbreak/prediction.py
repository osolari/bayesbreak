r"""Posterior-predictive scoring for fitted BayesBreak estimators (§``prediction``).

For a fitted segmenter :math:`\mathcal{M}` with MAP boundaries
:math:`(t_0, \dots, t_k)` and per-segment posterior hyperparameters
:math:`(\alpha_B, \beta_B)`, the posterior-predictive log-density of new
data is

.. math::
    \log p(y^{\mathrm{new}} \mid \mathcal{M}, t) =
      \sum_{B} \big[ H^{\mathrm{new}}_B + \log Z(\alpha_B + S^{\mathrm{new}}_B,
                                                \beta_B + W^{\mathrm{new}}_B)
                                            - \log Z(\alpha_B, \beta_B) \big].

Each family implements the block-level conjugate predictive as
``posterior_predictive_logpdf_block``; this module routes new data through
the fitted MAP segmentation and aggregates the scores.

It also implements the report's two prediction tasks:

- **(P1) Group membership** via :func:`predict_group` — Case A (pointwise),
  Case B (set-valued / multivariate units), Case C (vector-valued response,
  factorised EF).
- **(P2) Signal evaluation** via :func:`predict_map_signal`.

Diagnostics: cumulative held-out log-likelihood (HLL) traces and PIT
residuals (closed-CDF families only) for §``prediction-diagnostics``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import logsumexp
from .validation import check_sample_weight

if TYPE_CHECKING:
    from .base import BayesBreakSegmenter


FloatArray = NDArray[np.floating]


# -----------------------------------------------------------------------------
# Set-valued / multivariate units (Case B) container
# -----------------------------------------------------------------------------


@dataclass
class Unit:
    """A set-valued observation unit (Case B in §``prediction-inputs``).

    Attributes
    ----------
    interval : tuple[float, float]
        Outer interval :math:`[a_u, b_u]` of the unit (informational only).
    points : ndarray of shape (R_u,)
        Internal design points :math:`(x_{u r})` with :math:`a_u < x_{u r} \\le b_u`.
    values : ndarray of shape (R_u,)
        Internal observations :math:`(y_{u r})`.
    weights : ndarray of shape (R_u,) or None
        Optional per-point weights / exposures.
    metadata : dict
    """

    interval: tuple[float, float]
    points: FloatArray
    values: FloatArray
    weights: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).ravel()
        self.values = np.asarray(self.values, dtype=float).ravel()
        if self.points.shape != self.values.shape:
            raise ValueError("Unit.points and Unit.values must share the same length.")
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float).ravel()
            if self.weights.shape != self.values.shape:
                raise ValueError("Unit.weights must have the same length as values.")


# -----------------------------------------------------------------------------
# Pointwise (Case A) posterior-predictive
# -----------------------------------------------------------------------------


def _assign_to_map_blocks(x_design: FloatArray, x_new: FloatArray) -> NDArray[np.intp]:
    """Map each new ``x`` to the index of its nearest training design point.

    Implements the exported segment-assignment map of
    Definition ``def:segment-assignment-map``: under the exported MAP
    boundary vector, this routine carries a new design point to the
    training index of the segment that contains it.
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
    """Pointwise posterior-predictive log-density under the fitted MAP segmentation.

    Implements the Case A (pointwise) branch of the prediction interface
    of Definition ``def:prediction-cases``. Per-sample independence under
    the exported segmentation is the content of
    Assumption ``ass:prediction-independence``.
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

    training_pos = _assign_to_map_blocks(estimator.x_design_, x_new)
    boundaries = np.asarray(estimator.map_boundaries_, dtype=int)
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
    """Cumulative held-out log-predictive — eq. §``prediction-diagnostics``."""

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


# -----------------------------------------------------------------------------
# Set-valued / multivariate units (Case B)
# -----------------------------------------------------------------------------


def unit_log_likelihood(
    estimator: BayesBreakSegmenter,
    unit: Unit,
) -> float:
    """``log p(unit | M_g)`` — eq. ``unit-lik``.

    The unit's internal points are routed through the estimator's exported
    MAP segmentation; the per-block contributions sum to the unit log-lik.
    """

    if unit.points.size == 0:
        return 0.0
    weights = unit.weights if unit.weights is not None else np.ones(unit.points.size)
    total = posterior_predictive_logpdf(
        estimator, unit.points, unit.values, sample_weight=weights, per_sample=False
    )
    assert isinstance(total, float)
    return float(total)


def unit_responsibilities(
    estimators: Sequence[BayesBreakSegmenter],
    units: Sequence[Unit],
    prior: ArrayLike | None = None,
) -> FloatArray:
    """``r_{u g} = P(g | unit_u)`` — eq. ``unit-resp``.

    Returns shape ``(U, G)``.
    """

    G = len(estimators)
    U = len(units)
    if prior is None:
        log_pi = np.full(G, -np.log(G), dtype=float)
    else:
        pri = np.asarray(prior, dtype=float)
        if pri.shape != (G,):
            raise ValueError(f"prior must have shape ({G},); got {pri.shape}.")
        log_pi = np.log(np.maximum(pri / pri.sum(), 1e-300))

    log_u = np.zeros((U, G), dtype=float)
    for u, unit in enumerate(units):
        for g, est in enumerate(estimators):
            log_u[u, g] = log_pi[g] + unit_log_likelihood(est, unit)
    log_norm = logsumexp(log_u, axis=1, keepdims=True)
    return np.exp(log_u - log_norm)


# -----------------------------------------------------------------------------
# Group membership / signal-prediction routines (Algorithms ``predict-group``,
# ``predict-map``)
# -----------------------------------------------------------------------------


def predict_group(
    estimators: Sequence[BayesBreakSegmenter],
    new_data: ArrayLike | Sequence[Unit],
    y_new: ArrayLike | None = None,
    *,
    sample_weight: ArrayLike | None = None,
    prior: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute ``(ell_g, P(g | new))`` under each fitted group estimator.

    Two input shapes are supported:

    - Case A (pointwise): ``new_data`` is the design ``X^{new}`` (1-D / 2-D
      array); ``y_new`` is the response.
    - Case B (set-valued units): ``new_data`` is a sequence of :class:`Unit`
      and ``y_new`` is ignored.
    """

    G = len(estimators)
    if isinstance(new_data, Sequence) and not isinstance(new_data, np.ndarray):
        if all(isinstance(u, Unit) for u in new_data):
            ell_g = np.zeros(G, dtype=float)
            for g, est in enumerate(estimators):
                ell_g[g] = float(sum((unit_log_likelihood(est, u) for u in new_data), 0.0))
        else:
            raise TypeError("new_data must be array-like or a sequence of Unit objects.")
    else:
        if y_new is None:
            raise ValueError("Case A (array-like new_data) requires y_new.")
        ell_g = np.zeros(G, dtype=float)
        for g, est in enumerate(estimators):
            ell_g[g] = float(
                posterior_predictive_logpdf(est, new_data, y_new, sample_weight=sample_weight)
            )

    if prior is None:
        log_pi = np.full(G, -np.log(G), dtype=float)
    else:
        pri = np.asarray(prior, dtype=float)
        if pri.shape != (G,):
            raise ValueError(f"prior must have shape ({G},); got {pri.shape}.")
        log_pi = np.log(np.maximum(pri / pri.sum(), 1e-300))

    log_post = ell_g + log_pi
    log_post -= float(logsumexp(log_post))
    return ell_g, np.exp(log_post)


def predict_map_signal(
    estimator: BayesBreakSegmenter,
    X_star: ArrayLike,
    *,
    mode: str = "MAP",
) -> FloatArray:
    """Algorithm ``predict-map`` — return MAP / Bayes signal at query points."""

    mode = mode.lower()
    if mode == "map":
        return estimator.predict(X_star, mode="map")
    if mode == "bayes":
        return estimator.predict(X_star, mode="bayes")
    raise ValueError("mode must be 'MAP' or 'Bayes'.")


# -----------------------------------------------------------------------------
# Diagnostics: PIT residuals (§``prediction-diagnostics``)
# -----------------------------------------------------------------------------


def pit_residuals(
    estimator: BayesBreakSegmenter,
    X_new: ArrayLike,
    y_new: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
) -> FloatArray:
    """Probability-integral transform residuals for closed-CDF families.

    Under correct calibration the returned values are Uniform(0, 1).
    Implemented for Gaussian, Bernoulli, Beta, and Binomial families; other
    families raise :class:`NotImplementedError`. The CDF is evaluated under
    the per-point posterior-predictive of the segment containing the query.
    """

    from .base import BayesBreakSegmenter
    from .families import (
        BayesBreakBernoulli,
        BayesBreakBeta,
        BayesBreakBinomial,
        BayesBreakGaussian,
        BayesBreakPoisson,
    )

    if not isinstance(estimator, BayesBreakSegmenter):
        raise TypeError("pit_residuals requires a BayesBreakSegmenter.")

    X_arr = np.asarray(X_new, dtype=float)
    x_new = X_arr[:, 0] if X_arr.ndim == 2 else X_arr.ravel()
    y_arr = np.asarray(y_new, dtype=float)
    m = int(x_new.size)
    w_arr = check_sample_weight(sample_weight, m)

    training_pos = _assign_to_map_blocks(estimator.x_design_, x_new)
    boundaries = np.asarray(estimator.map_boundaries_, dtype=int)
    seg_index = np.searchsorted(boundaries, training_pos, side="right") - 1
    seg_index = np.clip(seg_index, 0, len(boundaries) - 2)

    pits = np.zeros(m, dtype=float)
    for s in range(len(boundaries) - 1):
        mask = seg_index == s
        if not np.any(mask):
            continue
        a, b = int(boundaries[s]), int(boundaries[s + 1])
        pits[mask] = _segment_pit(
            estimator,
            a,
            b,
            y_arr[mask],
            w_arr[mask],
            BayesBreakGaussian,
            BayesBreakPoisson,
            BayesBreakBernoulli,
            BayesBreakBeta,
            BayesBreakBinomial,
        )
    return pits


def _segment_pit(
    estimator: BayesBreakSegmenter,
    a: int,
    b: int,
    y: FloatArray,
    w: FloatArray,
    GaussianCls,
    PoissonCls,
    BernoulliCls,
    BetaCls,
    BinomialCls,
) -> FloatArray:
    """Per-segment PIT for the closed-CDF families enumerated above."""

    if isinstance(estimator, GaussianCls):
        from scipy.stats import norm

        nu = estimator.hyper_["nu"]
        rho2 = estimator.hyper_["rho2"]
        sigma2 = estimator.hyper_["sigma2"]
        w_train = estimator.sample_weight_[a:b]
        y_train = estimator._y_train_[a:b]
        Wseg = float(np.sum(w_train))
        Syseg = float(np.sum(w_train * y_train))
        mu_post = (rho2 * Syseg + sigma2 * nu) / (rho2 * Wseg + sigma2)
        rho2_post = (rho2 * sigma2) / (rho2 * Wseg + sigma2)
        var_pred = sigma2 / np.maximum(w, 1e-12) + rho2_post
        return norm.cdf(y, loc=mu_post, scale=np.sqrt(var_pred))

    if isinstance(estimator, PoissonCls):
        from scipy.stats import nbinom

        alpha = estimator.hyper_["alpha"]
        beta = estimator.hyper_["beta"]
        w_train = estimator.sample_weight_[a:b]
        y_train = estimator._y_train_[a:b]
        S = float(np.sum(w_train * y_train))
        W = float(np.sum(w_train))
        r = alpha + S
        # Predictive: NegBin(r, p = beta_post / (beta_post + w_new))
        p = (beta + W) / (beta + W + np.maximum(w, 1e-12))
        # Continuity correction for discrete CDF: PIT for discrete Y is not
        # uniformly Uniform(0,1); we use the standard randomized PIT.
        cdf = nbinom.cdf(y, n=r, p=p)
        cdf_minus = nbinom.cdf(y - 1, n=r, p=p)
        u = np.random.uniform(size=y.size)
        return cdf_minus + u * (cdf - cdf_minus)

    if isinstance(estimator, BernoulliCls):
        # Predictive: Bernoulli(p_hat); randomized PIT.
        alpha = estimator.hyper_["alpha"]
        beta = estimator.hyper_["beta"]
        w_train = estimator.sample_weight_[a:b]
        y_train = estimator._y_train_[a:b]
        S = float(np.sum(w_train * y_train))
        W = float(np.sum(w_train))
        p_hat = (alpha + S) / (alpha + beta + W)
        # P(Y = 0) = 1 - p_hat; P(Y = 1) = p_hat.
        u = np.random.uniform(size=y.size)
        cdf_minus = np.where(y == 0, 0.0, 1.0 - p_hat)
        cdf = np.where(y == 0, 1.0 - p_hat, 1.0)
        return cdf_minus + u * (cdf - cdf_minus)

    if isinstance(estimator, BetaCls):
        from scipy.stats import beta as beta_dist

        a0 = estimator.hyper_["alpha"]
        b0 = estimator.hyper_["beta"]
        kappa = estimator.concentration
        w_train = estimator.sample_weight_[a:b]
        y_train = estimator._y_train_[a:b]
        S = float(np.sum(w_train * (kappa * y_train)))
        W = float(np.sum(w_train) * kappa)
        a_post = a0 + S
        b_post = a0 + b0 + W - a_post
        return beta_dist.cdf(np.clip(y, 1e-12, 1 - 1e-12), a_post, b_post)

    if isinstance(estimator, BinomialCls):
        from scipy.stats import betabinom

        # Predictive: Beta-Binomial; if estimator was fit on n_trials = 1
        # this is a Bernoulli-like; otherwise we use estimator's stored n_arr.
        alpha = estimator.hyper_["alpha"]
        beta = estimator.hyper_["beta"]
        w_train = estimator.sample_weight_[a:b]
        y_train = estimator._y_train_[a:b]
        n_train = estimator._n_arr_[a:b]
        S = float(np.sum(w_train * y_train))
        N = float(np.sum(w_train * n_train))
        a_post = alpha + S
        b_post = beta + (N - S)
        # Assume n_trials = 1 for new points (matches the family's predictive).
        cdf = betabinom.cdf(y, n=1, a=a_post, b=b_post)
        cdf_minus = betabinom.cdf(y - 1, n=1, a=a_post, b=b_post)
        u = np.random.uniform(size=y.size)
        return cdf_minus + u * (cdf - cdf_minus)

    raise NotImplementedError(f"pit_residuals does not yet support {type(estimator).__name__}.")
