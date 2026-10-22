from __future__ import annotations

import math

import numpy as np
import pytest

from bayesbreak import BayesBreakBetaObs
from bayesbreak.base import BayesBreakSegmenter
from bayesbreak.prediction import posterior_predictive_logpdf
from bayesbreak.utils import gammaln, logsumexp


def _fit(quadrature_points: int = 96) -> BayesBreakBetaObs:
    y = np.array([0.18, 0.22, 0.35, 0.41, 0.56, 0.62])
    phi = np.array([12.0, 18.0, 15.0, 24.0, 20.0, 30.0])
    return BayesBreakBetaObs(
        k_max=1,
        estimate_hyper=False,
        alpha=2.5,
        beta=3.5,
        phi=phi,
        quadrature_points=quadrature_points,
    ).fit(np.arange(y.size).reshape(-1, 1), y)


def _reference_log_predictive(
    estimator: BayesBreakBetaObs,
    values: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(512)
    mu = 0.5 * (nodes + 1.0)
    log_weights = np.log(0.5 * weights)
    alpha0 = estimator.hyper_["alpha"]
    beta0 = estimator.hyper_["beta"]
    log_prior = (
        (alpha0 - 1.0) * np.log(mu)
        + (beta0 - 1.0) * np.log1p(-mu)
        - (math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0))
    )
    train = estimator._y_train_
    phi_train = estimator._phi_arr_
    log_train = np.zeros(mu.size)
    for observed, phi_value in zip(train, phi_train, strict=True):
        alpha = phi_value * mu
        beta = phi_value * (1.0 - mu)
        log_train += (
            math.lgamma(phi_value)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1.0) * math.log(observed)
            + (beta - 1.0) * math.log1p(-observed)
        )
    log_denominator = float(logsumexp(log_weights + log_prior + log_train))
    output = np.empty(values.size)
    for index, (observed, phi_value) in enumerate(zip(values, precision, strict=True)):
        alpha = phi_value * mu
        beta = phi_value * (1.0 - mu)
        log_new = (
            math.lgamma(phi_value)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1.0) * math.log(observed)
            + (beta - 1.0) * math.log1p(-observed)
        )
        output[index] = float(logsumexp(log_weights + log_prior + log_train + log_new))
        output[index] -= log_denominator
    return output


def test_beta_obs_overrides_gaussian_fallback() -> None:
    assert (
        BayesBreakBetaObs.posterior_predictive_logpdf_block
        is not BayesBreakSegmenter.posterior_predictive_logpdf_block
    )


def test_predictive_matches_high_order_numerical_reference() -> None:
    estimator = _fit()
    values = np.array([0.12, 0.48, 0.87])
    precision = np.array([10.0, 25.0, 40.0])
    observed = estimator.posterior_predictive_logpdf_block(
        a=0,
        b=estimator.n_,
        y_new=values,
        w_new=precision,
    )
    expected = _reference_log_predictive(estimator, values, precision)
    assert observed == pytest.approx(expected, abs=2e-6)


def test_predictive_density_normalizes_for_fixed_precision() -> None:
    estimator = _fit(128)
    nodes, weights = np.polynomial.legendre.leggauss(400)
    values = 0.5 * (nodes + 1.0)
    quadrature_weights = 0.5 * weights
    log_density = estimator.posterior_predictive_logpdf_block(
        a=0,
        b=estimator.n_,
        y_new=values,
        w_new=np.full(values.size, 20.0),
    )
    integral = float(np.sum(quadrature_weights * np.exp(log_density)))
    assert integral == pytest.approx(1.0, abs=2e-6)


def test_support_and_precision_edges_are_explicit() -> None:
    estimator = _fit()
    scores = estimator.posterior_predictive_logpdf_block(
        a=0,
        b=estimator.n_,
        y_new=np.array([0.0, 1e-12, 1.0 - 1e-12, 1.0, -0.1, 1.1]),
        w_new=np.full(6, 15.0),
    )
    assert scores[[0, 3, 4, 5]].tolist() == [-np.inf] * 4
    assert np.all(np.isfinite(scores[[1, 2]]))
    with pytest.raises(ValueError, match="precision"):
        estimator.posterior_predictive_logpdf_block(
            a=0,
            b=estimator.n_,
            y_new=np.array([0.5]),
            w_new=np.array([0.0]),
        )


def test_prediction_router_passes_family_precision() -> None:
    estimator = _fit()
    X = np.array([[1.0], [2.0], [3.0]])
    values = np.array([0.25, 0.45, 0.65])
    precision = np.array([8.0, 16.0, 32.0])
    routed = posterior_predictive_logpdf(
        estimator,
        X,
        values,
        sample_weight=precision,
        per_sample=True,
    )
    direct = estimator.posterior_predictive_logpdf_block(
        a=0,
        b=estimator.n_,
        y_new=values,
        w_new=precision,
    )
    assert routed == pytest.approx(direct)


def test_training_power_weights_are_separate_from_prediction_precision() -> None:
    y = np.array([0.2, 0.25, 0.7, 0.75])
    phi = np.full(y.size, 20.0)
    kwargs = {
        "k_max": 1,
        "estimate_hyper": False,
        "alpha": 2.0,
        "beta": 2.0,
        "phi": phi,
        "quadrature_points": 96,
    }
    unweighted = BayesBreakBetaObs(**kwargs).fit(np.arange(y.size), y)
    weighted = BayesBreakBetaObs(**kwargs).fit(
        np.arange(y.size),
        y,
        sample_weight=np.array([8.0, 8.0, 1.0, 1.0]),
    )
    value = np.array([0.3])
    precision = np.array([30.0])
    score_unweighted = unweighted.posterior_predictive_logpdf_block(
        a=0, b=y.size, y_new=value, w_new=precision
    )
    score_weighted = weighted.posterior_predictive_logpdf_block(
        a=0, b=y.size, y_new=value, w_new=precision
    )
    assert score_weighted != pytest.approx(score_unweighted)
