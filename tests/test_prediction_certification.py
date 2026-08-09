from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import beta as beta_distribution
from scipy.stats import betabinom

from bayesbreak import (
    BayesBreakBernoulli,
    BayesBreakBeta,
    BayesBreakBinomial,
    BayesBreakLogisticNormal,
)
from bayesbreak.prediction import posterior_predictive_logpdf


def test_bernoulli_predictive_matches_beta_bernoulli_reference() -> None:
    y_train = np.array([1.0, 0.0, 1.0])
    estimator = BayesBreakBernoulli(
        k_max=1,
        estimate_hyper=False,
        alpha=2.0,
        beta=3.0,
    ).fit(np.arange(y_train.size), y_train)
    values = np.array([0.0, 1.0])
    observed = posterior_predictive_logpdf(
        estimator,
        np.array([0.0, 2.0]),
        values,
        per_sample=True,
    )
    posterior_probability = (2.0 + y_train.sum()) / (2.0 + 3.0 + y_train.size)
    expected = np.array([math.log1p(-posterior_probability), math.log(posterior_probability)])
    assert observed == pytest.approx(expected)


def test_bernoulli_predictive_rejects_nonbinary_values() -> None:
    estimator = BayesBreakBernoulli(k_max=1).fit(np.arange(4), np.array([0.0, 1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\{0, 1\}"):
        posterior_predictive_logpdf(estimator, np.array([1.0]), np.array([0.5]))


def test_binomial_predictive_matches_beta_binomial_reference() -> None:
    y_train = np.array([2.0, 4.0])
    training_trials = np.array([5.0, 5.0])
    estimator = BayesBreakBinomial(
        k_max=1,
        estimate_hyper=False,
        n_trials=training_trials,
        alpha=2.0,
        beta=3.0,
    ).fit(np.arange(y_train.size), y_train)
    successes = np.array([1.0, 3.0])
    new_trials = np.array([2.0, 4.0])
    observed = posterior_predictive_logpdf(
        estimator,
        np.array([0.0, 1.0]),
        successes,
        sample_weight=new_trials,
        per_sample=True,
    )
    expected = betabinom.logpmf(successes, new_trials, 8.0, 7.0)
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize(
    ("successes", "trials", "message"),
    [
        (np.array([1.5]), np.array([2.0]), "integer counts"),
        (np.array([3.0]), np.array([2.0]), "0 <= y_new"),
        (np.array([1.0]), np.array([2.5]), "positive integers"),
        (np.array([0.0]), np.array([0.0]), "positive integers"),
    ],
)
def test_binomial_predictive_rejects_invalid_counts(
    successes: np.ndarray,
    trials: np.ndarray,
    message: str,
) -> None:
    estimator = BayesBreakBinomial(
        k_max=1,
        estimate_hyper=False,
        n_trials=np.array([2.0, 2.0]),
        alpha=2.0,
        beta=2.0,
    ).fit(np.arange(2), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match=message):
        posterior_predictive_logpdf(
            estimator,
            np.array([0.0]),
            successes,
            sample_weight=trials,
        )


def test_fractional_beta_predictive_matches_declared_beta_density() -> None:
    y_train = np.array([0.2, 0.4])
    estimator = BayesBreakBeta(
        k_max=1,
        estimate_hyper=False,
        concentration=10.0,
        alpha=2.0,
        beta=3.0,
    ).fit(np.arange(y_train.size), y_train)
    values = np.array([0.3, 0.7])
    observed = posterior_predictive_logpdf(
        estimator,
        np.array([0.0, 1.0]),
        values,
        per_sample=True,
    )
    expected = beta_distribution.logpdf(values, 8.0, 17.0)
    assert observed == pytest.approx(expected)


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_fractional_beta_predictive_rejects_outside_open_support(value: float) -> None:
    estimator = BayesBreakBeta(k_max=1).fit(np.arange(3), np.array([0.2, 0.4, 0.6]))
    with pytest.raises(ValueError, match="strictly in"):
        estimator.posterior_predictive_logpdf_block(
            a=0,
            b=3,
            y_new=np.array([value]),
            w_new=np.ones(1),
        )


def test_unsupported_logistic_normal_predictive_fails_explicitly() -> None:
    estimator = BayesBreakLogisticNormal(k_max=2, approx="quadrature").fit(
        np.arange(8),
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    with pytest.raises(NotImplementedError, match="does not implement"):
        posterior_predictive_logpdf(
            estimator,
            np.array([2.0]),
            np.array([1.0]),
        )
