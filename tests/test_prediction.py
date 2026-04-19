"""Posterior-predictive scoring correctness (§8 of the report)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from bayesbreak import BayesBreakGaussian, BayesBreakPoisson
from bayesbreak.prediction import (
    held_out_log_likelihood_trace,
    posterior_predictive_logpdf,
)


def test_gaussian_predictive_matches_closed_form():
    """Single-segment Gaussian predictive is Normal(mu_post, sigma2 / w + rho2_post)."""

    rng = np.random.default_rng(0)
    n = 40
    nu, rho2, sigma2 = 0.0, 1.0, 0.5
    y_train = rng.normal(1.2, math.sqrt(sigma2), size=n)
    est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2).fit(
        np.arange(n).reshape(-1, 1), y_train
    )

    # Posterior hyperparameters under a single block.
    mu_post = (rho2 * y_train.sum() + sigma2 * nu) / (rho2 * n + sigma2)
    rho2_post = (rho2 * sigma2) / (rho2 * n + sigma2)
    var_pred = sigma2 + rho2_post

    x_new = np.array([n // 2, n // 2 + 1])
    y_new = np.array([1.5, 0.8])
    per = posterior_predictive_logpdf(est, x_new, y_new, per_sample=True)

    expected = norm.logpdf(y_new, loc=mu_post, scale=math.sqrt(var_pred))
    assert np.allclose(per, expected, atol=1e-8)


def test_poisson_predictive_is_negative_binomial():
    """Poisson-Gamma predictive is Negative-Binomial(r=alpha_post, p=beta/(beta+1))."""

    rng = np.random.default_rng(1)
    n = 25
    alpha, beta = 2.0, 1.0
    y_train = rng.poisson(5.0, size=n)
    est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=alpha, beta=beta).fit(
        np.arange(n).reshape(-1, 1), y_train
    )

    S = int(y_train.sum())
    r = alpha + S
    p = (beta + n) / (beta + n + 1.0)

    x_new = np.array([0, 1])
    y_new = np.array([4.0, 7.0])
    per = posterior_predictive_logpdf(est, x_new, y_new, per_sample=True)

    from scipy.special import gammaln

    expected = (
        gammaln(r + y_new)
        - gammaln(y_new + 1.0)
        - gammaln(r)
        + r * math.log(p)
        + y_new * math.log(1.0 - p)
    )
    assert np.allclose(per, expected, atol=1e-8)


def test_score_method_uses_predictive():
    """score(X, y) returns mean posterior-predictive log-density."""

    rng = np.random.default_rng(2)
    n = 60
    y = rng.normal(size=n)
    est = BayesBreakGaussian(k_max=4).fit(np.arange(n).reshape(-1, 1), y)
    s = est.score(np.arange(n).reshape(-1, 1), y)

    total = posterior_predictive_logpdf(est, np.arange(n).reshape(-1, 1), y)
    assert s == pytest.approx(total / n, abs=1e-12)


def test_hll_trace_matches_prefix_sum():
    rng = np.random.default_rng(3)
    n = 30
    y = rng.normal(size=n)
    est = BayesBreakGaussian(k_max=1).fit(np.arange(n).reshape(-1, 1), y)

    trace = held_out_log_likelihood_trace(est, np.arange(n), y)
    per = posterior_predictive_logpdf(est, np.arange(n), y, per_sample=True)
    assert np.allclose(trace, np.cumsum(per), atol=1e-10)


def test_score_higher_for_correct_model():
    """Gaussian score should be higher on its training distribution than heavy-tailed noise."""

    rng = np.random.default_rng(4)
    n = 100
    y = rng.normal(size=n)
    est = BayesBreakGaussian(k_max=4).fit(np.arange(n).reshape(-1, 1), y)
    y_outlier = rng.standard_t(df=2, size=n) * 5.0
    assert est.score(np.arange(n).reshape(-1, 1), y) > est.score(
        np.arange(n).reshape(-1, 1), y_outlier
    )
