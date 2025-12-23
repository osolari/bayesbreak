from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import BayesBreakBeta, BayesBreakBinomial, BayesBreakGaussian, BayesBreakPoisson


def test_gaussian_requires_hyper_when_estimation_disabled():
    y = np.zeros(10)
    with pytest.raises(ValueError):
        BayesBreakGaussian(estimate_hyper=False, k_max=3).fit(y)


def test_gaussian_accepts_provided_hyper_when_estimation_disabled():
    y = np.zeros(10)
    m = BayesBreakGaussian(
        estimate_hyper=False, k_max=3, nu=0.0, rho2=1.0, sigma2=1.0
    ).fit(y)
    assert m.hyper_["nu"] == 0.0


def test_poisson_requires_hyper_when_estimation_disabled():
    y = np.ones(10)
    with pytest.raises(ValueError):
        BayesBreakPoisson(estimate_hyper=False, k_max=3).fit(y)


def test_binomial_requires_hyper_when_estimation_disabled():
    y = np.zeros(10)
    n_trials = np.ones(10)
    with pytest.raises(ValueError):
        BayesBreakBinomial(estimate_hyper=False, k_max=3, n_trials=n_trials).fit(y)


def test_beta_requires_hyper_when_estimation_disabled():
    y = np.full(10, 0.3)
    with pytest.raises(ValueError):
        BayesBreakBeta(estimate_hyper=False, k_max=3).fit(y)
