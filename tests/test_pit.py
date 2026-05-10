"""PIT residuals (§``prediction-diagnostics``) — uniformity under correct calibration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from bayesbreak import BayesBreakBernoulli, BayesBreakGaussian, BayesBreakPoisson
from bayesbreak.prediction import pit_residuals


def test_pit_uniform_for_in_sample_gaussian():
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(loc=0.0, scale=1.0, size=n)
    est = BayesBreakGaussian(k_max=4).fit(np.arange(n).reshape(-1, 1), y)
    pits = pit_residuals(est, np.arange(n).reshape(-1, 1), y)
    assert pits.shape == (n,)
    assert np.all((pits >= 0.0) & (pits <= 1.0))
    p_value = float(kstest(pits, "uniform").pvalue)
    assert p_value > 0.01


def test_pit_unsupported_family_raises():
    """Unsupported families surface a clean NotImplementedError."""

    rng = np.random.default_rng(0)
    n = 30
    # BetaObs has no closed-form CDF in our PIT helper.
    from bayesbreak import BayesBreakBetaObs

    y = rng.beta(20.0, 20.0, size=n)
    est = BayesBreakBetaObs(k_max=2).fit(np.arange(n).reshape(-1, 1), y)
    with pytest.raises(NotImplementedError):
        pit_residuals(est, np.arange(n).reshape(-1, 1), y)


def test_pit_in_unit_interval_for_discrete_families():
    """Bernoulli/Poisson PITs use the randomised PIT and stay in [0, 1]."""

    np.random.seed(0)
    rng = np.random.default_rng(0)
    n = 100
    y_b = rng.binomial(1, 0.4, size=n).astype(float)
    est_b = BayesBreakBernoulli(k_max=2).fit(np.arange(n).reshape(-1, 1), y_b)
    pits_b = pit_residuals(est_b, np.arange(n).reshape(-1, 1), y_b)
    assert np.all((pits_b >= 0) & (pits_b <= 1))

    y_p = rng.poisson(2.5, size=n).astype(float)
    est_p = BayesBreakPoisson(k_max=2).fit(np.arange(n).reshape(-1, 1), y_p)
    pits_p = pit_residuals(est_p, np.arange(n).reshape(-1, 1), y_p)
    assert np.all((pits_p >= 0) & (pits_p <= 1))
