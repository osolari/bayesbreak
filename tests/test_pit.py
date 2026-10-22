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


def test_pit_beta_obs_is_calibrated_out_of_sample():
    """BetaObs PIT via quadrature on the posterior over mu is uniform on
    fresh well-specified samples with matching per-observation precision."""

    from bayesbreak import BayesBreakBetaObs

    rng = np.random.default_rng(0)
    n_train = 400
    n_test = 400
    phi = 50.0
    mu_true = 0.3
    y_train = rng.beta(phi * mu_true, phi * (1.0 - mu_true), size=n_train)
    y_test = rng.beta(phi * mu_true, phi * (1.0 - mu_true), size=n_test)
    est = BayesBreakBetaObs(k_max=3, phi=phi).fit(
        np.arange(n_train).reshape(-1, 1), y_train
    )
    # Score on held-out points that lie inside the training index support.
    X_test = np.linspace(0.0, float(n_train - 1), n_test).reshape(-1, 1)
    pits = pit_residuals(
        est,
        X_test,
        y_test,
        sample_weight=np.full(n_test, phi),
    )
    assert pits.shape == (n_test,)
    assert np.all((pits >= 0.0) & (pits <= 1.0))
    p_value = float(kstest(pits, "uniform").pvalue)
    assert p_value > 0.01


def test_pit_beta_obs_respects_per_observation_phi():
    """Per-observation phi is honoured; predictive CDF changes with phi_new."""

    from bayesbreak import BayesBreakBetaObs

    rng = np.random.default_rng(1)
    n = 60
    mu_true = 0.4
    phi = np.linspace(20.0, 200.0, n)
    y = rng.beta(phi * mu_true, phi * (1.0 - mu_true), size=n)
    est = BayesBreakBetaObs(k_max=2, phi=phi).fit(np.arange(n).reshape(-1, 1), y)

    X = np.arange(n).reshape(-1, 1)
    pits_low = pit_residuals(est, X, y, sample_weight=np.full(n, 10.0))
    pits_high = pit_residuals(est, X, y, sample_weight=np.full(n, 500.0))
    # PITs at low precision (wide predictive) versus high precision (narrow
    # predictive) cannot be identical for the same y.
    assert np.any(np.abs(pits_low - pits_high) > 1e-6)


def test_pit_unsupported_family_raises():
    """Families with no CDF path still surface NotImplementedError."""

    from bayesbreak import BayesBreakLogisticNormal

    rng = np.random.default_rng(0)
    n = 30
    y = rng.binomial(1, 0.5, size=n).astype(float)
    est = BayesBreakLogisticNormal(k_max=2, approx="laplace").fit(
        np.arange(n).reshape(-1, 1), y
    )
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
