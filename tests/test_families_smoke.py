from __future__ import annotations

import numpy as np

from bayesbreak import (
    BayesBreakBeta,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakPoisson,
)


def _common_assertions(model, n: int) -> None:
    yhat = model.predict()
    assert yhat.shape == (n,)

    bnds = model.get_boundaries()
    assert bnds[0] == 0
    assert bnds[-1] == n
    assert all(bnds[i] < bnds[i + 1] for i in range(len(bnds) - 1))

    d1 = model.get_boundary_posteriors()
    assert d1.shape == (n - 1,)
    assert np.all(d1 >= -1e-12)
    assert np.all(d1 <= 1.0 + 1e-12)

    s = model.score()
    assert np.isfinite(s)


def test_gaussian_smoke_and_regression_curve():
    rng = np.random.default_rng(0)
    n = 150
    mu = np.r_[np.zeros(50), np.ones(50), -0.5 * np.ones(50)]
    y = mu + 0.25 * rng.standard_normal(n)

    m = BayesBreakGaussian(k_max=10, regression_curve="mix_k").fit(y)
    _common_assertions(m, n)

    brc = m.get_regression_curve()
    assert brc is not None
    assert brc.shape == (n,)
    assert np.all(np.isfinite(brc))


def test_poisson_smoke():
    rng = np.random.default_rng(1)
    n = 120
    lam = np.r_[2.0 * np.ones(40), 10.0 * np.ones(40), 4.0 * np.ones(40)]
    y = rng.poisson(lam)

    m = BayesBreakPoisson(k_max=10).fit(y)
    _common_assertions(m, n)


def test_binomial_smoke_with_trials():
    rng = np.random.default_rng(2)
    n = 100
    n_trials = rng.integers(20, 60, size=n)
    p = np.r_[0.1 * np.ones(50), 0.7 * np.ones(50)]
    y = rng.binomial(n_trials, p)

    m = BayesBreakBinomial(k_max=8, n_trials=n_trials).fit(y)
    _common_assertions(m, n)


def test_beta_smoke_fractional():
    rng = np.random.default_rng(3)
    n = 80
    # generate values in (0, 1)
    y = np.r_[rng.beta(2, 12, size=40), rng.beta(8, 3, size=40)]

    m = BayesBreakBeta(k_max=8, concentration=50.0).fit(y)
    _common_assertions(m, n)
