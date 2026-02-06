import numpy as np


def _allclose(a, b, rtol=1e-6, atol=1e-6) -> bool:
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol))


def test_sample_weight_gaussian_posterior_mean_whole_segment():
    """With k_max=1 the fitted curve must equal the closed-form posterior mean.

    The test uses fixed hyperparameters (estimate_hyper=False) so the posterior mean
    has an analytical expression.
    """
    from bayesbreak.families import BayesBreakGaussian

    y = np.array([1.0, 3.0, 2.0], dtype=float)
    w = np.array([2.0, 1.0, 0.5], dtype=float)

    nu, rho2, sigma2 = 0.0, 1.0, 1.0
    # posterior mean: (rho2 * sum(w*y) + sigma2*nu) / (rho2*sum(w) + sigma2)
    W = float(np.sum(w))
    Sy = float(np.sum(w * y))
    expected = (rho2 * Sy + sigma2 * nu) / (rho2 * W + sigma2)

    m = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2)
    m.fit(y, sample_weight=w)
    assert _allclose(m.predict(), np.full_like(y, expected))


def test_sample_weight_poisson_posterior_mean_whole_segment():
    from bayesbreak.families import BayesBreakPoisson

    y = np.array([2.0, 0.0, 1.0], dtype=float)
    w = np.array([1.0, 3.0, 2.0], dtype=float)

    alpha, beta = 2.0, 1.0
    S = float(np.sum(w * y))
    W = float(np.sum(w))
    expected = (alpha + S) / (beta + W)

    m = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=alpha, beta=beta)
    m.fit(y, sample_weight=w)
    assert _allclose(m.predict(), np.full_like(y, expected))


def test_sample_weight_binomial_posterior_mean_whole_segment():
    from bayesbreak.families import BayesBreakBinomial

    y = np.array([3.0, 1.0, 4.0], dtype=float)
    n_trials = 10
    w = np.array([1.0, 2.0, 1.0], dtype=float)

    alpha, beta = 1.5, 2.5
    S = float(np.sum(w * y))
    N = float(np.sum(w * n_trials))
    expected = (alpha + S) / (alpha + beta + N)

    m = BayesBreakBinomial(
        k_max=1,
        estimate_hyper=False,
        n_trials=n_trials,
        alpha=alpha,
        beta=beta,
    )
    m.fit(y, sample_weight=w)
    assert _allclose(m.predict(), np.full_like(y, expected))


def test_sample_weight_beta_fractional_posterior_mean_whole_segment():
    from bayesbreak.families import BayesBreakBeta

    y = np.array([0.2, 0.8, 0.6], dtype=float)
    w = np.array([1.0, 0.5, 2.0], dtype=float)
    kappa = 20.0

    alpha, beta = 1.0, 1.0
    S = float(np.sum(w * (kappa * y)))
    N = float(np.sum(w) * kappa)
    expected = (alpha + S) / (alpha + beta + N)

    m = BayesBreakBeta(
        k_max=1,
        estimate_hyper=False,
        concentration=kappa,
        alpha=alpha,
        beta=beta,
    )
    m.fit(y, sample_weight=w)
    assert _allclose(m.predict(), np.full_like(y, expected))


def test_bernoulli_family_beta_bernoulli():
    from bayesbreak.families import BayesBreakBernoulli

    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    w = np.array([1.0, 2.0, 1.0, 0.5], dtype=float)

    alpha, beta = 2.0, 3.0
    S = float(np.sum(w * y))
    W = float(np.sum(w))
    expected = (alpha + S) / (alpha + beta + W)

    m = BayesBreakBernoulli(k_max=1, estimate_hyper=False, alpha=alpha, beta=beta)
    m.fit(y, sample_weight=w)
    assert _allclose(m.predict(), np.full_like(y, expected))


def test_group_membership_and_map_signal():
    """Two-group toy problem: sequences differ by their global mean."""
    from bayesbreak.families import BayesBreakGaussian
    from bayesbreak.groups import BayesBreakGrouped

    rng = np.random.default_rng(0)
    n = 60

    # group A: mean ~ -2, group B: mean ~ +2
    XA = [rng.normal(-2.0, 0.4, size=n) for _ in range(4)]
    XB = [rng.normal(+2.0, 0.4, size=n) for _ in range(4)]
    X_train = XA + XB
    y_train = np.array(["A"] * len(XA) + ["B"] * len(XB), dtype=object)

    base = BayesBreakGaussian(k_max=8, estimate_hyper=True, regression_curve="none")
    clf = BayesBreakGrouped(base_estimator=base)
    clf.fit(X_train, y_train)

    x_test = rng.normal(+2.0, 0.4, size=n)
    pred = clf.predict([x_test])[0]
    assert pred == "B"

    curve = clf.map_signal([x_test])
    assert isinstance(curve, list) and curve[0].shape == (n,)


def test_multivariate_wrapper_shared_boundaries_smoke():
    from bayesbreak.families import BayesBreakGaussian
    from bayesbreak.multivariate import BayesBreakMultivariate

    rng = np.random.default_rng(1)
    n = 80
    # two correlated channels with shared changepoint at n//2
    y1 = np.r_[rng.normal(0.0, 0.2, size=n // 2), rng.normal(2.0, 0.2, size=n - n // 2)]
    y2 = np.r_[rng.normal(0.0, 0.2, size=n // 2), rng.normal(2.0, 0.2, size=n - n // 2)]
    Y = np.vstack([y1, y2]).T

    base = BayesBreakGaussian(k_max=10, estimate_hyper=True, regression_curve="none")
    mv = BayesBreakMultivariate(base_estimator=base, combine="shared")
    mv.fit(Y)

    assert mv.predict().shape == Y.shape
    bds = mv.get_boundaries()
    assert bds[0] == 0 and bds[-1] == n
