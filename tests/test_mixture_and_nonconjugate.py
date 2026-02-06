import numpy as np


def _make_gaussian_group_data(
    rng: np.random.Generator, n: int, means: list[float], cps: list[int], sigma: float = 0.5
) -> np.ndarray:
    """Piecewise-constant Gaussian data with changepoints cps (interior indices)."""
    boundaries = [0, *cps, n]
    y = np.empty(n, dtype=float)
    for (a, b), m in zip(zip(boundaries[:-1], boundaries[1:], strict=False), means, strict=False):
        y[a:b] = rng.normal(loc=m, scale=sigma, size=b - a)
    return y


def test_logistic_normal_smoke():
    from bayesbreak.families import BayesBreakLogisticNormal

    rng = np.random.default_rng(0)
    n = 80
    p = np.r_[np.full(30, 0.1), np.full(25, 0.8), np.full(25, 0.2)]
    y = rng.binomial(1, p).astype(float)

    for approx in ("laplace", "jj", "pg-vb", "ep"):
        m = BayesBreakLogisticNormal(k_max=10, approx=approx, regression_curve="fixed_k")
        m.fit(y)
        assert m.get_segment_count() >= 1
        bp = m.get_boundary_posteriors()
        assert bp.shape == (n - 1,)
        curve = m.get_regression_curve()
        assert curve is not None
        assert curve.shape == (n,)
        # probabilities are within [0,1]
        assert np.all((curve >= -1e-6) & (curve <= 1.0 + 1e-6))


def test_beta_obs_smoke():
    from bayesbreak.families import BayesBreakBetaObs

    rng = np.random.default_rng(0)
    n = 60
    # two regimes with different means
    mu = np.r_[np.full(30, 0.2), np.full(30, 0.75)]
    phi = 30.0
    # sample Beta via Gamma ratio
    a = mu * phi
    b = (1 - mu) * phi
    y = rng.beta(a, b)

    m = BayesBreakBetaObs(k_max=10, phi=phi, quad_points=24)
    m.fit(y)
    assert m.get_segment_count() >= 1
    assert m.get_boundary_posteriors().shape == (n - 1,)
    pc = m.predict()
    assert pc.shape == (n,)
    assert np.all((pc >= 0.0) & (pc <= 1.0))


def test_mixture_recovers_two_groups_gaussian():
    from sklearn.metrics import adjusted_rand_score

    from bayesbreak.families import BayesBreakGaussian
    from bayesbreak.mixture import BayesBreakMixture

    rng = np.random.default_rng(1)
    S = 20
    n = 120
    # group 0 and group 1 differ mainly by changepoint locations
    cps0, means0 = [40, 80], [0.0, 2.0, 0.0]
    cps1, means1 = [30, 70, 100], [1.0, -1.0, 1.0, -1.0]
    y_list = []
    z_true = []
    for s in range(S):
        if s < S // 2:
            y_list.append(_make_gaussian_group_data(rng, n, means0, cps0, sigma=0.6))
            z_true.append(0)
        else:
            y_list.append(_make_gaussian_group_data(rng, n, means1, cps1, sigma=0.6))
            z_true.append(1)
    z_true = np.asarray(z_true, dtype=int)

    # Mixture identification requires a sensible upper bound on the number of
    # segments. If k_max is set extremely large, multiple groups can explain all
    # sequences via a near-union segmentation.
    base = BayesBreakGaussian(k_max=5, regression_curve="none")
    mix = BayesBreakMixture(
        base_estimator=base,
        n_groups=2,
        k_max=5,
        max_iter=20,
        tol=1e-4,
        random_state=0,
    )
    mix.fit(y_list)

    z_hat = mix.predict(y_list)
    ari = adjusted_rand_score(z_true, z_hat)
    assert ari > 0.8
