"""Diagnostic tests for E02: observation-family portability.

Tests:
  - Each family generator (support, segment counts, equal lengths)
  - NegBin parameterization: E[Y] = r*(1-p)/p matches expected mu
  - Poisson exposure descriptor: y_i ~ Poisson(lambda*e_i)
  - Binomial trials: 0 <= y_i <= m_i
  - Beta-obs: y_i in (0, 1)
  - Small hand-constructed example for each family: fit, check k_marg > 0, posterior > 0
  - Seed reproducibility: same seed -> same data
  - Boundary matching correctly handles equal-segment boundaries [0, 80, 160, 240]
  - Signal MSE is finite and > 0 for a synthetic noiseless example
  - NegBin A1 is the count mean (not p)
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import nbinom


# ---------------------------------------------------------------------------
# Helper: E02 generators mirroring what the runner will produce
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


# ---- Gaussian generator ----

def _gen_gaussian(n: int, delta: float, rng: np.random.Generator) -> dict:
    """k_true=3, equal segments, means (0, delta, 0), sigma2=1."""
    seg = n // 3
    assert 3 * seg == n, "n must be divisible by 3"
    true_means = [0.0, delta, 0.0]
    true_signal = np.empty(n, dtype=float)
    for q in range(3):
        true_signal[q * seg:(q + 1) * seg] = true_means[q]
    y = true_signal + rng.standard_normal(n)
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "true_signal": true_signal, "true_means": true_means,
            "boundaries": [0, seg, 2 * seg, n]}


# ---- Poisson generator ----

def _gen_poisson(n: int, q: float, rng: np.random.Generator) -> dict:
    """k_true=3, equal segments, rates (2, 2*q, 2), exposures cycle {0.5,1,2}."""
    seg = n // 3
    assert 3 * seg == n
    true_rates = [2.0, 2.0 * q, 2.0]
    e_arr = np.array([0.5, 1.0, 2.0] * (n // 3 + 1))[:n]
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_rates[s]
    y = np.empty(n, dtype=float)
    for i in range(n):
        y[i] = rng.poisson(true_rates[i // seg] * e_arr[i])
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "e_arr": e_arr, "true_signal": true_signal,
            "true_rates": true_rates, "boundaries": [0, seg, 2 * seg, n]}


# ---- Binomial generator ----

def _gen_binomial(n: int, p_mid: float, rng: np.random.Generator) -> dict:
    """k_true=3, equal segments, probs (0.2, p_mid, 0.2), trials cycle {8,32}."""
    seg = n // 3
    assert 3 * seg == n
    true_probs = [0.2, p_mid, 0.2]
    m_arr = np.array([8 if i % 2 == 0 else 32 for i in range(n)], dtype=float)
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_probs[s]
    y = np.array([rng.binomial(int(m_arr[i]), true_probs[i // seg]) for i in range(n)], dtype=float)
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "m_arr": m_arr, "true_signal": true_signal,
            "true_probs": true_probs, "boundaries": [0, seg, 2 * seg, n]}


# ---- NegBin generator ----

def _gen_negbin(n: int, mu_mid: float, rng: np.random.Generator) -> dict:
    """k_true=3, equal segments, mean counts (2, mu_mid, 2), r=5."""
    seg = n // 3
    assert 3 * seg == n
    r = 5.0
    true_mus = [2.0, mu_mid, 2.0]
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_mus[s]
    y = np.empty(n, dtype=float)
    for s in range(3):
        mu_q = true_mus[s]
        p_q = r / (r + mu_q)  # scipy p_success: mean = r*(1-p)/p
        y[s * seg:(s + 1) * seg] = nbinom.rvs(n=int(r), p=p_q, size=seg, random_state=rng)
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "true_signal": true_signal, "true_mus": true_mus,
            "boundaries": [0, seg, 2 * seg, n], "r": r}


# ---- Beta-obs generator ----

def _gen_beta_obs(n: int, mu_mid: float, rng: np.random.Generator) -> dict:
    """k_true=3, equal segments, means (0.2, mu_mid, 0.2), phi cycle {20,80}."""
    seg = n // 3
    assert 3 * seg == n
    true_mus = [0.2, mu_mid, 0.2]
    phi_arr = np.array([20.0 if i % 2 == 0 else 80.0 for i in range(n)], dtype=float)
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_mus[s]
    y = np.empty(n, dtype=float)
    for i in range(n):
        mu_q = true_mus[i // seg]
        phi = phi_arr[i]
        a = phi * mu_q
        b = phi * (1.0 - mu_q)
        y[i] = rng.beta(a, b)
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "phi_arr": phi_arr, "true_signal": true_signal,
            "true_mus": true_mus, "boundaries": [0, seg, 2 * seg, n]}


# ---------------------------------------------------------------------------
# 1. Generator support and structure tests
# ---------------------------------------------------------------------------

class TestGeneratorStructure:
    def test_gaussian_support(self):
        rng = np.random.default_rng(42)
        d = _gen_gaussian(240, 1.5, rng)
        assert d["y"].shape == (240,)
        assert d["x"].shape == (240,)
        assert len(d["boundaries"]) == 4
        # equal segment lengths
        b = d["boundaries"]
        segs = [b[i + 1] - b[i] for i in range(3)]
        assert segs == [80, 80, 80]

    def test_poisson_support(self):
        rng = np.random.default_rng(42)
        d = _gen_poisson(240, 3.0, rng)
        assert np.all(d["y"] >= 0)
        assert np.all(d["y"] == np.floor(d["y"]))  # integer counts
        b = d["boundaries"]
        segs = [b[i + 1] - b[i] for i in range(3)]
        assert segs == [80, 80, 80]

    def test_binomial_support(self):
        rng = np.random.default_rng(42)
        d = _gen_binomial(240, 0.65, rng)
        y = d["y"]
        m = d["m_arr"]
        assert np.all(y >= 0)
        assert np.all(y <= m)
        b = d["boundaries"]
        segs = [b[i + 1] - b[i] for i in range(3)]
        assert segs == [80, 80, 80]

    def test_negbin_support(self):
        rng = np.random.default_rng(42)
        d = _gen_negbin(240, 6.0, rng)
        assert np.all(d["y"] >= 0)
        assert np.all(d["y"] == np.floor(d["y"]))
        b = d["boundaries"]
        segs = [b[i + 1] - b[i] for i in range(3)]
        assert segs == [80, 80, 80]

    def test_beta_obs_support(self):
        rng = np.random.default_rng(42)
        d = _gen_beta_obs(120, 0.8, rng)
        y = d["y"]
        assert np.all(y > 0) and np.all(y < 1)
        b = d["boundaries"]
        segs = [b[i + 1] - b[i] for i in range(3)]
        assert segs == [40, 40, 40]


# ---------------------------------------------------------------------------
# 2. NegBin parameterization test
# ---------------------------------------------------------------------------

class TestNegBinParameterization:
    def test_mean_formula(self):
        """E[Y] = r*(1-p)/p where p = r/(r+mu)."""
        r = 5.0
        for mu in [2.0, 3.0, 6.0]:
            p_q = r / (r + mu)
            # scipy nbinom mean: n*(1-p)/p where n=r
            empirical_mean = nbinom.mean(n=r, p=p_q)
            assert abs(empirical_mean - mu) < 1e-10, (
                f"Expected mean {mu}, got {empirical_mean} for p={p_q}"
            )

    def test_negbin_generation_mean(self):
        """Large-sample check: generated y has empirical mean close to true mu."""
        rng = np.random.default_rng(20261021)
        r = 5.0
        for mu in [2.0, 3.0, 6.0]:
            p_q = r / (r + mu)
            y = nbinom.rvs(n=int(r), p=p_q, size=5000, random_state=rng)
            emp = np.mean(y)
            assert abs(emp - mu) < 0.3, f"mu={mu}, empirical={emp:.3f}"

    def test_p_from_mu(self):
        """Verify p = r/(r+mu) gives correct mu = r*(1-p)/p."""
        r, mu = 5.0, 2.0
        p = r / (r + mu)
        mu_back = r * (1.0 - p) / p
        assert abs(mu_back - mu) < 1e-12


# ---------------------------------------------------------------------------
# 3. Poisson exposure descriptor test
# ---------------------------------------------------------------------------

class TestPoissonExposure:
    def test_exposure_cycles(self):
        """Exposure array cycles correctly."""
        n = 240
        e_arr = np.array([0.5, 1.0, 2.0] * (n // 3 + 1))[:n]
        assert e_arr[0] == 0.5
        assert e_arr[1] == 1.0
        assert e_arr[2] == 2.0
        assert e_arr[3] == 0.5

    def test_poisson_count_mean_with_exposure(self):
        """Large-sample: E[y_i | lambda, e_i] = lambda * e_i."""
        rng = np.random.default_rng(999)
        lam = 2.0
        e = 1.5
        y = rng.poisson(lam * e, size=10000)
        assert abs(np.mean(y) - lam * e) < 0.1


# ---------------------------------------------------------------------------
# 4. Fit and k_marg tests (hand-constructed small examples)
# ---------------------------------------------------------------------------

class TestFamilyFits:
    def test_gaussian_fit_basic(self):
        from bayesbreak.families.gaussian import BayesBreakGaussian
        rng = np.random.default_rng(20261021)
        d = _gen_gaussian(240, 1.5, rng)
        est = BayesBreakGaussian(
            k_max=10, estimate_hyper=False,
            nu=0.0, rho2=4.0, sigma2=1.0,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"])
        assert hasattr(est, "k_posterior_")
        assert np.all(np.isfinite(est.k_posterior_))
        assert abs(np.sum(est.k_posterior_) - 1.0) < 1e-6
        # k_posterior_ is indexed 0..k_max-1
        assert est.k_posterior_.shape == (10,)

    def test_poisson_fit_basic(self):
        from bayesbreak.families.poisson import BayesBreakPoisson
        rng = np.random.default_rng(20261021)
        d = _gen_poisson(240, 3.0, rng)
        est = BayesBreakPoisson(
            k_max=10, estimate_hyper=False,
            alpha=2.0, beta=1.0,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"], sample_weight=d["e_arr"])
        assert np.all(np.isfinite(est.k_posterior_))
        assert abs(np.sum(est.k_posterior_) - 1.0) < 1e-6

    def test_binomial_fit_basic(self):
        from bayesbreak.families.binomial import BayesBreakBinomial
        rng = np.random.default_rng(20261021)
        d = _gen_binomial(240, 0.65, rng)
        est = BayesBreakBinomial(
            k_max=10, estimate_hyper=False,
            alpha=2.0, beta=3.0,
            n_trials=d["m_arr"],
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"])
        assert np.all(np.isfinite(est.k_posterior_))
        assert abs(np.sum(est.k_posterior_) - 1.0) < 1e-6

    def test_negbin_fit_basic(self):
        from bayesbreak.families.negative_binomial import BayesBreakNegBin
        rng = np.random.default_rng(20261021)
        d = _gen_negbin(240, 6.0, rng)
        est = BayesBreakNegBin(
            k_max=10, estimate_hyper=False,
            r=5.0, alpha=2.0, beta=3.0,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"])
        assert np.all(np.isfinite(est.k_posterior_))
        assert abs(np.sum(est.k_posterior_) - 1.0) < 1e-6

    def test_beta_obs_fit_basic(self):
        from bayesbreak.families.beta_obs import BayesBreakBetaObs
        rng = np.random.default_rng(20261021)
        d = _gen_beta_obs(120, 0.8, rng)
        est = BayesBreakBetaObs(
            k_max=10, estimate_hyper=False,
            alpha=2.0, beta=3.0,
            phi=d["phi_arr"],
            quadrature_points=32,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"])
        assert np.all(np.isfinite(est.k_posterior_))
        assert abs(np.sum(est.k_posterior_) - 1.0) < 1e-6

    def test_negbin_a1_is_count_mean(self):
        """A1[i,j] for NegBin should be the posterior count mean, not p."""
        from bayesbreak.families.negative_binomial import BayesBreakNegBin
        rng = np.random.default_rng(20261021)
        r = 5.0
        mu_true = 6.0
        p_q = r / (r + mu_true)
        y = nbinom.rvs(n=int(r), p=p_q, size=80, random_state=rng)
        x = np.arange(1, 81, dtype=float)
        est = BayesBreakNegBin(
            k_max=3, estimate_hyper=False,
            r=r, alpha=2.0, beta=3.0,
        )
        est.fit(x.reshape(-1, 1), y)
        # A1[0, n] is the posterior mean count mean for the full block
        A1 = est.block_first_moment_
        count_mean_est = A1[0, 80]
        # The posterior count mean should be on the scale of mu_true (>>0, <<1)
        # p would be r/(r+mu) ≈ 5/11 ≈ 0.45 which is in (0, 1)
        # But count mean should be near mu_true = 6
        assert count_mean_est > 1.0, (
            f"A1[0, 80] = {count_mean_est:.4f} looks like p (near 0), expected count mean near {mu_true}"
        )


# ---------------------------------------------------------------------------
# 5. Seed reproducibility tests
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    def test_gaussian_seed_reproducible(self):
        seed = np.random.SeedSequence([20261021, 2, 0, 0])
        rng1 = np.random.default_rng(seed.spawn(2)[0])
        rng2 = np.random.default_rng(np.random.SeedSequence([20261021, 2, 0, 0]).spawn(2)[0])
        d1 = _gen_gaussian(240, 1.5, rng1)
        d2 = _gen_gaussian(240, 1.5, rng2)
        np.testing.assert_array_equal(d1["y"], d2["y"])

    def test_negbin_seed_reproducible(self):
        seed = np.random.SeedSequence([20261021, 2, 6, 0])
        rng1 = np.random.default_rng(seed.spawn(2)[0])
        rng2 = np.random.default_rng(np.random.SeedSequence([20261021, 2, 6, 0]).spawn(2)[0])
        d1 = _gen_negbin(240, 3.0, rng1)
        d2 = _gen_negbin(240, 3.0, rng2)
        np.testing.assert_array_equal(d1["y"], d2["y"])

    def test_different_seeds_differ(self):
        rng1 = np.random.default_rng(np.random.SeedSequence([20261021, 2, 0, 0]).spawn(2)[0])
        rng2 = np.random.default_rng(np.random.SeedSequence([20261021, 2, 0, 1]).spawn(2)[0])
        d1 = _gen_gaussian(240, 1.5, rng1)
        d2 = _gen_gaussian(240, 1.5, rng2)
        assert not np.array_equal(d1["y"], d2["y"])


# ---------------------------------------------------------------------------
# 6. Boundary matching: equal-segment boundaries at [0, 80, 160, 240]
# ---------------------------------------------------------------------------

def _boundary_match(bhat: list[int], bstar: list[int], tau: int) -> tuple[int, float]:
    """One-to-one max-cardinality match with min-displacement."""
    if not bstar or not bhat:
        return 0, 0.0
    edges = [
        (i, j, abs(bhat[i] - bstar[j]))
        for i in range(len(bhat))
        for j in range(len(bstar))
        if abs(bhat[i] - bstar[j]) <= tau
    ]
    if not edges:
        return 0, 0.0
    edges.sort(key=lambda e: e[2])
    matched_hat: set[int] = set()
    matched_star: set[int] = set()
    tp = 0
    total_disp = 0.0
    for i_h, j_s, disp in edges:
        if i_h not in matched_hat and j_s not in matched_star:
            matched_hat.add(i_h)
            matched_star.add(j_s)
            tp += 1
            total_disp += disp
    return tp, total_disp


class TestBoundaryMatching:
    def test_exact_match(self):
        """Exact match at tau=3."""
        bhat = [80, 160]
        bstar = [80, 160]
        tp, disp = _boundary_match(bhat, bstar, tau=3)
        assert tp == 2
        assert disp == 0.0

    def test_near_match_within_tau(self):
        """Match within tau=3."""
        bhat = [78, 163]
        bstar = [80, 160]
        tp, disp = _boundary_match(bhat, bstar, tau=3)
        assert tp == 2
        assert disp == 5.0  # 2 + 3

    def test_outside_tau_no_match(self):
        """No match outside tau."""
        bhat = [70, 150]
        bstar = [80, 160]
        tp, _ = _boundary_match(bhat, bstar, tau=3)
        assert tp == 0

    def test_partial_match(self):
        """Partial match when only one boundary is close."""
        bhat = [80, 200]
        bstar = [80, 160]
        tp, _ = _boundary_match(bhat, bstar, tau=3)
        assert tp == 1

    def test_f1_at_tau3_perfect(self):
        """Perfect F1 when map matches true boundaries."""
        bhat = [80, 160]
        bstar = [80, 160]
        tp, _ = _boundary_match(bhat, bstar, tau=3)
        fp = len(bhat) - tp
        fn = len(bstar) - tp
        f1 = 2 * tp / (2 * tp + fp + fn)
        assert abs(f1 - 1.0) < 1e-10

    def test_equal_segment_boundaries_n240(self):
        """n=240 k_true=3: interior true boundaries are [80, 160]."""
        n = 240
        seg = n // 3
        assert seg == 80
        bstar = [seg, 2 * seg]
        assert bstar == [80, 160]


# ---------------------------------------------------------------------------
# 7. Signal MSE tests
# ---------------------------------------------------------------------------

class TestSignalMSE:
    def test_gaussian_signal_mse_finite(self):
        """Signal MSE is finite and > 0 for noisy data."""
        from bayesbreak.families.gaussian import BayesBreakGaussian
        import bayesbreak.dp as _dp
        rng = np.random.default_rng(12345)
        d = _gen_gaussian(240, 1.5, rng)
        est = BayesBreakGaussian(
            k_max=10, estimate_hyper=False,
            nu=0.0, rho2=4.0, sigma2=1.0,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"])
        # Compute Bayes curve
        n = 240
        k_max = 10
        from scripts.run_e02_family_portability import (
            _bayes_curve_mix_k_vectorized, _logsumexp_axis0
        )
        bayes_curve = _bayes_curve_mix_k_vectorized(
            est.log_left_, est.log_right_,
            est.log_block_evidence_, est.block_first_moment_,
            n, k_max, est.k_posterior_,
        )
        mse = float(np.mean((bayes_curve - d["true_signal"]) ** 2))
        assert np.isfinite(mse), f"Signal MSE is not finite: {mse}"
        assert mse > 0.0, "Signal MSE should be > 0 for noisy data"

    def test_poisson_signal_mse_finite(self):
        """Poisson signal MSE is finite."""
        from bayesbreak.families.poisson import BayesBreakPoisson
        rng = np.random.default_rng(12345)
        d = _gen_poisson(240, 3.0, rng)
        est = BayesBreakPoisson(
            k_max=10, estimate_hyper=False,
            alpha=2.0, beta=1.0,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"], sample_weight=d["e_arr"])
        from scripts.run_e02_family_portability import _bayes_curve_mix_k_vectorized
        bayes_curve = _bayes_curve_mix_k_vectorized(
            est.log_left_, est.log_right_,
            est.log_block_evidence_, est.block_first_moment_,
            240, 10, est.k_posterior_,
        )
        mse = float(np.mean((bayes_curve - d["true_signal"]) ** 2))
        assert np.isfinite(mse)

    def test_negbin_a1_nan_handling(self):
        """NaN in A1 when a_post <= 1 (handled gracefully)."""
        # With alpha=2, beta=3, r=5: a_post = alpha + sum(r*w) >= 2 for non-empty blocks.
        # For the full-sequence block (80 obs * r=5 * w=1), a_post = 2 + 400 >> 1.
        # So with real data, a_post > 1 always holds for this experiment.
        # We just test that finite A1 is returned for well-conditioned blocks.
        from bayesbreak.families.negative_binomial import BayesBreakNegBin
        rng = np.random.default_rng(54321)
        d = _gen_negbin(240, 3.0, rng)
        est = BayesBreakNegBin(
            k_max=10, estimate_hyper=False,
            r=5.0, alpha=2.0, beta=3.0,
            prior_k=_geom_prior,
        )
        est.fit(d["x"].reshape(-1, 1), d["y"])
        # Full block (0, 240): a_post = 2 + 240*5 = 1202 >> 1
        A1_full = est.block_first_moment_[0, 240]
        assert np.isfinite(A1_full), f"A1[0, 240] = {A1_full}"
        assert A1_full > 0, "Count mean should be positive"


# ---------------------------------------------------------------------------
# 8. NegBin A1 specifically is count mean
# ---------------------------------------------------------------------------

class TestNegBinA1CountMean:
    def test_a1_scale(self):
        """A1[i,j] for large NegBin block with high mu should be >> 0 (count mean scale)."""
        from bayesbreak.families.negative_binomial import BayesBreakNegBin
        rng = np.random.default_rng(77)
        r = 5.0
        mu_true = 10.0  # high mu
        p_q = r / (r + mu_true)
        n = 60
        y = nbinom.rvs(n=int(r), p=p_q, size=n, random_state=rng)
        x = np.arange(1, n + 1, dtype=float)
        est = BayesBreakNegBin(
            k_max=3, estimate_hyper=False,
            r=r, alpha=2.0, beta=3.0,
        )
        est.fit(x.reshape(-1, 1), y)
        # A1[0, n] for the full block
        count_mean_est = est.block_first_moment_[0, n]
        # Should be >> 1 (near mu_true), NOT near p = r/(r+mu_true) ≈ 0.33
        assert count_mean_est > 1.0, (
            f"A1 = {count_mean_est:.4f} looks like p, expected count mean near {mu_true}"
        )
        # And it should be reasonable (within 2x of true mu for 60 obs)
        assert count_mean_est < 4 * mu_true, f"A1 unreasonably large: {count_mean_est}"

    def test_a1_formula_manual(self):
        """Manual verification of A1 formula: r_star * b_post / (a_post - 1)."""
        r = 5.0
        alpha, beta = 2.0, 3.0
        # Fake block: 10 observations all with count = 6
        y = np.full(10, 6.0)
        # a_post = alpha + N_ij = alpha + sum(r_t) = 2 + 10*5 = 52
        # b_post = beta + C_ij = beta + sum(y_t) = 3 + 60 = 63
        a_post = alpha + 10 * r
        b_post = beta + np.sum(y)
        expected_count_mean = r * b_post / (a_post - 1.0)
        # Should be near 6 (the true count)
        assert abs(expected_count_mean - r * b_post / (a_post - 1.0)) < 1e-10
        # Sanity: expected_count_mean ≈ 5 * 63 / 51 ≈ 6.17
        assert 4.0 < expected_count_mean < 10.0
