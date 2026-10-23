"""Oracle tests for Gaussian (known-precision) and Poisson (known-exposure) block evidence.

Expected values are computed from closed-form formulae derived independently of the
BayesBreak implementation, using paper Propositions 1 and 2 (§3b), plus independent
numerical quadrature for the generalized-Bayes (power-weight) cases.

Test organisation
-----------------
Gaussian — descriptor only (a_r = 1, w_r varies):
  unit precision baseline, single non-unit precision, heterogeneous precisions,
  posterior mean, posterior predictive.

Gaussian — power weights + descriptors combined (a_r ≠ 1, w_r ≠ 1):
  scalar power weight, heterogeneous power weights × heterogeneous precisions.
  Each case cross-checked against scipy.integrate.quad.

Poisson — descriptor only (a_r = 1, e_r varies):
  unit exposure baseline, single non-unit exposure, heterogeneous exposures,
  posterior rate, posterior predictive.

Poisson — power weights + descriptors combined:
  scalar power weight × unit exposure, heterogeneous power weights × exposures.
  Each case cross-checked against scipy.integrate.quad.

Zero-exposure edge cases:
  y=0, e=0: unit likelihood factor; contributes nothing to block evidence.
  y>0, e=0: impossible under Poisson; raises ValueError at fit() time.
  a=0, y>0, e=0: masked observation; contributes nothing, does not raise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma, norm, poisson

from bayesbreak import BayesBreakGaussian, BayesBreakPoisson


# ---------------------------------------------------------------------------
# Quadrature helpers (independent of BayesBreak implementation)
# ---------------------------------------------------------------------------

def _gaussian_logZ_quad(y, w, a, nu, rho2, sigma2):
    """Numerically integrate the power-weighted Normal-Normal marginal likelihood."""
    y, w, a = np.asarray(y, float), np.asarray(w, float), np.asarray(a, float)
    SAW = float(np.sum(a * w))
    SAWy = float(np.sum(a * w * y))
    prec_post = SAW / sigma2 + 1.0 / rho2
    mu_post = (SAWy / sigma2 + nu / rho2) / prec_post
    std_post = math.sqrt(1.0 / prec_post)

    def log_f(mu):
        ll = sum(float(ai) * norm.logpdf(float(yi), loc=mu, scale=math.sqrt(sigma2 / float(wi)))
                 for yi, wi, ai in zip(y, w, a))
        return ll + norm.logpdf(mu, loc=nu, scale=math.sqrt(rho2))

    log_peak = log_f(mu_post)
    val, _ = quad(
        lambda mu: math.exp(max(log_f(mu) - log_peak, -700)),
        mu_post - 15 * std_post,
        mu_post + 15 * std_post,
        limit=300,
    )
    return math.log(val) + log_peak


def _poisson_logZ_quad(y, e, a, alpha, beta):
    """Numerically integrate the power-weighted Gamma-Poisson marginal likelihood."""
    y, e, a = np.asarray(y, float), np.asarray(e, float), np.asarray(a, float)
    AY = float(np.sum(a * y))
    AE = float(np.sum(a * e))
    mode = max((alpha + AY - 1) / (beta + AE), 1e-8)

    def log_f(lam):
        if lam <= 0:
            return -1e300
        ll = sum(float(ai) * poisson.logpmf(int(yi), mu=float(ei) * lam)
                 for yi, ei, ai in zip(y, e, a))
        return ll + gamma.logpdf(lam, a=alpha, scale=1.0 / beta)

    log_peak = log_f(mode)
    val, _ = quad(
        lambda lam: math.exp(max(log_f(lam) - log_peak, -700)),
        1e-12, mode * 10 + 20,
        limit=300,
    )
    return math.log(val) + log_peak


# ---------------------------------------------------------------------------
# Gaussian — descriptor only (a_r = 1)
# ---------------------------------------------------------------------------


class TestGaussianKnownPrecisionOracle:
    """Proposition 1 (§3b): log A^(0)_B with per-observation precisions w_r, a_r = 1."""

    def _fit(self, y, w, nu=0.0, rho2=1.0, sigma2=1.0):
        n = len(y)
        return BayesBreakGaussian(
            k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2,
        ).fit(np.arange(n).reshape(-1, 1), np.asarray(y, float),
              sample_weight=np.asarray(w, float))

    def test_single_obs_unit_precision(self):
        """w=1 baseline: ΣlogW = 0; marginal is N(y; ν, σ²+ρ²)."""
        y0, nu, rho2, sigma2 = 2.5, 1.0, 2.0, 0.5
        est = self._fit([y0], [1.0], nu=nu, rho2=rho2, sigma2=sigma2)
        var = sigma2 + rho2
        expected = -0.5 * math.log(2 * math.pi * var) - 0.5 * (y0 - nu) ** 2 / var
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected, abs=1e-10)

    def test_single_obs_nonunit_precision(self):
        """w=4: ΣlogW = log(4); normalization differs from power-weight −W/2·log(2πσ²)."""
        # y=1, w=4, ν=0, ρ²=1, σ²=1  →  marginal N(1; 0, 5/4)
        est = self._fit([1.0], [4.0])
        var = 1.0 / 4.0 + 1.0
        expected = -0.5 * math.log(2 * math.pi * var) - 0.5 * 1.0 / var
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected, abs=1e-10)

    def test_multi_obs_heterogeneous_precisions(self):
        """y=[1,2], w=[2,3]: matrix-det-lemma reference −log(2π) − 5/3."""
        # Cov = diag([1/2, 1/3]) + 11^T; det=1; quad=10/3
        est = self._fit([1.0, 2.0], [2.0, 3.0])
        expected = -math.log(2 * math.pi) - 5.0 / 3.0
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_posterior_mean_nonunit_precision(self):
        """E[μ|(i,j]] = (ρ²·Σw·y + σ²·ν) / (ρ²·Σw + σ²)."""
        est = self._fit([1.0, 2.0], [2.0, 3.0])
        # Σw·y = 8; Σw = 5; μ_post = 8/6
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(8.0 / 6.0, abs=1e-10)

    def test_posterior_predictive_nonunit_precision(self):
        """Predictive is N(μ_post, σ²/w_new + ρ²_post)."""
        y_train = np.array([1.0, 2.0])
        w_train = np.array([2.0, 3.0])
        est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=0.0, rho2=1.0, sigma2=1.0
                                 ).fit(np.arange(2).reshape(-1, 1), y_train, sample_weight=w_train)
        mu_post, rho2_post = 8.0 / 6.0, 1.0 / 6.0
        y_new = np.array([0.5, 2.0])
        w_new = np.array([1.0, 4.0])
        var_new = 1.0 / w_new + rho2_post
        expected = -0.5 * (np.log(2 * math.pi * var_new) + (y_new - mu_post) ** 2 / var_new)
        observed = est.posterior_predictive_logpdf_block(a=0, b=2, y_new=y_new, w_new=w_new)
        np.testing.assert_allclose(observed, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Gaussian — generalized-Bayes power weights combined with descriptors
# ---------------------------------------------------------------------------


class TestGaussianPowerWeightOracle:
    """General formula: a_r ≠ 1 and w_r ≠ 1, cross-verified against quadrature."""

    TOL = 1e-6  # quadrature + analytic tolerance

    def _fit(self, y, w, a, nu=0.0, rho2=1.0, sigma2=1.0):
        n = len(y)
        return BayesBreakGaussian(
            k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2,
        ).fit(np.arange(n).reshape(-1, 1), np.asarray(y, float),
              sample_weight=np.asarray(w, float),
              likelihood_power=np.asarray(a, float))

    def test_scalar_power_unit_precision(self):
        """a=2, w=1: generalized-Bayes doubles each observation's weight."""
        # Reduces to a=1, w=1 case for unit precision but with SA=2 instead of 1
        est = self._fit([1.0], [1.0], [2.0])
        expected_quad = _gaussian_logZ_quad([1.0], [1.0], [2.0], 0.0, 1.0, 1.0)
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected_quad, abs=self.TOL)

    def test_scalar_power_nonunit_precision(self):
        """a=2, w=4, y=1, ν=0, ρ²=1, σ²=1: SA=2, SAW=8, SAlogW=2·log(4)."""
        est = self._fit([1.0], [4.0], [2.0])
        # Analytic: term1=-log(2π)+log(4), term2=-½log(9), term3=64/9-8)/2=-4/9
        SA, SAW, SAlogW = 2.0, 8.0, 2.0 * math.log(4)
        t1 = -0.5 * SA * math.log(2 * math.pi) + 0.5 * SAlogW
        t2 = -0.5 * math.log1p(SAW)
        t3 = 0.5 * (SAW ** 2 / (SAW + 1.0) - SAW)  # ysum_c=SAW*y=8, y2sum=SAW*y^2=8
        expected_analytic = t1 + t2 + t3
        expected_quad = _gaussian_logZ_quad([1.0], [4.0], [2.0], 0.0, 1.0, 1.0)
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected_analytic, abs=1e-10)
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected_quad, abs=self.TOL)

    def test_heterogeneous_power_and_precision(self):
        """a=[2,0.5], w=[2,3], y=[1,2], ν=0, ρ²=1, σ²=1: cross-checked by quadrature."""
        a_v, w_v, y_v = [2.0, 0.5], [2.0, 3.0], [1.0, 2.0]
        est = self._fit(y_v, w_v, a_v)
        a_arr, w_arr, y_arr = np.array(a_v), np.array(w_v), np.array(y_v)
        SA = float(np.sum(a_arr))
        SAW = float(np.sum(a_arr * w_arr))
        SAWy = float(np.sum(a_arr * w_arr * y_arr))
        SAWy2 = float(np.sum(a_arr * w_arr * y_arr ** 2))
        SAlogW = float(np.sum(a_arr * np.log(w_arr)))
        t1 = -0.5 * SA * math.log(2 * math.pi) + 0.5 * SAlogW
        t2 = -0.5 * math.log1p(SAW)
        denom = SAW + 1.0
        t3 = 0.5 * (SAWy ** 2 / denom - SAWy2)
        expected_analytic = t1 + t2 + t3
        expected_quad = _gaussian_logZ_quad(y_v, w_v, a_v, 0.0, 1.0, 1.0)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected_analytic, abs=1e-10)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected_quad, abs=self.TOL)

    def test_unit_power_recovers_descriptor_oracle(self):
        """a_r = 1 for all r: must match the Proposition 1 descriptor formula exactly."""
        # y=[1,2], w=[2,3]: expected −log(2π) − 5/3 from TestGaussianKnownPrecisionOracle
        est = self._fit([1.0, 2.0], [2.0, 3.0], [1.0, 1.0])
        expected = -math.log(2 * math.pi) - 5.0 / 3.0
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_power_weighted_posterior_mean(self):
        """Posterior mean uses Σa·w·y and Σa·w as effective precision-weighted statistics."""
        a_v, w_v, y_v = [2.0, 0.5], [2.0, 3.0], [1.0, 2.0]
        est = self._fit(y_v, w_v, a_v)
        a_arr, w_arr, y_arr = np.array(a_v), np.array(w_v), np.array(y_v)
        SAW = float(np.sum(a_arr * w_arr))
        SAWy = float(np.sum(a_arr * w_arr * y_arr))
        mu_expected = (1.0 * SAWy + 1.0 * 0.0) / (1.0 * SAW + 1.0)  # ρ²=σ²=1, ν=0
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(mu_expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Poisson — descriptor only (a_r = 1)
# ---------------------------------------------------------------------------


class TestPoissonKnownExposureOracle:
    """Proposition 2 (§3b): log A^(0)_B with per-observation exposures e_r, a_r = 1."""

    def _fit(self, y, w, alpha=1.0, beta=1.0):
        n = len(y)
        return BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta,
        ).fit(np.arange(n).reshape(-1, 1), np.asarray(y, float),
              sample_weight=np.asarray(w, float))

    def test_single_obs_unit_exposure(self):
        """w=1 baseline: H_B = −lgamma(y+1); reduces to standard Gamma-Poisson."""
        # y=3, e=1, α=1, β=1  →  A = 1/16
        est = self._fit([3], [1.0])
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(-4.0 * math.log(2.0), abs=1e-10)

    def test_single_obs_nonunit_exposure(self):
        """w=2, y=3: integral = 8/81 exactly; requires Σy·log(e) base-measure term."""
        # p(y=3;e=2,α=1,β=1) = ∫ e^{−2λ}(2λ)³/6·e^{−λ}dλ = 8/81
        est = self._fit([3], [2.0])
        expected = 3.0 * math.log(2.0) - 4.0 * math.log(3.0)
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected, abs=1e-10)

    def test_multi_obs_heterogeneous_exposures(self):
        """y=[1,4], w=[1,2], α=2, β=1: A^(0)_B = 15/512 exactly."""
        # Verified by scipy.integrate.quad: 0.029296875 = 15/512
        est = self._fit([1, 4], [1.0, 2.0], alpha=2.0, beta=1.0)
        expected = math.log(15.0) - 9.0 * math.log(2.0)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_posterior_rate_nonunit_exposure(self):
        """E[λ|(i,j]] = (α+Y_B)/(β+E_B) uses unweighted count Y_B."""
        est = self._fit([1, 4], [1.0, 2.0], alpha=2.0, beta=1.0)
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(7.0 / 4.0, abs=1e-10)

    def test_posterior_predictive_nonunit_exposure(self):
        """Predictive is NegBin(α_post, p = β_post/(β_post+e_new))."""
        y_train = np.array([1.0, 4.0])
        w_train = np.array([1.0, 2.0])
        est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=2.0, beta=1.0
                                ).fit(np.arange(2).reshape(-1, 1), y_train, sample_weight=w_train)
        alpha_post, beta_post = 7.0, 4.0
        y_new, w_new = np.array([2.0]), np.array([3.0])
        p = beta_post / (beta_post + 3.0)
        expected = (math.lgamma(alpha_post + 2.0) - math.lgamma(3.0) - math.lgamma(alpha_post)
                    + alpha_post * math.log(p) + 2.0 * math.log(1.0 - p))
        observed = float(est.posterior_predictive_logpdf_block(a=0, b=2, y_new=y_new, w_new=w_new)[0])
        assert observed == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Poisson — generalized-Bayes power weights combined with descriptors
# ---------------------------------------------------------------------------


class TestPoissonPowerWeightOracle:
    """General formula: a_r ≠ 1 and e_r ≠ 1, cross-verified against quadrature."""

    TOL = 1e-6

    def _fit(self, y, w, a, alpha=1.0, beta=1.0):
        n = len(y)
        return BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta,
        ).fit(np.arange(n).reshape(-1, 1), np.asarray(y, float),
              sample_weight=np.asarray(w, float),
              likelihood_power=np.asarray(a, float))

    def test_scalar_power_unit_exposure(self):
        """a=2, e=1, y=3: SaY=6, SaE=2; formula = −2·lgamma(4)+lgamma(7)−7·log(3)."""
        est = self._fit([3], [1.0], [2.0])
        expected_analytic = -2.0 * math.lgamma(4) + math.lgamma(7) - 7.0 * math.log(3.0)
        expected_quad = _poisson_logZ_quad([3], [1.0], [2.0], 1.0, 1.0)
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected_analytic, abs=1e-10)
        assert float(est.log_block_evidence_[0, 1]) == pytest.approx(expected_quad, abs=self.TOL)

    def test_heterogeneous_power_and_exposure(self):
        """a=[2,0.5], e=[1,2], y=[1,4], α=2, β=1: cross-checked by quadrature."""
        a_v, e_v, y_v = [2.0, 0.5], [1.0, 2.0], [1.0, 4.0]
        est = self._fit(y_v, e_v, a_v, alpha=2.0, beta=1.0)
        a_arr, e_arr, y_arr = np.array(a_v), np.array(e_v), np.array(y_v)
        SaY = float(np.sum(a_arr * y_arr))          # 2*1 + 0.5*4 = 4
        SaE = float(np.sum(a_arr * e_arr))          # 2*1 + 0.5*2 = 3
        SalogE = float(np.sum(a_arr * y_arr * np.log(np.maximum(e_arr, 1e-300))))
        aLfac = float(np.sum(a_arr * np.array([math.lgamma(2), math.lgamma(5)])))
        expected_analytic = (SalogE - aLfac + 2.0 * math.log(1.0) - math.lgamma(2.0)
                             + math.lgamma(2.0 + SaY) - (2.0 + SaY) * math.log(1.0 + SaE))
        expected_quad = _poisson_logZ_quad(y_v, e_v, a_v, 2.0, 1.0)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected_analytic, abs=1e-10)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected_quad, abs=self.TOL)

    def test_unit_power_recovers_descriptor_oracle(self):
        """a_r = 1 for all r: must match the Proposition 2 descriptor formula exactly."""
        est = self._fit([1, 4], [1.0, 2.0], [1.0, 1.0], alpha=2.0, beta=1.0)
        expected = math.log(15.0) - 9.0 * math.log(2.0)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_power_weighted_posterior_rate(self):
        """E[λ|block] = (α+SaY)/(β+SaE) uses power-weighted sums."""
        a_v, e_v, y_v = [2.0, 0.5], [1.0, 2.0], [1.0, 4.0]
        est = self._fit(y_v, e_v, a_v, alpha=2.0, beta=1.0)
        SaY = 2.0 * 1.0 + 0.5 * 4.0  # = 4
        SaE = 2.0 * 1.0 + 0.5 * 2.0  # = 3
        expected_rate = (2.0 + SaY) / (1.0 + SaE)
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(expected_rate, abs=1e-10)


# ---------------------------------------------------------------------------
# Zero-exposure edge cases
# ---------------------------------------------------------------------------


class TestPoissonZeroExposure:
    """Zero exposure: y=0 is a unit-likelihood factor; y>0 with e=0 is impossible.

    Semantics (exact, not sentinel-based):
      - y=0, e=0: P(0|λ,0) = e^{−0} = 1; contributes 0 to log block evidence.
      - y>0, e=0: P(y|λ,0) = 0 for any λ; raises ValueError at fit() time.
      - a=0, y>0, e=0: masked by zero power weight; contributes nothing, no error.
    """

    def test_zero_exposure_zero_count_contributes_nothing(self):
        """y=0, e=0: unit likelihood; block evidence is identical to the zero-obs case."""
        # Fit y=[3], e=[1] (bare).
        est_bare = BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0,
        ).fit(np.arange(1).reshape(-1, 1), np.array([3.0]),
              sample_weight=np.array([1.0]))
        # Fit y=[0, 3], e=[0, 1]: the first obs contributes nothing to evidence.
        est_padded = BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0,
        ).fit(np.arange(2).reshape(-1, 1), np.array([0.0, 3.0]),
              sample_weight=np.array([0.0, 1.0]))
        # Block (0,2) of padded dataset must equal block (0,1) of bare dataset.
        assert float(est_bare.log_block_evidence_[0, 1]) == pytest.approx(
            float(est_padded.log_block_evidence_[0, 2]), abs=1e-10
        )

    def test_positive_count_zero_exposure_raises(self):
        """y=1, e=0: P(1|λ,0) = 0 for all λ; must raise ValueError, not silently produce -inf."""
        with pytest.raises(ValueError, match="zero exposure"):
            BayesBreakPoisson(
                k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0,
            ).fit(np.arange(1).reshape(-1, 1), np.array([1.0]),
                  sample_weight=np.array([0.0]))

    def test_positive_count_zero_exposure_raises_lists_indices(self):
        """Informative error must report which observation indices are impossible."""
        with pytest.raises(ValueError, match=r"indices \[1\]"):
            BayesBreakPoisson(
                k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0,
            ).fit(np.arange(2).reshape(-1, 1), np.array([3.0, 2.0]),
                  sample_weight=np.array([1.0, 0.0]))

    def test_zero_power_impossible_obs_is_masked(self):
        """a=0, y=1, e=0: zero power masks the observation; no error, block unchanged."""
        # The zero-power obs contributes 0 to all statistics — same as absent.
        est_masked = BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0,
        ).fit(np.arange(2).reshape(-1, 1), np.array([1.0, 3.0]),
              sample_weight=np.array([0.0, 1.0]),
              likelihood_power=np.array([0.0, 1.0]))
        est_bare = BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0,
        ).fit(np.arange(1).reshape(-1, 1), np.array([3.0]),
              sample_weight=np.array([1.0]))
        # The two-obs fit (obs 0 masked) should give the same full-block evidence
        # as the single-obs fit.
        assert float(est_masked.log_block_evidence_[0, 2]) == pytest.approx(
            float(est_bare.log_block_evidence_[0, 1]), abs=1e-10
        )
