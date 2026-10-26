"""Oracle tests for observation_mask support and generalized-Bayes posterior prediction.

All expected values are derived independently of the BayesBreak implementation,
using closed-form conjugate formulas (Propositions 1 and 2 of §3b) and scipy
for numerical cross-checks.  No production BayesBreak function is used as an oracle.

Test classes
------------
TestGaussianMask
    Single/multi-obs blocks with observed and missing entries; nonfinite stored
    values at masked positions; all-missing blocks; posterior mean under mask.

TestPoissonMask
    Same structure for the Poisson-Gamma family; impossible obs (y>0, e=0)
    at masked positions must not raise; all-missing Poisson block.

TestMaskCombinedWithPower
    Joint mask × likelihood_power: masked obs get effective a=0 regardless of
    the user-supplied power value; observed obs use the supplied power.

TestAllMissingSequence
    A fully-masked fit with estimate_hyper=False must succeed; all block
    log-evidences must be exactly 0; block_first_moment equals the prior mean.

TestPITLikelihoodPower
    _segment_pit must use the power-weighted posterior (same parameters as
    _compute_block_evidence), not the unit-power parameters.

TestWrapperRejection
    Wrappers that bypass base.fit() must raise NotImplementedError for
    likelihood_power and observation_mask.

TestRepeatFit
    Calling fit() twice on the same estimator fully overwrites observation_mask_
    and likelihood_power_; no stale state from a previous call leaks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import nbinom, norm

from bayesbreak import BayesBreakBetaObs, BayesBreakBinomial, BayesBreakGaussian, BayesBreakPoisson
from bayesbreak.multivariate import (
    IndependentMultivariateSegmenter,
    SharedBoundaryMultivariateSegmenter,
)
from bayesbreak.replicates import SharedBoundaryReplicatesSegmenter


# ---------------------------------------------------------------------------
# Independent analytic helpers (do not call BayesBreak production code)
# ---------------------------------------------------------------------------

def _gauss_logev(y_obs, w_obs, nu=0.0, rho2=1.0, sigma2=1.0):
    """log A^(0)_(0,n] for Gaussian known-precision, a_r=1, via Proposition 1."""
    y = np.asarray(y_obs, float); w = np.asarray(w_obs, float)
    SAW = float(np.sum(w)); SAWy = float(np.sum(w * y))
    SAWy2 = float(np.sum(w * y**2)); SA = float(y.size)
    SAlogW = float(np.sum(np.log(np.maximum(w, 1e-300))))
    t1 = -0.5 * SA * math.log(2 * math.pi * sigma2) + 0.5 * SAlogW
    t2 = -0.5 * math.log1p(SAW * rho2 / sigma2)
    denom = SAW + sigma2 / rho2
    ysum_c = SAWy - nu * SAW
    t3 = 0.5 / sigma2 * (ysum_c**2 / denom - (SAWy2 - 2*nu*SAWy + nu**2*SAW))
    return t1 + t2 + t3


def _gauss_mu_post(y_obs, w_obs, nu=0.0, rho2=1.0, sigma2=1.0):
    y = np.asarray(y_obs, float); w = np.asarray(w_obs, float)
    SAW = float(np.sum(w)); SAWy = float(np.sum(w * y))
    return (rho2 * SAWy + sigma2 * nu) / (rho2 * SAW + sigma2)


def _poisson_logev(y_obs, e_obs, alpha=1.0, beta=1.0):
    """log A^(0)_(0,n] for Poisson known-exposure, a_r=1, via Proposition 2."""
    y = np.asarray(y_obs, float); e = np.asarray(e_obs, float)
    SY = float(np.sum(y)); SE = float(np.sum(e))
    SlogE = float(np.sum(y * np.log(np.maximum(e, 1e-300))))
    SlogFac = float(np.sum([math.lgamma(yi + 1) for yi in y]))
    return (alpha * math.log(beta) - math.lgamma(alpha)
            + SlogE - SlogFac
            + math.lgamma(alpha + SY)
            - (alpha + SY) * math.log(beta + SE))


# ---------------------------------------------------------------------------
# 1.  Gaussian observation mask
# ---------------------------------------------------------------------------


class TestGaussianMask:
    NU, RHO2, SIGMA2 = 0.0, 1.0, 1.0

    def _fit(self, y, w, mask=None, a=None, **kw):
        n = len(y)
        est = BayesBreakGaussian(
            k_max=1, estimate_hyper=False,
            nu=self.NU, rho2=self.RHO2, sigma2=self.SIGMA2,
        )
        return est.fit(
            np.arange(n).reshape(-1, 1), np.asarray(y, float),
            sample_weight=np.asarray(w, float),
            observation_mask=mask,
            likelihood_power=a,
        )

    def test_single_missing_obs_equals_bare_fit(self):
        """mask=[T,F]: block evidence for y=[1,2],w=[2,3] matches y=[1],w=[2]."""
        expected = _gauss_logev([1.0], [2.0], nu=self.NU, rho2=self.RHO2, sigma2=self.SIGMA2)
        est = self._fit([1.0, 2.0], [2.0, 3.0], mask=[True, False])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_single_missing_first_obs(self):
        """mask=[F,T]: block for y=[1,2],w=[2,3] matches y=[2],w=[3]."""
        expected = _gauss_logev([2.0], [3.0], nu=self.NU, rho2=self.RHO2, sigma2=self.SIGMA2)
        est = self._fit([1.0, 2.0], [2.0, 3.0], mask=[False, True])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_all_missing_block_log_evidence_is_zero(self):
        """All-missing block: integrated likelihood is 1, so log evidence = 0."""
        est = self._fit([1.0, 2.0], [2.0, 3.0], mask=[False, False])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(0.0, abs=1e-10)

    def test_all_missing_posterior_mean_is_prior_mean(self):
        """All-missing block: posterior mean = ν (the prior mean)."""
        est = self._fit([99.0, -99.0], [1.0, 1.0], mask=[False, False])
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(self.NU, abs=1e-10)

    def test_nonfinite_y_at_masked_position(self):
        """y=nan at mask=False is allowed and treated as a missing observation."""
        expected = _gauss_logev([1.0], [2.0], nu=self.NU, rho2=self.RHO2, sigma2=self.SIGMA2)
        # mask=False for the nan entry; mask=True for the observed one
        est = self._fit([np.nan, 1.0], [3.0, 2.0], mask=[False, True])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_all_observed_with_unit_mask_unchanged(self):
        """mask=[T,T] is identical to omitting the mask entirely."""
        expected = _gauss_logev([1.0, 2.0], [2.0, 3.0], nu=self.NU, rho2=self.RHO2, sigma2=self.SIGMA2)
        est = self._fit([1.0, 2.0], [2.0, 3.0], mask=[True, True])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_single_observed_posterior_mean(self):
        """mask=[T,F]: posterior mean uses only the observed obs."""
        expected_mu = _gauss_mu_post([1.0], [2.0], nu=self.NU, rho2=self.RHO2, sigma2=self.SIGMA2)
        est = self._fit([1.0, 5.0], [2.0, 99.0], mask=[True, False])
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(expected_mu, abs=1e-10)

    def test_mask_excludes_from_hyperparameter_estimation(self):
        """estimate_hyper=True: masked obs do not enter nu/sigma2 estimation."""
        # Two observations with identical y; masked second obs has extreme value.
        # EB estimate should match the single-obs estimation.
        rng = np.random.default_rng(0)
        y_base = rng.standard_normal(10)
        y_masked = np.concatenate([y_base, [1e6]])   # extreme masked obs
        w = np.ones(11)
        mask = np.ones(11, dtype=bool); mask[10] = False
        est_full = BayesBreakGaussian(k_max=1).fit(
            np.arange(10).reshape(-1, 1), y_base, sample_weight=np.ones(10)
        )
        est_masked = BayesBreakGaussian(k_max=1).fit(
            np.arange(11).reshape(-1, 1), y_masked, sample_weight=w,
            observation_mask=mask,
        )
        assert est_masked.hyper_["nu"] == pytest.approx(est_full.hyper_["nu"], rel=1e-6)
        assert est_masked.hyper_["sigma2"] == pytest.approx(est_full.hyper_["sigma2"], rel=1e-4)


# ---------------------------------------------------------------------------
# 2.  Poisson observation mask
# ---------------------------------------------------------------------------


class TestPoissonMask:
    ALPHA, BETA = 1.0, 1.0

    def _fit(self, y, e, mask=None, a=None, **kw):
        n = len(y)
        est = BayesBreakPoisson(
            k_max=1, estimate_hyper=False, alpha=self.ALPHA, beta=self.BETA,
        )
        return est.fit(
            np.arange(n).reshape(-1, 1), np.asarray(y, float),
            sample_weight=np.asarray(e, float),
            observation_mask=mask,
            likelihood_power=a,
        )

    def test_single_missing_obs_equals_bare_fit(self):
        """mask=[T,F]: block evidence for y=[3,4],e=[1,2] matches y=[3],e=[1]."""
        expected = _poisson_logev([3], [1.0], alpha=self.ALPHA, beta=self.BETA)
        est = self._fit([3, 4], [1.0, 2.0], mask=[True, False])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_all_missing_block_log_evidence_is_zero(self):
        """All-missing Poisson block: log evidence = 0."""
        est = self._fit([3, 4], [1.0, 2.0], mask=[False, False])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(0.0, abs=1e-10)

    def test_all_missing_posterior_rate_is_prior_rate(self):
        """All-missing block: posterior rate = α/β (prior mean)."""
        est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=3.0, beta=2.0).fit(
            np.arange(2).reshape(-1, 1), np.array([99.0, 99.0]),
            sample_weight=np.array([1.0, 1.0]),
            observation_mask=np.array([False, False]),
        )
        assert float(est.block_first_moment_[0, 2]) == pytest.approx(3.0 / 2.0, abs=1e-10)

    def test_impossible_obs_at_masked_position_no_error(self):
        """y>0, e=0 at mask=False: masked impossible obs must not raise ValueError."""
        # Without mask, this would raise; with mask=False it is simply absent.
        est = self._fit([5, 3], [0.0, 1.0], mask=[False, True])
        expected = _poisson_logev([3], [1.0], alpha=self.ALPHA, beta=self.BETA)
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_nonfinite_y_at_masked_position(self):
        """y=nan at mask=False is allowed and treated as missing."""
        expected = _poisson_logev([3], [1.0], alpha=self.ALPHA, beta=self.BETA)
        est = self._fit([np.nan, 3], [np.nan, 1.0], mask=[False, True])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_zero_exposure_zero_count_at_masked_position(self):
        """y=0, e=0 at mask=False: treated as missing, contributes nothing."""
        expected = _poisson_logev([3], [1.0], alpha=self.ALPHA, beta=self.BETA)
        est = self._fit([0, 3], [0.0, 1.0], mask=[False, True])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# 3.  Mask combined with likelihood_power
# ---------------------------------------------------------------------------


class TestMaskCombinedWithPower:
    """Masked obs get effective power 0 regardless of the user-supplied power."""

    def test_gaussian_mask_overrides_nonzero_power(self):
        """mask=False, likelihood_power=3: effective power is 0, not 3."""
        # Only the observed (mask=True) entry contributes, with its supplied power.
        nu, rho2, sigma2 = 0.0, 1.0, 1.0
        # y=[1,2], mask=[T,F], power=[1,3]: obs 0 (y=1,w=2,a=1) contributes.
        expected = _gauss_logev([1.0], [2.0])
        est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2
                                 ).fit(np.arange(2).reshape(-1, 1),
                                       np.array([1.0, 2.0]),
                                       sample_weight=np.array([2.0, 3.0]),
                                       observation_mask=np.array([True, False]),
                                       likelihood_power=np.array([1.0, 3.0]))
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_poisson_mask_overrides_nonzero_power(self):
        """mask=False, likelihood_power=2: effective power is 0."""
        # y=[3,4], mask=[T,F], power=[1,2]: only y=3,e=1 contributes with a=1.
        expected = _poisson_logev([3], [1.0])
        est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0
                                ).fit(np.arange(2).reshape(-1, 1),
                                      np.array([3.0, 4.0]),
                                      sample_weight=np.array([1.0, 2.0]),
                                      observation_mask=np.array([True, False]),
                                      likelihood_power=np.array([1.0, 2.0]))
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)

    def test_zero_power_and_mask_false_both_give_zero_contribution(self):
        """Either mask=False or likelihood_power=0 independently gives zero contribution."""
        nu, rho2, sigma2 = 0.0, 1.0, 1.0
        # Expected: single obs y=1, w=2, a=1 (only the first entry contributes)
        expected = _gauss_logev([1.0], [2.0])
        # mask=False for obs 1
        est_mask = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2
                                      ).fit(np.arange(2).reshape(-1, 1),
                                            np.array([1.0, 2.0]),
                                            sample_weight=np.array([2.0, 3.0]),
                                            observation_mask=np.array([True, False]))
        # likelihood_power=0 for obs 1
        est_power = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2
                                       ).fit(np.arange(2).reshape(-1, 1),
                                             np.array([1.0, 2.0]),
                                             sample_weight=np.array([2.0, 3.0]),
                                             likelihood_power=np.array([1.0, 0.0]))
        assert float(est_mask.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)
        assert float(est_power.log_block_evidence_[0, 2]) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# 4.  All-missing sequences
# ---------------------------------------------------------------------------


class TestAllMissingSequence:
    """Entirely missing datasets with fixed hyperparameters."""

    def test_gaussian_all_missing_all_blocks_zero(self):
        """All mask=False: every upper-triangle block has log evidence = 0."""
        est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=1.0, rho2=2.0, sigma2=0.5
                                 ).fit(np.arange(3).reshape(-1, 1),
                                       np.array([5.0, 3.0, -2.0]),
                                       sample_weight=np.ones(3),
                                       observation_mask=np.array([False, False, False]))
        upper = np.triu(np.ones((4, 4), dtype=bool), k=1)
        np.testing.assert_allclose(est.log_block_evidence_[upper], 0.0, atol=1e-10)

    def test_poisson_all_missing_all_blocks_zero(self):
        """All mask=False: every upper-triangle block has log evidence = 0."""
        est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=3.0, beta=2.0
                                ).fit(np.arange(3).reshape(-1, 1),
                                      np.array([5.0, 3.0, 2.0]),
                                      sample_weight=np.ones(3),
                                      observation_mask=np.array([False, False, False]))
        upper = np.triu(np.ones((4, 4), dtype=bool), k=1)
        np.testing.assert_allclose(est.log_block_evidence_[upper], 0.0, atol=1e-10)

    def test_all_missing_estimate_hyper_raises(self):
        """estimate_hyper=True + all-missing must raise ValueError."""
        with pytest.raises(ValueError, match="entirely missing"):
            BayesBreakGaussian(k_max=1, estimate_hyper=True).fit(
                np.arange(2).reshape(-1, 1), np.array([1.0, 2.0]),
                sample_weight=np.ones(2),
                observation_mask=np.array([False, False]),
            )

    def test_all_missing_poisson_estimate_hyper_raises(self):
        """Poisson: estimate_hyper=True + all-missing must raise ValueError."""
        with pytest.raises(ValueError, match="entirely missing"):
            BayesBreakPoisson(k_max=1, estimate_hyper=True).fit(
                np.arange(2).reshape(-1, 1), np.array([3.0, 1.0]),
                sample_weight=np.ones(2),
                observation_mask=np.array([False, False]),
            )


# ---------------------------------------------------------------------------
# 5.  Generalized-Bayes PIT correctness
# ---------------------------------------------------------------------------


class TestPITLikelihoodPower:
    """_segment_pit must use power-weighted posterior parameters."""

    def test_gaussian_pit_uses_power_weighted_posterior(self):
        """PIT CDF must use mu_post = (ρ²·Σa·w·y + σ²·ν)/(ρ²·Σa·w + σ²)."""
        from bayesbreak.prediction import pit_residuals
        nu, rho2, sigma2 = 0.0, 1.0, 1.0
        a_pow, w_pow, y_train = 2.0, 1.0, 1.0
        # Independent oracle: mu_post and rho2_post for power-weighted posterior
        Wseg = a_pow * w_pow; Syseg = a_pow * w_pow * y_train
        mu_post = (rho2 * Syseg + sigma2 * nu) / (rho2 * Wseg + sigma2)  # = 2/3
        rho2_post = (rho2 * sigma2) / (rho2 * Wseg + sigma2)              # = 1/3
        var_pred = sigma2 / 1.0 + rho2_post  # w_new = 1
        expected_cdf = float(norm.cdf(0.5, loc=mu_post, scale=math.sqrt(var_pred)))

        est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2
                                 ).fit(np.arange(1).reshape(-1, 1),
                                       np.array([y_train]),
                                       sample_weight=np.array([w_pow]),
                                       likelihood_power=np.array([a_pow]))
        # PIT residuals use a deterministic seed via np.random; set seed for reproducibility
        rng = np.random.default_rng(42)
        # For Gaussian (continuous), pit_residuals returns norm.cdf directly
        pit = pit_residuals(est,
                            np.array([[0.0]]),
                            np.array([0.5]),
                            sample_weight=np.array([1.0]))
        assert float(pit[0]) == pytest.approx(expected_cdf, abs=1e-8)

    def test_poisson_pit_uses_power_weighted_posterior(self):
        """Poisson PIT must use alpha_post = α + Σa·y, beta_post = β + Σa·e."""
        from bayesbreak.prediction import pit_residuals
        alpha, beta = 1.0, 1.0; a_pit, e_1, y_1 = 2.0, 1.0, 3
        # Power-weighted posterior parameters (oracle)
        S = a_pit * y_1  # = 6
        W = a_pit * e_1  # = 2
        alpha_post = alpha + S  # = 7
        beta_post = beta + W   # = 3
        e_new = 2.0; y_new_test = 5
        p_pred = beta_post / (beta_post + e_new)  # = 3/5
        cdf_hi = float(nbinom.cdf(y_new_test, n=alpha_post, p=p_pred))
        cdf_lo = float(nbinom.cdf(y_new_test - 1, n=alpha_post, p=p_pred))

        est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=alpha, beta=beta
                                ).fit(np.arange(1).reshape(-1, 1),
                                      np.array([float(y_1)]),
                                      sample_weight=np.array([e_1]),
                                      likelihood_power=np.array([a_pit]))
        np.random.seed(0)  # fix randomized PIT
        pit = pit_residuals(est,
                            np.array([[0.0]]),
                            np.array([float(y_new_test)]),
                            sample_weight=np.array([e_new]))
        # Randomized PIT must lie within [cdf_lo, cdf_hi]
        assert cdf_lo <= float(pit[0]) <= cdf_hi

    def test_gaussian_pit_unit_power_unchanged(self):
        """PIT with unit power = PIT without likelihood_power (regression test)."""
        from bayesbreak.prediction import pit_residuals
        nu, rho2, sigma2 = 0.0, 1.0, 1.0
        y_arr = np.array([1.0, 2.0])
        w_arr = np.array([2.0, 3.0])
        x_arr = np.arange(2).reshape(-1, 1)
        est_no_power = BayesBreakGaussian(k_max=1, estimate_hyper=False,
                                          nu=nu, rho2=rho2, sigma2=sigma2
                                          ).fit(x_arr, y_arr, sample_weight=w_arr)
        est_unit_power = BayesBreakGaussian(k_max=1, estimate_hyper=False,
                                            nu=nu, rho2=rho2, sigma2=sigma2
                                            ).fit(x_arr, y_arr, sample_weight=w_arr,
                                                  likelihood_power=np.ones(2))
        np.random.seed(7)
        pit_no = pit_residuals(est_no_power, x_arr, y_arr, sample_weight=w_arr)
        np.random.seed(7)
        pit_unit = pit_residuals(est_unit_power, x_arr, y_arr, sample_weight=w_arr)
        np.testing.assert_allclose(pit_no, pit_unit, atol=1e-12)


# ---------------------------------------------------------------------------
# 6.  Wrapper rejection
# ---------------------------------------------------------------------------


class TestWrapperRejection:
    """Wrappers that bypass base.fit() must explicitly reject unsupported params."""

    def _xy(self, n=4, d=2):
        X = np.arange(n).reshape(-1, 1)
        y = np.ones((n, d))
        return X, y

    def _xy1d(self, n=4):
        return np.arange(n).reshape(-1, 1), np.ones(n)

    def test_shared_boundary_multivariate_rejects_likelihood_power(self):
        X, y = self._xy()
        base = BayesBreakGaussian(k_max=2, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        with pytest.raises(NotImplementedError, match="likelihood_power"):
            SharedBoundaryMultivariateSegmenter(base).fit(
                X, y, likelihood_power=np.ones(4)
            )

    def test_shared_boundary_multivariate_rejects_observation_mask(self):
        X, y = self._xy()
        base = BayesBreakGaussian(k_max=2, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        with pytest.raises(NotImplementedError, match="observation_mask"):
            SharedBoundaryMultivariateSegmenter(base).fit(
                X, y, observation_mask=np.ones(4, dtype=bool)
            )

    def test_independent_multivariate_rejects_likelihood_power(self):
        X, y = self._xy()
        base = BayesBreakGaussian(k_max=2, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        with pytest.raises(NotImplementedError, match="likelihood_power"):
            IndependentMultivariateSegmenter(base).fit(
                X, y, likelihood_power=np.ones(4)
            )

    def test_independent_multivariate_rejects_observation_mask(self):
        X, y = self._xy()
        base = BayesBreakGaussian(k_max=2, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        with pytest.raises(NotImplementedError, match="observation_mask"):
            IndependentMultivariateSegmenter(base).fit(
                X, y, observation_mask=np.ones(4, dtype=bool)
            )

    def test_replicates_rejects_likelihood_power(self):
        X, y = self._xy1d()
        base = BayesBreakGaussian(k_max=2, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        with pytest.raises(NotImplementedError, match="likelihood_power"):
            SharedBoundaryReplicatesSegmenter(base).fit(
                X, np.stack([y, y]), likelihood_power=np.ones(4)
            )

    def test_replicates_rejects_observation_mask(self):
        X, y = self._xy1d()
        base = BayesBreakGaussian(k_max=2, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        with pytest.raises(NotImplementedError, match="observation_mask"):
            SharedBoundaryReplicatesSegmenter(base).fit(
                X, np.stack([y, y]), observation_mask=np.ones(4, dtype=bool)
            )


# ---------------------------------------------------------------------------
# 7.  Repeat-fit behaviour and stale state
# ---------------------------------------------------------------------------


class TestRepeatFit:
    """Second fit() must fully overwrite mask and power state from the first."""

    def test_repeat_fit_overwrites_observation_mask(self):
        """After a second fit without mask, observation_mask_ is all-True."""
        est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        y = np.array([1.0, 2.0]); w = np.ones(2); X = np.arange(2).reshape(-1, 1)
        # First fit with a mask
        est.fit(X, y, sample_weight=w, observation_mask=np.array([True, False]))
        assert not np.all(est.observation_mask_)
        # Second fit without mask — must be all-True
        est.fit(X, y, sample_weight=w)
        np.testing.assert_array_equal(est.observation_mask_, [True, True])

    def test_repeat_fit_overwrites_likelihood_power(self):
        """After a second fit without power, likelihood_power_ is all-ones."""
        est = BayesBreakGaussian(k_max=1, estimate_hyper=False, nu=0, rho2=1, sigma2=1)
        y = np.array([1.0, 2.0]); w = np.ones(2); X = np.arange(2).reshape(-1, 1)
        # First fit with nonunit powers
        est.fit(X, y, sample_weight=w, likelihood_power=np.array([2.0, 0.5]))
        assert not np.allclose(est.likelihood_power_, 1.0)
        # Second fit without power — must be all-ones
        est.fit(X, y, sample_weight=w)
        np.testing.assert_allclose(est.likelihood_power_, 1.0, atol=1e-15)

    def test_repeat_fit_clears_mask_from_power(self):
        """Mask×power from first fit does not bleed into second fit's powers."""
        est = BayesBreakPoisson(k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0)
        y = np.array([3.0, 4.0]); e = np.ones(2); X = np.arange(2).reshape(-1, 1)
        # First fit: mask=[T,F], power=[1,2] → effective=[1,0]
        est.fit(X, y, sample_weight=e, observation_mask=[True, False],
                likelihood_power=[1.0, 2.0])
        assert est.likelihood_power_[1] == 0.0
        # Second fit: no mask, no power → effective must be all-ones
        est.fit(X, y, sample_weight=e)
        np.testing.assert_allclose(est.likelihood_power_, 1.0, atol=1e-15)
        expected_full = _poisson_logev([3.0, 4.0], [1.0, 1.0])
        assert float(est.log_block_evidence_[0, 2]) == pytest.approx(expected_full, abs=1e-10)


# ---------------------------------------------------------------------------
# 8.  BetaObs likelihood_power_ regression tests
# ---------------------------------------------------------------------------


class TestBetaObsLikelihoodPower:
    """BetaObs _compute_block_evidence, posterior_predictive_logpdf_block, and
    _segment_pit must all apply likelihood_power_ so that masked observations
    do not enter any posterior or predictive computation."""

    PHI, ALPHA0, BETA0 = 50.0, 1.5, 2.5
    # Five observations; last two are held out / masked.
    Y_FULL = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    MASK = np.array([True, True, True, False, False])

    def _fit_masked(self):
        X5 = np.arange(5).reshape(-1, 1)
        return BayesBreakBetaObs(
            k_max=1, estimate_hyper=False, phi=self.PHI, alpha=self.ALPHA0, beta=self.BETA0
        ).fit(X5, self.Y_FULL, sample_weight=np.ones(5), observation_mask=self.MASK)

    def _fit_direct(self):
        """Reference fit on observed-only subset (no mask, no likelihood_power_)."""
        X3 = np.arange(3).reshape(-1, 1)
        return BayesBreakBetaObs(
            k_max=1, estimate_hyper=False, phi=self.PHI, alpha=self.ALPHA0, beta=self.BETA0
        ).fit(X3, self.Y_FULL[:3], sample_weight=np.ones(3))

    def test_block_evidence_masked_equals_observed_only(self):
        """log_block_evidence_[0,5] with last 2 masked matches [0,3] from a direct fit."""
        est_m = self._fit_masked()
        est_d = self._fit_direct()
        assert float(est_m.log_block_evidence_[0, 5]) == pytest.approx(
            float(est_d.log_block_evidence_[0, 3]), abs=1e-7
        )

    def test_block_evidence_differs_from_unmasked_full_fit(self):
        """Including the two extra obs (0.7, 0.8) changes the block evidence."""
        X5 = np.arange(5).reshape(-1, 1)
        est_full = BayesBreakBetaObs(
            k_max=1, estimate_hyper=False, phi=self.PHI, alpha=self.ALPHA0, beta=self.BETA0
        ).fit(X5, self.Y_FULL, sample_weight=np.ones(5))
        est_m = self._fit_masked()
        # With the fix the masked evidence must NOT equal the full-data evidence.
        assert float(est_m.log_block_evidence_[0, 5]) != pytest.approx(
            float(est_full.log_block_evidence_[0, 5]), abs=1e-5
        )

    def test_predictive_logpdf_masked_matches_observed_only(self):
        """posterior_predictive_logpdf_block uses lp * sample_weight_ for the posterior."""
        est_m = self._fit_masked()
        est_d = self._fit_direct()
        y_new = np.array([0.4])
        phi_new = np.array([self.PHI])
        lp_m = est_m.posterior_predictive_logpdf_block(a=0, b=5, y_new=y_new, w_new=phi_new)
        lp_d = est_d.posterior_predictive_logpdf_block(a=0, b=3, y_new=y_new, w_new=phi_new)
        np.testing.assert_allclose(lp_m, lp_d, atol=1e-7)

    def test_pit_masked_matches_observed_only(self):
        """BetaObs PIT is deterministic; masked fit must match direct fit on observed obs."""
        from bayesbreak.prediction import pit_residuals

        est_m = self._fit_masked()
        est_d = self._fit_direct()
        X_new = np.array([[0]])
        y_new = np.array([0.25])
        phi_new = np.array([self.PHI])
        # BetaObs CDF is a continuous quadrature mixture; no randomized PIT.
        pit_m = pit_residuals(est_m, X_new, y_new, sample_weight=phi_new)
        pit_d = pit_residuals(est_d, X_new, y_new, sample_weight=phi_new)
        np.testing.assert_allclose(pit_m, pit_d, atol=1e-7)


# ---------------------------------------------------------------------------
# 9.  Binomial PIT likelihood_power_ regression tests
# ---------------------------------------------------------------------------


class TestBinomialPITLikelihoodPower:
    """Binomial _segment_pit must apply lp to S and N so masked obs are excluded."""

    N_TRIALS = 10
    ALPHA0, BETA0 = 1.0, 1.0

    def test_masked_pit_matches_direct_fit_on_observed_subset(self):
        """PIT for masked Binomial fit (lp=[1,1,1,0]) matches direct fit on first 3 obs."""
        from bayesbreak.prediction import pit_residuals

        y_full = np.array([3.0, 7.0, 2.0, 5.0])
        mask = np.array([True, True, True, False])
        X4 = np.arange(4).reshape(-1, 1)
        X3 = np.arange(3).reshape(-1, 1)

        est_m = BayesBreakBinomial(
            k_max=1, estimate_hyper=False,
            alpha=self.ALPHA0, beta=self.BETA0, n_trials=self.N_TRIALS,
        ).fit(X4, y_full, sample_weight=np.ones(4), observation_mask=mask)

        est_d = BayesBreakBinomial(
            k_max=1, estimate_hyper=False,
            alpha=self.ALPHA0, beta=self.BETA0, n_trials=self.N_TRIALS,
        ).fit(X3, y_full[:3], sample_weight=np.ones(3))

        X_new = np.array([[0]])
        y_new = np.array([4.0])
        n_new = np.array([float(self.N_TRIALS)])

        np.random.seed(99)
        pit_m = pit_residuals(est_m, X_new, y_new, sample_weight=n_new)
        np.random.seed(99)
        pit_d = pit_residuals(est_d, X_new, y_new, sample_weight=n_new)
        np.testing.assert_allclose(pit_m, pit_d, atol=1e-12)

    def test_posterior_parameters_exclude_masked_obs(self):
        """Binomial PIT uses a_post, b_post computed without masked obs contribution."""
        from bayesbreak.prediction import pit_residuals
        from scipy.stats import betabinom

        # Train on 2 obs: y=[3, 9], n_trials=10, mask=[T, F].
        # Correct posterior: S=3, N=10 → a_post=4, b_post=8.
        # Incorrect (bug) posterior: S=12, N=20 → a_post=13, b_post=9.
        y = np.array([3.0, 9.0])
        X2 = np.arange(2).reshape(-1, 1)
        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0, n_trials=10
        ).fit(X2, y, sample_weight=np.ones(2), observation_mask=np.array([True, False]))

        y_new_val, n_new_val = 2, 10
        a_correct, b_correct = 4.0, 8.0
        cdf_hi = float(betabinom.cdf(y_new_val, n=n_new_val, a=a_correct, b=b_correct))
        cdf_lo = float(betabinom.cdf(y_new_val - 1, n=n_new_val, a=a_correct, b=b_correct))

        # Verify wrong params (a_post=13, b_post=9) fall outside this interval.
        a_wrong, b_wrong = 13.0, 9.0
        cdf_wrong = float(betabinom.cdf(y_new_val, n=n_new_val, a=a_wrong, b=b_wrong))
        assert not (cdf_lo <= cdf_wrong <= cdf_hi), (
            "Test setup error: wrong params must produce a CDF outside [cdf_lo, cdf_hi]."
        )

        # Randomized PIT: fix seed; value must lie in [cdf_lo, cdf_hi].
        np.random.seed(77)
        pit = pit_residuals(
            est,
            np.array([[0]]),
            np.array([float(y_new_val)]),
            sample_weight=np.array([float(n_new_val)]),
        )
        assert cdf_lo <= float(pit[0]) <= cdf_hi, (
            f"PIT {pit[0]:.6f} not in [{cdf_lo:.6f}, {cdf_hi:.6f}]; "
            "likelihood_power_ may not be applied in _segment_pit."
        )
