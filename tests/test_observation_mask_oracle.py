"""Oracle tests for observation_mask correctness in Binomial and NegBin families.

Seven required oracle cases:
  O1. All-missing block: log A = 0.
  O2. No-missing block: matches unmasked result exactly.
  O3. Partial mask (alternating): matches closed-form on observed subset.
  O4. Heterogeneous descriptors (Binomial: varying n_trials; NegBin: varying r).
  O5. Fractional likelihood powers: matches analytic powered block integral.
  O6. Degenerate n=1, fully masked: log A = 0.
  O7. E07 workaround equivalence: sample_weight=mask vs observation_mask paths.

Additional tests:
  - Repeat-fit state safety.
  - estimate_hyper=True with observation_mask.
  - Posterior predictive consistent with masked training posterior.
  - Regression: previously valid unmasked results unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import betaln, gammaln as sp_gammaln

from bayesbreak import BayesBreakBinomial, BayesBreakBetaObs, BayesBreakNegBin


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _X(n: int) -> np.ndarray:
    return np.arange(n, dtype=float).reshape(-1, 1)


def _logB(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _binom_block_log_evidence(
    y: np.ndarray, n_arr: np.ndarray, alpha: float, beta: float,
    weights: np.ndarray | None = None,
) -> float:
    """Closed-form log A^0 for a single Binomial block with optional effective weights."""
    w = np.ones(y.size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    S = float(np.sum(w * y))
    F = float(np.sum(w * (n_arr - y)))
    Lcomb = float(np.sum(w * (sp_gammaln(n_arr + 1) - sp_gammaln(y + 1) - sp_gammaln(n_arr - y + 1))))
    return Lcomb + _logB(alpha + S, beta + F) - _logB(alpha, beta)


def _negbin_block_log_evidence(
    y: np.ndarray, r_arr: np.ndarray, alpha: float, beta: float,
    weights: np.ndarray | None = None,
) -> float:
    """Closed-form log A^0 for a single NegBin block with optional effective weights."""
    w = np.ones(y.size, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    C = float(np.sum(w * y))
    N = float(np.sum(w * r_arr))
    H = float(np.sum(w * (sp_gammaln(y + r_arr) - sp_gammaln(y + 1) - sp_gammaln(r_arr))))
    return H + _logB(alpha + N, beta + C) - _logB(alpha, beta)


# ---------------------------------------------------------------------------
# O1: All-missing block
# ---------------------------------------------------------------------------

class TestO1AllMissing:
    """All observations masked -> log block evidence = 0."""

    def test_binomial_all_missing_single_block(self):
        n, alpha, beta = 15, 2.0, 3.0
        rng = np.random.default_rng(42)
        y = rng.binomial(10, 0.4, size=n).astype(float)
        mask = np.zeros(n, dtype=bool)  # all masked

        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=10
        ).fit(_X(n), y, observation_mask=mask)

        # A single fully-masked block contributes zero to every suffix sum,
        # so log A[0, n] = 0 + logB(alpha, beta) - logB(alpha, beta) = 0.
        assert est.log_block_evidence_[0, n] == pytest.approx(0.0, abs=1e-12)

    def test_negbin_all_missing_single_block(self):
        n, r, alpha, beta = 15, 3.0, 2.0, 3.0
        rng = np.random.default_rng(42)
        y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
        mask = np.zeros(n, dtype=bool)

        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, observation_mask=mask)

        assert est.log_block_evidence_[0, n] == pytest.approx(0.0, abs=1e-12)

    def test_binomial_all_missing_k_max_3(self):
        """With all observations masked every block scores 0; fit succeeds and
        k_posterior is a proper distribution."""
        n, alpha, beta = 20, 1.0, 1.0
        y = np.zeros(n, dtype=float)
        mask = np.zeros(n, dtype=bool)

        est = BayesBreakBinomial(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=5
        ).fit(_X(n), y, observation_mask=mask)

        # Every block evidence = 1 (log A = 0).  The posterior over k is still
        # well-defined (proportional to C_k * prior(k)), sums to 1.
        assert abs(float(np.sum(est.k_posterior_)) - 1.0) < 1e-10
        assert np.all(est.k_posterior_ >= 0)


# ---------------------------------------------------------------------------
# O2: No-missing block matches unmasked
# ---------------------------------------------------------------------------

class TestO2NoMissing:
    """All-True mask is identical to no mask."""

    def test_binomial_all_observed_mask_equals_no_mask(self):
        n, alpha, beta = 12, 2.0, 1.5
        rng = np.random.default_rng(7)
        y = rng.binomial(8, 0.3, size=n).astype(float)
        mask = np.ones(n, dtype=bool)

        est_nomask = BayesBreakBinomial(
            k_max=4, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=8
        ).fit(_X(n), y)
        est_mask = BayesBreakBinomial(
            k_max=4, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=8
        ).fit(_X(n), y, observation_mask=mask)

        np.testing.assert_allclose(
            est_mask.log_block_evidence_, est_nomask.log_block_evidence_, atol=1e-12
        )
        np.testing.assert_allclose(
            est_mask.k_posterior_, est_nomask.k_posterior_, atol=1e-12
        )

    def test_negbin_all_observed_mask_equals_no_mask(self):
        n, r, alpha, beta = 10, 4.0, 2.0, 2.0
        rng = np.random.default_rng(8)
        y = rng.negative_binomial(int(r), 0.5, size=n).astype(float)
        mask = np.ones(n, dtype=bool)

        est_nomask = BayesBreakNegBin(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y)
        est_mask = BayesBreakNegBin(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, observation_mask=mask)

        np.testing.assert_allclose(
            est_mask.log_block_evidence_, est_nomask.log_block_evidence_, atol=1e-12
        )


# ---------------------------------------------------------------------------
# O3: Partial mask matches closed-form on observed subset
# ---------------------------------------------------------------------------

class TestO3PartialMask:
    """Alternating mask: block [0,n] score equals closed-form over observed obs."""

    def test_binomial_alternating_mask_single_block(self):
        n, m_trials, alpha, beta = 16, 6, 2.0, 3.0
        rng = np.random.default_rng(99)
        y = rng.binomial(m_trials, 0.35, size=n).astype(float)
        mask = np.arange(n) % 2 == 0  # even indices observed

        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, observation_mask=mask)

        expected = _binom_block_log_evidence(
            y[mask], np.full(mask.sum(), float(m_trials)), alpha, beta
        )
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_negbin_alternating_mask_single_block(self):
        n, r, alpha, beta = 14, 2.0, 2.0, 4.0
        rng = np.random.default_rng(100)
        y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
        mask = np.arange(n) % 2 == 0

        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, observation_mask=mask)

        expected = _negbin_block_log_evidence(
            y[mask], np.full(mask.sum(), r), alpha, beta
        )
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_binomial_mask_matches_sample_weight_workaround(self):
        """observation_mask path equals sample_weight={0,1} workaround path."""
        n, m_trials, alpha, beta = 20, 10, 2.0, 3.0
        rng = np.random.default_rng(55)
        y = rng.binomial(m_trials, 0.4, size=n).astype(float)
        mask = rng.random(n) > 0.25  # ~75% observed

        est_mask = BayesBreakBinomial(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, observation_mask=mask)

        est_sw = BayesBreakBinomial(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, sample_weight=mask.astype(float))

        np.testing.assert_allclose(
            est_mask.log_block_evidence_, est_sw.log_block_evidence_, atol=1e-10
        )

    def test_negbin_mask_matches_sample_weight_workaround(self):
        n, r, alpha, beta = 18, 3.0, 2.0, 4.0
        rng = np.random.default_rng(56)
        y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
        mask = rng.random(n) > 0.3

        est_mask = BayesBreakNegBin(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, observation_mask=mask)

        est_sw = BayesBreakNegBin(
            k_max=3, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, sample_weight=mask.astype(float))

        np.testing.assert_allclose(
            est_mask.log_block_evidence_, est_sw.log_block_evidence_, atol=1e-10
        )


# ---------------------------------------------------------------------------
# O4: Heterogeneous descriptors
# ---------------------------------------------------------------------------

class TestO4HeterogeneousDescriptors:
    """Per-observation n_trials (Binomial) and r (NegBin) are correctly indexed."""

    def test_binomial_heterogeneous_n_trials_single_block(self):
        n, alpha, beta = 10, 1.5, 2.5
        rng = np.random.default_rng(20)
        n_arr = rng.integers(3, 15, size=n).astype(float)
        y = np.array([rng.binomial(int(m), 0.4) for m in n_arr], dtype=float)

        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=n_arr
        ).fit(_X(n), y)

        expected = _binom_block_log_evidence(y, n_arr, alpha, beta)
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_binomial_heterogeneous_n_trials_with_mask(self):
        n, alpha, beta = 12, 1.5, 2.5
        rng = np.random.default_rng(21)
        n_arr = rng.integers(5, 20, size=n).astype(float)
        y = np.array([rng.binomial(int(m), 0.3) for m in n_arr], dtype=float)
        mask = np.arange(n) % 3 != 0  # skip every third

        # With array n_trials and observation_mask, estimate_hyper must be False
        # (estimate_hyper=True with array n_trials requires hyperparameter
        # estimation from y_obs which cannot index into the full n_arr).
        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=n_arr
        ).fit(_X(n), y, observation_mask=mask)

        expected = _binom_block_log_evidence(y[mask], n_arr[mask], alpha, beta)
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_negbin_heterogeneous_r_single_block(self):
        n, alpha, beta = 10, 2.0, 3.0
        rng = np.random.default_rng(22)
        r_arr = rng.uniform(1.0, 5.0, size=n)
        y = np.array(
            [rng.negative_binomial(max(1, int(r)), 0.4) for r in r_arr], dtype=float
        )

        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r_arr
        ).fit(_X(n), y)

        expected = _negbin_block_log_evidence(y, r_arr, alpha, beta)
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_negbin_heterogeneous_r_with_mask(self):
        n, alpha, beta = 12, 2.0, 3.0
        rng = np.random.default_rng(23)
        r_arr = rng.uniform(0.5, 4.0, size=n)
        y = np.array(
            [rng.negative_binomial(max(1, int(r)), 0.45) for r in r_arr], dtype=float
        )
        mask = rng.random(n) > 0.4

        # estimate_hyper=False required with array r and observation_mask
        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r_arr
        ).fit(_X(n), y, observation_mask=mask)

        expected = _negbin_block_log_evidence(y[mask], r_arr[mask], alpha, beta)
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# O5: Fractional likelihood powers
# ---------------------------------------------------------------------------

class TestO5FractionalPowers:
    """likelihood_power=0.5 applies the half-power to all sufficient statistics."""

    def test_binomial_half_power_single_block(self):
        n, m_trials, alpha, beta = 10, 8, 2.0, 3.0
        rng = np.random.default_rng(30)
        y = rng.binomial(m_trials, 0.35, size=n).astype(float)
        a_pow = np.full(n, 0.5)

        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, likelihood_power=a_pow)

        expected = _binom_block_log_evidence(
            y, np.full(n, float(m_trials)), alpha, beta, weights=a_pow
        )
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_negbin_half_power_single_block(self):
        n, r, alpha, beta = 10, 3.0, 2.0, 3.0
        rng = np.random.default_rng(31)
        y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
        a_pow = np.full(n, 0.5)

        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, likelihood_power=a_pow)

        expected = _negbin_block_log_evidence(
            y, np.full(n, r), alpha, beta, weights=a_pow
        )
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_binomial_zero_power_equals_all_missing(self):
        """likelihood_power = 0 for all obs should give same result as all-masked."""
        n, m_trials, alpha, beta = 8, 5, 1.5, 2.0
        rng = np.random.default_rng(32)
        y = rng.binomial(m_trials, 0.5, size=n).astype(float)

        est_lp = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, likelihood_power=np.zeros(n))

        # All-missing block: mask=False everywhere
        mask = np.zeros(n, dtype=bool)
        est_mask = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, observation_mask=mask)

        assert est_lp.log_block_evidence_[0, n] == pytest.approx(
            est_mask.log_block_evidence_[0, n], abs=1e-12
        )


# ---------------------------------------------------------------------------
# O6: Degenerate n=1, fully masked
# ---------------------------------------------------------------------------

class TestO6DegenerateFullyMasked:
    """n=1 with observation_mask=[False]: log A[0, 1] = 0."""

    def test_binomial_n1_masked(self):
        y = np.array([7.0])
        mask = np.array([False])
        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=1.0, beta=1.0, n_trials=10
        ).fit(_X(1), y, observation_mask=mask)
        assert est.log_block_evidence_[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_negbin_n1_masked(self):
        y = np.array([5.0])
        mask = np.array([False])
        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=2.0
        ).fit(_X(1), y, observation_mask=mask)
        assert est.log_block_evidence_[0, 1] == pytest.approx(0.0, abs=1e-12)

    def test_binomial_n1_observed_matches_closed_form(self):
        """Unmasked n=1 matches the single-observation closed form."""
        y = np.array([3.0])
        m_trials, alpha, beta = 8.0, 2.0, 3.0
        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(1), y)
        expected = _binom_block_log_evidence(y, np.array([m_trials]), alpha, beta)
        assert est.log_block_evidence_[0, 1] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# O7: E07 workaround equivalence (all six affected cells)
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4) * (k - 1))


def _generate_overdispersed_poisson(n: int, k_true: int, bps: list[int], means: list[float],
                                     r_gen: float, rng: np.random.Generator) -> np.ndarray:
    y = np.empty(n, dtype=float)
    segs = [(0, bps[0]), (bps[0], bps[1]), (bps[1], n)]
    for (s, e), mu in zip(segs, means):
        p_nb = r_gen / (r_gen + mu)
        y[s:e] = rng.negative_binomial(int(r_gen), p_nb, size=e - s)
    return y


def _generate_beta_binomial(n: int, bps: list[int], means: list[float],
                              m_trials: int, icc: float, rng: np.random.Generator) -> np.ndarray:
    y = np.empty(n, dtype=float)
    phi = (1 - icc) / icc
    segs = [(0, bps[0]), (bps[0], bps[1]), (bps[1], n)]
    for (s, e), mu in zip(segs, means):
        a_i, b_i = mu * phi, (1 - mu) * phi
        p = rng.beta(a_i, b_i, size=e - s)
        y[s:e] = rng.binomial(m_trials, p, size=e - s)
    return y


def _apply_missingness(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    if kind == "mcar_10":
        miss = rng.random(n) < 0.10
        mask[miss] = False
    elif kind == "mcar_30":
        miss = rng.random(n) < 0.30
        mask[miss] = False
    elif kind == "single_block_15":
        start = rng.integers(0, n - 15)
        mask[start:start + 15] = False
    return mask


_E07_CELLS = [
    # (data_family, fit_family, missingness, cell_index)
    # Cell indices from the E07 manifest (cells 12-14 = negbin missingness,
    # cells 17-19 = binomial missingness).
    ("overdispersed_poisson", "negbin", "mcar_10", 12),
    ("overdispersed_poisson", "negbin", "mcar_30", 13),
    ("overdispersed_poisson", "negbin", "single_block_15", 14),
    ("beta_binomial", "binomial", "mcar_10", 17),
    ("beta_binomial", "binomial", "mcar_30", 18),
    ("beta_binomial", "binomial", "single_block_15", 19),
]


@pytest.mark.parametrize("data_family,fit_family,missingness,cell_idx", _E07_CELLS)
def test_O7_e07_workaround_equivalence(data_family, fit_family, missingness, cell_idx):
    """Corrected observation_mask path equals E07 sample_weight workaround.

    For each of the six affected E07 cells, verify that:
    - Block log-evidence matrices agree to within floating-point tolerance.
    - k_posterior distributions agree.
    - MAP boundary sets agree.
    """
    n = 256
    k_max = 10
    bps = [85, 170]
    # Parameters from E07_CONFIG in run_e07_misspecification.py
    means_p = [2.0, 5.0, 2.0]    # NegBin count means per segment
    means_bb = [0.2, 0.6, 0.2]   # BetaBinomial success probs per segment
    m_trials = 20
    icc = 0.1
    r_gen = 2.0
    alpha_nb, beta_nb = 2.0, 3.0
    alpha_bb, beta_bb = 2.0, 3.0
    k_model = 10

    # Use 5 representative replicates for speed (full cell sweep is in the runner).
    for rep in range(5):
        seed = np.random.SeedSequence([20261021, 7, cell_idx, rep])
        # Use spawn(2) to match the E07 runner: children[0]=data, children[1]=miss.
        children = seed.spawn(2)
        rng_data = np.random.default_rng(children[0])
        rng_miss = np.random.default_rng(children[1])

        if data_family == "overdispersed_poisson":
            y = _generate_overdispersed_poisson(n, 3, bps, means_p, r_gen, rng_data)
        else:
            y = _generate_beta_binomial(n, bps, means_bb, m_trials, icc, rng_data)

        obs_mask = _apply_missingness(missingness, n, rng_miss)
        X = np.arange(n, dtype=float).reshape(-1, 1)

        if fit_family == "negbin":
            # Workaround path (original E07): sample_weight = 0/1 mask, no observation_mask
            est_sw = BayesBreakNegBin(
                estimate_hyper=False, r=r_gen, alpha=alpha_nb, beta=beta_nb,
                k_max=k_model, regression_curve="none", prior_k=_geom_prior
            ).fit(X, y, sample_weight=obs_mask.astype(float))
            # Corrected path: observation_mask, sample_weight=default ones
            est_mask = BayesBreakNegBin(
                estimate_hyper=False, r=r_gen, alpha=alpha_nb, beta=beta_nb,
                k_max=k_model, regression_curve="none", prior_k=_geom_prior
            ).fit(X, y, observation_mask=obs_mask)
        else:
            # Workaround path
            est_sw = BayesBreakBinomial(
                estimate_hyper=False, n_trials=m_trials, alpha=alpha_bb, beta=beta_bb,
                k_max=k_model, regression_curve="none", prior_k=_geom_prior
            ).fit(X, y, sample_weight=obs_mask.astype(float))
            # Corrected path
            est_mask = BayesBreakBinomial(
                estimate_hyper=False, n_trials=m_trials, alpha=alpha_bb, beta=beta_bb,
                k_max=k_model, regression_curve="none", prior_k=_geom_prior
            ).fit(X, y, observation_mask=obs_mask)

        # Block log-evidence matrices must agree to floating-point precision.
        fin_sw = np.isfinite(est_sw.log_block_evidence_)
        fin_mask = np.isfinite(est_mask.log_block_evidence_)
        assert (fin_sw == fin_mask).all(), (
            f"Finite-mask mismatch: cell {cell_idx}, rep {rep}"
        )
        np.testing.assert_allclose(
            est_mask.log_block_evidence_[fin_mask],
            est_sw.log_block_evidence_[fin_sw],
            atol=1e-10,
            err_msg=f"log_block_evidence_ mismatch: cell={cell_idx}, rep={rep}, "
                    f"missingness={missingness}",
        )

        # k_posterior must agree.
        np.testing.assert_allclose(
            est_mask.k_posterior_, est_sw.k_posterior_, atol=1e-10,
            err_msg=f"k_posterior_ mismatch: cell={cell_idx}, rep={rep}",
        )

        # MAP boundaries must agree.
        assert est_mask.map_boundaries_ == est_sw.map_boundaries_, (
            f"MAP boundaries mismatch: cell={cell_idx}, rep={rep}"
        )


# ---------------------------------------------------------------------------
# Repeat-fit state safety
# ---------------------------------------------------------------------------

class TestRepeatFitStateSafety:
    """Fitting twice on different data gives consistent results (no state leak)."""

    def test_binomial_repeat_fit(self):
        rng = np.random.default_rng(200)
        n = 15
        y1 = rng.binomial(10, 0.3, size=n).astype(float)
        y2 = rng.binomial(10, 0.7, size=n).astype(float)

        est = BayesBreakBinomial(k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, n_trials=10)
        est.fit(_X(n), y1)
        lA0_first = est.log_block_evidence_[0, n]

        est.fit(_X(n), y2)
        lA0_second = est.log_block_evidence_[0, n]

        # Verify second result matches an independent estimator on y2.
        est2 = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, n_trials=10
        ).fit(_X(n), y2)
        assert lA0_second == pytest.approx(est2.log_block_evidence_[0, n], abs=1e-12)
        assert lA0_first != pytest.approx(lA0_second, abs=1e-4)

    def test_negbin_repeat_fit(self):
        rng = np.random.default_rng(201)
        n = 12
        y1 = rng.negative_binomial(3, 0.4, size=n).astype(float)
        y2 = rng.negative_binomial(3, 0.7, size=n).astype(float)

        est = BayesBreakNegBin(k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=3.0)
        est.fit(_X(n), y1)
        lA0_first = est.log_block_evidence_[0, n]

        est.fit(_X(n), y2)
        lA0_second = est.log_block_evidence_[0, n]

        est2 = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=3.0
        ).fit(_X(n), y2)
        assert lA0_second == pytest.approx(est2.log_block_evidence_[0, n], abs=1e-12)

    def test_binomial_repeat_fit_with_mask_then_no_mask(self):
        """Fitting with mask then without should not carry over _n_arr_ size."""
        rng = np.random.default_rng(202)
        n = 10
        y = rng.binomial(5, 0.4, size=n).astype(float)
        mask = np.arange(n) % 2 == 0  # 5 observed

        est = BayesBreakBinomial(k_max=1, estimate_hyper=False, alpha=1.5, beta=2.0, n_trials=5)
        est.fit(_X(n), y, observation_mask=mask)
        assert est._n_arr_.size == n  # must be n after masked fit

        est.fit(_X(n), y)  # unmasked second fit
        assert est._n_arr_.size == n

        expected = _binom_block_log_evidence(y, np.full(n, 5.0), 1.5, 2.0)
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-10)

    def test_negbin_repeat_fit_with_mask_then_no_mask(self):
        rng = np.random.default_rng(203)
        n = 10
        y = rng.negative_binomial(4, 0.5, size=n).astype(float)
        mask = np.arange(n) % 2 == 0

        est = BayesBreakNegBin(k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=4.0)
        est.fit(_X(n), y, observation_mask=mask)
        assert est._r_arr_.size == n

        est.fit(_X(n), y)
        assert est._r_arr_.size == n


# ---------------------------------------------------------------------------
# estimate_hyper=True with observation_mask
# ---------------------------------------------------------------------------

class TestEstimateHyperWithMask:
    """When estimate_hyper=True, hyperparameters use only observed data."""

    def test_binomial_estimate_hyper_scalar_n_trials(self):
        """estimate_hyper=True with scalar n_trials and observation_mask runs without error."""
        rng = np.random.default_rng(300)
        n, m_trials = 30, 8
        y = rng.binomial(m_trials, 0.4, size=n).astype(float)
        mask = rng.random(n) > 0.2  # ~80% observed

        est = BayesBreakBinomial(k_max=3, estimate_hyper=True, n_trials=m_trials).fit(
            _X(n), y, observation_mask=mask
        )
        assert hasattr(est, "hyper_")
        assert est.hyper_["alpha"] > 0
        assert est.hyper_["beta"] > 0
        assert est._n_arr_.size == n

    def test_negbin_estimate_hyper_scalar_r(self):
        rng = np.random.default_rng(301)
        n, r = 30, 3.0
        y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
        mask = rng.random(n) > 0.2

        est = BayesBreakNegBin(k_max=3, estimate_hyper=True, r=r).fit(
            _X(n), y, observation_mask=mask
        )
        assert hasattr(est, "hyper_")
        assert est.hyper_["alpha"] > 0
        assert est.hyper_["beta"] > 0
        assert est._r_arr_.size == n


# ---------------------------------------------------------------------------
# Posterior predictive consistent with masked training posterior
# ---------------------------------------------------------------------------

class TestPosteriorPredictiveWithMask:
    """posterior_predictive_logpdf_block uses masked-corrected posterior parameters."""

    def test_binomial_predictive_uses_masked_posterior(self):
        n, m_trials, alpha, beta = 12, 6, 2.0, 3.0
        rng = np.random.default_rng(400)
        y = rng.binomial(m_trials, 0.35, size=n).astype(float)
        mask = np.arange(n) % 3 != 0  # skip every third

        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=m_trials
        ).fit(_X(n), y, observation_mask=mask)

        # Posterior parameters from observed data only.
        y_obs = y[mask]
        S_obs = float(y_obs.sum())
        N_obs = float(len(y_obs) * m_trials)
        alpha_post = alpha + S_obs
        beta_post = beta + (N_obs - S_obs)

        y_new = np.array([0.0, 2.0, 5.0])
        w_new = np.full(3, float(m_trials))
        log_pred = est.posterior_predictive_logpdf_block(a=0, b=n, y_new=y_new, w_new=w_new)

        expected = np.array([
            betaln(alpha_post + s, beta_post + (m_trials - s)) - betaln(alpha_post, beta_post)
            + sp_gammaln(m_trials + 1) - sp_gammaln(s + 1) - sp_gammaln(m_trials - s + 1)
            for s in y_new
        ])
        np.testing.assert_allclose(log_pred, expected, atol=1e-10)

    def test_negbin_predictive_uses_masked_posterior(self):
        n, r, alpha, beta = 10, 4.0, 2.0, 3.0
        rng = np.random.default_rng(401)
        y = rng.negative_binomial(int(r), 0.45, size=n).astype(float)
        mask = rng.random(n) > 0.25

        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y, observation_mask=mask)

        y_obs = y[mask]
        n_obs = len(y_obs)
        a_post = alpha + n_obs * r
        b_post = beta + float(y_obs.sum())

        y_new = np.array([0.0, 3.0, 8.0])
        log_pred = est.posterior_predictive_logpdf_block(
            a=0, b=n, y_new=y_new, w_new=np.ones_like(y_new)
        )
        expected = np.array([
            sp_gammaln(yi + r) - sp_gammaln(yi + 1) - math.lgamma(r)
            + betaln(a_post + r, b_post + yi) - betaln(a_post, b_post)
            for yi in y_new
        ])
        np.testing.assert_allclose(log_pred, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Regression: previously valid unmasked results unchanged
# ---------------------------------------------------------------------------

class TestRegressionUnmaskedUnchanged:
    """Existing unmasked oracle tests still pass after the fix."""

    def test_binomial_unmasked_closed_form(self):
        """Matches the closed form from the original test_families.py."""
        rng = np.random.default_rng(2)
        n, n_trials_val, alpha, beta = 8, 5, 1.0, 1.0
        y = rng.binomial(n_trials_val, 0.4, size=n)
        est = BayesBreakBinomial(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, n_trials=n_trials_val
        ).fit(_X(n), y.astype(float))

        S = int(y.sum())
        N = n * n_trials_val
        expected = (
            float(np.sum([math.lgamma(n_trials_val + 1) - math.lgamma(int(yi) + 1)
                          - math.lgamma(n_trials_val - int(yi) + 1) for yi in y]))
            + _logB(alpha + S, beta + (N - S)) - _logB(alpha, beta)
        )
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, rel=1e-9)

    def test_negbin_unmasked_closed_form(self):
        """Matches the closed form from test_negbin.py."""
        rng = np.random.default_rng(0)
        n, r, alpha, beta = 25, 4.0, 2.0, 3.0
        y = rng.negative_binomial(int(r), 0.5, size=n).astype(float)
        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y)

        expected = _negbin_block_log_evidence(y, np.full(n, r), alpha, beta)
        assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-9)

    def test_negbin_observation_mean_target(self):
        """A1 targets E[r(1-p)/p | y], not the Beta mean of p."""
        rng = np.random.default_rng(1)
        n, r, alpha, beta = 30, 3.0, 2.0, 3.0
        y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y)

        a_post = alpha + n * r
        b_post = beta + float(y.sum())
        expected_obs_mean = r * b_post / (a_post - 1.0)
        assert float(est.block_first_moment_[0, n]) == pytest.approx(expected_obs_mean, rel=1e-9)


# ---------------------------------------------------------------------------
# A1 fix: array-valued descriptors + observation_mask + estimate_hyper=True
# ---------------------------------------------------------------------------

class TestA1ArrayDescriptorWithMask:
    """Array-valued descriptors use the observed subset for empirical Bayes estimation."""

    def test_binomial_array_n_trials_with_mask_descriptor_size(self):
        """Array n_trials + observation_mask + estimate_hyper=True: _n_arr_ has full size."""
        rng = np.random.default_rng(500)
        n = 40
        n_trials_arr = rng.integers(4, 20, size=n).astype(float)
        y = np.array([rng.binomial(int(m), 0.4) for m in n_trials_arr], dtype=float)
        mask = rng.random(n) > 0.25

        est = BayesBreakBinomial(
            k_max=3, estimate_hyper=True, n_trials=n_trials_arr
        ).fit(_X(n), y, observation_mask=mask)

        assert est._n_arr_.size == n
        assert est.hyper_["alpha"] > 0
        assert est.hyper_["beta"] > 0

    def test_binomial_array_n_trials_hyper_matches_observed_subset(self):
        """Array n_trials: masked fit yields same hypers as fitting on observed subset."""
        rng = np.random.default_rng(501)
        n = 30
        n_trials_arr = rng.integers(4, 16, size=n).astype(float)
        y = np.array([rng.binomial(int(m), 0.35) for m in n_trials_arr], dtype=float)
        mask = rng.random(n) > 0.3

        est_mask = BayesBreakBinomial(
            k_max=3, estimate_hyper=True, n_trials=n_trials_arr
        ).fit(_X(n), y, observation_mask=mask)

        y_obs = y[mask]
        n_trials_obs = n_trials_arr[mask]
        est_obs = BayesBreakBinomial(
            k_max=3, estimate_hyper=True, n_trials=n_trials_obs
        ).fit(_X(int(mask.sum())), y_obs)

        assert est_mask.hyper_["alpha"] == pytest.approx(est_obs.hyper_["alpha"], rel=1e-9)
        assert est_mask.hyper_["beta"] == pytest.approx(est_obs.hyper_["beta"], rel=1e-9)

    def test_negbin_array_r_with_mask_descriptor_size(self):
        """Array r + observation_mask + estimate_hyper=True: _r_arr_ has full size."""
        rng = np.random.default_rng(510)
        n = 35
        r_arr = rng.uniform(1.0, 5.0, size=n)
        y = np.array([rng.negative_binomial(max(1, int(r)), 0.5) for r in r_arr], dtype=float)
        mask = rng.random(n) > 0.3

        est = BayesBreakNegBin(
            k_max=3, estimate_hyper=True, r=r_arr
        ).fit(_X(n), y, observation_mask=mask)

        assert est._r_arr_.size == n
        assert est.hyper_["alpha"] > 0
        assert est.hyper_["beta"] > 0

    def test_negbin_array_r_hyper_matches_observed_subset(self):
        """Array r: masked fit yields same hypers as fitting on observed subset."""
        rng = np.random.default_rng(511)
        n = 30
        r_arr = rng.uniform(1.0, 4.0, size=n)
        y = np.array([rng.negative_binomial(max(1, int(r)), 0.45) for r in r_arr], dtype=float)
        mask = rng.random(n) > 0.25

        est_mask = BayesBreakNegBin(
            k_max=3, estimate_hyper=True, r=r_arr
        ).fit(_X(n), y, observation_mask=mask)

        y_obs = y[mask]
        r_obs = r_arr[mask]
        est_obs = BayesBreakNegBin(
            k_max=3, estimate_hyper=True, r=r_obs
        ).fit(_X(int(mask.sum())), y_obs)

        assert est_mask.hyper_["alpha"] == pytest.approx(est_obs.hyper_["alpha"], rel=1e-9)
        assert est_mask.hyper_["beta"] == pytest.approx(est_obs.hyper_["beta"], rel=1e-9)

    def test_betaobs_array_phi_with_mask_descriptor_size(self):
        """Array phi + observation_mask + estimate_hyper=True: _phi_arr_ has full size."""
        rng = np.random.default_rng(520)
        n = 25
        phi_arr = rng.uniform(10.0, 80.0, size=n)
        mu_true = 0.4
        y = np.clip(
            np.array([rng.beta(phi * mu_true, phi * (1 - mu_true)) for phi in phi_arr]),
            1e-6, 1 - 1e-6,
        )
        mask = rng.random(n) > 0.3

        est = BayesBreakBetaObs(
            k_max=2, estimate_hyper=True, phi=phi_arr
        ).fit(_X(n), y, observation_mask=mask)

        assert est._phi_arr_.size == n
        assert est.hyper_["alpha"] > 0
        assert est.hyper_["beta"] > 0

    def test_betaobs_array_phi_hyper_matches_observed_subset(self):
        """Array phi: masked fit yields same hypers as fitting on observed subset."""
        rng = np.random.default_rng(521)
        n = 25
        phi_arr = rng.uniform(15.0, 60.0, size=n)
        mu_true = 0.35
        y = np.clip(
            np.array([rng.beta(phi * mu_true, phi * (1 - mu_true)) for phi in phi_arr]),
            1e-6, 1 - 1e-6,
        )
        mask = rng.random(n) > 0.3

        est_mask = BayesBreakBetaObs(
            k_max=2, estimate_hyper=True, phi=phi_arr
        ).fit(_X(n), y, observation_mask=mask)

        y_obs = y[mask]
        phi_obs = phi_arr[mask]
        est_obs = BayesBreakBetaObs(
            k_max=2, estimate_hyper=True, phi=phi_obs
        ).fit(_X(int(mask.sum())), y_obs)

        assert est_mask.hyper_["alpha"] == pytest.approx(est_obs.hyper_["alpha"], rel=1e-9)
        assert est_mask.hyper_["beta"] == pytest.approx(est_obs.hyper_["beta"], rel=1e-9)

    def test_repeated_fit_does_not_use_stale_descriptor(self):
        """Fitting twice with different masks does not use descriptors from the first fit."""
        rng = np.random.default_rng(530)
        n = 20
        r_arr = rng.uniform(1.0, 3.0, size=n)
        y = np.array([rng.negative_binomial(max(1, int(r)), 0.5) for r in r_arr], dtype=float)
        mask1 = rng.random(n) > 0.3
        mask2 = rng.random(n) > 0.5  # different mask

        est = BayesBreakNegBin(k_max=3, estimate_hyper=True, r=r_arr)
        est.fit(_X(n), y, observation_mask=mask1)
        hyper1 = dict(est.hyper_)

        est.fit(_X(n), y, observation_mask=mask2)
        hyper2 = dict(est.hyper_)

        # Hypers from different masks may differ; key check is no ValueError.
        assert est._r_arr_.size == n
        # Both fits should give valid hypers.
        assert hyper1["alpha"] > 0 and hyper1["beta"] > 0
        assert hyper2["alpha"] > 0 and hyper2["beta"] > 0


# ---------------------------------------------------------------------------
# A2 fix: NegBin divergent posterior moments report nan
# ---------------------------------------------------------------------------

class TestA2NegBinDivergentMoment:
    """A2 fix: A1[i,j] is nan when a_post = alpha + sum(r) <= 1, not a silent 0."""

    def test_divergent_all_blocks_are_nan(self):
        """With very small r, a_post <= 1 for all blocks and A1 is all nan."""
        n = 5
        y = np.ones(n, dtype=float)
        # alpha=0.01, r=0.01: a_post = 0.01 + k*0.01 <= 0.06 for k<=5, all < 1.
        est = BayesBreakNegBin(
            k_max=n, estimate_hyper=False, alpha=0.01, beta=1.0, r=0.01
        ).fit(_X(n), y)

        upper = np.triu(np.ones((n + 1, n + 1), dtype=bool), k=1)
        assert np.all(np.isnan(est.block_first_moment_[upper]))

    def test_valid_moment_is_positive_finite(self):
        """With large alpha, a_post > 1 for all blocks and A1 is positive finite."""
        n = 5
        y = np.ones(n, dtype=float)
        est = BayesBreakNegBin(
            k_max=n, estimate_hyper=False, alpha=2.0, beta=1.0, r=1.0
        ).fit(_X(n), y)

        upper = np.triu(np.ones((n + 1, n + 1), dtype=bool), k=1)
        moments = est.block_first_moment_[upper]
        assert np.all(np.isfinite(moments))
        assert np.all(moments > 0)

    def test_moment_value_matches_formula(self):
        """A1[0,n] = r * b_post / (a_post - 1) for a single full-sequence block."""
        n, r, alpha, beta = 5, 2.0, 3.0, 1.0
        y = np.array([1.0, 3.0, 2.0, 4.0, 1.0])
        est = BayesBreakNegBin(
            k_max=n, estimate_hyper=False, alpha=alpha, beta=beta, r=r
        ).fit(_X(n), y)

        Nsum = n * r
        Csum = float(y.sum())
        a_post = alpha + Nsum
        b_post = beta + Csum
        expected = r * b_post / (a_post - 1.0)
        assert float(est.block_first_moment_[0, n]) == pytest.approx(expected, rel=1e-9)

    def test_boundary_a_post_exactly_one(self):
        """When a_post exactly equals 1, A1 should be nan (moment diverges at boundary)."""
        # alpha=0.9, r=0.1, n=1: a_post = 0.9 + 1*0.1 = 1.0, diverges.
        y = np.array([2.0])
        est = BayesBreakNegBin(
            k_max=1, estimate_hyper=False, alpha=0.9, beta=1.0, r=0.1
        ).fit(_X(1), y)

        assert np.isnan(float(est.block_first_moment_[0, 1]))

    # --- second moment (not stored, but formula properties verified here) ---

    def test_second_moment_not_stored(self):
        """block_second_moment_ is not implemented; only block_first_moment_ is available."""
        y = np.array([1.0, 2.0])
        est = BayesBreakNegBin(
            k_max=2, estimate_hyper=False, alpha=3.0, beta=1.0, r=1.0
        ).fit(_X(2), y)
        assert not hasattr(est, "block_second_moment_")
        assert hasattr(est, "block_first_moment_")

    def test_second_moment_formula_valid_above_two(self):
        """E[m^2] = r^2 b(b+1)/((a-1)(a-2)) is positive finite for a_post > 2."""
        # a_post=alpha+N=3.0+1*3=6.0 (n=3, r=1), b_post=beta+C=1.0+6.0=7.0.
        n, r, alpha, beta = 3, 1.0, 3.0, 1.0
        y = np.array([2.0, 1.0, 3.0])
        a_post = alpha + n * r        # = 6.0
        b_post = beta + float(y.sum())  # = 7.0
        second_moment = r**2 * b_post * (b_post + 1.0) / ((a_post - 1.0) * (a_post - 2.0))
        assert np.isfinite(second_moment) and second_moment > 0.0

    def test_second_moment_formula_diverges_at_a_post_le_2(self):
        """E[m^2] formula denominator is zero or negative when a_post <= 2."""
        # At a_post = 2: denominator = (2-1)*(2-2) = 0.
        assert (2.0 - 1.0) * (2.0 - 2.0) == 0.0
        # At a_post = 1.5: denominator = 0.5 * (-0.5) < 0.
        assert (1.5 - 1.0) * (1.5 - 2.0) < 0.0
        # At a_post = 0.5: denominator = (-0.5) * (-1.5) > 0 but a_post <= 1 also
        # means the first moment diverges, so the second moment is also undefined.
        assert (0.5 - 1.0) < 0.0  # a_post <= 1 already triggers first-moment divergence
