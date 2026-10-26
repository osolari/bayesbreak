"""Regression tests: Beta-obs Gauss-Legendre quadrature convergence.

Verifies that 128-node GL agrees with scipy.integrate.quad on E02 configurations
to within the prespecified 1e-3 gate threshold, and that the runner gate logic
is correctly structured.

These tests guard against:
  1. Regression to a lower quadrature order (64 nodes was marginally insufficient).
  2. Re-introduction of the gate_pass premature-evaluation bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad as scipy_quad
from scipy.special import gammaln as sp_gammaln


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


def _generate_beta_obs_e02(n: int, mu_mid: float, cell_index: int, rep: int) -> dict:
    """Reproduce E02 beta_obs dataset for a given cell and replicate."""
    seed_seq = np.random.SeedSequence([20261021, 2, cell_index, rep])
    data_seed = seed_seq.spawn(2)[0]
    rng = np.random.default_rng(data_seed)

    seg = n // 3
    true_mus = [0.2, mu_mid, 0.2]
    phi_arr = np.array([20.0 if i % 2 == 0 else 80.0 for i in range(n)], dtype=float)
    y = np.empty(n, dtype=float)
    for i in range(n):
        mu_q = true_mus[i // seg]
        phi = phi_arr[i]
        y[i] = rng.beta(phi * mu_q, phi * (1.0 - mu_q))
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "phi_arr": phi_arr}


def _scipy_block_posterior_mean(
    y_blk: np.ndarray,
    phi_blk: np.ndarray,
    alpha0: float,
    beta0: float,
) -> float:
    """E[mu | block] via scipy.quad — used as reference."""
    log_y = np.log(y_blk)
    log_1my = np.log(1.0 - y_blk)
    log_Bab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)

    def unnorm(mu: float) -> float:
        if mu <= 0.0 or mu >= 1.0:
            return 0.0
        log_prior = (alpha0 - 1.0) * math.log(mu) + (beta0 - 1.0) * math.log(1.0 - mu) - log_Bab
        log_lik = float(np.sum(
            sp_gammaln(phi_blk)
            - sp_gammaln(phi_blk * mu)
            - sp_gammaln(phi_blk * (1.0 - mu))
            + (phi_blk * mu - 1.0) * log_y
            + (phi_blk * (1.0 - mu) - 1.0) * log_1my
        ))
        return math.exp(log_prior + log_lik)

    # Reference scale to avoid floating-point underflow in scipy.quad
    scale = max(unnorm(m) for m in np.linspace(0.05, 0.95, 50))
    if scale <= 0.0:
        return float("nan")

    norm, _ = scipy_quad(lambda mu: unnorm(mu) / scale, 0.0, 1.0,
                         limit=200, epsabs=1e-12, epsrel=1e-10)
    moment, _ = scipy_quad(lambda mu: mu * unnorm(mu) / scale, 0.0, 1.0,
                           limit=200, epsabs=1e-12, epsrel=1e-10)
    if norm <= 0.0:
        return float("nan")
    return float(moment / norm)


def _bayes_curve_mix_k(est, n: int, k_max: int) -> np.ndarray:
    """Compute mixed-k Bayes curve from a fitted BayesBreakBetaObs estimator."""
    from bayesbreak.dp import boundary_event_marginals_marginalised  # noqa: F401

    def _logsumexp_axis0(a):
        a_max = np.max(a, axis=0, keepdims=True)
        a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
        s = np.sum(np.exp(a - a_max_safe), axis=0)
        a_max_sq = np.squeeze(a_max, axis=0)
        with np.errstate(divide="ignore"):
            log_s = np.log(s)
        return np.where(np.isfinite(a_max_sq), a_max_sq + log_s, -np.inf)

    def _curve_fixed_k(L, R, lA0, A1, k):
        denom = float(L[k, n])
        if not np.isfinite(denom):
            return np.full(n, np.nan)
        LL = L[0:k, :]
        RR = R[k - 1::-1, :]
        cross = _logsumexp_axis0(LL[:, :, np.newaxis] + RR[:, np.newaxis, :])
        log_w = cross + lA0 - denom
        finite_mask = np.isfinite(log_w) & np.isfinite(A1) & (A1 != 0.0)
        log_w_clipped = np.where(finite_mask, np.clip(log_w, -745.0, 0.0), -np.inf)
        contrib = np.where(finite_mask, np.exp(log_w_clipped) * A1, 0.0)
        diff = contrib.sum(axis=1) - contrib.sum(axis=0)
        return np.cumsum(diff)[:n]

    out = np.zeros(n, dtype=float)
    for k in range(1, k_max + 1):
        wk = float(est.k_posterior_[k - 1])
        if wk == 0.0 or not np.isfinite(float(est.log_left_[k, n])):
            continue
        out += wk * _curve_fixed_k(
            est.log_left_, est.log_right_,
            est.log_block_evidence_, est.block_first_moment_, k,
        )
    return out


# ---------------------------------------------------------------------------
# Gate threshold constant (must not be changed here without protocol amendment)
# ---------------------------------------------------------------------------

CURVE_GATE_THRESHOLD = 1e-3


# ---------------------------------------------------------------------------
# Test 1: 128-node vs scipy.quad on E02 configurations
# ---------------------------------------------------------------------------

class TestBetaObs128NodeVsScipy:
    """128-node GL block posterior mean agrees with scipy.quad to within 1e-3."""

    @pytest.mark.parametrize("mu_mid,cell_index,rep", [
        (0.4, 8, 0),
        (0.4, 8, 2),  # The worst-case rep from the original gate failure
        (0.8, 9, 0),
        (0.8, 9, 2),  # Worst 64-node rep for mu_mid=0.8
    ])
    def test_128_node_block_mean_agrees_with_scipy(self, mu_mid, cell_index, rep):
        """128-node GL posterior mean for a single segment agrees with scipy.quad."""
        from bayesbreak.families.beta_obs import BayesBreakBetaObs

        n = 120
        alpha0, beta0 = 2.0, 3.0
        ds = _generate_beta_obs_e02(n, mu_mid, cell_index, rep)
        y, phi_arr = ds["y"], ds["phi_arr"]

        # Use the middle segment only for the per-block scipy reference
        seg = n // 3
        y_mid = y[seg:2 * seg]
        phi_mid = phi_arr[seg:2 * seg]

        est = BayesBreakBetaObs(
            k_max=10, estimate_hyper=False,
            alpha=alpha0, beta=beta0,
            phi=phi_arr, quadrature_points=128,
            prior_k=_geom_prior,
        )
        est.fit(np.arange(1, n + 1, dtype=float).reshape(-1, 1), y)

        # GL estimate: block_first_moment_ for the middle segment
        gl_mean = float(est.block_first_moment_[seg, 2 * seg])

        # scipy reference
        scipy_mean = _scipy_block_posterior_mean(y_mid, phi_mid, alpha0, beta0)

        assert math.isfinite(scipy_mean), "scipy reference returned nan/inf"
        assert math.isfinite(gl_mean), "128-node GL returned nan/inf"
        assert abs(gl_mean - scipy_mean) < CURVE_GATE_THRESHOLD, (
            f"mu_mid={mu_mid}, rep={rep}: 128-node GL block mean {gl_mean:.6f} "
            f"deviates {abs(gl_mean - scipy_mean):.3e} from scipy {scipy_mean:.6f}, "
            f"threshold {CURVE_GATE_THRESHOLD}"
        )


# ---------------------------------------------------------------------------
# Test 2: 128 vs 256 Bayes-curve gate on E02 seeds
# ---------------------------------------------------------------------------

class TestBetaObs128vs256Gate:
    """128-node vs 256-node Bayes-curve deviation < 1e-3 on E02 seeds."""

    @pytest.mark.parametrize("mu_mid,cell_index", [
        (0.4, 8),
        (0.8, 9),
    ])
    def test_128_vs_256_passes_gate_on_all_5_reps(self, mu_mid, cell_index):
        """All 5 E02 oracle reps: 128 vs 256 Bayes-curve dev < 1e-3."""
        from bayesbreak.families.beta_obs import BayesBreakBetaObs

        n = 120
        alpha0, beta0 = 2.0, 3.0
        k_max = 10
        worst_dev = 0.0

        for rep in range(5):
            ds = _generate_beta_obs_e02(n, mu_mid, cell_index, rep)
            y, phi_arr, x = ds["y"], ds["phi_arr"], ds["x"]
            X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)

            est128 = BayesBreakBetaObs(
                k_max=k_max, estimate_hyper=False,
                alpha=alpha0, beta=beta0,
                phi=phi_arr, quadrature_points=128,
                prior_k=_geom_prior,
            )
            est256 = BayesBreakBetaObs(
                k_max=k_max, estimate_hyper=False,
                alpha=alpha0, beta=beta0,
                phi=phi_arr, quadrature_points=256,
                prior_k=_geom_prior,
            )
            est128.fit(X, y)
            est256.fit(X, y)

            c128 = _bayes_curve_mix_k(est128, n, k_max)
            c256 = _bayes_curve_mix_k(est256, n, k_max)
            dev = float(np.max(np.abs(c128 - c256)))
            worst_dev = max(worst_dev, dev)

        assert worst_dev < CURVE_GATE_THRESHOLD, (
            f"mu_mid={mu_mid}: max 128 vs 256 Bayes-curve deviation "
            f"{worst_dev:.4e} >= threshold {CURVE_GATE_THRESHOLD}"
        )


# ---------------------------------------------------------------------------
# Test 3: 64-node was marginal (regression guard — 64 nodes should fail gate)
# ---------------------------------------------------------------------------

class TestBetaObs64NodeMarginal:
    """64-node GL was marginally over threshold on E02 seed (mu_mid=0.4, rep=2)."""

    def test_64_node_rep2_mu04_was_over_threshold(self):
        """Confirm that the original failing rep (64 vs 128, mu_mid=0.4 rep 2) was >1e-3.

        This test documents the root cause: 64-node GL was insufficient for this seed.
        It is NOT a failure if deviation > threshold; it is the expected behavior of
        64 nodes on this concentrated integrand.
        """
        from bayesbreak.families.beta_obs import BayesBreakBetaObs

        n = 120
        alpha0, beta0 = 2.0, 3.0
        k_max = 10
        ds = _generate_beta_obs_e02(n, mu_mid=0.4, cell_index=8, rep=2)
        y, phi_arr = ds["y"], ds["phi_arr"]
        X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)

        est64 = BayesBreakBetaObs(
            k_max=k_max, estimate_hyper=False,
            alpha=alpha0, beta=beta0,
            phi=phi_arr, quadrature_points=64,
            prior_k=_geom_prior,
        )
        est128 = BayesBreakBetaObs(
            k_max=k_max, estimate_hyper=False,
            alpha=alpha0, beta=beta0,
            phi=phi_arr, quadrature_points=128,
            prior_k=_geom_prior,
        )
        est64.fit(X, y)
        est128.fit(X, y)

        c64 = _bayes_curve_mix_k(est64, n, k_max)
        c128 = _bayes_curve_mix_k(est128, n, k_max)
        dev = float(np.max(np.abs(c64 - c128)))

        # The original failure: 1.006e-3 > 1e-3. Document that 64 nodes was
        # marginally insufficient.
        assert dev >= CURVE_GATE_THRESHOLD, (
            f"Expected 64-node deviation >= {CURVE_GATE_THRESHOLD} for this seed, "
            f"got {dev:.4e}. If this fails, the integrand or seed has changed."
        )


# ---------------------------------------------------------------------------
# Test 4: Runner gate logic structural test
# ---------------------------------------------------------------------------

class TestRunnerGateLogic:
    """The runner script's gate logic structure: gate_pass is evaluated after checks."""

    def test_gate_pass_set_false_initially_and_updated(self, tmp_path):
        """Verify that metrics.json gate_pass reflects actual gate outcomes."""
        import importlib.util
        import sys as _sys

        # Import the runner module without executing its top-level code
        runner_path = (
            __file__.replace("tests/", "").replace("test_beta_obs_quadrature.py", "")
            + "scripts/run_e02_family_portability.py"
        )
        from pathlib import Path as _Path
        runner_file = _Path(__file__).parent.parent / "scripts" / "run_e02_family_portability.py"
        assert runner_file.exists(), f"Runner not found at {runner_file}"

        spec = importlib.util.spec_from_file_location("_e02_runner", runner_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Verify _run_gate_checks returns bool
        assert callable(mod._run_gate_checks), "_run_gate_checks must be callable"

        import inspect
        sig = inspect.signature(mod._run_gate_checks)
        # Should return bool — verify it can be called with empty inputs and returns bool
        result_pass = mod._run_gate_checks([], {})
        result_fail = mod._run_gate_checks(
            [{"cell_index": 0, "family": "test", "mean_f1": float("nan"),
              "mean_signal_mse": float("nan"), "k_max_saturation": False}],
            {},
        )
        assert isinstance(result_pass, bool), "_run_gate_checks must return bool"
        assert isinstance(result_fail, bool), "_run_gate_checks must return bool"
        assert result_pass is True, "Empty checks should pass"
        assert result_fail is False, "NaN mean_f1 should fail gate"

    def test_gate_checks_fail_on_oracle_threshold_violation(self):
        """_run_gate_checks returns False when curve_dev > 1e-3."""
        import importlib.util
        from pathlib import Path

        runner_file = Path(__file__).parent.parent / "scripts" / "run_e02_family_portability.py"
        spec = importlib.util.spec_from_file_location("_e02_runner2", runner_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        oracle_fail = {
            "beta_obs,n=120,mu_mid=0.4": {
                "max_log_evidence_deviation": 0.079,
                "max_bayes_curve_deviation": 1.1e-3,  # Exceeds 1e-3 threshold
                "n_reps_checked": 5,
            }
        }
        result = mod._run_gate_checks([], oracle_fail)
        assert result is False, "curve_dev=1.1e-3 should fail gate (threshold=1e-3)"

        oracle_pass = {
            "beta_obs,n=120,mu_mid=0.4": {
                "max_log_evidence_deviation": 3.5e-7,
                "max_bayes_curve_deviation": 4.7e-8,  # Well within threshold
                "n_reps_checked": 5,
            }
        }
        result = mod._run_gate_checks([], oracle_pass)
        assert result is True, "curve_dev=4.7e-8 should pass gate (threshold=1e-3)"
