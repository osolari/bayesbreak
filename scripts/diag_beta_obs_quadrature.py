"""Diagnostic: Beta-obs Gauss-Legendre quadrature convergence for E02 configurations.

Evaluates block log-marginal-likelihood and Bayes-curve at GL orders 32, 64, 128, 256,
512 and against scipy.integrate.quad as an independent high-accuracy reference.
Reports deviation from reference at each order, integrand concentration,
and log-domain stability.

Also re-evaluates the specific E02 seeds (reps 0-4) for the beta_obs,mu_mid=0.4 cell
that produced the largest Bayes-curve deviation in oracle_checks.json.

Usage:
    uv run python scripts/diag_beta_obs_quadrature.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad as scipy_quad
from scipy.special import gammaln as sp_gammaln

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure src is importable
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bayesbreak.families.beta_obs import BayesBreakBetaObs  # noqa: E402

# ---------------------------------------------------------------------------
# E02 configuration (mirrors run_e02_family_portability.py)
# ---------------------------------------------------------------------------

E02_PRIOR = {"alpha": 2.0, "beta": 3.0}
K_MAX = 10
SEED_BASE = [20261021, 2]


def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


# ---------------------------------------------------------------------------
# Bayes-curve helper (copied from runner)
# ---------------------------------------------------------------------------

def _logsumexp_axis0(a: np.ndarray) -> np.ndarray:
    a_max = np.max(a, axis=0, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=0)
    a_max_sq = np.squeeze(a_max, axis=0)
    with np.errstate(divide="ignore"):
        log_s = np.log(s)
    return np.where(np.isfinite(a_max_sq), a_max_sq + log_s, -np.inf)


def _bayes_curve_fixed_k(L, R, lA0, A1, n, k):
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


def _bayes_curve(L, R, lA0, A1, n, k_max, post_k):
    out = np.zeros(n)
    for k in range(1, k_max + 1):
        wk = float(post_k[k - 1])
        if wk == 0.0 or not np.isfinite(float(L[k, n])):
            continue
        out += wk * _bayes_curve_fixed_k(L, R, lA0, A1, n, k)
    return out


# ---------------------------------------------------------------------------
# scipy.integrate.quad reference for a single block
# ---------------------------------------------------------------------------

def _scipy_block_log_evidence(
    y_blk: np.ndarray,
    phi_blk: np.ndarray,
    alpha0: float,
    beta0: float,
) -> float:
    """High-accuracy log block marginal likelihood via scipy.quad."""
    log_y = np.log(y_blk)
    log_1my = np.log(1.0 - y_blk)
    log_Bab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)

    def integrand(mu: float) -> float:
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

    # Use log-scale to avoid underflow: find the mode first
    # Evaluate at several points to pick a safe reference
    mu_test = np.linspace(0.05, 0.95, 50)
    vals = np.array([integrand(m) for m in mu_test])
    max_val = float(vals.max())
    if max_val <= 0.0:
        return -np.inf

    # Normalised integrand to improve numerical stability
    def integrand_norm(mu: float) -> float:
        return integrand(mu) / max_val

    result, abserr = scipy_quad(
        integrand_norm, 0.0, 1.0,
        limit=200, epsabs=1e-12, epsrel=1e-10,
    )
    return math.log(result * max_val) if result > 0.0 else -np.inf


def _scipy_block_posterior_mean_mu(
    y_blk: np.ndarray,
    phi_blk: np.ndarray,
    alpha0: float,
    beta0: float,
) -> float:
    """E[mu | block] via scipy.quad (unnormalized integral ratio)."""
    log_y = np.log(y_blk)
    log_1my = np.log(1.0 - y_blk)
    log_Bab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)

    def unnorm_integrand(mu: float) -> float:
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

    # Reference scale
    mu_test = np.linspace(0.05, 0.95, 50)
    ref_scale = max(float(unnorm_integrand(m)) for m in mu_test)
    if ref_scale <= 0.0:
        return float("nan")

    norm, _ = scipy_quad(
        lambda mu: unnorm_integrand(mu) / ref_scale,
        0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-10,
    )
    mu_moment, _ = scipy_quad(
        lambda mu: mu * unnorm_integrand(mu) / ref_scale,
        0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-10,
    )
    if norm <= 0.0:
        return float("nan")
    return float(mu_moment / norm)


# ---------------------------------------------------------------------------
# Generate E02 beta_obs datasets (mirrors runner seed scheme)
# ---------------------------------------------------------------------------

def _generate_beta_obs_dataset(
    n: int,
    mu_mid: float,
    cell_index: int,
    rep: int,
) -> dict:
    """Reproduce E02 dataset generation for a specific cell and replicate."""
    seed_seq = np.random.SeedSequence([20261021, 2, cell_index, rep])
    data_seed = seed_seq.spawn(2)[0]
    rng = np.random.default_rng(data_seed)

    seg = n // 3
    true_mus = [0.2, mu_mid, 0.2]
    phi_arr = np.array([20.0 if i % 2 == 0 else 80.0 for i in range(n)], dtype=float)
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_mus[s]
    y = np.empty(n, dtype=float)
    for i in range(n):
        mu_q = true_mus[i // seg]
        phi = phi_arr[i]
        y[i] = rng.beta(phi * mu_q, phi * (1.0 - mu_q))
    x = np.arange(1, n + 1, dtype=float)
    return {"y": y, "x": x, "phi_arr": phi_arr, "true_signal": true_signal, "true_mus": true_mus}


# ---------------------------------------------------------------------------
# Quadrature order sweep for one dataset
# ---------------------------------------------------------------------------

def _quadrature_sweep(
    y: np.ndarray,
    phi_arr: np.ndarray,
    alpha0: float,
    beta0: float,
    n: int,
    orders: list[int],
    label: str,
) -> dict:
    """Evaluate block log-evidence and Bayes curve for several GL orders plus scipy.quad.

    Returns dict with per-order log_evidence, bayes_curve, and deviations from scipy ref.
    """
    logger.info("  Quadrature sweep for %s (n=%d)", label, n)
    x = np.arange(1, n + 1, dtype=float)

    # scipy reference: evaluate on the full data block [0, n)
    logger.info("    Computing scipy.quad reference for full block...")
    log_ev_ref = _scipy_block_log_evidence(y, phi_arr, alpha0, beta0)
    mu_mean_ref = _scipy_block_posterior_mean_mu(y, phi_arr, alpha0, beta0)
    logger.info("    scipy log_ev_ref = %.6f, mu_mean_ref = %.6f", log_ev_ref, mu_mean_ref)

    results: dict = {
        "scipy_ref_log_evidence_full_block": log_ev_ref,
        "scipy_ref_mu_mean_full_block": mu_mean_ref,
        "orders": {},
    }

    # Also store full Bayes-curve at each order (full fit)
    for order in orders:
        logger.info("    Fitting with quadrature_points=%d ...", order)
        est = BayesBreakBetaObs(
            k_max=K_MAX,
            estimate_hyper=False,
            alpha=alpha0, beta=beta0,
            phi=phi_arr,
            quadrature_points=order,
            prior_k=_geom_prior,
        )
        est.fit(x.reshape(-1, 1), y)

        log_ev = float(est.log_evidence_)
        curve = _bayes_curve(
            est.log_left_, est.log_right_,
            est.log_block_evidence_, est.block_first_moment_,
            n, K_MAX, est.k_posterior_,
        )
        # Block [0,n) single-segment posterior mean from block_first_moment_ vs scipy
        block_mu_gl = float(est.block_first_moment_[0, n])
        dev_log_ev_from_scipy = abs(log_ev - log_ev_ref) if math.isfinite(log_ev_ref) else float("nan")
        dev_mu_mean_from_scipy = abs(block_mu_gl - mu_mean_ref) if math.isfinite(mu_mean_ref) else float("nan")

        results["orders"][order] = {
            "log_evidence": log_ev,
            "block_mu_posterior_mean": block_mu_gl,
            "dev_log_ev_from_scipy": dev_log_ev_from_scipy,
            "dev_block_mu_from_scipy": dev_mu_mean_from_scipy,
            "bayes_curve": curve,
        }
        logger.info(
            "    order=%d: log_ev=%.6f, dev_from_scipy=%.3e, block_mu=%.5f, dev_mu_from_scipy=%.3e",
            order, log_ev, dev_log_ev_from_scipy, block_mu_gl, dev_mu_mean_from_scipy,
        )

    return results


# ---------------------------------------------------------------------------
# Bayes-curve deviation between consecutive orders
# ---------------------------------------------------------------------------

def _curve_deviation_table(sweep_results: dict, orders: list[int]) -> None:
    """Log a table of Bayes-curve max deviations between consecutive order pairs."""
    logger.info("  --- Consecutive-order Bayes-curve max deviations ---")
    ord_data = sweep_results["orders"]
    for lo, hi in zip(orders[:-1], orders[1:]):
        c_lo = ord_data[lo]["bayes_curve"]
        c_hi = ord_data[hi]["bayes_curve"]
        dev = float(np.max(np.abs(c_lo - c_hi)))
        status = "PASS" if dev < 1e-3 else "FAIL"
        logger.info(
            "    %3d vs %3d: max_curve_dev = %.4e  [threshold 1e-3 -> %s]",
            lo, hi, dev, status,
        )


# ---------------------------------------------------------------------------
# Integrand concentration diagnostic
# ---------------------------------------------------------------------------

def _integrand_concentration(
    y_blk: np.ndarray,
    phi_blk: np.ndarray,
    alpha0: float,
    beta0: float,
    n_eval: int = 200,
) -> None:
    """Log the concentration of the unnormalized integrand over (0,1)."""
    log_Bab = math.lgamma(alpha0) + math.lgamma(beta0) - math.lgamma(alpha0 + beta0)
    log_y = np.log(y_blk)
    log_1my = np.log(1.0 - y_blk)
    mu_grid = np.linspace(0.01, 0.99, n_eval)
    log_vals = np.empty(n_eval)
    for i, mu in enumerate(mu_grid):
        log_prior = (alpha0 - 1.0) * math.log(mu) + (beta0 - 1.0) * math.log(1.0 - mu) - log_Bab
        log_lik = float(np.sum(
            sp_gammaln(phi_blk)
            - sp_gammaln(phi_blk * mu)
            - sp_gammaln(phi_blk * (1.0 - mu))
            + (phi_blk * mu - 1.0) * log_y
            + (phi_blk * (1.0 - mu) - 1.0) * log_1my
        ))
        log_vals[i] = log_prior + log_lik

    log_vals_centered = log_vals - np.max(log_vals)
    density = np.exp(log_vals_centered)
    density /= np.trapezoid(density, mu_grid)

    # Report approximate mode and 0.95 credible interval
    dmu = mu_grid[1] - mu_grid[0]
    cum = np.cumsum(density * dmu)
    cum /= cum[-1]
    idx_lo = int(np.searchsorted(cum, 0.025))
    idx_hi = int(np.searchsorted(cum, 0.975))
    mode_idx = int(np.argmax(density))
    logger.info(
        "    Integrand: mode ~ %.4f, 95%% CI ~ [%.4f, %.4f]",
        mu_grid[mode_idx], mu_grid[idx_lo], mu_grid[idx_hi],
    )
    ci_width = mu_grid[idx_hi] - mu_grid[idx_lo]
    logger.info(
        "    95%% CI width = %.4f (narrow = concentrated, may need many GL nodes)",
        ci_width,
    )


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostics() -> None:
    logger.info("=" * 70)
    logger.info("Beta-obs quadrature convergence diagnostic (E02)")
    logger.info("=" * 70)

    alpha0 = E02_PRIOR["alpha"]
    beta0 = E02_PRIOR["beta"]
    GL_ORDERS = [32, 64, 128, 256, 512]

    # -----------------------------------------------------------------------
    # Section 1: Representative configuration sweep
    # -----------------------------------------------------------------------
    configs = [
        # (mu_mid, n, cell_index, label)
        (0.4, 120, 8, "mu_mid=0.4"),
        (0.8, 120, 9, "mu_mid=0.8"),
    ]

    for mu_mid, n, cell_index, cfg_label in configs:
        logger.info("")
        logger.info("--- Configuration: %s, n=%d ---", cfg_label, n)

        # Use rep=2 for mu_mid=0.4 (the failing rep: curve_dev=1.01e-3)
        # Use rep=0 for mu_mid=0.8
        rep = 2 if mu_mid == 0.4 else 0

        logger.info("  Generating E02 dataset (cell_index=%d, rep=%d)...", cell_index, rep)
        ds = _generate_beta_obs_dataset(n, mu_mid, cell_index, rep)
        y = ds["y"]
        phi_arr = ds["phi_arr"]

        # Use a single segment near the middle for the per-block reference
        seg = n // 3
        y_mid = y[seg:2 * seg]
        phi_mid = phi_arr[seg:2 * seg]

        logger.info("  Integrand concentration for middle segment (mu_mid=%.2f, phi~{20,80}):", mu_mid)
        _integrand_concentration(y_mid, phi_mid, alpha0, beta0)

        logger.info("  Full dataset quadrature sweep (GL orders %s):", GL_ORDERS)
        sweep = _quadrature_sweep(y, phi_arr, alpha0, beta0, n, GL_ORDERS, cfg_label)
        _curve_deviation_table(sweep, GL_ORDERS)

    # -----------------------------------------------------------------------
    # Section 2: Targeted sweep of E02 seeds that produced the largest deviation
    # -----------------------------------------------------------------------
    # oracle_checks.json: mu_mid=0.4, n_reps_checked=5
    # run_log shows per-rep curve_devs: rep0=6.24e-4, rep1=9.36e-4, rep2=1.01e-3, rep3=3.37e-4, rep4=8.06e-4
    logger.info("")
    logger.info("=" * 70)
    logger.info("Section 2: Per-rep Bayes-curve deviations for mu_mid=0.4 (reps 0-4)")
    logger.info("=" * 70)

    mu_mid = 0.4
    n = 120
    cell_index = 8  # cell index for beta_obs,mu_mid=0.4

    per_rep_results: list[dict] = []

    for rep in range(5):
        logger.info("")
        logger.info("  Rep %d", rep)
        ds = _generate_beta_obs_dataset(n, mu_mid, cell_index, rep)
        y = ds["y"]
        phi_arr = ds["phi_arr"]
        x = np.arange(1, n + 1, dtype=float)

        row: dict = {"rep": rep}
        for q1, q2 in [(64, 128), (128, 256)]:
            est1 = BayesBreakBetaObs(
                k_max=K_MAX, estimate_hyper=False,
                alpha=alpha0, beta=beta0,
                phi=phi_arr, quadrature_points=q1,
                prior_k=_geom_prior,
            )
            est2 = BayesBreakBetaObs(
                k_max=K_MAX, estimate_hyper=False,
                alpha=alpha0, beta=beta0,
                phi=phi_arr, quadrature_points=q2,
                prior_k=_geom_prior,
            )
            est1.fit(x.reshape(-1, 1), y)
            est2.fit(x.reshape(-1, 1), y)

            c1 = _bayes_curve(est1.log_left_, est1.log_right_,
                              est1.log_block_evidence_, est1.block_first_moment_,
                              n, K_MAX, est1.k_posterior_)
            c2 = _bayes_curve(est2.log_left_, est2.log_right_,
                              est2.log_block_evidence_, est2.block_first_moment_,
                              n, K_MAX, est2.k_posterior_)
            dev = float(np.max(np.abs(c1 - c2)))
            dev_ev = abs(float(est1.log_evidence_) - float(est2.log_evidence_))

            key_curve = f"curve_dev_{q1}_vs_{q2}"
            key_ev = f"log_ev_dev_{q1}_vs_{q2}"
            row[key_curve] = dev
            row[key_ev] = dev_ev
            status = "PASS" if dev < 1e-3 else "FAIL"
            logger.info(
                "    %d vs %d: curve_dev=%.4e [%s], log_ev_dev=%.4e",
                q1, q2, dev, status, dev_ev,
            )

        per_rep_results.append(row)

    # Summary table
    logger.info("")
    logger.info("Summary: mu_mid=0.4, per-rep deviations")
    logger.info("  %-5s  %-14s  %-14s  %-14s  %-14s",
                "rep", "64v128_curve", "64v128_PASS?", "128v256_curve", "128v256_PASS?")
    for row in per_rep_results:
        d64 = row.get("curve_dev_64_vs_128", float("nan"))
        d128 = row.get("curve_dev_128_vs_256", float("nan"))
        logger.info(
            "  %-5d  %-14.4e  %-14s  %-14.4e  %-14s",
            row["rep"],
            d64, "PASS" if d64 < 1e-3 else "FAIL",
            d128, "PASS" if d128 < 1e-3 else "FAIL",
        )

    max_64_128 = max(row["curve_dev_64_vs_128"] for row in per_rep_results)
    max_128_256 = max(row["curve_dev_128_vs_256"] for row in per_rep_results)
    logger.info("")
    logger.info("Max 64 vs 128 curve dev (5 reps, mu_mid=0.4): %.4e [gate %s]",
                max_64_128, "FAIL" if max_64_128 >= 1e-3 else "PASS")
    logger.info("Max 128 vs 256 curve dev (5 reps, mu_mid=0.4): %.4e [gate %s]",
                max_128_256, "FAIL" if max_128_256 >= 1e-3 else "PASS")

    # -----------------------------------------------------------------------
    # Section 3: mu_mid=0.8 reps 0-4
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("Section 3: Per-rep Bayes-curve deviations for mu_mid=0.8 (reps 0-4)")
    logger.info("=" * 70)

    mu_mid = 0.8
    n = 120
    cell_index = 9

    per_rep_08: list[dict] = []

    for rep in range(5):
        ds = _generate_beta_obs_dataset(n, mu_mid, cell_index, rep)
        y = ds["y"]
        phi_arr = ds["phi_arr"]
        x = np.arange(1, n + 1, dtype=float)

        row: dict = {"rep": rep}
        for q1, q2 in [(64, 128), (128, 256)]:
            est1 = BayesBreakBetaObs(
                k_max=K_MAX, estimate_hyper=False,
                alpha=alpha0, beta=beta0,
                phi=phi_arr, quadrature_points=q1,
                prior_k=_geom_prior,
            )
            est2 = BayesBreakBetaObs(
                k_max=K_MAX, estimate_hyper=False,
                alpha=alpha0, beta=beta0,
                phi=phi_arr, quadrature_points=q2,
                prior_k=_geom_prior,
            )
            est1.fit(x.reshape(-1, 1), y)
            est2.fit(x.reshape(-1, 1), y)

            c1 = _bayes_curve(est1.log_left_, est1.log_right_,
                              est1.log_block_evidence_, est1.block_first_moment_,
                              n, K_MAX, est1.k_posterior_)
            c2 = _bayes_curve(est2.log_left_, est2.log_right_,
                              est2.log_block_evidence_, est2.block_first_moment_,
                              n, K_MAX, est2.k_posterior_)
            dev = float(np.max(np.abs(c1 - c2)))
            dev_ev = abs(float(est1.log_evidence_) - float(est2.log_evidence_))

            key_curve = f"curve_dev_{q1}_vs_{q2}"
            key_ev = f"log_ev_dev_{q1}_vs_{q2}"
            row[key_curve] = dev
            row[key_ev] = dev_ev

        per_rep_08.append(row)

    logger.info("Summary: mu_mid=0.8, per-rep deviations")
    logger.info("  %-5s  %-14s  %-14s  %-14s  %-14s",
                "rep", "64v128_curve", "64v128_PASS?", "128v256_curve", "128v256_PASS?")
    for row in per_rep_08:
        d64 = row.get("curve_dev_64_vs_128", float("nan"))
        d128 = row.get("curve_dev_128_vs_256", float("nan"))
        logger.info(
            "  %-5d  %-14.4e  %-14s  %-14.4e  %-14s",
            row["rep"],
            d64, "PASS" if d64 < 1e-3 else "FAIL",
            d128, "PASS" if d128 < 1e-3 else "FAIL",
        )

    max_64_128_08 = max(row["curve_dev_64_vs_128"] for row in per_rep_08)
    max_128_256_08 = max(row["curve_dev_128_vs_256"] for row in per_rep_08)
    logger.info("")
    logger.info("Max 64 vs 128 curve dev (5 reps, mu_mid=0.8): %.4e [gate %s]",
                max_64_128_08, "FAIL" if max_64_128_08 >= 1e-3 else "PASS")
    logger.info("Max 128 vs 256 curve dev (5 reps, mu_mid=0.8): %.4e [gate %s]",
                max_128_256_08, "FAIL" if max_128_256_08 >= 1e-3 else "PASS")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_diagnostics()
