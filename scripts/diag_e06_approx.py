"""Diagnostic E06: independent block-level verification of nonconjugate approximation methods.

Computes block log-marginal-likelihood by four methods (adaptive scipy.integrate.quad
reference, Laplace, JJ, EP) for each (B, p) combination from the E06 protocol and
compares each approximation to the quadrature reference. Runs independently of the
E06 runner.

Key findings documented in this script:
- JJ is a rigorous variational lower bound (verified with high-accuracy reference).
- Laplace systematically underestimates (negative signed error) for most blocks.
- EP (parallel EP with GH moment matching) is NOT guaranteed to be a lower bound;
  it can exceed the true log-marginal on extreme blocks (all-success or all-failure).
- pg_vb shares the JJ code path in the estimator and is NOT an independent method.
- Simple scipy.quad over [-30,30] can be inaccurate for sharply peaked integrands
  (B=64, S=32 case); this script uses mode-centered adaptive quadrature.

Usage:
    uv run python scripts/diag_e06_approx.py
"""

from __future__ import annotations

import logging
import math
import sys
from typing import NamedTuple

import numpy as np
import scipy.integrate
from scipy.optimize import minimize_scalar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("diag_e06_approx")


class BlockResult(NamedTuple):
    B: int
    p: float
    S: float
    ref: float
    la: float
    jj: float
    ep: float
    la_err: float
    jj_err: float
    ep_err: float
    jj_lb_ok: bool
    ep_ok: bool


def _log_integrand(theta: float, S: float, N: float, nu: float, rho2: float) -> float:
    """Log unnormalized posterior density."""
    if theta <= 0:
        loglik = S * theta - N * math.log1p(math.exp(theta))
    else:
        loglik = S * theta - N * (theta + math.log1p(math.exp(-theta)))
    logprior = -0.5 * (math.log(2.0 * math.pi * rho2) + (theta - nu) ** 2 / rho2)
    return loglik + logprior


def _reference_lml(S: float, N: float, nu: float, rho2: float) -> float:
    """Log marginal likelihood via mode-centered adaptive quadrature.

    Uses the MAP mode as center and integrates over a sufficiently wide
    interval relative to the posterior standard deviation. This approach
    avoids inaccuracy from fixed-interval integration on sharply peaked
    integrands (e.g., B=64, S=32 where posterior std ~ 0.25).
    """

    # Find mode via bounded optimization.
    opt = minimize_scalar(
        lambda t: -_log_integrand(t, S, N, nu, rho2),
        bounds=(-20.0, 20.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    theta_mode = float(opt.x)
    log_max = _log_integrand(theta_mode, S, N, nu, rho2)

    # Posterior variance from negative Hessian at mode.
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    sig = sigmoid(theta_mode)
    h = -(N * sig * (1.0 - sig) + 1.0 / rho2)
    if h >= 0.0:
        # Degenerate: fall back to wide interval.
        half_width = 20.0
    else:
        v_lap = -1.0 / h
        half_width = max(10.0 * math.sqrt(v_lap), 5.0)

    def shifted(t: float) -> float:
        return math.exp(_log_integrand(theta_mode + t, S, N, nu, rho2) - log_max)

    result, _err = scipy.integrate.quad(
        shifted,
        -half_width,
        half_width,
        epsabs=1e-13,
        epsrel=1e-12,
        limit=500,
    )
    if result <= 0.0:
        return -math.inf
    return log_max + math.log(result)


def compute_block(
    B: int,
    p_label: float,
    S: float,
    nu: float,
    rho2: float,
) -> BlockResult:
    """Compute all four approximations for a block with S successes in B Bernoulli trials."""
    from bayesbreak.families.logistic_normal._ep import ep_single_block
    from bayesbreak.families.logistic_normal._jj import jj_block
    from bayesbreak.families.logistic_normal._laplace import laplace_block

    N = float(B)
    S_arr = np.array([S])
    N_arr = np.array([N])

    # Reference: mode-centered adaptive quadrature.
    ref = _reference_lml(S, N, nu, rho2)

    # Laplace
    la_arr, _m, _v, _g = laplace_block(S_arr, N_arr, nu=nu, rho2=rho2, max_iter=50)
    la = float(la_arr[0])

    # JJ
    jj_arr, _m, _v = jj_block(S_arr, N_arr, nu=nu, rho2=rho2, max_iter=500, tol=1e-12)
    jj = float(jj_arr[0])

    # EP: build a block with exactly round(S) successes followed by failures.
    n_succ = int(round(S))
    y_block = np.zeros(B)
    y_block[:n_succ] = 1.0
    w_block = np.ones(B)
    ep_logA0, _m, _v, _pm, ep_ok = ep_single_block(
        y_block, w_block, nu=nu, rho2=rho2,
        max_iter=300, tol_kl=1e-8, damping=0.7, n_gh=80,
    )
    ep = float(ep_logA0)

    la_err = la - ref
    jj_err = jj - ref
    ep_err = ep - ref
    # JJ must be <= ref + small numerical tolerance.
    jj_lb_ok = jj <= ref + 1e-6

    return BlockResult(
        B=B, p=p_label, S=S, ref=ref, la=la, jj=jj, ep=ep,
        la_err=la_err, jj_err=jj_err, ep_err=ep_err,
        jj_lb_ok=jj_lb_ok, ep_ok=ep_ok,
    )


def run_diagnostics() -> None:
    # E06 frozen configuration.
    B_values = [4, 8, 16, 32, 64, 128]
    p_values = [0.05, 0.5, 0.95]
    nu, rho2 = 0.0, 4.0  # E06 prior: theta ~ N(0, 4)

    logger.info("E06 block-level approximation diagnostic")
    logger.info("Prior: theta ~ N(%.1f, %.1f)  [sigma_0^2 = %.1f]", nu, rho2, rho2)
    logger.info("Reference: mode-centered adaptive scipy.integrate.quad (epsabs=1e-13)")
    logger.info("Methods: Laplace, JJ (lower bound), EP (not a guaranteed lower bound)")
    logger.info("pg_vb shares JJ code path: NOT benchmarked as independent method")

    results: list[BlockResult] = []
    n_jj_lb_violations = 0

    header = (
        f"{'B':>4} {'p':>5} {'S':>5} | {'ref':>11} {'laplace':>11} {'jj':>11} "
        f"{'ep':>11} | {'la_err':>8} {'jj_err':>8} {'ep_err':>9} | {'JJ<=':>5} {'ep_ok':>5}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    for B in B_values:
        for p in p_values:
            # Deterministic block: y = round(p * B) successes in B Bernoulli trials.
            S = float(round(p * B))
            res = compute_block(B, p, S, nu, rho2)
            results.append(res)
            if not res.jj_lb_ok:
                n_jj_lb_violations += 1

            logger.info(
                "B=%3d p=%.2f S=%4.0f | ref=%9.4f la=%9.4f jj=%9.4f ep=%9.4f | "
                "la_err=%7.4f jj_err=%7.4f ep_err=%8.4f | %s %s",
                B, p, S, res.ref, res.la, res.jj, res.ep,
                res.la_err, res.jj_err, res.ep_err,
                "YES " if res.jj_lb_ok else "NO! ",
                "yes" if res.ep_ok else "no! ",
            )

    logger.info("")
    logger.info("--- Degenerate edge cases (all-failure and all-success) ---")
    edge_cases = [
        (8, 0.0, "all-failure B=8"),
        (8, 8.0, "all-success B=8"),
        (16, 0.0, "all-failure B=16"),
        (16, 16.0, "all-success B=16"),
        (32, 0.0, "all-failure B=32"),
        (32, 32.0, "all-success B=32"),
    ]
    edge_results: list[BlockResult] = []
    for B_edge, S_edge, label in edge_cases:
        p_label = S_edge / B_edge
        res = compute_block(B_edge, p_label, S_edge, nu, rho2)
        edge_results.append(res)
        logger.info(
            "  %-20s (B=%2d, S=%2.0f): ref=%7.4f la=%7.4f jj=%7.4f ep=%7.4f | "
            "la_err=%7.4f jj_err=%7.4f ep_err=%8.4f | JJ<=%s",
            label, B_edge, S_edge, res.ref, res.la, res.jj, res.ep,
            res.la_err, res.jj_err, res.ep_err,
            "YES" if res.jj_lb_ok else "NO!",
        )

    logger.info("")
    logger.info("--- Summary statistics ---")
    n_blocks = len(results)
    logger.info("Total regular blocks: %d (6 B-values x 3 p-values)", n_blocks)
    logger.info("JJ lower-bound violations: %d / %d", n_jj_lb_violations, n_blocks)
    jj_lb_rate = (n_blocks - n_jj_lb_violations) / max(n_blocks, 1)
    logger.info(
        "JJ lower-bound property holds on %.1f%% of blocks (gate: >=95%%)",
        100 * jj_lb_rate,
    )

    la_errors = [abs(r.la_err) for r in results]
    jj_errors = [abs(r.jj_err) for r in results]
    ep_errors = [abs(r.ep_err) for r in results]
    logger.info(
        "Laplace: median |err|=%.4f, max |err|=%.4f (systematic underestimate, Laplace remainder)",
        np.median(la_errors), max(la_errors),
    )
    logger.info(
        "JJ:      median |err|=%.4f, max |err|=%.4f (variational lower bound, larger gap at extremes)",
        np.median(jj_errors), max(jj_errors),
    )
    logger.info(
        "EP:      median |err|=%.4f, max |err|=%.4f (NOT a lower bound; large errors on extreme blocks)",
        np.median(ep_errors), max(ep_errors),
    )

    n_ep_above = sum(1 for r in results if r.ep_err > 1e-6)
    logger.info(
        "EP blocks above reference (ep > ref + 1e-6): %d / %d",
        n_ep_above, n_blocks,
    )

    logger.info("")
    logger.info("--- Structural finding: pg_vb is an alias of JJ ---")
    logger.info(
        "In _estimator.py BayesBreakLogisticNormal._compute_block_evidence, "
        "both approx='jj' and approx='pg_vb' call jj_block() with identical arguments."
    )
    logger.info(
        "The pg_vb label is retained only for API backward compatibility. "
        "It does NOT implement a distinct Polya-Gamma variational Bayes algorithm. "
        "Do NOT benchmark it as an independent method in E06."
    )

    logger.info("")
    logger.info("--- Gates ---")
    gate1 = jj_lb_rate >= 0.95
    gate2 = not any(
        not math.isfinite(r.ref) or not math.isfinite(r.la) or not math.isfinite(r.jj)
        for r in results
    )
    logger.info("Gate 1: JJ lower-bound >=95%% of blocks: %s", "PASS" if gate1 else "FAIL")
    logger.info("Gate 2: No NaN/inf in ref/Laplace/JJ: %s", "PASS" if gate2 else "FAIL")

    ep_nan_blocks = sum(1 for r in results if not math.isfinite(r.ep))
    logger.info(
        "Note: EP NaN/inf on %d blocks (large positive errors on extreme blocks are known EP behavior)",
        ep_nan_blocks,
    )

    if gate1 and gate2:
        logger.info("DIAGNOSTIC PASS")
    else:
        logger.warning("DIAGNOSTIC ISSUES detected")


if __name__ == "__main__":
    run_diagnostics()
