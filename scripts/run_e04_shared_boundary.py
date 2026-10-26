"""E04: common-changepoint pooling via shared-boundary segmentation.

Protocol : .claude/skills/bayesbreak-release-audit/protocols/experiments.json §E04
Config   : .claude/skills/bayesbreak-release-audit/protocols/experiments.json

Design (from protocol):
  n=240, S in {1,4,16,64}. Gaussian sequences conditionally on a common
  three-segment template with boundaries at [80,160]. nonshared_fractions in
  {0, 0.25, 0.5}: fraction of sequences that use shifted subject-specific
  boundaries instead of the common template.
  Block means (0, 1.5, 0), sigma2=1, Normal(0,4) prior, k_max=8.
  Compare SharedBoundary (pooled) vs Independent (per-sequence) fits.
  100 datasets per cell, geometric-decay-log4 count prior.
  Sequence-level bootstrap resampling (2000 resamples).

Produces:
  results/confirmatory/E04_shared_boundary/<config_sha256[:16]>/
    manifest.json, metrics.json, metrics.csv, oracle_checks.json,
    figures/fig_E04_pooled.pdf/.png,
    tables/tab_E04_comparison.tex,
    run_log.txt, failed_runs.csv, replicates.csv

Usage (confirmatory):
    uv run python scripts/run_e04_shared_boundary.py

Usage (diagnostic pilot, 5 datasets per cell):
    uv run python scripts/run_e04_shared_boundary.py --pilot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# E04 configuration (frozen before first confirmatory run)
# ---------------------------------------------------------------------------

E04_CONFIG: dict[str, Any] = {
    "id": "E04",
    "name": "pooled_shared_boundaries",
    "protocol_revision": "2026-10-21-2",
    "type": "simulation",
    "n": 240,
    "S_values": [1, 4, 16, 64],
    "nonshared_fractions": [0, 0.25, 0.5],
    "true_boundaries": [80, 160],
    "k_true": 3,
    "gaussian_sigma2": 1.0,
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0},
    "block_means": [0.0, 1.5, 0.0],
    "k_max": 8,
    "replicates": 100,
    "seed_base": [20261021, 4],
    "prior": "geometric_decay_log4",
    "bootstrap_resamples": 2000,
    "boundary_tolerances": [0, 1, 3, 5],
    "boundary_tolerance_primary": 3,
    # Shift applied to boundaries for non-shared sequences (in index units)
    "nonshared_boundary_shift": 20,
    "methods": ["pooled", "independent"],
}

E04_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E04_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E04_shared_boundary"
    / E04_CONFIG_SHA256[:16]
)
FIGURES_DIR = RESULTS_BASE / "figures"
TABLES_DIR = RESULTS_BASE / "tables"

# Make _style.py importable from scripts/figures/
_STYLE_DIR = Path(__file__).parent / "figures"
if str(_STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(_STYLE_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _setup_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Count prior
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    """Geometric-decay count prior: p(k) ∝ exp(-log(4)*(k-1))."""
    return math.exp(-math.log(4.0) * (k - 1))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_multi_sequence_dataset(
    n: int,
    S: int,
    nonshared_fraction: float,
    true_boundaries: list[int],
    block_means: list[float],
    sigma2: float,
    nonshared_shift: int,
    rng: np.random.Generator,
) -> dict:
    """Generate S Gaussian sequences with possibly shifted boundaries.

    The first floor(S * (1 - nonshared_fraction)) sequences use the common
    template. The remainder use boundaries shifted by nonshared_shift.

    Returns dict with:
      - Y: shape (n, S) observations
      - boundaries_per_seq: list of length S, each a list [0, b1, b2, n]
      - common_boundaries: the shared template interior boundaries
    """
    n_shared = int(round(S * (1.0 - nonshared_fraction)))
    n_nonshared = S - n_shared

    b1, b2 = true_boundaries
    common_bint = [b1, b2]
    shift = nonshared_shift
    # Shifted boundaries: clamp to [1, n-1]
    b1_ns = max(1, min(n - 1, b1 + shift))
    b2_ns = max(b1_ns + 1, min(n - 1, b2 + shift))
    nonshared_bint = [b1_ns, b2_ns]

    Y = np.zeros((n, S), dtype=float)
    boundaries_per_seq: list[list[int]] = []

    sigma = math.sqrt(sigma2)
    for s in range(S):
        if s < n_shared:
            bint = common_bint
        else:
            bint = nonshared_bint

        bb0, bb1, bb2 = 0, bint[0], bint[1]
        true_signal = np.empty(n, dtype=float)
        true_signal[:bb1] = block_means[0]
        true_signal[bb1:bb2] = block_means[1]
        true_signal[bb2:] = block_means[2]

        Y[:, s] = true_signal + rng.standard_normal(n) * sigma
        boundaries_per_seq.append([0, bb1, bb2, n])

    return {
        "Y": Y,
        "boundaries_per_seq": boundaries_per_seq,
        "common_boundaries": [0] + common_bint + [n],
        "n_shared": n_shared,
        "n_nonshared": n_nonshared,
        "S": S,
        "n": n,
        "k_true": 3,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_pooled_inference(
    dataset: dict,
    k_max: int,
    sigma2: float,
    nu: float,
    rho2: float,
) -> dict:
    """Fit SharedBoundaryMultivariateSegmenter."""
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak.multivariate import SharedBoundaryMultivariateSegmenter
    from bayesbreak import dp as _dp
    from sklearn.base import clone

    n = dataset["n"]
    S = dataset["S"]
    Y = dataset["Y"]
    X = np.arange(n, dtype=float).reshape(-1, 1)

    base_est = BayesBreakGaussian(
        k_max=k_max,
        estimate_hyper=False,
        nu=nu,
        rho2=rho2,
        sigma2=sigma2,
        prior_k=_geom_prior,
    )

    # SharedBoundaryMultivariateSegmenter uses uniform p(k) by default.
    # We need to pass log_p_k explicitly to use geom_prior.
    # Since the class does not expose prior_k, we use a subclass or reconstruct.
    # For E04 correctness, we use the production interface directly and document
    # the uniform p(k) limitation.
    multi_est = SharedBoundaryMultivariateSegmenter(base_estimator=base_est, k_max=k_max)
    multi_est.fit(X, Y)

    # Compute boundary-marginalized posteriors using joint DP
    # (We need to rebuild lA0_joint to get marginalized-over-k boundary marginals)
    w_ones = np.ones(n, dtype=float)
    lA0_joint = None
    from sklearn.base import clone as sk_clone
    for s in range(S):
        est_s = sk_clone(base_est)
        est_s.likelihood_power_ = w_ones
        hyper_s = est_s._estimate_hyperparameters(Y[:, s], w_ones)
        lA0_s, _ = est_s._compute_block_evidence(Y[:, s], hyper_s, w_ones)
        lA0_joint = lA0_s.copy() if lA0_joint is None else lA0_joint + lA0_s

    log_left_j, log_right_j = _dp.forward_backward(lA0_joint, n, k_max)

    # Posterior over k with geometric prior
    log_p_k_full = base_est._build_log_p_k(k_max)
    log_C_k = _dp.compute_log_C_k(None, n, k_max)
    log_post_k_j, post_k_j, log_ev_j = _dp.posterior_over_k(
        log_left_j, n, k_max, log_C_k=log_C_k, log_p_k=log_p_k_full,
    )

    # Marginalized boundary event posteriors
    bm_margk = _dp.boundary_event_marginals_marginalised(
        log_left_j, log_right_j, log_post_k_j, n, k_max
    )

    # MAP boundaries and k_map from geometric prior
    valid = np.arange(1, k_max + 1)[np.isfinite(log_post_k_j)]
    k_map_geom = int(valid[int(np.argmax(log_post_k_j[valid - 1]))])
    map_boundaries_geom, _ = _dp.max_sum_segmentation(lA0_joint, k_map_geom)

    return {
        "k_map": k_map_geom,
        "map_boundaries": list(map_boundaries_geom),
        "k_posterior": post_k_j,
        "bm_margk": bm_margk,
        "log_evidence": float(log_ev_j),
    }


def _run_independent_inference(
    dataset: dict,
    k_max: int,
    sigma2: float,
    nu: float,
    rho2: float,
) -> list[dict]:
    """Fit BayesBreakGaussian independently to each sequence."""
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak.dp import boundary_event_marginals_marginalised

    n = dataset["n"]
    S = dataset["S"]
    Y = dataset["Y"]
    X = np.arange(n, dtype=float).reshape(-1, 1)

    results = []
    for s in range(S):
        est = BayesBreakGaussian(
            k_max=k_max,
            estimate_hyper=False,
            nu=nu,
            rho2=rho2,
            sigma2=sigma2,
            prior_k=_geom_prior,
        )
        est.fit(X, Y[:, s])
        bm_margk = boundary_event_marginals_marginalised(
            est.log_left_, est.log_right_, est.log_posterior_k_, n, k_max
        )
        results.append({
            "k_map": int(est.k_map_),
            "map_boundaries": list(est.map_boundaries_),
            "k_posterior": est.k_posterior_.copy(),
            "bm_margk": bm_margk,
            "log_evidence": float(est.log_evidence_),
        })
    return results


# ---------------------------------------------------------------------------
# Boundary matching and F1
# ---------------------------------------------------------------------------

def _boundary_match(bhat: list[int], bstar: list[int], tau: int) -> tuple[int, float]:
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
    matched_pred: set[int] = set()
    matched_true: set[int] = set()
    edges.sort(key=lambda e: (e[2], e[0], e[1]))
    n_matched = 0
    total_disp = 0.0
    for (i, j, d) in edges:
        if i not in matched_pred and j not in matched_true:
            matched_pred.add(i)
            matched_true.add(j)
            n_matched += 1
            total_disp += d
    return n_matched, total_disp


def _compute_f1(bhat: list[int], bstar: list[int], tau: int) -> tuple[float, float, float]:
    if not bstar and not bhat:
        return 1.0, 1.0, 1.0
    if not bstar:
        return 0.0, 1.0, 0.0
    if not bhat:
        return 1.0, 0.0, 0.0
    n_m, _ = _boundary_match(bhat, bstar, tau)
    prec = n_m / len(bhat) if bhat else 0.0
    rec = n_m / len(bstar) if bstar else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_replicate_metrics(
    dataset: dict,
    pooled_result: dict,
    indep_results: list[dict],
) -> dict:
    """Compute metrics for one replicate."""
    taus = E04_CONFIG["boundary_tolerances"]
    tau_primary = E04_CONFIG["boundary_tolerance_primary"]

    n = dataset["n"]
    S = dataset["S"]
    common_bint = [b for b in dataset["common_boundaries"][1:-1]]

    # ---- Pooled method metrics ----
    pooled_bint = [b for b in pooled_result["map_boundaries"][1:-1]]
    pooled_f1_tol = {}
    for tau in taus:
        _, _, f1 = _compute_f1(pooled_bint, common_bint, tau)
        pooled_f1_tol[tau] = f1

    pooled_f1 = pooled_f1_tol[tau_primary]
    k_true = dataset["k_true"]
    pooled_count_exact = int(pooled_result["k_map"] == k_true)
    pooled_count_err = abs(pooled_result["k_map"] - k_true)
    pooled_p_k_true = float(pooled_result["k_posterior"][k_true - 1]) if k_true <= len(pooled_result["k_posterior"]) else 0.0

    # Brier score for pooled boundary posteriors against common boundaries
    bm_p = pooled_result["bm_margk"]
    true_b_vec = np.zeros(n - 1, dtype=float)
    for b in common_bint:
        if 1 <= b <= n - 1:
            true_b_vec[b - 1] = 1.0
    pooled_brier = float(np.mean((bm_p - true_b_vec) ** 2))

    # ---- Independent method metrics (averaged over sequences, against their own truth) ----
    indep_f1_list = []
    indep_k_exact_list = []
    indep_k_err_list = []
    for s, ir in enumerate(indep_results):
        # Use the actual boundary for this sequence as truth
        bstar_s = [b for b in dataset["boundaries_per_seq"][s][1:-1]]
        bhat_s = [b for b in ir["map_boundaries"][1:-1]]
        _, _, f1_s = _compute_f1(bhat_s, bstar_s, tau_primary)
        indep_f1_list.append(f1_s)
        indep_k_exact_list.append(int(ir["k_map"] == k_true))
        indep_k_err_list.append(abs(ir["k_map"] - k_true))

    indep_f1 = float(np.mean(indep_f1_list))
    indep_count_exact = float(np.mean(indep_k_exact_list))
    indep_count_err = float(np.mean(indep_k_err_list))

    # Pooled signal MSE: average MAP curve MSE over all sequences
    # (Use pooled MAP boundaries to compute segment means per sequence, then MSE)
    pooled_map_bint_full = pooled_result["map_boundaries"]
    pooled_signal_mse_list = []
    for s in range(S):
        y_s = dataset["Y"][:, s]
        bstar_s = dataset["boundaries_per_seq"][s]
        # True signal
        true_signal = np.empty(n, dtype=float)
        for seg_idx, (a, b) in enumerate(zip(bstar_s[:-1], bstar_s[1:])):
            true_signal[a:b] = E04_CONFIG["block_means"][seg_idx]
        # Pooled MAP curve: posterior mean on each MAP segment
        map_curve = np.zeros(n, dtype=float)
        nu = E04_CONFIG["gaussian_prior"]["nu"]
        rho2 = E04_CONFIG["gaussian_prior"]["rho2"]
        sigma2 = E04_CONFIG["gaussian_sigma2"]
        for a, b in zip(pooled_map_bint_full[:-1], pooled_map_bint_full[1:]):
            n_seg = b - a
            rho2_n = 1.0 / (1.0 / rho2 + n_seg / sigma2)
            nu_n = rho2_n * (nu / rho2 + np.sum(y_s[a:b]) / sigma2)
            map_curve[a:b] = nu_n
        pooled_signal_mse_list.append(float(np.mean((map_curve - true_signal) ** 2)))

    pooled_signal_mse = float(np.mean(pooled_signal_mse_list))

    # Independent signal MSE
    indep_signal_mse_list = []
    for s, ir in enumerate(indep_results):
        y_s = dataset["Y"][:, s]
        bstar_s = dataset["boundaries_per_seq"][s]
        true_signal = np.empty(n, dtype=float)
        for seg_idx, (a, b) in enumerate(zip(bstar_s[:-1], bstar_s[1:])):
            true_signal[a:b] = E04_CONFIG["block_means"][seg_idx]
        map_curve = np.zeros(n, dtype=float)
        nu = E04_CONFIG["gaussian_prior"]["nu"]
        rho2 = E04_CONFIG["gaussian_prior"]["rho2"]
        sigma2 = E04_CONFIG["gaussian_sigma2"]
        for a, b in zip(ir["map_boundaries"][:-1], ir["map_boundaries"][1:]):
            n_seg = b - a
            rho2_n = 1.0 / (1.0 / rho2 + n_seg / sigma2)
            nu_n = rho2_n * (nu / rho2 + np.sum(y_s[a:b]) / sigma2)
            map_curve[a:b] = nu_n
        indep_signal_mse_list.append(float(np.mean((map_curve - true_signal) ** 2)))

    indep_signal_mse = float(np.mean(indep_signal_mse_list))

    return {
        "pooled_f1": pooled_f1,
        "pooled_f1_at_0": pooled_f1_tol.get(0, float("nan")),
        "pooled_f1_at_1": pooled_f1_tol.get(1, float("nan")),
        "pooled_f1_at_5": pooled_f1_tol.get(5, float("nan")),
        "pooled_count_exact": pooled_count_exact,
        "pooled_count_err": pooled_count_err,
        "pooled_p_k_true": pooled_p_k_true,
        "pooled_brier": pooled_brier,
        "pooled_signal_mse": pooled_signal_mse,
        "pooled_k_map": pooled_result["k_map"],
        "indep_f1": indep_f1,
        "indep_count_exact": indep_count_exact,
        "indep_count_err": indep_count_err,
        "indep_signal_mse": indep_signal_mse,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(values[idx]))
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def _aggregate_cell(rep_metrics: list[dict], cell: dict, boot_rng: np.random.Generator, n_boot: int) -> dict:
    n_reps = len(rep_metrics)
    if n_reps == 0:
        return {}

    def _arr(key: str) -> np.ndarray:
        vals = [m[key] for m in rep_metrics if not (isinstance(m[key], float) and math.isnan(m[key]))]
        return np.array(vals, dtype=float)

    scalar_keys = [
        "pooled_f1", "pooled_f1_at_0", "pooled_f1_at_1", "pooled_f1_at_5",
        "pooled_count_exact", "pooled_count_err", "pooled_p_k_true",
        "pooled_brier", "pooled_signal_mse",
        "indep_f1", "indep_count_exact", "indep_count_err", "indep_signal_mse",
    ]

    summary: dict[str, Any] = {
        "cell_index": cell["cell_index"],
        "S": cell["S"],
        "nonshared_fraction": cell["nonshared_fraction"],
        "n_replicates": n_reps,
    }

    for key in scalar_keys:
        arr = _arr(key)
        mean_val = float(np.mean(arr)) if len(arr) > 0 else float("nan")
        lo, hi = _bootstrap_ci(arr, n_boot, boot_rng)
        summary[f"mean_{key}"] = mean_val
        summary[f"{key}_ci_lo"] = lo
        summary[f"{key}_ci_hi"] = hi

    return summary


# ---------------------------------------------------------------------------
# Oracle checks (prior normalizer verification for the multivariate DP)
# ---------------------------------------------------------------------------

def _run_oracle_checks() -> dict:
    """Quick oracle: brute-force vs DP for n=8, S=2, k_max=3."""
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak.multivariate import SharedBoundaryMultivariateSegmenter
    from bayesbreak import dp as _dp
    from sklearn.base import clone as sk_clone

    n_small = 8
    S_small = 2
    k_max_check = 3
    sigma2 = E04_CONFIG["gaussian_sigma2"]
    nu = E04_CONFIG["gaussian_prior"]["nu"]
    rho2 = E04_CONFIG["gaussian_prior"]["rho2"]

    rng = np.random.default_rng([20261021, 4, 0, 0])
    Y = np.zeros((n_small, S_small), dtype=float)
    for s in range(S_small):
        Y[:, s] = rng.standard_normal(n_small) + np.where(np.arange(n_small) < 4, 0.0, 2.0)

    # Brute-force
    def log_nn_marg(y, s2, nu_, rho2_):
        n_ = len(y)
        if n_ == 0:
            return 0.0
        rho2_n = 1.0 / (1.0 / rho2_ + n_ / s2)
        nu_n = rho2_n * (nu_ / rho2_ + np.sum(y) / s2)
        return (
            -0.5 * n_ * math.log(2.0 * math.pi * s2)
            - 0.5 * np.sum(y ** 2) / s2
            + 0.5 * math.log(rho2_n / rho2_)
            + 0.5 * nu_n ** 2 / rho2_n
            - 0.5 * nu_ ** 2 / rho2_
        )

    all_parts = []
    for k in range(1, k_max_check + 1):
        for interior in combinations(range(1, n_small), k - 1):
            all_parts.append([0] + list(interior) + [n_small])

    prior_vals = np.array([_geom_prior(k) for k in range(1, k_max_check + 1)])
    log_prior_k = np.log(prior_vals / prior_vals.sum())

    # Compute L_k = sum_t prod A_s(t) per k (matching production formula)
    # Production: log P(y|k) = log L_k(n) - log C_k
    # log C_k = log C(n-1, k-1) for uniform partition prior
    from bayesbreak.utils import log_binom as _log_binom

    log_Lk_bf = {}
    for k in range(1, k_max_check + 1):
        k_parts = [p for p in all_parts if len(p) - 1 == k]
        log_ws = []
        for part in k_parts:
            lw = sum(
                log_nn_marg(Y[a:b, s], sigma2, nu, rho2)
                for s in range(S_small) for a, b in zip(part[:-1], part[1:])
            )
            log_ws.append(lw)
        finite = [lw for lw in log_ws if math.isfinite(lw)]
        if not finite:
            log_Lk_bf[k] = -math.inf
        else:
            max_lw = max(finite)
            log_Lk_bf[k] = max_lw + math.log(sum(math.exp(lw - max_lw) for lw in finite))

    # log P(y,k) = log p(k) + log L_k(n) - log C_k
    log_C_k_vals = {k: float(_log_binom(n_small - 1, k - 1)) for k in range(1, k_max_check + 1)}
    log_py_k = np.array([
        log_prior_k[k - 1] + log_Lk_bf[k] - log_C_k_vals[k]
        for k in range(1, k_max_check + 1)
    ], dtype=float)
    finite_pyk = log_py_k[np.isfinite(log_py_k)]
    if len(finite_pyk) == 0:
        logger.error("Oracle: no finite log_py_k values")
        return {"gate_pass": False, "max_bm_deviation": float("inf"), "log_evidence_deviation": float("inf")}
    max_pyk = float(np.max(finite_pyk))
    log_Z = max_pyk + math.log(sum(math.exp(v - max_pyk) for v in finite_pyk))
    log_post_k_bf = log_py_k - log_Z

    # Marginalized boundary event posteriors using log_post_k_bf weights
    bf_bm = np.zeros(n_small - 1, dtype=float)
    for k in range(1, k_max_check + 1):
        if not math.isfinite(log_post_k_bf[k - 1]):
            continue
        p_k = math.exp(log_post_k_bf[k - 1])
        k_parts = [p for p in all_parts if len(p) - 1 == k]
        # Compute p(partition | k, y)
        log_ws_k = []
        for part in k_parts:
            lw = sum(
                log_nn_marg(Y[a:b, s], sigma2, nu, rho2)
                for s in range(S_small) for a, b in zip(part[:-1], part[1:])
            )
            log_ws_k.append(lw)
        if not any(math.isfinite(lw) for lw in log_ws_k):
            continue
        max_wk = max(lw for lw in log_ws_k if math.isfinite(lw))
        log_Z_k = max_wk + math.log(sum(math.exp(lw - max_wk) for lw in log_ws_k if math.isfinite(lw)))
        for i, part in enumerate(k_parts):
            if not math.isfinite(log_ws_k[i]):
                continue
            p_part_given_k = math.exp(log_ws_k[i] - log_Z_k)
            for b in part[1:-1]:
                bf_bm[b - 1] += p_k * p_part_given_k

    # Production (with geom prior)
    base_est = BayesBreakGaussian(
        k_max=k_max_check,
        estimate_hyper=False,
        nu=nu, rho2=rho2, sigma2=sigma2,
        prior_k=_geom_prior,
    )
    w_ones = np.ones(n_small, dtype=float)
    lA0_joint = None
    for s in range(S_small):
        est_s = sk_clone(base_est)
        est_s.likelihood_power_ = w_ones
        hyper_s = est_s._estimate_hyperparameters(Y[:, s], w_ones)
        lA0_s, _ = est_s._compute_block_evidence(Y[:, s], hyper_s, w_ones)
        lA0_joint = lA0_s.copy() if lA0_joint is None else lA0_joint + lA0_s

    log_left_j, log_right_j = _dp.forward_backward(lA0_joint, n_small, k_max_check)
    log_p_k_full = base_est._build_log_p_k(k_max_check)
    log_C_k = _dp.compute_log_C_k(None, n_small, k_max_check)
    log_post_k_j, post_k_j, log_ev_j = _dp.posterior_over_k(
        log_left_j, n_small, k_max_check, log_C_k=log_C_k, log_p_k=log_p_k_full
    )
    bm_margk = _dp.boundary_event_marginals_marginalised(
        log_left_j, log_right_j, log_post_k_j, n_small, k_max_check
    )

    max_bm_dev = float(np.max(np.abs(bf_bm - bm_margk)))
    log_ev_dev = abs(log_Z - log_ev_j)

    logger.info(
        "Oracle check (n=%d, S=%d, k_max=%d): max_bm_dev=%.3e, log_ev_dev=%.3e",
        n_small, S_small, k_max_check, max_bm_dev, log_ev_dev,
    )

    return {
        "n_small": n_small,
        "S_small": S_small,
        "k_max_check": k_max_check,
        "max_bm_deviation": max_bm_dev,
        "log_evidence_deviation": log_ev_dev,
        "gate_pass": max_bm_dev < 1e-10 and log_ev_dev < 1e-10,
    }


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _generate_latex_table(cell_summaries: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{E04 shared-boundary pooling: pooled (SharedBoundary) vs independent "
        r"(per-sequence) F1@3 and count-exact rate. "
        r"95\% percentile bootstrap intervals (2000 resamples, sequence-level).}",
        r"\label{tab:planned-E04-paper}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"$S$ & Nonshared \% & Method & Mean F1@3 (95\% CI) & Count Exact (95\% CI) & Signal MSE (95\% CI) \\",
        r"\midrule",
    ]

    def fmt(v: float) -> str:
        return f"{v:.3f}" if math.isfinite(v) else "---"

    for S in E04_CONFIG["S_values"]:
        for nf in E04_CONFIG["nonshared_fractions"]:
            cs = next((c for c in cell_summaries if c.get("S") == S and abs(c.get("nonshared_fraction", -1) - nf) < 1e-9), None)
            if cs is None:
                continue

            for method, f1_key, ce_key, mse_key in [
                ("Pooled", "pooled_f1", "pooled_count_exact", "pooled_signal_mse"),
                ("Independent", "indep_f1", "indep_count_exact", "indep_signal_mse"),
            ]:
                f1 = cs.get(f"mean_{f1_key}", float("nan"))
                f1_lo = cs.get(f"{f1_key}_ci_lo", float("nan"))
                f1_hi = cs.get(f"{f1_key}_ci_hi", float("nan"))
                ce = cs.get(f"mean_{ce_key}", float("nan"))
                ce_lo = cs.get(f"{ce_key}_ci_lo", float("nan"))
                ce_hi = cs.get(f"{ce_key}_ci_hi", float("nan"))
                mse = cs.get(f"mean_{mse_key}", float("nan"))
                mse_lo = cs.get(f"{mse_key}_ci_lo", float("nan"))
                mse_hi = cs.get(f"{mse_key}_ci_hi", float("nan"))

                line = (
                    f"{S} & {int(nf * 100)}\\% & {method} & "
                    f"{fmt(f1)} [{fmt(f1_lo)}, {fmt(f1_hi)}] & "
                    f"{fmt(ce)} [{fmt(ce_lo)}, {fmt(ce_hi)}] & "
                    f"{fmt(mse)} [{fmt(mse_lo)}, {fmt(mse_hi)}] \\\\"
                )
                lines.append(line)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved LaTeX table to %s", output_path)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _make_figures(cell_summaries: list[dict], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    S_values = E04_CONFIG["S_values"]
    nfs = E04_CONFIG["nonshared_fractions"]
    nf_labels = {0: "0% non-shared", 0.25: "25% non-shared", 0.5: "50% non-shared"}
    colors = {0: "#1f77b4", 0.25: "#ff7f0e", 0.5: "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("E04 Shared-Boundary Pooling vs Independent Fits", fontsize=10)

    x_ticks = np.arange(len(S_values))
    x_labels = [f"S={s}" for s in S_values]

    for nf in nfs:
        pooled_f1 = []
        pooled_lo = []
        pooled_hi = []
        indep_f1 = []
        indep_lo = []
        indep_hi = []
        for S in S_values:
            cs = next((c for c in cell_summaries if c.get("S") == S and abs(c.get("nonshared_fraction", -1) - nf) < 1e-9), None)
            if cs is not None:
                pooled_f1.append(cs.get("mean_pooled_f1", float("nan")))
                pooled_lo.append(cs.get("pooled_f1_ci_lo", float("nan")))
                pooled_hi.append(cs.get("pooled_f1_ci_hi", float("nan")))
                indep_f1.append(cs.get("mean_indep_f1", float("nan")))
                indep_lo.append(cs.get("indep_f1_ci_lo", float("nan")))
                indep_hi.append(cs.get("indep_f1_ci_hi", float("nan")))
            else:
                for lst in [pooled_f1, pooled_lo, pooled_hi, indep_f1, indep_lo, indep_hi]:
                    lst.append(float("nan"))

        pa = np.array(pooled_f1)
        pl = np.array(pooled_lo)
        ph = np.array(pooled_hi)
        ia = np.array(indep_f1)
        il = np.array(indep_lo)
        ih = np.array(indep_hi)

        valid_p = np.isfinite(pa)
        valid_i = np.isfinite(ia)

        color = colors.get(nf, "black")
        label = nf_labels.get(nf, str(nf))

        if np.any(valid_p):
            axes[0].errorbar(
                x_ticks[valid_p] - 0.1, pa[valid_p],
                yerr=[pa[valid_p] - pl[valid_p], ph[valid_p] - pa[valid_p]],
                label=f"Pooled ({label})", color=color, marker="o", capsize=3,
            )
        if np.any(valid_i):
            axes[0].errorbar(
                x_ticks[valid_i] + 0.1, ia[valid_i],
                yerr=[ia[valid_i] - il[valid_i], ih[valid_i] - ia[valid_i]],
                label=f"Indep ({label})", color=color, marker="s", linestyle="--", capsize=3,
            )

    axes[0].set_title("Mean F1@3 by S and non-shared fraction")
    axes[0].set_xlabel("Number of sequences S")
    axes[0].set_ylabel("Mean F1@3")
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=6)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Pooled signal MSE vs S
    for nf in nfs:
        pooled_mse = []
        mse_lo = []
        mse_hi = []
        for S in S_values:
            cs = next((c for c in cell_summaries if c.get("S") == S and abs(c.get("nonshared_fraction", -1) - nf) < 1e-9), None)
            if cs is not None:
                pooled_mse.append(cs.get("mean_pooled_signal_mse", float("nan")))
                mse_lo.append(cs.get("pooled_signal_mse_ci_lo", float("nan")))
                mse_hi.append(cs.get("pooled_signal_mse_ci_hi", float("nan")))
            else:
                pooled_mse.append(float("nan"))
                mse_lo.append(float("nan"))
                mse_hi.append(float("nan"))

        ma = np.array(pooled_mse)
        ml = np.array(mse_lo)
        mh = np.array(mse_hi)
        valid = np.isfinite(ma)
        if np.any(valid):
            axes[1].errorbar(
                x_ticks[valid], ma[valid],
                yerr=[ma[valid] - ml[valid], mh[valid] - ma[valid]],
                label=f"Pooled ({nf_labels.get(nf, str(nf))})",
                color=colors.get(nf, "black"), marker="o", capsize=3,
            )

    axes[1].set_title("Pooled signal MSE by S and non-shared fraction")
    axes[1].set_xlabel("Number of sequences S")
    axes[1].set_ylabel("Mean signal MSE")
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels)
    axes[1].legend(fontsize=6)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fpath = figures_dir / f"fig_E04_pooled.{ext}"
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
    logger.info("Saved figures to %s", figures_dir / "fig_E04_pooled")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main(pilot: bool = False) -> None:
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RESULTS_BASE / "run_log.txt"
    _setup_logging(log_path)

    mode_str = "PILOT" if pilot else "CONFIRMATORY"
    n_reps = 5 if pilot else E04_CONFIG["replicates"]
    n_boot = E04_CONFIG["bootstrap_resamples"]
    n = E04_CONFIG["n"]
    k_max = E04_CONFIG["k_max"]
    sigma2 = E04_CONFIG["gaussian_sigma2"]
    nu = E04_CONFIG["gaussian_prior"]["nu"]
    rho2 = E04_CONFIG["gaussian_prior"]["rho2"]

    logger.info("=" * 70)
    logger.info("E04 shared-boundary pooling — %s run", mode_str)
    logger.info("Config SHA256: %s", E04_CONFIG_SHA256[:16])
    logger.info("n_reps=%d, n_boot=%d", n_reps, n_boot)
    logger.info("=" * 70)

    git_rev = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            git_rev = result.stdout.strip()
    except Exception:
        pass

    working_tree_dirty = False
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True, cwd=Path(__file__).parent.parent,
        )
        working_tree_dirty = result.returncode != 0
    except Exception:
        pass

    utc_start = datetime.now(tz=timezone.utc)

    # Build cell list: (S, nonshared_fraction) combinations
    cells = []
    cell_index = 0
    for S in E04_CONFIG["S_values"]:
        for nf in E04_CONFIG["nonshared_fractions"]:
            cells.append({
                "cell_index": cell_index,
                "S": S,
                "nonshared_fraction": nf,
                "n_reps": n_reps,
            })
            cell_index += 1

    n_cells = len(cells)
    logger.info("Total cells: %d (%d S values × %d nonshared_fraction values)",
                n_cells, len(E04_CONFIG["S_values"]), len(E04_CONFIG["nonshared_fractions"]))

    # Oracle checks
    logger.info("Running oracle checks...")
    oracle_results = _run_oracle_checks()
    oracle_gate_pass = oracle_results["gate_pass"]
    if not oracle_gate_pass:
        logger.error(
            "ORACLE GATE FAILED: max_bm_dev=%.3e, log_ev_dev=%.3e",
            oracle_results["max_bm_deviation"],
            oracle_results["log_evidence_deviation"],
        )
        sys.exit(1)
    logger.info("Oracle gate: PASS (max_bm_dev=%.3e)", oracle_results["max_bm_deviation"])

    cell_summaries: list[dict] = []
    all_replicate_records: list[dict] = []
    failed_runs: list[dict] = []

    n_cells_completed = 0
    n_cells_failed = 0
    n_replicates_completed = 0
    n_replicates_failed = 0

    for cell_num, cell in enumerate(cells, start=1):
        S = cell["S"]
        nf = cell["nonshared_fraction"]
        cell_idx = cell["cell_index"]
        label = f"S={S},nf={nf}"

        logger.info("[Cell %d/%d] %s", cell_num, n_cells, label)
        start_wall = time.perf_counter()

        rep_metrics: list[dict] = []

        for rep in range(n_reps):
            seed_seq = np.random.SeedSequence(
                [*E04_CONFIG["seed_base"], cell_idx, rep]
            )
            data_seed = seed_seq.spawn(2)[0]
            rng = np.random.default_rng(data_seed)

            try:
                dataset = _generate_multi_sequence_dataset(
                    n=n,
                    S=S,
                    nonshared_fraction=nf,
                    true_boundaries=E04_CONFIG["true_boundaries"],
                    block_means=E04_CONFIG["block_means"],
                    sigma2=sigma2,
                    nonshared_shift=E04_CONFIG["nonshared_boundary_shift"],
                    rng=rng,
                )

                pooled_result = _run_pooled_inference(dataset, k_max, sigma2, nu, rho2)
                indep_results = _run_independent_inference(dataset, k_max, sigma2, nu, rho2)

                metrics = _compute_replicate_metrics(dataset, pooled_result, indep_results)

                record = {
                    "cell_index": cell_idx,
                    "S": S,
                    "nonshared_fraction": nf,
                    "replicate": rep,
                    **{k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()},
                }
                rep_metrics.append(metrics)
                all_replicate_records.append(record)
                n_replicates_completed += 1

                if (rep + 1) % 20 == 0 or n_reps <= 10:
                    logger.info("  Rep %d/%d done", rep + 1, n_reps)

            except Exception as exc:
                logger.error("  FAILED rep=%d: %s", rep, exc, exc_info=True)
                failed_runs.append({
                    "cell_index": cell_idx,
                    "S": S,
                    "nonshared_fraction": nf,
                    "replicate": rep,
                    "error": str(exc),
                })
                n_replicates_failed += 1

        if not rep_metrics:
            logger.warning("Cell %s: all replicates failed!", label)
            n_cells_failed += 1
            continue

        boot_rng = np.random.default_rng([20261021, 4, cell_idx])
        summary = _aggregate_cell(rep_metrics, cell, boot_rng, n_boot)
        cell_summaries.append(summary)
        n_cells_completed += 1

        elapsed = time.perf_counter() - start_wall
        logger.info(
            "  Cell done | elapsed %.1fs | n_rep=%d | pooled_f1=%.3f | indep_f1=%.3f",
            elapsed,
            len(rep_metrics),
            summary.get("mean_pooled_f1", float("nan")),
            summary.get("mean_indep_f1", float("nan")),
        )

    utc_end = datetime.now(tz=timezone.utc)

    # Figures
    logger.info("Generating figures...")
    _make_figures(cell_summaries, FIGURES_DIR)

    # LaTeX table
    logger.info("Generating LaTeX table...")
    _generate_latex_table(cell_summaries, TABLES_DIR / "tab_E04_comparison.tex")

    # Metrics CSV
    metrics_csv_path = RESULTS_BASE / "metrics.csv"
    if cell_summaries:
        all_keys = sorted({k for c in cell_summaries for k in c.keys()})
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in cell_summaries:
                writer.writerow({k: row.get(k, "") for k in all_keys})
        logger.info("Saved metrics CSV to %s", metrics_csv_path)

    # Replicate CSV
    rep_csv_path = RESULTS_BASE / "replicates.csv"
    if all_replicate_records:
        all_rep_keys = sorted({k for r in all_replicate_records for k in r.keys()})
        with open(rep_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rep_keys)
            writer.writeheader()
            for row in all_replicate_records:
                writer.writerow({k: row.get(k, "") for k in all_rep_keys})

    # Failed runs CSV
    failed_csv_path = RESULTS_BASE / "failed_runs.csv"
    if failed_runs:
        with open(failed_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["cell_index", "S", "nonshared_fraction", "replicate", "error"])
            writer.writeheader()
            writer.writerows(failed_runs)
        logger.warning("Failed runs: %d", len(failed_runs))
    else:
        logger.info("No failed replicates.")

    # Oracle checks JSON
    oracle_checks_path = RESULTS_BASE / "oracle_checks.json"
    with open(oracle_checks_path, "w") as f:
        json.dump(oracle_results, f, indent=2)
    logger.info("Saved oracle checks to %s", oracle_checks_path)

    # Gate check
    all_gate_pass = oracle_gate_pass and n_cells_failed == 0
    gate_threshold_oracle = 1e-10

    logger.info("")
    logger.info("=" * 70)
    logger.info("--- Gate checks ---")
    logger.info(
        "Oracle max_bm_dev: %.3e [%s]",
        oracle_results["max_bm_deviation"],
        "PASS" if oracle_results["max_bm_deviation"] < gate_threshold_oracle else "FAIL",
    )
    logger.info(
        "Cells completed: %d/%d [%s]",
        n_cells_completed, n_cells,
        "PASS" if n_cells_failed == 0 else "FAIL",
    )

    # Metrics JSON
    metrics = {
        "experiment_id": "E04",
        "n_cells_planned": n_cells,
        "n_cells_completed": n_cells_completed,
        "n_cells_failed": n_cells_failed,
        "n_replicates_completed": n_replicates_completed,
        "n_replicates_failed": n_replicates_failed,
        "pilot_mode": pilot,
        "gate_pass": all_gate_pass,
        "cell_summaries": cell_summaries,
        "oracle_checks": oracle_results,
    }

    metrics_path = RESULTS_BASE / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # Manifest JSON
    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = "unknown"
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "unknown"

    src_dir = Path(__file__).parent.parent / "src" / "bayesbreak"
    source_file_sha256: dict[str, str] = {}
    for src_file in sorted(src_dir.rglob("*.py")):
        rel = str(src_file.relative_to(Path(__file__).parent.parent))
        with open(src_file, "rb") as f_src:
            source_file_sha256[rel] = hashlib.sha256(f_src.read()).hexdigest()

    manifest = {
        "experiment_id": "E04",
        "protocol_revision": E04_CONFIG["protocol_revision"],
        "config_sha256": E04_CONFIG_SHA256,
        "git_revision": git_rev,
        "working_tree_dirty": working_tree_dirty,
        "source_file_sha256": source_file_sha256,
        "python_version": sys.version,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "seed": E04_CONFIG["seed_base"],
        "input_sha256": "deterministic_generator",
        "source_kind": "gaussian_simulation_shared_boundaries",
        "family": "gaussian",
        "family_descriptor": "gaussian_known_variance",
        "prior": E04_CONFIG["prior"],
        "k_max": E04_CONFIG["k_max"],
        "k_true": E04_CONFIG["k_true"],
        "coordinate_axis": "regular_index_grid",
        "reference_type": "simulation_exact_inference",
        "fit_status": "completed" if n_cells_failed == 0 else "partial",
        "metric_schema": "E04_shared_boundary",
        "utc_start": utc_start.isoformat(),
        "utc_end": utc_end.isoformat(),
        "pilot_mode": pilot,
        "n_cells_planned": n_cells,
        "n_cells_completed": n_cells_completed,
        "limitation_prior_k": (
            "SharedBoundaryMultivariateSegmenter uses uniform p(k) for the joint DP. "
            "The per-sequence prior_k=geom_prior passed to the base_estimator does NOT "
            "propagate to the shared posterior. Pooled inference uses uniform p(k)."
        ),
    }

    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest to %s", manifest_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info("E04 %s complete in %.1fs", mode_str, (utc_end - utc_start).total_seconds())
    logger.info("Cells: %d planned, %d completed, %d failed", n_cells, n_cells_completed, n_cells_failed)
    logger.info("Replicates: %d completed, %d failed", n_replicates_completed, n_replicates_failed)
    logger.info("Config SHA256: %s", E04_CONFIG_SHA256[:16])
    logger.info("Results: %s", RESULTS_BASE)
    logger.info("=" * 70)

    if all_gate_pass:
        logger.info("All gate checks PASSED.")
        logger.info("E04 result: PASS")
    else:
        logger.error("Gate check FAILED — see above.")
        logger.error("E04 result: FAIL (gate check)")
        sys.exit(1)

    logger.info("gate_pass: %s", all_gate_pass)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E04 shared-boundary pooling runner")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode (5 datasets per cell, excluded from confirmatory metrics)",
    )
    args = parser.parse_args()
    main(pilot=args.pilot)
