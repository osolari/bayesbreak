"""E02: observation-family portability.

Protocol : .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E02
Config   : .claude/skills/bayesbreak-release-audit/protocols/experiments.json

Produces:
  results/confirmatory/E02_family_portability/<config_sha256[:16]>/
    manifest.json, metrics.json, metrics.csv, oracle_checks.json,
    figures/fig_E02_recovery.pdf/.png,
    tables/tab_E02_recovery.tex,
    run_log.txt, failed_runs.csv

Usage (confirmatory):
    python scripts/run_e02_family_portability.py

Usage (diagnostic pilot, 5 datasets per cell):
    python scripts/run_e02_family_portability.py --pilot
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
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import nbinom as scipy_nbinom

# ---------------------------------------------------------------------------
# E02 configuration (frozen before first confirmatory run)
# ---------------------------------------------------------------------------

E02_CONFIG: dict[str, Any] = {
    "id": "E02",
    "name": "family_portability",
    "protocol_revision": "2026-10-21-3",
    "type": "simulation",
    # Amendment 2026-10-23: confirmatory quadrature order raised from 64 to 128
    # after diagnostic confirmed 64-node result was marginally over threshold for
    # mu_mid=0.4 rep 2 (curve_dev=1.006e-3 vs threshold 1e-3). 128-node vs 256-node
    # deviations are ~1e-8, well within threshold. Gate comparison is now 128 vs 256.
    "beta_obs_quadrature_points": 128,
    "beta_obs_gate_comparison_points": 256,
    "k_true": 3,
    "k_max": 10,
    "n_conjugate": 240,
    "n_beta_quadrature": 120,
    "replicates_conjugate": 100,
    "replicates_beta": 50,
    "seed_base": [20261021, 2],
    "prior": "geometric_decay_log4",
    "bootstrap_resamples": 2000,
    "boundary_tolerances": [0, 1, 3, 5],
    "boundary_tolerance_primary": 3,
    # Family priors
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0, "sigma2": 1.0},
    "poisson_prior": {"alpha": 2.0, "beta": 1.0},
    "binomial_prior": {"alpha": 2.0, "beta": 3.0},
    "negbin_prior": {"r": 5.0, "alpha": 2.0, "beta": 3.0},
    "beta_obs_prior": {"alpha": 2.0, "beta": 3.0},
    # Cells: 10 total
    "cells": [
        # cell_index, family, n, param_name, param_value, n_reps
        {"cell_index": 0, "family": "gaussian", "n": 240, "param": "delta", "value": 0.5,   "n_reps": 100},
        {"cell_index": 1, "family": "gaussian", "n": 240, "param": "delta", "value": 1.5,   "n_reps": 100},
        {"cell_index": 2, "family": "poisson",  "n": 240, "param": "q",     "value": 1.5,   "n_reps": 100},
        {"cell_index": 3, "family": "poisson",  "n": 240, "param": "q",     "value": 3.0,   "n_reps": 100},
        {"cell_index": 4, "family": "binomial", "n": 240, "param": "p_mid", "value": 0.35,  "n_reps": 100},
        {"cell_index": 5, "family": "binomial", "n": 240, "param": "p_mid", "value": 0.65,  "n_reps": 100},
        {"cell_index": 6, "family": "negbin",   "n": 240, "param": "mu_mid","value": 3.0,   "n_reps": 100},
        {"cell_index": 7, "family": "negbin",   "n": 240, "param": "mu_mid","value": 6.0,   "n_reps": 100},
        {"cell_index": 8, "family": "beta_obs", "n": 120, "param": "mu_mid","value": 0.4,   "n_reps": 50},
        {"cell_index": 9, "family": "beta_obs", "n": 120, "param": "mu_mid","value": 0.8,   "n_reps": 50},
    ],
}

E02_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E02_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E02_family_portability"
    / E02_CONFIG_SHA256[:16]
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
# Section 1: Prior function
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    """Geometric-decay count prior: p(k) ∝ exp(-log(4)*(k-1))."""
    return math.exp(-math.log(4.0) * (k - 1))


# ---------------------------------------------------------------------------
# Section 2: Data generators (one per family)
# ---------------------------------------------------------------------------

def _generate_gaussian(
    n: int, delta: float, rng: np.random.Generator
) -> dict:
    """k_true=3, equal-length segments n/3, means (0, delta, 0), sigma2=1.

    Returns y, x, true_signal, true_means, boundaries, sample_weight=ones.
    """
    seg = n // 3
    assert 3 * seg == n, f"n={n} must be divisible by 3"
    true_means = [0.0, delta, 0.0]
    true_signal = np.empty(n, dtype=float)
    for q in range(3):
        true_signal[q * seg:(q + 1) * seg] = true_means[q]
    y = true_signal + rng.standard_normal(n)
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y,
        "x": x,
        "true_signal": true_signal,
        "true_means": true_means,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": None,
    }


def _generate_poisson(
    n: int, q: float, rng: np.random.Generator
) -> dict:
    """k_true=3, equal-length segments, rates (2, 2*q, 2), exposures cycle {0.5,1,2}.

    sample_weight = exposure array e_i.
    true_signal = rate per unit exposure (lambda_q) per position.
    """
    seg = n // 3
    assert 3 * seg == n
    true_rates = [2.0, 2.0 * q, 2.0]
    # Exposure: cycles {0.5, 1, 2}
    e_pattern = [0.5, 1.0, 2.0]
    e_arr = np.array([e_pattern[i % 3] for i in range(n)], dtype=float)
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_rates[s]
    # Generate: y_i ~ Poisson(lambda_q * e_i)
    y = np.array(
        [rng.poisson(true_rates[i // seg] * e_arr[i]) for i in range(n)],
        dtype=float,
    )
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y,
        "x": x,
        "true_signal": true_signal,  # per-unit-exposure rate (lambda_q)
        "true_rates": true_rates,
        "e_arr": e_arr,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": e_arr,
        "descriptor": {"e_arr": e_arr},
    }


def _generate_binomial(
    n: int, p_mid: float, rng: np.random.Generator
) -> dict:
    """k_true=3, equal-length segments, probs (0.2, p_mid, 0.2), trials cycle {8,32}.

    true_signal = probability p_q per position.
    """
    seg = n // 3
    assert 3 * seg == n
    true_probs = [0.2, p_mid, 0.2]
    m_arr = np.array([8 if i % 2 == 0 else 32 for i in range(n)], dtype=float)
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_probs[s]
    y = np.array(
        [rng.binomial(int(m_arr[i]), true_probs[i // seg]) for i in range(n)],
        dtype=float,
    )
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y,
        "x": x,
        "true_signal": true_signal,  # probability p_q
        "true_probs": true_probs,
        "m_arr": m_arr,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": {"m_arr": m_arr},
    }


def _generate_negbin(
    n: int, mu_mid: float, rng: np.random.Generator
) -> dict:
    """k_true=3, equal-length segments, count means (2, mu_mid, 2), r=5.

    PMF: NegBin(r, p) with p = r/(r+mu), mean = r*(1-p)/p = mu.
    true_signal = count mean mu_q per position.
    """
    seg = n // 3
    assert 3 * seg == n
    r = float(E02_CONFIG["negbin_prior"]["r"])
    true_mus = [2.0, mu_mid, 2.0]
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_mus[s]
    y = np.empty(n, dtype=float)
    for s in range(3):
        mu_q = true_mus[s]
        p_q = r / (r + mu_q)  # scipy p_success: mean = r*(1-p)/p
        y[s * seg:(s + 1) * seg] = scipy_nbinom.rvs(
            n=int(r), p=p_q, size=seg, random_state=rng
        )
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y,
        "x": x,
        "true_signal": true_signal,  # count mean mu_q
        "true_mus": true_mus,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": None,
        "r": r,
    }


def _generate_beta_obs(
    n: int, mu_mid: float, rng: np.random.Generator
) -> dict:
    """k_true=3, equal-length segments, means (0.2, mu_mid, 0.2), phi cycle {20,80}.

    true_signal = mu_q per position.
    """
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
        a_param = phi * mu_q
        b_param = phi * (1.0 - mu_q)
        y[i] = rng.beta(a_param, b_param)
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y,
        "x": x,
        "true_signal": true_signal,  # mu_q
        "true_mus": true_mus,
        "phi_arr": phi_arr,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": {"phi_arr": phi_arr},
    }


def generate_dataset(cell: dict, rng: np.random.Generator) -> dict:
    """Dispatch to the correct generator based on cell['family']."""
    family = cell["family"]
    n = cell["n"]
    param_val = cell["value"]
    if family == "gaussian":
        return _generate_gaussian(n, param_val, rng)
    elif family == "poisson":
        return _generate_poisson(n, param_val, rng)
    elif family == "binomial":
        return _generate_binomial(n, param_val, rng)
    elif family == "negbin":
        return _generate_negbin(n, param_val, rng)
    elif family == "beta_obs":
        return _generate_beta_obs(n, param_val, rng)
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Section 3: Vectorized Bayes regression curve (copied from E01 runner)
# ---------------------------------------------------------------------------

def _logsumexp_axis0(a: NDArray) -> NDArray:
    """Stable log-sum-exp over axis 0. Returns -inf where all inputs are -inf."""
    a_max = np.max(a, axis=0, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=0)
    a_max_sq = np.squeeze(a_max, axis=0)
    with np.errstate(divide="ignore"):
        log_s = np.log(s)
    return np.where(np.isfinite(a_max_sq), a_max_sq + log_s, -np.inf)


def _bayes_curve_fixed_k_vectorized(
    L: NDArray,
    R: NDArray,
    lA0: NDArray,
    A1: NDArray,
    n: int,
    k: int,
) -> NDArray:
    """Vectorized Bayes regression curve for fixed k."""
    denom = float(L[k, n])
    if not np.isfinite(denom):
        return np.full(n, np.nan, dtype=float)

    LL = L[0:k, :]
    RR = R[k - 1::-1, :]

    cross = _logsumexp_axis0(LL[:, :, np.newaxis] + RR[:, np.newaxis, :])
    log_w = cross + lA0 - denom

    finite_mask = np.isfinite(log_w) & np.isfinite(A1) & (A1 != 0.0)
    log_w_clipped = np.where(finite_mask, np.clip(log_w, -745.0, 0.0), -np.inf)
    contrib = np.where(finite_mask, np.exp(log_w_clipped) * A1, 0.0)

    diff = contrib.sum(axis=1) - contrib.sum(axis=0)
    return np.cumsum(diff)[:n]


def _bayes_curve_mix_k_vectorized(
    L: NDArray,
    R: NDArray,
    lA0: NDArray,
    A1: NDArray,
    n: int,
    k_max: int,
    post_k: NDArray,
) -> NDArray:
    """Mixed-k Bayes regression curve: Σ_k P(k|y) E[signal_t|y,k]."""
    out = np.zeros(n, dtype=float)
    for k in range(1, k_max + 1):
        wk = float(post_k[k - 1])
        if wk == 0.0 or not np.isfinite(float(L[k, n])):
            continue
        out += wk * _bayes_curve_fixed_k_vectorized(L, R, lA0, A1, n, k)
    return out


# ---------------------------------------------------------------------------
# Section 4: Inference (per family)
# ---------------------------------------------------------------------------

def run_inference(
    cell: dict,
    dataset: dict,
) -> dict:
    """Run exact posterior inference for one E02 dataset.

    Returns:
      p_k_given_y, log_p_k_given_y, k_marg, p_k_marg, p_k_max,
      log_evidence, boundary_probs_marginal, map_boundaries, bayes_curve
    """
    import bayesbreak.dp as _dp

    family = cell["family"]
    n = cell["n"]
    k_max = E02_CONFIG["k_max"]
    y = dataset["y"]
    x = dataset["x"]
    sw = dataset["sample_weight"]

    if family == "gaussian":
        from bayesbreak.families.gaussian import BayesBreakGaussian
        p = E02_CONFIG["gaussian_prior"]
        est = BayesBreakGaussian(
            k_max=k_max, estimate_hyper=False,
            nu=p["nu"], rho2=p["rho2"], sigma2=p["sigma2"],
            prior_k=_geom_prior,
        )
        est.fit(x.reshape(-1, 1), y)

    elif family == "poisson":
        from bayesbreak.families.poisson import BayesBreakPoisson
        p = E02_CONFIG["poisson_prior"]
        est = BayesBreakPoisson(
            k_max=k_max, estimate_hyper=False,
            alpha=p["alpha"], beta=p["beta"],
            prior_k=_geom_prior,
        )
        est.fit(x.reshape(-1, 1), y, sample_weight=sw)

    elif family == "binomial":
        from bayesbreak.families.binomial import BayesBreakBinomial
        p = E02_CONFIG["binomial_prior"]
        m_arr = dataset["m_arr"]
        est = BayesBreakBinomial(
            k_max=k_max, estimate_hyper=False,
            alpha=p["alpha"], beta=p["beta"],
            n_trials=m_arr,
            prior_k=_geom_prior,
        )
        est.fit(x.reshape(-1, 1), y)

    elif family == "negbin":
        from bayesbreak.families.negative_binomial import BayesBreakNegBin
        p = E02_CONFIG["negbin_prior"]
        est = BayesBreakNegBin(
            k_max=k_max, estimate_hyper=False,
            r=p["r"], alpha=p["alpha"], beta=p["beta"],
            prior_k=_geom_prior,
        )
        est.fit(x.reshape(-1, 1), y)

    elif family == "beta_obs":
        from bayesbreak.families.beta_obs import BayesBreakBetaObs
        p = E02_CONFIG["beta_obs_prior"]
        phi_arr = dataset["phi_arr"]
        est = BayesBreakBetaObs(
            k_max=k_max, estimate_hyper=False,
            alpha=p["alpha"], beta=p["beta"],
            phi=phi_arr,
            quadrature_points=E02_CONFIG["beta_obs_quadrature_points"],
            prior_k=_geom_prior,
        )
        est.fit(x.reshape(-1, 1), y)

    else:
        raise ValueError(f"Unknown family: {family}")

    # Marginalized boundary event probabilities P(b_i=1|y) for i=1..n-1
    bm_marginal = _dp.boundary_event_marginals_marginalised(
        est.log_left_, est.log_right_,
        est.log_posterior_k_, n, k_max,
    )

    # Vectorized mixed-k Bayes regression curve
    bayes_curve = _bayes_curve_mix_k_vectorized(
        est.log_left_, est.log_right_,
        est.log_block_evidence_, est.block_first_moment_,
        n, k_max, est.k_posterior_,
    )

    p_k = est.k_posterior_
    log_p_k = est.log_posterior_k_
    k_marg = int(est.k_map_)
    p_k_marg = float(p_k[k_marg - 1])
    p_k_max = float(p_k[k_max - 1])

    return {
        "p_k_given_y": p_k,
        "log_p_k_given_y": log_p_k,
        "k_marg": k_marg,
        "p_k_marg": p_k_marg,
        "p_k_max": p_k_max,
        "log_evidence": float(est.log_evidence_),
        "boundary_probs_marginal": bm_marginal,
        "map_boundaries": list(est.map_boundaries_),
        "bayes_curve": bayes_curve,
    }


# ---------------------------------------------------------------------------
# Section 5: Beta-obs quadrature convergence check
# ---------------------------------------------------------------------------

def _beta_obs_convergence_check(
    cell: dict,
    dataset: dict,
    n_check: int = 5,
    rng_seed: int = 20261021,
) -> dict:
    """Compare confirmatory quadrature order vs 2x nodes on one replicate.

    Confirmatory order: E02_CONFIG["beta_obs_quadrature_points"] (128).
    Gate comparison: E02_CONFIG["beta_obs_gate_comparison_points"] (256).

    Returns max_log_evidence_deviation and max_bayes_curve_deviation (128 vs 256).
    """
    from bayesbreak.families.beta_obs import BayesBreakBetaObs

    p = E02_CONFIG["beta_obs_prior"]
    phi_arr = dataset["phi_arr"]
    n = cell["n"]
    x = dataset["x"]
    y = dataset["y"]
    k_max = E02_CONFIG["k_max"]
    q_confirmatory = E02_CONFIG["beta_obs_quadrature_points"]
    q_gate = E02_CONFIG["beta_obs_gate_comparison_points"]

    est_conf = BayesBreakBetaObs(
        k_max=k_max, estimate_hyper=False,
        alpha=p["alpha"], beta=p["beta"],
        phi=phi_arr, quadrature_points=q_confirmatory,
        prior_k=_geom_prior,
    )
    est_gate = BayesBreakBetaObs(
        k_max=k_max, estimate_hyper=False,
        alpha=p["alpha"], beta=p["beta"],
        phi=phi_arr, quadrature_points=q_gate,
        prior_k=_geom_prior,
    )
    est_conf.fit(x.reshape(-1, 1), y)
    est_gate.fit(x.reshape(-1, 1), y)

    dev_ev = abs(float(est_conf.log_evidence_) - float(est_gate.log_evidence_))
    c_conf = _bayes_curve_mix_k_vectorized(
        est_conf.log_left_, est_conf.log_right_,
        est_conf.log_block_evidence_, est_conf.block_first_moment_,
        n, k_max, est_conf.k_posterior_,
    )
    c_gate = _bayes_curve_mix_k_vectorized(
        est_gate.log_left_, est_gate.log_right_,
        est_gate.log_block_evidence_, est_gate.block_first_moment_,
        n, k_max, est_gate.k_posterior_,
    )
    dev_curve = float(np.max(np.abs(c_conf - c_gate)))

    return {
        "max_log_evidence_deviation": float(dev_ev),
        "max_bayes_curve_deviation": float(dev_curve),
        "n_reps_checked": 1,
        "quadrature_comparison": f"{q_confirmatory}_vs_{q_gate}",
    }


# ---------------------------------------------------------------------------
# Section 6: Boundary matching (bipartite, one-to-one, max cardinality)
# ---------------------------------------------------------------------------

def _boundary_match(
    bhat: list[int],
    bstar: list[int],
    tau: int,
) -> tuple[int, float]:
    """One-to-one maximum-cardinality boundary match with min-displacement."""
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


def compute_f1_metrics(
    map_boundaries: list[int],
    true_boundaries: list[int],
    taus: list[int],
) -> dict:
    """Compute F1, precision, recall for all tolerances."""
    n_total = true_boundaries[-1]
    bhat = sorted(b for b in map_boundaries if 0 < b < n_total)
    bstar = sorted(b for b in true_boundaries if 0 < b < n_total)

    out: dict[str, float] = {}
    for tau in taus:
        tp, total_disp = _boundary_match(bhat, bstar, tau)
        fp = len(bhat) - tp
        fn = len(bstar) - tp
        denom_p = tp + fp
        denom_r = tp + fn
        denom_f = 2 * tp + fp + fn
        prec = tp / denom_p if denom_p > 0 else float("nan")
        rec = tp / denom_r if denom_r > 0 else float("nan")
        f1 = 2 * tp / denom_f if denom_f > 0 else float("nan")
        out[f"tp_at_{tau}"] = float(tp)
        out[f"fp_at_{tau}"] = float(fp)
        out[f"fn_at_{tau}"] = float(fn)
        out[f"precision_at_{tau}"] = prec
        out[f"recall_at_{tau}"] = rec
        out[f"f1_at_{tau}"] = f1
        out[f"mae_at_{tau}"] = total_disp / tp if tp > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Section 7: Per-replicate metrics
# ---------------------------------------------------------------------------

def compute_replicate_metrics(
    cell: dict,
    dataset: dict,
    inference: dict,
    taus: list[int],
) -> dict:
    """Compute all E02 metrics for one (dataset, inference) pair."""
    n = cell["n"]
    k_true = E02_CONFIG["k_true"]
    true_boundaries = dataset["boundaries"]
    true_signal = dataset["true_signal"]
    k_max = E02_CONFIG["k_max"]

    # 1. F1 and boundary detection
    boundary_metrics = compute_f1_metrics(
        inference["map_boundaries"],
        true_boundaries,
        taus,
    )

    # 2. Count metrics
    k_marg = inference["k_marg"]
    count_exact = int(k_marg == k_true)
    count_abs_error = abs(k_marg - k_true)

    p_k = inference["p_k_given_y"]
    if 1 <= k_true <= k_max:
        p_k_true = float(p_k[k_true - 1])
    else:
        p_k_true = 0.0

    p_k_max = inference["p_k_max"]

    # 3. Signal MSE (Bayes regression curve vs true signal)
    bayes_curve = inference["bayes_curve"]
    signal_mse = float(np.mean((bayes_curve - true_signal) ** 2))

    # 4. Brier score: boundary event probabilities P(b_i=1|y) for i=1..n-1
    bm = inference["boundary_probs_marginal"]  # length n-1
    true_interior = set(b for b in true_boundaries if 0 < b < n)
    z = np.array([float(i in true_interior) for i in range(1, n)], dtype=float)
    brier = float(np.mean((bm - z) ** 2))

    return {
        **boundary_metrics,
        "k_marg": k_marg,
        "k_true": k_true,
        "count_exact": count_exact,
        "count_abs_error": float(count_abs_error),
        "p_k_true_given_y": p_k_true,
        "p_k_max_given_y": p_k_max,
        "signal_mse": signal_mse,
        "brier": brier,
        "log_evidence": inference["log_evidence"],
    }


# ---------------------------------------------------------------------------
# Section 8: Bootstrap aggregation
# ---------------------------------------------------------------------------

def _bootstrap_mean_ci(
    values: list[float],
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    arr = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1 - alpha)))
    return mean, lo, hi


def aggregate_cell(
    replicate_metrics: list[dict],
    cell: dict,
    boot_rng: np.random.Generator,
    n_boot: int,
) -> dict:
    """Aggregate per-replicate metrics to cell-level summary with bootstrap CIs."""
    tau_primary = E02_CONFIG["boundary_tolerance_primary"]
    k_true = E02_CONFIG["k_true"]
    n_reps = len(replicate_metrics)

    def _ci(key: str) -> tuple[float, float, float]:
        vals = [m[key] for m in replicate_metrics if math.isfinite(m.get(key, float("nan")))]
        return _bootstrap_mean_ci(vals, n_boot, boot_rng)

    agg: dict[str, Any] = {
        "cell_index": cell["cell_index"],
        "family": cell["family"],
        "n": cell["n"],
        "param": cell["param"],
        "value": cell["value"],
        "k_true": k_true,
        "n_replicates": n_reps,
    }

    # Primary boundary metrics (tau=3)
    for key in ["f1", "precision", "recall", "mae"]:
        mean, lo, hi = _ci(f"{key}_at_{tau_primary}")
        agg[f"mean_{key}"] = mean
        agg[f"{key}_ci_lo"] = lo
        agg[f"{key}_ci_hi"] = hi

    # Sensitivity tolerances
    for tau in E02_CONFIG["boundary_tolerances"]:
        if tau != tau_primary:
            mean, lo, hi = _ci(f"f1_at_{tau}")
            agg[f"mean_f1_at_{tau}"] = mean

    # Count metrics
    for key in ["count_exact", "count_abs_error", "p_k_true_given_y", "p_k_max_given_y"]:
        mean, lo, hi = _ci(key)
        agg[f"mean_{key}"] = mean
        agg[f"{key}_ci_lo"] = lo
        agg[f"{key}_ci_hi"] = hi

    # Signal and calibration
    for key in ["signal_mse", "brier"]:
        mean, lo, hi = _ci(key)
        agg[f"mean_{key}"] = mean
        agg[f"{key}_ci_lo"] = lo
        agg[f"{key}_ci_hi"] = hi

    # Flag k_max saturation
    agg["k_max_saturation"] = bool(
        any(m["p_k_max_given_y"] >= 0.01 for m in replicate_metrics)
    )

    return agg


# ---------------------------------------------------------------------------
# Section 9: Figure generation
# ---------------------------------------------------------------------------

def _make_recovery_figure(
    cell_summaries: list[dict],
    out_path: Path,
) -> None:
    """Figure E02: F1@3 and P(k=3|y) by family and effect size with 95% CIs."""
    from _style import COLORS, COLOR_CYCLE, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.1)

    families = ["gaussian", "poisson", "binomial", "negbin", "beta_obs"]
    family_labels = {
        "gaussian": "Gaussian",
        "poisson": "Poisson",
        "binomial": "Binomial",
        "negbin": "NegBin",
        "beta_obs": "Beta-obs",
    }
    param_display = {
        "delta": "delta",
        "q": "q",
        "p_mid": "p_mid",
        "mu_mid": "mu_mid",
    }
    fam_colors = {fam: COLOR_CYCLE[i] for i, fam in enumerate(families)}
    # Circle = weak effect, triangle-up = strong effect
    effect_markers = ["o", "^"]
    effect_ms = [9, 9]

    # Large figure: two wide side-by-side panels with room for legend below.
    # Disable constrained_layout so subplots_adjust works (setup_style enables
    # it globally, which is incompatible with manual spacing adjustments).
    import matplotlib as _mpl
    _mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.28, wspace=0.32)
    x_pos = np.arange(len(families))
    legend_handles: list = []
    legend_labels: list = []

    def _param_str(c: dict) -> str:
        sym = param_display.get(c["param"], c["param"])
        return f"{sym} = {c['value']:.2f}"

    def _plot_metric(
        ax,
        metric_key: str, ci_lo_key: str, ci_hi_key: str,
        ylabel: str, title: str,
        collect_legend: bool,
    ) -> None:
        for i, fam in enumerate(families):
            fam_cells = [c for c in cell_summaries if c["family"] == fam]
            if not fam_cells:
                continue
            means = [c.get(metric_key, float("nan")) for c in fam_cells]
            los   = [c.get(ci_lo_key,  float("nan")) for c in fam_cells]
            his   = [c.get(ci_hi_key,  float("nan")) for c in fam_cells]
            x_jitter = x_pos[i] + np.linspace(-0.18, 0.18, len(fam_cells))
            for j, (m, lo, hi, c) in enumerate(zip(means, los, his, fam_cells)):
                if not math.isfinite(m):
                    continue
                yerr = np.array([[m - lo, hi - m]]).T if math.isfinite(lo) else None
                marker = effect_markers[j] if j < len(effect_markers) else "o"
                ms = effect_ms[j] if j < len(effect_ms) else 9
                leg_label = f"{family_labels[fam]}  {_param_str(c)}"
                container = ax.errorbar(
                    x_jitter[j], m, yerr=yerr,
                    fmt=marker, color=fam_colors[fam],
                    capsize=5, capthick=1.4, elinewidth=1.4,
                    markersize=ms, markeredgewidth=0.8,
                    label=leg_label,
                )
                if collect_legend:
                    legend_handles.append(container)
                    legend_labels.append(leg_label)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [family_labels[f] for f in families],
            rotation=25, ha="right", fontsize=13,
        )
        ax.set_ylabel(ylabel, fontsize=14)
        ax.annotate(title, xy=(0.02, 0.97), xycoords='axes fraction', fontsize=10, va='top')
        ax.set_ylim(-0.03, 1.08)
        ax.axhline(1.0, color=COLORS["grey"], lw=1.0, ls="--", zorder=0)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlim(-0.5, len(families) - 0.5)

    _plot_metric(
        axes[0], "mean_f1", "f1_ci_lo", "f1_ci_hi",
        "F1 @ τ=3",
        "Recovery F1 by family  (k_true = 3)",
        collect_legend=True,
    )
    _plot_metric(
        axes[1], "mean_p_k_true_given_y",
        "p_k_true_given_y_ci_lo", "p_k_true_given_y_ci_hi",
        "P(k = 3 | y)",
        "Segment-count selection by family",
        collect_legend=False,
    )

    # Figure-level legend: 5 columns × 2 rows, one entry per cell.
    # Title explains marker encoding.
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        fontsize=10.5,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.4,
        title="Circle (o) = weak effect     Triangle (^) = strong effect",
        title_fontsize=11,
    )

    save_figure(fig, out_path, formats=("pdf", "png"))
    logger.info("Saved recovery figure to %s", out_path)


# ---------------------------------------------------------------------------
# Section 10: LaTeX table generation
# ---------------------------------------------------------------------------

def _fmt(v: float | None, fmt: str = ".3f") -> str:
    if v is None or not math.isfinite(v):
        return "---"
    return f"{v:{fmt}}"


def _fmt_ci(mean: float, lo: float, hi: float, fmt: str = ".3f") -> str:
    if not math.isfinite(mean):
        return "---"
    return f"{mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def generate_latex_table(
    cell_summaries: list[dict],
    out_path: Path,
) -> None:
    """Generate tab_E02_recovery.tex from cell summaries."""
    family_labels = {
        "gaussian": r"Gaussian",
        "poisson": r"Poisson",
        "binomial": r"Binomial",
        "negbin": r"NegBin",
        "beta_obs": r"Beta-obs",
    }
    param_labels = {
        "delta": r"$\Delta$",
        "q": r"$q$",
        "p_mid": r"$p_{\rm mid}$",
        "mu_mid": r"$\mu_{\rm mid}$",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{E02: Observation-family portability. "
        r"$k_{\rm true}=3$, exact equal segment lengths, geometric-decay $p(k)$ prior, "
        r"$k_{\max}=10$. "
        r"Mean [95\% bootstrap CI] over 100 (conjugate) or 50 (Beta-obs) independent datasets per cell. "
        r"F1 at boundary tolerance $\tau=3$.}",
        r"\label{tab:E02-recovery-paper}",
        r"\footnotesize",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Family & Param & $n$ & F1@3 & P($k$=3) & Count MAE & Signal MSE & Brier \\",
        r"\midrule",
    ]
    current_family = None
    for c in cell_summaries:
        fam = c["family"]
        if fam != current_family:
            if current_family is not None:
                lines.append(r"\midrule")
            current_family = fam

        fam_label = family_labels.get(fam, fam)
        param_label = param_labels.get(c["param"], c["param"])
        param_val = f"{c['value']:.2f}"
        n_val = c["n"]

        row = (
            f"{fam_label} & {param_label}={param_val} & {n_val}"
            f" & {_fmt_ci(c.get('mean_f1', float('nan')), c.get('f1_ci_lo', float('nan')), c.get('f1_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_p_k_true_given_y', float('nan')), c.get('p_k_true_given_y_ci_lo', float('nan')), c.get('p_k_true_given_y_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_count_abs_error', float('nan')), c.get('count_abs_error_ci_lo', float('nan')), c.get('count_abs_error_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_signal_mse', float('nan')), c.get('signal_mse_ci_lo', float('nan')), c.get('signal_mse_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_brier', float('nan')), c.get('brier_ci_lo', float('nan')), c.get('brier_ci_hi', float('nan')))}"
            r" \\"
        )
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved LaTeX table to %s", out_path)


# ---------------------------------------------------------------------------
# Section 11: SHA256 helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "missing"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Section 12: Main execution loop
# ---------------------------------------------------------------------------

def run_e02(pilot: bool = False) -> None:
    """Execute the full E02 family portability experiment."""
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    log_path = RESULTS_BASE / "run_log.txt"
    _setup_logging(log_path)

    n_boot = 200 if pilot else E02_CONFIG["bootstrap_resamples"]
    mode_label = "PILOT" if pilot else "CONFIRMATORY"
    logger.info("=" * 70)
    logger.info("E02 family portability — %s run", mode_label)
    logger.info("Config SHA256: %s", E02_CONFIG_SHA256[:16])
    logger.info("n_boot=%d", n_boot)
    logger.info("=" * 70)

    utc_start = datetime.now(timezone.utc).isoformat()
    start_wall = time.perf_counter()

    cells = E02_CONFIG["cells"]
    logger.info("Total cells: %d", len(cells))

    all_replicate_records: list[dict] = []
    cell_summaries: list[dict] = []
    failed_runs: list[dict] = []
    oracle_checks: dict[str, Any] = {}

    taus = E02_CONFIG["boundary_tolerances"]

    for cell_idx, cell in enumerate(cells):
        family = cell["family"]
        n = cell["n"]
        cell_index = cell["cell_index"]
        n_reps_full = cell["n_reps"]
        n_replicates = 5 if pilot else n_reps_full

        label = f"{family},n={n},{cell['param']}={cell['value']}"
        logger.info("[Cell %d/%d] %s", cell_idx + 1, len(cells), label)

        rep_metrics: list[dict] = []
        oracle_devs: list[dict] = []

        for rep in range(n_replicates):
            # Seed: [20261021, 2, cell_index, replicate] per protocol
            seed_seq = np.random.SeedSequence([20261021, 2, cell_index, rep])
            data_seed = seed_seq.spawn(2)[0]
            rng = np.random.default_rng(data_seed)

            try:
                dataset = generate_dataset(cell, rng)
                infer = run_inference(cell, dataset)
                metrics = compute_replicate_metrics(cell, dataset, infer, taus)

                # Quadrature convergence check for Beta-obs (on first n_check reps)
                if family == "beta_obs" and rep < 5:
                    try:
                        dev = _beta_obs_convergence_check(cell, dataset)
                        oracle_devs.append(dev)
                        logger.info(
                            "  Beta-obs convergence rep=%d: log_ev_dev=%.2e, curve_dev=%.2e",
                            rep,
                            dev["max_log_evidence_deviation"],
                            dev["max_bayes_curve_deviation"],
                        )
                    except Exception as exc:
                        logger.warning("  Beta-obs convergence check failed rep=%d: %s", rep, exc)

                record = {
                    "cell_index": cell_index,
                    "family": family,
                    "n": n,
                    "param": cell["param"],
                    "value": cell["value"],
                    "replicate": rep,
                    **{k: float(v) if isinstance(v, np.floating) else v
                       for k, v in metrics.items()},
                }
                rep_metrics.append(metrics)
                all_replicate_records.append(record)

                if (rep + 1) % 10 == 0 or n_replicates <= 10:
                    logger.info("  Rep %d/%d done", rep + 1, n_replicates)

            except Exception as exc:
                logger.error("  FAILED rep=%d: %s", rep, exc, exc_info=True)
                failed_runs.append({
                    "cell_index": cell_index,
                    "family": family,
                    "n": n,
                    "param": cell["param"],
                    "value": cell["value"],
                    "replicate": rep,
                    "error": str(exc),
                })

        if not rep_metrics:
            logger.warning("Cell %s: all replicates failed!", label)
            continue

        # Bootstrap seed: deterministic per cell
        boot_rng = np.random.default_rng([20261021, 3, cell_index])
        summary = aggregate_cell(rep_metrics, cell, boot_rng, n_boot)
        cell_summaries.append(summary)

        # Store oracle convergence checks
        if oracle_devs:
            oracle_checks[label] = {
                "max_log_evidence_deviation": max(d["max_log_evidence_deviation"] for d in oracle_devs),
                "max_bayes_curve_deviation": max(d["max_bayes_curve_deviation"] for d in oracle_devs),
                "n_reps_checked": len(oracle_devs),
            }

        elapsed = time.perf_counter() - start_wall
        logger.info(
            "  Cell done | elapsed %.1fs | n_rep=%d | mean_f1=%.3f",
            elapsed,
            len(rep_metrics),
            summary.get("mean_f1", float("nan")),
        )

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    logger.info("Generating figures...")
    _make_recovery_figure(
        cell_summaries,
        FIGURES_DIR / "fig_E02_recovery",
    )

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    logger.info("Generating LaTeX table...")
    generate_latex_table(cell_summaries, TABLES_DIR / "tab_E02_recovery.tex")

    # -----------------------------------------------------------------------
    # Metrics CSV
    # -----------------------------------------------------------------------
    metrics_csv_path = RESULTS_BASE / "metrics.csv"
    if cell_summaries:
        all_keys = sorted({k for c in cell_summaries for k in c.keys()})
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in cell_summaries:
                writer.writerow({k: row.get(k, "") for k in all_keys})

    # -----------------------------------------------------------------------
    # Oracle checks JSON
    # -----------------------------------------------------------------------
    oracle_path = RESULTS_BASE / "oracle_checks.json"
    with open(oracle_path, "w") as f:
        json.dump(oracle_checks, f, indent=2)

    # -----------------------------------------------------------------------
    # Metrics JSON
    # -----------------------------------------------------------------------
    utc_end = datetime.now(timezone.utc).isoformat()
    n_cells_completed = len(cell_summaries)
    n_cells_planned = len(cells)
    n_reps_completed = len(all_replicate_records)
    n_reps_failed = len(failed_runs)

    # gate_pass is initially set to False; _run_gate_checks will compute the
    # true value after all oracle checks are evaluated.
    replicate_gate_pass = (n_reps_failed == 0 and n_cells_completed == n_cells_planned)
    metrics_json = {
        "experiment_id": "E02",
        "n_cells_planned": n_cells_planned,
        "n_cells_completed": n_cells_completed,
        "n_cells_failed": n_cells_planned - n_cells_completed,
        "n_replicates_completed": n_reps_completed,
        "n_replicates_failed": n_reps_failed,
        "pilot_mode": pilot,
        "gate_pass": False,  # will be updated after gate checks; not yet evaluated
        "cell_summaries": cell_summaries,
        "oracle_checks": oracle_checks,
    }
    metrics_json_path = RESULTS_BASE / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(
            metrics_json, f, indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else x,
        )

    # -----------------------------------------------------------------------
    # Failed runs
    # -----------------------------------------------------------------------
    failed_csv_path = RESULTS_BASE / "failed_runs.csv"
    if failed_runs:
        with open(failed_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failed_runs[0].keys()))
            writer.writeheader()
            writer.writerows(failed_runs)
        logger.warning(
            "%d failed replicate(s) written to %s", len(failed_runs), failed_csv_path
        )
    else:
        logger.info("No failed replicates.")

    # -----------------------------------------------------------------------
    # Source tree fingerprint
    # -----------------------------------------------------------------------
    repo_root = Path(__file__).parent.parent
    src_files = sorted(
        (repo_root / "src" / "bayesbreak").rglob("*.py"),
        key=lambda p: str(p),
    )
    src_hashes = {str(p.relative_to(repo_root)): _sha256_file(p) for p in src_files}

    # -----------------------------------------------------------------------
    # Git provenance
    # -----------------------------------------------------------------------
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_rev = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception:
        dirty = None

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------
    artifact_sha256 = _sha256_file(metrics_json_path)

    manifest = {
        "experiment_id": "E02",
        "protocol_revision": E02_CONFIG["protocol_revision"],
        "config_sha256": E02_CONFIG_SHA256,
        "git_revision": git_rev,
        "working_tree_dirty": dirty,
        "src_tree_sha256": _sha256_bytes(
            b"".join(p.read_bytes() for p in sorted(src_files, key=str))
        ),
        "source_file_sha256": src_hashes,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "matplotlib_version": __import__("matplotlib").__version__,
        "platform": platform.platform(),
        "seed": E02_CONFIG["seed_base"] + [0, 0],
        "input_sha256": "deterministic_generator",
        "source_kind": "multi_family_simulation",
        "families": ["gaussian", "poisson", "binomial", "negbin", "beta_obs"],
        "prior": "geometric_decay_log4",
        "k_max": E02_CONFIG["k_max"],
        "k_true": E02_CONFIG["k_true"],
        "coordinate_axis": "index_uniform",
        "reference_type": "simulation_exact_inference",
        "fit_status": "completed" if n_reps_failed == 0 else f"partial_{n_reps_failed}_failed",
        "metric_schema": "E02_family_portability",
        "artifact_sha256": artifact_sha256,
        "utc_start": utc_start,
        "utc_end": utc_end,
        "pilot_mode": pilot,
        "n_cells_planned": n_cells_planned,
        "n_cells_completed": n_cells_completed,
    }
    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall_time = time.perf_counter() - start_wall
    logger.info("=" * 70)
    logger.info("E02 %s complete in %.1fs", mode_label, wall_time)
    logger.info(
        "Cells: %d planned, %d completed, %d failed",
        n_cells_planned, n_cells_completed, n_cells_planned - n_cells_completed,
    )
    logger.info("Replicates: %d completed, %d failed", n_reps_completed, n_reps_failed)
    logger.info("Config SHA256: %s", E02_CONFIG_SHA256)
    logger.info("Results: %s", RESULTS_BASE)
    logger.info("=" * 70)

    # Gate checks — must run BEFORE final gate_pass is written to metrics.json.
    if not pilot:
        oracle_gate_pass = _run_gate_checks(cell_summaries, oracle_checks)
    else:
        oracle_gate_pass = True

    final_gate_pass = replicate_gate_pass and oracle_gate_pass
    metrics_json["gate_pass"] = final_gate_pass
    logger.info("gate_pass: %s", final_gate_pass)

    # Rewrite metrics.json with the corrected gate_pass value.
    with open(metrics_json_path, "w") as _f:
        json.dump(
            metrics_json, _f, indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else x,
        )

    if not pilot and not final_gate_pass:
        sys.exit(1)


def _run_gate_checks(cell_summaries: list[dict], oracle_checks: dict) -> bool:
    """Log gate-check results after confirmatory run.

    Gate thresholds (prespecified, must not be relaxed without protocol amendment):
      - Bayes curve deviation (confirmatory quadrature order vs 2x nodes): < 1e-3
        on the (0,1) signal scale.
      - All critical metrics (mean_f1, signal_mse) must be finite.

    Returns
    -------
    bool
        True if all gate checks pass, False otherwise.
    """
    logger.info("--- Gate checks ---")
    issues: list[str] = []

    for c in cell_summaries:
        f1 = c.get("mean_f1", float("nan"))
        if not math.isfinite(f1):
            issues.append(f"Cell {c['cell_index']} ({c['family']}): mean_f1 is NaN")
        mse = c.get("mean_signal_mse", float("nan"))
        if not math.isfinite(mse):
            issues.append(f"Cell {c['cell_index']} ({c['family']}): signal_mse is NaN")
        if c.get("k_max_saturation"):
            logger.warning(
                "k_max saturation detected: cell %d (%s)", c["cell_index"], c["family"]
            )

    if oracle_checks:
        # Prespecified convergence threshold: 1e-3 on signal scale.
        # Compares confirmatory quadrature order vs 2x nodes.
        _curve_threshold = 1e-3
        for label, dev in oracle_checks.items():
            max_dev_ev = dev.get("max_log_evidence_deviation", float("nan"))
            max_dev_curve = dev.get("max_bayes_curve_deviation", float("nan"))
            logger.info(
                "Beta-obs convergence check %s: log_ev_dev=%.2e, curve_dev=%.2e",
                label, max_dev_ev, max_dev_curve,
            )
            if math.isfinite(max_dev_curve) and max_dev_curve > _curve_threshold:
                issues.append(
                    f"Beta-obs quadrature deviation too large ({label}): "
                    f"curve_dev={max_dev_curve:.2e} > threshold={_curve_threshold:.2e}"
                )

    if issues:
        logger.error("GATE CHECK FAILURES:")
        for issue in issues:
            logger.error("  %s", issue)
        logger.error("E02 result: FAIL (gate check)")
        return False
    else:
        logger.info("All gate checks PASSED.")
        logger.info("E02 result: PASS")
        return True


def regenerate_figures(
    manuscript_dest: Path | None = None,
    tables_dest: Path | None = None,
) -> None:
    """Regenerate E02 figures and tables from existing metrics.json.

    Does not re-run the experiment.  Reads the frozen metrics.json from the
    results directory keyed by config SHA256, regenerates the recovery figure
    and the LaTeX recovery table, and optionally copies outputs to the
    manuscript shared directories.

    Args:
        manuscript_dest: If provided, copy figure .pdf/.png files here.
        tables_dest: If provided, copy .tex table files here.
    """
    import shutil
    _setup_logging()
    metrics_path = RESULTS_BASE / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json not found at {metrics_path}. Run the experiment first."
        )
    with open(metrics_path) as f:
        data = json.load(f)
    cell_summaries = data["cell_summaries"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    fig_stem = FIGURES_DIR / "fig_E02_recovery"
    _make_recovery_figure(cell_summaries, fig_stem)
    generate_latex_table(cell_summaries, TABLES_DIR / "tab_E02_recovery.tex")

    if manuscript_dest is not None:
        manuscript_dest = Path(manuscript_dest)
        manuscript_dest.mkdir(parents=True, exist_ok=True)
        for suffix in [".pdf", ".png"]:
            src = fig_stem.with_suffix(suffix)
            if src.exists():
                shutil.copy2(src, manuscript_dest / src.name)
                logger.info("Copied %s → %s", src.name, manuscript_dest)

    if tables_dest is not None:
        tables_dest = Path(tables_dest)
        tables_dest.mkdir(parents=True, exist_ok=True)
        table_src = TABLES_DIR / "tab_E02_recovery.tex"
        if table_src.exists():
            shutil.copy2(table_src, tables_dest / table_src.name)
            logger.info("Copied %s → %s", table_src.name, tables_dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E02 family portability experiment.")
    parser.add_argument(
        "--pilot", action="store_true",
        help="Run in pilot mode (5 datasets per cell, 200 bootstrap resamples)",
    )
    parser.add_argument(
        "--regenerate-figures", action="store_true",
        help="Regenerate figures and tables from existing metrics.json without re-running",
    )
    parser.add_argument(
        "--manuscript-dest", type=Path, default=None,
        help="Copy regenerated figures (.pdf/.png) to this directory",
    )
    parser.add_argument(
        "--tables-dest", type=Path, default=None,
        help="Copy regenerated tables (.tex) to this directory",
    )
    args = parser.parse_args()
    if args.regenerate_figures:
        regenerate_figures(
            manuscript_dest=args.manuscript_dest,
            tables_dest=args.tables_dest,
        )
    else:
        run_e02(pilot=args.pilot)
