"""E01: Gaussian segmentation calibration and uncertainty evaluation.

Protocol : .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E01
Config   : .claude/skills/bayesbreak-release-audit/protocols/experiments.json

Produces:
  results/confirmatory/E01_gaussian_calibration/<config_sha256>/
    manifest.json, metrics.json, metrics.csv, oracle_checks.json,
    figures/fig_E01_calibration.pdf, figures/fig_E01_recovery.pdf,
    tables/tab_P1_recovery.tex, run_log.txt, failed_runs.csv

Usage (confirmatory):
    python scripts/run_e01_gaussian_calibration.py

Usage (diagnostic pilot, 5 datasets per cell):
    python scripts/run_e01_gaussian_calibration.py --pilot
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

# ---------------------------------------------------------------------------
# E01 configuration (frozen before first confirmatory run)
# ---------------------------------------------------------------------------

E01_CONFIG: dict[str, Any] = {
    "id": "E01",
    "name": "gaussian_calibration",
    "type": "simulation",
    "n_values": [256, 512],
    "k_true": [1, 3, 6],
    "jump_sigma": [0.5, 1.0, 2.0],
    "minimum_generating_segment_length": 32,
    "sigma": 1.0,
    "gaussian_prior": {"mean": 0, "variance": 4},
    "k_max": 12,
    "replicates": 100,
    "seed_base": [20261021, 1],
    "prior": "geometric_decay_log4",
    "ablation_prior": "uniform",
    "ablation_k_true": [3, 6],
    "ablation_jump_sigma": [0.5, 2.0],
    "boundary_tolerances": [0, 1, 3, 5],
    "boundary_tolerance_primary": 3,
    "bootstrap_resamples": 2000,
    "calibration_bins": 10,
    "protocol_revision": "2026-10-21-2",
}

E01_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E01_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = Path(__file__).parent.parent / "results" / "confirmatory" / "E01_gaussian_calibration" / E01_CONFIG_SHA256[:16]
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
# Section 1: Cell enumeration
# ---------------------------------------------------------------------------

def enumerate_cells(include_ablation: bool = True) -> list[dict]:
    """Return the full E01 cell list with flat cell_index for seeding.

    Primary cells: 14 (7 per n-value, 2 n-values)
    Ablation cells: 8 (4 per n-value: k in {3,6} x delta in {0.5,2.0})
    """
    cells = []
    cell_index = 0
    for n in E01_CONFIG["n_values"]:
        n_cells_start = cell_index
        # k=1 null cell (delta=0)
        cells.append({
            "n": n, "k_true": 1, "delta": 0.0, "cell_index": cell_index,
            "ablation": False, "prior": "geometric_decay_log4",
        })
        cell_index += 1
        # k=3 cells
        for delta in E01_CONFIG["jump_sigma"]:
            cells.append({
                "n": n, "k_true": 3, "delta": delta, "cell_index": cell_index,
                "ablation": False, "prior": "geometric_decay_log4",
            })
            cell_index += 1
        # k=6 cells
        for delta in E01_CONFIG["jump_sigma"]:
            cells.append({
                "n": n, "k_true": 6, "delta": delta, "cell_index": cell_index,
                "ablation": False, "prior": "geometric_decay_log4",
            })
            cell_index += 1

        if include_ablation:
            for k_true in E01_CONFIG["ablation_k_true"]:
                for delta in E01_CONFIG["ablation_jump_sigma"]:
                    cells.append({
                        "n": n, "k_true": k_true, "delta": delta, "cell_index": cell_index,
                        "ablation": True, "prior": "uniform",
                    })
                    cell_index += 1
    return cells


# ---------------------------------------------------------------------------
# Section 2: Data generator
# ---------------------------------------------------------------------------

def _sample_boundaries_uniform(
    n: int, k: int, min_L: int, rng: np.random.Generator
) -> list[int]:
    """Sample k-1 boundary positions uniformly from admissible sets.

    Uses the stars-and-bars bijection to sample uniformly over all
    k-segment partitions of {1..n} with minimum segment length min_L.

    Maps uniformly chosen k-1 divider positions to segment excess lengths
    (each segment = min_L + excess), then converts to boundary positions.
    """
    n_excess = n - k * min_L
    if n_excess < 0:
        raise ValueError(
            f"n={n} too small for k={k} segments of min length {min_L}. "
            f"Need n >= {k * min_L}."
        )
    # Choose k-1 divider positions from {0, ..., n_excess + k - 2} without replacement
    # (bijection: dividers split n_excess + k - 1 items into k groups)
    total_slots = n_excess + k - 1
    if total_slots < k - 1:
        raise ValueError("Not enough slots for boundary sampling.")
    dividers = sorted(rng.choice(total_slots, size=k - 1, replace=False).tolist())

    # Compute excess for each segment from consecutive divider differences
    seg_excesses = []
    prev = -1
    for d in dividers:
        seg_excesses.append(d - prev - 1)
        prev = d
    seg_excesses.append(n_excess + k - 2 - prev)

    # Convert excess lengths to segment lengths and then to boundary positions
    seg_lengths = [min_L + e for e in seg_excesses]
    assert sum(seg_lengths) == n, f"Sum {sum(seg_lengths)} != n {n}"
    assert all(s >= min_L for s in seg_lengths), f"Segment too short: {seg_lengths}"

    boundaries = []
    pos = 0
    for length in seg_lengths[:-1]:
        pos += length
        boundaries.append(pos)
    return boundaries


def generate_dataset(
    n: int,
    k_true: int,
    delta: float,
    sigma: float,
    min_L: int,
    seed_sequence: np.random.SeedSequence,
) -> dict:
    """Generate one E01 Gaussian piecewise-constant sequence.

    Returns a dict with:
      y (n,), x (n,), true_boundaries (k+1,), true_means (k,),
      true_signal (n,), k_true, delta, sigma, n, seed_entropy
    """
    # Child generators with named roles so method-specific RNGs cannot
    # change data generation (per protocol seed scheme).
    data_seed, _ = seed_sequence.spawn(2)
    rng = np.random.default_rng(data_seed)

    x = np.arange(1, n + 1, dtype=float)

    if k_true == 1:
        true_boundaries_int = [0, n]
        initial_mean = rng.choice([0.0, 0.5])
        true_means = [float(initial_mean)]
    else:
        interior = _sample_boundaries_uniform(n, k_true, min_L, rng)
        true_boundaries_int = [0] + interior + [n]
        # Initial mean: 0 or 0.5 with equal probability
        initial_mean = rng.choice([0.0, 0.5])
        # First direction: +1 or -1 with equal probability
        first_dir = rng.choice([-1, 1])
        true_means = [float(initial_mean)]
        direction = int(first_dir)
        for _ in range(1, k_true):
            true_means.append(true_means[-1] + direction * delta)
            direction = -direction

    # Generate true signal and noisy observations
    true_signal = np.empty(n, dtype=float)
    for q, (a, b) in enumerate(
        zip(true_boundaries_int[:-1], true_boundaries_int[1:], strict=False)
    ):
        true_signal[a:b] = true_means[q]

    y = true_signal + rng.standard_normal(n) * sigma

    # Verify minimum segment lengths
    seg_lengths = [true_boundaries_int[q + 1] - true_boundaries_int[q] for q in range(k_true)]
    assert all(s >= min_L for s in seg_lengths), (
        f"Generated segment length violation: {seg_lengths}, min_L={min_L}"
    )

    return {
        "y": y,
        "x": x,
        "true_boundaries": true_boundaries_int,
        "true_means": true_means,
        "true_signal": true_signal,
        "k_true": k_true,
        "delta": delta,
        "sigma": sigma,
        "n": n,
        "min_segment_length": min_L,
        "seg_lengths": seg_lengths,
        "seed_entropy": list(seed_sequence.entropy) if hasattr(seed_sequence.entropy, "__iter__") else [seed_sequence.entropy],
    }


# ---------------------------------------------------------------------------
# Section 3: Vectorized Bayes regression curve (optimization)
# ---------------------------------------------------------------------------

def _logsumexp_axis0(a: NDArray) -> NDArray:
    """Stable log-sum-exp over axis 0. Returns -inf where all inputs are -inf."""
    a_max = np.max(a, axis=0, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=0)
    a_max_sq = np.squeeze(a_max, axis=0)
    # When all inputs are -inf, a_max is -inf and s=0; return -inf exactly.
    with np.errstate(divide="ignore"):
        log_s = np.log(s)  # s=0 → -inf (correct; warning suppressed)
    return np.where(np.isfinite(a_max_sq), a_max_sq + log_s, -np.inf)


def _bayes_curve_fixed_k_vectorized(
    L: NDArray,
    R: NDArray,
    lA0: NDArray,
    A1: NDArray,
    n: int,
    k: int,
) -> NDArray:
    """Vectorized Bayes regression curve for fixed k (no Python inner loops).

    Equivalent to dp.bayes_regression_curve_fixed_k with g=1 (index-uniform).
    Uses numpy broadcasting for the (n+1, n+1) weight matrix.
    """
    denom = float(L[k, n])
    if not np.isfinite(denom):
        return np.full(n, np.nan, dtype=float)

    # LL[p, i] = L[p, i] for p=0..k-1
    LL = L[0:k, :]   # shape (k, n+1)
    # RR[p, j] = R[k-1-p, j] for p=0..k-1 → R[k-1,j], ..., R[0,j]
    RR = R[k - 1::-1, :]   # shape (k, n+1)

    # cross[i,j] = logsumexp_p( L[p,i] + R[k-1-p,j] )
    # Broadcasting: (k, n+1, 1) + (k, 1, n+1) → (k, n+1, n+1), then lse over axis 0
    cross = _logsumexp_axis0(LL[:, :, np.newaxis] + RR[:, np.newaxis, :])  # (n+1, n+1)

    log_w = cross + lA0 - denom   # (n+1, n+1)

    # Suppress inadmissible blocks (lA0=-inf) and zero moments
    finite_mask = np.isfinite(log_w) & np.isfinite(A1) & (A1 != 0.0)
    # Clamp log_w to avoid underflow in exp (< -745 is effectively 0)
    log_w_clipped = np.where(finite_mask, np.clip(log_w, -745.0, 0.0), -np.inf)
    contrib = np.where(finite_mask, np.exp(log_w_clipped) * A1, 0.0)

    # Diff array trick: curve[t] = Σ_{i≤t<j} contrib[i,j]
    # diff[i] += Σ_j contrib[i,j], diff[j] -= Σ_i contrib[i,j]
    diff = contrib.sum(axis=1) - contrib.sum(axis=0)   # shape (n+1,)
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
    """Mixed-k Bayes regression curve: Σ_k P(k|y) E[μ_t|y,k]."""
    out = np.zeros(n, dtype=float)
    for k in range(1, k_max + 1):
        wk = float(post_k[k - 1])
        if wk == 0.0 or not np.isfinite(float(L[k, n])):
            continue
        out += wk * _bayes_curve_fixed_k_vectorized(L, R, lA0, A1, n, k)
    return out


# ---------------------------------------------------------------------------
# Section 4: Inference
# ---------------------------------------------------------------------------

def run_inference(
    y: NDArray,
    x: NDArray,
    n: int,
    prior_kind: str,
    k_max: int,
) -> dict:
    """Run exact Gaussian posterior inference for one E01 dataset.

    Returns:
      p_k_given_y (k_max,), log_p_k_given_y (k_max,),
      k_marg (int), p_k_marg (float),
      p_k_max (float), log_evidence (float),
      boundary_probs_marginal (n-1,),
      map_boundaries (list), bayes_curve (n,)
    """
    from bayesbreak.families.gaussian import BayesBreakGaussian
    import bayesbreak.dp as _dp

    # E01 Gaussian: sigma2=1, nu=0, rho2=4 (prior mean 0, prior variance 4)
    sigma2 = float(E01_CONFIG["sigma"]) ** 2  # = 1.0
    nu = float(E01_CONFIG["gaussian_prior"]["mean"])     # = 0.0
    rho2 = float(E01_CONFIG["gaussian_prior"]["variance"])  # = 4.0

    if prior_kind == "geometric_decay_log4":
        prior_fn = lambda k: math.exp(-math.log(4.0) * (k - 1))
    elif prior_kind == "uniform":
        prior_fn = None  # BayesBreakGaussian default = uniform 1/k_max
    else:
        raise ValueError(f"Unknown prior_kind={prior_kind!r}")

    est = BayesBreakGaussian(
        estimate_hyper=False,
        nu=nu, rho2=rho2, sigma2=sigma2,
        k_max=k_max,
        regression_curve="none",
        prior_k=prior_fn,
    )

    X_col = x.reshape(-1, 1)
    est.fit(X_col, y)

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

    p_k = est.k_posterior_          # shape (k_max,), 0-indexed by k-1
    log_p_k = est.log_posterior_k_   # shape (k_max,)
    k_marg = int(est.k_map_)         # marginal MAP count
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
# Section 5: Boundary matching (bipartite, one-to-one, max cardinality)
# ---------------------------------------------------------------------------

def _boundary_match(
    bhat: list[int],
    bstar: list[int],
    tau: int,
) -> tuple[int, float]:
    """One-to-one maximum-cardinality boundary match with minimum-displacement tie-breaking.

    Implements the metrics_and_statistics.md definition.
    Returns (TP, total_displacement_of_matched_pairs).
    """
    if not bstar or not bhat:
        return 0, 0.0

    # Build adjacency: (i_hat, j_star) pairs within tolerance
    edges = [
        (i, j, abs(bhat[i] - bstar[j]))
        for i in range(len(bhat))
        for j in range(len(bstar))
        if abs(bhat[i] - bstar[j]) <= tau
    ]
    if not edges:
        return 0, 0.0

    # Greedy augmenting-path maximum cardinality matching with min-displacement
    # Sort edges by displacement (stable: smaller displacement preferred at equal cardinality)
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
    k_true: int,
) -> dict:
    """Compute F1, precision, recall for all tolerances, plus MAE.

    For null sequences (k_true=1, B*=empty), reports FPR and expected # false
    boundaries instead of F1 (per protocol).
    """
    # Interior boundaries (exclude 0 and n)
    bhat = sorted(b for b in map_boundaries if b not in (0, map_boundaries[-1]))
    bstar = sorted(b for b in true_boundaries if b not in (0, true_boundaries[-1]))

    out: dict[str, float] = {}
    for tau in taus:
        tp, total_disp = _boundary_match(bhat, bstar, tau)
        fp = len(bhat) - tp
        fn = len(bstar) - tp
        if k_true == 1:
            # Null cell: no true boundaries; report false boundary count
            out[f"n_false_at_{tau}"] = float(fp)
            out[f"precision_at_{tau}"] = float("nan")
            out[f"recall_at_{tau}"] = float("nan")
            out[f"f1_at_{tau}"] = float("nan")
        else:
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
# Section 6: Per-replicate metrics
# ---------------------------------------------------------------------------

def compute_replicate_metrics(
    dataset: dict,
    inference: dict,
    taus: list[int],
) -> dict:
    """Compute all E01 metrics for one (dataset, inference) pair."""
    n = dataset["n"]
    k_true = dataset["k_true"]
    true_boundaries = dataset["true_boundaries"]
    true_signal = dataset["true_signal"]
    k_max = E01_CONFIG["k_max"]

    # 1. F1 and boundary detection
    boundary_metrics = compute_f1_metrics(
        inference["map_boundaries"],
        true_boundaries,
        taus,
        k_true,
    )

    # 2. Count metrics
    k_marg = inference["k_marg"]
    count_exact = int(k_marg == k_true)
    count_abs_error = abs(k_marg - k_true)

    # p(k_true|y): index k_true-1 in the posterior array
    p_k = inference["p_k_given_y"]
    if 1 <= k_true <= k_max:
        p_k_true = float(p_k[k_true - 1])
    else:
        p_k_true = 0.0

    p_k_max = inference["p_k_max"]

    # 3. Signal MSE (Bayes regression curve vs true signal)
    bayes_curve = inference["bayes_curve"]
    signal_mse = float(np.mean((bayes_curve - true_signal) ** 2))

    # 4. Calibration: boundary event probabilities P(b_i=1|y) for i=1..n-1
    bm = inference["boundary_probs_marginal"]   # length n-1, positions 1..n-1
    # True boundary indicators z[i] for i=1..n-1
    true_interior = set(b for b in true_boundaries if b not in (0, n))
    z = np.array([float(i in true_interior) for i in range(1, n)], dtype=float)

    # Brier score over interior positions
    brier = float(np.mean((bm - z) ** 2))

    # Log score (binary cross-entropy), clip probabilities for numerical stability
    eps = 1e-12
    p_clip = np.clip(bm, eps, 1 - eps)
    log_score = -float(np.mean(z * np.log(p_clip) + (1 - z) * np.log(1 - p_clip)))

    # Count how many predictions were clipped
    n_clipped = int(np.sum((bm < eps) | (bm > 1 - eps)))

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
        "log_score": log_score,
        "n_log_score_clipped": n_clipped,
        "log_evidence": inference["log_evidence"],
    }


# ---------------------------------------------------------------------------
# Section 7: Bootstrap aggregation
# ---------------------------------------------------------------------------

def _bootstrap_mean_ci(
    values: list[float],
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via 2000 bootstrap resamples."""
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
    tau_primary = E01_CONFIG["boundary_tolerance_primary"]  # 3
    k_true = cell["k_true"]
    n_reps = len(replicate_metrics)

    def _ci(key: str) -> tuple[float, float, float]:
        vals = [m[key] for m in replicate_metrics if math.isfinite(m.get(key, float("nan")))]
        return _bootstrap_mean_ci(vals, n_boot, boot_rng)

    agg: dict[str, Any] = {
        "n": cell["n"],
        "k_true": k_true,
        "delta": cell["delta"],
        "prior": cell["prior"],
        "ablation": cell["ablation"],
        "n_replicates": n_reps,
    }

    # Boundary metrics
    if k_true == 1:
        mean, lo, hi = _ci(f"n_false_at_{tau_primary}")
        agg[f"mean_n_false_at_{tau_primary}"] = mean
        agg[f"n_false_ci_lo"] = lo
        agg[f"n_false_ci_hi"] = hi
        # Pad F1 fields as nan for null cells
        agg["mean_f1"] = float("nan")
        agg["f1_ci_lo"] = float("nan")
        agg["f1_ci_hi"] = float("nan")
    else:
        for key in ["f1", "precision", "recall", "mae"]:
            mean, lo, hi = _ci(f"{key}_at_{tau_primary}")
            agg[f"mean_{key}"] = mean
            agg[f"{key}_ci_lo"] = lo
            agg[f"{key}_ci_hi"] = hi
        # Sensitivity tolerances
        for tau in E01_CONFIG["boundary_tolerances"]:
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
    for key in ["signal_mse", "brier", "log_score"]:
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
# Section 8: Calibration curve
# ---------------------------------------------------------------------------

def compute_calibration_curve(
    all_bm: list[NDArray],
    all_z: list[NDArray],
    n_bins: int = 10,
    n_boot: int = 2000,
    boot_rng: np.random.Generator | None = None,
) -> dict:
    """Compute reliability diagram data with 10 equal-mass bins.

    Args:
        all_bm: predicted boundary probabilities per replicate (n-1,)
        all_z: true boundary indicators per replicate (n-1,)
        n_bins: number of equal-mass bins (primary: 10)
    Returns dict with bin_centers, bin_mean_pred, bin_mean_obs, bin_counts, ece
    """
    if boot_rng is None:
        boot_rng = np.random.default_rng(20261021)

    # Pool all (prediction, truth) pairs
    all_p = np.concatenate(all_bm)
    all_y = np.concatenate(all_z)
    n_total = len(all_p)

    # Equal-mass bins by predicted probability
    sorted_idx = np.argsort(all_p)
    bin_size = n_total // n_bins
    bin_mean_pred = []
    bin_mean_obs = []
    bin_counts = []
    bin_edges = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size if b < n_bins - 1 else n_total
        idx = sorted_idx[start:end]
        bin_mean_pred.append(float(np.mean(all_p[idx])))
        bin_mean_obs.append(float(np.mean(all_y[idx])))
        bin_counts.append(int(len(idx)))
        bin_edges.append((float(all_p[sorted_idx[start]]), float(all_p[sorted_idx[end - 1]])))

    ece = float(np.sum([
        (bc / n_total) * abs(mp - mo)
        for bc, mp, mo in zip(bin_counts, bin_mean_pred, bin_mean_obs)
    ]))

    # Bootstrap CI for each bin's mean_obs (resample full sequences)
    n_seq = len(all_bm)
    bin_obs_boot = [[] for _ in range(n_bins)]
    for _ in range(n_boot):
        seq_idx = boot_rng.choice(n_seq, size=n_seq, replace=True)
        boot_p = np.concatenate([all_bm[i] for i in seq_idx])
        boot_y = np.concatenate([all_z[i] for i in seq_idx])
        sorted_boot = np.argsort(boot_p)
        boot_n = len(boot_p)
        bbin_size = boot_n // n_bins
        for b in range(n_bins):
            s = b * bbin_size
            e = s + bbin_size if b < n_bins - 1 else boot_n
            idx = sorted_boot[s:e]
            bin_obs_boot[b].append(float(np.mean(boot_y[idx])))
    bin_obs_ci_lo = [float(np.percentile(x, 2.5)) for x in bin_obs_boot]
    bin_obs_ci_hi = [float(np.percentile(x, 97.5)) for x in bin_obs_boot]

    return {
        "bin_mean_pred": bin_mean_pred,
        "bin_mean_obs": bin_mean_obs,
        "bin_counts": bin_counts,
        "bin_obs_ci_lo": bin_obs_ci_lo,
        "bin_obs_ci_hi": bin_obs_ci_hi,
        "bin_edges": bin_edges,
        "ece": ece,
        "n_total_boundary_positions": n_total,
    }


# ---------------------------------------------------------------------------
# Section 9: Figure generation
# ---------------------------------------------------------------------------

def _make_calibration_figure(
    calib_by_strata: dict,
    out_path: Path,
) -> None:
    """Figure E01(a): boundary reliability by k_true/delta strata with bootstrap bands.

    Because predicted P(boundary|y) is bounded by k_true/n << 1, a 0-1 linear
    scale would squish all points into the lower-left corner.  Each panel is
    zoomed to the data range with a 20% margin so calibration structure is
    visible.  Eight strata are arranged in a 2×4 grid.
    """
    from _style import COLORS, get_figsize, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    strata = list(calib_by_strata.items())
    n_panels = max(len(strata), 1)
    n_cols = min(n_panels, 4)
    n_rows = math.ceil(n_panels / n_cols)

    # 4.5 in per column, 4.2 in per row — large enough to read comfortably
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.2 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    ax_flat = axes.flatten()
    for ax in ax_flat[n_panels:]:
        ax.set_visible(False)

    blue = COLORS["blue"]
    for ax, (label, calib) in zip(ax_flat, strata):
        x = calib["bin_mean_pred"]
        y = calib["bin_mean_obs"]
        lo = calib["bin_obs_ci_lo"]
        hi = calib["bin_obs_ci_hi"]

        x_max = max(x) if x else 1.0
        y_max = max(max(y), max(hi)) if y else 1.0
        xy_max = max(x_max, y_max) * 1.22

        ax.plot([0, xy_max], [0, xy_max], color=COLORS["grey"], lw=1.5,
                ls="--", label="Perfect cal.")
        ax.plot(x, y, "o-", color=blue, ms=7, lw=2.0, label="Observed")
        ax.fill_between(x, lo, hi, alpha=0.20, color=blue)
        ax.set_xlabel("Predicted P(boundary | y)", fontsize=12)
        ax.set_ylabel("Observed frequency", fontsize=12)
        ax.annotate(label, xy=(0.02, 0.97), xycoords='axes fraction', fontsize=8, va='top')
        ax.set_xlim(0, xy_max)
        ax.set_ylim(0, xy_max)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10)
        ax.text(0.04, 0.88, f"ECE = {calib['ece']:.4f}",
                transform=ax.transAxes, fontsize=11)

    save_figure(fig, out_path, formats=("pdf", "png"))
    logger.info("Saved calibration figure to %s", out_path)


def _make_recovery_figure(
    cell_summaries: list[dict],
    out_path: Path,
) -> None:
    """Figure E01(b): F1@3 vs jump size (Δ/σ) per n and k_true with 95% CIs."""
    from _style import COLORS, COLOR_CYCLE, get_figsize, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    primary = [c for c in cell_summaries if not c["ablation"] and c["k_true"] > 1]
    if not primary:
        return

    n_values = sorted({c["n"] for c in primary})
    k_values = sorted({c["k_true"] for c in primary})
    k_colors = {k: c for k, c in zip(k_values, [COLORS["blue"], COLORS["red"]])}
    markers = {n: m for n, m in zip(sorted(n_values), ["o", "s"])}

    fig, axes = plt.subplots(1, len(n_values), figsize=(6.5 * len(n_values), 5.5),
                             constrained_layout=True, squeeze=False)
    for ax, n in zip(axes[0], n_values):
        for k in k_values:
            cells = [c for c in primary if c["n"] == n and c["k_true"] == k]
            cells.sort(key=lambda c: c["delta"])
            if not cells:
                continue
            deltas = [c["delta"] for c in cells]
            means = [c["mean_f1"] for c in cells]
            lo = [c["f1_ci_lo"] for c in cells]
            hi = [c["f1_ci_hi"] for c in cells]
            yerr = np.array([[m - l, h_ - m] for m, l, h_ in zip(means, lo, hi)]).T
            ax.errorbar(
                deltas, means, yerr=yerr,
                fmt=f"{markers.get(n, 'o')}-",
                color=k_colors.get(k, COLORS["grey"]),
                label=f"k={k}",
                capsize=3,
            )
        ax.set_xlabel(f"Δ/σ (signal-to-noise),  n = {n}")
        ax.set_ylabel("F1 @ τ=3")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color=COLORS["grey"], lw=0.8, ls="--")
        ax.legend()

    save_figure(fig, out_path, formats=("pdf", "png"))
    logger.info("Saved recovery figure to %s", out_path)


# ---------------------------------------------------------------------------
# Section 10: LaTeX table generation
# ---------------------------------------------------------------------------

def _fmt(v: float | None, fmt: str = ".3f") -> str:
    if v is None or not math.isfinite(v):
        return "---"
    return f"{v:{fmt}}"


def _fmt_ci(mean: float, lo: float, hi: float) -> str:
    if not math.isfinite(mean):
        return "---"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def generate_latex_table(
    cell_summaries: list[dict],
    out_path: Path,
) -> None:
    """Generate tab_P1_recovery.tex from cell summaries."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{E01: Gaussian recovery (primary + ablation). "
        r"Geometric-decay p(k) prior. "
        r"Mean [95\% bootstrap CI] over 100 independent datasets per cell. "
        r"F1 at boundary tolerance $\tau=3$; null cells report expected false-boundary count.}",
        r"\label{tab:E01-recovery-paper}",
        r"\footnotesize",
        r"\begin{tabular}{ccccccccc}",
        r"\toprule",
        r"$n$ & $k_{\rm true}$ & $\Delta/\sigma$ & Prior & F1@3 & P($k$=true) & Count MAE & Brier & Signal MSE \\",
        r"\midrule",
    ]
    # Sort: primary first, then ablation; by n, k_true, delta
    sorted_cells = sorted(
        cell_summaries,
        key=lambda c: (c["ablation"], c["n"], c["k_true"], c["delta"])
    )
    current_n = None
    for c in sorted_cells:
        n = c["n"]
        if n != current_n:
            if current_n is not None:
                lines.append(r"\midrule")
            current_n = n

        prior_label = "Unif." if c["prior"] == "uniform" else "Geom."
        k_true = c["k_true"]

        if k_true == 1:
            f1_str = f"FP={_fmt(c.get('mean_n_false_at_3'))}"
        else:
            f1_str = _fmt_ci(c.get("mean_f1", float("nan")), c.get("f1_ci_lo", float("nan")), c.get("f1_ci_hi", float("nan")))

        row = (
            f"{n} & {k_true} & {c['delta']:.1f} & {prior_label}"
            f" & {f1_str}"
            f" & {_fmt_ci(c.get('mean_p_k_true_given_y', float('nan')), c.get('p_k_true_given_y_ci_lo', float('nan')), c.get('p_k_true_given_y_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_count_abs_error', float('nan')), c.get('count_abs_error_ci_lo', float('nan')), c.get('count_abs_error_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_brier', float('nan')), c.get('brier_ci_lo', float('nan')), c.get('brier_ci_hi', float('nan')))}"
            f" & {_fmt_ci(c.get('mean_signal_mse', float('nan')), c.get('signal_mse_ci_lo', float('nan')), c.get('signal_mse_ci_hi', float('nan')))}"
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

def run_e01(pilot: bool = False) -> None:
    """Execute the full E01 Gaussian calibration experiment."""
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    log_path = RESULTS_BASE / "run_log.txt"
    _setup_logging(log_path)

    n_replicates = 5 if pilot else E01_CONFIG["replicates"]
    n_boot = 200 if pilot else E01_CONFIG["bootstrap_resamples"]
    mode_label = "PILOT" if pilot else "CONFIRMATORY"
    logger.info("=" * 70)
    logger.info("E01 Gaussian calibration — %s run", mode_label)
    logger.info("Config SHA256: %s", E01_CONFIG_SHA256[:16])
    logger.info("n_replicates=%d, n_boot=%d", n_replicates, n_boot)
    logger.info("=" * 70)

    utc_start = datetime.now(timezone.utc).isoformat()
    start_wall = time.perf_counter()

    cells = enumerate_cells(include_ablation=True)
    logger.info("Total cells: %d", len(cells))

    all_replicate_records: list[dict] = []
    cell_summaries: list[dict] = []
    failed_runs: list[dict] = []

    # Separate boundary probability arrays for calibration curves
    # Key: (n, k_true, prior) → list of (bm, z) arrays per replicate
    calib_pools: dict[tuple, tuple[list, list]] = {}

    taus = E01_CONFIG["boundary_tolerances"]
    min_L = E01_CONFIG["minimum_generating_segment_length"]
    k_max = E01_CONFIG["k_max"]

    for cell_idx, cell in enumerate(cells):
        n = cell["n"]
        k_true = cell["k_true"]
        delta = cell["delta"]
        prior_kind = cell["prior"]
        is_ablation = cell["ablation"]
        cell_index = cell["cell_index"]

        label = f"n={n},k={k_true},Δ={delta:.1f},prior={prior_kind[:4]}"
        logger.info("[Cell %d/%d] %s", cell_idx + 1, len(cells), label)

        rep_metrics: list[dict] = []
        bm_pool: list[NDArray] = []
        z_pool: list[NDArray] = []

        for rep in range(n_replicates):
            # Seed: [20261021, 1, cell_index, replicate] per protocol
            seed_seq = np.random.SeedSequence([20261021, 1, cell_index, rep])

            try:
                dataset = generate_dataset(n, k_true, delta, 1.0, min_L, seed_seq)
                infer = run_inference(dataset["y"], dataset["x"], n, prior_kind, k_max)
                metrics = compute_replicate_metrics(dataset, infer, taus)

                # Collect boundary probs for calibration
                bm = infer["boundary_probs_marginal"]
                true_interior = set(
                    b for b in dataset["true_boundaries"] if b not in (0, n)
                )
                z = np.array([float(i in true_interior) for i in range(1, n)], dtype=float)
                bm_pool.append(bm)
                z_pool.append(z)

                record = {
                    "n": n, "k_true": k_true, "delta": delta, "prior": prior_kind,
                    "cell_index": cell_index, "replicate": rep,
                    "ablation": is_ablation,
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
                    "n": n, "k_true": k_true, "delta": delta, "prior": prior_kind,
                    "cell_index": cell_index, "replicate": rep, "error": str(exc),
                })

        if not rep_metrics:
            logger.warning("Cell %s: all replicates failed!", label)
            continue

        # Bootstrap seed: deterministic per cell
        boot_rng = np.random.default_rng([20261021, 2, cell_index])
        summary = aggregate_cell(rep_metrics, cell, boot_rng, n_boot)
        cell_summaries.append(summary)

        # Accumulate calibration pool (group by strata for figure)
        strata_key = (n, k_true, prior_kind)
        if strata_key not in calib_pools:
            calib_pools[strata_key] = ([], [])
        calib_pools[strata_key][0].extend(bm_pool)
        calib_pools[strata_key][1].extend(z_pool)

        elapsed = time.perf_counter() - start_wall
        logger.info("  Cell done | elapsed %.1fs | n_rep=%d", elapsed, len(rep_metrics))

    # -----------------------------------------------------------------------
    # Calibration curves (per n, per k_true strata)
    # -----------------------------------------------------------------------
    logger.info("Computing calibration curves...")
    calib_results: dict[str, dict] = {}
    for (n, k_true, prior_kind), (bm_list, z_list) in calib_pools.items():
        if k_true == 1 or not bm_list:
            continue
        label = f"n={n},k={k_true},prior={prior_kind[:4]}"
        calib_rng = np.random.default_rng([20261021, 3, n, k_true])
        calib = compute_calibration_curve(
            bm_list, z_list, n_bins=E01_CONFIG["calibration_bins"],
            n_boot=n_boot, boot_rng=calib_rng,
        )
        calib_results[label] = calib
        logger.info("  Calibration %s: ECE=%.4f", label, calib["ece"])

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    logger.info("Generating figures...")
    _make_calibration_figure(
        calib_results,
        FIGURES_DIR / "fig_E01_calibration",
    )
    _make_recovery_figure(
        cell_summaries,
        FIGURES_DIR / "fig_E01_recovery",
    )

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    logger.info("Generating LaTeX table...")
    generate_latex_table(cell_summaries, TABLES_DIR / "tab_P1_recovery.tex")

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
    # Metrics JSON
    # -----------------------------------------------------------------------
    utc_end = datetime.now(timezone.utc).isoformat()
    n_cells_completed = len(cell_summaries)
    n_cells_planned = len(cells)
    n_reps_completed = len(all_replicate_records)
    n_reps_failed = len(failed_runs)

    metrics_json = {
        "experiment_id": "E01",
        "n_cells_planned": n_cells_planned,
        "n_cells_completed": n_cells_completed,
        "n_cells_failed": n_cells_planned - n_cells_completed,
        "n_replicates_planned": n_cells_planned * n_replicates,
        "n_replicates_completed": n_reps_completed,
        "n_replicates_failed": n_reps_failed,
        "pilot_mode": pilot,
        "gate_pass": (n_reps_failed == 0 and n_cells_completed == n_cells_planned),
        "cell_summaries": cell_summaries,
        "calibration_results": calib_results,
    }
    metrics_json_path = RESULTS_BASE / "metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # -----------------------------------------------------------------------
    # Per-replicate records (oracle_checks.json — same field as E00 for consistency)
    # -----------------------------------------------------------------------
    records_path = RESULTS_BASE / "oracle_checks.json"
    with open(records_path, "w") as f:
        json.dump(all_replicate_records, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # -----------------------------------------------------------------------
    # Failed runs
    # -----------------------------------------------------------------------
    failed_csv_path = RESULTS_BASE / "failed_runs.csv"
    if failed_runs:
        with open(failed_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failed_runs[0].keys()))
            writer.writeheader()
            writer.writerows(failed_runs)
        logger.warning("%d failed replicate(s) written to %s", len(failed_runs), failed_csv_path)
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
    artifact_sha256 = _sha256_file(records_path)

    manifest = {
        "experiment_id": "E01",
        "protocol_revision": E01_CONFIG["protocol_revision"],
        "config_sha256": E01_CONFIG_SHA256,
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
        "seed": E01_CONFIG["seed_base"] + [0, 0],
        "input_sha256": "deterministic_generator",
        "source_kind": "gaussian_simulation",
        "family": "gaussian",
        "family_descriptor": "Normal-Normal conjugate, sigma2=1, nu=0, rho2=4",
        "prior": "geometric_decay_log4",
        "k_max": E01_CONFIG["k_max"],
        "coordinate_axis": "index_uniform",
        "reference_type": "simulation_exact_inference",
        "fit_status": "completed" if n_reps_failed == 0 else f"partial_{n_reps_failed}_failed",
        "metric_schema": "E01_gaussian_calibration",
        "artifact_sha256": artifact_sha256,
        "utc_start": utc_start,
        "utc_end": utc_end,
        "pilot_mode": pilot,
        "n_cells_planned": n_cells_planned,
        "n_cells_completed": n_cells_completed,
        "n_replicates": n_replicates,
    }
    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall_time = time.perf_counter() - start_wall
    logger.info("=" * 70)
    logger.info("E01 %s complete in %.1fs", mode_label, wall_time)
    logger.info("Cells: %d planned, %d completed, %d failed",
                n_cells_planned, n_cells_completed, n_cells_planned - n_cells_completed)
    logger.info("Replicates: %d completed, %d failed", n_reps_completed, n_reps_failed)
    logger.info("gate_pass: %s", metrics_json["gate_pass"])
    logger.info("Results: %s", RESULTS_BASE)
    logger.info("=" * 70)


def regenerate_figures(
    manuscript_dest: Path | None = None,
    tables_dest: Path | None = None,
) -> None:
    """Regenerate E01 figures and tables from existing metrics.json.

    Does not re-run the experiment.  Reads the frozen metrics.json from the
    results directory keyed by config SHA256, regenerates figures and the
    LaTeX recovery table, and optionally copies outputs to the manuscript
    shared directories.

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
    calib_results = data.get("calibration_results", {})

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    calib_stem = FIGURES_DIR / "fig_E01_calibration"
    recovery_stem = FIGURES_DIR / "fig_E01_recovery"
    _make_calibration_figure(calib_results, calib_stem)
    _make_recovery_figure(cell_summaries, recovery_stem)
    generate_latex_table(cell_summaries, TABLES_DIR / "tab_E01_recovery.tex")

    if manuscript_dest is not None:
        manuscript_dest = Path(manuscript_dest)
        manuscript_dest.mkdir(parents=True, exist_ok=True)
        for stem in [calib_stem, recovery_stem]:
            for suffix in [".pdf", ".png"]:
                src = stem.with_suffix(suffix)
                if src.exists():
                    shutil.copy2(src, manuscript_dest / src.name)
                    logger.info("Copied %s → %s", src.name, manuscript_dest)

    if tables_dest is not None:
        tables_dest = Path(tables_dest)
        tables_dest.mkdir(parents=True, exist_ok=True)
        table_src = TABLES_DIR / "tab_E01_recovery.tex"
        if table_src.exists():
            shutil.copy2(table_src, tables_dest / table_src.name)
            logger.info("Copied %s → %s", table_src.name, tables_dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E01 Gaussian calibration experiment.")
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
        run_e01(pilot=args.pilot)
