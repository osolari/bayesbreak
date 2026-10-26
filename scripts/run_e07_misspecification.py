"""E07: misspecification and missing data.

Protocol: .claude/skills/bayesbreak-release-audit/protocols/experiments.json §E07
Reference: .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E07

Design:
  n=256, k_true=3, true boundaries (85, 170).
  Mean structure (0, 1.5, 0). Reference noise N(0, 1).
  Noise classes:
    gaussian_reference  -- N(0,1) noise, fit Gaussian (correct)
    student_t3          -- t_3/sqrt(3) noise, fit Gaussian (misspecified)
    overdispersed_poisson -- NegBin(r=2, mu) data, fit Poisson (misspecified) and NegBin (correct)
    beta_binomial       -- BetaBinomial(m=20, ICC=0.1) data, fit Binomial (misspecified)
  Missingness: none, mcar_10, mcar_30, single_block_15.
  100 replicates per cell, k_max=10.
  Seed: SeedSequence([20261021, 7, cell_index, replicate]);
        child 0 = data, child 1 = missingness.

Outputs in: results/confirmatory/E07_misspecification/<config_sha256[:16]>/
  manifest.json, metrics.json, metrics.csv, run_log.txt,
  failed_runs.csv, replicates.csv, figures/, tables/

Usage:
    uv run python scripts/run_e07_misspecification.py
    uv run python scripts/run_e07_misspecification.py --pilot
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
import scipy.stats

# ---------------------------------------------------------------------------
# Frozen E07 configuration
# ---------------------------------------------------------------------------

E07_CONFIG: dict[str, Any] = {
    "id": "E07",
    "name": "misspecification",
    "protocol_revision": "2026-10-21-2",
    "type": "simulation",
    "n": 256,
    "k_true": 3,
    "true_boundaries": [85, 170],
    "segment_means": [0.0, 1.5, 0.0],
    "gaussian_sigma2": 1.0,
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0},
    # Poisson reference rates (generating from NegBin with same means)
    "poisson_rates": [2.0, 5.0, 2.0],
    "poisson_prior": {"alpha": 2.0, "beta": 1.0},
    # NegBin generating dispersion and fit prior
    "negbin_r": 2.0,
    "negbin_prior": {"alpha": 2.0, "beta": 3.0},
    # Binomial trials and Beta-Binomial ICC
    "binomial_m": 20,
    "binomial_probs": [0.2, 0.6, 0.2],
    "betabinomial_icc": 0.1,
    "binomial_prior": {"alpha": 2.0, "beta": 3.0},
    # Single-block missingness: floor(0.15 * n) = 38
    "single_block_miss_frac": 0.15,
    "k_max": 10,
    "replicates": 100,
    "seed_base": [20261021, 7],
    "prior": "geometric_decay_log4",
    "bootstrap_resamples": 2000,
    "boundary_tolerance_primary": 3,
    "boundary_tolerances": [0, 1, 3, 5],
    "noise_classes": [
        "gaussian_reference",
        "student_t3",
        "overdispersed_poisson",
        "beta_binomial",
    ],
    "missingness_settings": ["none", "mcar_10", "mcar_30", "single_block_15"],
    "mcar_10_prob": 0.10,
    "mcar_30_prob": 0.30,
}

E07_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E07_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E07_misspecification"
    / E07_CONFIG_SHA256[:16]
)
FIGURES_DIR = RESULTS_BASE / "figures"
TABLES_DIR = RESULTS_BASE / "tables"

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
# Cell enumeration
# ---------------------------------------------------------------------------

def _build_cells() -> list[dict[str, Any]]:
    """Enumerate all 20 cells for E07.

    Each cell: data_family, fit_family, missingness, is_correct, cell_index.
    """
    cells: list[dict[str, Any]] = []
    idx = 0
    for miss in E07_CONFIG["missingness_settings"]:
        cells.append({
            "cell_index": idx,
            "data_family": "gaussian_reference",
            "fit_family": "gaussian",
            "missingness": miss,
            "is_correct": True,
        })
        idx += 1
    for miss in E07_CONFIG["missingness_settings"]:
        cells.append({
            "cell_index": idx,
            "data_family": "student_t3",
            "fit_family": "gaussian",
            "missingness": miss,
            "is_correct": False,
        })
        idx += 1
    for miss in E07_CONFIG["missingness_settings"]:
        cells.append({
            "cell_index": idx,
            "data_family": "overdispersed_poisson",
            "fit_family": "poisson",
            "missingness": miss,
            "is_correct": False,
        })
        idx += 1
    for miss in E07_CONFIG["missingness_settings"]:
        cells.append({
            "cell_index": idx,
            "data_family": "overdispersed_poisson",
            "fit_family": "negbin",
            "missingness": miss,
            "is_correct": True,
        })
        idx += 1
    for miss in E07_CONFIG["missingness_settings"]:
        cells.append({
            "cell_index": idx,
            "data_family": "beta_binomial",
            "fit_family": "binomial",
            "missingness": miss,
            "is_correct": False,
        })
        idx += 1
    return cells


CELLS = _build_cells()
assert len(CELLS) == 20, f"Expected 20 cells, got {len(CELLS)}"


# ---------------------------------------------------------------------------
# Geometric-decay count prior (p(k) ∝ exp(-log(4)*(k-1)))
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


# ---------------------------------------------------------------------------
# True signal construction
# ---------------------------------------------------------------------------

def _true_signal(n: int, boundaries: list[int], means: list[float]) -> np.ndarray:
    """Build true piecewise-constant signal of length n.

    boundaries: interior boundary positions (1-indexed), length k_true-1.
    means: segment means, length k_true.
    """
    sig = np.zeros(n, dtype=float)
    segs = [0] + boundaries + [n]
    for q, (lo, hi) in enumerate(zip(segs[:-1], segs[1:])):
        sig[lo:hi] = means[q]
    return sig


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _segment_of(i: int, boundaries: list[int]) -> int:
    """Return segment index (0-based) for observation index i."""
    for q, b in enumerate(boundaries):
        if i < b:
            return q
    return len(boundaries)


def _generate_dataset(
    data_family: str,
    rng_data: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate observations y and optional sample_weight (exposures).

    Returns (y, sample_weight). sample_weight is None for Gaussian/t3/Binomial.
    For Poisson: sample_weight = exposure (all 1s), returned so the caller can
    pass it to fit().
    """
    n = int(E07_CONFIG["n"])
    boundaries = list(E07_CONFIG["true_boundaries"])
    seg_means = list(E07_CONFIG["segment_means"])

    y = np.zeros(n, dtype=float)

    if data_family == "gaussian_reference":
        true_sig = _true_signal(n, boundaries, seg_means)
        noise = rng_data.standard_normal(n)
        y = true_sig + noise
        return y, None

    elif data_family == "student_t3":
        true_sig = _true_signal(n, boundaries, seg_means)
        # t_3 / sqrt(3) has variance 1
        t_noise = rng_data.standard_t(3, size=n) / math.sqrt(3.0)
        y = true_sig + t_noise
        return y, None

    elif data_family == "overdispersed_poisson":
        rates = list(E07_CONFIG["poisson_rates"])  # [2, 5, 2]
        r_disp = float(E07_CONFIG["negbin_r"])
        # Build per-observation rate array (vectorized)
        mu_arr = np.array([rates[_segment_of(i, boundaries)] for i in range(n)], dtype=float)
        p_arr = r_disp / (r_disp + mu_arr)
        y = rng_data.negative_binomial(int(r_disp), p_arr).astype(float)
        return y, np.ones(n, dtype=float)

    elif data_family == "beta_binomial":
        m = int(E07_CONFIG["binomial_m"])
        probs = list(E07_CONFIG["binomial_probs"])
        icc = float(E07_CONFIG["betabinomial_icc"])
        phi = (1.0 - icc) / icc  # = 9.0 when icc=0.1
        # Build per-observation Beta parameters (vectorized)
        mu_arr = np.array([probs[_segment_of(i, boundaries)] for i in range(n)], dtype=float)
        a_arr = mu_arr * phi
        b_arr = (1.0 - mu_arr) * phi
        p_draw = rng_data.beta(a_arr, b_arr)
        y = rng_data.binomial(m, p_draw).astype(float)
        return y, None

    else:
        raise ValueError(f"Unknown data_family={data_family!r}")


# ---------------------------------------------------------------------------
# Missingness mask generation
# ---------------------------------------------------------------------------

def _apply_missingness(
    missingness: str,
    n: int,
    rng_miss: np.random.Generator,
) -> np.ndarray:
    """Return boolean mask of shape (n,): True = observed, False = missing.

    single_block_15: floor(0.15*n)=38 consecutive obs, start uniformly drawn
    from [0, n-block_len].
    """
    mask = np.ones(n, dtype=bool)
    if missingness == "none":
        return mask
    elif missingness == "mcar_10":
        p_miss = float(E07_CONFIG["mcar_10_prob"])
        drop = rng_miss.random(n) < p_miss
        mask[drop] = False
    elif missingness == "mcar_30":
        p_miss = float(E07_CONFIG["mcar_30_prob"])
        drop = rng_miss.random(n) < p_miss
        mask[drop] = False
    elif missingness == "single_block_15":
        block_len = int(math.floor(float(E07_CONFIG["single_block_miss_frac"]) * n))
        max_start = n - block_len
        start = int(rng_miss.integers(0, max_start + 1))
        mask[start:start + block_len] = False
    else:
        raise ValueError(f"Unknown missingness={missingness!r}")
    return mask


# ---------------------------------------------------------------------------
# Vectorized Bayes curve (avoids slow Python double-loop in production DP)
# ---------------------------------------------------------------------------

def _logsumexp_axis0(a: np.ndarray) -> np.ndarray:
    """LogSumExp over axis 0, numerically stable."""
    amax = np.max(a, axis=0, keepdims=True)
    finite_max = np.isfinite(amax)
    # Substitute 0 for -inf amax to avoid (-inf)-(-inf)=nan in the subtraction.
    amax_safe = np.where(finite_max, amax, 0.0)
    shifted = np.where(finite_max, a - amax_safe, 0.0)
    out = np.where(
        finite_max[0],
        np.log(np.sum(np.exp(shifted), axis=0)) + amax[0],
        -np.inf,
    )
    return out


def _bayes_curve_fixed_k_vec(
    L: np.ndarray,
    R: np.ndarray,
    lA0: np.ndarray,
    A1: np.ndarray,
    n: int,
    k: int,
) -> np.ndarray:
    """Vectorized fixed-k Bayes regression curve E[mu_t|y,k]."""
    denom = float(L[k, n])
    if not np.isfinite(denom):
        return np.full(n, np.nan, dtype=float)
    LL = L[0:k, :]    # (k, n+1)
    RR = R[k-1::-1, :]  # (k, n+1)
    cross = _logsumexp_axis0(LL[:, :, np.newaxis] + RR[:, np.newaxis, :])  # (n+1, n+1)
    log_w = cross + lA0 - denom
    finite_mask = np.isfinite(log_w) & np.isfinite(A1) & (A1 != 0.0)
    log_w_clipped = np.where(finite_mask, np.clip(log_w, -745.0, 0.0), -np.inf)
    contrib = np.where(finite_mask, np.exp(log_w_clipped) * A1, 0.0)
    diff = contrib.sum(axis=1) - contrib.sum(axis=0)
    return np.cumsum(diff)[:n]


def _bayes_curve_mixed_k_vec(
    L: np.ndarray,
    R: np.ndarray,
    lA0: np.ndarray,
    A1: np.ndarray,
    n: int,
    k_max: int,
    post_k: np.ndarray,
) -> np.ndarray:
    """Vectorized mixed-k Bayes regression curve sum_k P(k|y) E[mu_t|y,k]."""
    out = np.zeros(n, dtype=float)
    for k in range(1, k_max + 1):
        wk = float(post_k[k - 1])
        if wk == 0.0 or not np.isfinite(float(L[k, n])):
            continue
        out += wk * _bayes_curve_fixed_k_vec(L, R, lA0, A1, n, k)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    obs_mask: np.ndarray,
    fit_family: str,
) -> dict:
    """Fit the model and return posterior quantities.

    Returns dict with:
      k_map, k_posterior (k_max,), log_evidence,
      map_boundaries, bayes_curve_mean (n,) or None,
      log_normalizer_ok (bool), fit_failed (bool), fail_reason (str)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    k_max = int(E07_CONFIG["k_max"])
    n = int(E07_CONFIG["n"])
    X = np.arange(1, n + 1, dtype=float).reshape(-1, 1)

    try:
        if fit_family == "gaussian":
            from bayesbreak.families.gaussian import BayesBreakGaussian
            est = BayesBreakGaussian(
                estimate_hyper=False,
                nu=float(E07_CONFIG["gaussian_prior"]["nu"]),
                rho2=float(E07_CONFIG["gaussian_prior"]["rho2"]),
                sigma2=float(E07_CONFIG["gaussian_sigma2"]),
                k_max=k_max,
                regression_curve="none",
                prior_k=_geom_prior,
            )
            est.fit(X, y, observation_mask=obs_mask)
            # Compute vectorized mixed-k Bayes curve for Gaussian signal MSE
            bayes_curve = _bayes_curve_mixed_k_vec(
                est.log_left_, est.log_right_,
                est.log_block_evidence_, est.block_first_moment_,
                n, k_max, est.k_posterior_,
            )

        elif fit_family == "poisson":
            from bayesbreak.families.poisson import BayesBreakPoisson
            est = BayesBreakPoisson(
                estimate_hyper=False,
                alpha=float(E07_CONFIG["poisson_prior"]["alpha"]),
                beta=float(E07_CONFIG["poisson_prior"]["beta"]),
                k_max=k_max,
                regression_curve="none",
                prior_k=_geom_prior,
            )
            sw = sample_weight if sample_weight is not None else np.ones(n)
            est.fit(X, y, sample_weight=sw, observation_mask=obs_mask)
            bayes_curve = None

        elif fit_family == "negbin":
            from bayesbreak.families.negative_binomial import BayesBreakNegBin
            est = BayesBreakNegBin(
                estimate_hyper=False,
                r=float(E07_CONFIG["negbin_r"]),
                alpha=float(E07_CONFIG["negbin_prior"]["alpha"]),
                beta=float(E07_CONFIG["negbin_prior"]["beta"]),
                k_max=k_max,
                regression_curve="none",
                prior_k=_geom_prior,
            )
            # NegBin._compute_block_evidence does not read likelihood_power_,
            # so observation_mask alone does not zero missing obs in block sums.
            # Pass obs_mask as sample_weight (0=missing, 1=observed) without
            # observation_mask so _r_arr_ is set to size n (full sequence).
            est.fit(X, y, sample_weight=obs_mask.astype(float))
            bayes_curve = None

        elif fit_family == "binomial":
            from bayesbreak.families.binomial import BayesBreakBinomial
            est = BayesBreakBinomial(
                estimate_hyper=False,
                n_trials=int(E07_CONFIG["binomial_m"]),
                alpha=float(E07_CONFIG["binomial_prior"]["alpha"]),
                beta=float(E07_CONFIG["binomial_prior"]["beta"]),
                k_max=k_max,
                regression_curve="none",
                prior_k=_geom_prior,
            )
            # Same workaround as NegBin: pass obs_mask as sample_weight.
            # Binomial._compute_block_evidence uses w to weight prefix sums;
            # w=0 for missing obs correctly excludes them from block integrals.
            est.fit(X, y, sample_weight=obs_mask.astype(float))
            bayes_curve = None

        else:
            raise ValueError(f"Unknown fit_family={fit_family!r}")

        log_evidence = float(est.log_evidence_)
        if not np.isfinite(log_evidence):
            return {
                "k_map": -1, "k_posterior": None, "log_evidence": log_evidence,
                "map_boundaries": [], "bayes_curve_mean": None,
                "log_normalizer_ok": False, "fit_failed": True,
                "fail_reason": f"non-finite log_evidence={log_evidence}",
            }

        k_posterior = np.array(est.k_posterior_, dtype=float)
        return {
            "k_map": int(est.k_map_),
            "k_posterior": k_posterior,
            "log_evidence": log_evidence,
            "map_boundaries": list(est.map_boundaries_),
            "bayes_curve_mean": (
                np.array(bayes_curve, dtype=float)
                if bayes_curve is not None
                else None
            ),
            "log_normalizer_ok": True,
            "fit_failed": False,
            "fail_reason": "",
        }

    except Exception as exc:
        return {
            "k_map": -1, "k_posterior": None, "log_evidence": float("nan"),
            "map_boundaries": [], "bayes_curve_mean": None,
            "log_normalizer_ok": False, "fit_failed": True,
            "fail_reason": repr(exc),
        }


# ---------------------------------------------------------------------------
# Boundary matching (one-to-one, max cardinality)
# ---------------------------------------------------------------------------

def _boundary_match(bhat: list[int], bstar: list[int], tau: int) -> int:
    if not bstar or not bhat:
        return 0
    edges = [
        (i, j, abs(bhat[i] - bstar[j]))
        for i in range(len(bhat))
        for j in range(len(bstar))
        if abs(bhat[i] - bstar[j]) <= tau
    ]
    if not edges:
        return 0
    edges.sort(key=lambda e: e[2])
    matched_hat: set[int] = set()
    matched_star: set[int] = set()
    tp = 0
    for i_h, j_s, _ in edges:
        if i_h not in matched_hat and j_s not in matched_star:
            matched_hat.add(i_h)
            matched_star.add(j_s)
            tp += 1
    return tp


def _f1_metrics(
    map_boundaries: list[int],
    true_boundaries: list[int],
    taus: list[int],
) -> dict[str, float]:
    # Interior boundaries only
    bhat = sorted(b for b in map_boundaries if b not in (0,) and b < int(E07_CONFIG["n"]))
    bstar = sorted(b for b in true_boundaries)
    out: dict[str, float] = {}
    for tau in taus:
        tp = _boundary_match(bhat, bstar, tau)
        fp = len(bhat) - tp
        fn = len(bstar) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        out[f"f1_tau{tau}"] = f1
        out[f"precision_tau{tau}"] = prec
        out[f"recall_tau{tau}"] = rec
    return out


# ---------------------------------------------------------------------------
# Credible set width for k
# ---------------------------------------------------------------------------

def _k_credible_width(k_posterior: np.ndarray, level: float = 0.95) -> int:
    """Width of smallest credible set for k covering >= level of mass.

    Width = max_k_in_set - min_k_in_set (in number of k values).
    """
    k_max = len(k_posterior)
    order = np.argsort(k_posterior)[::-1]
    cumulative = 0.0
    k_in_set: list[int] = []
    for idx in order:
        cumulative += float(k_posterior[idx])
        k_in_set.append(idx + 1)  # 1-indexed
        if cumulative >= level:
            break
    return max(k_in_set) - min(k_in_set) if len(k_in_set) > 1 else 0


# ---------------------------------------------------------------------------
# Per-replicate metric computation
# ---------------------------------------------------------------------------

def _compute_replicate_metrics(
    cell: dict[str, Any],
    replicate: int,
    rng_data: np.random.Generator,
    rng_miss: np.random.Generator,
) -> dict[str, Any]:
    """Run one replicate for a cell. Returns a flat metric dict."""
    n = int(E07_CONFIG["n"])
    k_true = int(E07_CONFIG["k_true"])
    true_boundaries = list(E07_CONFIG["true_boundaries"])
    true_means = list(E07_CONFIG["segment_means"])
    k_max = int(E07_CONFIG["k_max"])
    taus = list(E07_CONFIG["boundary_tolerances"])

    # Generate data
    y, sw = _generate_dataset(cell["data_family"], rng_data)

    # Apply missingness
    obs_mask = _apply_missingness(cell["missingness"], n, rng_miss)
    n_missing = int(np.sum(~obs_mask))

    # Fit
    result = _run_inference(y, sw, obs_mask, cell["fit_family"])

    base = {
        "cell_index": cell["cell_index"],
        "data_family": cell["data_family"],
        "fit_family": cell["fit_family"],
        "missingness": cell["missingness"],
        "is_correct": cell["is_correct"],
        "replicate": replicate,
        "n_missing": n_missing,
        "fit_failed": result["fit_failed"],
        "fail_reason": result["fail_reason"],
        "log_normalizer_ok": result["log_normalizer_ok"],
    }

    if result["fit_failed"]:
        nan_metrics: dict[str, Any] = {}
        for tau in taus:
            nan_metrics[f"f1_tau{tau}"] = float("nan")
            nan_metrics[f"precision_tau{tau}"] = float("nan")
            nan_metrics[f"recall_tau{tau}"] = float("nan")
        nan_metrics.update({
            "k_map": -1,
            "k_count_exact": float("nan"),
            "p_k_true": float("nan"),
            "p_k_max": float("nan"),
            "brier_k": float("nan"),
            "signal_mse": float("nan"),
            "k_credible_width_95": float("nan"),
        })
        base.update(nan_metrics)
        return base

    k_posterior = result["k_posterior"]
    k_map = result["k_map"]
    map_boundaries = result["map_boundaries"]

    # F1 metrics
    f1_m = _f1_metrics(map_boundaries, true_boundaries, taus)
    base.update(f1_m)

    # Count metrics
    p_k_true = float(k_posterior[k_true - 1]) if k_true <= k_max else 0.0
    p_k_max = float(k_posterior[k_max - 1])
    brier_k = float(np.sum((np.arange(1, k_max + 1) == k_true).astype(float) - k_posterior) ** 2
                    + np.sum(k_posterior ** 2) - 2.0 * p_k_true + 1.0)
    # Brier = sum_k (I{k=k_true} - P(k|y))^2
    brier_k = float(np.sum(((np.arange(1, k_max + 1) == k_true).astype(float) - k_posterior) ** 2))

    base["k_map"] = k_map
    base["k_count_exact"] = float(k_map == k_true)
    base["p_k_true"] = p_k_true
    base["p_k_max"] = p_k_max
    base["brier_k"] = brier_k

    # Credible set width
    base["k_credible_width_95"] = float(_k_credible_width(k_posterior, 0.95))

    # Signal MSE (only for gaussian families where true_means are in the response scale)
    seg_means = list(E07_CONFIG["segment_means"])
    true_sig = _true_signal(n, true_boundaries, seg_means)
    bayes_curve = result["bayes_curve_mean"]
    if bayes_curve is not None and (
        cell["data_family"] in ("gaussian_reference", "student_t3")
        and cell["fit_family"] == "gaussian"
    ):
        # Gaussian families: signal in original response scale
        base["signal_mse"] = float(np.mean((bayes_curve - true_sig) ** 2))
    else:
        # Poisson/NegBin/Binomial: signal MSE not directly comparable to Gaussian means
        # Report NaN to avoid cross-family signal comparisons
        base["signal_mse"] = float("nan")

    return base


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: list[float],
    n_boot: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    boots = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - level) / 2.0
    ci_lo = float(np.percentile(boots, 100 * alpha))
    ci_hi = float(np.percentile(boots, 100 * (1.0 - alpha)))
    return mean, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def _get_environment() -> dict[str, Any]:
    import numpy as np
    import scipy
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        env["git_revision"] = rev
    except Exception:
        env["git_revision"] = "unknown"
    return env


# ---------------------------------------------------------------------------
# Oracle checks (sanity, no data-peeking)
# ---------------------------------------------------------------------------

def _run_oracle_checks() -> dict:
    """Verify that boundary matching and Brier computations are correct.

    Uses n=8, k_true=2, boundary at 4 with perfect recovery.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    issues: list[str] = []

    # F1 oracle: perfect recovery
    bhat = [4]
    bstar = [4]
    tp = _boundary_match(bhat, bstar, tau=3)
    if tp != 1:
        issues.append(f"F1 oracle: expected TP=1 for perfect boundary, got {tp}")

    # F1 oracle: miss by tolerance
    bhat2 = [8]
    bstar2 = [4]
    tp2 = _boundary_match(bhat2, bstar2, tau=3)
    if tp2 != 0:
        issues.append(f"F1 oracle: expected TP=0 for boundary 8 vs 4 (tau=3), got {tp2}")

    # Brier oracle: perfect k prediction
    k_true = 2
    k_max = 10
    kp = np.zeros(k_max)
    kp[k_true - 1] = 1.0
    brier = float(np.sum(((np.arange(1, k_max + 1) == k_true).astype(float) - kp) ** 2))
    if abs(brier) > 1e-12:
        issues.append(f"Brier oracle: expected 0 for perfect k posterior, got {brier}")

    # Brier oracle: uniform k prediction
    kp_uniform = np.full(k_max, 1.0 / k_max)
    brier_uniform = float(
        np.sum(((np.arange(1, k_max + 1) == k_true).astype(float) - kp_uniform) ** 2)
    )
    expected_brier_uniform = (k_max - 1) * (1.0 / k_max) ** 2 + (1.0 - 1.0 / k_max) ** 2
    if abs(brier_uniform - expected_brier_uniform) > 1e-10:
        issues.append(
            f"Brier oracle: uniform mismatch {brier_uniform:.6f} vs {expected_brier_uniform:.6f}"
        )

    # Missingness oracle: MCAR coverage
    rng_test = np.random.default_rng(seed=42)
    mask = _apply_missingness("mcar_30", 10000, rng_test)
    miss_frac = float(np.mean(~mask))
    if abs(miss_frac - 0.30) > 0.02:
        issues.append(f"Missingness oracle: MCAR 30% fraction {miss_frac:.3f} far from 0.30")

    # Single block oracle
    rng_test2 = np.random.default_rng(seed=7)
    n_test = 256
    block_len = int(math.floor(0.15 * n_test))
    mask_sb = _apply_missingness("single_block_15", n_test, rng_test2)
    missing_indices = np.where(~mask_sb)[0]
    if len(missing_indices) != block_len:
        issues.append(
            f"Single-block oracle: expected {block_len} missing, got {len(missing_indices)}"
        )
    # Check contiguous
    if len(missing_indices) > 1:
        diffs = np.diff(missing_indices)
        if not np.all(diffs == 1):
            issues.append("Single-block oracle: missing indices are not contiguous")

    gate_pass = len(issues) == 0
    return {"gate_pass": gate_pass, "issues": issues}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(
    reps: list[dict[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return []

    import pandas as pd
    df = pd.DataFrame(reps)
    df_ok = df[~df["fit_failed"]].copy()

    tau = int(E07_CONFIG["boundary_tolerance_primary"])
    fig_paths: list[Path] = []

    # Figure: F1@tau by noise class and missingness, faceted
    miss_order = list(E07_CONFIG["missingness_settings"])
    noise_order = list(E07_CONFIG["noise_classes"])

    # Gaussian fit cells only for the Gaussian MSE panel
    # Main F1 plot: noise_class x missingness heatmap of mean F1
    fig, axes = plt.subplots(1, len(miss_order), figsize=(14, 5), sharey=True)
    fig.suptitle(f"E07 MAP boundary F1@{tau} by condition", fontsize=12)

    for ax, miss in zip(axes, miss_order):
        means: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for cell in CELLS:
            if cell["missingness"] != miss:
                continue
            cell_df = df_ok[
                (df_ok["cell_index"] == cell["cell_index"])
            ]
            vals = cell_df[f"f1_tau{tau}"].dropna().tolist()
            if vals:
                means.append(float(np.mean(vals)))
            else:
                means.append(float("nan"))
            fam_label = f"{cell['data_family'][:6]}\n{cell['fit_family']}"
            labels.append(fam_label)
            colors.append("steelblue" if cell["is_correct"] else "coral")

        x = np.arange(len(means))
        bars = ax.bar(x, [m if np.isfinite(m) else 0.0 for m in means], color=colors, width=0.7)
        ax.set_title(miss.replace("_", " "), fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.set_ylabel(f"Mean F1@{tau}")
        ax.axhline(1.0, color="gray", lw=0.5, ls="--")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Correct fit"),
        Patch(facecolor="coral", label="Misspecified fit"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fp = figures_dir / "fig_E07_robustness.pdf"
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    fig_paths.append(fp)

    # PNG version
    fp_png = figures_dir / "fig_E07_robustness.png"
    fig2, axes2 = plt.subplots(1, len(miss_order), figsize=(14, 5), sharey=True)
    fig2.suptitle(f"E07 MAP boundary F1@{tau} by condition", fontsize=12)
    for ax, miss in zip(axes2, miss_order):
        means2: list[float] = []
        labels2: list[str] = []
        colors2: list[str] = []
        for cell in CELLS:
            if cell["missingness"] != miss:
                continue
            cell_df2 = df_ok[(df_ok["cell_index"] == cell["cell_index"])]
            vals2 = cell_df2[f"f1_tau{tau}"].dropna().tolist()
            means2.append(float(np.mean(vals2)) if vals2 else float("nan"))
            fam_label2 = f"{cell['data_family'][:6]}\n{cell['fit_family']}"
            labels2.append(fam_label2)
            colors2.append("steelblue" if cell["is_correct"] else "coral")
        x2 = np.arange(len(means2))
        ax.bar(x2, [m if np.isfinite(m) else 0.0 for m in means2], color=colors2, width=0.7)
        ax.set_title(miss.replace("_", " "), fontsize=9)
        ax.set_xticks(x2)
        ax.set_xticklabels(labels2, fontsize=6, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        if ax == axes2[0]:
            ax.set_ylabel(f"Mean F1@{tau}")
        ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    fig2.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9)
    fig2.tight_layout(rect=[0, 0.06, 1, 1])
    fig2.savefig(fp_png, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    fig_paths.append(fp_png)

    return fig_paths


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def _generate_table(
    cell_summaries: list[dict[str, Any]],
    tables_dir: Path,
) -> Path:
    tau = int(E07_CONFIG["boundary_tolerance_primary"])
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{E07 misspecification and missing data: boundary recovery F1@"
        + str(tau)
        + r" (mean $\pm$ 95\% CI), posterior count accuracy, and Brier score for $k=k_\mathrm{true}$.}",
        r"\label{tab:E07-robustness}",
        r"\begin{tabular}{llccccc}",
        r"\hline",
        r"Data family & Fit & Missingness & F1@"
        + str(tau)
        + r" & Count exact & $p(k^\star|y)$ & Brier$_k$ \\",
        r"\hline",
    ]
    for cs in cell_summaries:
        f1_str = (
            f"{cs['mean_f1']:.3f} [{cs['ci_lo_f1']:.3f},{cs['ci_hi_f1']:.3f}]"
            if np.isfinite(cs["mean_f1"])
            else "---"
        )
        cnt_str = f"{cs['mean_count_exact']:.3f}" if np.isfinite(cs.get("mean_count_exact", float("nan"))) else "---"
        pk_str = f"{cs['mean_p_k_true']:.3f}" if np.isfinite(cs.get("mean_p_k_true", float("nan"))) else "---"
        brier_str = f"{cs['mean_brier']:.4f}" if np.isfinite(cs.get("mean_brier", float("nan"))) else "---"
        correct_mark = "" if cs["is_correct"] else r"$\dag$"
        data_fam = cs["data_family"].replace("_", " ")
        fit_fam = cs["fit_family"] + correct_mark
        miss = cs["missingness"].replace("_", " ")
        lines.append(
            f"{data_fam} & {fit_fam} & {miss} & {f1_str} & {cnt_str} & {pk_str} & {brier_str} \\\\"
        )
    lines += [
        r"\hline",
        r"\multicolumn{7}{l}{\small$\dag$ misspecified fit.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    tex_path = tables_dir / "tab_E07_robustness.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    return tex_path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main(pilot: bool = False) -> None:
    n_replicates = 5 if pilot else int(E07_CONFIG["replicates"])
    run_tag = "PILOT" if pilot else "CONFIRMATORY"

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RESULTS_BASE / "run_log.txt"
    _setup_logging(log_path)

    env = _get_environment()
    logger.info("E07 %s run started: %s", run_tag, datetime.now(timezone.utc).isoformat())
    logger.info("Config SHA256: %s", E07_CONFIG_SHA256)
    logger.info("Git revision: %s", env.get("git_revision", "unknown"))
    logger.info("Results dir: %s", RESULTS_BASE)
    logger.info("Cells: %d, replicates: %d, total fits: %d",
                len(CELLS), n_replicates, len(CELLS) * n_replicates)

    # Oracle checks
    oracle_result = _run_oracle_checks()
    logger.info("Oracle checks: gate_pass=%s, issues=%s",
                oracle_result["gate_pass"], oracle_result["issues"])
    if not oracle_result["gate_pass"]:
        logger.error("ORACLE GATE FAILED: %s", oracle_result["issues"])
        sys.exit(1)
    (RESULTS_BASE / "oracle_checks.json").write_text(
        json.dumps(oracle_result, indent=2)
    )

    t_run_start = time.monotonic()
    all_reps: list[dict[str, Any]] = []
    failed_reps: list[dict[str, Any]] = []

    n_total = len(CELLS) * n_replicates
    n_done = 0

    for cell in CELLS:
        ci = cell["cell_index"]
        logger.info(
            "Cell %d/%d: data=%s fit=%s miss=%s",
            ci, len(CELLS) - 1,
            cell["data_family"], cell["fit_family"], cell["missingness"],
        )
        for rep in range(n_replicates):
            seed_seq = np.random.SeedSequence(
                E07_CONFIG["seed_base"] + [ci, rep]
            )
            children = seed_seq.spawn(2)
            rng_data = np.random.default_rng(children[0])
            rng_miss = np.random.default_rng(children[1])

            metrics = _compute_replicate_metrics(cell, rep, rng_data, rng_miss)
            all_reps.append(metrics)

            if metrics["fit_failed"]:
                failed_reps.append(metrics)
                logger.warning(
                    "  rep=%d FAILED: %s", rep, metrics["fail_reason"][:120]
                )
            n_done += 1
            if n_done % 100 == 0:
                elapsed = time.monotonic() - t_run_start
                logger.info("  Progress: %d/%d (%.1fs elapsed)", n_done, n_total, elapsed)

    # Write replicates CSV
    if all_reps:
        fieldnames = sorted(
            set().union(*[set(r.keys()) for r in all_reps])
        )
        reps_path = RESULTS_BASE / "replicates.csv"
        with open(reps_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_reps:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    # Write failed runs CSV
    if failed_reps:
        fail_fieldnames = ["cell_index", "data_family", "fit_family", "missingness",
                           "replicate", "fail_reason"]
        with open(RESULTS_BASE / "failed_runs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fail_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in failed_reps:
                writer.writerow(row)
    else:
        (RESULTS_BASE / "failed_runs.csv").write_text("cell_index,replicate,fail_reason\n")

    # Bootstrap summaries per cell
    tau = int(E07_CONFIG["boundary_tolerance_primary"])
    boot_rng = np.random.default_rng(np.random.SeedSequence([20261021, 7, 999]))
    n_boot = int(E07_CONFIG["bootstrap_resamples"])
    cell_summaries: list[dict[str, Any]] = []

    for cell in CELLS:
        ci = cell["cell_index"]
        cell_reps = [r for r in all_reps if r["cell_index"] == ci and not r["fit_failed"]]
        n_cell = len(cell_reps)
        n_failed_cell = sum(1 for r in all_reps if r["cell_index"] == ci and r["fit_failed"])

        f1_vals = [r[f"f1_tau{tau}"] for r in cell_reps]
        mean_f1, ci_lo_f1, ci_hi_f1 = _bootstrap_ci(f1_vals, n_boot, boot_rng)

        count_exact_vals = [r["k_count_exact"] for r in cell_reps]
        mean_ce, ci_lo_ce, ci_hi_ce = _bootstrap_ci(count_exact_vals, n_boot, boot_rng)

        pk_vals = [r["p_k_true"] for r in cell_reps]
        mean_pk, _, _ = _bootstrap_ci(pk_vals, n_boot, boot_rng)

        brier_vals = [r["brier_k"] for r in cell_reps]
        mean_brier, _, _ = _bootstrap_ci(brier_vals, n_boot, boot_rng)

        mse_vals = [r["signal_mse"] for r in cell_reps if np.isfinite(r.get("signal_mse", float("nan")))]
        mean_mse = float(np.mean(mse_vals)) if mse_vals else float("nan")

        width_vals = [r["k_credible_width_95"] for r in cell_reps]
        mean_width = float(np.mean([v for v in width_vals if np.isfinite(v)])) if width_vals else float("nan")

        cs: dict[str, Any] = {
            "cell_index": ci,
            "data_family": cell["data_family"],
            "fit_family": cell["fit_family"],
            "missingness": cell["missingness"],
            "is_correct": cell["is_correct"],
            "n_completed": n_cell,
            "n_failed": n_failed_cell,
            "mean_f1": mean_f1,
            "ci_lo_f1": ci_lo_f1,
            "ci_hi_f1": ci_hi_f1,
            "mean_count_exact": mean_ce,
            "ci_lo_count_exact": ci_lo_ce,
            "ci_hi_count_exact": ci_hi_ce,
            "mean_p_k_true": mean_pk,
            "mean_brier": mean_brier,
            "mean_signal_mse": mean_mse,
            "mean_k_credible_width_95": mean_width,
        }
        cell_summaries.append(cs)

        logger.info(
            "  Cell %d (%s/%s/%s): n=%d failed=%d F1@%d=%.3f [%.3f,%.3f] "
            "cnt_exact=%.3f brier=%.4f",
            ci, cell["data_family"][:8], cell["fit_family"], cell["missingness"],
            n_cell, n_failed_cell, tau, mean_f1, ci_lo_f1, ci_hi_f1,
            mean_ce, mean_brier,
        )

    # Write metrics
    metrics_path = RESULTS_BASE / "metrics.json"
    metrics_path.write_text(json.dumps({
        "experiment_id": "E07",
        "run_tag": run_tag,
        "n_cells": len(CELLS),
        "n_replicates": n_replicates,
        "n_total_runs": len(all_reps),
        "n_total_failed": len(failed_reps),
        "cell_summaries": cell_summaries,
        "oracle_checks": oracle_result,
    }, indent=2))

    if cell_summaries:
        cs_keys = sorted(
            set().union(*[set(cs.keys()) for cs in cell_summaries])
        )
        metrics_csv_path = RESULTS_BASE / "metrics.csv"
        with open(metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cs_keys, extrasaction="ignore")
            writer.writeheader()
            for cs in cell_summaries:
                writer.writerow({k: cs.get(k, "") for k in cs_keys})

    # Generate table
    tex_path = _generate_table(cell_summaries, TABLES_DIR)
    logger.info("Table written: %s", tex_path)

    # Generate figures
    if not pilot:
        fig_paths = _plot_results(all_reps, FIGURES_DIR)
        for fp in fig_paths:
            logger.info("Figure written: %s", fp)

    # Gate checks
    n_failed_total = len(failed_reps)
    fail_rate = n_failed_total / max(len(all_reps), 1)
    gate_pass = (
        oracle_result["gate_pass"]
        and fail_rate < 0.05  # < 5% failure rate
    )
    logger.info("Gate: oracle=%s, fail_rate=%.1f%%, gate_pass=%s",
                oracle_result["gate_pass"], 100 * fail_rate, gate_pass)

    # Manifest
    wall_time = time.monotonic() - t_run_start

    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    artifact_sha256s: dict[str, str] = {}
    for p in sorted(RESULTS_BASE.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            artifact_sha256s[p.name] = _sha256_file(p)

    manifest = {
        "experiment_id": "E07",
        "protocol_revision": E07_CONFIG["protocol_revision"],
        "config_sha256": E07_CONFIG_SHA256,
        "git_revision": env.get("git_revision", "unknown"),
        "run_tag": run_tag,
        "utc_start": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall_time, 1),
        "python_version": env["python_version"],
        "numpy_version": env["numpy_version"],
        "scipy_version": env["scipy_version"],
        "n_cells": len(CELLS),
        "n_replicates": n_replicates,
        "n_total_runs": len(all_reps),
        "n_total_failed": len(failed_reps),
        "gate_pass": gate_pass,
        "results_dir": str(RESULTS_BASE),
        "artifact_sha256": artifact_sha256s,
    }
    (RESULTS_BASE / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("E07 %s complete in %.1fs. Gate: %s", run_tag, wall_time, gate_pass)
    if not gate_pass:
        logger.error("GATE FAILED: oracle=%s fail_rate=%.2f%%",
                     oracle_result["gate_pass"], 100 * fail_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E07 misspecification experiment")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot run: 5 replicates per cell (fast)")
    args = parser.parse_args()
    main(pilot=args.pilot)
