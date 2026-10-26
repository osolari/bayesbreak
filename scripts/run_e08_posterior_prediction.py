"""E08: held-out predictive distribution.

Protocol: .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E08
Config: .claude/skills/bayesbreak-release-audit/protocols/experiments.json

Scores:
  - Mean held-out log predictive density per observation (all families).
  - 95% interval coverage (Gaussian, BetaObs -- continuous).
  - Randomized PIT KS statistic vs Uniform(0,1) (Poisson, Binomial -- discrete).
  Null comparator: one-segment conjugate predictive (k_max=1 forced).

Produces:
  results/confirmatory/E08_posterior_prediction/<config_sha256[:16]>/
    manifest.json, metrics.json, metrics.csv,
    run_log.txt, failed_runs.csv

Usage (confirmatory, 50 reps, 4 windows each):
    python scripts/run_e08_posterior_prediction.py

Usage (pilot, 5 reps per cell, 2 windows):
    python scripts/run_e08_posterior_prediction.py --pilot
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
from scipy.stats import kstest

# ---------------------------------------------------------------------------
# E08 configuration
# ---------------------------------------------------------------------------

E08_CONFIG: dict[str, Any] = {
    "id": "E08",
    "name": "posterior_prediction",
    "protocol_revision": "2026-10-24-2",
    "type": "simulation",
    "k_true": 3,
    "k_max": 10,
    "n_conjugate": 240,
    "n_beta_quadrature": 120,
    "replicates": 50,
    "holdout_fraction_intervals": [
        [0.10, 0.25],
        [0.30, 0.45],
        [0.55, 0.70],
        [0.75, 0.90],
    ],
    "seed_base": [20261021, 8],
    "prior": "geometric_decay_log4",
    "bootstrap_resamples": 2000,
    # Quadrature for BetaObs (aligned with E02 amendment: 128 nodes).
    "beta_obs_quadrature_points": 128,
    # Family priors (from E02 protocol).
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0, "sigma2": 1.0},
    "poisson_prior": {"alpha": 2.0, "beta": 1.0},
    "binomial_prior": {"alpha": 2.0, "beta": 3.0},
    "beta_obs_prior": {"alpha": 2.0, "beta": 3.0},
    # 8 cells: 4 families x 2 effects each; no NegBin (E02 cells 6-7 are negbin, excluded).
    "cells": [
        {"cell_index": 0, "family": "gaussian",  "n": 240, "param": "delta",  "value": 0.5,  "n_reps": 50},
        {"cell_index": 1, "family": "gaussian",  "n": 240, "param": "delta",  "value": 1.5,  "n_reps": 50},
        {"cell_index": 2, "family": "poisson",   "n": 240, "param": "q",      "value": 1.5,  "n_reps": 50},
        {"cell_index": 3, "family": "poisson",   "n": 240, "param": "q",      "value": 3.0,  "n_reps": 50},
        {"cell_index": 4, "family": "binomial",  "n": 240, "param": "p_mid",  "value": 0.35, "n_reps": 50},
        {"cell_index": 5, "family": "binomial",  "n": 240, "param": "p_mid",  "value": 0.65, "n_reps": 50},
        {"cell_index": 6, "family": "beta_obs",  "n": 120, "param": "mu_mid", "value": 0.4,  "n_reps": 50},
        {"cell_index": 7, "family": "beta_obs",  "n": 120, "param": "mu_mid", "value": 0.8,  "n_reps": 50},
    ],
}

E08_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E08_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E08_posterior_prediction"
    / E08_CONFIG_SHA256[:16]
)
FIGURES_DIR = RESULTS_BASE / "figures"
TABLES_DIR = RESULTS_BASE / "tables"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


def _holdout_windows(n: int, fractions: list[list[float]]) -> list[tuple[int, int]]:
    """Compute holdout [start, end) index pairs from fraction endpoints.

    Rounds deterministically via round(f * n).
    """
    windows = []
    for lo, hi in fractions:
        start = round(lo * n)
        end = round(hi * n)
        assert 0 < start < end < n, f"Invalid window for n={n}: [{start}, {end})"
        windows.append((start, end))
    return windows


# ---------------------------------------------------------------------------
# Data generators (identical procedure to E02, with E08 seeds)
# ---------------------------------------------------------------------------

def _generate_gaussian(n: int, delta: float, rng: np.random.Generator) -> dict:
    seg = n // 3
    assert 3 * seg == n
    true_means = [0.0, delta, 0.0]
    true_signal = np.empty(n, dtype=float)
    for q in range(3):
        true_signal[q * seg:(q + 1) * seg] = true_means[q]
    y = true_signal + rng.standard_normal(n)
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y, "x": x, "true_signal": true_signal,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": None,
    }


def _generate_poisson(n: int, q: float, rng: np.random.Generator) -> dict:
    seg = n // 3
    assert 3 * seg == n
    true_rates = [2.0, 2.0 * q, 2.0]
    e_pattern = [0.5, 1.0, 2.0]
    e_arr = np.array([e_pattern[i % 3] for i in range(n)], dtype=float)
    true_signal = np.empty(n, dtype=float)
    for s in range(3):
        true_signal[s * seg:(s + 1) * seg] = true_rates[s]
    y = np.array(
        [rng.poisson(true_rates[i // seg] * e_arr[i]) for i in range(n)],
        dtype=float,
    )
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y, "x": x, "true_signal": true_signal,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": e_arr,
        "descriptor": {"e_arr": e_arr},
    }


def _generate_binomial(n: int, p_mid: float, rng: np.random.Generator) -> dict:
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
        "y": y, "x": x, "true_signal": true_signal,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": {"m_arr": m_arr},
    }


def _generate_beta_obs(n: int, mu_mid: float, rng: np.random.Generator) -> dict:
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
        y[i] = rng.beta(phi * mu_q, phi * (1.0 - mu_q))
    x = np.arange(1, n + 1, dtype=float)
    return {
        "y": y, "x": x, "true_signal": true_signal,
        "boundaries": [0, seg, 2 * seg, n],
        "sample_weight": np.ones(n, dtype=float),
        "descriptor": {"phi_arr": phi_arr},
    }


def generate_dataset(cell: dict, rng: np.random.Generator) -> dict:
    family = cell["family"]
    n = cell["n"]
    val = cell["value"]
    if family == "gaussian":
        return _generate_gaussian(n, val, rng)
    elif family == "poisson":
        return _generate_poisson(n, val, rng)
    elif family == "binomial":
        return _generate_binomial(n, val, rng)
    elif family == "beta_obs":
        return _generate_beta_obs(n, val, rng)
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------

def _make_estimator(family: str, config: dict, dataset: dict, k_max: int = None):
    """Return a freshly constructed (unfitted) estimator for this family."""
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak.families.poisson import BayesBreakPoisson
    from bayesbreak.families.binomial import BayesBreakBinomial
    from bayesbreak.families.beta_obs import BayesBreakBetaObs

    kmax = k_max if k_max is not None else config["k_max"]

    if family == "gaussian":
        p = config["gaussian_prior"]
        return BayesBreakGaussian(
            k_max=kmax, estimate_hyper=False,
            nu=p["nu"], rho2=p["rho2"], sigma2=p["sigma2"],
            prior_k=_geom_prior,
        )
    elif family == "poisson":
        p = config["poisson_prior"]
        return BayesBreakPoisson(
            k_max=kmax, estimate_hyper=False,
            alpha=p["alpha"], beta=p["beta"],
            prior_k=_geom_prior,
        )
    elif family == "binomial":
        p = config["binomial_prior"]
        m_arr = dataset["descriptor"]["m_arr"]
        return BayesBreakBinomial(
            k_max=kmax, estimate_hyper=False,
            alpha=p["alpha"], beta=p["beta"],
            n_trials=m_arr,
            prior_k=_geom_prior,
        )
    elif family == "beta_obs":
        p = config["beta_obs_prior"]
        phi_arr = dataset["descriptor"]["phi_arr"]
        return BayesBreakBetaObs(
            k_max=kmax, estimate_hyper=False,
            alpha=p["alpha"], beta=p["beta"],
            phi=phi_arr,
            quadrature_points=config["beta_obs_quadrature_points"],
            prior_k=_geom_prior,
        )
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Per-window scoring
# ---------------------------------------------------------------------------

def _holdout_sample_weight(family: str, dataset: dict, holdout_idx: NDArray) -> NDArray:
    """Return the per-observation descriptor weight for scoring held-out points.

    Semantics:
      Gaussian  -- unit weights (no per-observation descriptor for prediction)
      Poisson   -- exposure e_i (passed as sample_weight to posterior_predictive_logpdf)
      Binomial  -- n_trials m_i (passed as sample_weight / w_new to predictive block)
      BetaObs   -- precision phi_i (passed as sample_weight / w_new to predictive block)
    """
    if family == "gaussian":
        return np.ones(len(holdout_idx), dtype=float)
    elif family == "poisson":
        return dataset["descriptor"]["e_arr"][holdout_idx]
    elif family == "binomial":
        return dataset["descriptor"]["m_arr"][holdout_idx]
    elif family == "beta_obs":
        return dataset["descriptor"]["phi_arr"][holdout_idx]
    else:
        raise ValueError(f"Unknown family: {family}")


def _score_window(
    est,
    dataset: dict,
    family: str,
    holdout_idx: NDArray,
    pit_rng_seed: int,
) -> dict:
    """Score held-out observations for a single fitted estimator and window.

    Returns:
      mean_logpd, coverage_95 (continuous), pit_ks_stat, pit_ks_pval (discrete)
    """
    from bayesbreak.prediction import posterior_predictive_logpdf, pit_residuals

    x_full = dataset["x"]
    y_full = dataset["y"]
    x_holdout = x_full[holdout_idx].reshape(-1, 1)
    y_holdout = y_full[holdout_idx]
    w_holdout = _holdout_sample_weight(family, dataset, holdout_idx)
    n_holdout = len(holdout_idx)

    # Log predictive density per observation.
    log_preds = posterior_predictive_logpdf(
        est, x_holdout, y_holdout, sample_weight=w_holdout, per_sample=True,
        extrapolation="nearest",
    )
    mean_logpd = float(np.sum(log_preds)) / n_holdout

    result = {
        "mean_logpd": mean_logpd,
        "coverage_95": float("nan"),
        "pit_ks_stat": float("nan"),
        "pit_ks_pval": float("nan"),
    }

    # Set global seed for randomized PIT reproducibility.
    np.random.seed(pit_rng_seed % (2**31))

    if family in ("gaussian", "beta_obs"):
        # Continuous: interval coverage from PIT values.
        pits = pit_residuals(est, x_holdout, y_holdout, sample_weight=w_holdout,
                             extrapolation="nearest")
        result["coverage_95"] = float(np.mean((pits >= 0.025) & (pits <= 0.975)))
    elif family in ("poisson", "binomial"):
        # Discrete: randomized PIT, test vs Uniform(0, 1).
        pits = pit_residuals(est, x_holdout, y_holdout, sample_weight=w_holdout,
                             extrapolation="nearest")
        ks_result = kstest(pits, "uniform")
        result["pit_ks_stat"] = float(ks_result.statistic)
        result["pit_ks_pval"] = float(ks_result.pvalue)

    return result


# ---------------------------------------------------------------------------
# Main per-replicate runner
# ---------------------------------------------------------------------------

def run_replicate(
    cell: dict,
    rep: int,
    windows: list[tuple[int, int]],
    config: dict,
) -> list[dict]:
    """Run one replicate for a cell: generate data, fit per window, score.

    Returns a list of row dicts (one per window) for metrics.csv.
    """
    from bayesbreak.prediction import posterior_predictive_logpdf, pit_residuals

    cell_idx = cell["cell_index"]
    family = cell["family"]
    n = cell["n"]

    # Seed: SeedSequence([20261021, 8, cell_index, replicate]).spawn(1)[0]
    seed = np.random.SeedSequence([20261021, 8, cell_idx, rep])
    data_seed = seed.spawn(1)[0]
    rng = np.random.default_rng(data_seed)
    dataset = generate_dataset(cell, rng)

    x_full = dataset["x"]
    y_full = dataset["y"]
    sw_full = dataset["sample_weight"]  # observation weight for training

    rows = []
    for win_idx, (w_start, w_end) in enumerate(windows):
        holdout_idx = np.arange(w_start, w_end)
        n_holdout = len(holdout_idx)
        n_training = n - n_holdout

        # Observation mask: True = training, False = holdout.
        obs_mask = np.ones(n, dtype=bool)
        obs_mask[holdout_idx] = False

        # Seed for randomized PIT RNG.
        pit_seed = int(np.random.SeedSequence(
            [20261021, 8, cell_idx, rep, win_idx]
        ).generate_state(1)[0])

        try:
            # --- Fitted model ---
            est = _make_estimator(family, config, dataset)
            est.fit(x_full.reshape(-1, 1), y_full, sample_weight=sw_full,
                    observation_mask=obs_mask)
            k_map = int(est.k_map_)
            p_k_true = float(est.k_posterior_[config["k_true"] - 1])
            scores = _score_window(est, dataset, family, holdout_idx, pit_seed)

            # --- Null model (k_max=1, forced one-segment) ---
            est_null = _make_estimator(family, config, dataset, k_max=1)
            est_null.fit(x_full.reshape(-1, 1), y_full, sample_weight=sw_full,
                         observation_mask=obs_mask)
            null_scores = _score_window(est_null, dataset, family, holdout_idx, pit_seed + 1)

            row = {
                "cell_index": cell_idx,
                "family": family,
                "param": cell["param"],
                "value": cell["value"],
                "n": n,
                "replicate": rep,
                "window_index": win_idx,
                "window_start": w_start,
                "window_end": w_end,
                "n_holdout": n_holdout,
                "n_training": n_training,
                "k_map": k_map,
                "p_k_true": p_k_true,
                "mean_logpd": scores["mean_logpd"],
                "coverage_95": scores["coverage_95"],
                "pit_ks_stat": scores["pit_ks_stat"],
                "pit_ks_pval": scores["pit_ks_pval"],
                "null_mean_logpd": null_scores["mean_logpd"],
                "null_coverage_95": null_scores["coverage_95"],
                "null_pit_ks_stat": null_scores["pit_ks_stat"],
                "null_pit_ks_pval": null_scores["pit_ks_pval"],
                "logpd_diff": scores["mean_logpd"] - null_scores["mean_logpd"],
                "failed": False,
                "error": "",
            }
        except Exception as exc:
            log.exception("cell=%d rep=%d win=%d failed: %s", cell_idx, rep, win_idx, exc)
            row = {
                "cell_index": cell_idx, "family": family, "param": cell["param"],
                "value": cell["value"], "n": n, "replicate": rep,
                "window_index": win_idx, "window_start": w_start, "window_end": w_end,
                "n_holdout": n_holdout, "n_training": n_training,
                "k_map": -1, "p_k_true": float("nan"),
                "mean_logpd": float("nan"), "coverage_95": float("nan"),
                "pit_ks_stat": float("nan"), "pit_ks_pval": float("nan"),
                "null_mean_logpd": float("nan"), "null_coverage_95": float("nan"),
                "null_pit_ks_stat": float("nan"), "null_pit_ks_pval": float("nan"),
                "logpd_diff": float("nan"),
                "failed": True,
                "error": str(exc),
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Bootstrap summary
# ---------------------------------------------------------------------------

def _bootstrap_cell_summary(rows: list[dict], config: dict) -> dict:
    """Compute per-cell mean and bootstrap 95% CI for key metrics."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df[~df["failed"]]
    if df.empty:
        return {}

    B = config["bootstrap_resamples"]
    rng = np.random.default_rng(20261021)
    reps = df["replicate"].unique()
    n_reps = len(reps)

    summaries = {}
    for col in ["mean_logpd", "null_mean_logpd", "logpd_diff",
                "coverage_95", "null_coverage_95",
                "pit_ks_stat", "null_pit_ks_stat"]:
        if col not in df.columns:
            continue
        # Per-replicate mean (averaging over windows).
        rep_means = df.groupby("replicate")[col].mean().values
        if len(rep_means) == 0:
            continue
        point = float(np.mean(rep_means))
        boot = np.array([
            np.mean(rng.choice(rep_means, size=n_reps, replace=True))
            for _ in range(B)
        ])
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        summaries[col] = {"mean": point, "ci95_lo": lo, "ci95_hi": hi}

    return summaries


# ---------------------------------------------------------------------------
# Environment recording
# ---------------------------------------------------------------------------

def _record_environment() -> dict:
    def _git_rev() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            return "unknown"

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_revision": _git_rev(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run E08 posterior prediction experiment.")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode: 5 reps, 2 windows.")
    args = parser.parse_args()

    pilot = args.pilot
    n_reps_override = 5 if pilot else None
    windows_per_n: dict[int, list[tuple[int, int]]] = {}
    for cell in E08_CONFIG["cells"]:
        n = cell["n"]
        if n not in windows_per_n:
            fracs = E08_CONFIG["holdout_fraction_intervals"]
            if pilot:
                fracs = fracs[:2]  # first 2 windows only for pilot
            windows_per_n[n] = _holdout_windows(n, fracs)

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_BASE / "run_log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("E08 starting. pilot=%s config_sha256=%s", pilot, E08_CONFIG_SHA256)
    log.info("Results: %s", RESULTS_BASE)

    env = _record_environment()
    t_start = time.time()
    utc_start = datetime.now(timezone.utc).isoformat()

    all_rows: list[dict] = []
    failed_rows: list[dict] = []

    for cell in E08_CONFIG["cells"]:
        cell_idx = cell["cell_index"]
        family = cell["family"]
        n_reps = n_reps_override if n_reps_override else cell["n_reps"]
        windows = windows_per_n[cell["n"]]
        log.info("Cell %d (%s %s=%.2f) n_reps=%d n_windows=%d",
                 cell_idx, family, cell["param"], cell["value"], n_reps, len(windows))

        for rep in range(n_reps):
            rows = run_replicate(cell, rep, windows, E08_CONFIG)
            all_rows.extend(rows)
            failed = [r for r in rows if r["failed"]]
            failed_rows.extend(failed)
            if rep % 10 == 0:
                log.info("  cell=%d rep=%d/%d done", cell_idx, rep + 1, n_reps)

    t_elapsed = time.time() - t_start
    utc_end = datetime.now(timezone.utc).isoformat()
    log.info("All cells done. Elapsed %.1fs. Total rows=%d failed=%d",
             t_elapsed, len(all_rows), len(failed_rows))

    # Write CSV.
    csv_path = RESULTS_BASE / "metrics.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    log.info("Wrote %s", csv_path)

    # Per-cell summaries.
    import pandas as pd
    df_all = pd.DataFrame(all_rows)
    cell_summaries = {}
    for cell in E08_CONFIG["cells"]:
        crows = [r for r in all_rows if r["cell_index"] == cell["cell_index"] and not r["failed"]]
        cell_summaries[cell["cell_index"]] = _bootstrap_cell_summary(crows, E08_CONFIG)

    # Aggregate metrics JSON.
    metrics = {
        "n_total": len(all_rows),
        "n_failed": len(failed_rows),
        "elapsed_seconds": t_elapsed,
        "pilot": pilot,
        "cell_summaries": cell_summaries,
    }
    metrics_path = RESULTS_BASE / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote %s", metrics_path)

    # Failed runs.
    if failed_rows:
        fail_path = RESULTS_BASE / "failed_runs.csv"
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failed_rows)
        log.warning("Wrote failed_runs.csv: %d failures", len(failed_rows))

    # Manifest.
    manifest = {
        "id": E08_CONFIG["id"],
        "protocol_revision": E08_CONFIG["protocol_revision"],
        "config_sha256": E08_CONFIG_SHA256,
        "git_revision": env["git_revision"],
        "utc_start": utc_start,
        "utc_end": utc_end,
        "pilot": pilot,
        "n_cells": len(E08_CONFIG["cells"]),
        "n_total_rows": len(all_rows),
        "n_failed": len(failed_rows),
        "holdout_windows": {str(n): ws for n, ws in windows_per_n.items()},
        "environment": env,
        "output_files": {
            "metrics_csv": str(csv_path.relative_to(RESULTS_BASE)),
            "metrics_json": str(metrics_path.relative_to(RESULTS_BASE)),
        },
        "gate_pass": len(failed_rows) == 0,
    }
    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: gate_pass=%s", manifest["gate_pass"])

    if len(failed_rows) > 0:
        log.error("E08 GATE FAIL: %d failed rows.", len(failed_rows))
        sys.exit(1)
    log.info("E08 complete. gate_pass=True")


if __name__ == "__main__":
    main()
