"""E03: irregular physical grid and prior ablation.

Protocol : .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E03
Config   : .claude/skills/bayesbreak-release-audit/protocols/experiments.json

Design (from protocol):
  n=240, x_1=1, successive gaps=1 except gap between x_120 and x_121 equals G in {1,4,16}.
  Interior candidate edge u_i = (x_i + x_{i+1})/2 for i<n, u_0=x_1-0.5, u_n=x_n+0.5.
  True templates (60,120) and (60,180). Gaussian sigma2=1, block means (0,1.5,0),
  Normal(0,4) prior, kmax=8, 100 datasets per (G,truth) cell.
  4 prior settings × 6 cells = 24 cell-prior combinations.

Produces:
  results/confirmatory/E03_irregular_design/<config_sha256[:16]>/
    manifest.json, metrics.json, metrics.csv, oracle_checks.json,
    figures/fig_E03_gap_true.pdf/.png, fig_E03_gap_false.pdf/.png,
    tables/tab_E03_prior_comparison.tex,
    run_log.txt, failed_runs.csv

Usage (confirmatory):
    uv run python scripts/run_e03_irregular_design.py

Usage (diagnostic pilot, 5 datasets per cell):
    uv run python scripts/run_e03_irregular_design.py --pilot
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
# E03 configuration (frozen before first confirmatory run)
# ---------------------------------------------------------------------------

E03_CONFIG: dict[str, Any] = {
    "id": "E03",
    "name": "irregular_design_prior",
    "protocol_revision": "2026-10-21-2",
    "type": "simulation",
    "n": 240,
    "gap_start_idx": 120,   # x_120 and x_121 have a variable gap
    "grid_gaps": [1, 4, 16],
    "true_templates": [[60, 120], [60, 180]],
    "k_true": 3,
    "gaussian_sigma2": 1.0,
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0},
    "block_means": [0.0, 1.5, 0.0],
    "k_max": 8,
    "replicates": 100,
    "priors": ["uniform", "gap_hazard", "length_cohesion", "combined"],
    "seed_base": [20261021, 3],
    "prior": "geometric_decay_log4",
    "bootstrap_resamples": 2000,
    "boundary_tolerances": [0, 1, 3, 5],
    "boundary_tolerance_primary": 3,
    "gap_index_boundary": 120,  # the boundary index at the large gap
    "small_positive_constant": 1e-6,
}

E03_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E03_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E03_irregular_design"
    / E03_CONFIG_SHA256[:16]
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
# Section 2: Grid construction
# ---------------------------------------------------------------------------

def _build_observation_coords(n: int, gap: int, gap_start_idx: int) -> np.ndarray:
    """Build x_1, ..., x_n with a large gap between x_{gap_start_idx} and x_{gap_start_idx+1}.

    Protocol: x_1=1, successive gaps=1 except the gap between
    x_{gap_start_idx} and x_{gap_start_idx+1} equals G.
    Returns 0-indexed array x[0..n-1] where x[0]=1.
    """
    x = np.empty(n, dtype=float)
    x[0] = 1.0
    for i in range(1, n):
        if i == gap_start_idx:
            # gap between x[i-1] and x[i] is G (gap_start_idx is 0-indexed, so gap at position i)
            x[i] = x[i - 1] + gap
        else:
            x[i] = x[i - 1] + 1.0
    return x


def _build_boundary_coords(x: np.ndarray) -> np.ndarray:
    """Build n+1 candidate boundary coordinates from n observation coordinates.

    Protocol: u_i = (x_i + x_{i+1})/2 for 1 <= i < n (midpoints),
    u_0 = x_1 - 0.5, u_n = x_n + 0.5.
    Returns 0-indexed array u[0..n].
    """
    n = len(x)
    u = np.empty(n + 1, dtype=float)
    u[0] = x[0] - 0.5
    for i in range(1, n):
        u[i] = 0.5 * (x[i - 1] + x[i])
    u[n] = x[n - 1] + 0.5
    return u


# ---------------------------------------------------------------------------
# Section 3: Prior configuration for E03
# ---------------------------------------------------------------------------

def _build_prior_config(
    prior_name: str,
    x: np.ndarray,
    u: np.ndarray,
) -> Any:
    """Build PartitionPriorConfig for each prior setting.

    A) uniform: all local factors = 1
    B) gap_hazard: h_u(i) = max(x_{i+1} - x_i, small_pos) for each interior i
       (where i indexes into x, so the gap is x[i] - x[i-1] for i=1..n-1)
    C) length_cohesion: c_u(i,j) = (u_j - u_i) / median_gap, h=1
    D) combined: product of B and C
    """
    from bayesbreak.priors import PartitionPriorConfig

    n = len(x)
    small_pos = float(E03_CONFIG["small_positive_constant"])

    # Compute median gap between consecutive boundary coords
    u_gaps = np.diff(u)  # u[1]-u[0], ..., u[n]-u[n-1]
    median_gap = float(np.median(u_gaps))

    # For prior B, the hazard at interior boundary i (between obs i and i+1, 1-indexed)
    # equals max(x[i] - x[i-1], small_pos) for i=1..n-1 in 0-indexed x.
    # In the prior table, boundary index i means a cut is placed between obs[i-1] and obs[i]
    # (the segment [0..i) and [i..n)).
    # The protocol says h_u(i) = max(x_{i+1} - x_i, small_positive_constant) for i<n.
    # In 1-indexed x: h_u(i) = max(x_{i+1} - x_i, eps) for i = 1..n-1.
    # In 0-indexed x: gap at position i (boundary between x[i-1] and x[i]) = x[i] - x[i-1].
    # So hazard at boundary index b (1-indexed, b=1..n-1) = max(x[b] - x[b-1], eps).
    # But the protocol uses 1-indexed x: h_u(i) = max(x_{i+1} - x_i, eps).
    # In 0-indexed: h_u(b) = max(x[b] - x[b-1], eps) for b=1..n-1.
    # This is the gap to the right of x[b-1] (i.e., from x[b-1] to x[b]).
    hazard_params = {
        f"hazard_{b}": float(max(x[b] - x[b - 1], small_pos))
        for b in range(1, n)
    }

    if prior_name == "uniform":
        return PartitionPriorConfig(
            segment_cohesion="uniform",
            boundary_hazard="uniform",
        )
    elif prior_name == "gap_hazard":
        return PartitionPriorConfig(
            segment_cohesion="uniform",
            boundary_hazard="explicit",
            parameters=hazard_params,
        )
    elif prior_name == "length_cohesion":
        return PartitionPriorConfig(
            segment_cohesion="span",
            boundary_hazard="uniform",
            parameters={"cohesion_scale": median_gap, "cohesion_power": 1.0},
        )
    elif prior_name == "combined":
        params = dict(hazard_params)
        params["cohesion_scale"] = median_gap
        params["cohesion_power"] = 1.0
        return PartitionPriorConfig(
            segment_cohesion="span",
            boundary_hazard="explicit",
            parameters=params,
        )
    else:
        raise ValueError(f"Unknown prior_name: {prior_name!r}")


# ---------------------------------------------------------------------------
# Section 4: Data generation
# ---------------------------------------------------------------------------

def _generate_dataset(
    n: int,
    gap: int,
    gap_start_idx: int,
    true_template: list[int],
    block_means: list[float],
    sigma2: float,
    rng: np.random.Generator,
) -> dict:
    """Generate one Gaussian segmentation dataset.

    Protocol: k_true=3, block means (0,1.5,0). Boundaries at positions given
    by true_template (e.g., [60, 120] means boundaries at index 60 and 120,
    so segments [0..60), [60..120), [120..240)).
    """
    x = _build_observation_coords(n, gap, gap_start_idx)
    u = _build_boundary_coords(x)

    b1, b2 = true_template
    k_true = 3
    true_signal = np.empty(n, dtype=float)
    true_signal[:b1] = block_means[0]
    true_signal[b1:b2] = block_means[1]
    true_signal[b2:] = block_means[2]

    sigma = math.sqrt(sigma2)
    y = true_signal + rng.standard_normal(n) * sigma

    return {
        "y": y,
        "x": x,
        "u": u,
        "true_signal": true_signal,
        "boundaries": [0, b1, b2, n],
        "true_boundaries_interior": [b1, b2],
        "gap": gap,
        "gap_start_idx": gap_start_idx,
        "n": n,
        "k_true": k_true,
    }


# ---------------------------------------------------------------------------
# Section 5: Inference
# ---------------------------------------------------------------------------

def _run_inference(
    dataset: dict,
    prior_name: str,
    prior_config: Any,
) -> dict:
    """Fit BayesBreakGaussian with the specified prior configuration."""
    from bayesbreak.families.gaussian import BayesBreakGaussian

    y = dataset["y"]
    x = dataset["x"]
    u = dataset["u"]
    n = dataset["n"]

    est = BayesBreakGaussian(
        k_max=E03_CONFIG["k_max"],
        estimate_hyper=False,
        nu=E03_CONFIG["gaussian_prior"]["nu"],
        rho2=E03_CONFIG["gaussian_prior"]["rho2"],
        sigma2=E03_CONFIG["gaussian_sigma2"],
        partition_prior=prior_config,
        boundary_coordinates=u,
        prior_k=_geom_prior,
    )
    est.fit(x.reshape(-1, 1), y)

    from bayesbreak.dp import boundary_event_marginals_marginalised
    # Boundary marginals marginalized over k
    bm_margk = boundary_event_marginals_marginalised(
        est.log_left_, est.log_right_, est.log_posterior_k_, n, E03_CONFIG["k_max"]
    )
    # bm_margk[i-1] = P(b_i=1|y) for i=1..n-1 (0-indexed: bm_margk[i] = P(b_{i+1}=1|y))

    # p(b_120=1|y): boundary at position 120 (1-indexed in x, so interior index 120 in dp)
    # bm_margk has length n-1, indexed 0..n-2 where bm_margk[i] = P(b_{i+1}=1|y)
    # Position 120 in the protocol is the 120th observation boundary (between obs 120 and 121)
    # In 0-indexed terms: boundary at position 120 means split between x[119] and x[120],
    # which is interior index i=120 in the DP (bm_margk[119]).
    gap_boundary_idx = E03_CONFIG["gap_index_boundary"]  # = 120
    p_b120 = float(bm_margk[gap_boundary_idx - 1])  # bm_margk[119] = P(b_120=1|y)

    return {
        "est": est,
        "bm_margk": bm_margk,
        "p_b_gap": p_b120,
        "k_map": est.k_map_,
        "map_boundaries": est.map_boundaries_,
        "k_posterior": est.k_posterior_,
        "log_evidence": est.log_evidence_,
        "map_curve": est.map_curve_,
    }


# ---------------------------------------------------------------------------
# Section 6: Boundary matching
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
    # Greedy min-cost matching (Hungarian is overkill for small k)
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


def _compute_f1(
    bhat: list[int],
    bstar: list[int],
    tau: int,
) -> tuple[float, float, float]:
    """Return precision, recall, F1 for boundary detection at tolerance tau."""
    if not bstar and not bhat:
        return 1.0, 1.0, 1.0
    if not bstar:
        return 0.0, 1.0, 0.0
    if not bhat:
        return 1.0, 0.0, 0.0
    n_matched, _ = _boundary_match(bhat, bstar, tau)
    precision = n_matched / len(bhat) if bhat else 0.0
    recall = n_matched / len(bstar) if bstar else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Section 7: Metrics computation
# ---------------------------------------------------------------------------

def _compute_replicate_metrics(
    dataset: dict,
    infer: dict,
    prior_name: str,
) -> dict:
    """Compute all prescribed metrics for one replicate."""
    taus = E03_CONFIG["boundary_tolerances"]
    tau_primary = E03_CONFIG["boundary_tolerance_primary"]

    n = dataset["n"]
    true_bint = dataset["true_boundaries_interior"]  # e.g., [60, 120] or [60, 180]
    gap = dataset["gap"]
    gap_idx = E03_CONFIG["gap_index_boundary"]  # 120

    # MAP boundaries (interior only)
    map_bint = [b for b in infer["map_boundaries"][1:-1]]

    # F1 at each tolerance
    f1_tol = {}
    prec_tol = {}
    rec_tol = {}
    for tau in taus:
        prec, rec, f1 = _compute_f1(map_bint, true_bint, tau)
        f1_tol[tau] = f1
        prec_tol[tau] = prec
        rec_tol[tau] = rec

    f1_primary = f1_tol[tau_primary]
    prec_primary = prec_tol[tau_primary]
    rec_primary = rec_tol[tau_primary]

    # MAE
    n_matched, total_disp = _boundary_match(map_bint, true_bint, tau_primary)
    mae = total_disp / n_matched if n_matched > 0 else float("nan")

    # Count metrics
    k_true = dataset["k_true"]
    k_hat = infer["k_map"]
    count_exact = int(k_hat == k_true)
    count_abs_error = abs(k_hat - k_true)
    p_k_true = float(infer["k_posterior"][k_true - 1])
    p_k_max = float(infer["k_posterior"][E03_CONFIG["k_max"] - 1])

    # Signal MSE (Gaussian native scale)
    signal_mse = float(np.mean((infer["map_curve"] - dataset["true_signal"]) ** 2))

    # Brier score for boundary events (marginalized boundary posteriors)
    bm = infer["bm_margk"]  # length n-1
    true_b_vec = np.zeros(n - 1, dtype=float)
    for b in true_bint:
        if 1 <= b <= n - 1:
            true_b_vec[b - 1] = 1.0
    brier = float(np.mean((bm - true_b_vec) ** 2))

    # p(b_gap=1|y): posterior probability of boundary at the gap position
    p_b_gap = infer["p_b_gap"]

    return {
        "prior": prior_name,
        "gap": gap,
        "true_template": str(dataset["true_boundaries_interior"]),
        "f1": f1_primary,
        "precision": prec_primary,
        "recall": rec_primary,
        "mae": mae,
        "f1_at_0": f1_tol.get(0, float("nan")),
        "f1_at_1": f1_tol.get(1, float("nan")),
        "f1_at_5": f1_tol.get(5, float("nan")),
        "count_exact": count_exact,
        "count_abs_error": count_abs_error,
        "p_k_true": p_k_true,
        "p_k_max": p_k_max,
        "signal_mse": signal_mse,
        "brier": brier,
        "p_b_gap": p_b_gap,
        "k_map": k_hat,
    }


# ---------------------------------------------------------------------------
# Section 8: Bootstrapped cell-level aggregation
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% percentile bootstrap CI."""
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


def _aggregate_cell(
    rep_metrics: list[dict],
    cell: dict,
    boot_rng: np.random.Generator,
    n_boot: int,
) -> dict:
    """Aggregate replicate metrics to cell summary with bootstrap CIs."""
    n_reps = len(rep_metrics)
    if n_reps == 0:
        return {}

    def _arr(key: str) -> np.ndarray:
        vals = [m[key] for m in rep_metrics if not (isinstance(m[key], float) and math.isnan(m[key]))]
        return np.array(vals, dtype=float)

    scalar_keys = [
        "f1", "precision", "recall", "mae", "f1_at_0", "f1_at_1", "f1_at_5",
        "count_exact", "count_abs_error", "p_k_true", "p_k_max",
        "signal_mse", "brier", "p_b_gap",
    ]

    summary: dict[str, Any] = {
        "cell_index": cell["cell_index"],
        "prior": cell["prior"],
        "gap": cell["gap"],
        "true_template": str(cell["true_template"]),
        "n_replicates": n_reps,
    }

    for key in scalar_keys:
        arr = _arr(key)
        mean_val = float(np.mean(arr)) if len(arr) > 0 else float("nan")
        lo, hi = _bootstrap_ci(arr, n_boot, boot_rng)
        summary[f"mean_{key}"] = mean_val
        summary[f"{key}_ci_lo"] = lo
        summary[f"{key}_ci_hi"] = hi

    summary["k_max_saturation"] = bool(summary.get("mean_p_k_max", 0) >= 0.01)

    return summary


# ---------------------------------------------------------------------------
# Section 9: LaTeX table generation
# ---------------------------------------------------------------------------

def _generate_latex_table(
    cell_summaries: list[dict],
    output_path: Path,
) -> None:
    """Generate LaTeX table comparing priors across cells."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering\small",
        r"\caption{E03 irregular-design prior ablation: mean F1@3 and P(b$_{120}$=1$|$y) per cell. " +
        r"95\% percentile bootstrap intervals (2000 resamples of independent datasets).}",
        r"\label{tab:planned-E03-prior-paper}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Gap $G$ & Template & Prior & Mean F1@3 (95\% CI) & P(b$_{120}$=1$|$y) (95\% CI) & Mean Count Exact \\",
        r"\midrule",
    ]

    prior_labels = {
        "uniform": "A: Uniform",
        "gap_hazard": "B: Gap hazard",
        "length_cohesion": "C: Length cohesion",
        "combined": "D: Combined",
    }

    # Sort by gap, then template, then prior
    prior_order = ["uniform", "gap_hazard", "length_cohesion", "combined"]
    template_order = ["[60, 120]", "[60, 180]"]
    gap_order = [1, 4, 16]

    for gap in gap_order:
        for tmpl in template_order:
            for prior in prior_order:
                for cs in cell_summaries:
                    if cs.get("gap") == gap and cs.get("true_template") == tmpl and cs.get("prior") == prior:
                        f1 = cs.get("mean_f1", float("nan"))
                        f1_lo = cs.get("f1_ci_lo", float("nan"))
                        f1_hi = cs.get("f1_ci_hi", float("nan"))
                        pb = cs.get("mean_p_b_gap", float("nan"))
                        pb_lo = cs.get("p_b_gap_ci_lo", float("nan"))
                        pb_hi = cs.get("p_b_gap_ci_hi", float("nan"))
                        ce = cs.get("mean_count_exact", float("nan"))

                        def fmt(v: float) -> str:
                            return f"{v:.3f}" if math.isfinite(v) else "---"

                        plabel = prior_labels.get(prior, prior)
                        line = (
                            f"{gap} & {tmpl} & {plabel} & "
                            f"{fmt(f1)} [{fmt(f1_lo)}, {fmt(f1_hi)}] & "
                            f"{fmt(pb)} [{fmt(pb_lo)}, {fmt(pb_hi)}] & "
                            f"{fmt(ce)} \\\\"
                        )
                        lines.append(line)
                        break

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Saved LaTeX table to %s", output_path)


# ---------------------------------------------------------------------------
# Section 10: Figure generation
# ---------------------------------------------------------------------------

def _make_figures(
    cell_summaries: list[dict],
    figures_dir: Path,
) -> None:
    """Generate fig_E03_gap_true and fig_E03_gap_false."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from _style import setup_style, PRIOR_COLORS
        setup_style()
    except ImportError:
        logger.warning("matplotlib not available; skipping figures")
        return

    priors = ["uniform", "gap_hazard", "length_cohesion", "combined"]
    prior_labels = {
        "uniform": "A: Uniform",
        "gap_hazard": "B: Gap hazard",
        "length_cohesion": "C: Length cohesion",
        "combined": "D: Combined",
    }
    colors = PRIOR_COLORS
    markers = {"uniform": "o", "gap_hazard": "s", "length_cohesion": "^", "combined": "D"}
    gaps = E03_CONFIG["grid_gaps"]
    gap_labels = [f"G={g}" for g in gaps]

    for tmpl_str, fig_suffix, tmpl_desc in [
        ("[60, 120]", "gap_true", "Template (60,120): true boundary at gap"),
        ("[60, 180]", "gap_false", "Template (60,180): no boundary at gap"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Panel 1: F1@3 by gap and prior
        ax1 = axes[0]
        ax1.set_xlabel("Gap G")
        ax1.set_ylabel(f"Mean F1@3 — {tmpl_desc}")
        x_ticks = np.arange(len(gaps))

        for prior in priors:
            y_vals = []
            y_lo = []
            y_hi = []
            for gap in gaps:
                found = None
                for cs in cell_summaries:
                    if cs.get("gap") == gap and cs.get("true_template") == tmpl_str and cs.get("prior") == prior:
                        found = cs
                        break
                if found is not None:
                    y_vals.append(found.get("mean_f1", float("nan")))
                    y_lo.append(found.get("f1_ci_lo", float("nan")))
                    y_hi.append(found.get("f1_ci_hi", float("nan")))
                else:
                    y_vals.append(float("nan"))
                    y_lo.append(float("nan"))
                    y_hi.append(float("nan"))

            y_arr = np.array(y_vals)
            y_lo_arr = np.array(y_lo)
            y_hi_arr = np.array(y_hi)
            yerr_lo = y_arr - y_lo_arr
            yerr_hi = y_hi_arr - y_arr
            valid = np.isfinite(y_arr)
            if np.any(valid):
                ax1.errorbar(
                    x_ticks[valid], y_arr[valid],
                    yerr=[yerr_lo[valid], yerr_hi[valid]],
                    label=prior_labels[prior],
                    color=colors[prior], marker=markers[prior],
                    capsize=3, linestyle="-",
                )

        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(gap_labels)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)

        # Panel 2: P(b_120=1|y) by gap and prior
        ax2 = axes[1]
        ax2.set_xlabel("Gap G")
        ax2.set_ylabel(r"Mean $P(b_{120}=1|y)$ by gap and prior")

        for prior in priors:
            y_vals = []
            y_lo = []
            y_hi = []
            for gap in gaps:
                found = None
                for cs in cell_summaries:
                    if cs.get("gap") == gap and cs.get("true_template") == tmpl_str and cs.get("prior") == prior:
                        found = cs
                        break
                if found is not None:
                    y_vals.append(found.get("mean_p_b_gap", float("nan")))
                    y_lo.append(found.get("p_b_gap_ci_lo", float("nan")))
                    y_hi.append(found.get("p_b_gap_ci_hi", float("nan")))
                else:
                    y_vals.append(float("nan"))
                    y_lo.append(float("nan"))
                    y_hi.append(float("nan"))

            y_arr = np.array(y_vals)
            y_lo_arr = np.array(y_lo)
            y_hi_arr = np.array(y_hi)
            yerr_lo = y_arr - y_lo_arr
            yerr_hi = y_hi_arr - y_arr
            valid = np.isfinite(y_arr)
            if np.any(valid):
                ax2.errorbar(
                    x_ticks[valid], y_arr[valid],
                    yerr=[yerr_lo[valid], yerr_hi[valid]],
                    label=prior_labels[prior],
                    color=colors[prior], marker=markers[prior],
                    capsize=3, linestyle="-",
                )

        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(gap_labels)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fpath = figures_dir / f"fig_E03_{fig_suffix}.{ext}"
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
        logger.info("Saved figure %s", figures_dir / f"fig_E03_{fig_suffix}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section 11: Oracle check (prior normalization verification)
# ---------------------------------------------------------------------------

def _run_prior_oracle_checks(gap: int, gap_start_idx: int) -> dict:
    """Verify that the prior-only DP C_k matches brute-force enumeration for a small n.

    Use n=10 for speed; verify all four prior configurations.
    """
    from bayesbreak.priors import PartitionPriorConfig
    from bayesbreak.design_prior import build_log_prior_table

    n_small = 10
    x_small = _build_observation_coords(n_small, gap, gap_start_idx if gap_start_idx < n_small else n_small - 1)
    u_small = _build_boundary_coords(x_small)

    results = {}
    k_max_check = 4

    for prior_name in E03_CONFIG["priors"]:
        config = _build_prior_config(prior_name, x_small, u_small)
        table = build_log_prior_table(n_small, u_small, config)

        # Brute-force C_k via enumeration
        from itertools import combinations
        bf_ck = {}
        for k in range(1, k_max_check + 1):
            # Choose k-1 interior boundaries from 1..n_small-1
            log_weights = []
            for interior in combinations(range(1, n_small), k - 1):
                partition = [0] + list(interior) + [n_small]
                lp = sum(
                    table[start, stop]
                    for start, stop in zip(partition[:-1], partition[1:])
                )
                log_weights.append(lp)
            finite_lw = [lw for lw in log_weights if math.isfinite(lw)]
            if not finite_lw:
                bf_ck[k] = -math.inf
            else:
                max_lw = max(finite_lw)
                bf_ck[k] = max_lw + math.log(sum(math.exp(lw - max_lw) for lw in finite_lw))

        # DP C_k via forward recursion
        from bayesbreak.dp import compute_log_C_k
        dp_log_Ck = compute_log_C_k(table, n_small, k_max_check)

        max_diff = 0.0
        for k in range(1, k_max_check + 1):
            bf = bf_ck.get(k, -math.inf)
            dp = float(dp_log_Ck[k]) if k <= len(dp_log_Ck) - 1 else -math.inf
            if math.isfinite(bf) and math.isfinite(dp):
                max_diff = max(max_diff, abs(bf - dp))

        results[prior_name] = {
            "max_log_C_k_deviation": float(max_diff),
            "n_k_checked": k_max_check,
        }
        logger.info(
            "  Oracle check prior=%s, gap=%d: max_log_C_k_dev=%.3e",
            prior_name, gap, max_diff,
        )

    return results


# ---------------------------------------------------------------------------
# Section 12: Main runner
# ---------------------------------------------------------------------------

def main(pilot: bool = False) -> None:
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    log_path = RESULTS_BASE / "run_log.txt"
    _setup_logging(log_path)

    mode_str = "PILOT" if pilot else "CONFIRMATORY"
    n_reps = 5 if pilot else E03_CONFIG["replicates"]
    n_boot = E03_CONFIG["bootstrap_resamples"]

    logger.info("=" * 70)
    logger.info("E03 irregular design prior — %s run", mode_str)
    logger.info("Config SHA256: %s", E03_CONFIG_SHA256[:16])
    logger.info("n_reps=%d, n_boot=%d", n_reps, n_boot)
    logger.info("=" * 70)

    # Record environment
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

    # Build cell list: (G, true_template, prior) combinations
    # 6 base cells × 4 priors = 24 combinations
    cells = []
    cell_index = 0
    for gap in E03_CONFIG["grid_gaps"]:
        for true_template in E03_CONFIG["true_templates"]:
            for prior_name in E03_CONFIG["priors"]:
                cells.append({
                    "cell_index": cell_index,
                    "gap": gap,
                    "true_template": true_template,
                    "prior": prior_name,
                    "n_reps": n_reps,
                })
                cell_index += 1

    n_cells = len(cells)
    logger.info("Total cell-prior combinations: %d", n_cells)

    cell_summaries: list[dict] = []
    all_replicate_records: list[dict] = []
    failed_runs: list[dict] = []
    oracle_results: dict[str, Any] = {}

    n_cells_completed = 0
    n_cells_failed = 0
    n_replicates_completed = 0
    n_replicates_failed = 0

    # Run oracle checks for each gap value (once per gap)
    logger.info("Running prior oracle checks...")
    gap_start_idx = E03_CONFIG["gap_start_idx"]
    for gap in E03_CONFIG["grid_gaps"]:
        logger.info("Oracle checks for gap=%d:", gap)
        oracle_results[f"gap={gap}"] = _run_prior_oracle_checks(gap, gap_start_idx)

    for cell_num, cell in enumerate(cells, start=1):
        gap = cell["gap"]
        true_template = cell["true_template"]
        prior_name = cell["prior"]
        cell_idx = cell["cell_index"]
        label = f"gap={gap},template={true_template},prior={prior_name}"

        logger.info("[Cell %d/%d] %s", cell_num, n_cells, label)
        start_wall = time.perf_counter()

        # Build coordinate arrays for this cell
        x = _build_observation_coords(
            E03_CONFIG["n"], gap, gap_start_idx
        )
        u = _build_boundary_coords(x)
        prior_config = _build_prior_config(prior_name, x, u)

        rep_metrics: list[dict] = []

        for rep in range(n_reps):
            seed_seq = np.random.SeedSequence(
                [*E03_CONFIG["seed_base"], cell_idx, rep]
            )
            data_seed = seed_seq.spawn(2)[0]
            rng = np.random.default_rng(data_seed)

            try:
                dataset = _generate_dataset(
                    n=E03_CONFIG["n"],
                    gap=gap,
                    gap_start_idx=gap_start_idx,
                    true_template=true_template,
                    block_means=E03_CONFIG["block_means"],
                    sigma2=E03_CONFIG["gaussian_sigma2"],
                    rng=rng,
                )
                infer = _run_inference(dataset, prior_name, prior_config)
                metrics = _compute_replicate_metrics(dataset, infer, prior_name)

                record = {
                    "cell_index": cell_idx,
                    "gap": gap,
                    "true_template": str(true_template),
                    "prior": prior_name,
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
                    "gap": gap,
                    "true_template": str(true_template),
                    "prior": prior_name,
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
            "  Cell done | elapsed %.1fs | n_rep=%d | mean_f1=%.3f | mean_p_b_gap=%.3f",
            elapsed,
            len(rep_metrics),
            summary.get("mean_f1", float("nan")),
            summary.get("mean_p_b_gap", float("nan")),
        )

    utc_end = datetime.now(tz=timezone.utc)

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    logger.info("Generating figures...")
    _make_figures(cell_summaries, FIGURES_DIR)

    # -----------------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------------
    logger.info("Generating LaTeX table...")
    _generate_latex_table(cell_summaries, TABLES_DIR / "tab_E03_prior_comparison.tex")

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
        logger.info("Saved metrics CSV to %s", metrics_csv_path)

    # -----------------------------------------------------------------------
    # Replicate CSV
    # -----------------------------------------------------------------------
    rep_csv_path = RESULTS_BASE / "replicates.csv"
    if all_replicate_records:
        all_rep_keys = sorted({k for r in all_replicate_records for k in r.keys()})
        with open(rep_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rep_keys)
            writer.writeheader()
            for row in all_replicate_records:
                writer.writerow({k: row.get(k, "") for k in all_rep_keys})

    # -----------------------------------------------------------------------
    # Failed runs CSV
    # -----------------------------------------------------------------------
    failed_csv_path = RESULTS_BASE / "failed_runs.csv"
    if failed_runs:
        with open(failed_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["cell_index", "gap", "true_template", "prior", "replicate", "error"])
            writer.writeheader()
            writer.writerows(failed_runs)
        logger.warning("Failed runs: %d", len(failed_runs))
    else:
        logger.info("No failed replicates.")

    # -----------------------------------------------------------------------
    # Oracle checks JSON
    # -----------------------------------------------------------------------
    oracle_checks_path = RESULTS_BASE / "oracle_checks.json"
    with open(oracle_checks_path, "w") as f:
        json.dump(oracle_results, f, indent=2)
    logger.info("Saved oracle checks to %s", oracle_checks_path)

    # -----------------------------------------------------------------------
    # Metrics JSON
    # -----------------------------------------------------------------------
    metrics = {
        "experiment_id": "E03",
        "n_cells_planned": n_cells,
        "n_cells_completed": n_cells_completed,
        "n_cells_failed": n_cells_failed,
        "n_replicates_completed": n_replicates_completed,
        "n_replicates_failed": n_replicates_failed,
        "pilot_mode": pilot,
        "gate_pass": False,  # will be updated after gate checks
        "cell_summaries": cell_summaries,
        "oracle_checks": oracle_results,
    }

    # Gate check: prior oracle max_log_C_k_deviation < 1e-8 for all prior/gap combos
    all_gate_pass = True
    gate_threshold = 1e-8
    logger.info("")
    logger.info("=" * 70)
    logger.info("--- Gate checks ---")
    for gap_key, gap_oracles in oracle_results.items():
        for prior_name, oc in gap_oracles.items():
            dev = oc.get("max_log_C_k_deviation", float("inf"))
            gate_ok = dev < gate_threshold
            status = "PASS" if gate_ok else "FAIL"
            logger.info(
                "Prior oracle check %s / %s: max_log_C_k_dev=%.2e [%s]",
                gap_key, prior_name, dev, status,
            )
            if not gate_ok:
                all_gate_pass = False

    metrics["gate_pass"] = all_gate_pass

    metrics_path = RESULTS_BASE / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    # -----------------------------------------------------------------------
    # Manifest JSON
    # -----------------------------------------------------------------------
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

    # Source file hashes
    src_dir = Path(__file__).parent.parent / "src" / "bayesbreak"
    source_file_sha256: dict[str, str] = {}
    for src_file in sorted(src_dir.rglob("*.py")):
        rel = str(src_file.relative_to(Path(__file__).parent.parent))
        with open(src_file, "rb") as f_src:
            source_file_sha256[rel] = hashlib.sha256(f_src.read()).hexdigest()

    manifest = {
        "experiment_id": "E03",
        "protocol_revision": E03_CONFIG["protocol_revision"],
        "config_sha256": E03_CONFIG_SHA256,
        "git_revision": git_rev,
        "working_tree_dirty": working_tree_dirty,
        "source_file_sha256": source_file_sha256,
        "python_version": sys.version,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
        "platform": platform.platform(),
        "seed": E03_CONFIG["seed_base"],
        "input_sha256": "deterministic_generator",
        "source_kind": "gaussian_simulation_irregular_grid",
        "family": "gaussian",
        "family_descriptor": "gaussian_known_variance",
        "prior": E03_CONFIG["prior"],
        "k_max": E03_CONFIG["k_max"],
        "k_true": E03_CONFIG["k_true"],
        "coordinate_axis": "irregular_physical_grid",
        "reference_type": "simulation_exact_inference",
        "fit_status": "completed" if n_cells_failed == 0 else "partial",
        "metric_schema": "E03_irregular_design",
        "utc_start": utc_start.isoformat(),
        "utc_end": utc_end.isoformat(),
        "pilot_mode": pilot,
        "n_cells_planned": n_cells,
        "n_cells_completed": n_cells_completed,
    }

    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest to %s", manifest_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("E03 %s complete in %.1fs", mode_str, (utc_end - utc_start).total_seconds())
    logger.info("Cells: %d planned, %d completed, %d failed", n_cells, n_cells_completed, n_cells_failed)
    logger.info("Replicates: %d completed, %d failed", n_replicates_completed, n_replicates_failed)
    logger.info("Config SHA256: %s", E03_CONFIG_SHA256[:16])
    logger.info("Results: %s", RESULTS_BASE)
    logger.info("=" * 70)

    if all_gate_pass:
        logger.info("All gate checks PASSED.")
        logger.info("E03 result: PASS")
    else:
        logger.error("Gate check FAILED — see above.")
        logger.error("E03 result: FAIL (gate check)")
        sys.exit(1)

    logger.info("gate_pass: %s", all_gate_pass)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E03 irregular design prior runner")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode (5 datasets per cell, excluded from confirmatory metrics)",
    )
    args = parser.parse_args()
    main(pilot=args.pilot)
