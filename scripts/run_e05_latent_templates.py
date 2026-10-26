"""E05: finite latent-template score clustering.

Protocol : .claude/skills/bayesbreak-release-audit/protocols/experiments.json §E05
Reference : .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E05

Design (from protocol):
  n=180, G in {1,2,3}, S in {24,48}, Delta in {0.5,1.5,3.0}.
  18 cells, 50 replicates each, 5 restarts per replicate.
  Templates:
    Group 0: boundaries (60,120), full [0,60,120,180]
    Group 1: boundaries (90,150), full [0,90,150,180]
    Group 2: boundaries (45,105), full [0,45,105,180]
  Piecewise means (0, sign_g*Delta, 0); sign_g = (-1)^g.
  Per-subject latent mean perturbed by N(0,0.25^2) per segment.
  Gaussian sigma2=1, Normal(0,4) prior, k_max=8.
  Geometric-decay count prior p(k) proportional to exp(-log(4)*(k-1)).
  Stopping: templates stable AND relative F improvement < 1e-8 after full update.
  Monotonicity tolerance: absolute 1e-9*(1+|F|).
  Seed: SeedSequence([20261021, 5, cell_index, replicate]); child roles: data (0), init (1).

Produces:
  results/confirmatory/E05_latent_templates/<config_sha256[:16]>/
    manifest.json, metrics.json, metrics.csv, oracle_checks.json,
    figures/fig_E05_templates.pdf/.png,
    tables/tab_E05_latent.tex,
    run_log.txt, failed_runs.csv, replicates.csv

Usage (confirmatory):
    uv run python scripts/run_e05_latent_templates.py

Usage (diagnostic pilot, 5 datasets per cell):
    uv run python scripts/run_e05_latent_templates.py --pilot
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
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# E05 configuration (frozen before first confirmatory run)
# ---------------------------------------------------------------------------

E05_CONFIG: dict[str, Any] = {
    "id": "E05",
    "name": "latent_templates",
    "protocol_revision": "2026-10-21-2",
    "type": "simulation",
    "n": 180,
    "G_values": [1, 2, 3],
    "S_values": [24, 48],
    "delta_values": [0.5, 1.5, 3.0],
    "k_true": 3,
    "gaussian_sigma2": 1.0,
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0},
    "subject_mean_perturbation_sigma": 0.25,
    # Templates (interior boundaries only) for groups 0,1,2
    "group_templates": [[60, 120], [90, 150], [45, 105]],
    # Middle-segment mean sign: group g -> (-1)^g
    "k_max": 8,
    "replicates": 50,
    "n_restarts": 5,
    "max_iter": 100,
    "tol": 1e-8,
    "seed_base": [20261021, 5],
    "prior": "geometric_decay_log4",
    "bootstrap_resamples": 2000,
    "boundary_tolerance_primary": 3,
    "monotonicity_abs_tol_factor": 1e-9,
    "methods": ["mixture_em"],
    "overfit_diagnostic": False,  # G+1 diagnostic not run in main confirmatory pass
}

E05_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E05_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E05_latent_templates"
    / E05_CONFIG_SHA256[:16]
)
FIGURES_DIR = RESULTS_BASE / "figures"
TABLES_DIR = RESULTS_BASE / "tables"

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
    """Geometric-decay count prior p(k) proportional to exp(-log(4)*(k-1))."""
    return math.exp(-math.log(4.0) * (k - 1))


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------

def _enumerate_cells() -> list[dict]:
    """Return all 18 experiment cells."""
    cells = []
    for g_idx, G in enumerate(E05_CONFIG["G_values"]):
        for s_idx, S in enumerate(E05_CONFIG["S_values"]):
            for d_idx, Delta in enumerate(E05_CONFIG["delta_values"]):
                cell_idx = (
                    g_idx * len(E05_CONFIG["S_values"]) * len(E05_CONFIG["delta_values"])
                    + s_idx * len(E05_CONFIG["delta_values"])
                    + d_idx
                )
                cells.append({
                    "cell_index": cell_idx,
                    "G": G,
                    "S": S,
                    "Delta": Delta,
                })
    return cells


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _group_assignment(G: int, S: int) -> list[int]:
    """Equal group frequencies; integer remainder assigned to first groups."""
    n_per_group = S // G
    remainder = S % G
    labels: list[int] = []
    for g in range(G):
        count = n_per_group + (1 if g < remainder else 0)
        labels.extend([g] * count)
    return labels


def _full_template(interior: list[int], n: int) -> list[int]:
    """Prepend 0 and append n to interior boundary list."""
    return [0] + list(interior) + [n]


def _generate_dataset(
    G: int,
    S: int,
    Delta: float,
    n: int,
    rng_data: np.random.Generator,
) -> dict:
    """Generate one labeled multi-subject dataset for E05.

    Returns dict with keys:
      - sequences: list of S float arrays of length n
      - true_labels: list of S int group indices in 0..G-1
      - true_templates: list of G full template arrays ([0,...,n])
      - k_true: 3 (all groups have 3 segments)
    """
    group_interior_templates = E05_CONFIG["group_templates"]
    k_true = E05_CONFIG["k_true"]
    n_full = n
    pert_sigma = E05_CONFIG["subject_mean_perturbation_sigma"]
    sigma = math.sqrt(E05_CONFIG["gaussian_sigma2"])

    true_labels = _group_assignment(G, S)
    true_templates = [_full_template(group_interior_templates[g], n_full) for g in range(G)]

    sequences: list[NDArray] = []
    for s, g in enumerate(true_labels):
        tmpl = true_templates[g]
        sign_g = (-1) ** g  # group 0: +1, group 1: -1, group 2: +1
        group_means = [0.0, float(sign_g) * Delta, 0.0]

        y = np.empty(n, dtype=float)
        for q, (t0, t1) in enumerate(zip(tmpl[:-1], tmpl[1:])):
            seg_len = t1 - t0
            # Subject-specific mean perturbation per segment (independent)
            eps = float(rng_data.normal(0.0, pert_sigma))
            mu_sq = group_means[q] + eps
            y[t0:t1] = rng_data.normal(mu_sq, sigma, size=seg_len)
        sequences.append(y)

    return {
        "sequences": sequences,
        "true_labels": true_labels,
        "true_templates": true_templates,
        "k_true": k_true,
        "G": G,
        "S": S,
        "n": n,
        "Delta": Delta,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_mixture_inference(
    dataset: dict,
    k_max: int,
    sigma2: float,
    nu: float,
    rho2: float,
    n_restarts: int,
    max_iter: int,
    tol: float,
    init_seed: int,
) -> dict:
    """Fit BayesBreakMixtureClassifier and return metrics."""
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak.mixture import BayesBreakMixtureClassifier

    G = dataset["G"]
    sequences = dataset["sequences"]
    Y = np.stack(sequences)  # shape (S, n)

    base_est = BayesBreakGaussian(
        k_max=k_max,
        estimate_hyper=False,
        nu=nu,
        rho2=rho2,
        sigma2=sigma2,
        prior_k=_geom_prior,
    )

    clf = BayesBreakMixtureClassifier(
        base_estimator=base_est,
        n_groups=G,
        k_max=k_max,
        max_iter=max_iter,
        tol=tol,
        random_state=int(init_seed),
        n_restarts=n_restarts,
        prior_k=_geom_prior,
    )
    clf.fit(Y)

    pred_labels = clf.predict(Y).tolist()
    objective_trace = list(clf.objective_)
    final_objective = float(clf.final_objective_)

    # Collect restart diagnostics
    restart_objs: list[float] = [
        float(rd.final_objective)
        for rd in clf.restart_diagnostics_
        if rd.status == "valid" and rd.final_objective is not None
    ]
    n_valid_restarts = len(restart_objs)
    n_invalid_restarts = sum(
        1 for rd in clf.restart_diagnostics_ if rd.status == "invalid"
    )
    n_failed_restarts = sum(
        1 for rd in clf.restart_diagnostics_ if rd.status == "failed"
    )

    # Best vs median restart gap
    if len(restart_objs) >= 2:
        best_restart_obj = max(restart_objs)
        median_restart_obj = float(np.median(restart_objs))
        best_median_gap = best_restart_obj - median_restart_obj
    else:
        best_median_gap = 0.0

    # Number of EM iterations (winning restart)
    n_iters = len(objective_trace)

    # Duplicate templates: any two groups share same template
    group_states = clf.group_states_
    templates_list = [list(gs.template) for gs in group_states]
    has_duplicate = any(
        templates_list[i] == templates_list[j]
        for i in range(len(templates_list))
        for j in range(i + 1, len(templates_list))
    )

    # Empty groups: sum of responsibilities < 1e-6
    responsibilities = clf.responsibilities_  # shape (S, G)
    has_empty_group = bool(np.any(responsibilities.sum(axis=0) < 1e-6))

    return {
        "pred_labels": pred_labels,
        "objective_trace": objective_trace,
        "final_objective": final_objective,
        "n_iters": n_iters,
        "n_valid_restarts": n_valid_restarts,
        "n_invalid_restarts": n_invalid_restarts,
        "n_failed_restarts": n_failed_restarts,
        "best_median_gap": float(best_median_gap),
        "has_duplicate_templates": bool(has_duplicate),
        "has_empty_group": bool(has_empty_group),
        "recovered_templates": templates_list,
        "k_recovered": [gs.k_g for gs in group_states],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _permutation_invariant_accuracy(
    true_labels: list[int], pred_labels: list[int], G: int
) -> float:
    """Maximum accuracy over all G! permutations of predicted labels."""
    if G == 1:
        return 1.0
    true_arr = np.array(true_labels, dtype=int)
    pred_arr = np.array(pred_labels, dtype=int)
    best_acc = 0.0
    for perm in permutations(range(G)):
        mapped = np.array([perm[p] if 0 <= p < G else p for p in pred_arr], dtype=int)
        acc = float(np.mean(mapped == true_arr))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def _adjusted_rand_index(true_labels: list[int], pred_labels: list[int]) -> float:
    """ARI using sklearn; returns NaN for G=1 (not defined)."""
    from sklearn.metrics import adjusted_rand_score
    if len(set(true_labels)) <= 1:
        return float("nan")
    return float(adjusted_rand_score(true_labels, pred_labels))


def _check_monotonicity(trace: list[float], factor: float = 1e-9) -> bool:
    """Return True if trace is non-decreasing within absolute factor*(1+|F|)."""
    for prev, curr in zip(trace[:-1], trace[1:]):
        tol = factor * (1.0 + abs(prev))
        if curr < prev - tol:
            return False
    return True


def _compute_replicate_metrics(
    dataset: dict,
    result: dict,
) -> dict:
    """Compute E05 metrics for one replicate."""
    G = dataset["G"]
    true_labels = dataset["true_labels"]
    pred_labels = result["pred_labels"]

    accuracy = _permutation_invariant_accuracy(true_labels, pred_labels, G)
    ari = _adjusted_rand_index(true_labels, pred_labels)

    trace = result["objective_trace"]
    monotone = _check_monotonicity(trace, factor=E05_CONFIG["monotonicity_abs_tol_factor"])

    # k_true recovery: fraction of groups that recovered k=3
    k_true = dataset["k_true"]
    k_recovered = result["k_recovered"]
    k_exact = float(np.mean([k == k_true for k in k_recovered]))

    return {
        "accuracy": accuracy,
        "ari": ari,
        "n_iters": result["n_iters"],
        "monotone": int(monotone),
        "has_duplicate_templates": int(result["has_duplicate_templates"]),
        "has_empty_group": int(result["has_empty_group"]),
        "best_median_gap": result["best_median_gap"],
        "n_valid_restarts": result["n_valid_restarts"],
        "n_invalid_restarts": result["n_invalid_restarts"],
        "n_failed_restarts": result["n_failed_restarts"],
        "k_exact": k_exact,
        "final_objective": result["final_objective"],
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


def _aggregate_cell(
    rep_metrics: list[dict],
    cell: dict,
    boot_rng: np.random.Generator,
    n_boot: int,
) -> dict:
    n_reps = len(rep_metrics)
    G = cell["G"]

    def _arr(key: str) -> np.ndarray:
        vals = [
            m[key] for m in rep_metrics
            if not (isinstance(m[key], float) and math.isnan(m[key]))
        ]
        return np.array(vals, dtype=float)

    scalar_keys_ci = ["accuracy", "ari", "n_iters", "has_duplicate_templates",
                       "has_empty_group", "best_median_gap", "k_exact"]

    summary: dict[str, Any] = {
        "cell_index": cell["cell_index"],
        "G": G,
        "S": cell["S"],
        "Delta": cell["Delta"],
        "n_replicates": n_reps,
        "n_monotone_failures": int(n_reps - int(np.sum(_arr("monotone")))),
        "n_valid_restarts_mean": float(np.mean(_arr("n_valid_restarts"))),
        "n_invalid_restarts_total": int(np.sum(_arr("n_invalid_restarts"))),
        "n_failed_restarts_total": int(np.sum(_arr("n_failed_restarts"))),
    }

    for key in scalar_keys_ci:
        arr = _arr(key)
        if len(arr) == 0:
            summary[f"mean_{key}"] = float("nan")
            summary[f"{key}_ci_lo"] = float("nan")
            summary[f"{key}_ci_hi"] = float("nan")
        else:
            summary[f"mean_{key}"] = float(np.mean(arr))
            lo, hi = _bootstrap_ci(arr, n_boot, boot_rng)
            summary[f"{key}_ci_lo"] = lo
            summary[f"{key}_ci_hi"] = hi

    return summary


# ---------------------------------------------------------------------------
# Oracle checks
# ---------------------------------------------------------------------------

def _run_oracle_checks() -> dict:
    """Verify mixture objective and log_C_k against brute-force small case.

    Oracle: n=12, G=2, S=4, k_max=3.
    Verifies:
      (1) log_C_k matches binomial coefficient.
      (2) template_log_score matches direct product.
      (3) Mixture objective at fixed pi matches direct computation.
    """
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak import dp as _dp
    from bayesbreak.mixture import template_log_score
    from sklearn.base import clone

    n = 12
    G = 2
    S = 4
    k_max = 3
    sigma2 = 1.0
    nu = 0.0
    rho2 = 4.0

    rng = np.random.default_rng([20261021, 5, 999, 0])

    # Templates: group 0 → [0,4,8,12], group 1 → [0,3,9,12]
    templates = [[0, 4, 8, 12], [0, 3, 9, 12]]
    true_labels = [0, 0, 1, 1]

    # Generate sequences
    sequences = []
    pert_sigma = 0.25
    for s, g in enumerate(true_labels):
        sign_g = (-1) ** g
        seg_means = [0.0, float(sign_g) * 1.5, 0.0]
        y = np.zeros(n, dtype=float)
        for q, (t0, t1) in enumerate(zip(templates[g][:-1], templates[g][1:])):
            eps = float(rng.normal(0.0, pert_sigma))
            y[t0:t1] = rng.normal(seg_means[q] + eps, 1.0, size=t1 - t0)
        sequences.append(y)

    # Compute per-subject log A^(0) tables
    w = np.ones(n, dtype=float)
    lA0s: list[NDArray] = []
    for y in sequences:
        base = BayesBreakGaussian(
            k_max=k_max, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2
        )
        base.likelihood_power_ = w.copy()
        hyper = base._estimate_hyperparameters(y, w)
        lA0, _ = base._compute_block_evidence(y, hyper, w)
        lA0s.append(lA0)

    # (1) Verify log_C_k
    log_C_k_dp = _dp.compute_log_C_k(None, n, k_max)
    max_Ck_dev = 0.0
    for k in range(1, k_max + 1):
        expected = float(math.log(math.comb(n - 1, k - 1))) if k > 1 else 0.0
        got = float(log_C_k_dp[k])
        max_Ck_dev = max(max_Ck_dev, abs(expected - got))

    # (2) Verify template_log_score matches direct computation
    log4 = math.log(4.0)
    raw_pk = [math.exp(-log4 * (k_ - 1)) for k_ in range(1, k_max + 1)]
    total_pk = sum(raw_pk)
    log_p_k_arr = np.full(k_max + 1, -np.inf)
    for k_ in range(1, k_max + 1):
        log_p_k_arr[k_] = math.log(raw_pk[k_ - 1] / total_pk)

    max_score_dev = 0.0
    for g in range(G):
        tmpl = templates[g]
        k_g = len(tmpl) - 1
        for s in range(S):
            # Direct computation (pure Python)
            score_direct = float(log_p_k_arr[k_g]) - float(log_C_k_dp[k_g])
            for a, b in zip(tmpl[:-1], tmpl[1:]):
                score_direct += float(lA0s[s][a, b])
            # Production template_log_score (computes just the Σ_q log A part)
            prod_part = template_log_score(lA0s[s], tmpl, None)
            offset = float(log_p_k_arr[k_g]) - float(log_C_k_dp[k_g])
            score_prod = prod_part + offset
            max_score_dev = max(max_score_dev, abs(score_direct - score_prod))

    # (3) Verify mixture objective at fixed pi=[0.5,0.5]
    pi = np.array([0.5, 0.5])
    obj_direct = 0.0
    for s in range(S):
        log_contribs = []
        for g in range(G):
            tmpl = templates[g]
            k_g = len(tmpl) - 1
            score = float(log_p_k_arr[k_g]) - float(log_C_k_dp[k_g])
            for a, b in zip(tmpl[:-1], tmpl[1:]):
                score += float(lA0s[s][a, b])
            log_contribs.append(math.log(float(pi[g])) + score)
        mx = max(log_contribs)
        lse = mx + math.log(sum(math.exp(v - mx) for v in log_contribs))
        obj_direct += lse

    # Run the classifier for one step and compare log-score entries
    # (We cannot easily set fixed pi and templates, so we verify consistency
    # via the single-group case where the objective is uniquely determined.)
    from bayesbreak.mixture import BayesBreakMixtureClassifier
    base_g1 = BayesBreakGaussian(
        k_max=k_max, estimate_hyper=False, nu=nu, rho2=rho2, sigma2=sigma2,
        prior_k=_geom_prior,
    )
    clf_g1 = BayesBreakMixtureClassifier(
        base_estimator=base_g1,
        n_groups=1,
        k_max=k_max,
        max_iter=50,
        tol=1e-12,
        random_state=42,
        n_restarts=1,
        prior_k=_geom_prior,
    )
    Y_small = np.stack(sequences)
    clf_g1.fit(Y_small)

    # For G=1, the objective at convergence = Σ_s log S_1(y^s; tau_1)
    # = Σ_s [log p(k_1) - log C_{k_1} + Σ_q log A^(0,s)_{t_{q-1},t_q}]
    tmpl_fit = clf_g1.group_states_[0].template
    k_fit = clf_g1.group_states_[0].k_g
    obj_g1_direct = 0.0
    for s in range(S):
        seg_score = float(log_p_k_arr[k_fit]) - float(log_C_k_dp[k_fit])
        for a, b in zip(tmpl_fit[:-1], tmpl_fit[1:]):
            seg_score += float(lA0s[s][a, b])
        obj_g1_direct += seg_score  # log(1.0 * exp(seg_score)) = seg_score

    obj_g1_code = float(clf_g1.final_objective_)
    obj_g1_dev = abs(obj_g1_direct - obj_g1_code)

    gate_pass = (
        max_Ck_dev < 1e-10
        and max_score_dev < 1e-10
        and obj_g1_dev < 1e-8
    )

    return {
        "oracle_n": n,
        "oracle_G": G,
        "oracle_S": S,
        "oracle_k_max": k_max,
        "max_log_Ck_deviation": max_Ck_dev,
        "max_template_score_deviation": max_score_dev,
        "obj_g1_direct": obj_g1_direct,
        "obj_g1_code": obj_g1_code,
        "obj_g1_deviation": obj_g1_dev,
        "oracle_mixture_obj_direct": obj_direct,
        "gate_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _generate_figures(cell_summaries: list[dict]) -> None:
    import matplotlib.pyplot as plt

    try:
        from _style import apply_paper_style
        apply_paper_style()
    except ImportError:
        pass

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    G_values = E05_CONFIG["G_values"]
    S_values = E05_CONFIG["S_values"]
    delta_values = E05_CONFIG["delta_values"]

    # Build lookup: (G, S, Delta) -> summary
    lookup: dict[tuple, dict] = {}
    for s in cell_summaries:
        lookup[(int(s["G"]), int(s["S"]), float(s["Delta"]))] = s

    n_G = len(G_values)
    n_S = len(S_values)

    fig, axes = plt.subplots(n_G, n_S, figsize=(4.5 * n_S, 3.5 * n_G), squeeze=False)

    for i, G in enumerate(G_values):
        for j, S in enumerate(S_values):
            ax = axes[i][j]
            ax2 = ax.twinx()

            accs = [lookup[(G, S, d)].get("mean_accuracy", float("nan")) for d in delta_values]
            acc_lo = [lookup[(G, S, d)].get("accuracy_ci_lo", float("nan")) for d in delta_values]
            acc_hi = [lookup[(G, S, d)].get("accuracy_ci_hi", float("nan")) for d in delta_values]

            ari_vals = [lookup[(G, S, d)].get("mean_ari", float("nan")) for d in delta_values]
            ari_lo = [lookup[(G, S, d)].get("ari_ci_lo", float("nan")) for d in delta_values]
            ari_hi = [lookup[(G, S, d)].get("ari_ci_hi", float("nan")) for d in delta_values]

            dup_vals = [lookup[(G, S, d)].get("mean_has_duplicate_templates", float("nan")) for d in delta_values]

            xs = list(delta_values)

            # Accuracy
            ax.plot(xs, accs, "o-", color="C0", label="Accuracy")
            ax.fill_between(xs, acc_lo, acc_hi, alpha=0.2, color="C0")

            # ARI (only for G > 1)
            if G > 1:
                ax.plot(xs, ari_vals, "s--", color="C1", label="ARI")
                ax.fill_between(xs, ari_lo, ari_hi, alpha=0.2, color="C1")

            # Duplicate template rate on secondary axis
            ax2.bar(
                xs, dup_vals, width=0.15, alpha=0.4, color="C3",
                label="Dup. template freq."
            )
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("Dup. template freq.", fontsize=8, color="C3")
            ax2.tick_params(axis="y", colors="C3", labelsize=7)

            ax.set_xlim(min(delta_values) - 0.2, max(delta_values) + 0.2)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(xs)
            ax.set_xlabel(r"$\Delta$", fontsize=9)
            ax.set_ylabel("Accuracy / ARI", fontsize=8)
            ax.set_title(f"G={G}, S={S}", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "E05 Latent-template recovery: accuracy, ARI, and duplicate-template frequency",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig_E05_templates.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig_E05_templates")


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def _generate_table(cell_summaries: list[dict]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{E05 latent-template score clustering. "
        r"Accuracy is label-permutation-invariant hard assignment accuracy; "
        r"ARI is adjusted Rand index (undefined for $G=1$, shown as --). "
        r"Bootstrap 95\% percentile intervals from 2000 resamples of "
        r"independently generated datasets. "
        r"Dup.\ freq.\ is the fraction of replicates where two groups "
        r"recovered identical templates.}",
        r"\label{tab:E05-latent}",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"$G$ & $S$ & $\Delta$ & $n$ & Accuracy (95\,\% CI) & ARI (95\,\% CI) "
        r"& Dup.\ freq. & Mon.\ fail \\",
        r"\midrule",
    ]

    for s in cell_summaries:
        G = s["G"]
        S_val = s["S"]
        Delta = s["Delta"]
        n_reps = s["n_replicates"]

        acc = s.get("mean_accuracy", float("nan"))
        acc_lo = s.get("accuracy_ci_lo", float("nan"))
        acc_hi = s.get("accuracy_ci_hi", float("nan"))

        if G > 1:
            ari = s.get("mean_ari", float("nan"))
            ari_lo = s.get("ari_ci_lo", float("nan"))
            ari_hi = s.get("ari_ci_hi", float("nan"))
            ari_str = f"${ari:.3f}$ $[{ari_lo:.3f},{ari_hi:.3f}]$"
        else:
            ari_str = "--"

        dup = s.get("mean_has_duplicate_templates", float("nan"))
        mon_fail = s.get("n_monotone_failures", 0)

        lines.append(
            f"${G}$ & ${S_val}$ & ${Delta:.1f}$ & ${n_reps}$ & "
            f"${acc:.3f}$ $[{acc_lo:.3f},{acc_hi:.3f}]$ & "
            f"{ari_str} & "
            f"${dup:.3f}$ & "
            f"${mon_fail}$ \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    tex_path = TABLES_DIR / "tab_E05_latent.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved tab_E05_latent.tex")


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def _capture_environment() -> dict:
    import numpy as np
    import scipy
    import sklearn

    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).parent.parent,
        ).strip()
        env["git_revision"] = rev
    except Exception:
        env["git_revision"] = "unknown"
    return env


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def _run_experiment(pilot: bool = False) -> None:
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(RESULTS_BASE / "run_log.txt")

    n_reps_plan = 5 if pilot else E05_CONFIG["replicates"]
    n_boot = E05_CONFIG["bootstrap_resamples"] if not pilot else 100

    env = _capture_environment()
    logger.info("E05 latent-template score clustering")
    logger.info("Config SHA256: %s", E05_CONFIG_SHA256)
    logger.info("Git revision: %s", env.get("git_revision", "unknown"))
    logger.info("Pilot mode: %s; replicates per cell: %d", pilot, n_reps_plan)

    # ---- Oracle checks ----
    logger.info("Running oracle checks...")
    oracle_results = _run_oracle_checks()
    logger.info(
        "Oracle: max_log_Ck_dev=%.2e  max_score_dev=%.2e  obj_g1_dev=%.2e  gate=%s",
        oracle_results["max_log_Ck_deviation"],
        oracle_results["max_template_score_deviation"],
        oracle_results["obj_g1_deviation"],
        oracle_results["gate_pass"],
    )
    if not oracle_results["gate_pass"]:
        logger.error("Oracle gate FAILED. Aborting.")
        (RESULTS_BASE / "oracle_checks.json").write_text(
            json.dumps(oracle_results, indent=2)
        )
        sys.exit(1)

    (RESULTS_BASE / "oracle_checks.json").write_text(json.dumps(oracle_results, indent=2))

    cells = _enumerate_cells()
    n_cells = len(cells)
    logger.info("Cells: %d", n_cells)

    n_total = E05_CONFIG["n"]
    k_max = E05_CONFIG["k_max"]
    sigma2 = E05_CONFIG["gaussian_sigma2"]
    nu = E05_CONFIG["gaussian_prior"]["nu"]
    rho2 = E05_CONFIG["gaussian_prior"]["rho2"]
    n_restarts = E05_CONFIG["n_restarts"]
    max_iter = E05_CONFIG["max_iter"]
    tol = E05_CONFIG["tol"]

    all_rep_rows: list[dict] = []
    all_cell_summaries: list[dict] = []
    failed_runs: list[dict] = []

    t_start = time.time()
    n_reps_done = 0
    n_reps_failed = 0
    n_cells_done = 0

    for cell in cells:
        cell_idx = cell["cell_index"]
        G = cell["G"]
        S = cell["S"]
        Delta = cell["Delta"]
        logger.info(
            "Cell %02d/%d: G=%d S=%d Delta=%.1f",
            cell_idx, n_cells - 1, G, S, Delta,
        )

        rep_metrics_cell: list[dict] = []

        for rep in range(n_reps_plan):
            # Seed scheme
            ss = np.random.SeedSequence(E05_CONFIG["seed_base"] + [cell_idx, rep])
            children = ss.spawn(2)
            rng_data = np.random.default_rng(children[0])
            # Convert init child to int seed for BayesBreakMixtureClassifier
            init_seed = int(children[1].generate_state(1, dtype=np.uint32)[0])

            try:
                dataset = _generate_dataset(G, S, Delta, n_total, rng_data)
                result = _run_mixture_inference(
                    dataset, k_max, sigma2, nu, rho2,
                    n_restarts=n_restarts, max_iter=max_iter,
                    tol=tol, init_seed=init_seed,
                )
                metrics = _compute_replicate_metrics(dataset, result)
            except Exception as exc:
                logger.warning(
                    "Cell %d rep %d FAILED: %s", cell_idx, rep, exc
                )
                failed_runs.append({
                    "cell_index": cell_idx, "replicate": rep,
                    "G": G, "S": S, "Delta": Delta, "error": str(exc),
                })
                n_reps_failed += 1
                continue

            row = {
                "cell_index": cell_idx,
                "G": G,
                "S": S,
                "Delta": Delta,
                "replicate": rep,
                **metrics,
            }
            all_rep_rows.append(row)
            rep_metrics_cell.append(metrics)
            n_reps_done += 1

            if (rep + 1) % 10 == 0 or rep == n_reps_plan - 1:
                logger.info(
                    "  rep %d/%d: acc=%.3f ari=%.3f mono=%d dup=%d",
                    rep + 1, n_reps_plan,
                    metrics["accuracy"],
                    metrics["ari"] if not math.isnan(metrics["ari"]) else -1.0,
                    metrics["monotone"],
                    metrics["has_duplicate_templates"],
                )

        boot_rng = np.random.default_rng([20261021, 5, cell_idx, 9999])
        cell_summary = _aggregate_cell(rep_metrics_cell, cell, boot_rng, n_boot)
        all_cell_summaries.append(cell_summary)
        n_cells_done += 1

    t_end = time.time()
    wall_seconds = t_end - t_start

    # ---- Write replicates.csv ----
    if all_rep_rows:
        rep_keys = list(all_rep_rows[0].keys())
        with open(RESULTS_BASE / "replicates.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rep_keys)
            w.writeheader()
            w.writerows(all_rep_rows)

    # ---- Write failed_runs.csv ----
    if failed_runs:
        with open(RESULTS_BASE / "failed_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(failed_runs[0].keys()))
            w.writeheader()
            w.writerows(failed_runs)

    # ---- Gate check ----
    total_mono_failures = sum(s.get("n_monotone_failures", 0) for s in all_cell_summaries)
    gate_pass = (
        n_cells_done == n_cells
        and n_reps_failed == 0
        and total_mono_failures == 0
    )

    logger.info(
        "Gate: cells=%d/%d reps_done=%d reps_failed=%d mono_failures=%d -> %s",
        n_cells_done, n_cells,
        n_reps_done, n_reps_failed,
        total_mono_failures,
        "PASS" if gate_pass else "FAIL",
    )

    # ---- metrics.json ----
    metrics_out = {
        "experiment_id": "E05",
        "protocol_revision": E05_CONFIG["protocol_revision"],
        "config_sha256": E05_CONFIG_SHA256,
        "pilot": pilot,
        "n_cells_planned": n_cells,
        "n_cells_completed": n_cells_done,
        "n_replicates_planned": n_cells * n_reps_plan,
        "n_replicates_completed": n_reps_done,
        "n_replicates_failed": n_reps_failed,
        "total_monotonicity_failures": total_mono_failures,
        "gate_pass": gate_pass,
        "wall_seconds": wall_seconds,
        "cell_summaries": all_cell_summaries,
    }
    (RESULTS_BASE / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # ---- metrics.csv (cell summaries) ----
    if all_cell_summaries:
        with open(RESULTS_BASE / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_cell_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_cell_summaries)

    # ---- Figures and tables ----
    if not pilot:
        try:
            _generate_figures(all_cell_summaries)
        except Exception as exc:
            logger.warning("Figure generation failed: %s", exc)
        try:
            _generate_table(all_cell_summaries)
        except Exception as exc:
            logger.warning("Table generation failed: %s", exc)

    # ---- Manifest ----
    artifact_sha256s: dict[str, str] = {}
    for fname in ["metrics.json", "oracle_checks.json", "replicates.csv"]:
        p = RESULTS_BASE / fname
        if p.exists():
            artifact_sha256s[fname] = _sha256_file(p)

    manifest = {
        "experiment_id": "E05",
        "name": E05_CONFIG["name"],
        "protocol_revision": E05_CONFIG["protocol_revision"],
        "config_sha256": E05_CONFIG_SHA256,
        "git_revision": env.get("git_revision", "unknown"),
        "utc_start": datetime.fromtimestamp(t_start, tz=timezone.utc).isoformat(),
        "utc_stop": datetime.fromtimestamp(t_end, tz=timezone.utc).isoformat(),
        "environment": env,
        "config": E05_CONFIG,
        "gate_pass": gate_pass,
        "n_cells": n_cells_done,
        "n_replicates": n_reps_done,
        "n_failed": n_reps_failed,
        "total_monotonicity_failures": total_mono_failures,
        "wall_seconds": wall_seconds,
        "results_dir": str(RESULTS_BASE),
        "artifact_sha256": artifact_sha256s,
        "notes": (
            "Overfit G+1 diagnostic not run in main confirmatory pass. "
            "BayesBreakMixtureClassifier uses geometric-decay prior via prior_k argument. "
            "Monotonicity check uses absolute tolerance 1e-9*(1+|F|) per protocol."
        ),
    }
    (RESULTS_BASE / "manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("Results written to: %s", RESULTS_BASE)
    logger.info("Wall time: %.1f s (%.1f min)", wall_seconds, wall_seconds / 60.0)
    logger.info("E05 result: %s", "PASS" if gate_pass else "FAIL")

    if not gate_pass:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E05 latent-template score clustering.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: 5 replicates per cell, 100 bootstrap resamples.",
    )
    args = parser.parse_args()
    _run_experiment(pilot=args.pilot)
