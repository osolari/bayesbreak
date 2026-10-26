"""Diagnostic E06: downstream segmentation effects of approximate block evidence tables.

For n=96, k_true=3, boundaries at (32,64), theta levels logit((0.2, 0.8, 0.2)),
generates 5 independent Bernoulli sequences and evaluates the effect of each
approximation method on downstream segmentation (p(k|y), boundary marginals,
MAP partition, F1@3).

Compares each method's approximate block-evidence table to the quadrature reference.

Usage:
    uv run python scripts/diag_e06_segmentation.py
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("diag_e06_seg")


def _logit(p: float) -> float:
    return math.log(p) - math.log(1.0 - p)


def _f1_at_tau(pred_boundaries: list[int], true_boundaries: list[int], tau: int = 3) -> float:
    """One-to-one maximum-cardinality boundary F1 at tolerance tau."""
    if not true_boundaries:
        return 1.0 if not pred_boundaries else 0.0
    if not pred_boundaries:
        return 0.0

    # Greedy one-to-one matching within tolerance.
    matched_pred = set()
    matched_true = set()
    for tb in true_boundaries:
        best = None
        best_dist = tau + 1
        for idx, pb in enumerate(pred_boundaries):
            if idx in matched_pred:
                continue
            d = abs(pb - tb)
            if d <= tau and d < best_dist:
                best = idx
                best_dist = d
        if best is not None:
            matched_pred.add(best)
            matched_true.add(tb)

    tp = len(matched_pred)
    prec = tp / len(pred_boundaries)
    rec = tp / len(true_boundaries)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def run_segmentation_diagnostic() -> None:
    from bayesbreak.families.logistic_normal import BayesBreakLogisticNormal

    # E06 segmentation propagation setup.
    n = 96
    k_true = 3
    boundaries = [32, 64]  # interior boundaries
    theta_levels = [_logit(0.2), _logit(0.8), _logit(0.2)]
    n_seqs = 5
    k_max = 8
    nu, rho2 = 0.0, 4.0

    logger.info("E06 segmentation propagation diagnostic")
    logger.info("n=%d, k_true=%d, boundaries=%s", n, k_true, boundaries)
    logger.info("theta levels: logit(0.2)=%.4f, logit(0.8)=%.4f", _logit(0.2), _logit(0.8))
    logger.info("Methods: quadrature (reference), Laplace, JJ, EP")
    logger.info("pg_vb = JJ alias, not shown separately")

    # Generate 5 independent Bernoulli sequences.
    seqs: list[np.ndarray] = []
    for rep in range(n_seqs):
        ss = np.random.SeedSequence([20261021, 6, 9998, rep])
        rng = np.random.default_rng(ss)
        y = np.zeros(n, dtype=float)
        for seg_idx, (start, stop) in enumerate(
            zip([0] + boundaries, boundaries + [n])
        ):
            p_seg = 1.0 / (1.0 + math.exp(-theta_levels[seg_idx]))
            y[start:stop] = rng.binomial(1, p_seg, size=stop - start).astype(float)
        seqs.append(y)

    X_design = np.arange(n, dtype=float).reshape(-1, 1)
    methods = ["quadrature", "laplace", "jj", "ep"]

    # For each method and sequence, fit and collect results.
    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for rep, y in enumerate(seqs):
        logger.info("Sequence %d/%d", rep + 1, n_seqs)
        for method in methods:
            est = BayesBreakLogisticNormal(
                k_max=k_max,
                approx=method,
                nu=nu,
                rho2=rho2,
                estimate_hyper=False,
                max_iter=100,
                ep_tol_kl=1e-7,
                ep_damping=0.7,
            )
            est.fit(X_design, y)

            # p(k|y) for k=1..k_max
            pk = np.asarray(est.k_posterior_, dtype=float)  # shape (k_max,)
            # boundary marginals: p(b_i=1|y) for i in 1..n-1
            bm = np.asarray(est.boundary_marginals_, dtype=float)
            # MAP partition boundaries
            map_boundaries = list(est.map_boundaries_) if hasattr(est, 'map_boundaries_') else []
            # F1@3
            f1 = _f1_at_tau(map_boundaries, boundaries, tau=3)

            all_results[method].append({
                "pk": pk,
                "bm": bm,
                "map_boundaries": map_boundaries,
                "f1": f1,
            })

    # Reference is quadrature.
    ref_results = all_results["quadrature"]

    logger.info("")
    logger.info("--- Per-method comparison to quadrature reference ---")
    for method in ["laplace", "jj", "ep"]:
        max_pk_diff = []
        max_bm_diff = []
        map_mismatches = 0
        f1_scores = []
        ref_f1_scores = []

        for rep in range(n_seqs):
            ref = ref_results[rep]
            approx = all_results[method][rep]

            # p(k|y) comparison
            pk_ref = np.array(ref["pk"])
            pk_approx = np.array(approx["pk"])
            n_common = min(len(pk_ref), len(pk_approx))
            max_pk_diff.append(float(np.max(np.abs(pk_ref[:n_common] - pk_approx[:n_common]))))

            # Boundary marginals comparison
            bm_ref = np.array(ref["bm"])
            bm_approx = np.array(approx["bm"])
            n_bm = min(len(bm_ref), len(bm_approx))
            max_bm_diff.append(float(np.max(np.abs(bm_ref[:n_bm] - bm_approx[:n_bm]))))

            # MAP boundary mismatch
            if sorted(approx["map_boundaries"]) != sorted(ref["map_boundaries"]):
                map_mismatches += 1

            f1_scores.append(approx["f1"])
            ref_f1_scores.append(ref["f1"])

        logger.info(
            "Method: %-10s | max|pk_diff|=%6.4f (mean=%6.4f) | "
            "max|bm_diff|=%6.4f (mean=%6.4f) | "
            "MAP mismatches=%d/%d | mean F1@3=%.3f (ref=%.3f)",
            method.upper(),
            max(max_pk_diff), np.mean(max_pk_diff),
            max(max_bm_diff), np.mean(max_bm_diff),
            map_mismatches, n_seqs,
            np.mean(f1_scores), np.mean(ref_f1_scores),
        )

    logger.info("")
    logger.info("--- Per-sequence MAP boundaries ---")
    header_m = " ".join(f"{m[:4]:>12}" for m in methods)
    logger.info("Rep | True boundaries | %s", header_m)
    for rep in range(n_seqs):
        parts = []
        for m in methods:
            b = sorted(all_results[m][rep]["map_boundaries"])
            parts.append(f"{str(b):>12}")
        logger.info(
            " %2d | %-15s | %s",
            rep + 1, str(boundaries), " ".join(parts),
        )

    logger.info("")
    logger.info("--- F1@3 by method and sequence ---")
    for rep in range(n_seqs):
        row = " ".join(
            f"{all_results[m][rep]['f1']:.3f}" for m in methods
        )
        logger.info("Rep %d: %s  [methods: %s]", rep + 1, row, ", ".join(methods))

    logger.info("")
    logger.info("--- Quadrature reference results ---")
    logger.info("True boundaries: %s", boundaries)
    for rep in range(n_seqs):
        r = ref_results[rep]
        pk = r["pk"]
        pk_str = " ".join(f"k={k+1}:{v:.3f}" for k, v in enumerate(pk[:5]))
        logger.info("Rep %d: MAP=%s, F1=%.3f, p(k)=[%s]", rep + 1, r["map_boundaries"], r["f1"], pk_str)

    logger.info("")
    logger.info("DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    run_segmentation_diagnostic()
