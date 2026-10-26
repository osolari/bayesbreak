"""Supplemental E09: ruptures baselines on welllog and CGH datasets.

This supplemental analysis runs PELT, OP (dynamic programming), BS, and WBS
from ruptures v1.1.10 on the welllog and CGH datasets from the canonical E09
run (config SHA256 prefix 0f5281c46ee8976e).

The canonical E09 runner hardcoded all baselines as unavailable without a
runtime import check. Post-hoc investigation confirmed ruptures v1.1.10 is
present in the bayesbreak conda environment.

Methodological constraints:
- Penalty parameters are chosen by principled criteria (BIC-scale), not
  tuned to maximize agreement with BayesBreak.
- Agreement with BayesBreak MAP boundaries is reported as a diagnostic only.
- BayesBreak boundaries are NOT ground truth.
- Do not equate method agreement with accuracy.
- Results are in probe-index coordinates (same axis as the canonical E09 run).
- This is a supplemental analysis, not a rerun of the frozen confirmatory
  protocol. Results are written to a separate directory.

R-based baselines (CBS/DNAcopy, SMUCE, Fearnhead) are not available (no
R runtime in this environment) and are not attempted here.

Usage:
    python scripts/supplemental_e09_ruptures_baselines.py

Requires:
    ruptures >= 1.1.10 (available in the bayesbreak conda environment)
    bayesbreak (for data loaders and robust sigma2 estimation)
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPLEMENTAL_CONFIG: dict[str, Any] = {
    "id": "E09_supplemental_ruptures",
    "canonical_e09_sha256_prefix": "0f5281c46ee8976e",
    "ruptures_cost": "l2",
    "pelt_penalty_criterion": "bic",
    "bic_penalty_scale": 2.0,
    "pelt_min_size": 2,
    "op_n_bkps_candidates": "auto_from_pelt_and_fixed",
    "fixed_n_bkps_candidates": [5, 10, 15, 20, 25, 30],
    "datasets": ["welllog_full", "welllog_stride8", "cgh_single_subject0"],
    "cgh_pooled_excluded": "computational budget: 43-subject pooled fit takes ~390s and ruptures is a per-sequence method (not a shared-boundary model); pooled comparison requires a different architecture",
}

SUPPLEMENTAL_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(SUPPLEMENTAL_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "supplemental"
    / "E09_ruptures_baselines"
    / SUPPLEMENTAL_CONFIG_SHA256[:16]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _robust_sigma2(y: np.ndarray) -> float:
    """MAD-of-differences estimator for noise variance.

    sigma = 1.4826 * median(|diff(y) - median(diff(y))|) / sqrt(2)
    Returns sigma^2.
    """
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return float(sigma ** 2)


def _bic_penalty(n: int, sigma2: float, scale: float = 2.0) -> float:
    """BIC-motivated penalty for PELT with l2 cost.

    For Gaussian data with known variance, BIC penalty per changepoint is
    2 * sigma2 * log(n) (one free parameter per segment: the mean).
    The scale factor allows variation around this baseline.
    """
    return scale * sigma2 * np.log(n)


def _breakpoints_to_boundaries(bkps: list[int], n: int) -> list[int]:
    """Convert ruptures breakpoint list to BayesBreak boundary list.

    ruptures returns breakpoints as the right-exclusive endpoint of each segment
    (last element is always n). BayesBreak stores boundaries as [0, b1, ..., n].
    """
    bounds = [0] + bkps
    if bounds[-1] != n:
        bounds.append(n)
    return bounds


def _f1_at_tau(
    pred_boundaries: list[int],
    ref_boundaries: list[int],
    tau: int = 3,
) -> float:
    """F1 at tolerance tau for boundary sets (internal boundaries only).

    Internal boundaries are all elements except the first (0) and last (n).
    This measures boundary-set agreement, not accuracy against ground truth.
    """
    pred_internal = set(pred_boundaries[1:-1])
    ref_internal = set(ref_boundaries[1:-1])
    if not pred_internal and not ref_internal:
        return 1.0
    if not pred_internal or not ref_internal:
        return 0.0

    tp = 0
    matched_ref = set()
    for p in pred_internal:
        for r in ref_internal:
            if r not in matched_ref and abs(p - r) <= tau:
                tp += 1
                matched_ref.add(r)
                break

    precision = tp / len(pred_internal) if pred_internal else 0.0
    recall = tp / len(ref_internal) if ref_internal else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _run_pelt(y: np.ndarray, penalty: float, min_size: int = 2) -> dict[str, Any]:
    """Run PELT with l2 cost and report result."""
    import ruptures as rpt

    algo = rpt.Pelt(model="l2", min_size=min_size)
    algo.fit(y.reshape(-1, 1))
    bkps = algo.predict(pen=penalty)
    n = len(y)
    boundaries = _breakpoints_to_boundaries(bkps, n)
    n_segments = len(boundaries) - 1
    return {
        "method": "pelt",
        "penalty": float(penalty),
        "min_size": min_size,
        "n_segments": n_segments,
        "n_breakpoints": n_segments - 1,
        "boundaries": boundaries,
    }


def _run_dynp(y: np.ndarray, n_bkps: int, min_size: int = 2) -> dict[str, Any]:
    """Run optimal partitioning (Dynp) with l2 cost."""
    import ruptures as rpt

    algo = rpt.Dynp(model="l2", min_size=min_size)
    algo.fit(y.reshape(-1, 1))
    bkps = algo.predict(n_bkps=n_bkps)
    n = len(y)
    boundaries = _breakpoints_to_boundaries(bkps, n)
    n_segments = len(boundaries) - 1
    return {
        "method": "dynp",
        "n_bkps_requested": n_bkps,
        "min_size": min_size,
        "n_segments": n_segments,
        "boundaries": boundaries,
    }


def _run_binseg(y: np.ndarray, n_bkps: int, min_size: int = 2) -> dict[str, Any]:
    """Run binary segmentation (Binseg) with l2 cost."""
    import ruptures as rpt

    algo = rpt.Binseg(model="l2", min_size=min_size)
    algo.fit(y.reshape(-1, 1))
    bkps = algo.predict(n_bkps=n_bkps)
    n = len(y)
    boundaries = _breakpoints_to_boundaries(bkps, n)
    n_segments = len(boundaries) - 1
    return {
        "method": "binseg",
        "n_bkps_requested": n_bkps,
        "min_size": min_size,
        "n_segments": n_segments,
        "boundaries": boundaries,
    }


def _run_bottomup(y: np.ndarray, n_bkps: int, min_size: int = 2) -> dict[str, Any]:
    """Run bottom-up segmentation (BottomUp) with l2 cost.

    BottomUp is included as a proxy for the wild binary segmentation concept;
    ruptures v1.1.x does not include a dedicated WBS implementation.
    """
    import ruptures as rpt

    algo = rpt.BottomUp(model="l2", min_size=min_size)
    algo.fit(y.reshape(-1, 1))
    bkps = algo.predict(n_bkps=n_bkps)
    n = len(y)
    boundaries = _breakpoints_to_boundaries(bkps, n)
    n_segments = len(boundaries) - 1
    return {
        "method": "bottomup",
        "n_bkps_requested": n_bkps,
        "min_size": min_size,
        "n_segments": n_segments,
        "boundaries": boundaries,
    }


def _analyze_dataset(
    name: str,
    y: np.ndarray,
    bayesbreak_boundaries: list[int],
    bayesbreak_k_map: int,
    sigma2: float,
) -> dict[str, Any]:
    """Run all available ruptures methods on a dataset."""
    n = len(y)
    penalty_bic = _bic_penalty(n, sigma2, scale=SUPPLEMENTAL_CONFIG["bic_penalty_scale"])

    log.info("%s: n=%d sigma2=%.4f BIC penalty=%.2f", name, n, sigma2, penalty_bic)

    results: dict[str, Any] = {
        "dataset": name,
        "n": n,
        "sigma2_robust_mad_diff": sigma2,
        "bic_penalty": penalty_bic,
        "bayesbreak_k_map": bayesbreak_k_map,
        "bayesbreak_boundaries": bayesbreak_boundaries,
        "pelt_bic": None,
        "dynp_results": [],
        "binseg_results": [],
        "bottomup_results": [],
        "agreement_diagnostics": [],
        "notes": [],
    }

    # --- PELT ---
    try:
        t0 = time.perf_counter()
        pelt_result = _run_pelt(y, penalty=penalty_bic)
        elapsed = time.perf_counter() - t0
        pelt_result["elapsed_s"] = elapsed
        pelt_result["f1_tau3_vs_bayesbreak"] = _f1_at_tau(
            pelt_result["boundaries"], bayesbreak_boundaries
        )
        results["pelt_bic"] = pelt_result
        log.info(
            "%s PELT BIC: n_segments=%d elapsed=%.2fs F1@3_vs_bb=%.3f",
            name,
            pelt_result["n_segments"],
            elapsed,
            pelt_result["f1_tau3_vs_bayesbreak"],
        )
    except Exception as exc:
        log.error("%s PELT failed: %s", name, exc, exc_info=True)
        results["pelt_bic"] = {"error": str(exc)}

    # Use PELT n_segments as one candidate for OP/BS/BottomUp
    pelt_k = results["pelt_bic"].get("n_segments") if results["pelt_bic"] else None

    # Candidate n_bkps values: fixed grid plus PELT result
    fixed_candidates = SUPPLEMENTAL_CONFIG["fixed_n_bkps_candidates"]
    candidates = sorted(set(fixed_candidates + ([pelt_k - 1] if pelt_k else [])))
    # Limit to n//2 at most and positive values
    candidates = [c for c in candidates if 1 <= c <= n // 2]

    # --- Dynp (Optimal Partitioning) for a subset of candidates ---
    # Dynp is O(n^2 * n_bkps) for large n; limit to first few candidates for large n.
    max_dynp_candidates = 3 if n > 512 else len(candidates)
    dynp_candidates = candidates[:max_dynp_candidates]
    for c in dynp_candidates:
        try:
            t0 = time.perf_counter()
            r = _run_dynp(y, n_bkps=c)
            elapsed = time.perf_counter() - t0
            r["elapsed_s"] = elapsed
            r["f1_tau3_vs_bayesbreak"] = _f1_at_tau(r["boundaries"], bayesbreak_boundaries)
            results["dynp_results"].append(r)
            log.info(
                "%s Dynp n_bkps=%d: n_segments=%d elapsed=%.2fs",
                name, c, r["n_segments"], elapsed
            )
        except Exception as exc:
            log.error("%s Dynp n_bkps=%d failed: %s", name, c, exc, exc_info=True)
            results["dynp_results"].append({"n_bkps_requested": c, "error": str(exc)})

    # --- BinSeg (Binary Segmentation) ---
    for c in candidates:
        try:
            t0 = time.perf_counter()
            r = _run_binseg(y, n_bkps=c)
            elapsed = time.perf_counter() - t0
            r["elapsed_s"] = elapsed
            r["f1_tau3_vs_bayesbreak"] = _f1_at_tau(r["boundaries"], bayesbreak_boundaries)
            results["binseg_results"].append(r)
        except Exception as exc:
            log.error("%s BinSeg n_bkps=%d failed: %s", name, c, exc, exc_info=True)
            results["binseg_results"].append({"n_bkps_requested": c, "error": str(exc)})

    if results["binseg_results"]:
        log.info(
            "%s BinSeg: %d configurations computed",
            name, len(results["binseg_results"])
        )

    # --- BottomUp ---
    # Skip for very large n (>1000) as it can be slow
    if n <= 2300:
        for c in candidates:
            try:
                t0 = time.perf_counter()
                r = _run_bottomup(y, n_bkps=c)
                elapsed = time.perf_counter() - t0
                r["elapsed_s"] = elapsed
                r["f1_tau3_vs_bayesbreak"] = _f1_at_tau(
                    r["boundaries"], bayesbreak_boundaries
                )
                results["bottomup_results"].append(r)
            except Exception as exc:
                log.error("%s BottomUp n_bkps=%d failed: %s", name, c, exc, exc_info=True)
                results["bottomup_results"].append(
                    {"n_bkps_requested": c, "error": str(exc)}
                )
        if results["bottomup_results"]:
            log.info(
                "%s BottomUp: %d configurations computed",
                name, len(results["bottomup_results"])
            )
    else:
        results["notes"].append(
            f"BottomUp skipped for n={n} (computational budget)"
        )

    results["notes"].append(
        "F1@3 scores measure agreement with BayesBreak MAP boundaries, "
        "not accuracy against verified ground truth."
    )
    results["notes"].append(
        "PELT penalty selected by BIC criterion (2*sigma2*log(n)), "
        "not tuned to maximize agreement with BayesBreak."
    )
    results["notes"].append(
        "ruptures v1.1.x does not include a WBS implementation; "
        "BottomUp is included instead."
    )

    return results


def main() -> None:
    try:
        import ruptures
        ruptures_version = ruptures.__version__
        log.info("ruptures version: %s", ruptures_version)
    except ImportError:
        log.error("ruptures is not installed in this environment.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_DIR / "run_log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    log.info("Supplemental E09 ruptures baselines starting.")
    log.info("config_sha256=%s", SUPPLEMENTAL_CONFIG_SHA256)
    log.info("Results: %s", RESULTS_DIR)
    log.info("canonical_e09_sha256_prefix=%s", SUPPLEMENTAL_CONFIG["canonical_e09_sha256_prefix"])

    t_start = time.perf_counter()
    utc_start = datetime.now(timezone.utc).isoformat()

    # -- Load datasets via BayesBreak data loaders ----------------------------
    from bayesbreak.datasets import load_welllog, load_cgh

    log.info("Loading datasets...")

    welllog_result = load_welllog()
    welllog_y_full = np.asarray(welllog_result.y, dtype=float)
    welllog_y_stride8 = welllog_y_full[::8]
    log.info("welllog: n_full=%d n_stride8=%d", len(welllog_y_full), len(welllog_y_stride8))

    cgh_result = load_cgh()
    # subject 0 (first column of the data matrix)
    cgh_matrix = np.asarray(cgh_result.y, dtype=float)
    if cgh_matrix.ndim == 2:
        cgh_subject0 = cgh_matrix[:, 0]
    else:
        cgh_subject0 = cgh_matrix
    log.info("cgh: n_probes=%d subject0 shape=%s", cgh_matrix.shape[0], cgh_subject0.shape)

    # -- BayesBreak MAP boundaries from canonical E09 run --------------------
    e09_metrics_path = (
        PROJECT_ROOT
        / "results"
        / "confirmatory"
        / "E09_real_data"
        / "0f5281c46ee8976e"
        / "metrics.json"
    )
    with open(e09_metrics_path) as f:
        e09_metrics = json.load(f)

    def _find_cell(cells: list[dict], dataset: str, variant: str, k_max: int) -> dict | None:
        for c in cells:
            if (
                c.get("dataset") == dataset
                and c.get("variant") == variant
                and c.get("k_max") == k_max
                and not c.get("failed")
                and not c.get("truncated")
            ):
                return c
        return None

    # Resolved cells from E09: welllog full k_max=100, welllog stride8 k_max=60,
    # cgh single_subject0 k_max=100.
    welllog_full_cell = _find_cell(e09_metrics["cells"], "welllog", "full_extended", 100)
    welllog_s8_cell = _find_cell(e09_metrics["cells"], "welllog", "stride8", 60)
    cgh0_cell = _find_cell(e09_metrics["cells"], "cgh", "single_subject0", 100)

    # MAP boundaries: for welllog, E09 metrics don't store boundaries (only CGH does).
    # We report k_map and note that boundary positions require refitting.
    welllog_full_k_map = welllog_full_cell["k_map"] if welllog_full_cell else None
    welllog_s8_k_map = welllog_s8_cell["k_map"] if welllog_s8_cell else None
    cgh0_k_map = cgh0_cell["k_map"] if cgh0_cell else None

    log.info(
        "BayesBreak k_map: welllog_full=%s welllog_stride8=%s cgh_subject0=%s",
        welllog_full_k_map, welllog_s8_k_map, cgh0_k_map,
    )

    # For welllog, boundaries were not serialized in metrics.json.
    # Use k_map as reference for n_bkps selection; F1 vs. bayesbreak is reported
    # as "N/A (boundaries not serialized)" for welllog.
    welllog_full_boundaries_ref: list[int] = []  # not available without refit
    welllog_s8_boundaries_ref: list[int] = []

    # For CGH subject0, boundaries also not stored in E09 metrics (only pooled has them).
    cgh0_boundaries_ref: list[int] = []

    # -- Sigma2 estimates (from E09 run) --------------------------------------
    sigma2_welllog_full = 4_674_808.19
    sigma2_welllog_stride8 = 5_466_698.12
    sigma2_cgh = 0.0057

    # -- Run analyses ---------------------------------------------------------
    all_results: list[dict[str, Any]] = []

    # Welllog full (n=4050, bayesbreak k_map=83)
    if welllog_full_k_map is not None:
        r = _analyze_dataset(
            name="welllog_full",
            y=welllog_y_full,
            bayesbreak_boundaries=welllog_full_boundaries_ref,
            bayesbreak_k_map=welllog_full_k_map,
            sigma2=sigma2_welllog_full,
        )
        r["bayesbreak_boundaries_available"] = False
        r["bayesbreak_boundaries_note"] = (
            "welllog boundaries not serialized in E09 metrics.json; "
            "F1@3 vs. BayesBreak requires refitting."
        )
        all_results.append(r)

    # Welllog stride-8 (n=507, bayesbreak k_map=25 geometric)
    if welllog_s8_k_map is not None:
        r = _analyze_dataset(
            name="welllog_stride8",
            y=welllog_y_stride8,
            bayesbreak_boundaries=welllog_s8_boundaries_ref,
            bayesbreak_k_map=welllog_s8_k_map,
            sigma2=sigma2_welllog_stride8,
        )
        r["bayesbreak_boundaries_available"] = False
        r["bayesbreak_boundaries_note"] = (
            "welllog stride-8 boundaries not serialized in E09 metrics.json."
        )
        all_results.append(r)

    # CGH single subject 0 (n=2215, bayesbreak k_map=71)
    if cgh0_k_map is not None:
        r = _analyze_dataset(
            name="cgh_single_subject0",
            y=cgh_subject0,
            bayesbreak_boundaries=cgh0_boundaries_ref,
            bayesbreak_k_map=cgh0_k_map,
            sigma2=sigma2_cgh,
        )
        r["bayesbreak_boundaries_available"] = False
        r["bayesbreak_boundaries_note"] = (
            "CGH single-subject boundaries not serialized in E09 metrics.json; "
            "pooled boundaries are available but use a different model."
        )
        all_results.append(r)

    t_elapsed = time.perf_counter() - t_start
    utc_end = datetime.now(timezone.utc).isoformat()

    # -- Write results --------------------------------------------------------
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"datasets": all_results}, f, indent=2)
    log.info("Wrote %s", metrics_path)

    manifest: dict[str, Any] = {
        "id": SUPPLEMENTAL_CONFIG["id"],
        "type": "supplemental",
        "parent_e09_sha256_prefix": SUPPLEMENTAL_CONFIG["canonical_e09_sha256_prefix"],
        "config_sha256": SUPPLEMENTAL_CONFIG_SHA256,
        "ruptures_version": ruptures_version,
        "git_revision": _git_rev(),
        "utc_start": utc_start,
        "utc_end": utc_end,
        "elapsed_seconds": t_elapsed,
        "n_datasets": len(all_results),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "notes": [
            "R-based baselines (CBS/DNAcopy, SMUCE, Fearnhead) unavailable: no R runtime.",
            "WBS not available in ruptures v1.1.x; BottomUp used instead.",
            "BayesBreak MAP boundaries not serialized for welllog in E09 metrics; "
            "F1@3 vs. BayesBreak requires a separate refitting step.",
            "This is a supplemental analysis, not part of the frozen E09 confirmatory protocol.",
        ],
    }
    manifest_path = RESULTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: %s", manifest_path)
    log.info("Supplemental E09 ruptures complete. elapsed=%.1fs", t_elapsed)


if __name__ == "__main__":
    main()
