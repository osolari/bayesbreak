"""Supplemental E09: boundary marginals for resolved configurations.

Computes exact boundary-event marginals and boundary location posteriors for
the resolved E09 configurations using the BayesBreak DP. No MCMC required.

The BayesBreak DP computes these exactly as:
  boundary_marginals_[i]     = P(b_i=1 | y, k_map)  (boundary event marginal)
  boundary_location_posterior_ = P(t_p=h | y, k_map) (boundary location posterior)

These are conditioned on k_map. For resolved configurations, k_map is the MAP
segment count under the unrestricted model (within the tested k_max range).
For truncated configurations, these would be conditioned on a constrained MAP k.

Resolved configurations:
  - welllog full, geometric_decay_log4, k_max=100, k_map=83
  - welllog stride-8, geometric_decay_log4, k_max=60, k_map=25
  - welllog stride-8, uniform, k_max=60, k_map=37
  - welllog min_seg=5, geometric_decay_log4, k_max=100, k_map=61
  - welllog min_seg=10, geometric_decay_log4, k_max=100, k_map=56
  - CGH single_subject0, geometric_decay_log4, k_max=100, k_map=71

Usage:
    python scripts/supplemental_e09_boundary_marginals.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
    "id": "E09_supplemental_boundary_marginals",
    "canonical_e09_sha256_prefix": "0f5281c46ee8976e",
    "configurations": [
        {"dataset": "welllog_full", "prior": "geometric_decay_log4", "k_max": 100, "k_map": 83, "min_seg": 1},
        {"dataset": "welllog_stride8", "prior": "geometric_decay_log4", "k_max": 60, "k_map": 25, "min_seg": 1},
        {"dataset": "welllog_stride8", "prior": "uniform", "k_max": 60, "k_map": 37, "min_seg": 1},
        {"dataset": "welllog_full", "prior": "geometric_decay_log4", "k_max": 100, "k_map": 61, "min_seg": 5},
        {"dataset": "welllog_full", "prior": "geometric_decay_log4", "k_max": 100, "k_map": 56, "min_seg": 10},
        {"dataset": "cgh_single_subject0", "prior": "geometric_decay_log4", "k_max": 100, "k_map": 71, "min_seg": 1},
    ],
}

SUPPLEMENTAL_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(SUPPLEMENTAL_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "supplemental"
    / "E09_boundary_marginals"
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _geom_prior_log4(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


def _uniform_prior(k: int) -> float:
    return 1.0


def _min_len_prior(min_len: int):
    if min_len <= 1:
        return None
    def f(delta: float) -> float:
        return 1.0 if delta >= min_len else 0.0
    return f


def _robust_sigma2_from_data(y: np.ndarray) -> float:
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return float(sigma ** 2)


def _fit_and_extract_marginals(
    X: np.ndarray,
    y: np.ndarray,
    k_max: int,
    sigma2: float,
    prior_k,
    min_seg: int,
    expected_k_map: int,
    *,
    estimate_hyper: bool = True,
    nu: float | None = None,
    rho2: float | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit BayesBreakGaussian and extract boundary marginal summaries."""
    from bayesbreak import BayesBreakGaussian

    length_prior = _min_len_prior(min_seg)

    if estimate_hyper:
        est = BayesBreakGaussian(
            k_max=k_max,
            estimate_hyper=True,
            sigma2=sigma2,
            prior_k=prior_k,
            length_prior=length_prior,
        )
    else:
        est = BayesBreakGaussian(
            k_max=k_max,
            estimate_hyper=False,
            nu=nu,
            rho2=rho2,
            sigma2=sigma2,
            prior_k=prior_k,
            length_prior=length_prior,
        )

    t0 = time.perf_counter()
    est.fit(X, y, sample_weight=sample_weight)
    elapsed = time.perf_counter() - t0

    k_map = int(est.k_map_)
    n = len(y)

    if k_map != expected_k_map:
        log.warning(
            "k_map mismatch: expected %d, got %d. "
            "Results correspond to refitted k_map=%d.",
            expected_k_map, k_map, k_map,
        )

    # boundary_marginals_[i] = P(b_i=1 | y, k_map) for i in 0..n-1
    bm = est.boundary_marginals_
    # Invariant: sum of boundary marginals = k_map - 1
    bm_sum = float(np.sum(bm))

    # boundary_location_posterior_[p, h] = P(t_p=h | y, k_map)
    # shape: (k_map-1, n) where t_p is the p-th internal boundary position
    blp = est.boundary_location_posterior_

    # MAP boundaries (from max-sum DP)
    map_boundaries = list(est.map_boundaries_)

    # Segment-count posterior
    k_posterior = list(float(x) for x in est.k_posterior_)

    # Top-5 boundary positions by marginal probability
    bm_with_idx = sorted(enumerate(bm), key=lambda x: x[1], reverse=True)
    top5 = [(int(i), float(p)) for i, p in bm_with_idx[:5]]

    # For each MAP boundary, report P(b_i=1 | y, k_map)
    map_internal = map_boundaries[1:-1]
    map_boundary_marginals = [(int(b), float(bm[b])) for b in map_internal]

    return {
        "k_map": k_map,
        "k_map_matches_expected": k_map == expected_k_map,
        "n": n,
        "elapsed_s": elapsed,
        "boundary_marginals_sum": bm_sum,
        "boundary_marginals_expected_sum": float(k_map - 1),
        "map_boundaries": map_boundaries,
        "top5_boundary_positions_by_marginal": top5,
        "map_boundary_marginals_sample": map_boundary_marginals[:10],
        "k_posterior_at_k_map": float(est.k_posterior_[k_map - 1]) if k_map <= len(est.k_posterior_) else None,
        "segment_count_posterior": k_posterior,
        "boundary_location_posterior_shape": list(blp.shape) if blp is not None else None,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_DIR / "run_log.txt"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info("Supplemental E09 boundary marginals starting.")
    log.info("config_sha256=%s", SUPPLEMENTAL_CONFIG_SHA256)
    log.info("Results: %s", RESULTS_DIR)

    t_start = time.perf_counter()
    utc_start = datetime.now(timezone.utc).isoformat()

    # Load datasets
    from bayesbreak.datasets import load_welllog, load_cgh

    welllog_result = load_welllog()
    welllog_y_full = np.asarray(welllog_result.y, dtype=float)
    welllog_y_stride8 = welllog_y_full[::8]
    sigma2_welllog = 4_674_808.19
    sigma2_welllog_stride8 = 5_466_698.12

    cgh_result = load_cgh()
    cgh_matrix = np.asarray(cgh_result.y, dtype=float)
    cgh_subject0 = cgh_matrix[:, 0] if cgh_matrix.ndim == 2 else cgh_matrix
    # Per-probe sample weights for subject 0 (used by canonical E09 run).
    # Omitting these changes k_map (78 without vs 71 with weights).
    cgh_sw_matrix = np.asarray(cgh_result.sample_weight, dtype=float)
    cgh_sw_subject0 = cgh_sw_matrix[:, 0] if cgh_sw_matrix.ndim == 2 else cgh_sw_matrix
    sigma2_cgh = 0.0057

    log.info(
        "Loaded: welllog n=%d stride8 n=%d cgh_subject0 n=%d",
        len(welllog_y_full), len(welllog_y_stride8), len(cgh_subject0),
    )

    all_results: list[dict[str, Any]] = []

    configs = SUPPLEMENTAL_CONFIG["configurations"]
    for cfg in configs:
        dataset = cfg["dataset"]
        prior_name = cfg["prior"]
        k_max = cfg["k_max"]
        k_map_expected = cfg["k_map"]
        min_seg = cfg["min_seg"]

        prior_fn = _geom_prior_log4 if "geometric" in prior_name else _uniform_prior

        if "welllog_full" in dataset:
            y = welllog_y_full
            sigma2 = sigma2_welllog
            estimate_hyper = True
        elif "welllog_stride8" in dataset:
            y = welllog_y_stride8
            sigma2 = sigma2_welllog_stride8
            estimate_hyper = True
        elif "cgh_single_subject0" in dataset:
            y = cgh_subject0
            sigma2 = sigma2_cgh
            estimate_hyper = False
        else:
            log.error("Unknown dataset: %s", dataset)
            continue

        n = len(y)
        X = np.arange(n, dtype=float).reshape(-1, 1)

        log.info(
            "Fitting %s n=%d k_max=%d prior=%s min_seg=%d expected_k_map=%d...",
            dataset, n, k_max, prior_name, min_seg, k_map_expected,
        )

        try:
            kwargs: dict[str, Any] = {}
            if not estimate_hyper:
                kwargs["nu"] = 0.0
                kwargs["rho2"] = 4.0
            if "cgh_single_subject0" in dataset:
                kwargs["sample_weight"] = cgh_sw_subject0
            result = _fit_and_extract_marginals(
                X, y,
                k_max=k_max,
                sigma2=sigma2,
                prior_k=prior_fn,
                min_seg=min_seg,
                expected_k_map=k_map_expected,
                estimate_hyper=estimate_hyper,
                **kwargs,
            )
            result.update({
                "dataset": dataset,
                "prior": prior_name,
                "k_max": k_max,
                "min_seg": min_seg,
                "failed": False,
                "error": "",
            })
            log.info(
                "  done: k_map=%d marginals_sum=%.2f elapsed=%.1fs",
                result["k_map"], result["boundary_marginals_sum"], result["elapsed_s"],
            )
        except Exception as exc:
            log.error("  failed: %s", exc, exc_info=True)
            result = {
                "dataset": dataset,
                "prior": prior_name,
                "k_max": k_max,
                "min_seg": min_seg,
                "failed": True,
                "error": str(exc),
            }

        all_results.append(result)

    t_elapsed = time.perf_counter() - t_start
    utc_end = datetime.now(timezone.utc).isoformat()

    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"configurations": all_results}, f, indent=2)
    log.info("Wrote %s", metrics_path)

    manifest: dict[str, Any] = {
        "id": SUPPLEMENTAL_CONFIG["id"],
        "type": "supplemental",
        "parent_e09_sha256_prefix": SUPPLEMENTAL_CONFIG["canonical_e09_sha256_prefix"],
        "config_sha256": SUPPLEMENTAL_CONFIG_SHA256,
        "git_revision": _git_rev(),
        "utc_start": utc_start,
        "utc_end": utc_end,
        "elapsed_seconds": t_elapsed,
        "n_configurations": len(all_results),
        "n_failed": sum(1 for r in all_results if r.get("failed")),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "notes": [
            "Boundary marginals P(b_i=1|y,k_map) computed exactly via the BayesBreak DP.",
            "No MCMC or variational approximation is used.",
            "All marginals conditioned on k_map (MAP segment count).",
            "For resolved configurations, k_map is the MAP under the full tested range.",
            "This is a supplemental analysis, not part of the frozen E09 confirmatory protocol.",
        ],
    }
    manifest_path = RESULTS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest. elapsed=%.1fs", t_elapsed)
    log.info("Supplemental E09 boundary marginals complete.")


if __name__ == "__main__":
    main()
