"""E10: computational scaling benchmark.

Protocol: .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E10

Dominant computational cost is O(k_max * n^2) from the forward-backward DP
recursion (two nested loops over k and j/i, each summing over segments). The
block-evidence computation is also O(n^2) but has a smaller constant. This
benchmark measures wall-clock time for each stage separately across the (n,
k_max) grid specified in the E10 protocol.

Outputs
-------
results/confirmatory/E10_scaling/<config_sha256[:16]>/
    metrics.json, manifest.json

Usage:
    python scripts/run_e10_scaling.py
"""

from __future__ import annotations

# BLAS thread control: must be set before importing numpy or scipy.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# E10 configuration
# ---------------------------------------------------------------------------

E10_CONFIG: dict[str, Any] = {
    "id": "E10",
    "name": "scaling",
    "type": "benchmark",
    "n_values": [128, 256, 512, 1024, 2048],
    "k_max_values": [5, 10, 20],
    "warmup": 3,
    "repeats": 7,
    "blas_threads": 1,
    "paper_figure_label": "fig:planned-scaling-paper",
    "paper_table_label": None,
}

E10_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E10_CONFIG, sort_keys=True).encode("utf-8")
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E10_scaling"
    / E10_CONFIG_SHA256[:16]
)

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
# Prior
# ---------------------------------------------------------------------------

def _geom_prior_log4(k: int) -> float:
    """Geometric decay prior: p(k) proportional to (1/4)^(k-1)."""
    return math.exp(-math.log(4.0) * (k - 1))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_data(
    n: int,
    n_idx: int,
    k_max_idx: int,
    rep_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a piecewise-constant Gaussian signal.

    k_true=3 segments, boundaries at [0, n//3, 2*n//3, n],
    segment means [0.0, 2.0, 0.0], sigma=1.0.

    Returns X of shape (n, 1) and y of shape (n,).
    """
    ss = np.random.SeedSequence([20261021, 10, n_idx, k_max_idx, rep_idx])
    rng = np.random.default_rng(ss)

    b0, b1, b2, b3 = 0, n // 3, 2 * n // 3, n
    means = [0.0, 2.0, 0.0]
    signal = np.empty(n, dtype=float)
    signal[b0:b1] = means[0]
    signal[b1:b2] = means[1]
    signal[b2:b3] = means[2]

    noise = rng.standard_normal(n)
    y = signal + noise
    X = np.arange(n, dtype=float).reshape(-1, 1)
    return X, y


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _record_environment() -> dict[str, str]:
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
# Single cell benchmark
# ---------------------------------------------------------------------------

def _benchmark_cell(
    n: int,
    k_max: int,
    n_idx: int,
    k_max_idx: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run warmup and timed repeats for one (n, k_max) cell.

    Times six stages:
      1. Block evidence computation.
      2. Forward-backward DP.
      3. Posterior over k.
      4. Boundary event marginals at k_map.
      5. MAP backtracking.
      6. End-to-end fit (est.fit).

    Memory (peak tracemalloc) is measured for the end-to-end fit only.
    """
    from bayesbreak import BayesBreakGaussian
    from bayesbreak import dp as _dp

    result: dict[str, Any] = {
        "n": n,
        "k_max": k_max,
        "k_map": -1,
        "warmup_done": False,
        "n_repeats": repeats,
        "block_evidence_median_s": float("nan"),
        "block_evidence_std_s": float("nan"),
        "forward_backward_median_s": float("nan"),
        "forward_backward_std_s": float("nan"),
        "posterior_k_median_s": float("nan"),
        "posterior_k_std_s": float("nan"),
        "boundary_marginals_median_s": float("nan"),
        "boundary_marginals_std_s": float("nan"),
        "map_backtrack_median_s": float("nan"),
        "map_backtrack_std_s": float("nan"),
        "end_to_end_median_s": float("nan"),
        "end_to_end_std_s": float("nan"),
        "peak_memory_mb": float("nan"),
        "failed": False,
        "error": "",
    }

    try:
        # Generate the data arrays once (same arrays for warmup and timing).
        X, y = _generate_data(n, n_idx, k_max_idx, rep_idx=0)
        y_1d = y.ravel()

        # Build estimator with fixed hyperparameters per protocol.
        est = BayesBreakGaussian(
            k_max=k_max,
            estimate_hyper=False,
            nu=0.0,
            rho2=4.0,
            sigma2=1.0,
            prior_k=_geom_prior_log4,
        )

        # --- Warmup: full fits to populate caches and JIT paths ----------------
        for _ in range(warmup):
            est.fit(X, y_1d)
        result["warmup_done"] = True

        # After warmup, extract fitted intermediate arrays for staged timing.
        # These do not change between repeats for fixed data, so we use the
        # arrays from the last warmup fit.
        lA0 = est.log_block_evidence_           # (n+1, n+1) block log evidences
        hyper = est.hyper_                       # {'nu': ..., 'rho2': ..., 'sigma2': ...}
        w = est.sample_weight_                   # (n,) weight array (all ones here)
        log_C_k = est.log_C_k_                  # (k_max+1,) log partition normalizer
        k_map = int(est.k_map_)
        result["k_map"] = k_map

        # Build log_p_k from estimator (private method on base class).
        log_p_k = est._build_log_p_k(k_max)     # (k_max+1,) with slot 0 = -inf

        # log_g_table is None for the default uniform length prior (no length_prior set).
        log_g = est.log_g_table_                 # None in this configuration

        # --- Timed repeats: stage by stage ------------------------------------

        be_times: list[float] = []
        fb_times: list[float] = []
        pk_times: list[float] = []
        bm_times: list[float] = []
        mb_times: list[float] = []
        e2e_times: list[float] = []

        for rep_i in range(repeats):
            # Stage 1: block evidence
            t0 = time.perf_counter()
            lA0_rep, _ = est._compute_block_evidence(y_1d, hyper, w)
            be_times.append(time.perf_counter() - t0)

            # Stage 2: forward-backward
            t0 = time.perf_counter()
            log_left, log_right = _dp.forward_backward(lA0, n, k_max, log_g_table=log_g)
            fb_times.append(time.perf_counter() - t0)

            # Stage 3: posterior over k
            t0 = time.perf_counter()
            _dp.posterior_over_k(log_left, n, k_max, log_C_k=log_C_k, log_p_k=log_p_k)
            pk_times.append(time.perf_counter() - t0)

            # Stage 4: boundary event marginals at k_map
            t0 = time.perf_counter()
            _dp.boundary_event_marginals_fixed_k(log_left, log_right, n, k_map)
            bm_times.append(time.perf_counter() - t0)

            # Stage 5: MAP backtracking
            t0 = time.perf_counter()
            _dp.max_sum_segmentation(lA0, k_map, log_g_table=log_g)
            mb_times.append(time.perf_counter() - t0)

            # Stage 6: end-to-end fit
            t0 = time.perf_counter()
            est.fit(X, y_1d)
            e2e_times.append(time.perf_counter() - t0)

        # --- Peak memory: tracemalloc over one end-to-end fit ------------------
        tracemalloc.start()
        est_mem = BayesBreakGaussian(
            k_max=k_max,
            estimate_hyper=False,
            nu=0.0,
            rho2=4.0,
            sigma2=1.0,
            prior_k=_geom_prior_log4,
        )
        est_mem.fit(X, y_1d)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.update({
            "block_evidence_median_s": float(np.median(be_times)),
            "block_evidence_std_s": float(np.std(be_times, ddof=1)),
            "forward_backward_median_s": float(np.median(fb_times)),
            "forward_backward_std_s": float(np.std(fb_times, ddof=1)),
            "posterior_k_median_s": float(np.median(pk_times)),
            "posterior_k_std_s": float(np.std(pk_times, ddof=1)),
            "boundary_marginals_median_s": float(np.median(bm_times)),
            "boundary_marginals_std_s": float(np.std(bm_times, ddof=1)),
            "map_backtrack_median_s": float(np.median(mb_times)),
            "map_backtrack_std_s": float(np.std(mb_times, ddof=1)),
            "end_to_end_median_s": float(np.median(e2e_times)),
            "end_to_end_std_s": float(np.std(e2e_times, ddof=1)),
            "peak_memory_mb": float(peak_bytes) / (1024.0 ** 2),
        })

    except Exception as exc:
        log.error(
            "Cell n=%d k_max=%d failed: %s", n, k_max, exc, exc_info=True
        )
        result["failed"] = True
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_BASE / "run_log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("E10 scaling benchmark starting.")
    log.info("config_sha256=%s", E10_CONFIG_SHA256)
    log.info("Results directory: %s", RESULTS_BASE)

    env = _record_environment()
    t_start = time.perf_counter()
    utc_start = datetime.now(timezone.utc).isoformat()

    n_values: list[int] = E10_CONFIG["n_values"]
    k_max_values: list[int] = E10_CONFIG["k_max_values"]
    warmup: int = E10_CONFIG["warmup"]
    repeats: int = E10_CONFIG["repeats"]

    n_cells = len(n_values) * len(k_max_values)
    log.info(
        "Grid: n=%s k_max=%s => %d cells; warmup=%d repeats=%d",
        n_values, k_max_values, n_cells, warmup, repeats,
    )

    all_cells: list[dict[str, Any]] = []

    for n_idx, n in enumerate(n_values):
        for k_max_idx, k_max in enumerate(k_max_values):
            log.info("Cell n=%d k_max=%d starting...", n, k_max)
            cell = _benchmark_cell(
                n=n,
                k_max=k_max,
                n_idx=n_idx,
                k_max_idx=k_max_idx,
                warmup=warmup,
                repeats=repeats,
            )
            all_cells.append(cell)
            if cell["failed"]:
                log.error(
                    "Cell n=%d k_max=%d FAILED: %s",
                    n, k_max, cell["error"],
                )
            else:
                log.info(
                    "Cell n=%d k_max=%d done: k_map=%d end_to_end_median=%.4fs",
                    n, k_max, cell["k_map"], cell["end_to_end_median_s"],
                )

    t_elapsed = time.perf_counter() - t_start
    utc_end = datetime.now(timezone.utc).isoformat()

    n_failed = sum(1 for c in all_cells if c["failed"])
    gate_pass = (n_failed == 0)

    log.info(
        "All cells complete. elapsed=%.1fs cells=%d failed=%d gate_pass=%s",
        t_elapsed, len(all_cells), n_failed, gate_pass,
    )

    # Write metrics.json.
    metrics_path = RESULTS_BASE / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "n_cells": len(all_cells),
                "n_failed": n_failed,
                "elapsed_seconds": t_elapsed,
                "cells": all_cells,
            },
            f,
            indent=2,
        )
    log.info("Wrote %s", metrics_path)

    # Write manifest.json.
    manifest: dict[str, Any] = {
        "id": E10_CONFIG["id"],
        "config_sha256": E10_CONFIG_SHA256,
        "git_revision": env["git_revision"],
        "utc_start": utc_start,
        "utc_end": utc_end,
        "n_cells": len(all_cells),
        "n_failed": n_failed,
        "elapsed_seconds": t_elapsed,
        "gate_pass": gate_pass,
        "environment": env,
        "output_files": {
            "metrics_json": "metrics.json",
        },
    }
    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: gate_pass=%s", gate_pass)

    if n_failed > 0:
        log.error("E10 GATE FAIL: %d cells failed.", n_failed)
        sys.exit(1)
    log.info("E10 complete. gate_pass=True")


if __name__ == "__main__":
    main()
