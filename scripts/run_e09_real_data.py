"""E09: real data, priors, annotation verification, and baselines.

Protocol: .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E09

Datasets
--------
- well-log NMR (TCPD, n=4050 full; n=507 stride-8 subsample): Gaussian
- array-CGH log2-ratios (cran/ecp ACGH.RData, 2215 probes x 43 subjects):
  Gaussian shared-boundary via SharedBoundaryReplicatesSegmenter
- methylKit chr21 CpG methylation (n=1904, coverage-weighted BetaObs)
- S&P 500: EXCLUDED -- yfinance unavailable; load_spx() returns simulated
  fallback which MUST NOT be presented as observational data

Baselines
---------
ruptures (PELT, OP, BS, WBS) and rpy2 (CBS, SMUCE, Fearnhead) are NOT
installed. All comparators are marked UNAVAILABLE.

Prior sensitivity
-----------------
Count prior: geometric_decay_log4 (default) vs uniform.
k_max grid: {15, 30, 60} + extended where P(k=k_max|y) >= 0.01.
Minimum segment length sensitivity for welllog: {1, 5, 10} observations.

Truncation
----------
A cell is marked truncated when k_map = k_max OR P(k=k_max|y) >= 0.01.
Cells where k_max extension exceeds budget_seconds are marked budget_exceeded.

Outputs
-------
results/confirmatory/E09_real_data/<config_sha256[:16]>/
    manifest.json, metrics.json, run_log.txt

Usage (full run):
    python scripts/run_e09_real_data.py

Usage (pilot -- small k_max, skip extended grid):
    python scripts/run_e09_real_data.py --pilot
"""

from __future__ import annotations

import argparse
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
# E09 configuration
# ---------------------------------------------------------------------------

E09_CONFIG: dict[str, Any] = {
    "id": "E09",
    "name": "real_data",
    "protocol_revision": "2026-10-24-2",
    "type": "real_data",
    "k_max_grid": [15, 30, 60],
    # Extended k_max values used when standard grid shows truncation.
    "welllog_k_max_extended": 100,
    "methylation_k_max_extended": 120,
    # CGH pooled fit is expensive (~378s at k_max=30). Limit to the standard
    # grid; single-subject representative fit extends to k_max=100.
    "cgh_k_max_extended_single_subject": 100,
    "budget_seconds_per_cell": 600,
    "count_priors": ["geometric_decay_log4", "uniform"],
    "welllog_min_seg_lengths": [1, 5, 10],
    # CGH prior: nu=0 (log2-ratio neutral), rho2=4.0 (diffuse prior on means up to +-2).
    # These are appropriate for log2-ratio data centered near 0 with typical CNV excursions.
    "cgh_gaussian_prior_nu": 0.0,
    "cgh_gaussian_prior_rho2": 4.0,
    # Welllog: estimate_hyper=True so nu and rho2 adapt to ~10^5 raw NMR values.
    # sigma2 is fixed to the robust MAD-diff estimator in _robust_sigma2; nu and rho2
    # are estimated from data by BayesBreakGaussian's moment estimator.
    "welllog_gaussian_estimate_hyper": True,
    "welllog_gaussian_sigma2": "robust_mad_diff",
    "beta_obs_quadrature_points": 128,
    "cgh_n_subjects_pooled": 43,
    "seed_base": [20261021, 9],
    # Incomplete items documented.
    "incomplete": {
        "spx": "yfinance not installed; load_spx() returns simulated fallback. "
               "SPX excluded from E09 per protocol: never label simulated data as real.",
        "baselines_ruptures": "ruptures package not installed. PELT, OP, BS, WBS UNAVAILABLE.",
        "baselines_rpy2": "rpy2 package not installed. CBS (DNAcopy), SMUCE, Fearnhead UNAVAILABLE.",
    },
}

E09_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E09_CONFIG, sort_keys=True).encode()
).hexdigest()

RESULTS_BASE = (
    Path(__file__).parent.parent
    / "results" / "confirmatory" / "E09_real_data"
    / E09_CONFIG_SHA256[:16]
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
# Priors
# ---------------------------------------------------------------------------

def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


def _uniform_prior(k: int) -> float:
    return 1.0


def _get_prior(name: str):
    if name == "geometric_decay_log4":
        return _geom_prior
    elif name == "uniform":
        return _uniform_prior
    else:
        raise ValueError(f"Unknown count prior: {name!r}")


def _min_len_prior(min_len: int):
    """Return a length_prior function enforcing a minimum segment size.

    The BayesBreak length_prior API requires g(delta) > 0 for allowed blocks;
    the framework then stores log(g). For blocked segments, return 0.0 so the
    condition ``value > 0`` fails and the block is assigned -inf.
    """
    if min_len <= 1:
        return None
    def f(delta: float) -> float:
        return 1.0 if delta >= min_len else 0.0
    return f


# ---------------------------------------------------------------------------
# Robust sigma estimation for welllog
# ---------------------------------------------------------------------------

def _robust_sigma2(y: np.ndarray) -> float:
    """Estimate sigma^2 for welllog using MAD of differenced observations.

    Differencing cancels mean shifts across segments; the scale of diffs
    is sqrt(2)*sigma under Gaussian noise, so sigma = std_hat / sqrt(2).
    The MAD-based estimator is robust to outliers.
    """
    diffs = np.diff(y)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    sigma_hat = (mad * 1.4826) / math.sqrt(2.0)
    return max(sigma_hat ** 2, 1e-6)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_datasets() -> dict[str, Any]:
    """Load all real datasets; record source, shape, SHA256 provenance."""
    from bayesbreak.datasets import load_welllog, load_cgh, load_methylation
    from bayesbreak.datasets.spx import load_spx

    results: dict[str, Any] = {}

    # Welllog.
    wl = load_welllog()
    results["welllog"] = {
        "bundle": wl,
        "source": wl.source,
        "n": int(wl.y.shape[0]),
        "data_sha256": _sha256_array(wl.y),
    }
    log.info("welllog: source=%s n=%d", wl.source, wl.y.shape[0])
    if wl.source == "simulated":
        log.warning("welllog: SIMULATED FALLBACK. Results may not be used as real-data analysis.")

    # Welllog stride-8 subsample (n=507).
    idx8 = np.arange(0, wl.y.shape[0], 8)
    y8 = wl.y[idx8]
    X8 = wl.X[idx8]
    results["welllog_stride8"] = {
        "y": y8, "X": X8,
        "source": wl.source,
        "n": int(y8.shape[0]),
        "stride": 8,
        "data_sha256": _sha256_array(y8),
    }
    log.info("welllog_stride8: n=%d", y8.shape[0])

    # CGH (all 43 subjects for pooled; subject 0 for single-subject analysis).
    cgh = load_cgh()
    results["cgh"] = {
        "bundle": cgh,
        "source": cgh.source,
        "n_probes": int(cgh.y.shape[0]),
        "n_subjects": int(cgh.y.shape[1]) if cgh.y.ndim == 2 else 1,
        "data_sha256": _sha256_array(cgh.y),
    }
    log.info("cgh: source=%s shape=%s", cgh.source, cgh.y.shape)
    if cgh.source == "simulated":
        log.warning("cgh: SIMULATED FALLBACK. Results may not be used as real-data analysis.")

    # Methylation.
    meth = load_methylation()
    results["methylation"] = {
        "bundle": meth,
        "source": meth.source,
        "n": int(meth.y.shape[0]),
        "data_sha256": _sha256_array(meth.y),
        "descriptor_sha256": _sha256_array(meth.sample_weight),
    }
    log.info("methylation: source=%s n=%d", meth.source, meth.y.shape[0])
    if meth.source == "simulated":
        log.warning("methylation: SIMULATED FALLBACK. Results may not be used as real-data analysis.")

    # SPX -- document exclusion.
    spx = load_spx()
    results["spx"] = {
        "source": spx.source,
        "excluded": True,
        "reason": E09_CONFIG["incomplete"]["spx"],
    }
    log.info("spx: source=%s -- EXCLUDED (simulated fallback only)", spx.source)

    return results


def _sha256_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------

def _fit_gaussian(
    X: np.ndarray,
    y: np.ndarray,
    k_max: int,
    sigma2: float,
    prior_k,
    min_seg_len: int = 1,
    sample_weight: np.ndarray | None = None,
    *,
    nu: float | None = None,
    rho2: float | None = None,
    estimate_hyper: bool = False,
) -> dict[str, Any]:
    """Fit BayesBreakGaussian.

    When estimate_hyper=True, nu and rho2 are estimated from data (adapts to data scale).
    Pass nu and rho2 explicitly for fixed priors (appropriate when data scale is known, e.g. CGH).
    sigma2 is always used as provided (robust estimate of within-segment noise).
    """
    from bayesbreak import BayesBreakGaussian
    length_prior = _min_len_prior(min_seg_len)
    if estimate_hyper:
        est = BayesBreakGaussian(
            k_max=k_max,
            estimate_hyper=True,
            sigma2=sigma2,  # override sigma2 only; nu and rho2 estimated from data
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
    t0 = time.time()
    est.fit(X, y, sample_weight=sample_weight)
    elapsed = time.time() - t0
    return _extract_fit_result(est, elapsed)


def _fit_gaussian_pooled(
    X: np.ndarray,
    y_matrix: np.ndarray,
    sw_matrix: np.ndarray,
    k_max: int,
    sigma2: float,
    prior_k,
    nu: float = 0.0,
    rho2: float = 4.0,
) -> dict[str, Any]:
    from bayesbreak import BayesBreakGaussian, SharedBoundaryReplicatesSegmenter
    base_est = BayesBreakGaussian(
        k_max=k_max,
        estimate_hyper=False,
        nu=nu,
        rho2=rho2,
        sigma2=sigma2,
        prior_k=prior_k,
    )
    seg = SharedBoundaryReplicatesSegmenter(base_est)
    S = y_matrix.shape[1]
    t0 = time.time()
    seg.fit(
        X,
        [y_matrix[:, s] for s in range(S)],
        sample_weight=[sw_matrix[:, s] for s in range(S)],
    )
    elapsed = time.time() - t0
    result = _extract_fit_result(seg, elapsed)
    result["n_subjects"] = S
    result["log_evidence_pooled"] = float(seg.log_evidence_)
    result["map_boundaries"] = list(map(int, seg.map_boundaries_))
    return result


def _fit_beta_obs(
    X: np.ndarray,
    y: np.ndarray,
    phi: np.ndarray,
    k_max: int,
    prior_k,
) -> dict[str, Any]:
    from bayesbreak import BayesBreakBetaObs
    config = E09_CONFIG
    est = BayesBreakBetaObs(
        k_max=k_max,
        estimate_hyper=True,
        phi=phi,
        quadrature_points=config["beta_obs_quadrature_points"],
        prior_k=prior_k,
    )
    t0 = time.time()
    est.fit(X, y)
    elapsed = time.time() - t0
    return _extract_fit_result(est, elapsed)


def _extract_fit_result(est: Any, elapsed: float) -> dict[str, Any]:
    """Extract standardized fit result from a fitted estimator."""
    k_posterior = np.asarray(est.k_posterior_)
    k_map = int(est.k_map_)
    k_max = int(k_posterior.size)
    p_kmax = float(k_posterior[k_max - 1])
    truncated = (k_map == k_max) or (p_kmax >= 0.01)
    return {
        "k_map": k_map,
        "k_max": k_max,
        "p_k_kmax": p_kmax,
        "truncated": truncated,
        "k_posterior": k_posterior.tolist(),
        "log_evidence": float(est.log_evidence_),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Cell runners
# ---------------------------------------------------------------------------

def _run_welllog_cells(
    datasets: dict[str, Any],
    pilot: bool,
) -> list[dict[str, Any]]:
    """Run all welllog cells: full + stride-8, multiple k_max and priors."""
    config = E09_CONFIG
    rows: list[dict[str, Any]] = []
    wl_bundle = datasets["welllog"]["bundle"]
    wl8 = datasets["welllog_stride8"]
    y_full = wl_bundle.y
    X_full = wl_bundle.X
    y8 = wl8["y"]
    X8 = wl8["X"]
    sigma2_full = _robust_sigma2(y_full)
    sigma2_8 = _robust_sigma2(y8)
    log.info("welllog robust sigma2: full=%.2f stride8=%.2f", sigma2_full, sigma2_8)

    k_max_grid = [15] if pilot else config["k_max_grid"]
    priors = config["count_priors"]
    min_seg_lengths = [1] if pilot else config["welllog_min_seg_lengths"]

    for variant, y, X, sigma2 in [
        ("full", y_full, X_full, sigma2_full),
        ("stride8", y8, X8, sigma2_8),
    ]:
        for prior_name in priors:
            prior_fn = _get_prior(prior_name)
            for k_max in k_max_grid:
                log.info("welllog/%s k_max=%d prior=%s min_seg=1", variant, k_max, prior_name)
                try:
                    result = _fit_gaussian(X, y, k_max, sigma2, prior_fn,
                                          estimate_hyper=True)
                    row = _cell_row(
                        dataset="welllog", variant=variant, k_max=k_max,
                        prior=prior_name, min_seg_len=1, result=result,
                    )
                except Exception as exc:
                    log.exception("welllog/%s k_max=%d prior=%s failed", variant, k_max, prior_name)
                    row = _cell_row_failed(
                        dataset="welllog", variant=variant, k_max=k_max,
                        prior=prior_name, min_seg_len=1, error=str(exc),
                    )
                rows.append(row)

            # Extended k_max for full welllog when standard grid is truncated.
            if variant == "full" and not pilot:
                k_ext = config["welllog_k_max_extended"]
                if k_ext not in k_max_grid:
                    log.info("welllog/full k_max=%d prior=%s (truncation extension)", k_ext, prior_name)
                    try:
                        result = _fit_gaussian(X_full, y_full, k_ext, sigma2_full, prior_fn,
                                              estimate_hyper=True)
                        row = _cell_row(
                            dataset="welllog", variant="full_extended", k_max=k_ext,
                            prior=prior_name, min_seg_len=1, result=result,
                        )
                    except Exception as exc:
                        log.exception("welllog/full_extended k_max=%d failed", k_ext)
                        row = _cell_row_failed(
                            dataset="welllog", variant="full_extended", k_max=k_ext,
                            prior=prior_name, min_seg_len=1, error=str(exc),
                        )
                    rows.append(row)

        # Minimum segment length sensitivity (full welllog, geometric prior only).
        if variant == "full" and not pilot:
            prior_fn = _get_prior("geometric_decay_log4")
            k_ext = config["welllog_k_max_extended"]
            for min_seg_len in min_seg_lengths:
                if min_seg_len == 1:
                    continue  # already run above
                log.info("welllog/full k_max=%d min_seg=%d prior=geom", k_ext, min_seg_len)
                try:
                    result = _fit_gaussian(X_full, y_full, k_ext, sigma2_full, prior_fn,
                                          min_seg_len=min_seg_len, estimate_hyper=True)
                    row = _cell_row(
                        dataset="welllog", variant="full_extended", k_max=k_ext,
                        prior="geometric_decay_log4", min_seg_len=min_seg_len, result=result,
                    )
                except Exception as exc:
                    log.exception("welllog min_seg=%d failed", min_seg_len)
                    row = _cell_row_failed(
                        dataset="welllog", variant="full_extended", k_max=k_ext,
                        prior="geometric_decay_log4", min_seg_len=min_seg_len, error=str(exc),
                    )
                rows.append(row)

    return rows


def _run_cgh_cells(
    datasets: dict[str, Any],
    pilot: bool,
) -> list[dict[str, Any]]:
    """Run CGH cells: pooled (SharedBoundaryReplicatesSegmenter) + single-subject."""
    config = E09_CONFIG
    rows: list[dict[str, Any]] = []
    cgh_bundle = datasets["cgh"]["bundle"]

    if cgh_bundle.source == "simulated":
        log.warning("cgh: simulated data; skipping real-data CGH analysis.")
        return []

    if cgh_bundle.y.ndim != 2:
        log.warning("cgh: unexpected shape %s; expected 2D (probes x subjects).", cgh_bundle.y.shape)
        return []

    n_probes, S = cgh_bundle.y.shape
    X = cgh_bundle.X
    y_mat = cgh_bundle.y
    sw_mat = cgh_bundle.sample_weight

    # Robust sigma2: pool all subjects' differenced observations.
    sigma2_cgh = float(np.median([
        _robust_sigma2(y_mat[:, s]) for s in range(S)
    ]))
    log.info("cgh: pooled robust sigma2=%.4f across %d subjects", sigma2_cgh, S)

    k_max_grid = [15] if pilot else config["k_max_grid"]
    prior_fn = _get_prior("geometric_decay_log4")

    # Pooled (SharedBoundaryReplicatesSegmenter) cells.
    for k_max in k_max_grid:
        log.info("cgh/pooled k_max=%d n_subjects=%d", k_max, S)
        t_budget = config["budget_seconds_per_cell"]
        t0 = time.time()
        try:
            result = _fit_gaussian_pooled(
                X, y_mat, sw_mat, k_max, sigma2_cgh, prior_fn,
                nu=config["cgh_gaussian_prior_nu"],
                rho2=config["cgh_gaussian_prior_rho2"],
            )
            elapsed = time.time() - t0
            if elapsed > t_budget:
                log.warning("cgh/pooled k_max=%d: elapsed %.1fs exceeded budget %.0fs",
                            k_max, elapsed, t_budget)
                result["budget_note"] = f"elapsed {elapsed:.1f}s exceeded budget {t_budget}s"
            row = _cell_row(
                dataset="cgh", variant="pooled", k_max=k_max,
                prior="geometric_decay_log4", min_seg_len=1, result=result,
            )
        except Exception as exc:
            log.exception("cgh/pooled k_max=%d failed", k_max)
            row = _cell_row_failed(
                dataset="cgh", variant="pooled", k_max=k_max,
                prior="geometric_decay_log4", min_seg_len=1, error=str(exc),
            )
        rows.append(row)

    # Single-subject representative analysis (subject 0) at extended k_max.
    k_ext = config["cgh_k_max_extended_single_subject"]
    single_subj_grid = [15] if pilot else sorted(set(k_max_grid) | {k_ext})
    for k_max in single_subj_grid:
        log.info("cgh/single_subject0 k_max=%d", k_max)
        try:
            result = _fit_gaussian(
                X, y_mat[:, 0], k_max, sigma2_cgh, prior_fn,
                sample_weight=sw_mat[:, 0],
                nu=config["cgh_gaussian_prior_nu"],
                rho2=config["cgh_gaussian_prior_rho2"],
                estimate_hyper=False,
            )
            row = _cell_row(
                dataset="cgh", variant="single_subject0", k_max=k_max,
                prior="geometric_decay_log4", min_seg_len=1, result=result,
            )
        except Exception as exc:
            log.exception("cgh/single_subject0 k_max=%d failed", k_max)
            row = _cell_row_failed(
                dataset="cgh", variant="single_subject0", k_max=k_max,
                prior="geometric_decay_log4", min_seg_len=1, error=str(exc),
            )
        rows.append(row)

    return rows


def _run_methylation_cells(
    datasets: dict[str, Any],
    pilot: bool,
) -> list[dict[str, Any]]:
    """Run methylation cells: BetaObs with coverage-weighted precision."""
    config = E09_CONFIG
    rows: list[dict[str, Any]] = []
    meth_bundle = datasets["methylation"]["bundle"]

    if meth_bundle.source == "simulated":
        log.warning("methylation: simulated data; skipping real-data methylation analysis.")
        return []

    y = meth_bundle.y
    X = meth_bundle.X
    phi = meth_bundle.sample_weight  # per-CpG read coverage used as precision

    k_max_grid = [15] if pilot else config["k_max_grid"]
    k_ext = config["methylation_k_max_extended"]
    priors = config["count_priors"]

    for prior_name in priors:
        prior_fn = _get_prior(prior_name)
        for k_max in k_max_grid:
            log.info("methylation k_max=%d prior=%s", k_max, prior_name)
            try:
                result = _fit_beta_obs(X, y, phi, k_max, prior_fn)
                row = _cell_row(
                    dataset="methylation", variant="full", k_max=k_max,
                    prior=prior_name, min_seg_len=1, result=result,
                )
            except Exception as exc:
                log.exception("methylation k_max=%d prior=%s failed", k_max, prior_name)
                row = _cell_row_failed(
                    dataset="methylation", variant="full", k_max=k_max,
                    prior=prior_name, min_seg_len=1, error=str(exc),
                )
            rows.append(row)

        # Extended k_max.
        if not pilot and k_ext not in k_max_grid:
            log.info("methylation k_max=%d prior=%s (truncation extension)", k_ext, prior_name)
            try:
                result = _fit_beta_obs(X, y, phi, k_ext, prior_fn)
                row = _cell_row(
                    dataset="methylation", variant="full_extended", k_max=k_ext,
                    prior=prior_name, min_seg_len=1, result=result,
                )
            except Exception as exc:
                log.exception("methylation k_max=%d prior=%s (extended) failed", k_ext, prior_name)
                row = _cell_row_failed(
                    dataset="methylation", variant="full_extended", k_max=k_ext,
                    prior=prior_name, min_seg_len=1, error=str(exc),
                )
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _cell_row(
    dataset: str,
    variant: str,
    k_max: int,
    prior: str,
    min_seg_len: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "k_max": k_max,
        "count_prior": prior,
        "min_seg_len": min_seg_len,
        "k_map": result["k_map"],
        "p_k_kmax": result["p_k_kmax"],
        "truncated": result["truncated"],
        "log_evidence": result["log_evidence"],
        "elapsed_seconds": result["elapsed_seconds"],
        "failed": False,
        "error": "",
        "budget_note": result.get("budget_note", ""),
    }
    if "n_subjects" in result:
        row["n_subjects"] = result["n_subjects"]
    if "map_boundaries" in result:
        row["map_boundaries"] = result["map_boundaries"]
    return row


def _cell_row_failed(
    dataset: str,
    variant: str,
    k_max: int,
    prior: str,
    min_seg_len: int,
    error: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "variant": variant,
        "k_max": k_max,
        "count_prior": prior,
        "min_seg_len": min_seg_len,
        "k_map": -1,
        "p_k_kmax": float("nan"),
        "truncated": True,
        "log_evidence": float("nan"),
        "elapsed_seconds": float("nan"),
        "failed": True,
        "error": error,
        "budget_note": "",
    }


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run E09 real-data experiment.")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: k_max=15 only, no extended grid.")
    args = parser.parse_args()
    pilot = args.pilot

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    log_file = RESULTS_BASE / "run_log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("E09 starting. pilot=%s config_sha256=%s", pilot, E09_CONFIG_SHA256)
    log.info("Results: %s", RESULTS_BASE)
    log.info("INCOMPLETE: SPX=%s", E09_CONFIG["incomplete"]["spx"])
    log.info("INCOMPLETE: baselines=%s", E09_CONFIG["incomplete"]["baselines_ruptures"])
    log.info("INCOMPLETE: baselines=%s", E09_CONFIG["incomplete"]["baselines_rpy2"])

    env = _record_environment()
    t_start = time.time()
    utc_start = datetime.now(timezone.utc).isoformat()

    # Load datasets.
    log.info("Loading datasets...")
    datasets = _load_datasets()

    # Check for real data.
    real_data_ok = all(
        datasets[k]["source"] == "downloaded"
        for k in ("welllog", "cgh", "methylation")
    )
    if not real_data_ok:
        log.warning(
            "One or more datasets are SIMULATED. Results cannot be reported "
            "as real-data analysis per E09 protocol."
        )

    all_rows: list[dict[str, Any]] = []

    # Welllog cells.
    log.info("=== Welllog ===")
    wl_rows = _run_welllog_cells(datasets, pilot)
    all_rows.extend(wl_rows)
    log.info("Welllog: %d cells", len(wl_rows))

    # CGH cells.
    log.info("=== CGH ===")
    cgh_rows = _run_cgh_cells(datasets, pilot)
    all_rows.extend(cgh_rows)
    log.info("CGH: %d cells", len(cgh_rows))

    # Methylation cells.
    log.info("=== Methylation ===")
    meth_rows = _run_methylation_cells(datasets, pilot)
    all_rows.extend(meth_rows)
    log.info("Methylation: %d cells", len(meth_rows))

    t_elapsed = time.time() - t_start
    utc_end = datetime.now(timezone.utc).isoformat()
    n_failed = sum(1 for r in all_rows if r.get("failed"))
    n_truncated = sum(1 for r in all_rows if r.get("truncated") and not r.get("failed"))
    # Primary analysis gate: min_seg>1 cells are sensitivity analysis; failures don't gate.
    primary_failed = sum(
        1 for r in all_rows
        if r.get("failed") and r.get("min_seg_len", 1) == 1
    )
    sensitivity_failed = n_failed - primary_failed
    log.info("All cells done. elapsed=%.1fs cells=%d failed=%d (primary=%d sensitivity=%d) truncated=%d",
             t_elapsed, len(all_rows), n_failed, primary_failed, sensitivity_failed, n_truncated)

    # Write metrics JSON.
    metrics_path = RESULTS_BASE / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "n_cells": len(all_rows),
                "n_failed": n_failed,
                "n_failed_primary": primary_failed,
                "n_failed_sensitivity": sensitivity_failed,
                "n_truncated": n_truncated,
                "elapsed_seconds": t_elapsed,
                "pilot": pilot,
                "real_data_ok": real_data_ok,
                "dataset_provenance": {
                    k: {
                        kk: v for kk, v in datasets[k].items()
                        if kk not in ("bundle", "y", "X")
                    }
                    for k in datasets
                },
                "incomplete": E09_CONFIG["incomplete"],
                "cells": all_rows,
            },
            f, indent=2,
        )
    log.info("Wrote %s", metrics_path)

    # Manifest.
    manifest = {
        "id": E09_CONFIG["id"],
        "protocol_revision": E09_CONFIG["protocol_revision"],
        "config_sha256": E09_CONFIG_SHA256,
        "git_revision": env["git_revision"],
        "utc_start": utc_start,
        "utc_end": utc_end,
        "pilot": pilot,
        "n_cells": len(all_rows),
        "n_failed": n_failed,
        "n_failed_primary": primary_failed,
        "n_failed_sensitivity": sensitivity_failed,
        "n_truncated": n_truncated,
        "real_data_ok": real_data_ok,
        "environment": env,
        "output_files": {"metrics_json": "metrics.json"},
        "gate_pass": primary_failed == 0 and real_data_ok,
        "gate_notes": _gate_notes(datasets, all_rows),
    }
    manifest_path = RESULTS_BASE / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Wrote manifest: gate_pass=%s", manifest["gate_pass"])

    if sensitivity_failed > 0:
        log.warning(
            "E09 SENSITIVITY NOTE: %d sensitivity cells (min_seg>1) failed. "
            "These are documented but do not gate the primary analysis.",
            sensitivity_failed,
        )
    if primary_failed > 0:
        log.error("E09 GATE FAIL: %d primary cells failed.", primary_failed)
        sys.exit(1)
    if not real_data_ok:
        log.error("E09 GATE FAIL: simulated data fallback for one or more datasets.")
        sys.exit(1)
    log.info("E09 complete. gate_pass=True")


def _gate_notes(datasets: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    # Truncation summary.
    for dataset in ("welllog", "cgh", "methylation"):
        dataset_rows = [r for r in rows if r["dataset"] == dataset and not r.get("failed")]
        all_trunc = all(r.get("truncated") for r in dataset_rows) if dataset_rows else True
        max_k_max = max((r["k_max"] for r in dataset_rows), default=0)
        if all_trunc:
            notes.append(
                f"{dataset}: ALL CELLS TRUNCATED at k_max up to {max_k_max}. "
                "True posterior mode unknown."
            )
        else:
            resolved = [r for r in dataset_rows if not r.get("truncated")]
            min_resolved = min(r["k_max"] for r in resolved) if resolved else -1
            notes.append(f"{dataset}: truncation resolved at k_max={min_resolved}.")
    notes.append("SPX excluded: simulated fallback only.")
    notes.append("Baselines: ruptures and rpy2 not installed; all comparators unavailable.")
    return notes


if __name__ == "__main__":
    main()
