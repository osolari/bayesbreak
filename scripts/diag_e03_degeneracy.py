"""Diagnostic A1: unit-gap degeneracy investigation for E03.

When G=1 (all physical gaps equal), the gap_hazard prior assigns equal weight
to all interior boundary positions, making it mathematically equivalent to the
uniform prior.  This diagnostic verifies that claim by:

1. Generating ONE set of 10 datasets for (G=1, template=[60,120]) using the
   E03 runner's exact generator and seed logic (cell_idx=0 for uniform).
2. Fitting all 4 priors to EACH of the same 10 datasets.
3. Reporting the max absolute difference between gap_hazard and uniform
   posterior results (boundary marginals, log evidence, k_map).

Mathematical argument:
  The gap_hazard prior sets h_b = x[b] - x[b-1] for each interior boundary b.
  When G=1, all gaps x[i] - x[i-1] = 1 for all i.  Therefore h_b = 1 for all
  interior b.  Since log h_b = 0 for all b, the gap_hazard prior contributes
  zero to the log prior table, making it identical to the uniform prior.

Usage:
    uv run python scripts/diag_e03_degeneracy.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure src is importable
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse construction helpers from the E03 runner
_SCRIPTS = Path(__file__).parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# Inline copies of E03 runner helpers (to avoid import-level side effects)
# ---------------------------------------------------------------------------

# E03 config
E03_CONFIG = {
    "n": 240,
    "gap_start_idx": 120,
    "gaussian_sigma2": 1.0,
    "gaussian_prior": {"nu": 0.0, "rho2": 4.0},
    "block_means": [0.0, 1.5, 0.0],
    "k_max": 8,
    "seed_base": [20261021, 3],
    "small_positive_constant": 1e-6,
}

SMALL_POS = float(E03_CONFIG["small_positive_constant"])


def _build_observation_coords(n: int, gap: int, gap_start_idx: int) -> np.ndarray:
    x = np.empty(n, dtype=float)
    x[0] = 1.0
    for i in range(1, n):
        if i == gap_start_idx:
            x[i] = x[i - 1] + gap
        else:
            x[i] = x[i - 1] + 1.0
    return x


def _build_boundary_coords(x: np.ndarray) -> np.ndarray:
    n = len(x)
    u = np.empty(n + 1, dtype=float)
    u[0] = x[0] - 0.5
    for i in range(1, n):
        u[i] = 0.5 * (x[i - 1] + x[i])
    u[n] = x[n - 1] + 0.5
    return u


def _build_prior_config(prior_name: str, x: np.ndarray, u: np.ndarray):
    from bayesbreak.priors import PartitionPriorConfig

    n = len(x)
    u_gaps = np.diff(u)
    median_gap = float(np.median(u_gaps))

    hazard_params = {
        f"hazard_{b}": float(max(x[b] - x[b - 1], SMALL_POS))
        for b in range(1, n)
    }

    if prior_name == "uniform":
        return PartitionPriorConfig(segment_cohesion="uniform", boundary_hazard="uniform")
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


def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


def _generate_dataset(n, gap, gap_start_idx, true_template, block_means, sigma2, rng):
    x = _build_observation_coords(n, gap, gap_start_idx)
    u = _build_boundary_coords(x)
    b1, b2 = true_template
    true_signal = np.empty(n, dtype=float)
    true_signal[:b1] = block_means[0]
    true_signal[b1:b2] = block_means[1]
    true_signal[b2:] = block_means[2]
    sigma = math.sqrt(sigma2)
    y = true_signal + rng.standard_normal(n) * sigma
    return {"y": y, "x": x, "u": u}


def _run_inference(y, x, u, prior_name, prior_config):
    from bayesbreak.families.gaussian import BayesBreakGaussian
    from bayesbreak.dp import boundary_event_marginals_marginalised

    n = len(y)
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
    bm = boundary_event_marginals_marginalised(
        est.log_left_, est.log_right_, est.log_posterior_k_, n, E03_CONFIG["k_max"]
    )
    return {
        "bm": bm,
        "log_evidence": float(est.log_evidence_),
        "k_map": int(est.k_map_),
        "k_posterior": est.k_posterior_.copy(),
    }


# ---------------------------------------------------------------------------
# Mathematical verification: are all hazard params equal at G=1?
# ---------------------------------------------------------------------------

def verify_gap_hazard_uniform_equivalence_math(gap: int) -> dict:
    """Verify mathematically that gap_hazard == uniform when all gaps equal."""
    n = E03_CONFIG["n"]
    gap_start_idx = E03_CONFIG["gap_start_idx"]
    x = _build_observation_coords(n, gap, gap_start_idx)
    u = _build_boundary_coords(x)

    # Compute all hazard values for gap_hazard prior
    hazards = {b: float(max(x[b] - x[b - 1], SMALL_POS)) for b in range(1, n)}

    unique_vals = set(hazards.values())
    all_equal = len(unique_vals) == 1
    h_val = list(unique_vals)[0] if all_equal else None

    # For uniform prior: log h_b = 0 for all b
    # For gap_hazard: log h_b = log(hazard_b)
    # Equivalent iff log(hazard_b) = const for all b (i.e., all hazards equal)
    # (constant log differences only shift the normalizer, not the posterior)

    if all_equal:
        log_h = math.log(h_val) if h_val > 0 else -math.inf
        logger.info(
            "Gap G=%d: all %d hazard values = %.6f  ->  log h_b = %.6f (constant) "
            "=> gap_hazard is EQUIVALENT to uniform up to a constant",
            gap, len(hazards), h_val, log_h,
        )
    else:
        logger.info(
            "Gap G=%d: hazard values are NOT all equal. Unique values: %s",
            gap, sorted(unique_vals)[:5],
        )

    return {
        "gap": gap,
        "all_hazards_equal": all_equal,
        "hazard_value": h_val,
        "n_unique": len(unique_vals),
    }


# ---------------------------------------------------------------------------
# Paired comparison: same 10 datasets, all 4 priors
# ---------------------------------------------------------------------------

def run_paired_comparison(gap: int = 1, true_template: list = None, n_datasets: int = 10):
    """Fit all 4 priors to the same datasets; report max absolute differences."""
    if true_template is None:
        true_template = [60, 120]

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "Paired comparison: gap=%d, template=%s, n_datasets=%d",
        gap, true_template, n_datasets,
    )
    logger.info("=" * 70)

    n = E03_CONFIG["n"]
    gap_start_idx = E03_CONFIG["gap_start_idx"]
    x = _build_observation_coords(n, gap, gap_start_idx)
    u = _build_boundary_coords(x)

    prior_names = ["uniform", "gap_hazard", "length_cohesion", "combined"]
    prior_configs = {p: _build_prior_config(p, x, u) for p in prior_names}

    # Use cell_idx=0 (uniform's seed) for dataset generation — same seed for all priors
    # This is the "paired" design the confirmatory run did NOT use.
    DATASETS_SEED_BASE = [20261021, 3, 0]  # same as cell_idx=0 in the runner

    results_per_dataset = []

    for rep in range(n_datasets):
        seed_seq = np.random.SeedSequence([*DATASETS_SEED_BASE, rep])
        data_seed = seed_seq.spawn(2)[0]
        rng = np.random.default_rng(data_seed)

        ds = _generate_dataset(
            n=n,
            gap=gap,
            gap_start_idx=gap_start_idx,
            true_template=true_template,
            block_means=E03_CONFIG["block_means"],
            sigma2=E03_CONFIG["gaussian_sigma2"],
            rng=rng,
        )

        per_prior = {}
        for pname in prior_names:
            res = _run_inference(ds["y"], ds["x"], ds["u"], pname, prior_configs[pname])
            per_prior[pname] = res

        results_per_dataset.append(per_prior)

    # Compute differences: gap_hazard vs uniform (on the same datasets)
    max_bm_diff = 0.0
    max_logevid_diff = 0.0
    k_map_agree = 0

    for rep, per_prior in enumerate(results_per_dataset):
        bm_u = per_prior["uniform"]["bm"]
        bm_gh = per_prior["gap_hazard"]["bm"]
        diff_bm = float(np.max(np.abs(bm_u - bm_gh)))
        max_bm_diff = max(max_bm_diff, diff_bm)

        le_u = per_prior["uniform"]["log_evidence"]
        le_gh = per_prior["gap_hazard"]["log_evidence"]
        diff_le = abs(le_u - le_gh)
        max_logevid_diff = max(max_logevid_diff, diff_le)

        if per_prior["uniform"]["k_map"] == per_prior["gap_hazard"]["k_map"]:
            k_map_agree += 1

        logger.info(
            "  Rep %2d: bm_diff=%.2e  logevid_diff=%.2e  k_map uniform=%d gap_hazard=%d",
            rep,
            diff_bm,
            diff_le,
            per_prior["uniform"]["k_map"],
            per_prior["gap_hazard"]["k_map"],
        )

    logger.info("")
    logger.info("Summary (gap_hazard vs uniform, same datasets, G=%d):", gap)
    logger.info("  Max boundary marginal absolute diff : %.3e", max_bm_diff)
    logger.info("  Max log_evidence absolute diff      : %.3e", max_logevid_diff)
    logger.info("  k_map agreement rate                : %d/%d", k_map_agree, n_datasets)

    return {
        "gap": gap,
        "max_bm_diff": max_bm_diff,
        "max_logevid_diff": max_logevid_diff,
        "k_map_agree": k_map_agree,
        "n_datasets": n_datasets,
    }


# ---------------------------------------------------------------------------
# Verify the prior table values directly
# ---------------------------------------------------------------------------

def verify_prior_table_equality(gap: int) -> dict:
    """Check that gap_hazard and uniform produce identical prior tables at G."""
    from bayesbreak.design_prior import build_log_prior_table

    n = E03_CONFIG["n"]
    gap_start_idx = E03_CONFIG["gap_start_idx"]
    x = _build_observation_coords(n, gap, gap_start_idx)
    u = _build_boundary_coords(x)

    cfg_uniform = _build_prior_config("uniform", x, u)
    cfg_gh = _build_prior_config("gap_hazard", x, u)

    table_u = build_log_prior_table(n, u, cfg_uniform)
    table_gh = build_log_prior_table(n, u, cfg_gh)

    # For uniform: all finite entries are 0.0
    # For gap_hazard: each entry table[i,j] = log(h_j) where h_j = x[j]-x[j-1]
    # When G=1, all h_j = 1, so log(h_j) = 0.0  => identical tables

    finite_mask = np.isfinite(table_u) & np.isfinite(table_gh)
    if finite_mask.sum() == 0:
        logger.warning("No finite table entries found!")
        return {"gap": gap, "max_table_diff": float("nan"), "n_finite": 0}

    diff = np.abs(table_u[finite_mask] - table_gh[finite_mask])
    max_diff = float(np.max(diff))
    n_finite = int(finite_mask.sum())

    logger.info(
        "Prior table comparison (gap=%d): max |table_uniform - table_gap_hazard| = %.3e  "
        "(over %d finite entries)",
        gap, max_diff, n_finite,
    )

    # Also check the actual table values
    finite_vals_u = table_u[finite_mask]
    finite_vals_gh = table_gh[finite_mask]
    logger.info(
        "  Uniform table finite values: min=%.6f, max=%.6f",
        float(np.min(finite_vals_u)), float(np.max(finite_vals_u)),
    )
    logger.info(
        "  Gap-hazard table finite values: min=%.6f, max=%.6f",
        float(np.min(finite_vals_gh)), float(np.max(finite_vals_gh)),
    )

    return {"gap": gap, "max_table_diff": max_diff, "n_finite": n_finite}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("E03 Unit-gap Degeneracy Diagnostic (diag_e03_degeneracy.py)")
    logger.info("=" * 70)

    # Part 1: Mathematical verification
    logger.info("\n--- Part 1: Mathematical argument (G=1) ---")
    math_result_g1 = verify_gap_hazard_uniform_equivalence_math(gap=1)
    logger.info("\n--- Part 1: Mathematical argument (G=4 for comparison) ---")
    math_result_g4 = verify_gap_hazard_uniform_equivalence_math(gap=4)
    logger.info("\n--- Part 1: Mathematical argument (G=16 for comparison) ---")
    math_result_g16 = verify_gap_hazard_uniform_equivalence_math(gap=16)

    # Part 2: Prior table equality
    logger.info("\n--- Part 2: Prior table equality at G=1 ---")
    table_result_g1 = verify_prior_table_equality(gap=1)
    logger.info("\n--- Part 2: Prior table equality at G=4 (should differ) ---")
    table_result_g4 = verify_prior_table_equality(gap=4)

    # Part 3: Paired dataset comparison at G=1
    logger.info("\n--- Part 3: Paired comparison at G=1, template=[60,120] ---")
    paired_g1 = run_paired_comparison(gap=1, true_template=[60, 120], n_datasets=10)

    # Part 4: Summary and conclusion
    logger.info("")
    logger.info("=" * 70)
    logger.info("CONCLUSION")
    logger.info("=" * 70)

    g1_eq = math_result_g1["all_hazards_equal"]
    g1_table_diff = table_result_g1["max_table_diff"]
    g1_paired_bm_diff = paired_g1["max_bm_diff"]
    g1_paired_le_diff = paired_g1["max_logevid_diff"]

    logger.info("G=1 all hazard values equal:           %s", g1_eq)
    logger.info("G=1 max prior table diff (u vs gh):    %.3e", g1_table_diff)
    logger.info("G=1 paired max BM diff (u vs gh):      %.3e", g1_paired_bm_diff)
    logger.info("G=1 paired max log_ev diff (u vs gh):  %.3e", g1_paired_le_diff)

    g4_eq = math_result_g4["all_hazards_equal"]
    g4_table_diff = table_result_g4["max_table_diff"]
    logger.info("G=4 all hazard values equal:           %s  (expected: False)", g4_eq)
    logger.info("G=4 max prior table diff (u vs gh):    %.3e  (expected: large)", g4_table_diff)

    if g1_eq and g1_table_diff < 1e-12 and g1_paired_bm_diff < 1e-12:
        logger.info(
            "\nVERDICT: gap_hazard IS NUMERICALLY IDENTICAL to uniform at G=1."
        )
        logger.info(
            "The observed F1/P(b_gap) differences in the E03 confirmatory run "
            "are SAMPLING VARIABILITY from different independent datasets "
            "(unpaired design: each cell used a different cell_idx in seed)."
        )
        logger.info(
            "The E03 report's claim 'gap_hazard ≈ uniform at G=1' is CORRECT "
            "but should be understood as a population-level statement; "
            "the finite-sample comparison was unpaired."
        )
    else:
        logger.warning(
            "\nVERDICT: gap_hazard is NOT identical to uniform at G=1. "
            "Max BM diff = %.3e, table diff = %.3e",
            g1_paired_bm_diff, g1_table_diff,
        )


if __name__ == "__main__":
    main()
