"""Diagnostic B1: independently verify the irregular design prior for E03.

Constructs a small example with 5 observations at coordinates [0.0, 1.2, 3.5, 3.7, 10.0],
k_max=3. Enumerates all admissible partitions (k=1,2,3) and:
1. Computes the prior probability of each partition using the production code
2. Computes manually from the prior definition
3. Verifies they match
4. Verifies the sum-product DP gives the same result as brute force

For each of the four E03 prior settings:
A) uniform local factors
B) gap_hazard: h_u(i) = u_i+1 - u_i (gap to next candidate coordinate)
C) length_cohesion: c_u(i,j) = (u_j - u_i) / median_gap
D) combined: product of B and C

Usage:
    uv run python scripts/diag_e03_prior.py
"""

from __future__ import annotations

import logging
import math
import sys
from itertools import pairwise
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

from bayesbreak.priors import PartitionPriorConfig, partition_log_prior  # noqa: E402
from bayesbreak.design_prior import build_log_prior_table  # noqa: E402

# ---------------------------------------------------------------------------
# Small example setup
# ---------------------------------------------------------------------------

# 5 observations at irregular coordinates
X_OBS = [0.0, 1.2, 3.5, 3.7, 10.0]
N = len(X_OBS)
K_MAX = 3


def _build_boundary_coords(x_obs: list[float]) -> list[float]:
    """Build n+1 candidate boundary coordinates from n observation coords.

    u_0 = x[0] - 0.5 * (x[1] - x[0])
    u_i = (x[i-1] + x[i]) / 2   for 1 <= i < n
    u_n = x[-1] + 0.5 * (x[-1] - x[-2])
    """
    n = len(x_obs)
    u = []
    first_gap = x_obs[1] - x_obs[0]
    last_gap = x_obs[-1] - x_obs[-2]
    u.append(x_obs[0] - 0.5 * first_gap)
    for i in range(1, n):
        u.append(0.5 * (x_obs[i - 1] + x_obs[i]))
    u.append(x_obs[-1] + 0.5 * last_gap)
    return u


U_COORDS = _build_boundary_coords(X_OBS)
logger.info("Observation coordinates: %s", X_OBS)
logger.info("Candidate boundary coordinates: %s", [f"{u:.4f}" for u in U_COORDS])


# ---------------------------------------------------------------------------
# Enumerate all admissible partitions
# ---------------------------------------------------------------------------

def _enumerate_partitions(n: int, k_max: int) -> list[list[int]]:
    """Enumerate all ordered partitions [0, i1, i2, ..., n] with k <= k_max segments."""
    partitions = []
    for k in range(1, k_max + 1):
        if k == 1:
            partitions.append([0, n])
            continue
        # k-1 interior boundaries chosen from 1..n-1 (strictly increasing)
        def _gen(start: int, remaining: int, prefix: list[int]):
            if remaining == 0:
                partitions.append(prefix + [n])
                return
            for b in range(start, n):
                _gen(b + 1, remaining - 1, prefix + [b])
        _gen(1, k - 1, [0])
    return partitions


ALL_PARTITIONS = _enumerate_partitions(N, K_MAX)
logger.info("Total admissible partitions (n=%d, k_max=%d): %d", N, K_MAX, len(ALL_PARTITIONS))


# ---------------------------------------------------------------------------
# Manual prior computation helpers
# ---------------------------------------------------------------------------

def _gaps_from_u(u: list[float]) -> list[float]:
    """Gap between consecutive boundary coordinates: g_i = u_{i+1} - u_i."""
    return [u[i + 1] - u[i] for i in range(len(u) - 1)]


U_GAPS = _gaps_from_u(U_COORDS)
MEDIAN_GAP = float(np.median(U_GAPS))
logger.info("Boundary-coordinate gaps: %s", [f"{g:.4f}" for g in U_GAPS])
logger.info("Median gap: %.4f", MEDIAN_GAP)


def _manual_log_prior_A(partition: list[int]) -> float:
    """Prior A: uniform — log prior = 0 for all admissible partitions."""
    return 0.0


def _manual_log_prior_B(partition: list[int]) -> float:
    """Prior B: gap hazard h_u(i) = u_{i+1} - u_i for each interior boundary.

    Log prior = sum over interior boundaries i of log(u_{i+1} - u_i).
    (Cohesion = 1 for all segments; no cohesion term.)
    """
    total = 0.0
    for idx in range(1, len(partition) - 1):
        b = partition[idx]  # interior boundary index in [1, n-1]
        # hazard factor for cutting at position b: h_u(b) = u_{b+1} - u_b = U_GAPS[b]
        gap = U_GAPS[b]
        if gap <= 0:
            return -math.inf
        total += math.log(gap)
    return total


def _manual_log_prior_C(partition: list[int]) -> float:
    """Prior C: length cohesion c_u(i,j) = (u_j - u_i) / median_gap.

    Log prior = sum over segments (i,j) of log((u_j - u_i) / median_gap).
    (Hazard = 1 for all boundaries; no hazard term.)
    """
    total = 0.0
    for start, stop in pairwise(partition):
        span = U_COORDS[stop] - U_COORDS[start]
        if span <= 0:
            return -math.inf
        total += math.log(span / MEDIAN_GAP)
    return total


def _manual_log_prior_D(partition: list[int]) -> float:
    """Prior D: combined = B * C."""
    return _manual_log_prior_B(partition) + _manual_log_prior_C(partition)


def _normalize_log_priors(log_priors: list[float]) -> list[float]:
    """Normalize log prior values to probabilities."""
    max_lp = max(lp for lp in log_priors if math.isfinite(lp))
    raw = [math.exp(lp - max_lp) if math.isfinite(lp) else 0.0 for lp in log_priors]
    total = sum(raw)
    return [r / total for r in raw]


# ---------------------------------------------------------------------------
# Production-code prior computation using partition_log_prior
# ---------------------------------------------------------------------------

def _production_log_prior_A(partition: list[int]) -> float:
    """Production code: uniform."""
    config = PartitionPriorConfig(segment_cohesion="uniform", boundary_hazard="uniform")
    return partition_log_prior(partition, U_COORDS, config)


def _production_log_prior_B(partition: list[int]) -> float:
    """Production code: explicit gap hazard.

    h_u(i) = U_GAPS[i] for interior i. We pass as explicit hazard parameters.
    """
    params = {f"hazard_{i}": U_GAPS[i] for i in range(1, N)}
    config = PartitionPriorConfig(
        segment_cohesion="uniform",
        boundary_hazard="explicit",
        parameters=params,
    )
    return partition_log_prior(partition, U_COORDS, config)


def _production_log_prior_C(partition: list[int]) -> float:
    """Production code: span cohesion with scale = median_gap."""
    config = PartitionPriorConfig(
        segment_cohesion="span",
        boundary_hazard="uniform",
        parameters={"cohesion_scale": MEDIAN_GAP, "cohesion_power": 1.0},
    )
    return partition_log_prior(partition, U_COORDS, config)


def _production_log_prior_D(partition: list[int]) -> float:
    """Production code: combined (explicit gap hazard + span cohesion)."""
    params = {f"hazard_{i}": U_GAPS[i] for i in range(1, N)}
    params["cohesion_scale"] = MEDIAN_GAP
    params["cohesion_power"] = 1.0
    config = PartitionPriorConfig(
        segment_cohesion="span",
        boundary_hazard="explicit",
        parameters=params,
    )
    return partition_log_prior(partition, U_COORDS, config)


# ---------------------------------------------------------------------------
# DP normalizer via build_log_prior_table
# ---------------------------------------------------------------------------

def _dp_log_normalizer(config: PartitionPriorConfig, k_max: int) -> dict[int, float]:
    """Sum the prior table over all admissible partitions with k segments via DP.

    Returns dict {k: log C_k} computed by summing exp(table[i,j]) factors.
    This should match brute-force enumeration.
    """
    table = build_log_prior_table(N, U_COORDS, config)
    # C_k = sum over all k-partitions of product of local prior factors
    # Use forward DP: dp[k][j] = log sum of products over all k-segment partitions ending at j
    dp = np.full((k_max + 1, N + 1), -np.inf, dtype=float)
    dp[0, 0] = 0.0  # empty partition ends at 0
    for k in range(1, k_max + 1):
        for j in range(k, N + 1):
            # dp[k][j] = logsumexp over i < j of dp[k-1][i] + table[i, j]
            for i in range(k - 1, j):
                val = dp[k - 1, i] + table[i, j]
                if math.isfinite(val):
                    cur = dp[k, j]
                    if math.isfinite(cur):
                        dp[k, j] = max(cur, val) + math.log1p(math.exp(min(cur, val) - max(cur, val)))
                    else:
                        dp[k, j] = val
    return {k: float(dp[k, N]) for k in range(1, k_max + 1)}


# ---------------------------------------------------------------------------
# Verification: compare manual vs production for all prior settings
# ---------------------------------------------------------------------------

PRIORS = {
    "A_uniform": (_manual_log_prior_A, _production_log_prior_A),
    "B_gap_hazard": (_manual_log_prior_B, _production_log_prior_B),
    "C_length_cohesion": (_manual_log_prior_C, _production_log_prior_C),
    "D_combined": (_manual_log_prior_D, _production_log_prior_D),
}


def _verify_prior(label: str, manual_fn, production_fn) -> bool:
    logger.info("")
    logger.info("--- Prior %s ---", label)

    manual_log_priors = [manual_fn(p) for p in ALL_PARTITIONS]
    production_log_priors = [production_fn(p) for p in ALL_PARTITIONS]

    # Check log-scale agreement (up to an additive constant from normalization)
    # Compare finite differences
    finite_pairs = [
        (m, pr)
        for m, pr in zip(manual_log_priors, production_log_priors)
        if math.isfinite(m) and math.isfinite(pr)
    ]
    if not finite_pairs:
        logger.error("  No finite pairs found!")
        return False

    # The two should agree up to a common additive constant
    diffs = [pr - m for m, pr in finite_pairs]
    diff_range = max(diffs) - min(diffs)
    logger.info("  Manual vs production log-prior diff range (should be ~0): %.3e", diff_range)

    ok = diff_range < 1e-10
    if ok:
        logger.info("  PASS: log-prior values agree up to additive constant (range=%.2e)", diff_range)
    else:
        logger.error("  FAIL: log-prior values disagree (range=%.2e)", diff_range)
        for i, (p, m, pr) in enumerate(zip(ALL_PARTITIONS, manual_log_priors, production_log_priors)):
            logger.error("    partition=%s: manual=%.6f, production=%.6f", p, m, pr)

    # Verify normalized probabilities match
    manual_probs = _normalize_log_priors(manual_log_priors)
    production_probs = _normalize_log_priors(production_log_priors)
    max_prob_diff = max(abs(m - pr) for m, pr in zip(manual_probs, production_probs))
    logger.info("  Max normalized probability difference: %.3e", max_prob_diff)
    if max_prob_diff < 1e-12:
        logger.info("  PASS: normalized probabilities agree")
    else:
        logger.error("  FAIL: normalized probabilities differ by %.2e", max_prob_diff)
        ok = False

    # Show top-5 partitions by probability
    ranked = sorted(zip(manual_probs, ALL_PARTITIONS), reverse=True)[:5]
    logger.info("  Top-5 partitions by probability:")
    for prob, part in ranked:
        logger.info("    p=%.5f  partition=%s  (k=%d)", prob, part, len(part) - 1)

    return ok


def _verify_dp_normalizer(label: str, config: PartitionPriorConfig) -> bool:
    """Compare DP-computed C_k with brute-force enumeration for each k."""
    logger.info("  DP normalizer check for %s:", label)

    # Brute force C_k
    bf_ck = {}
    for k in range(1, K_MAX + 1):
        k_partitions = [p for p in ALL_PARTITIONS if len(p) - 1 == k]
        if not k_partitions:
            continue
        log_weights = []
        for p in k_partitions:
            lp = partition_log_prior(p, U_COORDS, config)
            log_weights.append(lp)
        # logsumexp
        finite_lw = [lw for lw in log_weights if math.isfinite(lw)]
        if not finite_lw:
            bf_ck[k] = -math.inf
        else:
            max_lw = max(finite_lw)
            bf_ck[k] = max_lw + math.log(sum(math.exp(lw - max_lw) for lw in finite_lw))

    dp_ck = _dp_log_normalizer(config, K_MAX)

    ok = True
    for k in range(1, K_MAX + 1):
        bf = bf_ck.get(k, -math.inf)
        dp = dp_ck.get(k, -math.inf)
        if not math.isfinite(bf) and not math.isfinite(dp):
            continue
        if not math.isfinite(bf) or not math.isfinite(dp):
            logger.error("    k=%d: bf=%s, dp=%s — mismatch in finiteness", k, bf, dp)
            ok = False
            continue
        diff = abs(bf - dp)
        status = "PASS" if diff < 1e-10 else "FAIL"
        logger.info("    k=%d: bf=%.6f, dp=%.6f, diff=%.2e [%s]", k, bf, dp, diff, status)
        if status == "FAIL":
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Run all checks
# ---------------------------------------------------------------------------

def run_verification() -> None:
    logger.info("=" * 70)
    logger.info("E03 Prior Diagnostic: small-case verification")
    logger.info("n=%d observations at x=%s", N, X_OBS)
    logger.info("k_max=%d, total partitions=%d", K_MAX, len(ALL_PARTITIONS))
    logger.info("=" * 70)

    all_pass = True

    # Check 1: manual vs production log-prior agreement
    logger.info("\n=== CHECK 1: Manual vs production log-prior agreement ===")
    config_map = {
        "A_uniform": PartitionPriorConfig(segment_cohesion="uniform", boundary_hazard="uniform"),
        "B_gap_hazard": PartitionPriorConfig(
            segment_cohesion="uniform",
            boundary_hazard="explicit",
            parameters={f"hazard_{i}": U_GAPS[i] for i in range(1, N)},
        ),
        "C_length_cohesion": PartitionPriorConfig(
            segment_cohesion="span",
            boundary_hazard="uniform",
            parameters={"cohesion_scale": MEDIAN_GAP, "cohesion_power": 1.0},
        ),
        "D_combined": PartitionPriorConfig(
            segment_cohesion="span",
            boundary_hazard="explicit",
            parameters={
                **{f"hazard_{i}": U_GAPS[i] for i in range(1, N)},
                "cohesion_scale": MEDIAN_GAP,
                "cohesion_power": 1.0,
            },
        ),
    }

    for label, (manual_fn, production_fn) in PRIORS.items():
        ok = _verify_prior(label, manual_fn, production_fn)
        if not ok:
            all_pass = False

    # Check 2: DP normalizer matches brute force
    logger.info("\n=== CHECK 2: DP normalizer vs brute-force C_k ===")
    for label, config in config_map.items():
        logger.info("\n--- Prior %s ---", label)
        ok = _verify_dp_normalizer(label, config)
        if not ok:
            all_pass = False

    # Check 3: All probabilities sum to 1
    logger.info("\n=== CHECK 3: Partition probabilities sum to 1 ===")
    for label, (manual_fn, _) in PRIORS.items():
        log_priors = [manual_fn(p) for p in ALL_PARTITIONS]
        probs = _normalize_log_priors(log_priors)
        total = sum(probs)
        ok = abs(total - 1.0) < 1e-12
        logger.info("  %s: sum(probs)=%.12f [%s]", label, total, "PASS" if ok else "FAIL")
        if not ok:
            all_pass = False

    logger.info("\n" + "=" * 70)
    if all_pass:
        logger.info("ALL CHECKS PASSED")
    else:
        logger.error("SOME CHECKS FAILED — see above")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_verification()
