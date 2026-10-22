"""Property-based invariants for the DP engine.

These tests use Hypothesis to sample random small-``n`` data and verify
DP invariants that must hold for any admissible log-block-evidence
matrix:

- ``Σ_k P(k|y) = 1`` and ``log p(y)`` is finite;
- forward/backward duality ``L[k_map, n] = R[k_map, 0]``;
- ``Σ_i P(b_i = 1 | y, k) = k − 1`` for every ``k`` with positive support;
- max-sum log-joint ≥ marginal-mode segmentation log-joint at fixed ``k``;
- for ``n ≤ 8``, exhaustive enumeration reproduces sum-product's log evidence
  to machine precision.

The tests target the DP engine directly (``bayesbreak.dp``) so failures
localise to that module regardless of the family the block scores came
from.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bayesbreak import dp as _dp


def _sum_product_partition_enum(log_block_evidence: np.ndarray, k: int) -> float:
    """Brute-force ``log Σ_t ∏_q A^{(0)}_{t_{q-1} t_q}`` over all ``k``-segmentations."""

    n = log_block_evidence.shape[0] - 1
    if k < 1 or k > n:
        return -np.inf
    scores = []
    interior_positions = list(range(1, n))
    for cut_set in combinations(interior_positions, k - 1):
        cuts = [0, *cut_set, n]
        s = 0.0
        for i, j in zip(cuts[:-1], cuts[1:], strict=True):
            block = log_block_evidence[i, j]
            if not np.isfinite(block):
                s = -np.inf
                break
            s += block
        if np.isfinite(s):
            scores.append(s)
    if not scores:
        return -np.inf
    arr = np.array(scores)
    m = float(np.max(arr))
    return float(m + np.log(np.sum(np.exp(arr - m))))


def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    import math

    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _random_finite_score_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Full-support triangular log-block-evidence matrix (all upper-tri finite)."""

    mat = np.full((n + 1, n + 1), -np.inf, dtype=float)
    tri = np.triu_indices(n + 1, k=1)
    mat[tri] = rng.normal(loc=0.0, scale=1.0, size=len(tri[0]))
    return mat


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    n=st.integers(min_value=2, max_value=8),
    k_max=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_log_evidence_matches_exhaustive_enumeration(n: int, k_max: int, seed: int) -> None:
    """For ``n ≤ 8`` the sum-product ``log p(y)`` under a uniform ``p(k)`` and
    the flat length prior must equal the log-partition of the exhaustive
    enumeration."""

    k_max = min(k_max, n)
    rng = np.random.default_rng(seed)
    score = _random_finite_score_matrix(rng, n)

    left, _ = _dp.forward_backward(score, n, k_max)
    log_C_k = _dp.compute_log_C_k(None, n, k_max)
    _, _, log_evidence = _dp.posterior_over_k(left, n, k_max)

    # Direct enumeration: log p(y) = logsumexp_k [log p(k) + log Z_k - log C_k]
    # with p(k) uniform on {1, ..., k_max}.
    log_p_k = -np.log(float(k_max))
    logs = []
    for k in range(1, k_max + 1):
        log_Zk = _sum_product_partition_enum(score, k)
        if np.isfinite(log_Zk) and np.isfinite(log_C_k[k]):
            logs.append(log_p_k + log_Zk - log_C_k[k])
    assert logs, "no admissible k in enumeration path"
    arr = np.array(logs)
    m = float(np.max(arr))
    log_evidence_enum = float(m + np.log(np.sum(np.exp(arr - m))))
    assert log_evidence == pytest.approx(log_evidence_enum, rel=0, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=14),
    k_max=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_posterior_over_k_sums_to_one(n: int, k_max: int, seed: int) -> None:
    k_max = min(k_max, n)
    rng = np.random.default_rng(seed)
    score = _random_finite_score_matrix(rng, n)
    left, _ = _dp.forward_backward(score, n, k_max)
    _, pk, _ = _dp.posterior_over_k(left, n, k_max)
    assert np.isfinite(pk).all()
    assert float(pk.sum()) == pytest.approx(1.0, abs=1e-10)


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=14),
    k=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_boundary_event_marginals_sum_to_k_minus_one(n: int, k: int, seed: int) -> None:
    """``Σ_i P(b_i = 1 | y, k) = k − 1`` — the Corollary boundary-event
    normalisation identity."""

    k = min(k, n)
    if k < 2:
        return
    rng = np.random.default_rng(seed)
    score = _random_finite_score_matrix(rng, n)
    left, right = _dp.forward_backward(score, n, k)
    marginals = _dp.boundary_event_marginals_fixed_k(left, right, n, k)
    assert marginals.shape == (n - 1,)
    assert float(marginals.sum()) == pytest.approx(k - 1, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=12),
    k=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_forward_backward_duality(n: int, k: int, seed: int) -> None:
    """``L[k, n] == R[k, 0]`` — sum-product duality identity
    ``prop:fb-duality``."""

    k = min(k, n)
    rng = np.random.default_rng(seed)
    score = _random_finite_score_matrix(rng, n)
    left, right = _dp.forward_backward(score, n, k)
    ln = float(left[k, n])
    r0 = float(right[k, 0])
    assert ln == pytest.approx(r0, rel=0, abs=1e-9)


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=10),
    k=st.integers(min_value=2, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_max_sum_score_matches_exhaustive(n: int, k: int, seed: int) -> None:
    """Max-sum joint MAP matches the maximum over all ``k``-segmentations."""

    k = min(k, n)
    if k < 1:
        return
    rng = np.random.default_rng(seed)
    score = _random_finite_score_matrix(rng, n)
    boundaries, log_joint = _dp.max_sum_segmentation(score, k)

    # Verify score returned matches the sum along the recovered path.
    path_score = float(
        sum(score[i, j] for i, j in zip(boundaries[:-1], boundaries[1:], strict=True))
    )
    assert log_joint == pytest.approx(path_score, abs=1e-9)

    # Exhaustive maximum.
    best = -np.inf
    for cut_set in combinations(range(1, n), k - 1):
        cuts = [0, *cut_set, n]
        s = 0.0
        for i, j in zip(cuts[:-1], cuts[1:], strict=True):
            s += float(score[i, j])
        if s > best:
            best = s
    assert log_joint == pytest.approx(best, abs=1e-9)
