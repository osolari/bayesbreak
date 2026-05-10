"""DP-layer correctness: brute-force equivalence + joint MAP vs marginal-topk.

These are **conceptual** tests, not smoke tests: they verify the DP layer
computes the mathematically correct quantity by brute-enumerating all
admissible partitions on small problem sizes.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from bayesbreak import dp
from bayesbreak.utils import logsumexp


def brute_force_log_evidence(lA0: np.ndarray, n: int, k: int) -> float:
    """``log sum_t prod_q A0_{t_{q-1}, t_q}`` enumerated over all k-partitions."""

    if k == 1:
        return float(lA0[0, n])
    totals: list[float] = []
    for bps in itertools.combinations(range(1, n), k - 1):
        t = (0, *bps, n)
        s = 0.0
        ok = True
        for a, b in zip(t[:-1], t[1:], strict=False):
            v = float(lA0[a, b])
            if not np.isfinite(v):
                ok = False
                break
            s += v
        if ok:
            totals.append(s)
    return float(logsumexp(np.array(totals))) if totals else -np.inf


def brute_force_map(lA0: np.ndarray, n: int, k: int) -> tuple[tuple[int, ...], float]:
    """Joint MAP boundary vector by brute enumeration."""

    best = (None, -np.inf)
    iterator = [(0,)] if k == 1 else itertools.combinations(range(1, n), k - 1)
    for bps in iterator:
        t = (0, *(bps if k > 1 else ()), n)
        s = sum(float(lA0[a, b]) for a, b in zip(t[:-1], t[1:], strict=False))
        if s > best[1]:
            best = (t, s)
    return best


@pytest.fixture
def random_log_block_evidence():
    rng = np.random.default_rng(0)
    n = 7
    la = np.full((n + 1, n + 1), -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            la[i, j] = float(rng.normal(loc=-0.1, scale=0.5))
    return la, n


class TestSumProduct:
    def test_forward_matches_brute_force(self, random_log_block_evidence):
        la, n = random_log_block_evidence
        k_max = 4
        L, _ = dp.forward_backward(la, n=n, k_max=k_max)
        for k in range(1, k_max + 1):
            assert float(L[k, n]) == pytest.approx(brute_force_log_evidence(la, n, k), abs=1e-10)

    def test_backward_sanity(self, random_log_block_evidence):
        la, n = random_log_block_evidence
        _, R = dp.forward_backward(la, n=n, k_max=4)
        # R[1, i] is log p(y_{i+1:n} | single segment)
        for i in range(n):
            assert float(R[1, i]) == pytest.approx(float(la[i, n]), abs=1e-12)

    def test_posterior_k_sums_to_one(self, random_log_block_evidence):
        la, n = random_log_block_evidence
        L, _ = dp.forward_backward(la, n=n, k_max=4)
        _, post_k, _ = dp.posterior_over_k(L, n=n, k_max=4)
        assert float(np.sum(post_k)) == pytest.approx(1.0, abs=1e-10)
        assert np.all(post_k >= -1e-12)

    def test_boundary_event_marginal_bounds(self, random_log_block_evidence):
        la, n = random_log_block_evidence
        k_max = 4
        L, R = dp.forward_backward(la, n=n, k_max=k_max)
        for k in range(2, k_max + 1):
            d1 = dp.boundary_event_marginals_fixed_k(L, R, n=n, k=k)
            assert d1.shape == (n - 1,)
            assert np.all((d1 >= -1e-12) & (d1 <= 1.0 + 1e-12))
            # Sums to k - 1 by construction.
            assert float(np.sum(d1)) == pytest.approx(k - 1, abs=1e-10)


class TestMaxSum:
    def test_map_matches_brute_force(self, random_log_block_evidence):
        la, n = random_log_block_evidence
        for k in range(1, 5):
            bnd, score = dp.max_sum_segmentation(la, k=k)
            expected_t, expected_score = brute_force_map(la, n, k)
            assert tuple(bnd) == expected_t
            assert score == pytest.approx(expected_score, abs=1e-12)

    def test_map_distinct_from_marginal_topk(self):
        """Joint-MAP-vs-marginal-mode counterexample from §`rem:marg-vs-joint`.

        For ``n = 5`` and ``k = 3``, place the joint posterior on four ordered
        configurations with weights ``0.30, 0.28, 0.22, 0.20`` on
        ``(1,4),(2,3),(2,4),(3,4)``. The marginal modes are then ``t1=2`` and
        ``t2=4``, so the marginal-mode vector ``(2,4)`` is feasible but is *not*
        the joint MAP ``(1,4)``.
        """

        n = 5
        la = np.full((n + 1, n + 1), -np.inf)
        # Used blocks (path-specific assignment so that the four target paths
        # have log-scores log(0.30), log(0.28), log(0.22), log(0.20)).
        la[0, 1] = 0.0
        la[0, 2] = 0.0
        la[0, 3] = 0.0
        la[1, 4] = math.log(0.30) - math.log(0.20)  # 0.4054651
        la[2, 3] = 0.0
        la[2, 4] = math.log(0.22) - math.log(0.20)  # 0.0953102
        la[3, 4] = 0.0
        la[3, 5] = math.log(0.28)  # -1.2729657
        la[4, 5] = math.log(0.20)  # -1.6094379

        # Joint MAP under k=3.
        bnd, score = dp.max_sum_segmentation(la, k=3)
        assert bnd == [0, 1, 4, 5]
        assert score == pytest.approx(math.log(0.30), abs=1e-9)

        # Marginal modes diverge from the joint MAP.
        L, R = dp.forward_backward(la, n=n, k_max=3)
        bp = dp.boundary_location_posterior(L, R, n=n, k=3)
        assert int(np.argmax(bp[0])) == 2  # marginal mode of t1
        assert int(np.argmax(bp[1])) == 4  # marginal mode of t2

        # Boundary-event marginals are also consistent with the constructed
        # joint posterior: index 4 has the largest event probability (~0.72).
        d1 = dp.boundary_event_marginals_fixed_k(L, R, n=n, k=3)
        assert int(np.argmax(d1)) == 3  # 0-indexed: position 4 → index 3

    def test_invalid_k_raises(self, random_log_block_evidence):
        la, n = random_log_block_evidence
        with pytest.raises(ValueError):
            dp.max_sum_segmentation(la, k=0)
        with pytest.raises(ValueError):
            dp.max_sum_segmentation(la, k=n + 1)


class TestBayesCurve:
    def test_fixed_k_gaussian_matches_segment_mean(self):
        """For a 1-segment Gaussian block, the curve is the constant posterior mean."""

        from bayesbreak import BayesBreakGaussian

        rng = np.random.default_rng(0)
        n = 30
        y = rng.normal(5.0, 0.1, size=n)
        est = BayesBreakGaussian(k_max=1, regression_curve="fixed_k").fit(
            np.arange(n).reshape(-1, 1), y
        )
        assert est.bayes_curve_mean_ is not None
        assert np.std(est.bayes_curve_mean_) < 1e-8
        assert np.mean(est.bayes_curve_mean_) == pytest.approx(est.map_segment_means_[0], abs=1e-6)
