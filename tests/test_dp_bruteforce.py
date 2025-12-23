from __future__ import annotations

import itertools

import numpy as np

from bayesbreak.base import BayesBreakBase
from bayesbreak.utils import logsumexp


def brute_force_L(lA0: np.ndarray, n: int, k: int) -> float:
    """Brute-force compute log evidence for exactly k segments over n points."""
    if k == 1:
        return float(lA0[0, n])

    # choose k-1 breakpoints among {1, ..., n-1}
    logs: list[float] = []
    for bps in itertools.combinations(range(1, n), k - 1):
        t = (0, *bps, n)
        total = 0.0
        ok = True
        for a, b in zip(t[:-1], t[1:]):
            val = lA0[a, b]
            if not np.isfinite(val):
                ok = False
                break
            total += float(val)
        if ok:
            logs.append(total)

    return float(logsumexp(np.array(logs))) if logs else -np.inf


def test_left_recursion_matches_bruteforce_small():
    rng = np.random.default_rng(0)
    n = 7
    k_max = 4

    lA0 = np.full((n + 1, n + 1), -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            lA0[i, j] = float(rng.normal(loc=-0.1, scale=0.5))

    L, R = BayesBreakBase._compute_left_right_recursions(lA0, n=n, k_max=k_max)

    for k in range(1, k_max + 1):
        got = float(L[k, n])
        expected = brute_force_L(lA0, n=n, k=k)
        assert abs(got - expected) < 1e-10

    # basic right recursion sanity: evidence for exactly 1 segment from i to n
    for i in range(n):
        assert abs(float(R[1, i]) - float(lA0[i, n])) < 1e-12
