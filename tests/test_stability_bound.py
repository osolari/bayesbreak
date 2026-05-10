"""Non-conjugate stability bound (Proposition ``stability``).

Verify that perturbing every block log-evidence by a uniform amount ``ε``
shifts the segment-count log-odds by at most ``(k + k') ε`` and the
boundary-event log-odds by at most ``2 k ε``.
"""

from __future__ import annotations

import numpy as np

from bayesbreak.dp import (
    boundary_event_marginals_fixed_k,
    forward_backward,
    posterior_over_k,
)


def test_k_odds_within_bound():
    rng = np.random.default_rng(0)
    n = 12
    k_max = 5
    eps = 0.04
    la = np.full((n + 1, n + 1), -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            la[i, j] = float(rng.normal(loc=-0.1, scale=0.4))
    delta = rng.uniform(-eps, eps, size=la.shape)
    la_pert = la + delta * (la > -np.inf)

    L0, _ = forward_backward(la, n=n, k_max=k_max)
    L1, _ = forward_backward(la_pert, n=n, k_max=k_max)
    log_pk0, _, _ = posterior_over_k(L0, n=n, k_max=k_max)
    log_pk1, _, _ = posterior_over_k(L1, n=n, k_max=k_max)

    for k in range(1, k_max + 1):
        for kp in range(1, k_max + 1):
            if k == kp:
                continue
            exact = log_pk0[k - 1] - log_pk0[kp - 1]
            approx = log_pk1[k - 1] - log_pk1[kp - 1]
            bound = (k + kp) * eps
            # Allow a small numerical slack on top of the (k+k')ε bound.
            assert abs(approx - exact) <= bound + 1e-9


def test_boundary_odds_within_bound():
    rng = np.random.default_rng(1)
    n = 14
    k = 4
    eps = 0.03
    la = np.full((n + 1, n + 1), -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            la[i, j] = float(rng.normal(loc=-0.1, scale=0.4))
    delta = rng.uniform(-eps, eps, size=la.shape)
    la_pert = la + delta * (la > -np.inf)

    L0, R0 = forward_backward(la, n=n, k_max=k)
    L1, R1 = forward_backward(la_pert, n=n, k_max=k)
    bm0 = boundary_event_marginals_fixed_k(L0, R0, n=n, k=k)
    bm1 = boundary_event_marginals_fixed_k(L1, R1, n=n, k=k)

    bound = 2 * k * eps + 1e-9
    # Compare log-odds between any pair of interior indices.
    eps_clip = 1e-12
    log_bm0 = np.log(np.clip(bm0, eps_clip, 1.0)) - np.log(np.clip(1 - bm0, eps_clip, 1.0))
    log_bm1 = np.log(np.clip(bm1, eps_clip, 1.0)) - np.log(np.clip(1 - bm1, eps_clip, 1.0))
    deltas = []
    for i in range(n - 1):
        for j in range(n - 1):
            if i == j:
                continue
            exact = log_bm0[i] - log_bm0[j]
            approx = log_bm1[i] - log_bm1[j]
            deltas.append(abs(approx - exact))
    assert max(deltas) <= bound
