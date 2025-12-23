from __future__ import annotations

import math

import numpy as np

from bayesbreak.utils import log_binom, logsumexp


def test_logsumexp_basic_matches_manual():
    a = np.array([0.0, math.log(2.0), math.log(3.0)])
    expected = math.log(1.0 + 2.0 + 3.0)
    got = float(logsumexp(a))
    assert abs(got - expected) < 1e-12


def test_logsumexp_all_minus_inf():
    a = np.array([-np.inf, -np.inf])
    got = float(logsumexp(a))
    assert got == -np.inf


def test_log_binom_symmetry():
    for n in [1, 2, 5, 10]:
        for k in range(n + 1):
            assert abs(log_binom(n, k) - log_binom(n, n - k)) < 1e-12
