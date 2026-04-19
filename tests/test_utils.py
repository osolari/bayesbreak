"""Tests for pure-math helpers in :mod:`bayesbreak.utils`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayesbreak.utils import gammaln, log_binom, logsumexp


class TestLogsumexp:
    def test_matches_manual(self):
        a = np.array([0.0, math.log(2.0), math.log(3.0)])
        assert float(logsumexp(a)) == pytest.approx(math.log(6.0))

    def test_all_minus_inf_returns_minus_inf(self):
        assert float(logsumexp(np.array([-np.inf, -np.inf]))) == -np.inf

    def test_axis(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = logsumexp(a, axis=1)
        assert out.shape == (2,)
        assert np.allclose(out, [np.log(np.exp(1) + np.exp(2)), np.log(np.exp(3) + np.exp(4))])

    def test_keepdims(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert logsumexp(a, axis=1, keepdims=True).shape == (2, 1)

    def test_neg_inf_as_zero_contribution(self):
        a = np.array([0.0, -np.inf, 1.0])
        expected = np.log(np.exp(0.0) + np.exp(1.0))
        assert float(logsumexp(a)) == pytest.approx(expected)


class TestLogBinom:
    @pytest.mark.parametrize("n,k,expected", [(5, 2, np.log(10)), (10, 0, 0.0), (10, 10, 0.0)])
    def test_known_values(self, n, k, expected):
        assert log_binom(n, k) == pytest.approx(expected)

    def test_symmetry(self):
        for n in [1, 2, 5, 10]:
            for k in range(n + 1):
                assert log_binom(n, k) == pytest.approx(log_binom(n, n - k))

    def test_out_of_range_returns_neg_inf(self):
        assert log_binom(5, -1) == -np.inf
        assert log_binom(5, 6) == -np.inf


class TestGammaln:
    def test_integer_identity(self):
        assert gammaln(5) == pytest.approx(math.log(math.factorial(4)))

    def test_array(self):
        assert gammaln(np.array([1.0, 2.0, 3.0])).shape == (3,)
