"""Tests for utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak.utils import (
    as_1d_float_array,
    check_sample_weight,
    gammaln,
    log_binom,
    logsumexp,
)


class TestLogsumexp:
    """Tests for logsumexp function."""

    def test_logsumexp_basic(self):
        """Basic logsumexp computation."""
        a = np.array([1.0, 2.0, 3.0])
        result = logsumexp(a)
        expected = np.log(np.exp(1.0) + np.exp(2.0) + np.exp(3.0))
        assert np.isclose(result, expected)

    def test_logsumexp_axis(self):
        """logsumexp along an axis."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp(a, axis=1)
        assert result.shape == (2,)

    def test_logsumexp_with_neginf(self):
        """logsumexp should handle -inf correctly."""
        a = np.array([0.0, -np.inf, 1.0])
        result = logsumexp(a)
        expected = np.log(np.exp(0.0) + 0.0 + np.exp(1.0))
        assert np.isclose(result, expected)

    def test_logsumexp_keepdims(self):
        """logsumexp with keepdims=True."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp(a, axis=1, keepdims=True)
        assert result.shape == (2, 1)


class TestLogBinom:
    """Tests for log_binom function."""

    def test_log_binom_basic(self):
        """Basic log binomial coefficient."""
        # C(5, 2) = 10
        result = log_binom(5, 2)
        assert np.isclose(result, np.log(10))

    def test_log_binom_edge_cases(self):
        """Edge cases for log_binom."""
        # C(n, 0) = 1
        assert np.isclose(log_binom(10, 0), 0.0)
        # C(n, n) = 1
        assert np.isclose(log_binom(10, 10), 0.0)


class TestGammaln:
    """Tests for gammaln function."""

    def test_gammaln_integers(self):
        """gammaln of integers should give log((n-1)!)."""
        # log(Gamma(5)) = log(4!) = log(24)
        result = gammaln(5)
        assert np.isclose(result, np.log(24))

    def test_gammaln_array(self):
        """gammaln should work with arrays."""
        a = np.array([1.0, 2.0, 3.0])
        result = gammaln(a)
        assert result.shape == (3,)


class TestAs1dFloatArray:
    """Tests for as_1d_float_array helper."""

    def test_converts_list(self):
        """Should convert list to numpy array."""
        result = as_1d_float_array([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_raises_on_2d(self):
        """Should raise on 2D array."""
        arr = np.array([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError):
            as_1d_float_array(arr)

    def test_raises_on_2d_matrix(self):
        """Should raise on non-squeezable 2D array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError):
            as_1d_float_array(arr)


class TestCheckSampleWeight:
    """Tests for check_sample_weight helper."""

    def test_none_returns_ones(self):
        """None sample_weight should return array of ones."""
        result = check_sample_weight(None, 5)
        assert np.allclose(result, np.ones(5))

    def test_validates_length(self):
        """Should raise if length doesn't match n."""
        with pytest.raises(ValueError):
            check_sample_weight(np.ones(3), 5)

    def test_validates_non_negative(self):
        """Should raise if weights are negative."""
        with pytest.raises(ValueError):
            check_sample_weight(np.array([1.0, -1.0, 1.0]), 3)


class TestRequireFitted:
    """Tests for require_fitted decorator."""

    def test_raises_before_fit(self):
        """Should raise RuntimeError if not fitted."""
        from bayesbreak import BayesBreakGaussian

        model = BayesBreakGaussian()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict()
