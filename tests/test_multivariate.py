"""Comprehensive tests for BayesBreak multivariate functionality."""

from __future__ import annotations

import numpy as np

from bayesbreak import BayesBreakGaussian, BayesBreakMultivariate, BayesBreakPoisson


class TestMultivariateBasic:
    """Basic multivariate segmentation tests."""

    def test_multivariate_gaussian_fit(self):
        """Test basic multivariate Gaussian segmentation."""
        rng = np.random.default_rng(42)
        n = 100
        n_channels = 3

        # Generate data with shared boundaries
        Y = np.zeros((n, n_channels))
        for ch in range(n_channels):
            signal = np.r_[np.zeros(40), np.ones(30) * (ch + 1), np.zeros(30)]
            Y[:, ch] = signal + 0.2 * rng.standard_normal(n)

        base = BayesBreakGaussian(k_max=10)
        mv = BayesBreakMultivariate(base)
        mv.fit(Y)

        # Check that boundaries were detected
        boundaries = mv.get_boundaries()
        assert len(boundaries) >= 3  # Including 0 and n
        assert boundaries[0] == 0
        assert boundaries[-1] == n

    def test_multivariate_predict_shape(self):
        """Test that predict returns correct shape."""
        rng = np.random.default_rng(123)
        n, d = 80, 4
        Y = rng.standard_normal((n, d))

        mv = BayesBreakMultivariate(BayesBreakGaussian(k_max=5))
        mv.fit(Y)

        Y_pred = mv.predict()
        assert Y_pred.shape == (n, d)

    def test_multivariate_with_poisson(self):
        """Test multivariate with Poisson family."""
        rng = np.random.default_rng(456)
        n = 60
        n_channels = 2

        Y = np.zeros((n, n_channels), dtype=float)
        for ch in range(n_channels):
            rates = np.r_[np.ones(30) * 2, np.ones(30) * 8]
            Y[:, ch] = rng.poisson(rates)

        mv = BayesBreakMultivariate(BayesBreakPoisson(k_max=8))
        mv.fit(Y)

        # Should fit successfully
        assert mv.k_ml_ is not None
        assert mv.k_ml_ >= 1


class TestMultivariateEdgeCases:
    """Edge case tests for multivariate."""

    def test_single_channel(self):
        """Multivariate with single channel should work like univariate."""
        rng = np.random.default_rng(789)
        n = 50
        y = np.r_[np.zeros(25), np.ones(25)] + 0.1 * rng.standard_normal(n)
        Y = y.reshape(-1, 1)

        # Univariate
        uni = BayesBreakGaussian(k_max=5)
        uni.fit(y)

        # Multivariate
        mv = BayesBreakMultivariate(BayesBreakGaussian(k_max=5))
        mv.fit(Y)

        # Should get same or similar segment count
        assert abs(uni.get_segment_count() - mv.k_ml_) <= 1


class TestMultivariateScore:
    """Score computation tests."""

    def test_log_evidence_finite(self):
        """Log evidence should be finite."""
        rng = np.random.default_rng(111)
        Y = rng.standard_normal((40, 2))

        mv = BayesBreakMultivariate(BayesBreakGaussian(k_max=5))
        mv.fit(Y)

        score = mv.score()
        assert np.isfinite(score)
