"""Tests for the multivariate wrappers (shared / independent)."""

from __future__ import annotations

import numpy as np

from bayesbreak import (
    BayesBreakGaussian,
    IndependentMultivariateSegmenter,
    SharedBoundaryMultivariateSegmenter,
)


class TestSharedBoundary:
    def test_shape_and_shared_boundaries(self, multivariate_data):
        X, Y, true_b = multivariate_data
        est = SharedBoundaryMultivariateSegmenter(BayesBreakGaussian(), k_max=8).fit(X, Y)
        assert est.map_curve_.shape == Y.shape
        assert est.map_segment_means_.shape == (est.k_map_, Y.shape[1])
        # Same MAP boundaries apply to all channels.
        for b in true_b[1:-1]:
            assert any(abs(bb - b) <= 3 for bb in est.map_boundaries_)

    def test_boundaries_stable_across_runs(self, multivariate_data):
        X, Y, _ = multivariate_data
        b1 = (
            SharedBoundaryMultivariateSegmenter(BayesBreakGaussian(), k_max=6)
            .fit(X, Y)
            .map_boundaries_
        )
        b2 = (
            SharedBoundaryMultivariateSegmenter(BayesBreakGaussian(), k_max=6)
            .fit(X, Y)
            .map_boundaries_
        )
        assert b1 == b2

    def test_predict_shape_on_new_points(self, multivariate_data):
        X, Y, _ = multivariate_data
        est = SharedBoundaryMultivariateSegmenter(BayesBreakGaussian(), k_max=6).fit(X, Y)
        X_new = X[::10]
        out = est.predict(X_new)
        assert out.shape == (X_new.shape[0], Y.shape[1])


class TestIndependent:
    def test_per_channel_independence(self, multivariate_data):
        X, Y, _ = multivariate_data
        est = IndependentMultivariateSegmenter(BayesBreakGaussian(), k_max=8).fit(X, Y)
        assert len(est.channel_estimators_) == Y.shape[1]
        assert est.map_curve_.shape == Y.shape

    def test_log_evidence_sums_channel_evidences(self, multivariate_data):
        X, Y, _ = multivariate_data
        est = IndependentMultivariateSegmenter(BayesBreakGaussian(), k_max=5).fit(X, Y)
        total = sum(float(e.log_evidence_) for e in est.channel_estimators_)
        assert np.isclose(est.log_evidence_, total, rtol=1e-10)
