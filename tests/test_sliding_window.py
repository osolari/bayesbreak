"""Smoke tests for the sliding-window decomposition (§5b "Computational
regime"). The decomposition is an approximation, not an exact extension of
``thm:dp-correctness``; the tests therefore verify (i) that the fallback
to the full DP is exact when the sequence fits in one window, (ii) that
boundaries near the centre of a long sequence are still recovered up to
a small tolerance, and (iii) that the stitched curve has the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian, SlidingWindowSegmenter


def test_sliding_window_single_window_matches_base_estimator():
    """When ``n <= window_size`` the wrapper is the base estimator."""
    rng = np.random.default_rng(0)
    n = 40
    y = np.r_[rng.normal(0.0, 0.2, n // 2), rng.normal(2.0, 0.2, n // 2)]
    X = np.arange(n).reshape(-1, 1)

    base = BayesBreakGaussian(k_max=4)
    sw = SlidingWindowSegmenter(base, window_size=200, overlap=0).fit(X, y)
    direct = BayesBreakGaussian(k_max=4).fit(X, y)

    assert sw.map_boundaries_ == list(direct.map_boundaries_)
    assert sw.log_evidence_ == pytest.approx(float(direct.log_evidence_), rel=1e-12)
    assert np.allclose(sw.map_curve_, direct.map_curve_)


def test_sliding_window_recovers_well_separated_boundary():
    """On a 200-point signal with a single large jump at index 100, the
    decomposition should recover a boundary within a few indices."""
    rng = np.random.default_rng(1)
    n = 200
    y = np.r_[rng.normal(-1.0, 0.2, 100), rng.normal(2.0, 0.2, 100)]
    X = np.arange(n).reshape(-1, 1)

    sw = SlidingWindowSegmenter(BayesBreakGaussian(k_max=3), window_size=80, overlap=20).fit(X, y)
    interior = [b for b in sw.map_boundaries_ if 0 < b < n]
    assert interior, f"no interior boundary recovered (map={sw.map_boundaries_})"
    closest = min(abs(b - 100) for b in interior)
    assert closest <= 8, f"recovered boundaries {interior} too far from 100"


def test_sliding_window_boundary_marginals_in_unit_interval():
    rng = np.random.default_rng(2)
    n = 120
    y = np.r_[rng.normal(0.0, 0.3, 60), rng.normal(1.5, 0.3, 60)]
    X = np.arange(n).reshape(-1, 1)
    sw = SlidingWindowSegmenter(BayesBreakGaussian(k_max=3), window_size=50, overlap=15).fit(X, y)
    bm = sw.boundary_marginals_
    assert bm.shape == (n - 1,)
    assert np.all((bm >= -1e-12) & (bm <= 1.0 + 1e-12))


def test_sliding_window_predict_returns_finite_curve_on_training_grid():
    rng = np.random.default_rng(3)
    n = 90
    y = np.r_[rng.normal(0.0, 0.3, 30), rng.normal(2.0, 0.3, 30), rng.normal(-1.0, 0.3, 30)]
    X = np.arange(n).reshape(-1, 1)
    sw = SlidingWindowSegmenter(BayesBreakGaussian(k_max=4), window_size=40, overlap=10).fit(X, y)
    pred = sw.predict(X)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


def test_sliding_window_rejects_bad_overlap():
    base = BayesBreakGaussian(k_max=3)
    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowSegmenter(base, window_size=10, overlap=10).fit(
            np.arange(20).reshape(-1, 1), np.zeros(20)
        )
    with pytest.raises(ValueError, match="overlap"):
        SlidingWindowSegmenter(base, window_size=10, overlap=-1).fit(
            np.arange(20).reshape(-1, 1), np.zeros(20)
        )
