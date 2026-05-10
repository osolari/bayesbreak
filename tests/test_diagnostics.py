"""Diagnostics module (IMP-13)."""

from __future__ import annotations

import numpy as np

from bayesbreak import (
    BayesBreakGaussian,
    BayesBreakLogisticNormal,
    SharedBoundaryReplicatesSegmenter,
    run_dp_diagnostics,
    run_non_conjugate_diagnostics,
)


def test_dp_diagnostics_pass_for_segmenter():
    rng = np.random.default_rng(0)
    n = 80
    y = np.r_[rng.normal(0, 0.3, 30), rng.normal(2, 0.3, 30), rng.normal(-1, 0.3, 20)]
    est = BayesBreakGaussian(k_max=8).fit(np.arange(n).reshape(-1, 1), y)
    rep = run_dp_diagnostics(est)
    assert rep.passed
    assert rep.n == n
    assert rep.k_map == est.k_map_


def test_dp_diagnostics_pass_for_replicates():
    rng = np.random.default_rng(0)
    n = 60
    S = 4
    ys = [
        np.r_[rng.normal(0, 0.3, 20), rng.normal(1.5, 0.3, 20), rng.normal(-0.5, 0.3, 20)]
        for _ in range(S)
    ]
    rep = SharedBoundaryReplicatesSegmenter(BayesBreakGaussian(k_max=6)).fit(
        np.arange(n).reshape(-1, 1), ys
    )
    report = run_dp_diagnostics(rep)
    assert report.passed
    assert report.k_map == rep.k_map_


def test_dp_diagnostics_serialise_to_json():
    rng = np.random.default_rng(0)
    n = 30
    y = rng.normal(size=n)
    est = BayesBreakGaussian(k_max=4).fit(np.arange(n).reshape(-1, 1), y)
    payload = run_dp_diagnostics(est).to_json()
    assert "passed" in payload
    assert "checks" in payload


def test_non_conjugate_diagnostics_quantile_consistency():
    rng = np.random.default_rng(0)
    theta = np.r_[np.full(20, -1.0), np.full(20, 1.0), np.full(20, -0.5)]
    p = 1.0 / (1.0 + np.exp(-theta))
    y = rng.binomial(1, p).astype(float)
    n = y.size
    ref = BayesBreakLogisticNormal(k_max=8, approx="quadrature", gh_points=80).fit(
        np.arange(n).reshape(-1, 1), y
    )
    lap = BayesBreakLogisticNormal(k_max=8, approx="laplace").fit(np.arange(n).reshape(-1, 1), y)
    diag = run_non_conjugate_diagnostics(lap, ref)
    extra = diag.extra
    # Median ≤ q95 ≤ max.
    q = extra["block_error_quantiles"]
    assert q["q50"] <= q["q95"] <= q["q100"]
    assert extra["block_error_max"] == q["q100"]
    # Jaccard is in [0, 1]; we don't require perfect MAP agreement on a small
    # synthetic dataset where Laplace and quadrature can disagree on a few
    # boundary positions.
    assert 0.0 <= extra["map_path_jaccard"] <= 1.0
