"""Shared-boundary replicates: pooled evidence factorisation (Theorem ``multisubject``)."""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian, SharedBoundaryReplicatesSegmenter
from bayesbreak.dp import max_sum_segmentation


def test_pooled_log_evidence_factorises():
    """Pooled log A^(0)_{ij} = Σ_s subject log A^(0,s)_{ij} (Theorem ``multisubject``)."""

    rng = np.random.default_rng(0)
    n = 24
    S = 3
    means = np.r_[np.full(8, 0.0), np.full(8, 1.5), np.full(8, -0.5)]
    sigma = 0.3
    ys = [means + sigma * rng.standard_normal(n) for _ in range(S)]

    rep = SharedBoundaryReplicatesSegmenter(
        BayesBreakGaussian(k_max=4, estimate_hyper=False, nu=0.0, rho2=10.0, sigma2=sigma**2)
    )
    rep.fit(np.arange(n).reshape(-1, 1), ys)

    # Per-subject standalone fits (same hyperparameters).
    per_subject_lA0 = []
    for s in range(S):
        e = BayesBreakGaussian(
            k_max=4, estimate_hyper=False, nu=0.0, rho2=10.0, sigma2=sigma**2
        ).fit(np.arange(n).reshape(-1, 1), ys[s])
        per_subject_lA0.append(e.log_block_evidence_)

    pooled = rep.log_block_evidence_
    expected = sum(per_subject_lA0)
    finite = np.isfinite(pooled) & np.isfinite(expected)
    assert np.allclose(pooled[finite], expected[finite], atol=1e-9)


def test_single_subject_reduces_to_bare_segmenter():
    """With S=1, replicates reduces to the bare BayesBreakGaussian."""

    rng = np.random.default_rng(1)
    n = 30
    y = np.r_[np.full(15, 0.0), np.full(15, 2.0)] + 0.3 * rng.standard_normal(n)

    bare = BayesBreakGaussian(k_max=4).fit(np.arange(n).reshape(-1, 1), y)
    rep = SharedBoundaryReplicatesSegmenter(BayesBreakGaussian(k_max=4)).fit(
        np.arange(n).reshape(-1, 1), [y]
    )
    assert rep.k_map_ == bare.k_map_
    assert rep.map_boundaries_ == bare.map_boundaries_
    assert rep.log_evidence_ == pytest.approx(bare.log_evidence_, rel=1e-8)


def test_map_segmentation_matches_pooled_max_sum():
    """rep.map_boundaries_ matches max-sum DP on the pooled lA0 matrix."""

    rng = np.random.default_rng(2)
    n = 40
    S = 4
    means = np.r_[np.full(15, -0.5), np.full(10, 1.0), np.full(15, 0.0)]
    ys = [means + 0.4 * rng.standard_normal(n) for _ in range(S)]

    rep = SharedBoundaryReplicatesSegmenter(BayesBreakGaussian(k_max=6)).fit(
        np.arange(n).reshape(-1, 1), ys
    )
    expected, _ = max_sum_segmentation(rep.log_block_evidence_, k=rep.k_map_)
    assert rep.map_boundaries_ == expected
