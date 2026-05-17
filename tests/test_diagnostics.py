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
from bayesbreak.diagnostics import run_prior_sensitivity, select_n_groups_by_holdout


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
    # Worst-case TV bound on P(k|y) derived from prop:stability:
    # empirical TV ≤ exp(2·k_max·ε) − 1.
    assert "pk_tv_empirical" in extra
    assert "pk_tv_upper_bound" in extra
    assert extra["pk_tv_empirical"] <= extra["pk_tv_upper_bound"] + 1e-9
    # The TV check should be present with the failure_mode tag and have passed.
    tv_check = next(c for c in diag.checks if c.name == "pk_tv_bound_check")
    assert tv_check.failure_mode == "tv-bound"
    assert tv_check.passed
    # Theoretical rate annotation from prop:uniform-bounds: Laplace on
    # reachable blocks is O(n^-1). The field should be populated and the
    # estimator's approx attribute round-trips.
    assert extra["approx_routine"] == "laplace"
    assert "O(n^-1)" in extra["theoretical_rate"]
    # rate_violated may be True or False but must be a bool (not None) here.
    assert isinstance(extra["theoretical_rate_violated"], bool)


def test_prior_sensitivity_reports_variation_per_variant():
    """§6 6-C1 planned diagnostic: ``run_prior_sensitivity`` reports
    ``Δ p(k|y)`` and ``Δ P(b_i|y,k_map)`` for each perturbation of
    ``p(k)`` or ``g``.
    """

    rng = np.random.default_rng(0)
    n = 80
    y = np.r_[rng.normal(0, 0.3, 30), rng.normal(2, 0.3, 30), rng.normal(-1, 0.3, 20)]
    est = BayesBreakGaussian(k_max=8).fit(np.arange(n).reshape(-1, 1), y)

    rep = run_prior_sensitivity(est)
    # Three p(k) perturbations + two g variants = 5 entries by default.
    assert len(rep.extra["variants"]) == 5
    for variant in rep.extra["variants"]:
        for field_ in ("delta_pk_max", "delta_pk_tv", "delta_bm_max", "delta_bm_l1"):
            assert field_ in variant
            assert variant[field_] >= 0.0
        # Magnitudes are bounded: |Δ p(k|y)| ≤ 1, TV ≤ 1.
        assert variant["delta_pk_max"] <= 1.0 + 1e-9
        assert variant["delta_pk_tv"] <= 1.0 + 1e-9
    # Every check carries the prior-sensitivity failure mode tag.
    for c in rep.checks:
        assert c.failure_mode == "prior-sensitivity"


def test_select_n_groups_by_holdout_picks_small_g_on_homogeneous_data():
    """§5b ``rem:teicher-overspec`` mitigation: K-fold held-out
    log-likelihood prefers the smallest defensible ``G`` on data that
    actually comes from a single template (no real heterogeneity)."""

    rng = np.random.default_rng(0)
    n = 40
    # All sequences share the same boundary structure (single template):
    # left half mean = -0.5, right half mean = +0.5.
    seqs = [np.r_[rng.normal(-0.5, 0.3, n // 2), rng.normal(0.5, 0.3, n // 2)] for _ in range(8)]
    rep = select_n_groups_by_holdout(
        BayesBreakGaussian(k_max=3),
        seqs,
        g_grid=(1, 2),
        n_folds=2,
        n_restarts=1,
        max_iter=6,
        random_state=0,
    )
    extra = rep.extra
    assert extra["g_grid"] == [1, 2]
    assert extra["n_sequences"] == 8
    # Held-out marginal log-likelihood reported for each G.
    assert len(extra["mean_test_loglik"]) == 2
    # best_g exists and is one of the candidates.
    assert extra["best_g"] in (1, 2)
    # Every per-G check carries the teicher-overspec failure mode tag.
    for c in rep.checks:
        assert c.failure_mode == "teicher-overspec"
    # Exactly one check (best_g) is marked passing.
    passing = [c for c in rep.checks if c.passed]
    assert len(passing) == 1
    assert int(passing[0].name.split("=")[1]) == extra["best_g"]


def test_select_n_groups_by_holdout_recovers_two_groups_on_separable_data():
    """K-fold held-out log-likelihood prefers ``G=2`` when the data
    actually has two clearly different templates."""

    rng = np.random.default_rng(1)
    n = 40
    # Group A: jump at n/2; Group B: jump at n/4. Distinct templates.
    group_a = [np.r_[rng.normal(0.0, 0.2, n // 2), rng.normal(2.0, 0.2, n // 2)] for _ in range(6)]
    group_b = [
        np.r_[rng.normal(0.0, 0.2, n // 4), rng.normal(-2.0, 0.2, n - n // 4)] for _ in range(6)
    ]
    seqs = group_a + group_b
    rep = select_n_groups_by_holdout(
        BayesBreakGaussian(k_max=3),
        seqs,
        g_grid=(1, 2),
        n_folds=2,
        n_restarts=2,
        max_iter=8,
        random_state=0,
    )
    # Two-template data should prefer G=2 over G=1 on average.
    means = rep.extra["mean_test_loglik"]
    assert rep.extra["best_g"] == 2
    assert means[1] >= means[0]
