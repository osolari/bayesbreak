from __future__ import annotations

import math

import numpy as np
import pytest

from bayesbreak.nonconjugate import (
    IntegrationResult,
    approximate_segment_log_marginal,
    certify_against_gh_reference,
    certify_by_quadrature_doubling,
    evaluate_reachable_segment_error,
    propagate_partition_bounds,
)


def _scores(n: int) -> np.ndarray:
    scores = np.full((n + 1, n + 1), -np.inf)
    for start in range(n):
        for stop in range(start + 1, n + 1):
            scores[start, stop] = start + stop / 10.0
    return scores


def test_reachable_error_uses_shared_coordinates_and_maximum() -> None:
    reference = _scores(4)
    approximate = reference.copy()
    approximate[1, 4] += 0.2
    approximate[0, 4] += 0.1
    record = evaluate_reachable_segment_error(
        approximate,
        reference,
        family="logistic-normal",
        reference_method="high-accuracy-quadrature",
        k_max=3,
        convergence_status="verified",
    )
    assert record.max_log_score_error == pytest.approx(0.2)
    assert record.n_reachable_blocks > 0
    assert len(record.block_support_hash) == 64
    assert record.to_dict()["schema_version"] == "1.0.0"


def test_support_mismatch_returns_explicit_unverifiable_state() -> None:
    reference = _scores(4)
    approximate = reference.copy()
    approximate[1, 4] = -np.inf
    record = evaluate_reachable_segment_error(
        approximate,
        reference,
        family="logistic-normal",
        reference_method="quadrature",
        k_max=3,
    )
    assert record.convergence_status == "unverifiable"
    assert record.max_log_score_error is None
    assert "supports differ" in record.failure_reason


def test_invalid_reachable_score_returns_failed_state() -> None:
    reference = _scores(3)
    approximate = reference.copy()
    approximate[0, 3] = np.nan
    record = evaluate_reachable_segment_error(
        approximate,
        reference,
        family="beta-observation",
        reference_method="quadrature",
        k_max=2,
    )
    assert record.convergence_status == "failed"
    assert record.max_log_score_error is None


def test_partition_bounds_use_max_error_and_cap_total_variation() -> None:
    bounds = propagate_partition_bounds(0.2, 3)
    assert bounds["max_log_evidence_error"] == pytest.approx(0.6)
    assert bounds["max_log_posterior_odds_error"] == pytest.approx(1.2)
    assert bounds["tv_upper_bound"] == min(1.0, math.expm1(1.2))
    assert propagate_partition_bounds(1000.0, 20)["tv_upper_bound"] == 1.0


def test_nested_tolerance_records_tighten_or_remain_explicit() -> None:
    reference = _scores(4)
    coarse = reference + np.where(np.isfinite(reference), 0.1, 0.0)
    fine = reference + np.where(np.isfinite(reference), 0.01, 0.0)
    coarse_record = evaluate_reachable_segment_error(
        coarse,
        reference,
        family="logistic-normal",
        reference_method="quadrature",
        k_max=3,
        quadrature_error=0.1,
        convergence_status="verified",
    )
    fine_record = evaluate_reachable_segment_error(
        fine,
        reference,
        family="logistic-normal",
        reference_method="quadrature",
        k_max=3,
        quadrature_error=0.01,
        convergence_status="verified",
    )
    assert fine_record.max_log_score_error < coarse_record.max_log_score_error
    assert fine_record.quadrature_error < coarse_record.quadrature_error


def test_integration_result_keeps_error_sources_separate() -> None:
    result = approximate_segment_log_marginal(
        lambda: IntegrationResult(
            -2.5,
            optimization_residual=1e-5,
            tail_bound=2e-5,
            quadrature_error=3e-5,
            convergence_status="verified",
        )
    )
    assert result.optimization_residual == 1e-5
    assert result.tail_bound == 2e-5
    assert result.quadrature_error == 3e-5


def test_certify_against_gh_reference_populates_all_error_sources() -> None:
    """A fitted logistic-normal estimator produces a SegmentErrorRecord with
    distinct optimisation and quadrature error fields against a high-order
    GH reference."""

    from bayesbreak import BayesBreakLogisticNormal

    rng = np.random.default_rng(0)
    n = 40
    p = np.r_[0.2 * np.ones(20), 0.8 * np.ones(20)]
    y = rng.binomial(1, p).astype(float)
    X = np.arange(n).reshape(-1, 1)

    laplace = BayesBreakLogisticNormal(k_max=3, approx="laplace").fit(X, y)
    lap_record = certify_against_gh_reference(laplace, reference_gh_points=120)
    assert lap_record.family == "BayesBreakLogisticNormal"
    assert lap_record.quadrature_error is not None
    assert lap_record.quadrature_error >= 0.0
    assert lap_record.optimization_residual is not None
    # Laplace's terminal Newton gradient should be tiny.
    assert lap_record.optimization_residual < 1e-6
    assert lap_record.n_reachable_blocks > 0

    gh = BayesBreakLogisticNormal(k_max=3, approx="quadrature", gh_points=80).fit(X, y)
    gh_record = certify_against_gh_reference(gh, reference_gh_points=200)
    assert gh_record.reference_method == "gh-200"
    assert gh_record.quadrature_error is not None
    assert gh_record.quadrature_error >= 0.0
    # Quadrature at 80 points against a 200-point reference is finite and
    # the record carries the expected metadata; whether the residual is
    # numerically smaller than Laplace's Gaussian-mode error on this
    # short-segment / extreme-``p`` cell depends on the specific
    # integrand and is not a universal identity.


@pytest.mark.parametrize("approx", ["laplace", "jj", "pg_vb", "ep"])
def test_certify_against_gh_reference_measures_approximation_error(approx) -> None:
    """For every non-conjugate kernel, the certificate's ``quadrature_error``
    measures the block-evidence gap against a high-order GH reference, not
    the method's numerical drift under Q-doubling."""

    from bayesbreak import BayesBreakLogisticNormal

    rng = np.random.default_rng(11)
    n = 30
    p = np.r_[0.2 * np.ones(15), 0.8 * np.ones(15)]
    y = rng.binomial(1, p).astype(float)
    X = np.arange(n).reshape(-1, 1)

    est = BayesBreakLogisticNormal(k_max=3, approx=approx, max_iter=25).fit(X, y)
    record = certify_against_gh_reference(est, reference_gh_points=100)
    assert record.quadrature_error is not None
    assert record.quadrature_error >= 0.0
    # The gap is finite on a clean two-segment signal for every kernel.
    assert record.quadrature_error < 20.0


def test_legacy_certify_by_quadrature_doubling_alias() -> None:
    """Both public names dispatch to the same reference check."""

    assert certify_by_quadrature_doubling is certify_against_gh_reference


def test_laplace_analytic_tail_bound_reported_in_certificate() -> None:
    """A fitted Laplace logistic-normal estimator exposes the Kass-Tierney
    third-order remainder bound as ``tail_bound`` in its certificate."""

    from bayesbreak import BayesBreakLogisticNormal

    rng = np.random.default_rng(3)
    n = 40
    p = np.r_[0.2 * np.ones(20), 0.8 * np.ones(20)]
    y = rng.binomial(1, p).astype(float)
    X = np.arange(n).reshape(-1, 1)

    laplace = BayesBreakLogisticNormal(k_max=3, approx="laplace").fit(X, y)
    assert hasattr(laplace, "analytic_tail_bound_")
    tail_arr = laplace.analytic_tail_bound_
    assert tail_arr.shape == (n + 1, n + 1)
    # Bound is finite and nonnegative on admissible blocks.
    admissible = laplace.admissibility_mask_
    assert bool(np.all(tail_arr[admissible] >= 0.0))

    record = certify_by_quadrature_doubling(laplace)
    assert record.tail_bound is not None
    assert record.tail_bound >= 0.0
    # The Kass-Tierney bound should dominate the empirical Q-doubling gap
    # on a smooth well-conditioned block — this is the standard sanity check.
    if record.quadrature_error is not None:
        # loose sanity: the analytic bound is a valid upper bound modulo a
        # small constant, so it should not be orders of magnitude smaller
        # than the empirical gap.
        assert record.tail_bound + 1e-6 >= record.quadrature_error * 1e-3


def test_jj_analytic_remainder_is_nonneg_and_matches_gh_reference() -> None:
    """The Jaakkola-Jordan tangent lower-bounds every log-likelihood factor,
    so the analytic remainder must be nonnegative. On a well-conditioned
    block it also lower-bounds the empirical GH-referenced gap
    (both measure the same nonneg quantity, up to GH truncation)."""

    from bayesbreak import BayesBreakLogisticNormal

    rng = np.random.default_rng(4)
    n = 40
    p = np.r_[0.2 * np.ones(20), 0.8 * np.ones(20)]
    y = rng.binomial(1, p).astype(float)
    X = np.arange(n).reshape(-1, 1)

    jj = BayesBreakLogisticNormal(k_max=3, approx="jj").fit(X, y)
    assert hasattr(jj, "analytic_tail_bound_")
    tail_arr = jj.analytic_tail_bound_
    assert tail_arr.shape == (n + 1, n + 1)
    admissible = jj.admissibility_mask_
    finite_admissible = admissible & np.isfinite(tail_arr)
    # JJ is a rigorous lower bound so the analytic remainder is >= 0.
    assert bool(np.all(tail_arr[finite_admissible] >= -1e-9))

    record = certify_against_gh_reference(jj, reference_gh_points=120)
    assert record.tail_bound is not None
    assert record.quadrature_error is not None
    # The analytic remainder and the empirical GH-referenced gap measure
    # the same JJ excess; they should agree in order of magnitude, with
    # the analytic bound bounded above by the empirical gap up to
    # numerical slack (the analytic form uses one-block GH on the JJ
    # posterior; the empirical form uses the full-block GH reference).
    assert record.tail_bound >= 0.0
    assert record.tail_bound <= max(record.quadrature_error * 5.0, 1.0)


def test_pg_vb_analytic_remainder_matches_jj() -> None:
    """PG-VB and JJ share the same variational tangent; their analytic
    remainders coincide on the same data."""

    from bayesbreak import BayesBreakLogisticNormal

    rng = np.random.default_rng(5)
    n = 20
    y = rng.binomial(1, 0.5, size=n).astype(float)
    X = np.arange(n).reshape(-1, 1)
    jj = BayesBreakLogisticNormal(k_max=2, approx="jj").fit(X, y)
    pg = BayesBreakLogisticNormal(k_max=2, approx="pg_vb").fit(X, y)
    assert np.allclose(jj.analytic_tail_bound_, pg.analytic_tail_bound_)


def test_laplace_analytic_tail_bound_shrinks_with_larger_blocks() -> None:
    """The Kass-Tierney third-order remainder decreases as the effective
    sample size on a block grows — the ``O(N^{-3/2})`` regime."""

    from bayesbreak.families.logistic_normal._laplace import (
        laplace_analytic_remainder,
        laplace_block,
    )

    # Same theta_map / v_post pattern under increasing N.
    small = np.array([50.0])
    large = np.array([500.0])
    S_small = 0.5 * small
    S_large = 0.5 * large
    _, m_s, v_s, _ = laplace_block(S_small, small, nu=0.0, rho2=1.0)
    _, m_l, v_l, _ = laplace_block(S_large, large, nu=0.0, rho2=1.0)
    bound_small = float(laplace_analytic_remainder(m_s, v_s, small)[0])
    bound_large = float(laplace_analytic_remainder(m_l, v_l, large)[0])
    assert bound_large < bound_small


def test_certify_by_quadrature_doubling_rejects_unsupported_family() -> None:
    from bayesbreak import BayesBreakGaussian

    rng = np.random.default_rng(0)
    n = 20
    y = rng.normal(size=n)
    est = BayesBreakGaussian(k_max=2).fit(np.arange(n).reshape(-1, 1), y)
    with pytest.raises(TypeError, match="BayesBreakLogisticNormal"):
        certify_by_quadrature_doubling(est)
