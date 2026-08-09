from __future__ import annotations

import math

import numpy as np
import pytest

from bayesbreak.nonconjugate import (
    IntegrationResult,
    approximate_segment_log_marginal,
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
