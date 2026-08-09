from __future__ import annotations

import numpy as np

from scripts.phase6_misspecification_rerun import (
    generate_logistic_cell,
    generate_shared_cell,
    generate_standard_cell,
    interval_summary,
    piecewise_mean,
    summarize,
)


def test_piecewise_mean_returns_strict_partition() -> None:
    values, boundaries = piecewise_mean([3, 2, 4], [0.0, 1.0, -1.0])
    assert values.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
    assert boundaries == [0, 3, 5, 9]


def test_standard_generators_are_deterministic_and_match_plan_dimensions() -> None:
    expected = {
        "null-gaussian": (120, [0, 120]),
        "heavy-tail-gaussian": (120, [0, 40, 80, 120]),
        "zero-inflated-poisson": (120, [0, 40, 80, 120]),
        "dense-gaussian": (120, list(range(0, 121, 10))),
        "short-segment-gaussian": (100, [0, 48, 52, 100]),
        "prior-conflict-gaussian": (120, [0, 40, 80, 120]),
    }
    for cell_id, (n, boundaries) in expected.items():
        first = generate_standard_cell(cell_id, 123)
        second = generate_standard_cell(cell_id, 123)
        assert np.array_equal(first["values"], second["values"])
        assert first["values"].size == n
        assert first["true_boundaries"] == boundaries


def test_shared_and_logistic_generators_preserve_declared_truth() -> None:
    shared = generate_shared_cell(456)
    assert len(shared["sequences"]) == 8
    assert all(values.shape == (120,) for values in shared["sequences"])
    assert shared["common_boundaries"] == [0, 40, 80, 120]
    assert shared["subject_boundaries"][:2] == [[0, 40, 60, 80, 120]] * 2

    logistic = generate_logistic_cell(789)
    assert logistic["values"].shape == (80,)
    assert logistic["true_boundaries"] == [0, 40, 80]
    assert set(np.unique(logistic["values"])).issubset({0.0, 1.0})


def test_summary_retains_failed_and_reversed_outcomes() -> None:
    records = [
        {"status": "failed", "cell": "null-gaussian", "wall_seconds": 0.1},
        {
            "status": "executed",
            "cell": "null-gaussian",
            "wall_seconds": 0.2,
            "boundary_metrics": {"f1": 0.0},
            "k_error": 2,
            "posterior_k_entropy": 0.5,
            "missed_change_count": 0,
            "false_discovery_count": 2,
        },
    ]
    summary = summarize(records, ["null-gaussian"])["cells"]["null-gaussian"]
    assert summary["n_runs"] == 2
    assert summary["n_failed"] == 1
    assert summary["failure_rate"] == 0.5
    assert summary["false_discovery_count"]["mean"] == 2.0


def test_interval_summary_handles_single_pilot_value() -> None:
    assert interval_summary([3.0]) == {
        "mean": 3.0,
        "standard_error": 0.0,
        "ci95_lower": 3.0,
        "ci95_upper": 3.0,
        "min": 3.0,
        "max": 3.0,
    }
