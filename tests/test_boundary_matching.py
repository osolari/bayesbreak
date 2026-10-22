from __future__ import annotations

import pytest

from bayesbreak.metrics import (
    MATCHING_RULE,
    boundary_metrics,
    match_boundaries_one_to_one,
)


def _pairs(matches):
    return [(match.predicted_index, match.reference_index) for match in matches]


def test_hand_computed_matching_and_metrics() -> None:
    predicted = [2.0, 9.0, 20.0]
    reference = [1.0, 10.0, 14.0]
    result = boundary_metrics(predicted, reference, 2.0, "simulated-truth")
    assert _pairs(result.matches) == [(0, 0), (1, 1)]
    assert result.precision == pytest.approx(2.0 / 3.0)
    assert result.recall == pytest.approx(2.0 / 3.0)
    assert result.f1 == pytest.approx(2.0 / 3.0)
    assert result.matched_mae == pytest.approx(1.0)
    assert result.matching_rule == MATCHING_RULE


def test_maximum_cardinality_precedes_total_distance() -> None:
    matches = match_boundaries_one_to_one([0.0, 2.0], [1.0, 3.0], 2.0)
    assert len(matches) == 2
    assert sum(match.distance for match in matches) == pytest.approx(2.0)


def test_distance_ties_are_deterministic() -> None:
    matches = match_boundaries_one_to_one([0.0, 2.0], [1.0], 1.0)
    assert _pairs(matches) == [(0, 0)]


def test_duplicates_are_distinct_but_never_reused() -> None:
    matches = match_boundaries_one_to_one([1.0, 1.0], [1.0, 1.0], 0.0)
    assert _pairs(matches) == [(0, 0), (1, 1)]
    assert len({match.predicted_index for match in matches}) == 2
    assert len({match.reference_index for match in matches}) == 2


@pytest.mark.parametrize(
    "predicted, reference, expected",
    [
        ([], [], (1.0, 1.0, 1.0)),
        ([], [1.0], (0.0, 0.0, 0.0)),
        ([1.0], [], (0.0, 0.0, 0.0)),
    ],
)
def test_empty_cases_have_declared_metrics(predicted, reference, expected) -> None:
    result = boundary_metrics(predicted, reference, 1.0, "external-truth")
    assert (result.precision, result.recall, result.f1) == expected
    assert result.matched_mae is None


def test_zero_matches_use_na_mae() -> None:
    result = boundary_metrics([0.0], [10.0], 1.0, "external-truth")
    assert result.matches == ()
    assert result.mae_or_na is None


def test_symmetry_preserves_cardinality_distance_and_f1() -> None:
    forward = boundary_metrics([0.0, 5.0, 9.0], [1.0, 8.0], 2.0, "truth")
    reverse = boundary_metrics([1.0, 8.0], [0.0, 5.0, 9.0], 2.0, "truth")
    assert len(forward.matches) == len(reverse.matches)
    assert forward.matched_mae == pytest.approx(reverse.matched_mae)
    assert forward.f1 == pytest.approx(reverse.f1)


def test_tolerance_boundary_is_inclusive_and_version_is_recorded() -> None:
    result = boundary_metrics(
        [0.0],
        [2.0],
        2.0,
        "external-truth",
        "metric-test-v2",
        prediction_axis="probe-coordinate",
        reference_axis="probe-coordinate",
    )
    assert len(result.matches) == 1
    assert result.metric_version == "metric-test-v2"
    assert result.prediction_axis == result.reference_axis == "probe-coordinate"
    payload = result.to_dict()
    assert payload["schema_version"] == "1.0.0"
    assert payload["matching_rule"] == MATCHING_RULE
    assert payload["matched_pairs"] == [[0, 0]]
