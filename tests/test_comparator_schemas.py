from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bayesbreak.comparators import (
    FAILURE_ID_AXIS_MISMATCH,
    ComparatorInputSchema,
    ComparatorValidationError,
    TuningBudget,
    validate_axis,
)


def _budget() -> TuningBudget:
    return TuningBudget(
        parameter_evaluations=12,
        selection_rule="training-only fixed grid",
        data_access="training-only",
        tuning_stratum="equal-evaluation-count",
    )


def test_valid_raw_multisequence_matrix_route() -> None:
    request = ComparatorInputSchema(
        values=np.arange(12, dtype=float).reshape(3, 4),
        coordinate_axis=[10.0, 20.0, 30.0, 40.0],
        task_type="multisequence",
        tuning_budget=_budget(),
        metadata={"source_kind": "raw-observations", "dataset": "cgh"},
    )
    assert request.task_type == "multisequence"
    assert request.tuning_budget.tuning_stratum == "equal-evaluation-count"


def test_run_script_builds_same_validated_raw_route() -> None:
    from scripts.run_comparators import build_comparator_request

    request = build_comparator_request(
        np.arange(12, dtype=float).reshape(3, 4),
        np.arange(4, dtype=float),
        task_type="multisequence",
        parameter_evaluations=4,
        selection_rule="fixed grid",
        data_access="training-only",
        tuning_stratum="equal-budget",
        dataset="cgh",
    )
    assert np.asarray(request.values).shape == (3, 4)
    assert request.metadata["source_kind"] == "raw-observations"


def test_flattened_multisequence_input_is_rejected() -> None:
    with pytest.raises(ComparatorValidationError, match="unflattened"):
        validate_axis(np.arange(12), np.arange(12), task_type="multisequence")


def test_historical_cached_cgh_map_curve_is_rejected_deterministically() -> None:
    with pytest.raises(ComparatorValidationError) as error:
        ComparatorInputSchema(
            values=np.zeros((43, 2215)),
            coordinate_axis=np.arange(2215),
            task_type="multisequence",
            tuning_budget=_budget(),
            metadata={"source_kind": "cached-map-curve", "result_id": "RES-BB-CMP-002"},
        )
    assert error.value.failure_id == FAILURE_ID_AXIS_MISMATCH
    assert FAILURE_ID_AXIS_MISMATCH in str(error.value)


def test_axis_length_and_orientation_must_match_observation_dimension() -> None:
    with pytest.raises(ComparatorValidationError, match="does not match"):
        validate_axis(np.zeros((4, 3)), np.arange(4), task_type="multisequence")


def test_nonfinite_values_or_axis_are_rejected() -> None:
    with pytest.raises(ComparatorValidationError, match="values must be finite"):
        validate_axis([[1.0, np.nan], [2.0, 3.0]], [0.0, 1.0])
    with pytest.raises(ComparatorValidationError, match="strictly increasing"):
        validate_axis([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])


@pytest.mark.parametrize("field", ["selection_rule", "data_access", "tuning_stratum"])
def test_tuning_budget_requires_explicit_stratum_fields(field: str) -> None:
    values = {
        "parameter_evaluations": 1,
        "selection_rule": "fixed",
        "data_access": "training-only",
        "tuning_stratum": "equal-budget",
    }
    values[field] = ""
    with pytest.raises(ValueError, match=field):
        TuningBudget(**values)


def test_cached_baseline_script_preserves_rejected_cgh_as_diagnostic(monkeypatch) -> None:
    from scripts.tables import baselines_on_cached

    cached = SimpleNamespace(
        n_=4,
        map_curve_=np.zeros((3, 4)),
        map_boundaries_=[0, 2, 4],
    )
    monkeypatch.setattr(baselines_on_cached, "_load_fit", lambda name: {"rep": cached})
    result = baselines_on_cached.run_on("fig7_cgh", "cgh")
    assert result["execution_status"] == "rejected-before-comparator"
    assert result["failure_id"] == FAILURE_ID_AXIS_MISMATCH
    assert result["scientific_interpretation"] == "diagnostic-only"
