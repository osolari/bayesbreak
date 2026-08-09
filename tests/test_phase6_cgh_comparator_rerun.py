from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bayesbreak.baselines._types import BaselineResult
from bayesbreak.comparators import FAILURE_ID_AXIS_MISMATCH, ComparatorValidationError
from scripts.phase6_cgh_comparator_rerun import (
    algorithm_signal,
    build_raw_request,
    choose_pelt_candidate,
    comparator_record,
    exact_boundary_jaccard,
    verify_source,
)


def test_raw_request_and_algorithm_use_declared_transposes() -> None:
    probe_by_subject = np.arange(24, dtype=float).reshape(8, 3)
    request = build_raw_request(probe_by_subject, np.arange(8, dtype=float))
    assert np.asarray(request.values).shape == (3, 8)
    assert np.array_equal(algorithm_signal(request), probe_by_subject)
    assert request.tuning_budget.tuning_stratum == "matched-k-agreement-only"


def test_subject_by_probe_input_cannot_be_misdeclared_as_probe_by_subject() -> None:
    subject_by_probe = np.arange(24, dtype=float).reshape(3, 8)
    with pytest.raises(ComparatorValidationError, match="does not match"):
        build_raw_request(subject_by_probe, np.arange(8, dtype=float))


def test_wrong_source_hash_aborts_with_axis_failure(tmp_path: Path) -> None:
    source = tmp_path / "ACGH.RData"
    source.write_bytes(b"not the authorized source")
    with pytest.raises(RuntimeError, match=FAILURE_ID_AXIS_MISMATCH):
        verify_source(source)


def test_pelt_selection_uses_count_then_neutral_multiplier() -> None:
    candidates = [
        {"n_bkps": 4, "multiplier": 0.5, "penalty": 5.0},
        {"n_bkps": 5, "multiplier": 4.0, "penalty": 40.0},
        {"n_bkps": 5, "multiplier": 2.0, "penalty": 20.0},
    ]
    assert choose_pelt_candidate(candidates, target_n_bkps=5) == candidates[2]


def test_exact_boundary_jaccard_handles_empty_and_partial_sets() -> None:
    assert exact_boundary_jaccard([], []) == 1.0
    assert exact_boundary_jaccard([10, 20], [20, 30]) == pytest.approx(1 / 3)


def test_comparator_record_preserves_wrapper_extra_metadata() -> None:
    result = BaselineResult(
        algorithm="wild_binary_segmentation",
        package="ruptures",
        package_version="test",
        n=20,
        boundaries=np.asarray([10], dtype=np.intp),
        extra={"candidate_selection": "candidate-constrained-dynamic-programming"},
    )
    record = comparator_record(result, [10], runtime=0.1, tuning={"n_bkps": 1})
    assert record["extra"] == result.extra
