from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bayesbreak.provenance import ResultRecord, read_sidecar
from scripts.finalize_phase6_cgh_comparator import comparator_rows, validate_payload

ROOT = Path(__file__).parents[1]
RESULT_DIR = ROOT / "results" / "phase6" / "RES-BB-CMP-003"


def _payload() -> dict:
    return json.loads((RESULT_DIR / "results.json").read_text(encoding="utf-8"))


def test_full_cgh_result_passes_finalizer_validation() -> None:
    payload = _payload()
    validate_payload(payload)
    assert payload["source"]["matrix_shape"] == [2215, 43]
    assert payload["bayesbreak"]["shared"]["k_map"] == 15


def test_pelt_nonattainment_is_distinct_from_exact_matched_k() -> None:
    rows = {row["algorithm"]: row for row in comparator_rows(_payload())}
    assert rows["pelt"]["n_bkps"] == 11
    assert rows["pelt"]["count_status"] == "closest-grid-count-mismatch"
    assert all(
        row["count_status"] == "exact-matched-k"
        for algorithm, row in rows.items()
        if algorithm != "pelt"
    )


def test_finalizer_rejects_comparator_axis_drift() -> None:
    payload = copy.deepcopy(_payload())
    payload["comparators"][0]["boundary_metrics_tau3"]["prediction_axis"] = "flattened-index"
    with pytest.raises(ValueError, match="prediction axis"):
        validate_payload(payload)


def test_cgh_sidecar_hashes_all_release_artifacts() -> None:
    record = read_sidecar(RESULT_DIR / "result_sidecar.json")
    assert isinstance(record, ResultRecord)
    assert record.result_id == "RES-BB-CMP-003"
    assert record.parent_result_id == "RES-BB-CMP-002"
    for name, expected_hash in record.output_hashes.items():
        path = ROOT / record.artifacts[name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
