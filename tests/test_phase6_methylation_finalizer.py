from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bayesbreak.provenance import ResultRecord, read_sidecar
from scripts.finalize_phase6_methylation_predictive import split_rows, validate_payload

ROOT = Path(__file__).parents[1]
RESULT_DIR = ROOT / "results" / "phase6" / "RES-BB-RD-008Q"


def _payload() -> dict:
    return json.loads((RESULT_DIR / "results.json").read_text(encoding="utf-8"))


def test_full_methylation_result_passes_finalizer_validation() -> None:
    payload = _payload()
    validate_payload(payload)
    assert payload["summary"]["total_denominator"] == 1520
    assert len(split_rows(payload)) == 10


def test_finalizer_rejects_clipping_or_invalid_precision() -> None:
    clipped = copy.deepcopy(_payload())
    clipped["records"][0]["prediction_metadata"]["extrapolation"] = "clip"
    with pytest.raises(ValueError, match="provenance"):
        validate_payload(clipped)

    invalid_precision = copy.deepcopy(_payload())
    invalid_precision["records"][0]["phi_new_min"] = 0.0
    with pytest.raises(ValueError, match="precision"):
        validate_payload(invalid_precision)


def test_finalizer_rejects_score_denominator_drift() -> None:
    payload = copy.deepcopy(_payload())
    payload["summary"]["total_denominator"] = 1519
    with pytest.raises(ValueError, match="denominator"):
        validate_payload(payload)


def test_methylation_sidecar_hashes_all_release_artifacts() -> None:
    record = read_sidecar(RESULT_DIR / "result_sidecar.json")
    assert isinstance(record, ResultRecord)
    assert record.result_id == "RES-BB-RD-008Q"
    assert record.parent_result_id == "RES-BB-RD-007Q"
    assert record.split_hash == "76c7f52af33b45e9d9385fe368bfe0d621e93bdde89ae4a9cfe494918e8a1941"
    for name, expected_hash in record.output_hashes.items():
        path = ROOT / record.artifacts[name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
