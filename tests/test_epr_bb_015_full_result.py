from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesbreak.provenance import InterpretationStatus, ResultRecord, read_sidecar
from scripts.finalize_epr_bb_015_result import failure_rows, validate_payload

ROOT = Path(__file__).parents[1]
RESULT_DIR = ROOT / "results" / "phase6" / "RES-BB-SYN-006"


def test_full_result_matches_pending_sidecar() -> None:
    result_path = RESULT_DIR / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    sidecar = read_sidecar(RESULT_DIR / "result_sidecar.json")
    assert isinstance(sidecar, ResultRecord)
    assert sidecar.scientific_interpretation is InterpretationStatus.PENDING
    expected_artifacts = {
        "results": RESULT_DIR / "results.json",
        "summary_table": RESULT_DIR / "failure_summary.csv",
        "summary_report": RESULT_DIR / "SUMMARY.md",
        "summary_figure": RESULT_DIR / "failure_map.png",
    }
    assert set(sidecar.output_hashes) == set(expected_artifacts)
    for name, path in expected_artifacts.items():
        assert sidecar.output_hashes[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["scientific_interpretation"] == "pending-independent-review"
    assert result["code"]["relevant_paths_clean"] is True
    assert len(result["records"]) == 400
    assert all(record["status"] == "executed" for record in result["records"])
    assert all(
        len(record[name]) == 64
        for record in result["records"]
        for name in ("data_hash", "truth_hash", "effective_config_hash")
    )


def test_full_result_retains_every_ep_timeout() -> None:
    result = json.loads((RESULT_DIR / "results.json").read_text(encoding="utf-8"))
    logistic_records = [
        record for record in result["records"] if record["cell"] == "logistic-approximation-failure"
    ]
    assert len(logistic_records) == 50
    assert all(record["methods"]["ep"]["status"] == "timed-out" for record in logistic_records)
    assert all(
        record["methods"]["ep"]["timeout_scope"] == "ep-fit-only" for record in logistic_records
    )


def test_full_result_failure_map_uses_predeclared_indicators() -> None:
    result = json.loads((RESULT_DIR / "results.json").read_text(encoding="utf-8"))
    validate_payload(result)
    rows = {row["cell"]: row for row in failure_rows(result)}
    assert rows["null-gaussian"]["primary_failure_rate"] == 0.68
    assert rows["zero-inflated-poisson"]["primary_failure_rate"] == 1.0
    assert rows["dense-gaussian"]["primary_failure_rate"] == 1.0
    assert rows["prior-conflict-gaussian"]["primary_failure_rate"] == 1.0
    assert rows["shared-boundary-heterogeneity"]["primary_failure_rate"] == 1.0
    assert rows["logistic-approximation-failure"]["primary_failure_rate"] == 1.0


def test_full_result_is_registered_as_pending_review() -> None:
    registry_path = (
        ROOT / "docs" / "manuscript" / "shared" / "metadata" / "result_interpretation.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [record for record in registry["results"] if record["result_id"] == "RES-BB-SYN-006"]
    assert len(matches) == 1
    assert matches[0]["interpretation_status"] == "pending independent scientific review"
