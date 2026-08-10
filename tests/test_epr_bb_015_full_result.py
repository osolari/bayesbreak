from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bayesbreak.provenance import InterpretationStatus, ResultRecord, read_sidecar

ROOT = Path(__file__).parents[1]
RESULT_DIR = ROOT / "results" / "phase6" / "RES-BB-SYN-006"


def test_full_result_matches_pending_sidecar() -> None:
    result_path = RESULT_DIR / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    sidecar = read_sidecar(RESULT_DIR / "result_sidecar.json")
    assert isinstance(sidecar, ResultRecord)
    assert sidecar.scientific_interpretation is InterpretationStatus.PENDING
    assert sidecar.output_hashes["results"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
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
