from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesbreak.provenance import (
    SCHEMA_VERSION,
    InterpretationStatus,
    LineageStatus,
    ResultRecord,
    ResultStatus,
    read_sidecar,
    validate_result_record,
    write_sidecar,
)

ROOT = Path(__file__).parents[1]
ARCHIVED_RESULTS = ROOT / "docs" / "manuscript" / "shared" / "figures" / "results"
MIGRATION_MANIFEST = (
    ROOT
    / "docs"
    / "manuscript"
    / "revision_artifacts"
    / "adoption"
    / "LEGACY_ABSOLUTE_PATH_MIGRATIONS.json"
)
SHA256 = "a" * 64


def _record(**overrides) -> ResultRecord:
    values = {
        "result_id": "RES-BB-TEST-001",
        "execution_status": ResultStatus.EXECUTED,
        "scientific_interpretation": InterpretationStatus.VALID_FOR_STATED_INTERPRETATION,
        "lineage_status": LineageStatus.ORIGINAL,
        "parent_result_id": None,
        "data_hash": SHA256,
        "config_hash": SHA256,
        "code_hash": SHA256,
        "environment_hash": SHA256,
        "coordinate_metadata": {
            "prediction_axis": "observation-index",
            "reference_axis": "observation-index",
            "reference_type": "simulated-truth",
        },
        "metrics": {"f1": 1.0},
        "artifacts": {"table": "results/test/table.json"},
        "output_hashes": {"table": SHA256},
    }
    values.update(overrides)
    return ResultRecord(**values)


def test_execution_and_interpretation_statuses_are_distinct() -> None:
    assert ResultStatus.EXECUTED.value == "executed"
    assert InterpretationStatus.VALID_FOR_STATED_INTERPRETATION.value != "executed"
    assert InterpretationStatus.EXCLUDED_FROM_INTENDED_CONCLUSION.value != "planned"


def test_executed_result_requires_primary_hashes() -> None:
    with pytest.raises(ValueError, match="data_hash"):
        validate_result_record(_record(data_hash=None))


def test_planned_result_may_declare_pending_hashes() -> None:
    record = _record(
        execution_status=ResultStatus.PLANNED,
        scientific_interpretation=InterpretationStatus.PENDING,
        data_hash=None,
        config_hash=None,
        code_hash=None,
        environment_hash=None,
        output_hashes={},
    )
    validate_result_record(record)


def test_corrected_release_requires_lineage_and_output_hashes() -> None:
    with pytest.raises(ValueError, match="parent_result_id"):
        validate_result_record(_record(lineage_status=LineageStatus.CORRECTED))
    with pytest.raises(ValueError, match="output hash"):
        validate_result_record(
            _record(
                lineage_status=LineageStatus.CORRECTED,
                parent_result_id="RES-BB-OLD-001",
                output_hashes={},
            )
        )


def test_release_mode_rejects_absolute_or_parent_artifact_paths() -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        validate_result_record(_record(artifacts={"figure": "/tmp/figure.pdf"}))
    with pytest.raises(ValueError, match="repository-relative"):
        validate_result_record(_record(artifacts={"figure": "results/../figure.pdf"}))


def test_write_and_read_current_sidecar(tmp_path: Path) -> None:
    path = write_sidecar(tmp_path / "record.json", _record())
    assert path.read_text(encoding="utf-8").endswith("\n")
    loaded = read_sidecar(path)
    assert loaded == _record()
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize(
    "filename",
    [
        "fig6_welllog.json",
        "fig7_cgh.json",
        "fig8_spx.json",
        "fig9_methylation.json",
        "realdata_metrics.json",
        "baselines_metrics.json",
    ],
)
def test_archived_sidecars_have_read_only_compatibility_paths(filename: str) -> None:
    path = ARCHIVED_RESULTS / filename
    before = path.read_bytes()
    loaded = read_sidecar(path, migration_manifest=MIGRATION_MANIFEST)
    assert isinstance(loaded, dict)
    if "figure_path" in loaded:
        assert not Path(loaded["figure_path"]).is_absolute()
    assert path.read_bytes() == before


def test_legacy_migration_verifies_immutable_asset_hash(tmp_path: Path) -> None:
    source = ARCHIVED_RESULTS / "fig6_welllog.json"
    tampered = tmp_path / "shared" / "figures" / "results" / source.name
    tampered.parent.mkdir(parents=True)
    tampered.write_bytes(source.read_bytes() + b"\n")
    assert (
        hashlib.sha256(tampered.read_bytes()).hexdigest()
        != hashlib.sha256(source.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="hash does not match"):
        read_sidecar(tampered, migration_manifest=MIGRATION_MANIFEST)


def test_versioned_schema_declares_runtime_contract() -> None:
    schema = json.loads((ROOT / "schemas" / "result_sidecar.schema.json").read_text())
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert "coordinate_metadata" in schema["required"]
