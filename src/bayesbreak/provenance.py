"""Versioned results and portable, validated experiment metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "1.0.0"

_RESULT_ID_PATTERN = re.compile(r"^RES-BB-[A-Z0-9]+(?:-[A-Z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ResultStatus(str, Enum):
    """Execution lifecycle for a result record."""

    PLANNED = "planned"
    EXECUTED = "executed"
    FAILED = "failed"
    BLOCKED = "blocked"


class InterpretationStatus(str, Enum):
    """Scientific use status, kept separate from execution lifecycle."""

    PENDING = "pending"
    VALID_FOR_STATED_INTERPRETATION = "valid-for-stated-interpretation"
    EXCLUDED_FROM_INTENDED_CONCLUSION = "excluded-from-intended-conclusion"
    REAL_DIAGNOSTIC = "real-diagnostic"
    IMPLEMENTATION_VERIFICATION = "implementation-verification"


class LineageStatus(str, Enum):
    """Whether a result is original or corrects an archived result."""

    ORIGINAL = "original"
    CORRECTED = "corrected"


@dataclass(frozen=True)
class ResultRecord:
    """A versioned, release-validatable BayesBreak result sidecar."""

    result_id: str
    execution_status: ResultStatus
    scientific_interpretation: InterpretationStatus
    lineage_status: LineageStatus
    parent_result_id: str | None
    data_hash: str | None
    config_hash: str | None
    code_hash: str | None
    environment_hash: str | None
    coordinate_metadata: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    weights_hash: str | None = None
    split_hash: str | None = None
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible canonical representation."""

        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "lineage_status": self.lineage_status.value,
            "parent_result_id": self.parent_result_id,
            "execution_status": self.execution_status.value,
            "scientific_interpretation": self.scientific_interpretation.value,
            "data_hash": self.data_hash,
            "weights_hash": self.weights_hash,
            "split_hash": self.split_hash,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "environment_hash": self.environment_hash,
            "coordinate_metadata": dict(self.coordinate_metadata),
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
            "output_hashes": dict(self.output_hashes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResultRecord:
        """Construct and validate a record parsed from JSON."""

        try:
            record = cls(
                schema_version=str(payload["schema_version"]),
                result_id=str(payload["result_id"]),
                lineage_status=LineageStatus(payload["lineage_status"]),
                parent_result_id=_optional_string(payload.get("parent_result_id")),
                execution_status=ResultStatus(payload["execution_status"]),
                scientific_interpretation=InterpretationStatus(
                    payload["scientific_interpretation"]
                ),
                data_hash=_optional_string(payload.get("data_hash")),
                weights_hash=_optional_string(payload.get("weights_hash")),
                split_hash=_optional_string(payload.get("split_hash")),
                config_hash=_optional_string(payload.get("config_hash")),
                code_hash=_optional_string(payload.get("code_hash")),
                environment_hash=_optional_string(payload.get("environment_hash")),
                coordinate_metadata=_mapping(payload, "coordinate_metadata"),
                metrics=_mapping(payload, "metrics"),
                artifacts=_string_mapping(payload, "artifacts"),
                output_hashes=_string_mapping(payload, "output_hashes"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result record: {exc}") from exc
        validate_result_record(record)
        return record


def validate_result_record(record: ResultRecord, *, release_mode: bool = True) -> None:
    """Validate identifiers, lineage, hashes, metadata, and artifact paths."""

    if record.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported result schema version {record.schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    _validate_result_id(record.result_id, "result_id")

    if record.lineage_status is LineageStatus.CORRECTED:
        if record.parent_result_id is None:
            raise ValueError("Corrected results require parent_result_id")
        if record.parent_result_id == record.result_id:
            raise ValueError("A corrected result cannot name itself as its parent")
        _validate_result_id(record.parent_result_id, "parent_result_id")
    elif record.parent_result_id is not None:
        raise ValueError("parent_result_id requires lineage_status='corrected'")

    required_hashes = {
        "data_hash": record.data_hash,
        "config_hash": record.config_hash,
        "code_hash": record.code_hash,
        "environment_hash": record.environment_hash,
    }
    if record.execution_status is ResultStatus.EXECUTED:
        missing = [name for name, value in required_hashes.items() if value is None]
        if missing:
            raise ValueError(f"Executed results require hashes: {', '.join(missing)}")

    for name, value in {
        **required_hashes,
        "weights_hash": record.weights_hash,
        "split_hash": record.split_hash,
    }.items():
        _validate_optional_hash(value, name)
    for name, value in record.output_hashes.items():
        _validate_optional_hash(value, f"output_hashes[{name!r}]")

    if record.lineage_status is LineageStatus.CORRECTED:
        if record.execution_status is not ResultStatus.EXECUTED:
            raise ValueError("Corrected release records must have execution_status='executed'")
        if not record.output_hashes:
            raise ValueError("Corrected release records require at least one output hash")

    required_coordinate_fields = {"prediction_axis", "reference_axis", "reference_type"}
    missing_coordinate_fields = required_coordinate_fields - record.coordinate_metadata.keys()
    if missing_coordinate_fields:
        missing = ", ".join(sorted(missing_coordinate_fields))
        raise ValueError(f"coordinate_metadata is missing: {missing}")

    if release_mode:
        for name, value in record.artifacts.items():
            _validate_relative_path(value, f"artifacts[{name!r}]")


def write_sidecar(
    path: str | Path,
    record: ResultRecord,
    *,
    release_mode: bool = True,
) -> Path:
    """Validate and atomically write a deterministic result sidecar."""

    validate_result_record(record, release_mode=release_mode)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_sidecar(path: str | Path) -> ResultRecord | dict[str, Any]:
    """Read a validated current record or an existing portable figure-metrics JSON.

    Old path-migration manifests are not part of the active BayesBreak
    experiment protocol. A non-versioned metrics dictionary may be read for
    descriptive figures only; it cannot be promoted to a validated run record.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Result metadata must contain a JSON object: {source}")
    if "schema_version" in payload and "result_id" in payload:
        return ResultRecord.from_dict(payload)
    absolute_fields = _absolute_path_fields(payload)
    if absolute_fields:
        raise ValueError(f"Result metadata contains absolute paths: {', '.join(absolute_fields)}")
    return payload


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload[name]
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _string_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, str]:
    value = _mapping(payload, name)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise TypeError(f"{name} must map strings to strings")
    return value  # type: ignore[return-value]


def _optional_string(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise TypeError("Expected a string or null")


def _validate_result_id(value: str, name: str) -> None:
    if not _RESULT_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a canonical RES-BB identifier")


def _validate_optional_hash(value: str | None, name: str) -> None:
    if value is not None and not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_relative_path(value: str, name: str) -> None:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ValueError(f"{name} must be a repository-relative path")


def _absolute_path_fields(payload: Mapping[str, Any]) -> list[str]:
    return [
        name
        for name, value in payload.items()
        if isinstance(value, str)
        and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute())
    ]
