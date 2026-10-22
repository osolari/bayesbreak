"""Comparator input, coordinate-axis, and tuning-budget validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

FAILURE_ID_AXIS_MISMATCH = "FAIL-BB-002"


class ComparatorValidationError(ValueError):
    """Comparator request rejected before execution or metric computation."""

    failure_id = FAILURE_ID_AXIS_MISMATCH


@dataclass(frozen=True)
class TuningBudget:
    parameter_evaluations: int
    selection_rule: str
    data_access: str
    tuning_stratum: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameter_evaluations, int)
            or isinstance(self.parameter_evaluations, bool)
            or self.parameter_evaluations < 0
        ):
            raise ValueError("parameter_evaluations must be a nonnegative integer")
        for name in ("selection_rule", "data_access", "tuning_stratum"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be explicit and nonempty")


@dataclass(frozen=True)
class ComparatorInputSchema:
    values: Sequence[Sequence[float]]
    coordinate_axis: Sequence[float]
    task_type: str
    tuning_budget: TuningBudget
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.task_type.strip():
            raise ValueError("task_type must be explicit and nonempty")
        source_kind = self.metadata.get("source_kind")
        if source_kind is None:
            raise ValueError("metadata.source_kind is required")
        validate_axis(
            self.values,
            self.coordinate_axis,
            task_type=self.task_type,
            source_kind=source_kind,
        )


def validate_axis(
    values: Sequence[Sequence[float]],
    coordinate_axis: Sequence[float],
    *,
    task_type: str = "multisequence",
    source_kind: str = "raw-observations",
) -> None:
    """Reject values whose observation dimension or provenance mismatches the axis."""

    array = np.asarray(values, dtype=float)
    axis = np.asarray(coordinate_axis, dtype=float)
    if axis.ndim != 1 or axis.size == 0:
        raise ComparatorValidationError("coordinate_axis must be a nonempty vector")
    if np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
        raise ComparatorValidationError("coordinate_axis must be finite and strictly increasing")
    if np.any(~np.isfinite(array)):
        raise ComparatorValidationError("comparator values must be finite")

    normalized_task = task_type.strip().lower().replace("_", "-")
    if normalized_task == "multisequence":
        if source_kind != "raw-observations":
            raise ComparatorValidationError(
                f"{FAILURE_ID_AXIS_MISMATCH}: multisequence comparators require "
                "raw-observations; fitted curves and cached map traces are invalid"
            )
        if array.ndim != 2 or array.shape[0] < 2:
            raise ComparatorValidationError(
                "multisequence comparator values must be an unflattened "
                "sequence-by-coordinate matrix"
            )
        if array.shape[1] != axis.size:
            raise ComparatorValidationError(
                f"comparator observation dimension {array.shape[1]} does not match "
                f"coordinate axis length {axis.size}"
            )
        return
    if normalized_task == "univariate":
        if source_kind != "raw-observations":
            raise ComparatorValidationError("univariate comparators require raw-observations")
        if array.ndim != 1 or array.size != axis.size:
            raise ComparatorValidationError(
                "univariate comparator values must match coordinate_axis length"
            )
        return
    raise ValueError(f"Unknown comparator task_type: {task_type!r}")
