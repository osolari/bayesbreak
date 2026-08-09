"""Canonical one-to-one boundary matching and metric records."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

MATCHING_RULE = "maximum-cardinality-then-minimum-distance-one-to-one"
DEFAULT_METRIC_VERSION = "boundary-metrics-1.0.0"


@dataclass(frozen=True)
class BoundaryMatch:
    predicted_index: int
    reference_index: int
    distance: float


@dataclass(frozen=True)
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float
    matched_mae: float | None
    matches: Sequence[BoundaryMatch]
    tolerance: float
    reference_type: str
    metric_version: str
    prediction_axis: str
    reference_axis: str
    matching_rule: str = MATCHING_RULE

    @property
    def mae_or_na(self) -> float | None:
        return self.matched_mae

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "prediction_axis": self.prediction_axis,
            "reference_axis": self.reference_axis,
            "tolerance": self.tolerance,
            "matching_rule": self.matching_rule,
            "reference_type": self.reference_type,
            "matched_pairs": [
                [match.predicted_index, match.reference_index] for match in self.matches
            ],
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mae_or_na": self.mae_or_na,
            "metric_version": self.metric_version,
        }


def match_boundaries_one_to_one(
    predicted: Sequence[float],
    reference: Sequence[float],
    tolerance: float,
) -> tuple[BoundaryMatch, ...]:
    """Find maximum-cardinality then minimum-distance one-to-one matches."""

    predicted_values = _finite_vector(predicted, "predicted")
    reference_values = _finite_vector(reference, "reference")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    n_predicted = predicted_values.size
    n_reference = reference_values.size
    if n_predicted == 0 or n_reference == 0:
        return ()

    distances = np.abs(predicted_values[:, None] - reference_values[None, :])
    eligible = distances <= tolerance_value
    maximum_matches = min(n_predicted, n_reference)
    reward = (maximum_matches + 1.0) * (tolerance_value + 1.0)
    size = n_predicted + n_reference
    costs = np.zeros((size, size), dtype=float)
    real_costs = np.zeros((n_predicted, n_reference), dtype=float)
    real_costs[eligible] = distances[eligible] - reward
    costs[:n_predicted, :n_reference] = real_costs

    rows, columns = linear_sum_assignment(costs)
    matches = [
        BoundaryMatch(int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if row < n_predicted and column < n_reference and eligible[row, column]
    ]
    return tuple(sorted(matches, key=lambda item: (item.predicted_index, item.reference_index)))


def boundary_metrics(
    predicted: Sequence[float],
    reference: Sequence[float],
    tolerance: float,
    reference_type: str,
    metric_version: str = DEFAULT_METRIC_VERSION,
    *,
    prediction_axis: str = "observation-index",
    reference_axis: str = "observation-index",
) -> BoundaryMetrics:
    """Compute precision/recall/F1 and matched MAE from canonical matches."""

    predicted_values = _finite_vector(predicted, "predicted")
    reference_values = _finite_vector(reference, "reference")
    if not reference_type:
        raise ValueError("reference_type must be nonempty")
    if not metric_version:
        raise ValueError("metric_version must be nonempty")
    if not prediction_axis or not reference_axis:
        raise ValueError("prediction and reference axes must be nonempty")
    matches = match_boundaries_one_to_one(predicted_values, reference_values, tolerance)
    matched = len(matches)

    if predicted_values.size:
        precision = matched / predicted_values.size
    else:
        precision = 1.0 if reference_values.size == 0 else 0.0
    if reference_values.size:
        recall = matched / reference_values.size
    else:
        recall = 1.0 if predicted_values.size == 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_mae = math.fsum(match.distance for match in matches) / matched if matched else None
    return BoundaryMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        matched_mae=matched_mae,
        matches=matches,
        tolerance=float(tolerance),
        reference_type=reference_type,
        metric_version=metric_version,
        prediction_axis=prediction_axis,
        reference_axis=reference_axis,
    )


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} boundaries must be one-dimensional")
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} boundaries must be finite")
    return vector
