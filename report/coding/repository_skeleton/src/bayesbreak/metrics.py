"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


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


def match_boundaries_one_to_one(predicted: Sequence[float], reference: Sequence[float], tolerance: float) -> Sequence[BoundaryMatch]:
    """Maximum-cardinality, then minimum-distance one-to-one matching."""
    raise NotImplementedError("CODE-BB-008: boundary matching is not implemented in the skeleton")


def boundary_metrics(predicted: Sequence[float], reference: Sequence[float], tolerance: float, reference_type: str, metric_version: str) -> BoundaryMetrics:
    raise NotImplementedError("CODE-BB-008: boundary metrics are not implemented in the skeleton")
