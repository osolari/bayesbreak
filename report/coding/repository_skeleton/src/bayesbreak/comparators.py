"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TuningBudget:
    parameter_evaluations: int
    selection_rule: str
    data_access: str


@dataclass(frozen=True)
class ComparatorInputSchema:
    values: Sequence[Sequence[float]]
    coordinate_axis: Sequence[float]
    task_type: str
    tuning_budget: TuningBudget
    metadata: Mapping[str, str]


def validate_axis(values: Sequence[Sequence[float]], coordinate_axis: Sequence[float]) -> None:
    raise NotImplementedError("CODE-BB-009: comparator-axis validation is not implemented in the skeleton")
