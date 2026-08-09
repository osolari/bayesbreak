"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PartitionPriorConfig:
    segment_cohesion: str
    boundary_hazard: str
    parameters: Mapping[str, float]


def log_cohesion(start: int, stop: int, x: Sequence[float] | None, config: PartitionPriorConfig) -> float:
    raise NotImplementedError("CODE-BB-001: segment cohesion is not implemented in the skeleton")


def partition_log_prior(boundaries: Sequence[int], x: Sequence[float] | None, config: PartitionPriorConfig) -> float:
    raise NotImplementedError("CODE-BB-001: partition prior is not implemented in the skeleton")
