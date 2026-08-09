"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from typing import Sequence
from .priors import PartitionPriorConfig


def log_boundary_hazard(boundary: int, x: Sequence[float], config: PartitionPriorConfig) -> float:
    """Interior-boundary prior factor; terminal boundaries are excluded."""
    raise NotImplementedError("CODE-BB-001: boundary hazard is not implemented in the skeleton")


def local_partition_score(start: int, stop: int, is_terminal: bool, x: Sequence[float] | None, config: PartitionPriorConfig) -> float:
    raise NotImplementedError("CODE-BB-002: local prior factor integration is not implemented in the skeleton")
