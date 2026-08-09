"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .block_api import SegmentScoreTable


@dataclass(frozen=True)
class SharedBoundaryInput:
    coordinate_axis: Sequence[float]
    sequence_tables: Sequence[SegmentScoreTable]


def aggregate_block_log_evidence(data: SharedBoundaryInput) -> SegmentScoreTable:
    """Sum aligned sequence-specific segment log marginal likelihoods."""
    raise NotImplementedError("CODE-BB-004: stable shared-boundary aggregation is not implemented in the skeleton")
