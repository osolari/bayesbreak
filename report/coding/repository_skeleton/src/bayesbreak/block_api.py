"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .base import SegmentModel


@dataclass(frozen=True)
class SegmentScoreTable:
    """Dense upper-triangular segment log-marginal-likelihood table."""

    n: int
    values: Sequence[Sequence[float]]
    reachable_mask: Sequence[Sequence[bool]]
    family: str


def build_segment_score_table(model: SegmentModel, n: int, min_segment_length: int = 1) -> SegmentScoreTable:
    """Build all reachable segment scores (CODE-BB-002 and family tasks)."""
    raise NotImplementedError("CODE-BB-002: segment-score construction is not implemented in the skeleton")
