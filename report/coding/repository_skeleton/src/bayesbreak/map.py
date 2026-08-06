"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .block_api import SegmentScoreTable


@dataclass(frozen=True)
class MAPPartitionResult:
    boundaries: Sequence[int]
    segment_count: int
    log_posterior_score: float
    tie_breaking_rule: str


def max_sum_partition(log_scores: SegmentScoreTable, k_max: int, log_p_k: Sequence[float], tie_breaking_rule: str) -> MAPPartitionResult:
    """Return the joint MAP partition using max-sum recursion and backtracking."""
    raise NotImplementedError("CODE-BB-002: max-sum dynamic programming is not implemented in the skeleton")
