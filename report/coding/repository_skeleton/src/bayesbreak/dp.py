"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .block_api import SegmentScoreTable


@dataclass(frozen=True)
class SumProductResult:
    log_marginal_by_k: Sequence[float]
    log_normalizer: float
    segment_count_posterior: Sequence[float]


def sum_product_partition(log_scores: SegmentScoreTable, k_max: int, log_p_k: Sequence[float]) -> SumProductResult:
    """Compute posterior sums over admissible contiguous partitions."""
    raise NotImplementedError("CODE-BB-002: sum-product dynamic programming is not implemented in the skeleton")
