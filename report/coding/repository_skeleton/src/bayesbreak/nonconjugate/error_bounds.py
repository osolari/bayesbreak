"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentErrorRecord:
    family: str
    block_support_hash: str
    reference_method: str
    max_log_score_error: float | None
    optimization_residual: float | None
    tail_bound: float | None
    quadrature_error: float | None
    convergence_status: str


def evaluate_reachable_segment_error(*args: object, **kwargs: object) -> SegmentErrorRecord:
    raise NotImplementedError("CODE-BB-005: segment-error assessment is not implemented in the skeleton")


def propagate_partition_bounds(max_log_score_error: float, k_max: int) -> dict[str, float]:
    raise NotImplementedError("CODE-BB-005: partition-error propagation utility is not implemented in the skeleton")
