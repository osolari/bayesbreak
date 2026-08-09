"""Structured records for approximate nonconjugate segment integration."""

from .error_bounds import (
    SegmentErrorRecord,
    evaluate_reachable_segment_error,
    propagate_partition_bounds,
)
from .integration import IntegrationResult, approximate_segment_log_marginal

__all__ = [
    "IntegrationResult",
    "SegmentErrorRecord",
    "approximate_segment_log_marginal",
    "evaluate_reachable_segment_error",
    "propagate_partition_bounds",
]
