"""Validated outputs for numerical segment-integration routines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class IntegrationResult:
    log_marginal: float
    optimization_residual: float | None = None
    tail_bound: float | None = None
    quadrature_error: float | None = None
    convergence_status: str = "unverifiable"


class SegmentIntegrator(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> IntegrationResult: ...


def approximate_segment_log_marginal(
    integrator: SegmentIntegrator,
    *args: object,
    **kwargs: object,
) -> IntegrationResult:
    """Execute an integrator and validate its distinct numerical-error fields."""

    result = integrator(*args, **kwargs)
    if not isinstance(result, IntegrationResult):
        raise TypeError("integrator must return IntegrationResult")
    if not math.isfinite(result.log_marginal):
        raise FloatingPointError("integrator returned a nonfinite log marginal")
    for name in ("optimization_residual", "tail_bound", "quadrature_error"):
        value = getattr(result, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative when provided")
    if result.convergence_status not in {"verified", "unverifiable", "failed"}:
        raise ValueError("Unknown convergence_status")
    return result
