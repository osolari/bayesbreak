"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class OrderedObservations:
    """Validated ordered observations and family-specific descriptors."""

    y: Sequence[float]
    x: Sequence[float] | None = None
    descriptors: Mapping[str, Sequence[float] | float | int] = field(default_factory=dict)
    likelihood_power_weights: Sequence[float] | None = None


@dataclass(frozen=True)
class SegmentPosteriorMoments:
    mean: float
    variance: float | None = None
    extra: Mapping[str, float] = field(default_factory=dict)


class SegmentModel(Protocol):
    """Observation-family interface required by the partition algorithms."""

    family_name: str

    def segment_log_marginal(self, start: int, stop: int) -> float:
        """Return log p(y[start:stop] | family, prior)."""
        ...

    def segment_posterior_moments(self, start: int, stop: int) -> SegmentPosteriorMoments:
        """Return posterior moments for one candidate segment."""
        ...

    def posterior_predictive_logpdf(self, start: int, stop: int, y_new: float, **descriptors: Any) -> float:
        """Return a family-specific posterior-predictive log density or mass."""
        ...
