"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from .dp import SumProductResult


@dataclass(frozen=True)
class PosteriorSummaries:
    boundary_probabilities: Sequence[float]
    ordered_boundary_marginals: Sequence[Sequence[float]]
    signal_mean: Sequence[float]
    signal_variance: Sequence[float] | None = None


def posterior_summaries(result: SumProductResult, *, n: int) -> PosteriorSummaries:
    raise NotImplementedError("CODE-BB-002: posterior summary extraction is not implemented in the skeleton")
