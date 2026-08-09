"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LatentGroupConfig:
    n_groups: int
    n_restarts: int
    max_iterations: int
    tolerance: float
    tie_breaking_rule: str


@dataclass(frozen=True)
class LatentGroupResult:
    responsibilities: Sequence[Sequence[float]]
    templates: Sequence[Sequence[int]]
    group_weights: Sequence[float]
    objective_trace: Sequence[float]
    final_objective: float
    selected_restart: int


def fit_latent_groups(sequence_template_scores: Sequence[Sequence[float]], config: LatentGroupConfig) -> LatentGroupResult:
    """Optimize the stated finite latent-group criterion."""
    raise NotImplementedError("CODE-BB-003: latent-group optimization is not implemented in the skeleton")
