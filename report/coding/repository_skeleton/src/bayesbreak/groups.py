"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
from .replicates import SharedBoundaryInput


@dataclass(frozen=True)
class KnownGroupFitRequest:
    groups: Mapping[str, SharedBoundaryInput]
    k_max: int


def fit_known_groups(request: KnownGroupFitRequest) -> Mapping[str, object]:
    raise NotImplementedError("CODE-BB-004: known-group fitting is not implemented in the skeleton")
