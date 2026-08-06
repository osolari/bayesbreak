"""Interface-only BayesBreak skeleton.

No scientific algorithm is implemented in this module. See the canonical
coding handoff and the referenced CODE-BB task before adding behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ExtrapolationPolicy(str, Enum):
    ERROR = "error"
    LEFT_ENDPOINT = "left_endpoint"
    RIGHT_ENDPOINT = "right_endpoint"
    EXPLICIT_MODEL = "explicit_model"


@dataclass(frozen=True)
class PredictionRequest:
    coordinates: Sequence[float]
    extrapolation: ExtrapolationPolicy = ExtrapolationPolicy.ERROR


def assign_to_partition(coordinates: Sequence[float], fitted_coordinates: Sequence[float], boundaries: Sequence[int], extrapolation: ExtrapolationPolicy) -> Sequence[int]:
    raise NotImplementedError("CODE-BB-007: coordinate assignment and extrapolation are not implemented in the skeleton")


def posterior_predictive_logpdf(*args: object, **kwargs: object) -> Sequence[float]:
    raise NotImplementedError("CODE-BB-006/007: family-specific posterior prediction is not implemented in the skeleton")
