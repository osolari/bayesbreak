"""Factorized ordered-partition priors."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise


DEFAULT_PRIOR_K_LABEL = "geometric_0.5"
"""Default segment-count prior label: normalized geometric-decay on 1..k_max.

The geometric form gives more prior mass to small segment counts than the
uniform alternative. The numerical operating characteristics depend on the
data-generating mechanism and are evaluated in the E01 experiment protocol.
"""


def _geometric_p_k(rate: float) -> Callable[[int], float]:
    """Return a callable ``k -> rate**(k - 1)`` (unnormalised)."""

    if not 0.0 < rate < 1.0:
        raise ValueError("geometric prior rate must be in the open interval (0, 1)")

    def _pk(k: int) -> float:
        return float(rate) ** (int(k) - 1)

    _pk.__name__ = f"geometric_{rate}"
    return _pk


def _uniform_p_k(k: int) -> float:  # noqa: ARG001 - callable signature contract
    return 1.0


_NAMED_PRIOR_K: dict[str, Callable[[int], float]] = {
    "uniform": _uniform_p_k,
    "geometric_0.5": _geometric_p_k(0.5),
    "geometric_0.1": _geometric_p_k(0.1),
}


def resolve_prior_k(
    prior_k: Callable[[int], float] | str | None,
) -> Callable[[int], float]:
    """Normalise the user-facing ``prior_k`` argument to a callable.

    - ``None`` resolves to the shipped default (``geometric_0.5``).
    - A string name resolves against :data:`_NAMED_PRIOR_K`.
    - A callable is returned as-is.

    Raises :class:`ValueError` for unknown names.
    """

    if prior_k is None:
        return _NAMED_PRIOR_K[DEFAULT_PRIOR_K_LABEL]
    if isinstance(prior_k, str):
        try:
            return _NAMED_PRIOR_K[prior_k]
        except KeyError as exc:
            names = ", ".join(sorted(_NAMED_PRIOR_K))
            raise ValueError(
                f"Unknown prior_k name {prior_k!r}; expected one of {{{names}}} or a callable"
            ) from exc
    if callable(prior_k):
        return prior_k
    raise TypeError(f"prior_k must be None, a name, or a callable; got {type(prior_k)!r}")


@dataclass(frozen=True)
class PartitionPriorConfig:
    """Configuration for segment cohesion and interior-boundary hazard."""

    segment_cohesion: str = "uniform"
    boundary_hazard: str = "uniform"
    parameters: Mapping[str, float] = field(default_factory=dict)


def log_cohesion(
    start: int,
    stop: int,
    x: Sequence[float] | None,
    config: PartitionPriorConfig,
) -> float:
    """Return the log cohesion for the complete segment ``(start, stop]``."""

    coordinates = _validated_coordinates(x)
    _validate_segment(start, stop, coordinates)
    return _log_cohesion_validated(start, stop, coordinates, config)


def _log_cohesion_validated(
    start: int,
    stop: int,
    coordinates: tuple[float, ...] | None,
    config: PartitionPriorConfig,
) -> float:
    span = float(stop - start) if coordinates is None else coordinates[stop] - coordinates[start]

    min_length = _nonnegative_parameter(config, "min_segment_length", default=1.0)
    min_span = _nonnegative_parameter(config, "min_segment_span", default=0.0)
    max_span = _maximum_span(config)
    if stop - start < min_length or span < min_span or span > max_span:
        return -math.inf

    mode = _normalise_mode(config.segment_cohesion)
    if mode in {"uniform", "constant", "none"}:
        return 0.0
    if mode in {"span", "span-power", "length", "length-power"}:
        power = _finite_parameter(
            config,
            "cohesion_power",
            fallback="power",
            default=1.0,
        )
        scale = _positive_parameter(config, "cohesion_scale", default=1.0)
        return power * math.log(span / scale)
    if mode == "explicit":
        factor = _required_nonnegative_parameter(
            config,
            f"cohesion_{start}_{stop}",
        )
        return -math.inf if factor == 0.0 else math.log(factor)
    raise ValueError(f"Unknown segment_cohesion mode: {config.segment_cohesion!r}")


def partition_log_prior(
    boundaries: Sequence[int],
    x: Sequence[float] | None,
    config: PartitionPriorConfig,
) -> float:
    """Return the unnormalized log prior of an ordered partition."""

    from .design_prior import log_boundary_hazard

    points = tuple(boundaries)
    if len(points) < 2 or points[0] != 0:
        raise ValueError("boundaries must start at 0 and contain at least one segment")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in points):
        raise TypeError("boundaries must contain integers")
    if any(left >= right for left, right in pairwise(points)):
        raise ValueError("boundaries must be strictly increasing")

    coordinates = _validated_coordinates(x)
    if coordinates is not None and points[-1] != len(coordinates) - 1:
        raise ValueError("the terminal boundary must match the coordinate support")
    hazard_coordinates: Sequence[float] = (
        coordinates if coordinates is not None else tuple(float(i) for i in range(points[-1] + 1))
    )

    total = 0.0
    for index, (start, stop) in enumerate(pairwise(points)):
        cohesion = log_cohesion(start, stop, coordinates, config)
        if not math.isfinite(cohesion):
            return -math.inf
        total += cohesion
        if index < len(points) - 2:
            hazard = log_boundary_hazard(stop, hazard_coordinates, config)
            if not math.isfinite(hazard):
                return -math.inf
            total += hazard
    return total


def _validated_coordinates(x: Sequence[float] | None) -> tuple[float, ...] | None:
    if x is None:
        return None
    coordinates = tuple(float(value) for value in x)
    if len(coordinates) < 2:
        raise ValueError("x must contain at least two boundary coordinates")
    if any(not math.isfinite(value) for value in coordinates):
        raise ValueError("x must contain only finite boundary coordinates")
    if any(left >= right for left, right in pairwise(coordinates)):
        raise ValueError("x must be strictly increasing")
    return coordinates


def _validate_segment(
    start: int,
    stop: int,
    coordinates: tuple[float, ...] | None,
) -> None:
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(stop, int)
        or isinstance(stop, bool)
    ):
        raise TypeError("segment boundaries must be integers")
    if start < 0 or stop <= start:
        raise ValueError("segment boundaries must satisfy 0 <= start < stop")
    if coordinates is not None and stop >= len(coordinates):
        raise ValueError("segment stop exceeds the coordinate support")


def _normalise_mode(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _finite_parameter(
    config: PartitionPriorConfig,
    name: str,
    *,
    fallback: str | None = None,
    default: float,
) -> float:
    raw = config.parameters.get(name)
    if raw is None and fallback is not None:
        raw = config.parameters.get(fallback)
    value = default if raw is None else float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative_parameter(
    config: PartitionPriorConfig,
    name: str,
    *,
    default: float,
) -> float:
    value = _finite_parameter(config, name, default=default)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_parameter(
    config: PartitionPriorConfig,
    name: str,
    *,
    default: float,
) -> float:
    value = _finite_parameter(config, name, default=default)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _maximum_span(config: PartitionPriorConfig) -> float:
    value = float(config.parameters.get("max_segment_span", math.inf))
    if math.isnan(value) or value < 0:
        raise ValueError("max_segment_span must be nonnegative")
    return value


def _required_nonnegative_parameter(config: PartitionPriorConfig, name: str) -> float:
    if name not in config.parameters:
        raise ValueError(f"Missing required prior parameter: {name}")
    return _nonnegative_parameter(config, name, default=0.0)
