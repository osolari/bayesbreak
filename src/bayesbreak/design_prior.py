"""Design-dependent interior-boundary hazards."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .priors import (
    PartitionPriorConfig,
    _log_cohesion_validated,
    _normalise_mode,
    _validated_coordinates,
)

FloatArray = NDArray[np.floating]


def log_boundary_hazard(
    boundary: int,
    x: Sequence[float],
    config: PartitionPriorConfig,
) -> float:
    """Return the log factor for an interior boundary at ``boundary``."""

    coordinates = _validated_coordinates(x)
    assert coordinates is not None
    if not isinstance(boundary, int) or isinstance(boundary, bool):
        raise TypeError("boundary must be an integer")
    if boundary <= 0 or boundary >= len(coordinates) - 1:
        raise ValueError("boundary hazard is defined only for interior boundaries")
    return _log_boundary_hazard_validated(boundary, coordinates, config)


def _log_boundary_hazard_validated(
    boundary: int,
    coordinates: tuple[float, ...],
    config: PartitionPriorConfig,
) -> float:
    mode = _normalise_mode(config.boundary_hazard)
    if mode in {"uniform", "constant", "none"}:
        return 0.0
    if mode in {
        "poisson",
        "poisson-occupancy",
        "poisson-interval-occupancy",
        "fixed-count-poisson-occupancy",
    }:
        integrated_key = f"integrated_intensity_{boundary}"
        if integrated_key in config.parameters:
            integrated_intensity = float(config.parameters[integrated_key])
        else:
            intensity = _first_parameter(config, "poisson_intensity", "intensity", "rate")
            integrated_intensity = intensity * (coordinates[boundary] - coordinates[boundary - 1])
        if not math.isfinite(integrated_intensity) or integrated_intensity < 0:
            raise ValueError("integrated Poisson intensity must be finite and nonnegative")
        if integrated_intensity == 0.0:
            return -math.inf
        return integrated_intensity + math.log1p(-math.exp(-integrated_intensity))
    if mode == "explicit":
        key = f"hazard_{boundary}"
        if key not in config.parameters:
            key = f"boundary_hazard_{boundary}"
        if key not in config.parameters:
            raise ValueError(f"Missing required prior parameter: hazard_{boundary}")
        factor = float(config.parameters[key])
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("explicit boundary hazards must be finite and nonnegative")
        return -math.inf if factor == 0.0 else math.log(factor)
    raise ValueError(f"Unknown boundary_hazard mode: {config.boundary_hazard!r}")


def local_partition_score(
    start: int,
    stop: int,
    is_terminal: bool,
    x: Sequence[float] | None,
    config: PartitionPriorConfig,
) -> float:
    """Return ``log c_x(start, stop)`` plus any interior hazard at ``stop``."""

    coordinates = _validated_coordinates(x)
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError("start must be an integer")
    if not isinstance(stop, int) or isinstance(stop, bool):
        raise TypeError("stop must be an integer")
    if start < 0 or stop <= start:
        raise ValueError("segment boundaries must satisfy 0 <= start < stop")
    if coordinates is not None and stop >= len(coordinates):
        raise ValueError("segment stop exceeds the coordinate support")

    cohesion = _log_cohesion_validated(start, stop, coordinates, config)
    if not math.isfinite(cohesion) or is_terminal:
        return cohesion
    hazard_coordinates = (
        coordinates if coordinates is not None else tuple(float(i) for i in range(stop + 2))
    )
    hazard = _log_boundary_hazard_validated(stop, hazard_coordinates, config)
    return cohesion + hazard


def build_log_prior_table(
    n: int,
    x: Sequence[float] | None,
    config: PartitionPriorConfig,
) -> FloatArray:
    """Build all local prior factors after validating coordinate support once."""

    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    coordinates = _validated_coordinates(x)
    if coordinates is None:
        coordinates = tuple(float(i) for i in range(n + 1))
    elif len(coordinates) != n + 1:
        raise ValueError(f"x must contain n+1={n + 1} boundary coordinates")

    table = np.full((n + 1, n + 1), -np.inf, dtype=float)
    for start in range(n):
        for stop in range(start + 1, n + 1):
            cohesion = _log_cohesion_validated(start, stop, coordinates, config)
            if not math.isfinite(cohesion):
                continue
            table[start, stop] = cohesion
            if stop < n:
                hazard = _log_boundary_hazard_validated(stop, coordinates, config)
                if not math.isfinite(hazard):
                    table[start, stop] = -math.inf
                else:
                    table[start, stop] += hazard
    return table


def _first_parameter(config: PartitionPriorConfig, *names: str) -> float:
    for name in names:
        if name in config.parameters:
            value = float(config.parameters[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            return value
    choices = ", ".join(names)
    raise ValueError(f"Poisson boundary hazard requires one of: {choices}")
