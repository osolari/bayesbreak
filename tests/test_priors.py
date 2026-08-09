from __future__ import annotations

import itertools
import math

import pytest

from bayesbreak.design_prior import log_boundary_hazard
from bayesbreak.priors import PartitionPriorConfig, log_cohesion, partition_log_prior


def _partitions(n: int, k: int):
    for interior in itertools.combinations(range(1, n), k - 1):
        yield (0, *interior, n)


def _normalised_partition_weights(n: int, k: int, x, config):
    scores = [partition_log_prior(partition, x, config) for partition in _partitions(n, k)]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def test_uniform_factors_give_uniform_fixed_count_prior() -> None:
    config = PartitionPriorConfig()
    weights = _normalised_partition_weights(5, 3, None, config)
    assert weights == pytest.approx([1.0 / math.comb(4, 2)] * math.comb(4, 2))


def test_cohesion_and_hazard_are_separate_factors() -> None:
    x = [0.0, 1.0, 2.0, 4.0, 7.0]
    config = PartitionPriorConfig(
        segment_cohesion="span-power",
        boundary_hazard="explicit",
        parameters={"cohesion_power": 2.0, "hazard_2": 3.0},
    )
    expected = 2.0 * math.log(2.0) + math.log(3.0) + 2.0 * math.log(5.0)
    assert partition_log_prior((0, 2, 4), x, config) == pytest.approx(expected)


def test_terminal_endpoint_receives_no_hazard() -> None:
    config = PartitionPriorConfig(
        boundary_hazard="explicit",
        parameters={"hazard_2": 2.0},
    )
    assert partition_log_prior((0, 2, 4), None, config) == pytest.approx(math.log(2.0))


def test_poisson_interval_uses_fixed_count_occupancy_odds() -> None:
    x = [0.0, 0.2, 1.7, 2.0]
    config = PartitionPriorConfig(
        boundary_hazard="poisson-occupancy",
        parameters={"intensity": 0.7},
    )
    integrated_intensity = 0.7 * (x[2] - x[1])
    observed = log_boundary_hazard(2, x, config)
    assert observed == pytest.approx(math.log(math.expm1(integrated_intensity)), abs=1e-14)
    assert observed != pytest.approx(math.log1p(-math.exp(-integrated_intensity)))


def test_equal_poisson_intervals_recover_uniform_fixed_count_prior() -> None:
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    config = PartitionPriorConfig(
        boundary_hazard="poisson-occupancy",
        parameters={"intensity": 0.4},
    )
    assert _normalised_partition_weights(4, 2, x, config) == pytest.approx([1.0 / 3.0] * 3)


def test_irregular_poisson_hazard_favours_wider_candidate_interval() -> None:
    x = [0.0, 0.2, 1.7, 2.0]
    config = PartitionPriorConfig(
        boundary_hazard="poisson-occupancy",
        parameters={"intensity": 0.7},
    )
    assert log_boundary_hazard(2, x, config) > log_boundary_hazard(1, x, config)


def test_zero_support_is_preserved() -> None:
    config = PartitionPriorConfig(parameters={"min_segment_length": 2.0})
    assert log_cohesion(0, 1, None, config) == -math.inf
    assert partition_log_prior((0, 1, 4), None, config) == -math.inf


def test_hazard_rejects_noninterior_boundaries() -> None:
    config = PartitionPriorConfig()
    with pytest.raises(ValueError, match="interior"):
        log_boundary_hazard(0, [0.0, 1.0, 2.0], config)
    with pytest.raises(ValueError, match="interior"):
        log_boundary_hazard(2, [0.0, 1.0, 2.0], config)


def test_partition_requires_complete_coordinate_support() -> None:
    with pytest.raises(ValueError, match="terminal"):
        partition_log_prior((0, 2), [0.0, 1.0, 2.0, 3.0], PartitionPriorConfig())
