from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian, dp
from bayesbreak.design_prior import build_log_prior_table, local_partition_score
from bayesbreak.priors import PartitionPriorConfig, partition_log_prior
from bayesbreak.utils import logsumexp


def _partitions(n: int, k: int):
    for interior in itertools.combinations(range(1, n), k - 1):
        yield (0, *interior, n)


def _score_partition(boundaries, block_scores, prior_scores) -> float:
    return sum(
        float(block_scores[start, stop] + prior_scores[start, stop])
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=False)
    )


def test_local_partition_score_excludes_terminal_hazard() -> None:
    config = PartitionPriorConfig(
        boundary_hazard="explicit",
        parameters={"hazard_2": 4.0},
    )
    x = [0.0, 1.0, 2.0]
    assert local_partition_score(0, 2, True, x, config) == 0.0
    assert local_partition_score(0, 2, False, None, config) == pytest.approx(math.log(4.0))


def test_prior_table_matches_partition_factorization() -> None:
    n = 5
    x = [0.0, 0.2, 0.5, 1.8, 2.0, 3.0]
    config = PartitionPriorConfig(
        segment_cohesion="span-power",
        boundary_hazard="poisson-occupancy",
        parameters={"cohesion_power": 0.5, "intensity": 0.7},
    )
    table = build_log_prior_table(n, x, config)
    for k in range(1, 5):
        for boundaries in _partitions(n, k):
            local_sum = sum(
                float(table[start, stop])
                for start, stop in zip(boundaries[:-1], boundaries[1:], strict=False)
            )
            assert local_sum == pytest.approx(partition_log_prior(boundaries, x, config))


def test_sum_product_normalizer_and_map_match_exhaustive_enumeration() -> None:
    rng = np.random.default_rng(20260805)
    for _ in range(1000):
        n = int(rng.integers(2, 8))
        k_max = int(rng.integers(1, n + 1))
        widths = rng.uniform(0.05, 2.0, size=n)
        x = np.concatenate(([0.0], np.cumsum(widths)))
        config = PartitionPriorConfig(
            segment_cohesion="span-power",
            boundary_hazard="poisson-occupancy",
            parameters={
                "cohesion_power": float(rng.uniform(-1.0, 1.0)),
                "intensity": float(rng.uniform(0.05, 1.5)),
                "min_segment_length": float(rng.integers(1, min(3, n) + 1)),
            },
        )
        prior_scores = build_log_prior_table(n, x, config)
        block_scores = np.full((n + 1, n + 1), -np.inf)
        for start in range(n):
            for stop in range(start + 1, n + 1):
                block_scores[start, stop] = float(rng.normal())
        if n >= 4 and rng.random() < 0.3:
            start = int(rng.integers(0, n - 1))
            stop = int(rng.integers(start + 1, n + 1))
            block_scores[start, stop] = -np.inf

        left, right = dp.forward_backward(
            block_scores,
            n,
            k_max,
            log_g_table=prior_scores,
        )
        normalizers = dp.compute_log_C_k(prior_scores, n, k_max)
        assert np.allclose(left[:, n], right[:, 0], equal_nan=True)
        for k in range(1, k_max + 1):
            partitions = list(_partitions(n, k))
            prior_totals = np.array(
                [
                    sum(
                        float(prior_scores[start, stop])
                        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=False)
                    )
                    for boundaries in partitions
                ]
            )
            joint_totals = np.array(
                [
                    _score_partition(boundaries, block_scores, prior_scores)
                    for boundaries in partitions
                ]
            )
            expected_normalizer = float(logsumexp(prior_totals))
            expected_evidence = float(logsumexp(joint_totals))
            assert normalizers[k] == pytest.approx(expected_normalizer, abs=1e-10)
            assert left[k, n] == pytest.approx(expected_evidence, abs=1e-10)
            if np.isfinite(expected_evidence):
                boundaries, map_score = dp.max_sum_segmentation(
                    block_scores,
                    k,
                    log_g_table=prior_scores,
                )
                expected_index = int(np.argmax(joint_totals))
                assert tuple(boundaries) == partitions[expected_index]
                assert map_score == pytest.approx(float(joint_totals[expected_index]), abs=1e-12)


def test_zero_prior_support_is_preserved_by_both_semirings() -> None:
    n = 4
    config = PartitionPriorConfig(parameters={"min_segment_length": 2.0})
    prior_scores = build_log_prior_table(n, None, config)
    block_scores = np.zeros((n + 1, n + 1))
    left, _ = dp.forward_backward(block_scores, n, 2, log_g_table=prior_scores)
    assert left[2, n] == 0.0
    boundaries, score = dp.max_sum_segmentation(block_scores, 2, log_g_table=prior_scores)
    assert boundaries == [0, 2, 4]
    assert score == 0.0


def test_estimator_routes_identical_prior_table_to_sum_and_max_semirings() -> None:
    x = np.array([0.0, 0.2, 0.5, 1.8, 2.0, 3.0])
    y = np.array([0.0, 0.1, 0.2, 1.0, 1.1])
    config = PartitionPriorConfig(
        segment_cohesion="span-power",
        boundary_hazard="poisson-occupancy",
        parameters={"cohesion_power": 0.5, "intensity": 0.7},
    )
    estimator = BayesBreakGaussian(
        k_max=3,
        boundary_coordinates=x,
        partition_prior=config,
    ).fit(x[:-1], y)
    expected_table = build_log_prior_table(y.size, x, config)
    assert np.array_equal(estimator.log_prior_table_, expected_table)

    expected_left, expected_right = dp.forward_backward(
        estimator.log_block_evidence_,
        y.size,
        3,
        log_g_table=expected_table,
    )
    assert np.allclose(estimator.log_left_, expected_left)
    assert np.allclose(estimator.log_right_, expected_right)
    expected_boundaries, expected_score = dp.max_sum_segmentation(
        estimator.log_block_evidence_,
        estimator.k_map_,
        log_g_table=expected_table,
    )
    assert estimator.map_boundaries_ == expected_boundaries
    assert estimator.log_joint_map_ == pytest.approx(expected_score)
