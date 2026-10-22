"""Shared pytest fixtures: deterministic synthetic data for BayesBreak tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class SyntheticData:
    X: np.ndarray  # (n, 1) design matrix
    y: np.ndarray  # response
    true_boundaries: list[int]  # inclusive endpoints [0, b1, ..., n]
    true_means: np.ndarray  # per-segment ground truth


def _piecewise(
    means: list[float],
    segment_lengths: list[int],
    noise_fn,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    n = int(sum(segment_lengths))
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.empty(n, dtype=float)
    boundaries = [0]
    cursor = 0
    for mu, length in zip(means, segment_lengths, strict=False):
        y[cursor : cursor + length] = noise_fn(mu, length, rng)
        cursor += length
        boundaries.append(cursor)
    return X, y, boundaries, np.asarray(means, dtype=float)


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(0)
    X, y, b, mu = _piecewise(
        [0.0, 2.0, -1.0, 3.0],
        [40, 40, 40, 40],
        lambda m, L, r: m + 0.25 * r.standard_normal(L),
        rng,
    )
    return SyntheticData(X=X, y=y, true_boundaries=b, true_means=mu)


@pytest.fixture
def poisson_data():
    rng = np.random.default_rng(1)
    X, y, b, mu = _piecewise(
        [2.0, 12.0, 3.0],
        [40, 40, 40],
        lambda m, L, r: r.poisson(m, size=L).astype(float),
        rng,
    )
    return SyntheticData(X=X, y=y, true_boundaries=b, true_means=mu)


@pytest.fixture
def binomial_data():
    rng = np.random.default_rng(2)
    n_trials = 20
    X, y, b, mu = _piecewise(
        [0.1, 0.7, 0.3],
        [40, 40, 40],
        lambda m, L, r: r.binomial(n_trials, m, size=L).astype(float),
        rng,
    )
    return SyntheticData(X=X, y=y, true_boundaries=b, true_means=mu), n_trials


@pytest.fixture
def bernoulli_data():
    rng = np.random.default_rng(3)
    X, y, b, mu = _piecewise(
        [0.15, 0.8, 0.25],
        [40, 40, 40],
        lambda m, L, r: r.binomial(1, m, size=L).astype(float),
        rng,
    )
    return SyntheticData(X=X, y=y, true_boundaries=b, true_means=mu)


@pytest.fixture
def beta_data():
    rng = np.random.default_rng(4)
    X, y, b, mu = _piecewise(
        [0.2, 0.7, 0.3],
        [40, 40, 40],
        lambda m, L, r: np.clip(r.beta(50 * m, 50 * (1 - m), size=L), 1e-3, 1 - 1e-3),
        rng,
    )
    return SyntheticData(X=X, y=y, true_boundaries=b, true_means=mu)


@pytest.fixture
def multivariate_data():
    rng = np.random.default_rng(5)
    n = 120
    X = np.arange(n, dtype=float).reshape(-1, 1)
    # Shared boundaries at 40, 80
    y = np.zeros((n, 3))
    for c, (m1, m2, m3) in enumerate([(0.0, 2.0, -1.0), (1.0, -1.0, 2.0), (-2.0, 0.0, 3.0)]):
        y[:, c] = np.r_[
            m1 + 0.2 * rng.standard_normal(40),
            m2 + 0.2 * rng.standard_normal(40),
            m3 + 0.2 * rng.standard_normal(40),
        ]
    return X, y, [0, 40, 80, 120]


@pytest.fixture
def grouped_sequences():
    """Two groups with different boundary structures."""

    rng = np.random.default_rng(6)
    n = 80
    group_a = [np.r_[rng.normal(0, 0.2, 40), rng.normal(2, 0.2, 40)] for _ in range(4)]
    group_b = [np.r_[rng.normal(1, 0.2, 30), rng.normal(-1, 0.2, 50)] for _ in range(4)]
    X_list = [np.asarray(s, dtype=float) for s in group_a + group_b]
    labels = np.array([0] * 4 + [1] * 4)
    X_arr = np.stack(X_list)
    return X_arr, labels, n
