from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian
from bayesbreak.dp import max_sum_segmentation
from bayesbreak.groups import shared_fit
from bayesbreak.replicates import SharedBoundaryInput, aggregate_block_log_evidence


def _table(n: int, **entries: float) -> np.ndarray:
    table = np.full((n + 1, n + 1), -np.inf)
    for key, value in entries.items():
        start, stop = (int(part) for part in key.split("_"))
        table[start, stop] = value
    return table


def test_aggregate_block_log_evidence_uses_accurate_finite_sum() -> None:
    tables = [
        _table(2, **{"0_1": 1e16, "1_2": -700.0}),
        _table(2, **{"0_1": 1.0, "1_2": 1400.0}),
        _table(2, **{"0_1": -1e16, "1_2": -699.0}),
    ]
    pooled = aggregate_block_log_evidence(SharedBoundaryInput([0.0, 1.0, 2.0], tables))
    assert pooled[0, 1] == 1.0
    assert pooled[1, 2] == 1.0


def test_aggregate_preserves_intersection_of_structural_support() -> None:
    first = _table(2, **{"0_1": 2.0, "0_2": 3.0, "1_2": 4.0})
    second = _table(2, **{"0_1": 5.0, "0_2": 7.0})
    pooled = aggregate_block_log_evidence(SharedBoundaryInput([0.0, 1.0, 2.0], [first, second]))
    assert pooled[0, 1] == 7.0
    assert pooled[0, 2] == 10.0
    assert pooled[1, 2] == -np.inf


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_aggregate_rejects_nonfinite_scores_other_than_zero_support(bad: float) -> None:
    table = _table(2, **{"0_1": bad})
    with pytest.raises(FloatingPointError, match="finite values or -inf"):
        aggregate_block_log_evidence(SharedBoundaryInput([0.0, 1.0, 2.0], [table]))


def test_stable_aggregation_preserves_adversarial_map_ranking() -> None:
    first = _table(
        3,
        **{"0_1": 700.0, "1_3": -699.0, "0_2": -700.0, "2_3": 700.5},
    )
    second = _table(
        3,
        **{"0_1": -699.0, "1_3": 700.0, "0_2": 700.0, "2_3": -700.0},
    )
    pooled = aggregate_block_log_evidence(
        SharedBoundaryInput([0.0, 1.0, 2.0, 3.0], [first, second])
    )
    boundaries, score = max_sum_segmentation(pooled, 2)
    assert boundaries == [0, 1, 3]
    assert score == pytest.approx(2.0)


def test_shared_fit_matches_direct_replicate_model() -> None:
    rng = np.random.default_rng(7)
    sequences = np.stack([rng.normal(size=12), rng.normal(size=12)])
    fitted = shared_fit(
        BayesBreakGaussian(k_max=3),
        np.arange(12).reshape(-1, 1),
        sequences,
    )
    assert fitted.S_ == 2
    assert np.all(
        np.isfinite(fitted.block_posterior_mean_[np.isfinite(fitted.log_block_evidence_)])
    )
