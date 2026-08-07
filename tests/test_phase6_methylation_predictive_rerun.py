from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.phase6_methylation_predictive_rerun import (
    BLOCK_SIZE,
    EXPECTED_N,
    N_SPLITS,
    build_splits,
    interval_summary,
    map_boundaries_to_original,
    verify_source,
)


def test_stratified_blocks_are_deterministic_disjoint_and_in_support() -> None:
    first = build_splits()
    second = build_splits()
    assert len(first) == len(second) == N_SPLITS
    occupied: set[int] = set()
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left["test_indices"], right["test_indices"])
        assert left["test_indices"].size == BLOCK_SIZE
        assert left["train_indices"][0] == 0
        assert left["train_indices"][-1] == EXPECTED_N - 1
        assert not occupied.intersection(left["test_indices"].tolist())
        occupied.update(left["test_indices"].tolist())


def test_block_builder_rejects_strata_smaller_than_block() -> None:
    with pytest.raises(ValueError, match="stratum"):
        build_splits(n=20, n_splits=10, block_size=3)


def test_boundary_mapping_returns_original_cpg_indices() -> None:
    train_indices = np.array([0, 1, 5, 6, 7, 9])
    assert map_boundaries_to_original(train_indices, [0, 2, 4, 6]) == [5, 7]


def test_interval_summary_is_exact_for_constant_values() -> None:
    summary = interval_summary([2.5] * 10)
    assert summary["mean"] == 2.5
    assert summary["standard_error"] == 0.0
    assert summary["ci95_lower"] == summary["ci95_upper"] == 2.5


def test_wrong_source_hash_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "test1.myCpG.txt"
    source.write_text("not the authorized methylKit source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash"):
        verify_source(source)
