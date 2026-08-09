from __future__ import annotations

import numpy as np

from scripts.phase6_latent_group_rerun import (
    STRESS_CELLS,
    generate_dataset,
    label_invariant_accuracy,
    match_template_distance,
)


def test_archived_cell_reproduces_declared_shape_and_templates() -> None:
    values, labels, templates = generate_dataset(STRESS_CELLS[0], 123)
    assert values.shape == (24, 80)
    assert set(labels.tolist()) == {0, 1}
    assert templates == [[0, 26, 53, 80], [0, 20, 60, 80]]


def test_label_invariant_accuracy_handles_permutation() -> None:
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([1, 1, 0, 0])
    assert label_invariant_accuracy(truth, predicted) == 1.0


def test_template_distance_is_zero_for_permuted_exact_templates() -> None:
    truth = [[0, 20, 60, 80], [0, 26, 53, 80]]
    distance, matches = match_template_distance(list(reversed(truth)), truth)
    assert distance == 0.0
    assert len(matches) == 2


def test_stress_grid_covers_registered_factors() -> None:
    assert any(cell.n_sequences != 24 for cell in STRESS_CELLS)
    assert any(cell.n != 80 for cell in STRESS_CELLS)
    assert any(cell.sigma != 1.0 for cell in STRESS_CELLS)
    assert any(cell.separation != 1.0 for cell in STRESS_CELLS)
    assert any(cell.group0_fraction != 0.5 for cell in STRESS_CELLS)
    assert any(cell.n_groups_fit != 2 for cell in STRESS_CELLS)
    assert any(cell.duplicate_templates for cell in STRESS_CELLS)
