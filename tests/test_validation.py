"""Tests for input-validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak.validation import (
    check_sample_weight,
    check_segmentation_input,
    require_fitted,
)


class TestSegmentationInput:
    def test_1d_X_accepted(self):
        x, y, w = check_segmentation_input(np.arange(5), np.zeros(5))
        assert x.shape == (5,) and y.shape == (5,) and w.shape == (5,)

    def test_2d_X_accepted(self):
        X = np.arange(10).reshape(5, 2).astype(float)
        x, _, _ = check_segmentation_input(X, np.zeros(5))
        assert np.allclose(x, X[:, 0])

    def test_multivariate_y(self):
        Y = np.zeros((5, 3))
        _, y, _ = check_segmentation_input(np.arange(5), Y, multivariate=True)
        assert y.shape == (5, 3)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            check_segmentation_input(np.arange(5), np.zeros(4))

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            check_segmentation_input(np.arange(5), np.array([0.0, np.nan, 0.0, 0.0, 0.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            check_segmentation_input(np.array([]), np.array([]))


class TestSampleWeight:
    def test_none_returns_ones(self):
        assert np.allclose(check_sample_weight(None, 3), 1.0)

    def test_scalar_broadcasts(self):
        assert np.allclose(check_sample_weight(2.0, 4), 2.0)

    def test_array_passthrough(self):
        w = np.array([0.5, 1.0, 1.5])
        assert np.allclose(check_sample_weight(w, 3), w)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            check_sample_weight(np.ones(2), 3)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            check_sample_weight(np.array([1.0, -0.1]), 2)


class TestRequireFitted:
    def test_raises_when_attribute_missing(self):
        class Dummy:
            attr_ = None

        with pytest.raises(RuntimeError, match="not fitted"):
            require_fitted(Dummy(), ["attr_"])

    def test_passes_when_set(self):
        class Dummy:
            attr_ = 1

        require_fitted(Dummy(), ["attr_"])  # no error
