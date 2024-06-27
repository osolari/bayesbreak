import numpy as np
from numpy.testing import assert_allclose

from bprseg.estimator.mlutils import log_add, indexLA0, est_glob_param

import pytest


# log_add tests


def test_log_add_single_column():
    """
    Test case for a 1D array.
    """
    x = np.array([-np.inf, -1, 0, 1])
    assert np.isclose(log_add(x), 1.3862943611198906)


def test_log_add_multiple_columns():
    """
    Test case for a 2D array with multiple columns.
    """
    x = np.array([[-np.inf, -2, 0], [0, 0, 0], [1, -1, -1]])
    expected = np.array([1.31326169, 0.74193734, 1.55144471])
    np.testing.assert_almost_equal(log_add(x), expected)


def test_log_add_inf_values():
    """
    Test case for an array with all -inf values.
    """
    x = np.array([-np.inf, -np.inf, -np.inf])
    assert log_add(x) == -np.inf


def test_log_add_zero_length_array():
    """
    Test case for an empty array.
    """
    x = np.array([])
    assert log_add(x) == -np.inf


# indexLA0 tests


def test_indexLA0_single_row_single_column():
    """
    Test case for a single row and a single column.
    """
    assert np.array_equal(indexLA0(1, 1, 5), [1])


def test_indexLA0_single_row_range_column():
    """
    Test case for a single row and a range of columns.
    """
    assert np.array_equal(indexLA0(1, [1, 3], 5), [1, 2, 3])


def test_indexLA0_range_row_single_column():
    """
    Test case for a range of rows and a single column.
    """
    assert np.array_equal(indexLA0([1, 3], 3, 5), [3, 6, 9])


def test_indexLA0_complex_case():
    """
    Test case for a complex range of rows and columns.
    """
    assert np.array_equal(indexLA0(2, [1, 3], 5), [4, 5, 6])


if __name__ == "__main__":
    pytest.main()
