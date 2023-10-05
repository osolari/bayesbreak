import numpy as np
from numpy.testing import assert_allclose

from bayescp.estimator.utils import log_add, index_la0, est_glob_param


def test_log_add_1d():
    # Test for a 1D input array
    x = np.array([1.0, 2.0, 3.0])
    expected_result = np.log(np.sum(np.exp(x)))  # Compute the expected result
    assert_allclose(log_add(x), expected_result, atol=1e-10)


def test_log_add_2d():
    # Test for a 2D input array
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected_result = np.log(np.sum(np.exp(x), axis=0))  # Compute the expected result
    assert_allclose(log_add(x), expected_result, atol=1e-10)


def test_index_la0_single_values():
    # Test for single row and single column
    r = 3
    c = 5
    n = 10
    expected_result = np.array([c + (r - 1) * (n - r // 2)])
    assert np.array_equal(index_la0(r, c, n), expected_result)


def test_index_la0_column_range():
    # Test for single row and column range
    r = 3
    c_range = (5, 8)
    n = 10
    expected_result = np.array([6, 7, 8])  # Computed manually using the formula
    assert np.array_equal(index_la0(r, c_range, n), expected_result)


def test_index_la0_row_range():
    # Test for row range
    r_range = (2, 4)
    c = 5
    n = 10
    expected_result = np.array([9, 14, 19])  # Computed manually using the formula
    assert np.array_equal(index_la0(r_range, c, n), expected_result)


def test_est_glob_param():
    y = [1, 2, 3, 4, 5]
    expected_result = {"nu": 3.0, "rho_square": 2.0, "sigma_square": 2.5}
    assert est_glob_param(y) == expected_result
