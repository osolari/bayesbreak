import tempfile
import numpy as np
from numpy.testing import assert_allclose

from bprseg.estimator.mlutils import (
    computeA10,
    computeLA0Vect,
    computeMBPCR,
    computePCReg,
    computeRecursions,
    computeRegrCurve,
    log_add,
    indexLA0,
    est_glob_param,
    print_est_profile,
)

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


# est_glob_param tests


def test_est_glob_param_default():
    y = [1, 2, 3, 4]
    result = est_glob_param(y)
    assert np.isclose(result["nu"], 2.5)
    assert np.isclose(result["rho_square"], 1.25)
    assert np.isclose(result["sigma_square"], 2.0)


def test_est_glob_param_with_nu():
    y = [1, 2, 3, 4]
    result = est_glob_param(y, nu=2.0)
    assert np.isclose(result["nu"], 2.0)
    assert np.isclose(result["rho_square"], 1.25)
    assert np.isclose(result["sigma_square"], 2.0)


def test_est_glob_param_with_sigma_square():
    y = [1, 2, 3, 4]
    result = est_glob_param(y, sigma_square=1.0)
    assert np.isclose(result["nu"], 2.5)
    assert np.isclose(result["rho_square"], 1.25)
    assert np.isclose(result["sigma_square"], 1.0)


def test_est_glob_param_type_est_rho_0():
    y = [1, 2, 3, 4]
    result = est_glob_param(y, type_est_rho=0)
    assert np.isclose(result["nu"], 2.5)
    assert np.isclose(result["rho_square"], 1.25)
    assert np.isclose(result["sigma_square"], 2.0)


def test_est_glob_param_invalid_type_est_rho():
    y = [1, 2, 3, 4]
    with pytest.raises(
        ValueError, match="Error: wrong value for the parameter type_est_rho"
    ):
        est_glob_param(y, type_est_rho=2)


# computeLA0Vect tests
def test_computeLA0Vect_basic():
    y = [1, 2, 3, 4]
    nu = 2.5
    rho_square = 1.0
    sigma_square = 1.0
    result = computeLA0Vect(y, nu, rho_square, sigma_square)
    expected_length = sum(range(len(y) + 2))
    assert len(result) == expected_length
    assert np.isneginf(result[-1])


def test_computeLA0Vect_custom():
    y = [2, 3, 1, 5]
    nu = 2.75
    rho_square = 1.25
    sigma_square = 2.0
    result = computeLA0Vect(y, nu, rho_square, sigma_square)
    expected_length = sum(range(len(y) + 2))
    assert len(result) == expected_length
    assert np.isneginf(result[-1])


# computeA10 tests
def test_computeA10_single_j():
    i = 0
    j = 3
    y = [1, 2, 3, 4]
    nu = 2.5
    rho_square = 1.0
    sigma_square = 1.0
    result = computeA10(i, j, y, nu, rho_square, sigma_square)
    expected = (rho_square * sum(y[i + 1 : j]) + sigma_square * nu) / (
        (j - i) * rho_square + sigma_square
    )
    assert np.isclose(result, expected)


def test_computeA10_range_j():
    i = 0
    j = [2, 3]
    y = [1, 2, 3, 4]
    nu = 2.5
    rho_square = 1.0
    sigma_square = 1.0
    result = computeA10(i, j, y, nu, rho_square, sigma_square)
    ysum = np.cumsum(y[i + 1 : j[1]] - nu)
    dist = np.arange(j[0] - i, j[1] - i + 1)
    expected = (rho_square * (ysum[dist - 1] + dist * nu) + sigma_square * nu) / (
        dist * rho_square + sigma_square
    )
    assert np.allclose(result, expected)


def test_computeA10_invalid_j():
    i = 0
    j = [2, 3, 4]  # Invalid j length
    y = [1, 2, 3, 4]
    nu = 2.5
    rho_square = 1.0
    sigma_square = 1.0
    with pytest.raises(ValueError, match="Error: wrong value for parameter j"):
        computeA10(i, j, y, nu, rho_square, sigma_square)


# computeRecursions tests
def test_computeRecursions_basic():
    lA0 = np.array(
        [-np.inf, 0.1, 0.2, -np.inf, -np.inf, 0.3, -np.inf, 0.4, -np.inf, -np.inf]
    )
    n = 3
    kMax = 2
    result = computeRecursions(lA0, n, kMax)
    assert result["lL"].shape == (kMax + 1, n + 1)
    assert result["lR"].shape == (kMax + 1, n + 1)


def test_computeRecursions_custom():
    lA0 = np.random.rand(10)
    n = 4
    kMax = 3
    result = computeRecursions(lA0, n, kMax)
    assert result["lL"].shape == (kMax + 1, n + 1)
    assert result["lR"].shape == (kMax + 1, n + 1)


# computeRegrCurve tests
def test_computeRegrCurve_type1():
    y = np.random.rand(10)
    n = len(y)
    kMax = 5
    lL = np.random.rand(kMax + 1, n + 1)
    lR = np.random.rand(kMax + 1, n + 1)
    lA0 = np.random.rand(sum(range(n + 2)))
    nu = np.mean(y)
    rho_square = 1.0
    sigma_square = 1.0
    option = 2
    regrEst = computeRegrCurve(
        y,
        typeRegr=1,
        n=n,
        kMax=kMax,
        lL=lL,
        lR=lR,
        lA0=lA0,
        nu=nu,
        rho_square=rho_square,
        sigma_square=sigma_square,
        option=option,
    )
    assert regrEst.shape == (n,)


def test_computeRegrCurve_type2():
    y = np.random.rand(10)
    n = len(y)
    kMax = 5
    lL = np.random.rand(kMax + 1, n + 1)
    lR = np.random.rand(kMax + 1, n + 1)
    lA0 = np.random.rand(sum(range(n + 2)))
    nu = np.mean(y)
    rho_square = 1.0
    sigma_square = 1.0
    option = np.random.rand(kMax)
    regrEst = computeRegrCurve(
        y,
        typeRegr=2,
        n=n,
        kMax=kMax,
        lL=lL,
        lR=lR,
        lA0=lA0,
        nu=nu,
        rho_square=rho_square,
        sigma_square=sigma_square,
        option=option,
    )
    assert regrEst.shape == (kMax, n)


# computePCReg tests
def test_computePCReg():
    y = np.array([1, 2, 3, 4, 5])
    lA0 = np.random.rand(51)
    lL = np.random.rand(51, 6)
    lR = np.random.rand(51, 6)
    nu = 1.0
    rho_square = 0.5
    sigma_square = 0.2
    kMax = 50
    regr = None

    result = computePCReg(y, lA0, lL, lR, nu, rho_square, sigma_square, kMax, regr)

    assert "kml" in result
    assert "boundaries" in result
    assert "postProbT" in result
    assert "estPC" in result
    assert "estRegr" in result or regr is None


# computeMBPCR( tests
def test_computeMBPCR():
    y = np.array([1, 2, 3, 4, 5])
    kMax = 50
    nu = None
    rho_square = None
    sigma_square = None
    type_est_rho = 1
    regr = None

    result = computeMBPCR(y, kMax, nu, rho_square, sigma_square, type_est_rho, regr)

    assert "estK" in result
    assert "estBoundaries" in result
    assert "estPC" in result
    assert "regrCurve" in result or regr is None
    assert "nu" in result
    assert "rhoSquare" in result
    assert "sigmaSquare" in result
    assert "postProbT" in result


# print_est_profile test
def test_print_est_profile():
    snp_name = np.array(["snp1", "snp2", "snp3", "snp4"])
    chr = np.array([1, 1, 2, 2])
    position = np.array([100, 200, 300, 400])
    logratio = np.array([0.1, 0.2, -0.1, -0.2])
    chr_to_be_printed = [1, 2]
    est_pc = np.array([0.15, 0.25, -0.05, -0.15])
    est_boundaries = {1: [1], 2: [1]}
    post_prob_t = {1: [0.9], 2: [0.8]}
    regr_curve = np.array([0.12, 0.22, -0.08, -0.18])
    regr = 1

    # Call the function
    with tempfile.TemporaryDirectory() as tempdir:
        print_est_profile(
            path=f"{tempdir}",
            sample_name="sample1",
            snp_name=snp_name,
            chr=chr,
            position=position,
            logratio=logratio,
            chr_to_be_printed=chr_to_be_printed,
            est_pc=est_pc,
            est_boundaries=est_boundaries,
            post_prob_t=post_prob_t,
            regr_curve=regr_curve,
            regr=regr,
        )
