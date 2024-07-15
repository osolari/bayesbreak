import os
import tempfile
import numpy as np
from numpy.testing import assert_allclose

from bprseg.constants import DATA_RESOURCES_DIR
from bprseg.estimator.mlutils import (
    computeA10,
    computeLA0Vect,
    computeMBPCR,
    computePCReg,
    computeRecursions,
    computeRegrCurve,
    est_profile_with_mbpcr,
    import_cn_data,
    log_add,
    indexLA0,
    est_glob_param,
    plot_est_profile,
    print_est_profile,
)

import pytest


# log_add tests


def test_log_add_simple():

    i = [0.0001, 0.0003, 0.000006]
    x = np.log(i)

    # Compute log of sum of exponentials
    y = log_add(x)
    assert np.isclose(y, -7.809157)

    # Verification
    z = np.sum(i)
    z_exp_y = np.exp(y)

    assert np.isclose(z, z_exp_y)


def test_log_add_single_column():
    """
    Test case for a 1D array.
    """
    x = np.array([-np.inf, -1, 0, 1])
    assert np.isclose(log_add(x), 1.4076059644443804)


def test_log_add_multiple_columns():
    """
    Test case for a 2D array with multiple columns.
    """
    x = np.array([[-np.inf, -2, 0], [0, 0, 0], [1, -1, -1]])
    expected = np.array([1.3132617, 0.407606, 0.8619948])
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
    assert log_add(x).tolist() == x.tolist()


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
    assert np.array_equal(indexLA0([1, 3], 3, 5), [3, 7, 10])


def test_indexLA0_complex_case():
    """
    Test case for a complex range of rows and columns.
    """
    assert np.array_equal(indexLA0(2, [1, 3], 5), [6, 7])


# est_glob_param tests


def test_est_glob_param_rec10k():
    rec10k = import_cn_data(
        os.path.join(DATA_RESOURCES_DIR, "rec10k.tsv"), n_row_skip=1
    )
    out = est_glob_param(rec10k["logratio"])

    assert all(
        [
            np.isclose(x, y, atol=1e-6)
            for x, y in zip(
                out.values(),
                {
                    "nu": -0.024038,
                    "rho_square": 0.088963,
                    "sigma_square": 0.597142,
                }.values(),
            )
        ]
    )


def test_est_glob_param_default():
    y = [1, 2, 3, 4]
    result = est_glob_param(y)
    assert np.isclose(result["nu"], 2.75)
    assert np.isclose(result["rho_square"], 0.1875)
    assert np.isclose(result["sigma_square"], 1.5)


def test_est_glob_param_with_nu():
    y = [1, 2, 3, 4]
    result = est_glob_param(y, nu=2.0)
    assert np.isclose(result["nu"], 2.0)
    assert np.isclose(result["rho_square"], 0.1875)
    assert np.isclose(result["sigma_square"], 1.5)


def test_est_glob_param_with_sigma_square():
    y = [1, 2, 3, 4]
    result = est_glob_param(y, sigma_square=1.0)
    assert np.isclose(result["nu"], 2.75)
    assert np.isclose(result["rho_square"], 0.1875)
    assert np.isclose(result["sigma_square"], 1.0)


def test_est_glob_param_type_est_rho_0():
    y = [1, 2, 3, 4]
    result = est_glob_param(y, type_est_rho=0)
    assert np.isclose(result["nu"], 2.75)
    assert np.isclose(result["rho_square"], 0.1875)
    assert np.isclose(result["sigma_square"], 1.5)


def test_est_glob_param_invalid_type_est_rho():
    y = [1, 2, 3, 4]
    with pytest.raises(
        ValueError, match="Error: wrong value for the parameter type_est_rho"
    ):
        est_glob_param(y, type_est_rho=2)


# computeLA0Vect tests
def test_computeLA0Vect_basic():
    y = np.array([1, 2, 3, 4])
    nu = 2.5
    rho_square = 1.0
    sigma_square = 1.0
    result = computeLA0Vect(y, nu, rho_square, sigma_square)
    expected_length = sum(range(len(y) + 2))
    assert len(result) == expected_length
    assert np.isneginf(result[-1])


def test_computeLA0Vect_custom():
    y = np.array([2, 3, 1, 5])
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


@pytest.fixture
def setup_data():
    path = "test_output/"
    if not os.path.exists(path):
        os.makedirs(path)
    sample_name = "sample1"
    snp_name = np.array(["snp1", "snp2", "snp3"])
    chr = np.array(["chr1", "chr1", "chr2"])
    position = np.array([100, 200, 300])
    logratio = np.array([0.5, -0.5, 0.1])
    chr_to_be_printed = ["chr1", "chr2"]
    est_pc = np.array([0.4, -0.3, 0.05])
    est_boundaries = {"chr1": [0, 1], "chr2": [2]}
    post_prob_t = {"chr1": [0.8, 0.2], "chr2": [0.9]}
    regr_curve = np.array([0.3, -0.2, 0.08])
    regr = 1

    return (
        path,
        sample_name,
        snp_name,
        chr,
        position,
        logratio,
        chr_to_be_printed,
        est_pc,
        est_boundaries,
        post_prob_t,
        regr_curve,
        regr,
    )


def test_print_est_profile(setup_data):
    (
        path,
        sample_name,
        snp_name,
        chr,
        position,
        logratio,
        chr_to_be_printed,
        est_pc,
        est_boundaries,
        post_prob_t,
        regr_curve,
        regr,
    ) = setup_data

    print_est_profile(
        path,
        sample_name,
        snp_name,
        chr,
        position,
        logratio,
        chr_to_be_printed,
        est_pc,
        est_boundaries,
        post_prob_t,
        regr_curve,
        regr,
    )

    # Check if files are created
    assert os.path.exists(f"{path}{sample_name}_mBPCRestimate.txt")
    assert os.path.exists(f"{path}{sample_name}_mBPCRbreakpoints")


# import_cn_data tests


@pytest.fixture
def setup_test_file():
    # Setup a test file
    test_path = "test_cn_data.txt"
    data = """SNP1\tchr1\t100\t0.5
SNP2\tchr1\t200\t1.5
SNP3\tchr2\t300\t2.5
"""
    with open(test_path, "w") as f:
        f.write(data)
    yield test_path
    # Cleanup
    os.remove(test_path)


def test_import_cn_data_with_log_ratio(setup_test_file):
    test_path = setup_test_file
    result = import_cn_data(test_path, n_row_skip=0, if_log_ratio=1)

    assert result["snp_name"] == ["SNP1", "SNP2", "SNP3"]
    assert result["chr"] == ["chr1", "chr1", "chr2"]
    assert result["position"] == [100, 200, 300]
    assert result["logratio"] == [0.5, 1.5, 2.5]


def test_import_cn_data_without_log_ratio(setup_test_file):
    test_path = setup_test_file
    result = import_cn_data(test_path, n_row_skip=0, if_log_ratio=0)

    expected_logratio = np.log([0.5, 1.5, 2.5]) - 1

    assert result["snp_name"] == ["SNP1", "SNP2", "SNP3"]
    assert result["chr"] == ["chr1", "chr1", "chr2"]
    assert result["position"] == [100, 200, 300]
    np.testing.assert_almost_equal(result["logratio"], expected_logratio.tolist())


def test_import_cn_data_invalid_log_ratio(setup_test_file):
    test_path = setup_test_file
    with pytest.raises(
        ValueError, match="Invalid value for if_log_ratio: must be either 0 or 1"
    ):
        import_cn_data(test_path, n_row_skip=0, if_log_ratio=2)


# est_profile_with_mbpcr tests
@pytest.fixture
def sample_data():
    snp_name = ["SNP1", "SNP2", "SNP3", "SNP4"]
    chr = [1, 1, 2, 2]
    position = [100, 200, 300, 400]
    logratio = [0.5, 0.6, 0.7, 0.8]
    chr_to_be_analyzed = [1, 2]
    max_probe_number = 2
    return snp_name, chr, position, logratio, chr_to_be_analyzed, max_probe_number


def test_est_profile_with_mbpcr(sample_data):
    (
        snp_name,
        chr,
        position,
        logratio,
        chr_to_be_analyzed,
        max_probe_number,
    ) = sample_data
    result = est_profile_with_mbpcr(
        snp_name=snp_name,
        chr=chr,
        position=position,
        logratio=logratio,
        chr_to_be_analyzed=chr_to_be_analyzed,
        max_probe_number=max_probe_number,
    )

    assert "est_pc" in result
    assert "est_boundaries" in result
    assert "post_prob_t" in result
    assert len(result["est_pc"]) == len(snp_name)
    assert len(result["est_boundaries"]) == len(chr_to_be_analyzed)


# plot_est_profile tests
@pytest.fixture
def sample_plot_data():
    chr = [1, 1, 2, 2]
    position = [100, 200, 300, 400]
    logratio = [0.5, 0.6, 0.7, 0.8]
    chr_to_be_plotted = [1, 2]
    est_pc = [0.55, 0.65, 0.75, 0.85]
    max_probe_number = 2
    regr_curve = [0.54, 0.64, 0.74, 0.84]
    return (
        chr,
        position,
        logratio,
        chr_to_be_plotted,
        est_pc,
        max_probe_number,
        regr_curve,
    )


def test_plot_est_profile(sample_plot_data):
    (
        chr,
        position,
        logratio,
        chr_to_be_plotted,
        est_pc,
        max_probe_number,
        regr_curve,
    ) = sample_plot_data
    plot_est_profile(
        chr=chr,
        position=position,
        logratio=logratio,
        chr_to_be_plotted=chr_to_be_plotted,
        est_pc=est_pc,
        max_probe_number=max_probe_number,
        regr_curve=regr_curve,
    )
    assert True  # Check if the function runs without error
