import numpy as np

import pandas as pd
from scipy.special import comb

import matplotlib.pyplot as plt


def log_add(x: np.ndarray):
    """
    Computes the log of the sum of exponentials of input elements.

    Parameters:
    x (np.ndarray): input array which can be either 1D or 2D.

    Returns:
    float or np.ndarray: The result of log-sum-exp computation.
    """
    if not x.size:
        return x
    # Check if the input is a 1D array
    if x.ndim == 1:
        # Find the maximum value in the array
        y = np.max(x)
        # If the maximum value is -inf, return -inf
        if y == -np.inf:
            return y
        else:
            # Compute log-sum-exp in a numerically stable way
            return y + np.log(np.sum(np.exp(x - y)))
    else:
        # Find the maximum value for each column
        y = np.max(x, axis=0)
        # Create an array by tiling the max values for each column
        Y = np.tile(y, (x.shape[0], 1))
        # Compute log-sum-exp for each column in a numerically stable way
        return y + np.log(np.sum(np.exp(x - Y), axis=0))


def indexLA0(r, c, n):
    """
    Computes an index based on the provided rows, columns, and size parameter.

    Parameters:
    r (int or list): Row index or range of row indices.
    c (int or list): Column index or range of column indices.
    n (int): Size parameter.

    Returns:
    np.ndarray: Computed indices.
    """
    if isinstance(r, int):  # r is a single integer
        if isinstance(c, int):  # c is a single integer
            index = np.array([c + (r - 1) * (n - r / 2)])
        else:  # c is a list or range
            c1 = np.arange(c[0], c[1] + 1)
            c1 = c1[c1 >= r]
            index = c1 + (r - 1) * (n - r / 2)
        return index
    else:  # r is a list or range
        r1 = np.arange(r[0], r[1] + 1)
        r1 = r1[r1 <= c]
        index = c + (r1 - 1) * (n - r1 / 2)
        return index


def est_glob_param(y, nu=None, rho_square=None, sigma_square=None, type_est_rho=1):
    """
    Estimation of global parameters.

    Parameters:
    y (list or np.ndarray): Input data.
    nu (float, optional): Parameter nu. Defaults to None.
    rho_square (float, optional): Parameter rho^2. Defaults to None.
    sigma_square (float, optional): Parameter sigma^2. Defaults to None.
    type_est_rho (int, optional): Type of rho^2 estimator. Defaults to 1.

    Returns:
    dict: Dictionary containing estimated nu, rho_square, and sigma_square.
    """
    print("Estimation of global parameters")
    n = len(y)
    y = np.append(y, y[0])  # Append the first element to the end of the array
    m = np.sum(y)
    s = np.sum(y**2)
    l = np.sum((y[:n] - y[1 : n + 1]) ** 2)

    if nu is None:
        nu = m / n
    if sigma_square is None:
        sigma_square = l / (2 * n)
    if rho_square is None:
        if type_est_rho == 1:
            # Computation of rho^2 hat 1 estimator
            rho_square = abs(np.sum((y[:n] - m / n) * (y[1 : n + 1] - m / n))) / n
        elif type_est_rho == 0:
            # Computation of rho^2 hat estimator
            rho_square = s / n - (m / n) ** 2
        else:
            raise ValueError("Error: wrong value for the parameter type_est_rho")

    return {"nu": nu, "rho_square": rho_square, "sigma_square": sigma_square}


def computeLA0Vect(y, nu, rho_square, sigma_square):
    """
    Computes the log(A^0) vector based on the provided parameters.

    Parameters:
    y (list or np.ndarray): Input data.
    nu (float): Parameter nu.
    rho_square (float): Parameter rho^2.
    sigma_square (float): Parameter sigma^2.

    Returns:
    np.ndarray: Computed log(A^0) vector.
    """
    n = len(y)
    print("Computation of log(A^0)")
    lA0 = np.full(sum(range(n + 2)), -np.inf)  # Initialize with -Inf

    for i in range(1, n + 1):
        dist = (np.arange(1, n + 2) - i)[i : (n + 1)]
        ysum = np.cumsum(y[i - 1 : n] - nu)
        y2sum = np.cumsum((y[i - 1 : n] - nu) ** 2)
        indices = indexLA0(i, [i + 1, n + 1], n + 1)
        lA0[indices.astype(int)] = (
            -0.5 * dist * np.log(2 * np.pi * sigma_square)
            - 0.5 * np.log(1 + dist * rho_square / sigma_square)
            + 0.5
            / sigma_square
            * (ysum**2 / (dist + sigma_square / rho_square) - y2sum)
        )

    return lA0


def computeA10(i, j, y, nu, rho_square, sigma_square):
    """
    Computes the A1^0 value based on the provided parameters.

    Parameters:
    i (int): Index i.
    j (int or list): Index j or range of indices.
    y (list or numpy.ndarray): Input data.
    nu (float): Parameter nu.
    rho_square (float): Parameter rho^2.
    sigma_square (float): Parameter sigma^2.

    Returns:
    float or numpy.ndarray: Computed A1^0 value(s).
    """
    if isinstance(j, list):
        if len(j) == 2:
            ysum = np.cumsum(y[i : j[1]] - nu)
            dist = np.arange(j[0] - i, j[1] - i + 1)
            a = (rho_square * (ysum[dist - 1] + dist * nu) + sigma_square * nu) / (
                dist * rho_square + sigma_square
            )
        else:
            raise ValueError("Error: wrong value for parameter j")
    else:
        a = (rho_square * np.sum(y[i:j] - nu) + sigma_square * nu) / (
            (j - i) * rho_square + sigma_square
        )
    return a


def computeRecursions(lA0, n, kMax=50):
    """
    Computes the left and right recursions based on the provided parameters.

    Parameters:
    lA0 (numpy.ndarray): Log(A^0) vector.
    n (int): Size parameter.
    kMax (int): Maximum number of recursions.

    Returns:
    dict: Dictionary containing left ('lL') and right ('lR') recursions.
    """
    print("Computation of left and right recursions")
    lL = np.full((kMax + 1, n + 1), -np.inf)
    lR = np.full((kMax + 1, n + 1), -np.inf)
    lL[0, 0] = 0
    lR[0, n] = 0

    for k in range(1, kMax + 1):
        for j in range(1, n + 2):
            if (j - 1) >= k:
                lL[k, j - 1] = log_add(
                    lL[k - 1, k - 1 : (j - 1)] + lA0[indexLA0([k, j - 1], j - 1, n + 1)]
                )
            if (j + 1) <= (n + 1 - k):
                lR[k, j - 1] = log_add(
                    lA0[indexLA0(j - 1, [j, n + 1 - k], n + 1)]
                    + lR[k - 1, j : (n + 1 - k)]
                )

    return {"lL": lL, "lR": lR}


def computeRegrCurve(
    y,
    typeRegr=1,
    n=50,
    kMax=50,
    lL=None,
    lR=None,
    lA0=None,
    nu=0,
    rho_square=0,
    sigma_square=0,
    option=None,
):
    """
    Computes the Bayesian regression curve.

    Parameters:
    y (numpy.ndarray): Input data.
    typeRegr (int): Type of regression (1 or 2).
    n (int): Size parameter.
    kMax (int): Maximum number of recursions.
    lL (numpy.ndarray): Left recursion matrix.
    lR (numpy.ndarray): Right recursion matrix.
    lA0 (numpy.ndarray): Log(A^0) vector.
    nu (float): Parameter nu.
    rho_square (float): Parameter rho^2.
    sigma_square (float): Parameter sigma^2.
    option (numpy.ndarray): Option parameter (kml or lC).

    Returns:
    numpy.ndarray: Computed regression estimates.
    """
    if typeRegr == 1:
        print("Computation of Bayesian regression Curve")
        kml = option
        f = lA0 - lL[kml, n]
        for i in range(1, n + 1):
            a = np.concatenate(
                ([0], computeA10(i - 1, [i, n], y, nu, rho_square, sigma_square))
            )
            if kml > 1:
                f[indexLA0(i, [i, n + 1], n + 1)] = a * np.exp(
                    f[indexLA0(i, [i, n + 1], n + 1)]
                    + np.log(
                        np.dot(
                            np.exp(lL[1:kml, i]),
                            np.exp(lR[kml - np.arange(1, kml + 1), i : n + 1]),
                        )
                    )
                )
            else:
                f[indexLA0(i, [i, n + 1], n + 1)] = a * np.exp(
                    f[indexLA0(i, [i, n + 1], n + 1)]
                    + np.log(np.exp(lL[kml, i]) * np.exp(lR[1, i : n + 1]))
                )

        regrEst = np.zeros(n + 1)
        for h in range(1, n + 1):
            regrEst[h] = (
                regrEst[h - 1]
                + np.sum(f[indexLA0(h, [h + 1, n + 1], n + 1)])
                - np.sum(f[indexLA0([1, h], h, n + 1)])
            )
        regrEst = regrEst[1:]
    else:
        print("Computation of Bayesian regression Curve Ak")
        lC = option
        ff = np.zeros((kMax, n + 1))
        for k in range(1, kMax + 1):
            f = lA0 - lL[k, n]
            for i in range(1, n + 1):
                a = np.concatenate(
                    ([0], computeA10(i - 1, [i, n], y, nu, rho_square, sigma_square))
                )
                if k > 1:
                    f[indexLA0(i, [i, n + 1], n + 1)] = a * np.exp(
                        f[indexLA0(i, [i, n + 1], n + 1)]
                        + np.log(
                            np.dot(
                                np.exp(lL[1:k, i]),
                                np.exp(lR[k - np.arange(1, k + 1), i : n + 1]),
                            )
                        )
                    )
                else:
                    f[indexLA0(i, [i, n + 1], n + 1)] = a * np.exp(
                        f[indexLA0(i, [i, n + 1], n + 1)]
                        + np.log(np.exp(lL[k, i]) * np.exp(lR[1, i : n + 1]))
                    )
            S1 = np.zeros(n + 1)
            for h in range(1, n + 1):
                S1[h] = (
                    S1[h - 1]
                    + np.sum(f[indexLA0(h, [h + 1, n + 1], n + 1)])
                    - np.sum(f[indexLA0([1, h], h, n + 1)])
                )
            ff[k - 1, :] = S1

        regrEst = np.dot(np.exp(lC.T), ff)
        regrEst = regrEst[1:, :]

    return regrEst


def computePCReg(y, lA0, lL, lR, nu, rho_square, sigma_square, kMax=50, regr=None):
    """
    Determines the PC Regression.

    Parameters:
    y (numpy.ndarray): Input data.
    lA0 (numpy.ndarray): Log(A^0) vector.
    lL (numpy.ndarray): Left recursion matrix.
    lR (numpy.ndarray): Right recursion matrix.
    nu (float): Parameter nu.
    rho_square (float): Parameter rho^2.
    sigma_square (float): Parameter sigma^2.
    kMax (int): Maximum number of recursions.
    regr (int or None): Regression type.

    Returns:
    dict: Contains kml, boundaries, postProbT, estPC, and estRegr.
    """
    n = len(y)
    probK = 1 / kMax
    print("Determination of PC Regression")

    lC = (
        lL[np.arange(1, kMax + 1), n]
        - np.log(comb(n - 1, np.arange(1, kMax + 1) - 1))
        + np.log(probK)
    )
    lE = log_add(lC)
    lC -= lE

    ek = np.sum(np.arange(1, kMax + 1) * np.exp(lC))
    w = np.where(lC > -np.inf)[0]
    err = w**2 - 2 * w * ek
    kml = w[np.argmin(err)]

    if kml > 1:
        dd = np.zeros((kMax - 1, n - 1))
        for kk in range(2, kMax + 1):
            for i in range(2, n + 1):
                dd[kk - 2, i - 2] = log_add(
                    lL[1:kk, i - 1] + lR[kk - 1 :: -1, i - 1] - lL[kk, n]
                )

        d1 = np.zeros(n - 1)
        for i in range(1, n):
            d1[i - 1] = np.exp(log_add(dd[:, i - 1] + lC[1:kMax]))

        s1 = np.sort(d1)
        boundaries = np.zeros(kml - 1, dtype=int)
        i = 0
        while i < kml - 1:
            indices = np.where(d1 == s1[n - i - 1])[0]
            if len(indices) == 1:
                boundaries[i] = indices[0]
                i += 1
            else:
                boundaries[i : i + len(indices)] = indices
                i += len(indices)

        boundaries = np.sort(boundaries)
        boundaries = np.concatenate(([0], boundaries, [n]))
    else:
        boundaries = np.array([0, n])
        d1 = np.zeros(n - 1)

    if regr is not None:
        if regr == 1:
            estRegr = computeRegrCurve(
                y, regr, n, kMax, lL, lR, lA0, nu, rho_square, sigma_square, kml
            )
        else:
            estRegr = computeRegrCurve(
                y, regr, n, kMax, lL, lR, lA0, nu, rho_square, sigma_square, lC
            )
    else:
        estRegr = None

    mT = np.zeros(n)
    for k in range(kml):
        m1 = computeA10(
            boundaries[k], boundaries[k + 1], y, nu, rho_square, sigma_square
        )
        mT[boundaries[k] : (boundaries[k + 1])] = m1

    return {
        "kml": kml,
        "boundaries": boundaries,
        "postProbT": d1,
        "estPC": mT,
        "estRegr": estRegr,
    }


def computeMBPCR(
    y, kMax=50, nu=None, rho_square=None, sigma_square=None, type_est_rho=1, regr=None
):
    """
    Computes the MBPCR.

    Parameters:
    y (numpy.ndarray): Input data.
    kMax (int): Maximum number of recursions.
    nu (float): Parameter nu.
    rho_square (float): Parameter rho^2.
    sigma_square (float): Parameter sigma^2.
    type_est_rho (int): Type of rho estimation.
    regr (int or None): Regression type.

    Returns:
    dict: Contains estK, estBoundaries, estPC, regrCurve, nu, rhoSquare, sigmaSquare, postProbT.
    """
    n = len(y)
    kMax = min(kMax, n)

    if nu is None or rho_square is None or sigma_square is None:
        results = est_glob_param(y, nu, rho_square, sigma_square, type_est_rho)
        nu = results["nu"]
        rho_square = results["rho_square"]
        sigma_square = results["sigma_square"]

    lA0 = computeLA0Vect(y, nu, rho_square, sigma_square)
    recursions = computeRecursions(lA0, n, kMax)
    lL = recursions["lL"]
    lR = recursions["lR"]

    if regr is not None and regr not in [1, 2]:
        print("Error: wrong value for parameter regr")
        return None

    pc_reg_results = computePCReg(
        y, lA0, lL, lR, nu, rho_square, sigma_square, kMax, regr
    )

    return {
        "estK": pc_reg_results["kml"],
        "estBoundaries": pc_reg_results["boundaries"][1:],
        "estPC": pc_reg_results["estPC"],
        "regrCurve": pc_reg_results["estRegr"],
        "nu": nu,
        "rhoSquare": rho_square,
        "sigmaSquare": sigma_square,
        "postProbT": pc_reg_results["postProbT"],
    }


def print_est_profile(
    path="",
    sample_name="",
    snp_name=None,
    chr=None,
    position=None,
    logratio=None,
    chr_to_be_printed=None,
    est_pc=None,
    est_boundaries=None,
    post_prob_t=None,
    regr_curve=None,
    regr=None,
):
    """
    Print and save estimated profile data to files.

    Parameters:
    path (str): Path for the output files.
    sample_name (str): Sample name.
    snp_name (list): List of SNP names.
    chr (list): List of chromosomes.
    position (list): List of positions.
    logratio (list): List of log ratios.
    chr_to_be_printed (list): List of chromosomes to be printed.
    est_pc (list): List of estimated PC values.
    est_boundaries (list): List of estimated boundaries.
    post_prob_t (list): List of posterior probabilities.
    regr_curve (list): List of regression curves.
    regr (int): Regression type.

    Returns:
    None
    """
    if (
        snp_name is None
        or chr is None
        or position is None
        or logratio is None
        or chr_to_be_printed is None
        or est_pc is None
    ):
        print("Error: Missing required parameter(s)")
        return None

    path_results1 = f"{path}{sample_name}_mBPCRestimate.txt"
    data11 = pd.DataFrame(
        {
            "SNP_name": snp_name,
            "chromosome": chr,
            "position": position,
            "rawLog2ratio": logratio,
            "mBPCR_estimate": est_pc,
        }
    )
    data11.to_csv(path_results1, sep="\t", index=False, header=False)

    if est_boundaries is not None:
        path_results2 = f"{path}{sample_name}_mBPCRbreakpoints.txt"
        if post_prob_t is not None:
            data21 = pd.DataFrame(
                {
                    "SNP_name(start)": [],
                    "SNP_name(end)": [],
                    "chromosome": [],
                    "position(start)": [],
                    "position(end)": [],
                    "n_probes": [],
                    "mBPCR_estimate": [],
                    "breakpointPostProb": [],
                }
            )
        else:
            data21 = pd.DataFrame(
                {
                    "SNP_name(start)": [],
                    "SNP_name(end)": [],
                    "chromosome": [],
                    "position(start)": [],
                    "position(end)": [],
                    "n_probes": [],
                    "mBPCR_estimate": [],
                }
            )
        data21.to_csv(path_results2, sep="\t", index=False, header=False)
    else:
        if post_prob_t is not None:
            print("Error: estBoundaries=NULL while posteriorProbT!=NULL")
            return None

    if regr_curve is not None and not np.all(np.isnan(regr_curve)):
        if regr is None or regr not in [1, 2]:
            print("Error: wrong value for parameter regr")
            return None
        else:
            if regr == 1:
                path_results3 = f"{path}{sample_name}_mBRCestimate.txt"
                data31 = pd.DataFrame(
                    {
                        "SNP_name": snp_name,
                        "chromosome": chr,
                        "position": position,
                        "rawLog2ratio": logratio,
                        "mBRC_estimate": regr_curve,
                    }
                )
            else:
                path_results3 = f"{path}{sample_name}_BRCAkestimate.txt"
                data31 = pd.DataFrame(
                    {
                        "SNP_name": snp_name,
                        "chromosome": chr,
                        "position": position,
                        "rawLog2ratio": logratio,
                        "BRCAk_estimate": regr_curve,
                    }
                )
            data31.to_csv(path_results3, sep="\t", index=False, header=False)

    for j in chr_to_be_printed:
        data12 = pd.DataFrame(
            {
                "SNP_name": snp_name[chr == j],
                "chromosome": chr[chr == j],
                "position": position[chr == j],
                "rawLog2ratio": logratio[chr == j],
                "mBPCR_estimate": est_pc[chr == j],
            }
        )
        data12.to_csv(path_results1, sep="\t", index=False, header=False, mode="a")

        if est_boundaries is not None:
            start_boundaries = est_boundaries[j] + 1
            if len(est_pc[chr == j]) != len(chr[chr == j]):
                if est_boundaries[j][-1] == len(chr[chr == j]):
                    start_boundaries = np.concatenate(
                        (
                            [1],
                            np.where(np.isnan(est_pc[chr == j]))[0] + 1,
                            start_boundaries[:-1],
                        )
                    )
                    est_boundaries[j] = np.concatenate(
                        (np.where(np.isnan(est_pc[chr == j]))[0], est_boundaries[j])
                    )
                    if post_prob_t is not None:
                        post_prob_t[j] = np.concatenate(([np.nan], post_prob_t[j]))
                else:
                    start_boundaries = np.concatenate(
                        (
                            [1],
                            start_boundaries[:-1],
                            [np.where(np.isnan(est_pc[chr == j]))[0] + 1],
                        )
                    )
                    est_boundaries[j] = np.concatenate(
                        (est_boundaries[j], [len(chr[chr == j])])
                    )
                    if post_prob_t is not None:
                        post_prob_t[j] = np.concatenate((post_prob_t[j], [np.nan]))
            else:
                start_boundaries = np.concatenate(([1], start_boundaries[:-1]))

            if post_prob_t is not None:
                data22 = pd.DataFrame(
                    {
                        "SNP_name(start)": snp_name[chr == j][start_boundaries],
                        "SNP_name(end)": snp_name[chr == j][est_boundaries[j]],
                        "chromosome": [j] * len(est_boundaries[j]),
                        "position(start)": position[chr == j][start_boundaries],
                        "position(end)": position[chr == j][est_boundaries[j]],
                        "n_probes": est_boundaries[j] - start_boundaries + 1,
                        "mBPCR_estimate": est_pc[chr == j][est_boundaries[j]],
                        "breakpointPostProb": post_prob_t[j],
                    }
                )
            else:
                data22 = pd.DataFrame(
                    {
                        "SNP_name(start)": snp_name[chr == j][start_boundaries],
                        "SNP_name(end)": snp_name[chr == j][est_boundaries[j]],
                        "chromosome": [j] * len(est_boundaries[j]),
                        "position(start)": position[chr == j][start_boundaries],
                        "position(end)": position[chr == j][est_boundaries[j]],
                        "n_probes": est_boundaries[j] - start_boundaries + 1,
                        "mBPCR_estimate": est_pc[chr == j][est_boundaries[j]],
                    }
                )
            data22.to_csv(path_results2, sep="\t", index=False, header=False, mode="a")

        if regr_curve is not None and not np.all(np.isnan(regr_curve)):
            data32 = pd.DataFrame(
                {
                    "SNP_name": snp_name[chr == j],
                    "chromosome": chr[chr == j],
                    "position": position[chr == j],
                    "rawLog2ratio": logratio[chr == j],
                    "regrCurve": regr_curve[chr == j],
                }
            )
            data32.to_csv(path_results3, sep="\t", index=False, header=False, mode="a")


def import_cn_data(path, n_row_skip, if_log_ratio=1):
    """
    Import CN data from a file.

    Parameters:
    path (str): Path to the input file.
    n_row_skip (int): Number of rows to skip at the beginning of the file.
    if_log_ratio (int): Indicator whether the log ratio is already provided (1) or needs to be calculated (0).

    Returns:
    dict: A dictionary with keys 'snp_name', 'chr', 'position', and 'logratio'.
    """
    # Read the table, skipping the specified number of rows
    results = pd.read_csv(path, sep="\t", skiprows=n_row_skip, header=None)

    # Check if we need to compute the log ratio
    if if_log_ratio == 1:
        return {
            "snp_name": results[0].tolist(),
            "chr": results[1].tolist(),
            "position": results[2].tolist(),
            "logratio": results[3].tolist(),
        }
    elif if_log_ratio == 0:
        logratio = np.log(results[3].astype(float)) - 1
        return {
            "snp_name": results[0].tolist(),
            "chr": results[1].tolist(),
            "position": results[2].tolist(),
            "logratio": logratio.tolist(),
        }

    # In case of an invalid if_log_ratio value, raise an error
    raise ValueError("Invalid value for if_log_ratio: must be either 0 or 1")


def est_profile_with_mbpcr(
    path="",
    sample_name="",
    snp_name=None,
    chr=None,
    position=None,
    logratio=None,
    chr_to_be_analyzed=None,
    max_probe_number=None,
    rho_square=None,
    k_max=50,
    nu=None,
    sigma_square=None,
    type_est_rho=1,
    regr=None,
):
    """
    Estimate profile using MBPCR.

    Parameters:
    path (str): Path to save the results.
    sample_name (str): Sample name.
    snp_name (list): List of SNP names.
    chr (list): List of chromosomes.
    position (list): List of positions.
    logratio (list): List of log ratios.
    chr_to_be_analyzed (list): Chromosomes to be analyzed.
    max_probe_number (int): Maximum number of probes.
    rho_square (float): Rho squared value.
    k_max (int): Maximum value for k.
    nu (float): Nu value.
    sigma_square (float): Sigma squared value.
    type_est_rho (int): Type of rho estimation.
    regr (int): Regression parameter.

    Returns:
    dict: Dictionary containing estimated profile data.
    """
    pos_centromere = [
        124200000,
        93400000,
        91700000,
        50900000,
        47700000,
        60500000,
        58900000,
        45200000,
        50600000,
        40300000,
        52900000,
        35400000,
        16000000,
        15600000,
        17000000,
        38200000,
        22200000,
        16100000,
        28500000,
        27100000,
        12300000,
        11800000,
    ]

    # Estimate global parameters if not provided
    if nu is None or rho_square is None or sigma_square is None:
        results = est_glob_param(logratio, nu, rho_square, sigma_square, type_est_rho)
        nu = results["nu"]
        rho_square = results["rho_square"]
        sigma_square = results["sigma_square"]

    index_no = []
    est_pc = np.full(len(snp_name), np.nan)
    est_boundaries = [None] * len(chr_to_be_analyzed)
    post_prob_t = [None] * len(chr_to_be_analyzed)
    regr_curve = np.full(len(snp_name), np.nan)

    for j in chr_to_be_analyzed:
        y = logratio[np.array(chr) == j]
        n = len(y)

        if n <= max_probe_number:
            print(f"Estimation of the profile of chromosome {j}")
            results = computeMBPCR(
                y, k_max, nu, rho_square, sigma_square, type_est_rho, regr
            )
            est_pc[np.array(chr) == j] = results["est_pc"]
            if regr is not None:
                regr_curve[np.array(chr) == j] = results["regr_curve"]
            est_boundaries[chr_to_be_analyzed.index(j)] = results["est_boundaries"]
            post_prob_t[chr_to_be_analyzed.index(j)] = np.append(
                results["post_prob_t"][results["est_boundaries"][:-1]], 1
            )

        else:
            a = pos_centromere[j - 1] if j != "X" else pos_centromere[22]
            a = np.argmax(position[np.array(chr) == j] > a)
            bounds1, post_prob1 = [], []

            if a > max_probe_number and n - a > max_probe_number:
                print(
                    f"Warning: the profile of chromosome {j} has not been estimated because of its size"
                )
                index_no.append(chr_to_be_analyzed.index(j))
                continue

            print(f"Estimation of the profile of chromosome {j}")
            if a <= max_probe_number:
                y = y[:a]
                results = computeMBPCR(
                    y, k_max, nu, rho_square, sigma_square, type_est_rho, regr
                )
                est_pc[np.array(chr) == j][:a] = results["est_pc"]
                if regr is not None:
                    regr_curve[np.array(chr) == j][:a] = results["regr_curve"]
                bounds1.extend(results["est_boundaries"])
                post_prob1.extend(
                    np.append(results["post_prob_t"][results["est_boundaries"][:-1]], 1)
                )

            if n - a <= max_probe_number:
                y = logratio[np.array(chr) == j][a:]
                results = computeMBPCR(
                    y, k_max, nu, rho_square, sigma_square, type_est_rho, regr
                )
                est_pc[np.array(chr) == j][a:] = results["est_pc"]
                if regr is not None:
                    regr_curve[np.array(chr) == j][a:] = results["regr_curve"]
                bounds1.extend(a + np.array(results["est_boundaries"]))
                post_prob1.extend(
                    np.append(results["post_prob_t"][results["est_boundaries"][:-1]], 1)
                )

            if a <= max_probe_number or n - a <= max_probe_number:
                est_boundaries[chr_to_be_analyzed.index(j)] = bounds1
                post_prob_t[chr_to_be_analyzed.index(j)] = post_prob1

            if a <= max_probe_number and n - a > max_probe_number:
                print(
                    f"Warning: the profile of arm q of chromosome {j} has not been estimated because of its size"
                )

            if a > max_probe_number and n - a <= max_probe_number:
                print(
                    f"Warning: the profile of arm p of chromosome {j} has not been estimated because of its size"
                )

    if path:
        if index_no:
            print_est_profile(
                path,
                sample_name,
                snp_name,
                chr,
                position,
                logratio,
                chr_to_be_analyzed=[
                    c for i, c in enumerate(chr_to_be_analyzed) if i not in index_no
                ],
                est_pc=est_pc,
                est_boundaries=[
                    e for i, e in enumerate(est_boundaries) if i not in index_no
                ],
                post_prob_t=[p for i, p in enumerate(post_prob_t) if i not in index_no],
                regr_curve=regr_curve,
                regr=regr,
            )
        else:
            print_est_profile(
                path,
                sample_name,
                snp_name,
                chr,
                position,
                logratio,
                chr_to_be_analyzed=chr_to_be_analyzed,
                est_pc=est_pc,
                est_boundaries=est_boundaries,
                post_prob_t=post_prob_t,
                regr_curve=regr_curve,
                regr=regr,
            )

    if regr is not None:
        return {
            "est_pc": est_pc,
            "est_boundaries": est_boundaries,
            "post_prob_t": post_prob_t,
            "regr_curve": regr_curve,
        }
    return {
        "est_pc": est_pc,
        "est_boundaries": est_boundaries,
        "post_prob_t": post_prob_t,
    }


def plot_est_profile(
    path="",
    sample_name="",
    chr=None,
    position=None,
    logratio=None,
    chr_to_be_plotted=None,
    est_pc=None,
    max_probe_number=None,
    legend_position="lower left",
    regr_curve=None,
    regr=None,
):
    """
    Plot estimated profile.

    Parameters:
    path (str): Path to save the plots.
    sample_name (str): Sample name.
    chr (list): List of chromosomes.
    position (list): List of positions.
    logratio (list): List of log ratios.
    chr_to_be_plotted (list): Chromosomes to be plotted.
    est_pc (list): Estimated profile data.
    max_probe_number (int): Maximum number of probes.
    legend_position (str): Position of the legend.
    regr_curve (list): Regression curve data.
    regr (int): Regression parameter.

    Returns:
    None
    """
    pos_centromere = [
        124200000,
        93400000,
        91700000,
        50900000,
        47700000,
        60500000,
        58900000,
        45200000,
        50600000,
        40300000,
        52900000,
        35400000,
        16000000,
        15600000,
        17000000,
        38200000,
        22200000,
        16100000,
        28500000,
        27100000,
        12300000,
        11800000,
    ]

    for j in chr_to_be_plotted:
        plt.figure()
        plt.scatter(
            position[np.array(chr) == j],
            logratio[np.array(chr) == j],
            color="grey",
            s=2,
        )
        plt.xlabel(f"Chromosome {j}")
        plt.ylabel("log2ratio")
        plt.title(sample_name)

        if est_pc is not None:
            if regr is None:
                plt.legend(["mBPCR"], loc=legend_position)
            else:
                if regr == 1:
                    plt.legend(["mBPCR", "BRC with K_2"], loc=legend_position)
                elif regr == 2:
                    plt.legend(["mBPCR", "BRCAk"], loc=legend_position)
                else:
                    print("Error: wrong value for parameter regr")
                    return None

            n = np.sum(np.array(chr) == j)
            if n <= max_probe_number:
                plt.plot(
                    position[np.array(chr) == j],
                    est_pc[np.array(chr) == j],
                    color="blue",
                )
                if regr is not None and (regr == 1 or regr == 2):
                    plt.plot(
                        position[np.array(chr) == j],
                        regr_curve[np.array(chr) == j],
                        color="red",
                    )
            else:
                a = pos_centromere[j - 1] if j != "X" else pos_centromere[22]
                a = np.argmax(position[np.array(chr) == j] > a)
                plt.plot(
                    position[np.array(chr) == j][:a],
                    est_pc[np.array(chr) == j][:a],
                    color="blue",
                )
                plt.plot(
                    position[np.array(chr) == j][a:],
                    est_pc[np.array(chr) == j][a:],
                    color="blue",
                )
                if regr is not None and (regr == 1 or regr == 2):
                    plt.plot(
                        position[np.array(chr) == j][:a],
                        regr_curve[np.array(chr) == j][:a],
                        color="red",
                    )
                    plt.plot(
                        position[np.array(chr) == j][a:],
                        regr_curve[np.array(chr) == j][a:],
                        color="red",
                    )

            if path:
                if regr is None:
                    plt.savefig(f"{path}{sample_name}_chr{j}_mBPCR.eps")
                else:
                    if regr == 1:
                        plt.savefig(f"{path}{sample_name}_chr{j}_mBPCR&mBRC.eps")
                    elif regr == 2:
                        plt.savefig(f"{path}{sample_name}_chr{j}_mBPCR&BRCAk.eps")
        else:
            if regr == 1:
                plt.legend(["BRC with K_2"], loc=legend_position)
            elif regr == 2:
                plt.legend(["BRCAk"], loc=legend_position)
            else:
                print("Error: wrong value for parameter regr")
                return None

            n = np.sum(np.array(chr) == j)
            if n <= max_probe_number:
                plt.plot(
                    position[np.array(chr) == j],
                    regr_curve[np.array(chr) == j],
                    color="blue",
                )
            else:
                a = pos_centromere[j - 1] if j != "X" else pos_centromere[22]
                a = np.argmax(position[np.array(chr) == j] > a)
                plt.plot(
                    position[np.array(chr) == j][:a],
                    regr_curve[np.array(chr) == j][:a],
                    color="blue",
                )
                plt.plot(
                    position[np.array(chr) == j][a:],
                    regr_curve[np.array(chr) == j][a:],
                    color="blue",
                )

            if path:
                if regr == 1:
                    plt.savefig(f"{path}{sample_name}_chr{j}_mBRC.eps")
                elif regr == 2:
                    plt.savefig(f"{path}{sample_name}_chr{j}_BRCAk.eps")

        plt.close()
