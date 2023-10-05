import numpy as np


def log_add(x):
    """
    Compute the logarithm of the sum of exponentials in a numerically stable way.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    float: The logarithm of the sum of exponentials.
    """
    if x.ndim == 1:
        # For a 1D array, compute the logarithm of the sum of exponentials
        y = np.max(x)
        if y == -np.inf:
            return y
        else:
            return y + np.log(np.sum(np.exp(x - y)))
    else:
        # For a 2D array, compute the logarithm of the sum of exponentials along each column
        y = np.max(x, axis=0)
        Y = np.tile(y, (x.shape[0], 1))
        return y + np.log(np.sum(np.exp(x - Y), axis=0))


def index_la0(r, c, n):
    """
    Compute the index using the LA0 indexing formula.

    Parameters:
    r (int or tuple): Row index or range.
    c (int or tuple): Column index or range.
    n (int): Total number of columns.

    Returns:
    numpy.ndarray: Computed indices.
    """
    if isinstance(r, int):
        if isinstance(c, int):
            # Single row and single column
            index = c + (r - 1) * (n - r // 2)
        else:
            # Single row and column range
            c1 = list(range(c[0], c[1] + 1))
            c1 = [x for x in c1 if x >= r]
            index = np.array(c1) + (r - 1) * (n - r // 2)
    else:
        # Row range
        r1 = list(range(r[0], r[1] + 1))
        r1 = [x for x in r1 if x <= c]
        index = np.array(c) + (np.array(r1) - 1) * (n - np.array(r1) // 2)
    return index


def est_glob_param(y, nu=None, rho_square=None, sigma_square=None, type_est_rho=1):
    """
    Estimate global parameters.

    Parameters:
    y (list or numpy.ndarray): Input data.
    nu (float): Value for nu parameter. If None, it will be estimated.
    rho_square (float): Value for rho_square parameter. If None, it will be estimated.
    sigma_square (float): Value for sigma_square parameter. If None, it will be estimated.
    type_est_rho (int): Type of estimation for rho_square. 0 for one estimator, 1 for another.

    Returns:
    dict: Dictionary containing estimated parameters nu, rho_square, sigma_square.
    """
    print("Estimation of global parameters")
    n = len(y)
    y = np.append(y, y[0])
    m = np.sum(y)
    s = np.sum(y**2)
    l = np.sum((y[0:n] - y[1 : (n + 1)]) ** 2)

    if nu is None:
        nu = m / n

    if sigma_square is None:
        sigma_square = l / (2 * n)

    if rho_square is None:
        if type_est_rho == 1:
            rho_square = np.abs(np.sum((y[0:n] - m / n) * (y[1 : (n + 1)] - m / n))) / n
        elif type_est_rho == 0:
            rho_square = s / n - (m / n) ** 2
        else:
            print("Error: wrong value for the parameter typeEstRho")

    return {"nu": nu, "rho_square": rho_square, "sigma_square": sigma_square}
