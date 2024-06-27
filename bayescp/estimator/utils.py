import numpy


# def log_add(x):
#     """
#     Compute the logarithm of the sum of exponentials in a numerically stable way.

#     Parameters:
#     x (numpy.ndarray): input array.

#     Returns:
#     float: The logarithm of the sum of exponentials.
#     """

#     if x.ndim == 1:
#         # For a 1D array, compute the logarithm of the sum of exponentials
#         y = numpy.max(x)
#         if y == -numpy.inf:
#             return y
#         else:
#             return y + numpy.log(numpy.sum(numpy.exp(x - y)))
#     else:
#         # For a 2D array, compute the logarithm of the sum of exponentials along each column
#         y = numpy.max(x, axis=0)
#         Y = numpy.tile(y, (x.shape[0], 1))
#         return y + numpy.log(numpy.sum(numpy.exp(x - Y), axis=0))


def log_add(x: numpy.ndarray):
    """
    Computes the log of the sum of exponentials of input elements.

    Parameters:
    x (numpy.ndarray): input array which can be either 1D or 2D.

    Returns:
    float or numpy.ndarray: The result of log-sum-exp computation.
    """

    # Check if the input is a 1D array
    if x.ndim == 1:
        # Find the maximum value in the array
        y = numpy.max(x)
        # If the maximum value is -inf, return -inf
        if y == -numpy.inf:
            return y
        else:
            # Compute log-sum-exp in a numerically stable way
            return y + numpy.log(numpy.sum(numpy.exp(x - y)))
    else:
        # Find the maximum value for each column
        y = numpy.max(x, axis=0)
        # Create an array by tiling the max values for each column
        Y = numpy.tile(y, (x.shape[0], 1))
        # Compute log-sum-exp for each column in a numerically stable way
        return y + numpy.log(numpy.sum(numpy.exp(x - Y), axis=0))


# def index_la0(r, c, n):
#     """
#     Compute the index using the LA0 indexing formula.

#     Parameters:
#     r (int or tuple): Row index or range.
#     c (int or tuple): Column index or range.
#     n (int): Total number of columns.

#     Returns:
#     numpy.ndarray: Computed indices.
#     """
#     if isinstance(r, int):
#         if isinstance(c, int):
#             # Single row and single column
#             index = c + (r - 1) * (n - r // 2)
#         else:
#             # Single row and column range
#             c1 = list(range(c[0], c[1] + 1))
#             c1 = [x for x in c1 if x >= r]
#             index = numpy.array(c1) + (r - 1) * (n - r // 2)
#     else:
#         # Row range
#         r1 = list(range(r[0], r[1] + 1))
#         r1 = [x for x in r1 if x <= c]
#         index = numpy.array(c) + (numpy.array(r1) - 1) * (n - numpy.array(r1) // 2)
#     return index


def indexLA0(r, c, n):
    """
    Computes an index based on the provided rows, columns, and size parameter.

    Parameters:
    r (int or list): Row index or range of row indices.
    c (int or list): Column index or range of column indices.
    n (int): Size parameter.

    Returns:
    numpy.ndarray: Computed indices.
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
    Estimate global parameters.

    Parameters:
    y (list or numpy.ndarray): input data.
    nu (float): Value for nu parameter. If None, it will be estimated.
    rho_square (float): Value for rho_square parameter. If None, it will be estimated.
    sigma_square (float): Value for sigma_square parameter. If None, it will be estimated.
    type_est_rho (int): Type of estimation for rho_square. 0 for one estimator, 1 for another.

    Returns:
    dict: Dictionary containing estimated parameters nu, rho_square, sigma_square.
    """
    print("Estimation of global parameters")
    n = len(y)
    y = numpy.append(y, y[0])
    m = numpy.sum(y)
    s = numpy.sum(y**2)
    l = numpy.sum((y[0:n] - y[1 : (n + 1)]) ** 2)

    if nu is None:
        nu = m / n

    if sigma_square is None:
        sigma_square = l / (2 * n)

    if rho_square is None:
        if type_est_rho == 1:
            rho_square = (
                numpy.abs(numpy.sum((y[0:n] - m / n) * (y[1 : (n + 1)] - m / n))) / n
            )
        elif type_est_rho == 0:
            rho_square = s / n - (m / n) ** 2
        else:
            print("Error: wrong value for the parameter typeEstRho")

    return {"nu": nu, "rho_square": rho_square, "sigma_square": sigma_square}
