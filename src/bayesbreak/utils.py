"""Utility functions for :mod:`bayesbreak`.

The core BayesBreak dynamic program is written in log-space for numerical
stability. This module provides:

- A stable :func:`logsumexp` implementation (SciPy-backed when available).
- Small combinatorial helpers such as :func:`log_binom`.
- Lightweight input validation helpers.

The project keeps SciPy optional; when SciPy is not installed we fall back to
Python's :func:`math.lgamma`.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


# -----------------------------------------------------------------------------
# logsumexp
# -----------------------------------------------------------------------------

try:  # SciPy-backed version (preferred)
    from scipy.special import logsumexp as _scipy_logsumexp
except Exception:  # pragma: no cover
    _scipy_logsumexp = None


def logsumexp(
    a: ArrayLike, axis: Optional[int | tuple[int, ...]] = None, keepdims: bool = False
) -> np.ndarray:
    """Compute ``log(sum(exp(a)))`` in a numerically stable way.

    Parameters
    ----------
    a:
        Input array in log-space.
    axis:
        Axis or axes over which the sum is taken. ``None`` sums all elements.
    keepdims:
        Whether to keep reduced dimensions.

    Returns
    -------
    ndarray
        The log-sum-exp of the input.

    Notes
    -----
    - When SciPy is installed, we delegate to ``scipy.special.logsumexp``.
    - This function treats ``-inf`` as a valid log-probability (contributing 0
      in linear space).
    """

    if _scipy_logsumexp is not None:
        return _scipy_logsumexp(a, axis=axis, keepdims=keepdims)

    # Minimal dependency fallback.
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)

    # If all values are -inf along a reduction axis, max is -inf. Substituting
    # 0.0 prevents invalid operations; exp(-inf - 0) = 0 still.
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
    out = a_max_safe + np.log(s)

    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


# -----------------------------------------------------------------------------
# Combinatorics / special functions
# -----------------------------------------------------------------------------

try:
    from scipy.special import gammaln as _scipy_gammaln
except Exception:  # pragma: no cover
    _scipy_gammaln = None


def gammaln(x: ArrayLike) -> np.ndarray:
    """Compute ``log(Gamma(x))`` element-wise.

    This function uses SciPy when available and otherwise falls back to
    :func:`math.lgamma`.
    """

    if _scipy_gammaln is not None:
        return _scipy_gammaln(x)
    x_arr = np.asarray(x, dtype=float)
    # ``np.vectorize`` is slower than SciPy but keeps SciPy optional.
    return np.vectorize(math.lgamma, otypes=[float])(x_arr)


def log_binom(n: int, k: int) -> float:
    """Return ``log(\binom{n}{k})``.

    Parameters
    ----------
    n:
        Integer ``n``.
    k:
        Integer ``k``.

    Returns
    -------
    float
        The natural logarithm of the binomial coefficient. Returns ``-inf`` if
        ``k`` is outside ``[0, n]``.
    """

    if k < 0 or k > n:
        return -np.inf
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


def as_1d_float_array(x: ArrayLike, *, name: str = "array") -> FloatArray:
    """Convert input to a contiguous 1D float array.

    Parameters
    ----------
    x:
        Input array-like.
    name:
        Name used in error messages.

    Returns
    -------
    ndarray
        1D array of dtype ``float``.
    """

    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(arr)


def check_sample_weight(
    sample_weight: Optional[Union[float, int, ArrayLike]], n: int
) -> FloatArray:
    """Validate and normalize ``sample_weight``.

    The BayesBreak codebase treats sample weights as **power-likelihood**
    exponents: each per-observation log-likelihood contribution is multiplied
    by ``w_i``. This is a standard and convenient way to represent
    heteroscedastic weights, exposures, or replicated observations.

    Parameters
    ----------
    sample_weight:
        ``None`` (all ones), a scalar (broadcasted), or a 1D array-like of
        length ``n``.
    n:
        Number of observations.

    Returns
    -------
    ndarray
        A contiguous float array of shape ``(n,)`` with nonnegative entries.

    Raises
    ------
    ValueError
        If the weights are not compatible with ``n`` or contain invalid values.
    """

    if sample_weight is None:
        return np.ones(n, dtype=float)

    if np.isscalar(sample_weight):
        w = np.full(n, sample_weight, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)

    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"sample_weight must be 1D with length {n}; got shape {w.shape}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(w < 0):  # type: ignore[operator]
        raise ValueError("sample_weight must be nonnegative.")
    return np.ascontiguousarray(w)


def require_fitted(obj: Any, attrs: list[str]) -> None:
    """Raise an informative error if any of ``attrs`` is missing/None."""

    missing = [a for a in attrs if getattr(obj, a, None) is None]
    if missing:
        raise RuntimeError(
            "Estimator is not fitted yet. Call fit() before using this method. "
            f"Missing attributes: {missing}."
        )
