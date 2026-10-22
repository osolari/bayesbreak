"""Pure-math helpers used by the BayesBreak DP and block families.

Input validation lives in :mod:`bayesbreak.validation`.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


try:
    from scipy.special import logsumexp as _scipy_logsumexp
except Exception:  # pragma: no cover - SciPy is optional
    _scipy_logsumexp = None


def logsumexp(
    a: ArrayLike, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> np.ndarray:
    """Stable ``log(sum(exp(a)))`` with SciPy fallback when available."""

    if _scipy_logsumexp is not None:
        return _scipy_logsumexp(a, axis=axis, keepdims=keepdims)

    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    s = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
    out = a_max_safe + np.log(s)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


try:
    from scipy.special import gammaln as _scipy_gammaln
except Exception:  # pragma: no cover
    _scipy_gammaln = None


def gammaln(x: ArrayLike) -> np.ndarray:
    """Element-wise ``log Gamma(x)`` with SciPy fallback."""

    if _scipy_gammaln is not None:
        return _scipy_gammaln(x)
    x_arr = np.asarray(x, dtype=float)
    return np.vectorize(math.lgamma, otypes=[float])(x_arr)


def log_binom(n: int, k: int) -> float:
    r"""Return ``log C(n, k)``; returns ``-inf`` when ``k`` is out of range."""

    if k < 0 or k > n:
        return -np.inf
    return math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
