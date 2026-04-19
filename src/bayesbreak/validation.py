"""Input-validation helpers for BayesBreak estimators.

BayesBreak operates on *ordered* data: a one-dimensional response sequence
(optionally multivariate) indexed by a design vector ``X``. scikit-learn, by
contrast, expects ``X`` to be a two-dimensional design matrix. This module
bridges the two conventions by turning sklearn-style ``(X, y)`` inputs into the
``(x_design, y, sample_weight)`` triple the DP layer consumes.

The public entry point is :func:`check_segmentation_input`; the helpers
:func:`check_sample_weight` and :func:`require_fitted` are also re-used by
wrapper estimators (multivariate / grouped / mixture).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def check_segmentation_input(
    X: ArrayLike,
    y: ArrayLike,
    *,
    sample_weight: ArrayLike | None = None,
    multivariate: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate and coerce ``(X, y, sample_weight)`` for a segmenter.

    Parameters
    ----------
    X : array-like of shape (n,) or (n, p)
        Design points / locations for the ordered sequence. When 2-D, the first
        column is interpreted as the design coordinate; the remaining columns
        are accepted but ignored (reserved for future covariate support).
    y : array-like of shape (n,) or (n, d)
        Ordered response. Univariate segmenters require 1-D; multivariate
        segmenters require 2-D.
    sample_weight : array-like of shape (n,), scalar, or None
        Per-observation exposure / precision weight. Interpreted as a power-
        likelihood exponent (see :func:`check_sample_weight`).
    multivariate : bool, default=False
        If True, ``y`` must be 2-D; if False, ``y`` must be 1-D.

    Returns
    -------
    x_design : ndarray of shape (n,), float
        1-D contiguous design-point array (strictly increasing is recommended
        but not enforced).
    y_arr : ndarray of shape (n,) or (n, d), float
        Response array matching the ``multivariate`` contract.
    w_arr : ndarray of shape (n,), float
        Sample weights. ``None`` input becomes all ones.

    Raises
    ------
    ValueError
        If shapes are inconsistent, arrays are empty, or values are non-finite.
    """

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        x_design = X_arr
    elif X_arr.ndim == 2:
        if X_arr.shape[1] == 0:
            raise ValueError("X must have at least one column when 2-D.")
        x_design = X_arr[:, 0]
    else:
        raise ValueError(f"X must be 1-D or 2-D, got shape {X_arr.shape}.")

    n = int(x_design.size)
    if n == 0:
        raise ValueError("Input sequence X must be non-empty.")
    if not np.all(np.isfinite(x_design)):
        raise ValueError("X must contain only finite values.")

    y_arr = np.asarray(y, dtype=float)
    if multivariate:
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.ndim != 2:
            raise ValueError(f"Multivariate y must be 2-D, got shape {y_arr.shape}.")
    else:
        if y_arr.ndim != 1:
            raise ValueError(
                f"y must be 1-D for the univariate segmenter, got shape {y_arr.shape}."
            )

    if y_arr.shape[0] != n:
        raise ValueError(
            f"X and y must agree on the sample dimension: X has {n}, y has {y_arr.shape[0]}."
        )
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite values.")

    w_arr = check_sample_weight(sample_weight, n)
    return (
        np.ascontiguousarray(x_design),
        np.ascontiguousarray(y_arr),
        w_arr,
    )


def check_sample_weight(sample_weight: float | int | ArrayLike | None, n: int) -> FloatArray:
    """Validate and normalise ``sample_weight``.

    Sample weights are treated as **power-likelihood** exponents: each per-
    observation log-likelihood is multiplied by ``w_i``. This is distinct from
    scikit-learn's replication-weight semantics and is documented explicitly
    in the user-facing docstrings.

    Parameters
    ----------
    sample_weight : None, scalar, or array-like of shape (n,)
        Input weights. ``None`` produces all ones.
    n : int
        Expected length of the weight vector.

    Returns
    -------
    ndarray of shape (n,)
        Contiguous float array with non-negative finite entries.
    """

    if sample_weight is None:
        return np.ones(n, dtype=float)
    if np.isscalar(sample_weight):
        return np.full(n, float(sample_weight), dtype=float)

    w = np.asarray(sample_weight, dtype=float)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError(f"sample_weight must be 1-D with length {n}; got shape {w.shape}.")
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")
    return np.ascontiguousarray(w)


def require_fitted(obj: Any, attrs: list[str]) -> None:
    """Raise a uniform error if ``obj`` is missing any fitted attribute."""

    missing = [a for a in attrs if getattr(obj, a, None) is None]
    if missing:
        raise RuntimeError(
            "Estimator is not fitted yet. Call fit() before using this method. "
            f"Missing attributes: {missing}."
        )
