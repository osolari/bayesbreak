"""Shared numerical utilities for the logistic-normal family."""

from __future__ import annotations

import math

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""

    return 1.0 / (1.0 + np.exp(-x))


def mackay_logistic_normal_mean(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    r"""MacKay's approximation to ``E[\sigma(Z)]`` for ``Z ~ N(m, v)``.

    Used by Laplace / JJ / PG-VB routines to compute the posterior-mean
    sigmoid without an explicit quadrature pass.
    """

    return sigmoid(m / np.sqrt(1.0 + (math.pi * v) / 8.0))
