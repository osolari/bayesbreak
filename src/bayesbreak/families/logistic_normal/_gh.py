"""Gauss--Hermite quadrature for the Bernoulli-logistic block evidence."""

from __future__ import annotations

import math

import numpy as np

from ...utils import logsumexp
from ._util import sigmoid


def gh_moments_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    n_gh: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GH quadrature for ``logA0`` and posterior moments (per block)."""

    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    if n_gh < 5:
        raise ValueError("n_gh must be >= 5 for stable quadrature.")

    x, w = np.polynomial.hermite.hermgauss(n_gh)
    theta = nu + math.sqrt(2.0 * rho2) * x  # (n_gh,)

    loglik = np.outer(S, theta) - np.outer(N, np.logaddexp(0.0, theta))
    logw = np.log(w)[None, :]
    log_terms = logw + loglik
    logZ_unnorm = logsumexp(log_terms, axis=1)
    logZ = logZ_unnorm - 0.5 * math.log(math.pi)

    w_norm = np.exp(log_terms - logZ_unnorm[:, None])
    m = w_norm @ theta
    m2 = w_norm @ (theta * theta)
    v = np.maximum(0.0, m2 - m * m)
    p_mean = w_norm @ sigmoid(theta)
    return logZ, m, v, p_mean
