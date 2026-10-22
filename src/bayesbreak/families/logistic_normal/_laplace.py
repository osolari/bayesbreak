"""Laplace approximation for the Bernoulli-logistic block evidence."""

from __future__ import annotations

import math

import numpy as np

from ._util import sigmoid


def laplace_analytic_remainder(
    theta_map: np.ndarray,
    v_post: np.ndarray,
    N: np.ndarray,
) -> np.ndarray:
    r"""Kass--Tierney third-order Laplace remainder bound (per block).

    For an integral ``\int \exp(h(\theta)) d\theta`` approximated by
    ``\exp(h(\theta^*)) \sqrt{2\pi v}`` at the mode ``\theta^*`` with
    curvature ``|h''(\theta^*)| = 1/v``, the standard third-order remainder
    (Kass and Tierney 1989 §2; Tierney and Kadane 1986) is

    .. math::
        |\log Z - \log Z_{\text{Laplace}}| \le
            \tfrac{1}{8} v^2 |h''''(\theta^*)|
            + \tfrac{5}{24} v^3 [h'''(\theta^*)]^2 + O(v^{7/2}).

    For the Bernoulli-logistic block with a normal prior on the log-odds,

    - ``h'''(\theta) = -N \sigma'(\theta) (1 - 2 \sigma(\theta))``;
    - ``h''''(\theta) = -N \sigma'(\theta) (1 - 6 \sigma + 6 \sigma^2)``.

    where ``\sigma' = \sigma (1 - \sigma) \in [0, 1/4]``. The prior
    contribution to derivatives of order >= 3 is zero (it is quadratic in
    ``\theta``). The returned array is a per-block scalar upper bound; it
    is reported as ``tail_bound`` in :class:`SegmentErrorRecord`.
    """

    theta_map = np.asarray(theta_map, dtype=float)
    v_post = np.asarray(v_post, dtype=float)
    N = np.asarray(N, dtype=float)

    sig = sigmoid(theta_map)
    sig_prime = sig * (1.0 - sig)  # sigma'(theta) in [0, 1/4]
    h3 = -N * sig_prime * (1.0 - 2.0 * sig)  # h'''(theta*)
    h4 = -N * sig_prime * (1.0 - 6.0 * sig + 6.0 * sig * sig)  # h''''(theta*)

    return (v_post * v_post) * (np.abs(h4) / 8.0) + (v_post ** 3) * (h3 * h3) * (5.0 / 24.0)


def laplace_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    max_iter: int = 25,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised 1-D Laplace approximation for many blocks.

    Returns
    -------
    logA0 : ndarray
        Laplace-approximate log segment marginal likelihood.
    m, v : ndarray
        Approximate posterior mean and variance of ``\\theta``.
    residual : ndarray
        Terminal ``|\\nabla \\log p(\\theta_{MAP} | y)|`` per block. Reported
        as ``optimization_residual`` in the nonconjugate certificate.
    """

    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)

    p0 = np.clip(S / np.maximum(N, 1e-12), 1e-6, 1.0 - 1e-6)
    theta = np.log(p0) - np.log1p(-p0)
    theta = 0.5 * theta + 0.5 * nu

    inv_rho2 = 1.0 / max(rho2, 1e-12)

    for _ in range(max_iter):
        sig = sigmoid(theta)
        g = (S - N * sig) - (theta - nu) * inv_rho2
        h = -(N * sig * (1.0 - sig) + inv_rho2)
        step = g / np.maximum(-h, 1e-12)
        theta_new = theta + step
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new
            break
        theta = theta_new

    sig = sigmoid(theta)
    h = -(N * sig * (1.0 - sig) + inv_rho2)
    v = 1.0 / np.maximum(-h, 1e-12)
    m = theta

    loglik = S * theta - N * np.logaddexp(0.0, theta)
    logprior = -0.5 * (math.log(2.0 * math.pi * rho2) + (theta - nu) ** 2 / rho2)
    logpost = loglik + logprior

    logA0 = logpost + 0.5 * math.log(2.0 * math.pi) - 0.5 * np.log(np.maximum(-h, 1e-300))
    grad_final = np.abs((S - N * sig) - (theta - nu) * inv_rho2)
    return logA0, m, v, grad_final
