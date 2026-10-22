"""Jaakkola--Jordan variational bound for the Bernoulli-logistic block."""

from __future__ import annotations

import math

import numpy as np


def _safe_lambda_jj(xi: np.ndarray) -> np.ndarray:
    """``lambda(xi) = tanh(xi/2) / (4 xi)`` with the ``xi -> 0`` limit ``1/8``."""

    xi = np.asarray(xi, dtype=float)
    out = np.empty_like(xi)
    small = np.abs(xi) < 1e-10
    out[small] = 1.0 / 8.0
    xs = xi[~small]
    out[~small] = np.tanh(xs / 2.0) / (4.0 * xs)
    return out


def jj_analytic_remainder(
    m: np.ndarray,
    v: np.ndarray,
    S: np.ndarray,
    N: np.ndarray,
    *,
    n_gh: int = 60,
) -> np.ndarray:
    r"""JJ tangent-gap diagnostic under the fitted Gaussian variational distribution; NOT an evidence-error certificate.

    The Jaakkola--Jordan variational tangent bounds ``log σ(x)`` from
    below at every ``x``, so ``log Z >= log Z_JJ`` for every block. At
    the JJ fixed point ``ξ² = m² + v`` the excess is

    .. math::
        \mathrm{gap}_{q}
        \;=\;
        \mathbb{E}_{q}\!\bigl[\log p(y \mid \theta) - T(\theta;\xi)\bigr]

    where ``T`` is the JJ tangent-lower-bound of the log-likelihood and
    ``q = \mathcal{N}(m, v)`` is the JJ-implied Gaussian posterior over
    the log-odds. For the Bernoulli-logistic block with sufficient
    statistics ``(S, N)`` and effective weighted sample size ``N``,

    .. math::
        \log p(y \mid \theta)
        = S \theta - N \log(1 + e^{\theta}), \qquad
        T(\theta; \xi)
        = S \theta - N \bigl[\lambda(\xi) \theta^{2}
                + \tfrac{1}{2}(\theta + \xi)
                + \log(1 + e^{-\xi})
                - \lambda(\xi) \xi^{2}\bigr].

    Both terms are integrated in one-block Gauss--Hermite quadrature
    against ``\mathcal{N}(m, v)``; the ``S`` terms cancel exactly,
    leaving a computable, non-negative per-block remainder that
    quantifies the JJ underapproximation without a global constant.
    """

    m = np.atleast_1d(np.asarray(m, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    N = np.atleast_1d(np.asarray(N, dtype=float))
    xi = np.sqrt(m * m + v)
    lam = _safe_lambda_jj(xi)

    x, w = np.polynomial.hermite.hermgauss(n_gh)
    scale = math.sqrt(2.0)
    w_norm = w / math.sqrt(math.pi)
    # theta samples under N(m, v) at Gauss-Hermite nodes: (B, Q)
    theta = m[:, None] + scale * np.sqrt(v)[:, None] * x[None, :]

    E_softplus = (w_norm[None, :] * np.logaddexp(0.0, theta)).sum(axis=1)
    # Under N(m, v): E[theta] = m, E[theta^2] = m^2 + v. The JJ upper
    # tangent for log(1 + e^theta), evaluated in expectation, is
    #   E_q[T(theta; xi)] = -log sigma(xi) + 0.5(m + xi) + lambda(xi)(m^2 + v - xi^2).
    # At the JJ fixed point xi^2 = m^2 + v the last term is exactly zero;
    # we keep it in the formula so a residual on that identity from an
    # unconverged fit is reported honestly.
    E_upper_tangent = (
        np.logaddexp(0.0, -xi)
        + 0.5 * (m + xi)
        + lam * (m * m + v - xi * xi)
    )
    # This is an expectation under q, not log Z - log Z_JJ.
    return N * (E_upper_tangent - E_softplus)


def jj_block(
    S: np.ndarray,
    N: np.ndarray,
    nu: float,
    rho2: float,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JJ variational lower bound on the binomial-logistic block evidence."""

    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    inv_rho2 = 1.0 / max(rho2, 1e-12)

    p0 = np.clip(S / np.maximum(N, 1e-12), 1e-6, 1.0 - 1e-6)
    m = 0.5 * (np.log(p0) - np.log1p(-p0)) + 0.5 * nu
    v = np.full_like(m, rho2)
    xi = np.sqrt(m * m + v)

    for _ in range(max_iter):
        lam = _safe_lambda_jj(xi)
        A = inv_rho2 + 2.0 * N * lam
        v_new = 1.0 / np.maximum(A, 1e-300)
        b = nu * inv_rho2 + (S - 0.5 * N)
        m_new = b * v_new
        xi_new = np.sqrt(m_new * m_new + v_new)
        if np.max(np.abs(xi_new - xi)) < tol:
            m, v, xi = m_new, v_new, xi_new
            break
        m, v, xi = m_new, v_new, xi_new

    lam = _safe_lambda_jj(xi)
    A = inv_rho2 + 2.0 * N * lam
    b = nu * inv_rho2 + (S - 0.5 * N)

    psi_xi = np.logaddexp(xi / 2.0, -xi / 2.0)

    const = N * lam * (xi * xi) - N * psi_xi - 0.5 * (nu * nu) * inv_rho2
    logZ = (
        -0.5 * math.log(2.0 * math.pi * rho2)
        + const
        + 0.5 * math.log(2.0 * math.pi)
        - 0.5 * np.log(np.maximum(A, 1e-300))
        + 0.5 * (b * b) / np.maximum(A, 1e-300)
    )
    return logZ, m, v
