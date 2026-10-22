"""``BayesBreakLogisticNormal`` estimator (façade over the approximation kernels)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ...base import BayesBreakSegmenter
from ...priors import PartitionPriorConfig
from ._ep import ep_i_sweep, ep_single_block
from ._gh import gh_moments_block
from ._jj import jj_analytic_remainder, jj_block
from ._laplace import laplace_analytic_remainder, laplace_block
from ._util import mackay_logistic_normal_mean

Approx = Literal["laplace", "jj", "pg_vb", "pg-vb", "ep", "gh", "quadrature"]


class BayesBreakLogisticNormal(BayesBreakSegmenter):
    """BayesBreak for Bernoulli data with a Normal prior on the log-odds.

    Parameters
    ----------
    k_max : int, default=50
    estimate_hyper : bool, default=True
    approx : {"laplace", "jj", "pg_vb"/"pg-vb", "ep", "gh", "quadrature"}
        Block-evidence approximation kernel. See ``prop:uniform-bounds`` for
        the theoretical error scaling of each choice.
    nu, rho2 : float, optional
        Fixed hyperparameters; override the EB estimates when supplied.
    gh_points : int, default=25
        Number of Gauss--Hermite points for ``"ep"``/``"quadrature"``.
    ep_tol_kl : float, default=1e-4
        Parallel-EP stopping tolerance on the largest per-site natural
        parameter update.
    ep_damping : float, default=0.7
        Damping coefficient in natural-parameter space.
    ep_warm_start_iter : int, default=3
        Number of EP passes to run when warm-starting from the previous
        block's fit before falling back to a full iteration budget.
    """

    def __init__(
        self,
        k_max: int = 50,
        estimate_hyper: bool = True,
        approx: Approx = "laplace",
        regression_curve: Literal["none", "fixed_k", "mix_k"] = "none",
        length_prior: Callable[[float], float] | None = None,
        boundary_coordinates: ArrayLike | None = None,
        prior_k: Callable[[int], float] | None = None,
        nu: float | None = None,
        rho2: float | None = None,
        gh_points: int = 25,
        max_iter: int = 50,
        ep_tol_kl: float = 1e-4,
        ep_damping: float = 0.7,
        ep_warm_start_iter: int = 3,
        partition_prior: PartitionPriorConfig | None = None,
    ):
        super().__init__(
            k_max=k_max,
            estimate_hyper=estimate_hyper,
            regression_curve=regression_curve,
            length_prior=length_prior,
            boundary_coordinates=boundary_coordinates,
            prior_k=prior_k,
            partition_prior=partition_prior,
        )
        # Accept both "pg_vb" and "pg-vb" (paper notation).
        self.approx = str(approx).lower().replace("-", "_")
        self.nu = nu
        self.rho2 = rho2
        self.gh_points = int(gh_points)
        self.max_iter = int(max_iter)
        self.ep_tol_kl = float(ep_tol_kl)
        self.ep_damping = float(ep_damping)
        self.ep_warm_start_iter = int(ep_warm_start_iter)

    def _estimate_hyperparameters(
        self, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> dict[str, float]:
        if not self.estimate_hyper and self.nu is not None and self.rho2 is not None:
            return {"nu": float(self.nu), "rho2": float(self.rho2)}

        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != y.shape:
                raise ValueError("sample_weight must have same shape as y")

        # Empirical Bayes: match moments on the probability scale using a
        # logit transform. Stabilises the prior for segmentation.
        eps = 1e-3
        p = np.clip(y, eps, 1.0 - eps)
        z = np.log(p) - np.log1p(-p)
        wsum = float(np.sum(w))
        nu = float(np.sum(w * z) / max(wsum, 1e-12))
        var = float(np.sum(w * (z - nu) ** 2) / max(wsum, 1e-12))
        rho2 = max(var, 1e-6)

        if self.nu is not None:
            nu = float(self.nu)
        if self.rho2 is not None:
            rho2 = float(self.rho2)
        return {"nu": nu, "rho2": rho2}

    def _compute_block_evidence(
        self, y: np.ndarray, hyper: dict[str, float], sample_weight: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        n = int(y.size)
        nu = float(hyper["nu"])
        rho2 = max(float(hyper["rho2"]), 1e-12)

        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)
            if w.shape != y.shape:
                raise ValueError("sample_weight must have same shape as y")

        Sw = np.zeros(n + 1, dtype=float)
        Sw[1:] = np.cumsum(w * y)
        Nw = np.zeros(n + 1, dtype=float)
        Nw[1:] = np.cumsum(w)

        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        A1 = np.zeros((n + 1, n + 1), dtype=float)
        ep_converged = np.ones((n + 1, n + 1), dtype=bool) if self.approx == "ep" else None
        opt_residual = np.zeros((n + 1, n + 1), dtype=float)
        # Per-block analytic tail bound. Populated for Laplace (Kass-Tierney
        # third-order remainder) and JJ / PG-VB (rigorous JJ excess under the
        # fitted Gaussian posterior via one-block Gauss-Hermite quadrature).
        analytic_tail = (
            np.zeros((n + 1, n + 1), dtype=float)
            if self.approx in ("laplace", "jj", "pg_vb")
            else None
        )

        for i in range(n):
            j = np.arange(i + 1, n + 1)
            S = Sw[j] - Sw[i]
            N = Nw[j] - Nw[i]

            if self.approx == "laplace":
                logA0, m, v, laplace_grad = laplace_block(
                    S, N, nu=nu, rho2=rho2, max_iter=self.max_iter
                )
                p_mean = mackay_logistic_normal_mean(m, v)
                opt_residual[i, j] = laplace_grad
                assert analytic_tail is not None
                analytic_tail[i, j] = laplace_analytic_remainder(m, v, N)
            elif self.approx in ("jj", "pg_vb"):
                logA0, m, v = jj_block(S, N, nu=nu, rho2=rho2, max_iter=self.max_iter)
                p_mean = mackay_logistic_normal_mean(m, v)
                assert analytic_tail is not None
                analytic_tail[i, j] = jj_analytic_remainder(m, v, S, N)
            elif self.approx == "ep":
                logA0, m, v, p_mean, conv, ep_step = ep_i_sweep(
                    y,
                    w,
                    i=i,
                    j_indices=j,
                    nu=nu,
                    rho2=rho2,
                    max_iter=self.max_iter,
                    tol_kl=self.ep_tol_kl,
                    damping=self.ep_damping,
                    n_gh=self.gh_points,
                    warm_start_iter=self.ep_warm_start_iter,
                )
                assert ep_converged is not None
                ep_converged[i, j] = conv
                opt_residual[i, j] = ep_step
            elif self.approx in ("gh", "quadrature"):
                n_gh = self.gh_points if self.approx == "gh" else max(self.gh_points, 80)
                logA0, m, v, p_mean = gh_moments_block(S, N, nu=nu, rho2=rho2, n_gh=n_gh)
            else:
                raise ValueError(
                    "approx must be one of "
                    "{'laplace','jj','pg_vb','ep','gh','quadrature'}, "
                    f"got {self.approx!r}."
                )

            lA0[i, j] = logA0
            A1[i, j] = p_mean

        diag = np.arange(n + 1)
        lA0[diag, diag] = -np.inf
        if self.approx == "ep":
            assert ep_converged is not None
            np.fill_diagonal(ep_converged, True)
            self.ep_converged_ = ep_converged
            self.ep_all_converged_ = bool(ep_converged[np.triu_indices(n + 1, k=1)].all())
        # Per-block optimisation residual (Newton gradient for laplace,
        # site-update magnitude for ep). Consumed by
        # ``nonconjugate.certify_by_quadrature_doubling``.
        self.optimization_residual_ = opt_residual
        # Per-block analytic tail bound (only populated for Laplace);
        # consumed by ``nonconjugate.certify_by_quadrature_doubling`` as
        # the ``tail_bound`` field of :class:`SegmentErrorRecord`.
        if analytic_tail is not None:
            self.analytic_tail_bound_ = analytic_tail
        return lA0, A1

    def posterior_predictive_logpdf_block(
        self,
        *,
        a: int,
        b: int,
        y_new: np.ndarray,
        w_new: np.ndarray,
    ) -> np.ndarray:
        r"""Bernoulli predictive with ``p = E[\sigma(\theta) | y_{(a,b]}]``.

        The block-level posterior over ``\theta`` is approximated as
        Gaussian by the selected ``approx`` routine; the posterior mean of
        the sigmoid is the same quantity that populates the family's
        ``block_first_moment_[a, b]`` up to the shared block evidence, and
        we recover it from those tables so the predictive is
        family-consistent regardless of approximation.
        """

        y_new = np.asarray(y_new, dtype=float)
        if np.any(~np.isfinite(y_new)) or np.any((y_new != 0.0) & (y_new != 1.0)):
            raise ValueError("LogisticNormal prediction requires y_new in {0, 1}")
        p_mean = float(self.block_first_moment_[a, b])
        if not (0.0 < p_mean < 1.0):
            raise RuntimeError(
                f"Block ({a}, {b}] is not admissible for posterior prediction"
            )
        p_mean = min(max(p_mean, 1e-12), 1.0 - 1e-12)
        return y_new * math.log(p_mean) + (1.0 - y_new) * math.log(1.0 - p_mean)

    def _segment_posterior_mean(
        self,
        a: int,
        b: int,
        y: np.ndarray,
        hyper: dict[str, float],
        sample_weight: np.ndarray | None = None,
    ) -> float:
        y = np.asarray(y, dtype=float)
        nu = float(hyper["nu"])
        rho2 = max(float(hyper["rho2"]), 1e-12)
        if sample_weight is None:
            w = np.ones_like(y)
        else:
            w = np.asarray(sample_weight, dtype=float)

        S = float(np.sum(w[a:b] * y[a:b]))
        N = float(np.sum(w[a:b]))
        if self.approx == "laplace":
            _, m, v, _ = laplace_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, max_iter=self.max_iter
            )
            return float(mackay_logistic_normal_mean(m, v)[0])
        if self.approx in ("jj", "pg_vb"):
            _, m, v = jj_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, max_iter=self.max_iter
            )
            return float(mackay_logistic_normal_mean(m, v)[0])
        if self.approx == "ep":
            _, _, _, p_mean, _ = ep_single_block(
                y[a:b],
                w[a:b],
                nu=nu,
                rho2=rho2,
                n_gh=self.gh_points,
                max_iter=self.max_iter,
                tol_kl=self.ep_tol_kl,
                damping=self.ep_damping,
            )
            return float(p_mean)
        if self.approx in ("gh", "quadrature"):
            n_gh = self.gh_points if self.approx == "gh" else max(self.gh_points, 80)
            _, m, v, p_mean = gh_moments_block(
                np.array([S]), np.array([N]), nu=nu, rho2=rho2, n_gh=n_gh
            )
            return float(p_mean[0])
        raise ValueError(f"Unknown approx={self.approx!r}")
