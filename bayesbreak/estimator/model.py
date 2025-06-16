# ------------------------------------------------------------------------------------
# ProcessSegmentation: Bayesian Piecewise Constant Regression
# ------------------------------------------------------------------------------------
# Implements the algorithm described in:
# Marcus Hutter (2006), "Bayesian Regression of Piecewise Constant Functions"
# arXiv:math/0606315v1, IDSIA-14-05
# https://arxiv.org/abs/math/0606315v1
# ------------------------------------------------------------------------------------
# This class follows the scikit-learn API and design, exposing `fit` and `predict`.
# ------------------------------------------------------------------------------------


from typing import Union
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, RegressorMixin


class ProcessSegmentation(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        k_max=50,
        nu=None,
        rho_sq=None,
        sigma_sq=None,
        type_est_rho=1,
        regression_mode=None,
        num_jobs: int = 1,
        seed: Union[int, None] = None,
    ):
        # See paper Section 4–7 for definitions of nu, rho_sq, sigma_sq.
        self.k_max = k_max

        self.nu = nu
        self.rho_sq = rho_sq
        self.sigma_sq = sigma_sq
        self.type_est_rho = type_est_rho
        self.regression_mode = regression_mode

        self.num_jobs = num_jobs
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def _estimate_global_parameters(self, y):
        """
        Step 1: Estimate hyperparameters nu, rho^2, sigma^2 (See Section 7 in paper).
        """
        n = len(y)
        nu = np.mean(y) if self.nu is None else self.nu
        sigma_sq = (
            (np.sum((y[1:] - y[:-1]) ** 2) / (2 * (n - 1)))
            if self.sigma_sq is None
            else self.sigma_sq
        )
        if self.rho_sq is None:
            if self.type_est_rho == 1:
                rho_sq = abs(np.sum((y[:-1] - nu) * (y[1:] - nu))) / (n - 1)
            elif self.type_est_rho == 0:
                rho_sq = np.var(y, ddof=1)
            else:
                raise ValueError("type_est_rho must be 0 or 1")
        else:
            rho_sq = self.rho_sq
        return nu, rho_sq, sigma_sq

    def _compute_logA0(self, y, nu, rho_sq, sigma_sq):
        """
        Step 2: Compute log(A^0_ij), the log evidence for y[i:j] assuming it's a single segment.
        (See Eq. 25–27 in paper.)
        """
        n = len(y)
        logA0 = np.full((n + 1, n + 1), -np.inf)
        for i in range(n):
            y_cumsum = 0.0
            y2_cumsum = 0.0
            for j in range(i + 1, n + 1):
                dist = j - i
                y_cumsum += y[j - 1] - nu
                y2_cumsum += (y[j - 1] - nu) ** 2
                denom = dist + sigma_sq / rho_sq
                log_norm = -0.5 * dist * np.log(2 * np.pi * sigma_sq) - 0.5 * np.log(
                    1 + dist * rho_sq / sigma_sq
                )
                quad_term = 0.5 / sigma_sq * (y_cumsum**2 / denom - y2_cumsum)
                logA0[i, j] = log_norm + quad_term
        return logA0

    def _compute_recursions(self, logA0, n, k_max):
        """
        Step 3: Compute forward (logL) and backward (logR) recursions for dynamic programming.
        See Equations (17–21) in the paper.
        """
        logL = np.full((k_max + 1, n + 1), -np.inf)
        logR = np.full((k_max + 1, n + 1), -np.inf)
        logL[0, 0] = 0
        logR[0, n] = 0
        for k in range(1, k_max + 1):
            for j in range(n + 1):
                if j >= k:
                    logL[k, j] = logsumexp(
                        [logL[k - 1, h] + logA0[h, j] for h in range(k - 1, j)]
                    )
                if j <= n - k:
                    logR[k, j] = logsumexp(
                        [
                            logA0[j, h] + logR[k - 1, h]
                            for h in range(j + 1, n + 1 - (k - 1))
                        ]
                    )
        return logL, logR

    def _compute_posterior(self, y):
        """
        Step 4: Compute posterior quantities: MAP k, boundary probabilities, segmentation,
        piecewise constant levels, and Bayesian regression curve.
        (See Sections 5, 6, 8 in the paper.)
        """
        n, k_max = self.n_, self.k_max
        logL, logR, logA0 = self.logL_, self.logR_, self.logA0_
        logC = np.array(
            [
                logL[k, n] - np.log(np.math.comb(n - 1, k - 1))
                for k in range(1, k_max + 1)
            ]
        )
        logC -= logsumexp(logC)
        C = np.exp(logC)
        k_vals = np.arange(1, k_max + 1)
        k_map = k_vals[np.argmax(C)]
        B = np.zeros(n - 1)
        for p in range(1, k_map):
            for h in range(1, n):
                if logL[p, h] > -np.inf and logR[k_map - p, h] > -np.inf:
                    B[h - 1] += np.exp(logL[p, h] + logR[k_map - p, h] - logL[k_map, n])
        boundaries = [0]
        current = 0
        for p in range(1, k_map):
            scores = [
                logL[p, h] + logR[k_map - p, h] if h > current else -np.inf
                for h in range(n + 1)
            ]
            best_h = np.argmax(scores)
            boundaries.append(best_h)
            current = best_h
        boundaries.append(n)
        mu_hat = np.zeros(n)
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            yseg = y[a:b]
            mu = (self.rho_sq_ * np.sum(yseg) + self.sigma_sq_ * self.nu_) / (
                len(yseg) * self.rho_sq_ + self.sigma_sq_
            )
            mu_hat[a:b] = mu
        mu_hat_reg = np.zeros(n)
        for t in range(n):
            numer = 0.0
            denom = 0.0
            for k, ck in zip(k_vals, C):
                for m in range(1, k + 1):
                    for i in range(t + 1):
                        for j in range(t + 1, n + 1):
                            if logL[m - 1, i] > -np.inf and logR[k - m, j] > -np.inf:
                                log_weight = (
                                    logL[m - 1, i]
                                    + logR[k - m, j]
                                    + logA0[i, j]
                                    - logL[k, n]
                                )
                                weight = np.exp(log_weight) * ck
                                yseg = y[i:j]
                                mu = (
                                    self.rho_sq_ * np.sum(yseg)
                                    + self.sigma_sq_ * self.nu_
                                ) / (len(yseg) * self.rho_sq_ + self.sigma_sq_)
                                numer += weight * mu
                                denom += weight
            mu_hat_reg[t] = numer / denom if denom > 0 else mu_hat[t]
        return {
            "k_map": k_map,
            "boundaries": boundaries,
            "mu_hat_pc": mu_hat,
            "mu_hat_reg": mu_hat_reg,
            "boundary_probs": B,
        }
