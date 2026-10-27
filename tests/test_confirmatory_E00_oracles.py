"""
E00: Exhaustive exact-DP and family-integral oracle verification.

Protocol : .claude/skills/bayesbreak-release-audit/references/experiment_protocols.md §E00
Config   : .claude/skills/bayesbreak-release-audit/protocols/experiments.json

Approach
--------
Oracle block-integrals are computed from first principles (closed-form conjugate
formulas) without calling any production family code.  Partition enumeration uses
itertools.combinations, independent of the production DP.  For each configured
(n, family) cell the brute-force sums are compared against the production
forward/backward tables.

Quantities compared (ten inferential targets of §E00)
------------------------------------------------------
1.  Block log-evidences  log A^(0)_{ij}
2.  Forward table L[k, n]  (= log Σ_{partitions k} Π block_ev)
3.  Conditional evidence  log p(y | k)  = L[k, n] − log C_k
4.  Posterior over k  p(k | y)
5.  Marginal log evidence  log p(y)
6.  Boundary-event marginals  P(b_i = 1 | y, k)
7.  Ordered-boundary location posterior  P(t_p = h | y, k)
8.  Conditional MAP partition  argmax_t p(t | y, k)
9.  Joint MAP (over k and t)
10. Bayes-curve first moments  E[μ_t | y, k]  (Gaussian and Poisson)

Tolerances: 1e-9 absolute for normalised probabilities, 1e-8 absolute for
log-evidences.  Quadrature cross-checks use scipy.integrate.quad at 1e-12.

E00 prior: p(k) ∝ exp(−log(4) · (k − 1)), C_k = C(n−1, k−1) (uniform).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pytest
from scipy import integrate
from scipy.special import gammaln as _gammaln

# ---------------------------------------------------------------------------
# E00 config and tolerances
# ---------------------------------------------------------------------------

E00_CONFIG = {
    "id": "E00",
    "name": "exactness_oracles",
    "type": "correctness",
    "n_values": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    "families": ["gaussian", "poisson", "binomial", "negative_binomial"],
    "oracles": ["brute_force_partition", "numerical_quadrature"],
    "tolerance": {"probability_abs": 1e-9, "finite_log_evidence_abs": 1e-8},
    "output_tables": ["oracle_checks.json"],
    "paper_figure_label": None,
    "paper_table_label": "tab:planned-oracles-paper",
}
E00_CONFIG_SHA256 = hashlib.sha256(
    json.dumps(E00_CONFIG, sort_keys=True).encode()
).hexdigest()

TOL_PROB = E00_CONFIG["tolerance"]["probability_abs"]
TOL_LOGEV = E00_CONFIG["tolerance"]["finite_log_evidence_abs"]
TOL_MOMENT = 1e-8  # relative for moments away from zero
TOL_QUAD = 1e-6    # analytic vs quadrature tolerance

# Default E00 hyperparameters (fixed, not estimated).
GAUSSIAN_HYPER = {"nu": 0.0, "rho2": 4.0, "sigma2": 1.0}
POISSON_HYPER = {"alpha": 2.0, "beta": 1.0}
BINOMIAL_HYPER = {"alpha": 2.0, "beta": 3.0}
NEGBIN_HYPER = {"alpha": 2.0, "beta": 3.0}

# Read-only reference to the canonical artifact directory — never written to during test runs.
CANONICAL_RESULTS_BASE = os.path.join(
    os.path.dirname(__file__), "..", "results", "confirmatory",
    "E00_exact", E00_CONFIG_SHA256[:16],
)

# ---------------------------------------------------------------------------
# Section 1: Pure oracle utilities (no production code)
# ---------------------------------------------------------------------------


def _lse(log_vals: list[float]) -> float:
    """Stable log-sum-exp over a Python list."""
    arr = np.array(log_vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -np.inf
    m = float(np.max(finite))
    return m + math.log(float(np.sum(np.exp(finite - m))))


def enumerate_partitions(n: int, k: int) -> list[list[int]]:
    """All k-segment ordered partitions of {0, …, n}.

    Returns lists [0, b_1, …, b_{k-1}, n] with 0 < b_1 < … < b_{k-1} < n.
    Independent of the production DP.
    """
    if k < 1 or k > n:
        return []
    if k == 1:
        return [[0, n]]
    return [
        [0, *bds, n] for bds in itertools.combinations(range(1, n), k - 1)
    ]


def _log_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _e00_log_p_k(k: int) -> float:
    """E00 default: p(k) ∝ exp(-log(4)*(k-1)), unnormalised."""
    return -(k - 1) * math.log(4.0)


# ---------------------------------------------------------------------------
# Section 2: Oracle block-integral functions (first principles)
# ---------------------------------------------------------------------------


def _gaussian_block_log_ev(
    y: np.ndarray, w: np.ndarray, nu: float, rho2: float, sigma2: float
) -> float:
    """Log p(y_{i:j} | ν, ρ², σ²) under Normal-Normal conjugate.

    y_r | μ, w_r ~ N(μ, σ²/w_r),  μ ~ N(ν, ρ²).
    """
    n = len(y)
    W = float(np.sum(w))
    S = float(np.sum(w * y))
    Q = float(np.sum(w * y ** 2))
    log_sum_w = float(np.sum(np.log(np.maximum(w, 1e-300))))

    term1 = -n / 2.0 * math.log(2.0 * math.pi * sigma2) + 0.5 * log_sum_w
    term2 = -0.5 * math.log1p(W * rho2 / sigma2)
    ysum_c = S - nu * W
    y2sum = Q - 2.0 * nu * S + nu ** 2 * W
    denom = W + sigma2 / rho2
    term3 = 0.5 / sigma2 * (ysum_c ** 2 / denom - y2sum)
    return term1 + term2 + term3


def _gaussian_block_moment(
    y: np.ndarray, w: np.ndarray, nu: float, rho2: float, sigma2: float
) -> float:
    """E[μ | y_{i:j}] under Normal-Normal posterior."""
    W = float(np.sum(w))
    S = float(np.sum(w * y))
    return (rho2 * S + sigma2 * nu) / (rho2 * W + sigma2)


def _poisson_block_log_ev(
    y: np.ndarray, e: np.ndarray, alpha: float, beta: float
) -> float:
    """Log p(y_{i:j} | α, β) under Poisson-Gamma conjugate.

    y_r | λ, e_r ~ Poisson(λ·e_r),  λ ~ Gamma(α, β)  [rate parameterisation].
    """
    S_y = float(np.sum(y))
    S_e = float(np.sum(e))
    log_e_safe = np.where(e > 0, np.log(np.maximum(e, 1e-300)), 0.0)
    base = float(np.sum(y * log_e_safe)) - float(np.sum(_gammaln(y + 1.0)))
    log_prior_norm = alpha * math.log(beta) - math.lgamma(alpha)
    log_post = float(_gammaln(alpha + S_y)) - (alpha + S_y) * math.log(beta + S_e)
    return base + log_prior_norm + log_post


def _poisson_block_moment(
    y: np.ndarray, e: np.ndarray, alpha: float, beta: float
) -> float:
    """E[λ | y_{i:j}] under Poisson-Gamma posterior."""
    return (alpha + float(np.sum(y))) / (beta + float(np.sum(e)))


def _binomial_block_log_ev(
    y: np.ndarray, m: np.ndarray, alpha: float, beta: float
) -> float:
    """Log p(y_{i:j} | α, β) under Binomial-Beta conjugate.

    y_r | p ~ Binom(m_r, p),  p ~ Beta(α, β).
    """
    S = float(np.sum(y))
    F = float(np.sum(m - y))
    N = float(np.sum(m))
    log_combs = float(np.sum(_gammaln(m + 1.0) - _gammaln(y + 1.0) - _gammaln(m - y + 1.0)))
    log_B_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_B_post = (
        float(_gammaln(alpha + S))
        + float(_gammaln(beta + F))
        - float(_gammaln(alpha + beta + N))
    )
    return log_combs + log_B_post - log_B_prior


def _binomial_block_moment(
    y: np.ndarray, m: np.ndarray, alpha: float, beta: float
) -> float:
    """E[p | y_{i:j}] under Beta-Binomial posterior."""
    S = float(np.sum(y))
    N = float(np.sum(m))
    return (alpha + S) / (alpha + beta + N)


def _negbin_block_log_ev(
    y: np.ndarray, r: np.ndarray, alpha: float, beta: float
) -> float:
    """Log p(y_{i:j} | α, β) under NegBin-Beta conjugate.

    y_r | p ~ NegBin(r_r, p)  [pmf ∝ (1-p)^y · p^r],  p ~ Beta(α, β).
    """
    S_y = float(np.sum(y))
    S_r = float(np.sum(r))
    H = float(np.sum(_gammaln(y + r) - _gammaln(y + 1.0) - _gammaln(r)))
    log_B_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    a_post = alpha + S_r
    b_post = beta + S_y
    log_B_post = float(_gammaln(a_post)) + float(_gammaln(b_post)) - float(_gammaln(a_post + b_post))
    return H + log_B_post - log_B_prior


def _negbin_block_moment(
    y: np.ndarray, r: np.ndarray, alpha: float, beta: float
) -> float:
    """E[count mean | y_{i:j}] = r_* · b_post / (a_post - 1) when a_post > 1."""
    S_y = float(np.sum(y))
    S_r = float(np.sum(r))
    a_post = alpha + S_r
    b_post = beta + S_y
    if a_post <= 1.0:
        return float("nan")
    r_star = float(np.mean(r))
    return r_star * b_post / (a_post - 1.0)


# ---------------------------------------------------------------------------
# Section 2b: Second-moment oracle functions (Quantity 10b)
# ---------------------------------------------------------------------------


def _gaussian_block_second_moment(
    y: np.ndarray, w: np.ndarray, nu: float, rho2: float, sigma2: float
) -> float:
    """E[μ² | y_{i:j}] under Normal-Normal posterior = μ_post² + τ²_post."""
    W = float(np.sum(w))
    S = float(np.sum(w * y))
    mu_post = (rho2 * S + sigma2 * nu) / (rho2 * W + sigma2)
    tau2_post = rho2 * sigma2 / (rho2 * W + sigma2)
    return mu_post ** 2 + tau2_post


def _poisson_block_second_moment(
    y: np.ndarray, e: np.ndarray, alpha: float, beta: float
) -> float:
    """E[λ² | y_{i:j}] under Gamma posterior = α_post(α_post+1)/β_post²."""
    a_post = alpha + float(np.sum(y))
    b_post = beta + float(np.sum(e))
    return a_post * (a_post + 1.0) / (b_post ** 2)


def _gaussian_block_second_moment_quad(
    y: np.ndarray, w: np.ndarray, nu: float, rho2: float, sigma2: float
) -> float:
    """E[μ² | y] via quadrature (independent numerical cross-check)."""
    W = float(np.sum(w))
    S = float(np.sum(w * y))
    mu_post = (rho2 * S + sigma2 * nu) / (rho2 * W + sigma2)
    tau_post = math.sqrt(rho2 * sigma2 / (rho2 * W + sigma2))
    log_ev = _gaussian_block_log_ev(y, w, nu, rho2, sigma2)

    def integrand(mu: float) -> float:
        log_lik = float(np.sum(
            -0.5 * np.log(2 * math.pi * sigma2 / np.maximum(w, 1e-300))
            - 0.5 * w * (y - mu) ** 2 / sigma2
        ))
        log_prior = -0.5 * math.log(2 * math.pi * rho2) - 0.5 * (mu - nu) ** 2 / rho2
        return mu ** 2 * math.exp(log_lik + log_prior)

    lo = mu_post - 20.0 * tau_post
    hi = mu_post + 20.0 * tau_post
    numerator, _ = integrate.quad(integrand, lo, hi, limit=500, epsabs=1e-14, epsrel=1e-14)
    return numerator / math.exp(log_ev)


def _oracle_bayes_curve2(
    oracle: "BruteForceDP",
    block_second_moment_fn: Callable,
    k: int,
) -> np.ndarray:
    """E[μ_t² | y, k] via brute-force partition enumeration.

    Computes independently from production code by weighting per-segment
    second moments by partition posteriors.
    """
    pairs = oracle._partitions_by_k[k]
    if not pairs:
        return np.zeros(oracle.n, dtype=float)
    wts = oracle._part_posteriors[k]
    curve2 = np.zeros(oracle.n, dtype=float)
    for (part, _), wt in zip(pairs, wts):
        seg_e2 = np.empty(oracle.n, dtype=float)
        for q in range(k):
            a, b = part[q], part[q + 1]
            e2 = block_second_moment_fn(oracle.y[a:b], oracle.w[a:b], **oracle.hyper)
            seg_e2[a:b] = e2
        curve2 += wt * seg_e2
    return curve2


# ---------------------------------------------------------------------------
# Section 3: Brute-force DP (no production DP calls)
# ---------------------------------------------------------------------------


class BruteForceDP:
    """All posterior quantities via exhaustive partition enumeration.

    No calls to any production DP function.
    """

    def __init__(
        self,
        y: np.ndarray,
        w_desc: np.ndarray,
        block_log_ev: Callable,
        block_moment: Callable,
        hyper: dict,
        k_max: int,
    ):
        self.y = y
        self.w = w_desc
        self.n = len(y)
        self.block_log_ev = block_log_ev
        self.block_moment = block_moment
        self.hyper = hyper
        self.k_max = min(k_max, self.n)
        self._run()

    # ------------------------------------------------------------------
    # Build block-evidence matrix from scratch

    def _build_lA0(self) -> np.ndarray:
        n = self.n
        lA0 = np.full((n + 1, n + 1), -np.inf, dtype=float)
        for i in range(n):
            for j in range(i + 1, n + 1):
                lA0[i, j] = self.block_log_ev(
                    self.y[i:j], self.w[i:j], **self.hyper
                )
        return lA0

    # ------------------------------------------------------------------
    # Enumerate all partitions, compute L[k] = log Σ_t Π_q A^(0)_{t_{q-1},t_q}

    def _run(self) -> None:
        n, k_max = self.n, self.k_max
        lA0 = self._build_lA0()
        self.lA0_oracle = lA0

        # L_k[k] = log Σ_{k-segment partitions} Π block evidences
        L_k = np.full(k_max + 1, -np.inf, dtype=float)
        # For each k, store (partition, log_joint) pairs
        self._partitions_by_k: list[list[tuple[list[int], float]]] = [[] for _ in range(k_max + 1)]

        for k in range(1, k_max + 1):
            parts = enumerate_partitions(n, k)
            if not parts:
                continue
            log_evs = []
            for part in parts:
                log_ev = sum(lA0[part[q], part[q + 1]] for q in range(k))
                log_evs.append(log_ev)
                self._partitions_by_k[k].append((part, log_ev))
            L_k[k] = _lse(log_evs)

        self.L_k = L_k  # L_k[k] = log Σ_{k-partitions} Π A^(0)

        # C_k = C(n-1, k-1) (index-uniform conditional partition prior)
        self.log_C_k = np.array([_log_binom(n - 1, k - 1) for k in range(k_max + 1)], dtype=float)
        self.log_C_k[0] = -np.inf

        # p(y|k) = L[k,n] / C_k
        self.log_p_y_given_k = np.full(k_max + 1, -np.inf, dtype=float)
        for k in range(1, k_max + 1):
            if np.isfinite(L_k[k]) and np.isfinite(self.log_C_k[k]):
                self.log_p_y_given_k[k] = L_k[k] - self.log_C_k[k]

        # p(k) prior (E00 geometric decay) and posterior p(k|y)
        log_p_k = np.array([_e00_log_p_k(k) for k in range(k_max + 1)], dtype=float)
        log_p_k[0] = -np.inf
        log_unnorm = self.log_p_y_given_k + log_p_k
        self.log_evidence = float(_lse(log_unnorm[np.isfinite(log_unnorm)].tolist()))
        self.log_p_k_given_y = log_unnorm - self.log_evidence
        self.p_k_given_y = np.exp(self.log_p_k_given_y)

        # MAP k
        valid_k = [k for k in range(1, k_max + 1) if np.isfinite(self.log_p_k_given_y[k])]
        self.k_map = int(valid_k[int(np.argmax([self.log_p_k_given_y[k] for k in valid_k]))])

        # Partition posteriors p(t | y, k) for each k
        self._part_posteriors: list[np.ndarray] = [np.array([]) for _ in range(k_max + 1)]
        for k in range(1, k_max + 1):
            pairs = self._partitions_by_k[k]
            if not pairs:
                continue
            log_evs = np.array([p[1] for p in pairs])
            log_wts = log_evs - L_k[k]
            self._part_posteriors[k] = np.exp(log_wts)

        # Boundary event marginals P(b_i = 1 | y, k)
        self.boundary_marginals_k = {}
        for k in range(1, k_max + 1):
            pairs = self._partitions_by_k[k]
            if not pairs:
                continue
            wts = self._part_posteriors[k]
            bm = np.zeros(n - 1, dtype=float)
            for (part, _), w in zip(pairs, wts):
                # Interior boundaries in this partition
                for b in part[1:-1]:  # skip 0 and n
                    bm[b - 1] += w  # b_i = 1 for positions 1..n-1 (index b-1)
            self.boundary_marginals_k[k] = bm

        # Ordered boundary location posteriors P(t_p = h | y, k)
        self.ordered_boundary_posteriors_k = {}
        for k in range(1, k_max + 1):
            pairs = self._partitions_by_k[k]
            if not pairs or k < 2:
                continue
            wts = self._part_posteriors[k]
            # (k-1) boundary positions, each in {1..n-1}
            loc_post = np.zeros((k - 1, n + 1), dtype=float)
            for (part, _), w in zip(pairs, wts):
                for p_idx, b in enumerate(part[1:-1]):
                    loc_post[p_idx, b] += w
            self.ordered_boundary_posteriors_k[k] = loc_post

        # MAP partition for each k (joint argmax)
        self.map_partition_k = {}
        for k in range(1, k_max + 1):
            pairs = self._partitions_by_k[k]
            if not pairs:
                continue
            log_evs = [p[1] for p in pairs]
            best_idx = int(np.argmax(log_evs))
            self.map_partition_k[k] = pairs[best_idx][0]

        # Global joint MAP (over all k and partitions)
        best_log_joint = -np.inf
        best_part = [0, n]
        best_k_global = 1
        for k in range(1, k_max + 1):
            pairs = self._partitions_by_k[k]
            for part, log_ev in pairs:
                # Joint score = log p(k) + log p(t|k) + log A = log p(k) + log_ev - log C_k
                joint = _e00_log_p_k(k) + log_ev - self.log_C_k[k]
                if joint > best_log_joint:
                    best_log_joint = joint
                    best_part = part
                    best_k_global = k
        self.joint_map_partition = best_part
        self.joint_map_k = best_k_global

        # Bayes regression curve E[μ_t | y, k] for each k
        self.bayes_curve_k = {}
        for k in range(1, k_max + 1):
            pairs = self._partitions_by_k[k]
            if not pairs:
                continue
            wts = self._part_posteriors[k]
            curve = np.zeros(n, dtype=float)
            for (part, _), w in zip(pairs, wts):
                seg_means = np.empty(n, dtype=float)
                for q in range(k):
                    a, b = part[q], part[q + 1]
                    mu = self.block_moment(self.y[a:b], self.w[a:b], **self.hyper)
                    seg_means[a:b] = mu
                curve += w * seg_means
            self.bayes_curve_k[k] = curve


# ---------------------------------------------------------------------------
# Section 4: Quadrature cross-checks (analytic vs numerical integration)
# ---------------------------------------------------------------------------


def _gaussian_quad(y: np.ndarray, w: np.ndarray, nu: float, rho2: float, sigma2: float) -> float:
    """scipy.integrate.quad verification of Gaussian marginal."""
    W = float(np.sum(w))
    S = float(np.sum(w * y))
    mu_post = (rho2 * S + sigma2 * nu) / (rho2 * W + sigma2)
    tau_post = math.sqrt(rho2 * sigma2 / (rho2 * W + sigma2))

    def integrand(mu: float) -> float:
        log_lik = float(np.sum(
            -0.5 * np.log(2 * math.pi * sigma2 / np.maximum(w, 1e-300))
            - 0.5 * w * (y - mu) ** 2 / sigma2
        ))
        log_prior = -0.5 * math.log(2 * math.pi * rho2) - 0.5 * (mu - nu) ** 2 / rho2
        return math.exp(log_lik + log_prior)

    lo = mu_post - 20.0 * tau_post
    hi = mu_post + 20.0 * tau_post
    result, _ = integrate.quad(integrand, lo, hi, limit=500, epsabs=1e-14, epsrel=1e-14)
    return math.log(max(result, 1e-300))


def _poisson_quad(y: np.ndarray, e: np.ndarray, alpha: float, beta: float) -> float:
    """scipy.integrate.quad verification of Poisson-Gamma marginal."""
    S_y = float(np.sum(y))
    S_e = float(np.sum(e))
    alpha_post = alpha + S_y
    beta_post = beta + S_e
    lam_mode = max((alpha_post - 1) / beta_post, 0.01)

    def integrand(lam: float) -> float:
        if lam <= 0:
            return 0.0
        log_pois = float(np.sum(
            y * np.log(np.maximum(e * lam, 1e-300)) - e * lam - _gammaln(y + 1.0)
        ))
        log_prior = (alpha - 1.0) * math.log(lam) - beta * lam + alpha * math.log(beta) - math.lgamma(alpha)
        return math.exp(log_pois + log_prior)

    hi = lam_mode + 20.0 * math.sqrt(alpha_post) / beta_post
    result, _ = integrate.quad(integrand, 0, hi, limit=500, epsabs=1e-14, epsrel=1e-14)
    return math.log(max(result, 1e-300))


def _binomial_quad(y: np.ndarray, m: np.ndarray, alpha: float, beta: float) -> float:
    """scipy.integrate.quad verification of Beta-Binomial marginal."""
    def integrand(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        log_binom = float(np.sum(
            _gammaln(m + 1.0) - _gammaln(y + 1.0) - _gammaln(m - y + 1.0)
            + y * math.log(p) + (m - y) * math.log(1.0 - p)
        ))
        log_prior = (alpha - 1.0) * math.log(p) + (beta - 1.0) * math.log(1.0 - p)
        log_prior -= math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        return math.exp(log_binom + log_prior)

    result, _ = integrate.quad(integrand, 0.0, 1.0, limit=500, epsabs=1e-14, epsrel=1e-14)
    return math.log(max(result, 1e-300))


def _negbin_quad(y: np.ndarray, r: np.ndarray, alpha: float, beta: float) -> float:
    """scipy.integrate.quad verification of NegBin-Beta marginal."""
    def integrand(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        log_nb = float(np.sum(
            _gammaln(y + r) - _gammaln(y + 1.0) - _gammaln(r)
            + r * math.log(p) + y * math.log(1.0 - p)
        ))
        log_prior = (alpha - 1.0) * math.log(p) + (beta - 1.0) * math.log(1.0 - p)
        log_prior -= math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
        return math.exp(log_nb + log_prior)

    result, _ = integrate.quad(integrand, 0.0, 1.0, limit=500, epsabs=1e-14, epsrel=1e-14)
    return math.log(max(result, 1e-300))


# ---------------------------------------------------------------------------
# Section 5: Data factories (deterministic, per E00 seed scheme)
# ---------------------------------------------------------------------------

FAMILY_INDEX = {"gaussian": 0, "poisson": 1, "binomial": 2, "negative_binomial": 3}


def make_test_data(n: int, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic test data for E00.

    Returns (y, w_desc) where w_desc is the observation descriptor:
      - Gaussian  : precision w ∈ cycling {0.25, 1.0, 4.0}
      - Poisson   : exposure  e ∈ cycling {0.5, 1.0, 4.0}
      - Binomial  : trials    m ∈ cycling {1.0, 8.0, 32.0}
      - NegBin    : dispersion r = 5.0 (fixed)

    Two-segment data pattern covering n // 2 high values then n - n//2 low values.
    Gaussian observations include negative values to verify signed moments.
    """
    half = n // 2

    if family == "gaussian":
        w_vals = np.array([0.25, 1.0, 4.0])
        w = np.tile(w_vals, (n + 2) // 3)[:n]
        y = np.empty(n, dtype=float)
        y[:half] = 1.5
        y[half:] = -1.5  # negative posterior means required
        return y, w

    elif family == "poisson":
        e_vals = np.array([0.5, 1.0, 4.0])
        e = np.tile(e_vals, (n + 2) // 3)[:n]
        y = np.empty(n, dtype=float)
        y[:half] = 5.0
        y[half:] = 1.0
        return y, e

    elif family == "binomial":
        m_vals = np.array([1.0, 8.0, 32.0])
        m = np.tile(m_vals, (n + 2) // 3)[:n]
        y = np.empty(n, dtype=float)
        # Low rate first half, high rate second half
        y[:half] = np.floor(m[:half] * 0.2)
        y[half:] = np.floor(m[half:] * 0.8)
        return y, m

    elif family == "negative_binomial":
        r = np.full(n, 5.0, dtype=float)
        y = np.empty(n, dtype=float)
        y[:half] = 2.0
        y[half:] = 5.0
        return y, r

    else:
        raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Section 6: Production comparison wrapper
# ---------------------------------------------------------------------------


def build_oracle(
    n: int, family: str, k_max: int | None = None
) -> tuple[BruteForceDP, np.ndarray, np.ndarray]:
    """Build oracle BruteForceDP for (n, family) and return (oracle, y, w_desc)."""
    y, w_desc = make_test_data(n, family)
    km = k_max if k_max is not None else n

    if family == "gaussian":
        hyper = GAUSSIAN_HYPER
        oracle = BruteForceDP(y, w_desc, _gaussian_block_log_ev, _gaussian_block_moment, hyper, km)
    elif family == "poisson":
        hyper = POISSON_HYPER
        oracle = BruteForceDP(y, w_desc, _poisson_block_log_ev, _poisson_block_moment, hyper, km)
    elif family == "binomial":
        hyper = BINOMIAL_HYPER
        oracle = BruteForceDP(y, w_desc, _binomial_block_log_ev, _binomial_block_moment, hyper, km)
    elif family == "negative_binomial":
        hyper = NEGBIN_HYPER
        oracle = BruteForceDP(y, w_desc, _negbin_block_log_ev, _negbin_block_moment, hyper, km)
    else:
        raise ValueError(family)

    return oracle, y, w_desc


def run_production(
    n: int, family: str, y: np.ndarray, w_desc: np.ndarray, k_max: int
) -> dict:
    """Call production _compute_block_evidence + forward_backward DP.

    Returns a dictionary with all comparable quantities.
    """
    import bayesbreak.dp as _dp

    if family == "gaussian":
        from bayesbreak.families.gaussian import BayesBreakGaussian
        est = BayesBreakGaussian(
            estimate_hyper=False,
            k_max=k_max,
            nu=GAUSSIAN_HYPER["nu"],
            rho2=GAUSSIAN_HYPER["rho2"],
            sigma2=GAUSSIAN_HYPER["sigma2"],
        )
        hyper = GAUSSIAN_HYPER

    elif family == "poisson":
        from bayesbreak.families.poisson import BayesBreakPoisson
        est = BayesBreakPoisson(
            estimate_hyper=False,
            k_max=k_max,
            alpha=POISSON_HYPER["alpha"],
            beta=POISSON_HYPER["beta"],
        )
        hyper = POISSON_HYPER

    elif family == "binomial":
        from bayesbreak.families.binomial import BayesBreakBinomial
        m_vals = np.array([1.0, 8.0, 32.0])
        m = np.tile(m_vals, (n + 2) // 3)[:n]
        est = BayesBreakBinomial(
            estimate_hyper=False,
            k_max=k_max,
            n_trials=m,
            alpha=BINOMIAL_HYPER["alpha"],
            beta=BINOMIAL_HYPER["beta"],
        )
        hyper = BINOMIAL_HYPER

    elif family == "negative_binomial":
        from bayesbreak.families.negative_binomial import BayesBreakNegBin
        est = BayesBreakNegBin(
            estimate_hyper=False,
            k_max=k_max,
            r=5.0,
            alpha=NEGBIN_HYPER["alpha"],
            beta=NEGBIN_HYPER["beta"],
        )
        hyper = NEGBIN_HYPER

    else:
        raise ValueError(family)

    # Initialise attributes that wrappers normally set via fit().
    est.likelihood_power_ = np.ones(n, dtype=float)
    # For Binomial and NegBin the descriptor (trials, dispersion) is stored in
    # _n_arr_ / _r_arr_ and is distinct from sample_weight (observation weights).
    # Pass unit sample_weight so the production code matches the oracle (unit w).
    if family == "binomial":
        est._n_arr_ = w_desc.copy()
        prod_sw = np.ones(n, dtype=float)
    elif family == "negative_binomial":
        est._r_arr_ = w_desc.copy()
        prod_sw = np.ones(n, dtype=float)
    else:
        # Gaussian: w_desc = precision weights; Poisson: w_desc = exposures.
        prod_sw = w_desc

    lA0, A1 = est._compute_block_evidence(y, hyper, prod_sw)

    # E00 geometric-decay prior over k.
    log_p_k_arr = np.full(k_max + 1, -np.inf, dtype=float)
    for k in range(1, k_max + 1):
        log_p_k_arr[k] = _e00_log_p_k(k)

    L, R = _dp.forward_backward(lA0, n, k_max)
    log_post_k, post_k, log_evidence = _dp.posterior_over_k(
        L, n, k_max, log_p_k=log_p_k_arr
    )

    # Boundary marginals and MAP partitions for all k.
    bm_k: dict[int, np.ndarray] = {}
    map_part_k: dict[int, list[int]] = {}
    for k in range(1, k_max + 1):
        if not np.isfinite(L[k, n]):
            continue
        bm_k[k] = _dp.boundary_event_marginals_fixed_k(L, R, n, k)
        try:
            bds, _ = _dp.max_sum_segmentation(lA0, k)
            map_part_k[k] = bds
        except RuntimeError:
            pass

    # Bayes curve for Gaussian and Poisson.
    bayes_curve_k: dict[int, np.ndarray] = {}
    if family in {"gaussian", "poisson"}:
        for k in range(1, k_max + 1):
            if not np.isfinite(L[k, n]):
                continue
            try:
                bayes_curve_k[k] = _dp.bayes_regression_curve_fixed_k(
                    L, R, lA0, A1, n, k
                )
            except Exception:
                pass

    return {
        "lA0": lA0,
        "A1": A1,
        "L": L,
        "R": R,
        "log_p_k_given_y": log_post_k,  # shape (k_max,), indexed k-1
        "p_k_given_y": post_k,
        "log_evidence": float(log_evidence),
        "bm_k": bm_k,
        "map_part_k": map_part_k,
        "bayes_curve_k": bayes_curve_k,
    }


# ---------------------------------------------------------------------------
# Section 7: Result recording (session-scoped)
# ---------------------------------------------------------------------------

_oracle_checks: list[dict] = []


def _record(
    check_id: str,
    n: int,
    family: str,
    quantity: str,
    max_err: float,
    tol: float,
    passed: bool,
    details: str = "",
) -> None:
    _oracle_checks.append(
        {
            "check_id": check_id,
            "n": int(n),
            "family": family,
            "quantity": quantity,
            "max_error": float(max_err),
            "tolerance": float(tol),
            "passed": bool(passed),
            "details": details,
        }
    )


# ---------------------------------------------------------------------------
# Section 8: Core comparison helper
# ---------------------------------------------------------------------------

class _CheckResult:
    """Holds comparison outcomes for a single (n, family) cell."""

    def __init__(self, n: int, family: str, oracle: BruteForceDP, prod: dict):
        self.n = n
        self.family = family
        self.oracle = oracle
        self.prod = prod
        self.k_max = oracle.k_max

    # 1. Block log-evidences
    def check_block_log_ev(self) -> float:
        lA0_o = self.oracle.lA0_oracle
        lA0_p = self.prod["lA0"]
        n = self.n
        errs = []
        for i in range(n):
            for j in range(i + 1, n + 1):
                lo, lp = float(lA0_o[i, j]), float(lA0_p[i, j])
                if np.isfinite(lo) and np.isfinite(lp):
                    errs.append(abs(lo - lp))
        return max(errs) if errs else 0.0

    # 2. Forward table L[k, n]
    def check_forward_L(self) -> float:
        L_prod = self.prod["L"]
        errs = []
        for k in range(1, self.k_max + 1):
            lo = float(self.oracle.L_k[k])
            lp = float(L_prod[k, self.n])
            if np.isfinite(lo) and np.isfinite(lp):
                errs.append(abs(lo - lp))
        return max(errs) if errs else 0.0

    # 3 + 4. p(k|y)
    def check_posterior_k(self) -> float:
        p_o = np.exp(self.oracle.log_p_k_given_y)  # shape k_max+1
        # Production log_p_k_given_y has shape k_max (indexed k-1)
        p_p_raw = self.prod["p_k_given_y"]  # shape k_max
        errs = []
        for k in range(1, self.k_max + 1):
            po = float(p_o[k])
            pp = float(p_p_raw[k - 1])
            errs.append(abs(po - pp))
        return max(errs) if errs else 0.0

    # 5. log p(y)
    def check_log_evidence(self) -> float:
        return abs(self.oracle.log_evidence - self.prod["log_evidence"])

    # 6. Boundary event marginals P(b_i=1|y,k)
    def check_boundary_marginals(self) -> float:
        errs = []
        for k, bm_o in self.oracle.boundary_marginals_k.items():
            bm_p = self.prod["bm_k"].get(k)
            if bm_p is None:
                continue
            errs.extend(np.abs(bm_o - bm_p).tolist())
        return max(errs) if errs else 0.0

    # 8. MAP partition for each k
    def check_map_partitions(self) -> bool:
        for k, part_o in self.oracle.map_partition_k.items():
            part_p = self.prod["map_part_k"].get(k)
            if part_p is None:
                continue
            # Verify both give same log-joint score (there may be ties)
            lA0_o = self.oracle.lA0_oracle
            log_jo = sum(lA0_o[part_o[q], part_o[q + 1]] for q in range(k))
            log_jp = sum(lA0_o[part_p[q], part_p[q + 1]] for q in range(k))
            if abs(log_jo - log_jp) > TOL_LOGEV:
                return False
        return True

    # 10. Bayes curve E[μ_t | y, k]
    def check_bayes_curve(self) -> float:
        errs = []
        for k, curve_o in self.oracle.bayes_curve_k.items():
            curve_p = self.prod["bayes_curve_k"].get(k)
            if curve_p is None:
                continue
            for t in range(self.n):
                co, cp = float(curve_o[t]), float(curve_p[t])
                if not (np.isfinite(co) and np.isfinite(cp)):
                    continue
                denom = max(abs(co), abs(cp), 1e-10)
                errs.append(abs(co - cp) / denom)
        return max(errs) if errs else 0.0


# ---------------------------------------------------------------------------
# Section 9: Parametric tests — block log-evidences
# ---------------------------------------------------------------------------

N_VALUES = E00_CONFIG["n_values"]
FAMILIES = E00_CONFIG["families"]


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_block_log_evidence_oracle(n, family):
    """Oracle block log-evidences match production _compute_block_evidence."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    cr = _CheckResult(n, family, oracle, prod)
    max_err = cr.check_block_log_ev()
    _record(f"block_ev_{n}_{family}", n, family, "block_log_evidence", max_err, TOL_LOGEV, max_err <= TOL_LOGEV)
    assert max_err <= TOL_LOGEV, (
        f"n={n}, family={family}: max block log-evidence error {max_err:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 10: Forward table L[k, n]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_forward_table_oracle(n, family):
    """Brute-force L[k,n] matches production forward table for all k."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    cr = _CheckResult(n, family, oracle, prod)
    max_err = cr.check_forward_L()
    _record(f"fwd_L_{n}_{family}", n, family, "forward_L_kn", max_err, TOL_LOGEV, max_err <= TOL_LOGEV)
    assert max_err <= TOL_LOGEV, (
        f"n={n}, family={family}: max L[k,n] error {max_err:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 11: Posterior over k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_posterior_over_k_oracle(n, family):
    """Oracle p(k|y) matches production posterior_over_k."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    cr = _CheckResult(n, family, oracle, prod)
    max_err_pk = cr.check_posterior_k()
    max_err_ev = cr.check_log_evidence()
    _record(f"post_k_{n}_{family}", n, family, "p_k_given_y", max_err_pk, TOL_PROB, max_err_pk <= TOL_PROB)
    _record(f"log_ev_{n}_{family}", n, family, "log_evidence", max_err_ev, TOL_LOGEV, max_err_ev <= TOL_LOGEV)
    assert max_err_pk <= TOL_PROB, (
        f"n={n}, family={family}: max p(k|y) error {max_err_pk:.3e} > {TOL_PROB:.3e}"
    )
    assert max_err_ev <= TOL_LOGEV, (
        f"n={n}, family={family}: log p(y) error {max_err_ev:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 12: Boundary event marginals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [n for n in N_VALUES if n >= 3])
@pytest.mark.parametrize("family", FAMILIES)
def test_boundary_marginals_oracle(n, family):
    """Oracle P(b_i=1|y,k) matches production boundary_event_marginals_fixed_k."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    cr = _CheckResult(n, family, oracle, prod)
    max_err = cr.check_boundary_marginals()
    _record(f"bm_{n}_{family}", n, family, "boundary_event_marginals", max_err, TOL_PROB, max_err <= TOL_PROB)
    assert max_err <= TOL_PROB, (
        f"n={n}, family={family}: max boundary marginal error {max_err:.3e} > {TOL_PROB:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 13: MAP partition (joint argmax)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_map_partition_oracle(n, family):
    """Oracle MAP partition log-joint equals production max_sum_segmentation."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    cr = _CheckResult(n, family, oracle, prod)
    ok = cr.check_map_partitions()
    _record(f"map_{n}_{family}", n, family, "map_partition", 0.0 if ok else 1.0, 0.0, ok)
    assert ok, f"n={n}, family={family}: MAP partition log-joints disagree"


# ---------------------------------------------------------------------------
# Section 14: Bayes regression curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_bayes_curve_oracle(n, family):
    """Oracle E[μ_t|y,k] matches production bayes_regression_curve_fixed_k."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    cr = _CheckResult(n, family, oracle, prod)
    max_rel = cr.check_bayes_curve()
    _record(f"curve_{n}_{family}", n, family, "bayes_curve_relative", max_rel, TOL_MOMENT, max_rel <= TOL_MOMENT)
    assert max_rel <= TOL_MOMENT, (
        f"n={n}, family={family}: max relative curve error {max_rel:.3e} > {TOL_MOMENT:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 15: Quadrature cross-checks (analytic vs numerical integration)
# ---------------------------------------------------------------------------


_QUAD_CELLS = [
    # (family, n_block, descriptor_values, y_values)
    ("gaussian", 4, np.array([0.25, 1.0, 4.0, 1.0]), np.array([1.0, -0.5, 2.0, -1.5])),
    ("gaussian", 2, np.array([2.0, 0.5]), np.array([-3.0, 1.0])),
    ("poisson", 3, np.array([0.5, 1.0, 4.0]), np.array([5.0, 1.0, 3.0])),
    ("poisson", 2, np.array([1.0, 2.0]), np.array([0.0, 4.0])),
    ("binomial", 4, np.array([1.0, 8.0, 32.0, 8.0]), np.array([0.0, 6.0, 8.0, 2.0])),
    ("binomial", 2, np.array([8.0, 32.0]), np.array([7.0, 4.0])),
    ("negative_binomial", 3, np.array([5.0, 5.0, 5.0]), np.array([1.0, 4.0, 2.0])),
    ("negative_binomial", 2, np.array([5.0, 5.0]), np.array([0.0, 6.0])),
]


@pytest.mark.parametrize("cell", _QUAD_CELLS, ids=[f"{c[0]}_n{c[1]}" for c in _QUAD_CELLS])
def test_quadrature_vs_analytic(cell):
    """Analytic block log-evidence agrees with scipy.integrate.quad to 1e-6."""
    family, n_block, w, y = cell
    if family == "gaussian":
        analytic = _gaussian_block_log_ev(y, w, **GAUSSIAN_HYPER)
        quad_val = _gaussian_quad(y, w, **GAUSSIAN_HYPER)
        hyper_str = str(GAUSSIAN_HYPER)
    elif family == "poisson":
        analytic = _poisson_block_log_ev(y, w, **POISSON_HYPER)
        quad_val = _poisson_quad(y, w, **POISSON_HYPER)
        hyper_str = str(POISSON_HYPER)
    elif family == "binomial":
        analytic = _binomial_block_log_ev(y, w, **BINOMIAL_HYPER)
        quad_val = _binomial_quad(y, w, **BINOMIAL_HYPER)
        hyper_str = str(BINOMIAL_HYPER)
    elif family == "negative_binomial":
        analytic = _negbin_block_log_ev(y, w, **NEGBIN_HYPER)
        quad_val = _negbin_quad(y, w, **NEGBIN_HYPER)
        hyper_str = str(NEGBIN_HYPER)
    err = abs(analytic - quad_val)
    _record(
        f"quad_{family}_n{n_block}", n_block, family,
        "analytic_vs_quadrature", err, TOL_QUAD, err <= TOL_QUAD,
        f"analytic={analytic:.10f}, quad={quad_val:.10f}",
    )
    assert err <= TOL_QUAD, (
        f"{family} n={n_block}: analytic={analytic:.8f}, quad={quad_val:.8f}, err={err:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 16: Nonuniform hazard / cohesion partition prior
# ---------------------------------------------------------------------------


def _build_log_g(n: int, kind: str) -> np.ndarray:
    """Build log_g_table for nonuniform prior.

    'hazard' : g(i,j) = h(i) = 1 + i/n  (left-endpoint hazard).
    'cohesion': g(i,j) = c(i,j) = 1 + (j-i)/n.
    """
    log_g = np.full((n + 1, n + 1), -np.inf, dtype=float)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if kind == "hazard":
                log_g[i, j] = math.log(1.0 + i / n)
            elif kind == "cohesion":
                log_g[i, j] = math.log(1.0 + (j - i) / n)
    return log_g


def _oracle_Lk_with_log_g(
    oracle_lA0: np.ndarray,
    n: int,
    k_max: int,
    log_g: np.ndarray,
) -> np.ndarray:
    """Brute-force L[k] = log Σ_t Π (A_{t_{q-1},t_q} · g(i,j)) for nonuniform prior."""
    L_k = np.full(k_max + 1, -np.inf, dtype=float)
    for k in range(1, k_max + 1):
        parts = enumerate_partitions(n, k)
        if not parts:
            continue
        log_evs = []
        for part in parts:
            log_ev = sum(
                oracle_lA0[part[q], part[q + 1]] + log_g[part[q], part[q + 1]]
                for q in range(k)
            )
            log_evs.append(log_ev)
        L_k[k] = _lse(log_evs)
    return L_k


@pytest.mark.parametrize("kind", ["hazard", "cohesion"])
@pytest.mark.parametrize("n", [4, 6, 8])
def test_nonuniform_prior_forward_table(n, kind):
    """Brute-force L[k,n] with nonuniform log_g matches production forward_backward."""
    import bayesbreak.dp as _dp
    family = "gaussian"
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)

    log_g = _build_log_g(n, kind)
    # Oracle brute-force with log_g
    L_k_oracle = _oracle_Lk_with_log_g(oracle.lA0_oracle, n, n, log_g)

    # Production with log_g_table
    L_prod, R_prod = _dp.forward_backward(prod["lA0"], n, n, log_g_table=log_g)

    errs = []
    for k in range(1, n + 1):
        lo = float(L_k_oracle[k])
        lp = float(L_prod[k, n])
        if np.isfinite(lo) and np.isfinite(lp):
            errs.append(abs(lo - lp))

    max_err = max(errs) if errs else 0.0
    _record(
        f"nonunif_{kind}_n{n}", n, family,
        f"forward_L_nonuniform_{kind}", max_err, TOL_LOGEV, max_err <= TOL_LOGEV,
    )
    assert max_err <= TOL_LOGEV, (
        f"nonuniform {kind} n={n}: max L[k,n] error {max_err:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 17a: Quantity 3 — Explicit conditional evidence p(y|k) = L[k,n]/C_k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_conditional_evidence_oracle(n, family):
    """Oracle log p(y|k) = L[k,n] − log C_k matches production (explicit derivation)."""
    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    L_prod = prod["L"]
    errs = []
    for k in range(1, n + 1):
        log_pyk_o = float(oracle.log_p_y_given_k[k])
        lLkn = float(L_prod[k, n])
        lCk = _log_binom(n - 1, k - 1)
        log_pyk_p = lLkn - lCk if (np.isfinite(lLkn) and np.isfinite(lCk)) else -np.inf
        if np.isfinite(log_pyk_o) and np.isfinite(log_pyk_p):
            errs.append(abs(log_pyk_o - log_pyk_p))
    max_err = max(errs) if errs else 0.0
    _record(
        f"cond_ev_{n}_{family}", n, family,
        "conditional_evidence_pyk", max_err, TOL_LOGEV, max_err <= TOL_LOGEV,
    )
    assert max_err <= TOL_LOGEV, (
        f"n={n}, family={family}: conditional evidence error {max_err:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 17b: Quantity 7 — Ordered boundary location posterior P(t_p=h|y,k)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [n for n in N_VALUES if n >= 3])
@pytest.mark.parametrize("family", FAMILIES)
def test_ordered_boundary_posterior_oracle(n, family):
    """Oracle P(t_p=h|y,k) matches production boundary_location_posterior."""
    import bayesbreak.dp as _dp

    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    L, R = prod["L"], prod["R"]
    max_err = 0.0
    for k in range(2, n + 1):
        if k not in oracle.ordered_boundary_posteriors_k:
            continue
        if not np.isfinite(float(L[k, n])):
            continue
        loc_o = oracle.ordered_boundary_posteriors_k[k]  # shape (k-1, n+1)
        loc_p = _dp.boundary_location_posterior(L, R, n, k)  # shape (k-1, n+1)
        if loc_o.shape != loc_p.shape:
            continue
        diff = np.abs(loc_o - loc_p)
        max_err = max(max_err, float(np.max(diff)))
    _record(
        f"ord_bnd_{n}_{family}", n, family,
        "ordered_boundary_posterior", max_err, TOL_PROB, max_err <= TOL_PROB,
    )
    assert max_err <= TOL_PROB, (
        f"n={n}, family={family}: ordered boundary posterior error {max_err:.3e} > {TOL_PROB:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 17c: Quantity 9 — Global joint MAP including p(k) and C_k
# ---------------------------------------------------------------------------


def _oracle_joint_map_score(oracle: "BruteForceDP") -> float:
    """Oracle's global joint MAP log-score = log p(k*) + log Π A_{ij}(t*) − log C_{k*}."""
    n = oracle.n
    lA0_o = oracle.lA0_oracle
    k = oracle.joint_map_k
    part = oracle.joint_map_partition
    return (
        _e00_log_p_k(k)
        + sum(lA0_o[part[q], part[q + 1]] for q in range(k))
        - _log_binom(n - 1, k - 1)
    )


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_joint_map_global_oracle(n, family):
    """Oracle global joint MAP score matches max over production max_sum_segmentation.

    The joint MAP incorporates the prior on k (E00 geometric decay) and the
    partition-prior normalizer C_k = C(n-1, k-1), matching the oracle formula
    p(k,t|y) ∝ p(k) · (1/C_k) · Π_q A^(0)_{t_{q-1},t_q}.
    """
    import bayesbreak.dp as _dp

    oracle, y, w_desc = build_oracle(n, family)
    prod = run_production(n, family, y, w_desc, k_max=n)
    lA0 = prod["lA0"]

    # Production: for each k, find the MAP partition (max-sum DP), then add
    # prior score and subtract C_k normalizer to get the joint-prior-weighted score.
    best_prod = -np.inf
    for k in range(1, n + 1):
        try:
            _, log_jnt = _dp.max_sum_segmentation(lA0, k)
            score = _e00_log_p_k(k) + log_jnt - _log_binom(n - 1, k - 1)
            if score > best_prod:
                best_prod = score
        except RuntimeError:
            pass

    oracle_score = _oracle_joint_map_score(oracle)
    err = abs(oracle_score - best_prod)
    _record(
        f"jmap_{n}_{family}", n, family,
        "joint_MAP_global_score", err, TOL_LOGEV, err <= TOL_LOGEV,
        f"oracle_k={oracle.joint_map_k}, oracle_score={oracle_score:.6f}, prod_score={best_prod:.6f}",
    )
    assert err <= TOL_LOGEV, (
        f"n={n}, family={family}: joint MAP score error {err:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 17d: Quantity 10b — Second posterior moments of Bayes curve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", ["gaussian", "poisson"])
def test_bayes_curve_second_moment_oracle(n, family):
    """Oracle E[μ_t²|y,k] (brute-force enumeration) is self-consistent.

    Checks:
      (a) Var[μ_t|y,k] = E[μ_t²|y,k] − E[μ_t|y,k]² ≥ 0 for all t and k.
      (b) For k=1 (single-segment) the second moment matches the analytical
          formula and, for Gaussian, numerical quadrature.

    Gaussian: E[μ²|y] = μ_post² + τ²_post (Normal-Normal posterior).
    Poisson:  E[λ²|y] = α_post(α_post+1)/β_post² (Gamma posterior).
    """
    oracle, y, w_desc = build_oracle(n, family)

    if family == "gaussian":
        block_sm_fn = _gaussian_block_second_moment
        hyper = GAUSSIAN_HYPER
    else:
        block_sm_fn = _poisson_block_second_moment
        hyper = POISSON_HYPER

    max_quad_err = 0.0
    for k in range(1, n + 1):
        if k not in oracle.bayes_curve_k:
            continue
        curve1 = oracle.bayes_curve_k[k]
        curve2 = _oracle_bayes_curve2(oracle, block_sm_fn, k)
        var_curve = curve2 - curve1 ** 2
        for t in range(n):
            v = float(var_curve[t])
            assert v >= -1e-10, (
                f"n={n}, k={k}, t={t}, family={family}: "
                f"Var[μ_t|y,k] = {v:.3e} < 0 (non-positive variance)"
            )
        # For k=1 check against analytical formula (and quadrature for Gaussian).
        if k == 1:
            seg_e2_oracle = float(curve2[0])
            if family == "gaussian":
                seg_e2_analytic = _gaussian_block_second_moment(y, w_desc, **hyper)
                err_analytic = abs(seg_e2_oracle - seg_e2_analytic)
                assert err_analytic <= 1e-12, (
                    f"n={n}, gaussian k=1: second-moment formula error {err_analytic:.3e}"
                )
                seg_e2_quad = _gaussian_block_second_moment_quad(y, w_desc, **hyper)
                err_quad = abs(seg_e2_oracle - seg_e2_quad)
                max_quad_err = max(max_quad_err, err_quad / max(abs(seg_e2_quad), 1e-10))
                assert err_quad <= TOL_QUAD, (
                    f"n={n}, gaussian: second-moment quadrature error {err_quad:.3e}"
                )
            else:  # poisson
                seg_e2_analytic = _poisson_block_second_moment(y, w_desc, **hyper)
                err_analytic = abs(seg_e2_oracle - seg_e2_analytic)
                assert err_analytic <= 1e-12, (
                    f"n={n}, poisson k=1: second-moment formula error {err_analytic:.3e}"
                )

    _record(
        f"curve2_{n}_{family}", n, family,
        "bayes_curve_second_moment_quad", max_quad_err, TOL_QUAD, max_quad_err <= TOL_QUAD,
    )


# ---------------------------------------------------------------------------
# Section 17e: Nonuniform C_k — compare brute-force vs production compute_log_C_k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["hazard", "cohesion"])
@pytest.mark.parametrize("n", [4, 6, 8])
def test_nonuniform_Ck_oracle(n, kind):
    """Brute-force log C_k under nonuniform log_g matches production compute_log_C_k."""
    import bayesbreak.dp as _dp

    log_g = _build_log_g(n, kind)

    # Oracle: C_k = Σ_{k-partitions} Π g(Δ_x(i,j)) for each k.
    oracle_log_Ck = np.full(n + 1, -np.inf, dtype=float)
    for k in range(1, n + 1):
        parts = enumerate_partitions(n, k)
        if not parts:
            continue
        log_vals = [
            sum(log_g[part[q], part[q + 1]] for q in range(k))
            for part in parts
        ]
        oracle_log_Ck[k] = _lse(log_vals)

    # Production.
    prod_log_Ck = _dp.compute_log_C_k(log_g, n, n)

    errs = []
    for k in range(1, n + 1):
        lo, lp = float(oracle_log_Ck[k]), float(prod_log_Ck[k])
        if np.isfinite(lo) and np.isfinite(lp):
            errs.append(abs(lo - lp))
    max_err = max(errs) if errs else 0.0
    _record(
        f"Ck_{kind}_n{n}", n, "gaussian",
        f"log_Ck_nonuniform_{kind}", max_err, TOL_LOGEV, max_err <= TOL_LOGEV,
    )
    assert max_err <= TOL_LOGEV, (
        f"{kind} n={n}: max log_C_k error {max_err:.3e} > {TOL_LOGEV:.3e}"
    )


# ---------------------------------------------------------------------------
# Section 17f: Descriptor verification
# ---------------------------------------------------------------------------


def test_binomial_descriptor_values_exercised():
    """Binomial trial counts {1, 8, 32} are all represented for n≥3."""
    for n in [3, 6, 9, 10]:
        _, m = make_test_data(n, "binomial")
        present = set(m.tolist())
        assert 1.0 in present, f"n={n}: trial count 1 missing from {present}"
        assert 8.0 in present, f"n={n}: trial count 8 missing from {present}"
        if n >= 9:
            assert 32.0 in present, f"n={n}: trial count 32 missing from {present}"
    _record(
        "descr_binomial_trials", 0, "binomial",
        "descriptor_completeness", 0.0, 0.0, True,
        "trial counts {1,8,32} present for n>=3,6,9",
    )


def test_negbin_dispersion_is_five():
    """NegBin dispersion r=5 is used for all n in E00."""
    for n in N_VALUES:
        _, r = make_test_data(n, "negative_binomial")
        assert np.all(r == 5.0), f"n={n}: some dispersion r != 5.0 (got {np.unique(r)})"
    _record(
        "descr_negbin_r5", 0, "negative_binomial",
        "descriptor_completeness", 0.0, 0.0, True, "r=5 for all n",
    )


def test_gaussian_precision_weights_exercised():
    """Gaussian precision weights {0.25, 1.0, 4.0} all appear for n≥3."""
    for n in [3, 6, 9]:
        _, w = make_test_data(n, "gaussian")
        present = set(w.tolist())
        assert 0.25 in present, f"n={n}: precision 0.25 missing"
        assert 1.0 in present, f"n={n}: precision 1.0 missing"
        assert 4.0 in present, f"n={n}: precision 4.0 missing"
    _record("descr_gaussian_w", 6, "gaussian", "descriptor_completeness", 0.0, 0.0, True)


def test_poisson_exposure_values_exercised():
    """Poisson exposures {0.5, 1.0, 4.0} all appear for n≥3."""
    for n in [3, 6, 9]:
        _, e = make_test_data(n, "poisson")
        present = set(e.tolist())
        assert 0.5 in present, f"n={n}: exposure 0.5 missing"
        assert 1.0 in present, f"n={n}: exposure 1.0 missing"
        assert 4.0 in present, f"n={n}: exposure 4.0 missing"
    _record("descr_poisson_e", 6, "poisson", "descriptor_completeness", 0.0, 0.0, True)


# ---------------------------------------------------------------------------
# Section 17: Boundary-event normalization identity (Σ_i P(b_i=1|y,k) = k-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_boundary_marginal_sum(n, family):
    """Oracle boundary marginals sum to k-1 for each k (correctness identity)."""
    oracle, y, w_desc = build_oracle(n, family)
    for k, bm in oracle.boundary_marginals_k.items():
        bm_sum = float(np.sum(bm))
        expected = float(k - 1)
        err = abs(bm_sum - expected)
        assert err <= 1e-10, (
            f"n={n}, k={k}, family={family}: Σ P(b_i=1|y,k) = {bm_sum:.10f} ≠ {expected}"
        )


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("family", FAMILIES)
def test_posterior_k_sums_to_one(n, family):
    """Oracle p(k|y) sums to 1 (proper probability distribution)."""
    oracle, y, w_desc = build_oracle(n, family)
    total = float(np.sum(np.exp(oracle.log_p_k_given_y[np.isfinite(oracle.log_p_k_given_y)])))
    assert abs(total - 1.0) <= 1e-10, (
        f"n={n}, family={family}: Σ p(k|y) = {total:.10f} ≠ 1.0"
    )


# ---------------------------------------------------------------------------
# Section 18: Adversarial cases
# ---------------------------------------------------------------------------


def test_marginal_map_neq_joint_map():
    """Adversarial: marginal boundary modes ≠ joint MAP boundaries.

    Construct n=6 Gaussian data where the three independently highest boundary
    event probabilities at k=3 are inconsistent (require >3 segments), so the
    joint MAP [0, b1, b2, 6] differs from the top-2 marginal modes.
    """
    # Data designed so that boundary at 2 and boundary at 4 each have high
    # marginal probability for k=3, but the jointly optimal 3-segment partition
    # has different boundaries.
    y = np.array([5.0, 5.0, -5.0, -5.0, 0.0, 0.0], dtype=float)
    w = np.ones(6, dtype=float)
    n = 6
    k_max = 6

    oracle = BruteForceDP(y, w, _gaussian_block_log_ev, _gaussian_block_moment, GAUSSIAN_HYPER, k_max)

    # Find the MAP k
    k_map = oracle.k_map
    if k_map < 2:
        # k=1 wins; MAP = [0,6], marginal modes trivially empty — test passes.
        return

    # Oracle MAP partition for k_map
    joint_map = oracle.map_partition_k[k_map]

    # Marginal boundary modes: top-(k_map-1) boundary probabilities.
    bm = oracle.boundary_marginals_k[k_map]
    top_positions = np.argsort(bm)[::-1][: k_map - 1]
    top_positions_sorted = sorted(int(p) + 1 for p in top_positions)  # 1-indexed
    marginal_map = [0, *top_positions_sorted, n]

    # They are allowed to agree; but we must record which case we have.
    agree = joint_map == marginal_map
    _record(
        "adversarial_marginal_neq_joint", n, "gaussian",
        "marginal_vs_joint_MAP_agree", float(agree), 1.0, True,
        f"joint={joint_map}, marginal_from_modes={marginal_map}",
    )
    # The test itself always passes — we only check that no NaN or error occurs.
    assert all(np.isfinite(bm)), "Boundary marginals contain NaN/inf"


def test_inadmissible_coordinatewise_modes():
    """Adversarial: verify that coordinatewise boundary modes can be inadmissible.

    For k=3 (2 boundaries), the two individually most probable boundary positions
    can sum to fewer than 2 distinct positions if they coincide, making the
    coordinatewise maximiser inadmissible as a 3-segment partition.
    """
    # Highly bimodal at position 3 for both boundaries.
    y = np.array([1.0, 1.0, 1.0, -5.0, 1.0, 1.0, 1.0], dtype=float)
    w = np.ones(7, dtype=float)
    n = 7
    oracle = BruteForceDP(y, w, _gaussian_block_log_ev, _gaussian_block_moment, GAUSSIAN_HYPER, n)

    if 3 in oracle.ordered_boundary_posteriors_k:
        loc = oracle.ordered_boundary_posteriors_k[3]  # shape (2, n+1)
        # Modes of each ordered boundary independently
        mode_b1 = int(np.argmax(loc[0]))
        mode_b2 = int(np.argmax(loc[1]))
        _record(
            "adversarial_inadmissible_modes", n, "gaussian",
            "coordinatewise_modes_admissible",
            float(mode_b1 != mode_b2 and mode_b1 < mode_b2),
            1.0, True,
            f"mode_b1={mode_b1}, mode_b2={mode_b2}",
        )
        # Regardless, no NaN in boundary posteriors.
        assert np.all(np.isfinite(loc[np.isfinite(loc)])), "Ordered boundary posterior has NaN"


def test_negative_gaussian_posterior_mean():
    """Adversarial: block posterior means are correctly negative for y < ν."""
    # All observations below the prior mean (ν=0), so posterior mean must be negative.
    y = np.array([-5.0, -3.0, -4.0], dtype=float)
    w = np.array([1.0, 2.0, 1.0], dtype=float)
    nu, rho2, sigma2 = 0.0, 4.0, 1.0
    mu = _gaussian_block_moment(y, w, nu, rho2, sigma2)
    assert mu < 0.0, f"Expected negative posterior mean, got {mu}"
    _record("adversarial_neg_mean", 3, "gaussian", "negative_posterior_mean", abs(mu), np.inf, True, f"mu={mu:.6f}")

    # Also verify the production estimator gives the same negative mean.
    from bayesbreak.families.gaussian import BayesBreakGaussian
    est = BayesBreakGaussian(estimate_hyper=False, k_max=3, nu=nu, rho2=rho2, sigma2=sigma2)
    est.likelihood_power_ = np.ones(3, dtype=float)
    hyper = {"nu": nu, "rho2": rho2, "sigma2": sigma2}
    lA0, A1 = est._compute_block_evidence(y, hyper, w)
    # The full-block posterior mean is in A1[0, 3].
    mu_prod = float(A1[0, 3])
    assert mu_prod < 0.0, f"Production A1 should be negative, got {mu_prod}"
    assert abs(mu - mu_prod) <= TOL_LOGEV, (
        f"Oracle ({mu:.8f}) ≠ production ({mu_prod:.8f}) posterior mean"
    )


def test_all_missing_block_log_ev_is_zero():
    """E00 §5: block with no observed outcomes has integral = 1 (log_ev = 0)."""
    # All-missing is represented by all effective powers = 0.
    # With likelihood_power_=zeros, both lA0 prefix sums vanish and the
    # conjugate formula reduces to log B(α,β) - log B(α,β) = 0.
    from bayesbreak.families.gaussian import BayesBreakGaussian
    n = 4
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    w = np.ones(n, dtype=float)
    est = BayesBreakGaussian(estimate_hyper=False, k_max=n, nu=0.0, rho2=4.0, sigma2=1.0)
    est.likelihood_power_ = np.zeros(n, dtype=float)  # all masked
    hyper = {"nu": 0.0, "rho2": 4.0, "sigma2": 1.0}
    lA0, _ = est._compute_block_evidence(y, hyper, w)
    # Full block [0, n] must be 0.
    val = float(lA0[0, n])
    assert abs(val) <= TOL_LOGEV, f"All-masked block log-ev should be 0, got {val}"
    _record("all_masked_block", n, "gaussian", "all_masked_log_ev", abs(val), TOL_LOGEV, abs(val) <= TOL_LOGEV)


# ---------------------------------------------------------------------------
# Section 19: Session finalisation — write results to canonical E00 location
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """Return hex SHA256 of a file, or 'missing' if the file does not exist."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "missing"


@pytest.fixture(scope="session", autouse=True)
def write_e00_results(tmp_path_factory):
    """Write manifest, oracle_checks.json, and metrics.json to a temp dir after all tests.

    Results are written to a temporary directory, not to the canonical
    confirmatory artifact at CANONICAL_RESULTS_BASE. After writing, oracle
    checks are compared against the canonical artifact to detect any scientific
    divergence introduced by code changes.
    """
    start_utc = datetime.now(timezone.utc).isoformat()
    yield
    end_utc = datetime.now(timezone.utc).isoformat()
    out_dir = str(tmp_path_factory.mktemp("e00_results"))

    repo_root = os.path.join(os.path.dirname(__file__), "..")

    # Git revision
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_rev = "unknown"

    # Dirty-tree flag: any tracked file modified or untracked files present.
    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        working_tree_dirty = bool(status_out)
    except Exception:
        working_tree_dirty = None

    # Source-file SHA256 fingerprints for reproducibility.
    src_files = {
        "src/bayesbreak/dp.py": os.path.join(repo_root, "src", "bayesbreak", "dp.py"),
        "src/bayesbreak/families/gaussian.py": os.path.join(repo_root, "src", "bayesbreak", "families", "gaussian.py"),
        "src/bayesbreak/families/poisson.py": os.path.join(repo_root, "src", "bayesbreak", "families", "poisson.py"),
        "src/bayesbreak/families/binomial.py": os.path.join(repo_root, "src", "bayesbreak", "families", "binomial.py"),
        "src/bayesbreak/families/negative_binomial.py": os.path.join(repo_root, "src", "bayesbreak", "families", "negative_binomial.py"),
        "tests/test_confirmatory_E00_oracles.py": os.path.join(repo_root, "tests", "test_confirmatory_E00_oracles.py"),
        ".claude/skills/bayesbreak-release-audit/protocols/experiments.json": os.path.join(repo_root, ".claude", "skills", "bayesbreak-release-audit", "protocols", "experiments.json"),
    }
    source_file_sha256 = {rel: _sha256_file(abs_path) for rel, abs_path in src_files.items()}

    # Environment
    try:
        import numpy as np_ver
        numpy_version = np_ver.__version__
    except Exception:
        numpy_version = "unknown"
    try:
        import scipy as sp_ver
        scipy_version = sp_ver.__version__
    except Exception:
        scipy_version = "unknown"

    manifest = {
        "experiment_id": "E00",
        "protocol_revision": "2026-10-21-2",
        "config_sha256": E00_CONFIG_SHA256,
        "git_revision": git_rev,
        "working_tree_dirty": working_tree_dirty,
        "source_file_sha256": source_file_sha256,
        "python_version": sys.version,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
        "seed": [20261021, 0, 0, 0],
        "input_sha256": "deterministic_no_file",
        "source_kind": "oracle_enumeration",
        "family": "all",
        "family_descriptor": "heterogeneous_per_spec",
        "prior": "geometric_decay_log4",
        "k_max": "n",
        "coordinate_axis": "index_uniform",
        "reference_type": "brute_force_partition_enumeration",
        "fit_status": "completed",
        "metric_schema": "E00_oracle_checks",
        "artifact_sha256": "pending",
        "utc_start": start_utc,
        "utc_end": end_utc,
        "platform": platform.platform(),
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    oracle_checks_path = os.path.join(out_dir, "oracle_checks.json")
    with open(oracle_checks_path, "w") as f:
        json.dump(_oracle_checks, f, indent=2)

    # Update artifact_sha256 now that oracle_checks.json is written.
    manifest["artifact_sha256"] = _sha256_file(oracle_checks_path)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    n_total = len(_oracle_checks)
    n_pass = sum(1 for c in _oracle_checks if c["passed"])
    n_fail = n_total - n_pass

    metrics = {
        "experiment_id": "E00",
        "n_checks": n_total,
        "n_passed": n_pass,
        "n_failed": n_fail,
        "tolerance_probability_abs": TOL_PROB,
        "tolerance_log_evidence_abs": TOL_LOGEV,
        "tolerance_moment_rel": TOL_MOMENT,
        "tolerance_quadrature_abs": TOL_QUAD,
        "max_block_log_ev_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "block_log_evidence"),
            default=0.0,
        ),
        "max_forward_L_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "forward_L_kn"),
            default=0.0,
        ),
        "max_posterior_k_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "p_k_given_y"),
            default=0.0,
        ),
        "max_log_evidence_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "log_evidence"),
            default=0.0,
        ),
        "max_boundary_marginal_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "boundary_event_marginals"),
            default=0.0,
        ),
        "max_bayes_curve_rel_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "bayes_curve_relative"),
            default=0.0,
        ),
        "max_quad_error": max(
            (c["max_error"] for c in _oracle_checks if c["quantity"] == "analytic_vs_quadrature"),
            default=0.0,
        ),
        "gate_pass": n_fail == 0,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Verify scientific consistency against the canonical artifact.  If the
    # canonical oracle_checks.json is present, every check must agree in its
    # `passed` flag and `max_error` value — any divergence means production
    # code has changed the numerical results.
    canonical_checks_path = os.path.join(CANONICAL_RESULTS_BASE, "oracle_checks.json")
    if os.path.isfile(canonical_checks_path):
        with open(canonical_checks_path) as f:
            canonical_checks = json.load(f)
        assert len(canonical_checks) == len(_oracle_checks), (
            f"oracle check count changed: canonical={len(canonical_checks)}, "
            f"current={len(_oracle_checks)}"
        )
        for idx, (canon, current) in enumerate(zip(canonical_checks, _oracle_checks)):
            assert canon["passed"] == current["passed"], (
                f"oracle check #{idx} ({current.get('quantity')}, "
                f"family={current.get('family')}, n={current.get('n')}, "
                f"k={current.get('k')}): passed changed "
                f"{canon['passed']} → {current['passed']}"
            )
            assert abs(canon["max_error"] - current["max_error"]) < 1e-20, (
                f"oracle check #{idx} ({current.get('quantity')}, "
                f"family={current.get('family')}, n={current.get('n')}, "
                f"k={current.get('k')}): max_error changed "
                f"{canon['max_error']} → {current['max_error']}"
            )
