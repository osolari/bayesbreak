#!/usr/bin/env python3
"""Independent finite-case and numerical checks for BayesBreak Phase 6.

These checks do not replace proofs. They independently evaluate representative closed-form
segment marginal likelihoods, dynamic-programming identities, prior calculations, and the
finite latent-group Jensen decomposition used in the manuscript.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import betaln, gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "revision_artifacts" / "phase6"
JSON_OUT = OUT_DIR / "FORMAL_NUMERICAL_VERIFICATION.json"
MD_OUT = OUT_DIR / "FORMAL_NUMERICAL_VERIFICATION.md"
RNG = np.random.default_rng(20260805)


@dataclass
class Check:
    check_id: str
    description: str
    passed: bool
    metrics: dict[str, float | int | str | list[float]]
    tolerance: str


def relerr(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def partitions(n: int, k: int):
    for cuts in itertools.combinations(range(1, n), k - 1):
        yield (0, *cuts, n)


def log_weight(t: tuple[int, ...], loga: np.ndarray) -> float:
    return float(sum(loga[t[q], t[q + 1]] for q in range(len(t) - 1)))


def forward_log(loga: np.ndarray, kmax: int) -> np.ndarray:
    n = loga.shape[0] - 1
    f = np.full((kmax + 1, n + 1), -np.inf)
    f[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for j in range(k, n + 1):
            vals = [f[k - 1, i] + loga[i, j] for i in range(k - 1, j) if np.isfinite(f[k - 1, i])]
            if vals:
                f[k, j] = logsumexp(vals)
    return f


def backward_log(loga: np.ndarray, kmax: int) -> np.ndarray:
    n = loga.shape[0] - 1
    b = np.full((kmax + 1, n + 1), -np.inf)
    b[0, n] = 0.0
    for r in range(1, kmax + 1):
        for i in range(n - r, -1, -1):
            vals = [loga[i, j] + b[r - 1, j] for j in range(i + 1, n + 1) if np.isfinite(b[r - 1, j])]
            if vals:
                b[r, i] = logsumexp(vals)
    return b


def maxsum(loga: np.ndarray, k: int) -> tuple[float, tuple[int, ...]]:
    n = loga.shape[0] - 1
    v = np.full((k + 1, n + 1), -np.inf)
    bp = np.full((k + 1, n + 1), -1, dtype=int)
    v[0, 0] = 0.0
    for r in range(1, k + 1):
        for j in range(r, n + 1):
            candidates = [(v[r - 1, i] + loga[i, j], i) for i in range(r - 1, j) if np.isfinite(v[r - 1, i])]
            if candidates:
                best_val, best_i = max(candidates, key=lambda z: (z[0], -z[1]))
                v[r, j] = best_val
                bp[r, j] = best_i
    cuts = [n]
    j = n
    for r in range(k, 0, -1):
        j = int(bp[r, j])
        if j < 0:
            raise RuntimeError("unreachable max-sum state")
        cuts.append(j)
    return float(v[k, n]), tuple(reversed(cuts))


def check_gaussian() -> Check:
    y = np.array([-0.8, 0.3, 1.1, 0.4])
    w = np.array([0.7, 1.2, 2.0, 0.9])
    sigma2, nu, rho2 = 1.4, -0.2, 0.8
    n = y.size
    W = float(w.sum())
    S = float(np.dot(w, y))
    Q = float(np.dot(w, (y - nu) ** 2))
    formula_log = (
        -0.5 * n * math.log(2 * math.pi * sigma2)
        + 0.5 * float(np.log(w).sum())
        - 0.5 * math.log1p(rho2 * W / sigma2)
        - Q / (2 * sigma2)
        + (S - nu * W) ** 2 / (2 * (sigma2 * W + sigma2**2 / rho2))
    )

    def integrand(mu: float) -> float:
        ll = -0.5 * n * math.log(2 * math.pi * sigma2) + 0.5 * float(np.log(w).sum())
        ll -= float(np.dot(w, (y - mu) ** 2)) / (2 * sigma2)
        lp = -0.5 * math.log(2 * math.pi * rho2) - (mu - nu) ** 2 / (2 * rho2)
        return math.exp(ll + lp)

    numeric, err = quad(integrand, -np.inf, np.inf, epsabs=1e-13, epsrel=1e-13, limit=300)
    formula = math.exp(formula_log)
    e = relerr(formula, numeric)
    return Check("MATH-BB-001", "Gaussian-known-variance segment marginal likelihood versus direct integration", e < 2e-11, {"formula": formula, "quadrature": numeric, "quad_abs_error": err, "relative_error": e}, "relative error < 2e-11")


def check_poisson() -> Check:
    y = np.array([2, 0, 3, 1])
    exposure = np.array([0.5, 1.2, 0.8, 1.6])
    a0, b0 = 1.7, 2.3
    C, W = int(y.sum()), float(exposure.sum())
    H = float(np.dot(y, np.log(exposure)) - gammaln(y + 1).sum())
    log_formula = H + a0 * math.log(b0) + gammaln(a0 + C) - gammaln(a0) - (a0 + C) * math.log(b0 + W)

    def integrand(lam: float) -> float:
        if lam <= 0:
            return 0.0
        ll = H + C * math.log(lam) - W * lam
        lp = a0 * math.log(b0) - gammaln(a0) + (a0 - 1) * math.log(lam) - b0 * lam
        return math.exp(ll + lp)

    numeric, err = quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-13, limit=300)
    formula = math.exp(log_formula)
    e = relerr(formula, numeric)
    return Check("MATH-BB-002", "Gamma-Poisson segment marginal likelihood versus direct integration", e < 2e-11, {"formula": formula, "quadrature": numeric, "quad_abs_error": err, "relative_error": e}, "relative error < 2e-11")


def check_binomial() -> Check:
    y = np.array([1, 3, 0, 2])
    m = np.array([2, 4, 1, 3])
    a0, b0 = 2.2, 1.4
    C, M = int(y.sum()), int(m.sum())
    H = float((gammaln(m + 1) - gammaln(y + 1) - gammaln(m - y + 1)).sum())
    log_formula = H + betaln(a0 + C, b0 + M - C) - betaln(a0, b0)

    def integrand(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        ll = H + C * math.log(p) + (M - C) * math.log1p(-p)
        lp = (a0 - 1) * math.log(p) + (b0 - 1) * math.log1p(-p) - betaln(a0, b0)
        return math.exp(ll + lp)

    numeric, err = quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-13, points=[0.25, 0.5, 0.75], limit=300)
    formula = math.exp(log_formula)
    e = relerr(formula, numeric)
    return Check("MATH-BB-003", "Beta-Binomial segment marginal likelihood versus direct integration", e < 2e-11, {"formula": formula, "quadrature": numeric, "quad_abs_error": err, "relative_error": e}, "relative error < 2e-11")


def check_negbin() -> Check:
    y = np.array([0, 2, 1, 4])
    r = np.array([2.0, 2.0, 3.0, 1.0])
    a0, b0 = 1.9, 2.4
    C, N = float(y.sum()), float(r.sum())
    H = float((gammaln(y + r) - gammaln(y + 1) - gammaln(r)).sum())
    log_formula = H + betaln(a0 + N, b0 + C) - betaln(a0, b0)

    def integrand(p: float) -> float:
        if p <= 0 or p >= 1:
            return 0.0
        # Manuscript parameterization: P(Y=y|r,p)=choose(y+r-1,y) p^r (1-p)^y.
        ll = H + N * math.log(p) + C * math.log1p(-p)
        lp = (a0 - 1) * math.log(p) + (b0 - 1) * math.log1p(-p) - betaln(a0, b0)
        return math.exp(ll + lp)

    numeric, err = quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-13, points=[0.25, 0.5, 0.75], limit=300)
    formula = math.exp(log_formula)
    e = relerr(formula, numeric)
    return Check("MATH-BB-004", "Beta-negative-binomial segment marginal likelihood versus direct integration", e < 2e-11, {"formula": formula, "quadrature": numeric, "quad_abs_error": err, "relative_error": e}, "relative error < 2e-11")


def check_betaobs() -> Check:
    y = np.array([0.22, 0.63, 0.48])
    phi = np.array([8.0, 12.0, 10.0])
    a0, b0 = 2.5, 3.0

    def log_integrand(mu: float, power: int = 0) -> float:
        if mu <= 0 or mu >= 1:
            return -np.inf
        a = phi * mu
        b = phi * (1 - mu)
        ll = float(((a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - np.vectorize(betaln)(a, b)).sum())
        lp = (a0 - 1) * math.log(mu) + (b0 - 1) * math.log1p(-mu) - betaln(a0, b0)
        return ll + lp + power * math.log(mu)

    # A log offset avoids overflow/underflow and is shared by reference and GL rules.
    grid = np.linspace(1e-5, 1 - 1e-5, 2000)
    offset = max(log_integrand(float(x)) for x in grid)

    def ref_integrand(mu: float, power: int = 0) -> float:
        return math.exp(log_integrand(mu, power) - offset)

    reference_scaled, ref_err = quad(lambda z: ref_integrand(z, 0), 0, 1, epsabs=1e-12, epsrel=1e-12, limit=500, points=[0.01, 0.1, 0.5, 0.9, 0.99])
    reference = math.exp(offset) * reference_scaled
    errors: list[float] = []
    values: list[float] = []
    for q in (8, 16, 32, 64, 128):
        x, w = leggauss(q)
        mu = (x + 1) / 2
        ww = w / 2
        scaled = float(sum(float(ww_i) * math.exp(log_integrand(float(mu_i), 0) - offset) for mu_i, ww_i in zip(mu, ww, strict=True)))
        val = math.exp(offset) * scaled
        values.append(val)
        errors.append(relerr(val, reference))
    passed = errors[-1] < 2e-9
    return Check("MATH-BB-005", "Beta-observation one-dimensional Gauss-Legendre segment integration versus adaptive quadrature", passed, {"reference": reference, "quad_scaled_abs_error": ref_err, "node_counts": [8, 16, 32, 64, 128], "approximations": values, "relative_errors": errors}, "128-node relative error < 2e-9; no monotonicity assertion")


def make_random_loga(n: int) -> np.ndarray:
    loga = np.full((n + 1, n + 1), -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            length = j - i
            loga[i, j] = RNG.normal(loc=-0.08 * length, scale=0.35) + 0.04 * math.sin(i + 2 * j)
    return loga


def exhaustive_by_k(loga: np.ndarray, kmax: int, logpk: np.ndarray | None = None):
    n = loga.shape[0] - 1
    records = []
    for k in range(1, kmax + 1):
        for t in partitions(n, k):
            lw = log_weight(t, loga)
            if logpk is not None:
                lw += float(logpk[k])
            records.append((k, t, lw))
    return records


def check_sumproduct_map_marginals_curves() -> list[Check]:
    n, kmax = 7, 4
    loga = make_random_loga(n)
    f = forward_log(loga, kmax)
    b = backward_log(loga, kmax)
    records = exhaustive_by_k(loga, kmax)
    exhaustive_z = {k: logsumexp([lw for kk, _, lw in records if kk == k]) for k in range(1, kmax + 1)}
    dp_err = max(abs(float(f[k, n]) - float(exhaustive_z[k])) for k in exhaustive_z)

    map_gap = 0.0
    same_partitions = True
    for k in range(1, kmax + 1):
        v, t_dp = maxsum(loga, k)
        v_ex, t_ex = max(((lw, t) for kk, t, lw in records if kk == k), key=lambda z: (z[0], tuple(-x for x in z[1])))
        map_gap = max(map_gap, abs(v - v_ex))
        same_partitions = same_partitions and t_dp == t_ex

    # Ordered-boundary and boundary-event marginals at fixed k.
    k = 4
    z = exhaustive_z[k]
    ordered_ex = np.zeros((k + 1, n + 1))
    event_ex = np.zeros(n + 1)
    for kk, t, lw in records:
        if kk != k:
            continue
        p = math.exp(lw - z)
        for q, h in enumerate(t[1:-1], start=1):
            ordered_ex[q, h] += p
            event_ex[h] += p
    ordered_dp = np.zeros_like(ordered_ex)
    for q in range(1, k):
        for h in range(q, n - (k - q) + 1):
            ordered_dp[q, h] = math.exp(f[q, h] + b[k - q, h] - z)
    event_dp = ordered_dp.sum(axis=0)
    ordered_err = float(np.max(np.abs(ordered_dp - ordered_ex)))
    event_err = float(np.max(np.abs(event_dp - event_ex)))

    # Bayes-curve moments under a segment-count prior, verified by exhaustive posterior averaging.
    logpk = np.full(kmax + 1, -np.inf)
    raw = np.array([0.0, 0.38, 0.31, 0.20, 0.11])
    for kk in range(1, kmax + 1):
        logpk[kk] = math.log(raw[kk])
    all_records = exhaustive_by_k(loga, kmax, logpk)
    logz_all = logsumexp([lw for _, _, lw in all_records])
    block_mean = np.full((n + 1, n + 1), np.nan)
    block_second = np.full((n + 1, n + 1), np.nan)
    for i in range(n):
        for j in range(i + 1, n + 1):
            mean = 0.17 * (i + j) - 0.3
            var = 0.12 + 0.01 * (j - i)
            block_mean[i, j] = mean
            block_second[i, j] = var + mean**2
    mean_ex = np.zeros(n)
    second_ex = np.zeros(n)
    for _, t, lw in all_records:
        p = math.exp(lw - logz_all)
        for i, j in zip(t[:-1], t[1:], strict=True):
            mean_ex[i:j] += p * block_mean[i, j]
            second_ex[i:j] += p * block_second[i, j]
    mean_dp = np.zeros(n)
    second_dp = np.zeros(n)
    for kk in range(1, kmax + 1):
        for q in range(0, kk):
            left_segments = q
            right_segments = kk - q - 1
            for i in range(n):
                if not np.isfinite(f[left_segments, i]):
                    continue
                for j in range(i + 1, n + 1):
                    if not np.isfinite(b[right_segments, j]):
                        continue
                    lp = logpk[kk] + f[left_segments, i] + loga[i, j] + b[right_segments, j] - logz_all
                    p = math.exp(lp)
                    mean_dp[i:j] += p * block_mean[i, j]
                    second_dp[i:j] += p * block_second[i, j]
    mean_err = float(np.max(np.abs(mean_dp - mean_ex)))
    second_err = float(np.max(np.abs(second_dp - second_ex)))

    return [
        Check("MATH-BB-006", "Sum-product fixed-count partition evidence versus exhaustive enumeration", dp_err < 2e-12, {"max_absolute_log_evidence_error": dp_err, "n": n, "kmax": kmax}, "absolute log-evidence error < 2e-12"),
        Check("MATH-BB-007", "Max-sum recursion and backtracking versus exhaustive joint MAP search", map_gap < 2e-12 and same_partitions, {"max_objective_gap": map_gap, "all_partitions_identical": str(same_partitions)}, "objective gap < 2e-12 and identical partition"),
        Check("MATH-BB-008", "Ordered-boundary and boundary-event marginals versus exhaustive posterior enumeration", max(ordered_err, event_err) < 2e-12, {"ordered_boundary_max_abs_error": ordered_err, "boundary_event_max_abs_error": event_err}, "maximum absolute error < 2e-12"),
        Check("MATH-BB-009", "Posterior mean and second-moment curves from block-cover decomposition versus exhaustive averaging", max(mean_err, second_err) < 3e-12, {"mean_curve_max_abs_error": mean_err, "second_moment_curve_max_abs_error": second_err}, "maximum absolute error < 3e-12"),
    ]


def check_shared_pooling() -> Check:
    n, kmax, s_count = 6, 3, 4
    logas = [make_random_loga(n) for _ in range(s_count)]
    pooled = np.sum(np.stack(logas, axis=0), axis=0)
    f = forward_log(pooled, kmax)
    max_err = 0.0
    for k in range(1, kmax + 1):
        ex = logsumexp([sum(log_weight(t, a) for a in logas) for t in partitions(n, k)])
        max_err = max(max_err, abs(float(f[k, n]) - float(ex)))
    return Check("MATH-BB-010", "Shared-boundary pooling by products of sequence-specific segment marginal likelihoods", max_err < 2e-12, {"sequences": s_count, "max_absolute_log_evidence_error": max_err}, "absolute log-evidence error < 2e-12")


def check_stability() -> Check:
    n, kmax, eps = 6, 4, 0.018
    loga = make_random_loga(n)
    delta = np.full_like(loga, np.nan)
    pert = np.full_like(loga, -np.inf)
    for i in range(n):
        for j in range(i + 1, n + 1):
            d = RNG.uniform(-eps, eps)
            delta[i, j] = d
            pert[i, j] = loga[i, j] + d
    f, fh = forward_log(loga, kmax), forward_log(pert, kmax)
    state_bound_ok = True
    max_scaled = 0.0
    for k in range(1, kmax + 1):
        for j in range(k, n + 1):
            if np.isfinite(f[k, j]) and np.isfinite(fh[k, j]):
                diff = abs(float(fh[k, j] - f[k, j]))
                max_scaled = max(max_scaled, diff / (k * eps))
                state_bound_ok &= diff <= k * eps + 1e-12
    # Posterior state-ratio/TV generic lemma.
    true_logw = RNG.normal(size=17)
    d = RNG.uniform(-0.23, 0.23, size=17)
    eta = float(np.max(np.abs(d)))
    p = np.exp(true_logw - logsumexp(true_logw))
    ph = np.exp(true_logw + d - logsumexp(true_logw + d))
    ratios = ph / p
    ratio_ok = float(ratios.min()) >= math.exp(-2 * eta) - 1e-13 and float(ratios.max()) <= math.exp(2 * eta) + 1e-13
    tv = 0.5 * float(np.abs(ph - p).sum())
    tv_bound = min(1.0, math.exp(2 * eta) - 1)
    tv_ok = tv <= tv_bound + 1e-13
    passed = bool(state_bound_ok and ratio_ok and tv_ok)
    return Check("MATH-BB-011", "Conditional propagation of uniform segment log-score error through dynamic-programming evidences and normalized finite distributions", passed, {"epsilon": eps, "maximum_state_error_as_fraction_of_k_epsilon": max_scaled, "eta_generic": eta, "minimum_probability_ratio": float(ratios.min()), "maximum_probability_ratio": float(ratios.max()), "total_variation": tv, "total_variation_bound": tv_bound}, "state error <= k epsilon; probability ratios within exp(+-2 eta); TV <= min(1, exp(2 eta)-1)")


def check_poisson_occupancy() -> Check:
    lam = np.array([0.15, 0.7, 1.1, 0.4, 1.6])
    p = 1 - np.exp(-lam)
    m = 2
    subsets = list(itertools.combinations(range(len(lam)), m))
    direct = []
    odds = []
    wrong = []
    for b in subsets:
        B = set(b)
        direct.append(float(np.prod([p[j] if j in B else 1 - p[j] for j in range(len(lam))])))
        odds.append(float(np.prod([math.exp(lam[j]) - 1 for j in B])))
        wrong.append(float(np.prod([p[j] for j in B])))
    direct = np.array(direct) / np.sum(direct)
    odds = np.array(odds) / np.sum(odds)
    wrong = np.array(wrong) / np.sum(wrong)
    err = float(np.max(np.abs(direct - odds)))
    wrong_gap = float(np.max(np.abs(direct - wrong)))
    passed = err < 2e-15 and wrong_gap > 1e-3
    return Check("MATH-BB-012", "Fixed-count Poisson interval occupancy gives local odds exp(Lambda_j)-1 after conditioning", passed, {"max_abs_error_odds_formula": err, "max_abs_gap_if_occupancy_probabilities_used_instead": wrong_gap, "candidate_intervals": len(lam), "occupied_intervals": m}, "odds formula error < 2e-15 and occupancy-probability alternative demonstrably differs")


def check_latent_jensen() -> Check:
    S, G = 5, 3
    pi = np.array([0.22, 0.31, 0.47])
    score = np.exp(RNG.normal(loc=-1.0, scale=0.8, size=(S, G)))
    weighted = score * pi[None, :]
    r = weighted / weighted.sum(axis=1, keepdims=True)
    F = float(np.log(weighted.sum(axis=1)).sum())
    Q = float(np.sum(r * (np.log(pi)[None, :] + np.log(score) - np.log(r))))
    tight_gap = abs(F - Q)

    # Exact decomposition of one group template contribution.
    g = 1
    n_g = float(r[:, g].sum())
    k = 3
    logpk = math.log(0.24)
    logCk = math.log(35.0)
    block_logA = RNG.normal(size=(S, k))
    loggamma = RNG.normal(loc=-0.05, scale=0.08, size=k)
    direct = 0.0
    for s in range(S):
        direct += float(r[s, g]) * (logpk - logCk + float(block_logA[s].sum()) + float(loggamma.sum()))
    decomposed = n_g * (logpk - logCk) + float(np.sum(r[:, g, None] * block_logA)) + n_g * float(loggamma.sum())
    decomp_gap = abs(direct - decomposed)
    return Check("MATH-BB-013", "Jensen minorizer tightness and responsibility-weighted latent-template score decomposition", tight_gap < 2e-13 and decomp_gap < 2e-13, {"jensen_tightness_gap": tight_gap, "template_decomposition_gap": decomp_gap, "effective_group_weight_n_g": n_g}, "both absolute gaps < 2e-13")


def main() -> int:
    checks: list[Check] = [
        check_gaussian(),
        check_poisson(),
        check_binomial(),
        check_negbin(),
        check_betaobs(),
        *check_sumproduct_map_marginals_curves(),
        check_shared_pooling(),
        check_stability(),
        check_poisson_occupancy(),
        check_latent_jensen(),
    ]
    passed = all(c.passed for c in checks)
    payload = {
        "verification_id": "VERIFY-BB-PHASE6-MATH-001",
        "date": "2026-08-05",
        "purpose": "Independent finite-case and numerical verification of central formulas and recursions; not a substitute for formal proofs.",
        "seed": 20260805,
        "passed": passed,
        "checks": [asdict(c) for c in checks],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Phase 6 formal and numerical verification",
        "",
        "This independent check evaluates representative closed-form segment marginal likelihoods, finite dynamic-programming identities, the fixed-count Poisson occupancy calculation, numerical-error propagation, and the latent-template Jensen decomposition. It is a numerical verification record, not a replacement for the proofs in the manuscript.",
        "",
        f"Overall status: **{'passed' if passed else 'failed'}**.",
        "",
        "| ID | Check | Status | Tolerance |",
        "|---|---|---:|---|",
    ]
    for c in checks:
        lines.append(f"| `{c.check_id}` | {c.description} | {'PASS' if c.passed else 'FAIL'} | {c.tolerance} |")
    lines.extend(["", "## Detailed metrics", ""])
    for c in checks:
        lines.extend([f"### {c.check_id}", "", "```json", json.dumps(c.metrics, indent=2), "```", ""])
    MD_OUT.write_text("\n".join(lines))

    print(json.dumps({"passed": passed, "checks": len(checks), "failed": [c.check_id for c in checks if not c.passed]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
