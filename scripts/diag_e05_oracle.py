"""E05 standalone oracle diagnostic: verifies mixture objective computation.

Runs independent of the E05 runner to confirm log_C_k, template_log_score,
and G=1 mixture objective match direct computation for a small example.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bayesbreak.families.gaussian import BayesBreakGaussian
from bayesbreak import dp as _dp
from bayesbreak.mixture import BayesBreakMixtureClassifier, template_log_score
from sklearn.base import clone


def _geom_prior(k: int) -> float:
    return math.exp(-math.log(4.0) * (k - 1))


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    n, G, S, k_max = 12, 2, 4, 3
    sigma2 = 1.0
    nu, rho2 = 0.0, 4.0
    templates = [[0, 4, 8, 12], [0, 3, 9, 12]]
    true_labels = [0, 0, 1, 1]

    rng = np.random.default_rng([20261021, 5, 999, 0])
    pert_sigma = 0.25
    sequences = []
    for s, g in enumerate(true_labels):
        sign_g = (-1) ** g
        seg_means = [0.0, float(sign_g) * 1.5, 0.0]
        y = np.zeros(n, dtype=float)
        for q, (t0, t1) in enumerate(zip(templates[g][:-1], templates[g][1:])):
            eps = float(rng.normal(0.0, pert_sigma))
            y[t0:t1] = rng.normal(seg_means[q] + eps, 1.0, size=t1 - t0)
        sequences.append(y)

    w = np.ones(n, dtype=float)
    lA0s = []
    for y in sequences:
        base = BayesBreakGaussian(k_max=k_max, estimate_hyper=False,
                                   nu=nu, rho2=rho2, sigma2=sigma2)
        base.likelihood_power_ = w.copy()
        hyper = base._estimate_hyperparameters(y, w)
        lA0, _ = base._compute_block_evidence(y, hyper, w)
        lA0s.append(lA0)

    log_C_k_dp = _dp.compute_log_C_k(None, n, k_max)
    max_Ck_dev = max(
        abs(float(log_C_k_dp[k]) - (math.log(math.comb(n - 1, k - 1)) if k > 1 else 0.0))
        for k in range(1, k_max + 1)
    )
    log.info("max_log_Ck_deviation = %.2e  (expected < 1e-10)", max_Ck_dev)

    log4 = math.log(4.0)
    raw_pk = [math.exp(-log4 * (k_ - 1)) for k_ in range(1, k_max + 1)]
    total_pk = sum(raw_pk)
    log_p_k_arr = np.full(k_max + 1, -np.inf)
    for k_ in range(1, k_max + 1):
        log_p_k_arr[k_] = math.log(raw_pk[k_ - 1] / total_pk)

    max_score_dev = 0.0
    for g in range(G):
        t = templates[g]
        kg = len(t) - 1
        for s in range(S):
            score_direct = float(log_p_k_arr[kg]) - float(log_C_k_dp[kg])
            for a, b in zip(t[:-1], t[1:]):
                score_direct += float(lA0s[s][a, b])
            prod_part = template_log_score(lA0s[s], t, None)
            score_prod = prod_part + float(log_p_k_arr[kg]) - float(log_C_k_dp[kg])
            max_score_dev = max(max_score_dev, abs(score_direct - score_prod))
    log.info("max_template_score_deviation = %.2e  (expected < 1e-10)", max_score_dev)

    base_g1 = BayesBreakGaussian(k_max=k_max, estimate_hyper=False,
                                   nu=nu, rho2=rho2, sigma2=sigma2,
                                   prior_k=_geom_prior)
    clf = BayesBreakMixtureClassifier(
        base_estimator=base_g1, n_groups=1, k_max=k_max,
        max_iter=50, tol=1e-12, random_state=42, n_restarts=1, prior_k=_geom_prior,
    )
    clf.fit(np.stack(sequences))
    tmpl_fit = clf.group_states_[0].template
    k_fit = clf.group_states_[0].k_g
    obj_direct = sum(
        float(log_p_k_arr[k_fit]) - float(log_C_k_dp[k_fit])
        + sum(float(lA0s[s][a, b]) for a, b in zip(tmpl_fit[:-1], tmpl_fit[1:]))
        for s in range(S)
    )
    obj_code = float(clf.final_objective_)
    obj_dev = abs(obj_direct - obj_code)
    log.info("obj_g1_deviation = %.2e  (expected < 1e-8)", obj_dev)

    all_ok = max_Ck_dev < 1e-10 and max_score_dev < 1e-10 and obj_dev < 1e-8
    log.info("Oracle gate: %s", "PASS" if all_ok else "FAIL")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
