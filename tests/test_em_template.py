"""Latent-template EM (§``latent-em`` / Algorithm ``multi-em``).

Verifies the report's three convergence criteria:

(i) templates stable;
(ii) finite-template mixture objective ``ℓ_⋆`` change below tolerance;
(iii) deterministic tie-breaking would re-emit the same templates.
"""

from __future__ import annotations

import numpy as np

from bayesbreak import BayesBreakGaussian, BayesBreakMixtureClassifier
from bayesbreak.dp import max_sum_segmentation


def _two_group_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 60
    S_per = 8
    seqs = []
    truth = []
    for _ in range(S_per):
        means = np.r_[np.full(20, 0.0), np.full(20, 2.0), np.full(20, -1.0)]
        seqs.append(means + 0.3 * rng.standard_normal(n))
        truth.append(0)
    for _ in range(S_per):
        means = np.r_[np.full(30, 0.5), np.full(30, -1.5)]
        seqs.append(means + 0.3 * rng.standard_normal(n))
        truth.append(1)
    return np.stack(seqs), np.array(truth)


def test_objective_monotone():
    """The finite-template mixture objective ``ℓ_⋆`` is non-decreasing."""

    y, _ = _two_group_dataset(0)
    mix = BayesBreakMixtureClassifier(
        BayesBreakGaussian(k_max=8),
        n_groups=2,
        k_max=8,
        n_restarts=2,
        random_state=0,
        max_iter=20,
    ).fit(y)
    obj = mix.objective_
    assert all(b - a > -1e-6 for a, b in zip(obj, obj[1:], strict=False))


def test_recovers_latent_groups():
    """Two clearly separated groups are recovered (modulo label permutation)."""

    y, truth = _two_group_dataset(0)
    mix = BayesBreakMixtureClassifier(
        BayesBreakGaussian(k_max=8),
        n_groups=2,
        k_max=8,
        n_restarts=3,
        random_state=42,
        max_iter=30,
    ).fit(y)
    pred = mix.predict(y)
    acc = max(float(np.mean(pred == truth)), float(np.mean(pred != truth)))
    assert acc >= 0.95


def test_template_certifies_max_sum():
    """Each fitted template re-emerges from a max-sum DP on B^(g) (criterion iii)."""

    y, _ = _two_group_dataset(0)
    mix = BayesBreakMixtureClassifier(
        BayesBreakGaussian(k_max=8),
        n_groups=2,
        k_max=8,
        n_restarts=2,
        random_state=0,
        max_iter=30,
    ).fit(y)

    log_A0_subjects = mix._log_A0_subjects_  # type: ignore[attr-defined]
    r = mix.responsibilities_
    finite_mask = np.ones_like(log_A0_subjects[0], dtype=bool)
    for la in log_A0_subjects:
        finite_mask &= np.isfinite(la)

    for g, gs in enumerate(mix.group_states_):
        n_g = float(r[:, g].sum())
        B = np.zeros_like(log_A0_subjects[0])
        for s in range(len(log_A0_subjects)):
            B[finite_mask] += float(r[s, g]) * log_A0_subjects[s][finite_mask]
        B[~finite_mask] = -np.inf
        if mix.log_g_table_ is not None:
            B[finite_mask] += n_g * mix.log_g_table_[finite_mask]
        # Verify the stored template equals the max-sum optimum at k_g.
        recovered, _ = max_sum_segmentation(B, k=gs.k_g)
        assert recovered == gs.template
