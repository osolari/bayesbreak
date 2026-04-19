"""Tests for the supervised (grouped) and latent-group (mixture) classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import (
    BayesBreakGaussian,
    BayesBreakGroupedClassifier,
    BayesBreakMixtureClassifier,
)


def _make_two_group_sequences(n_per_group: int = 5, n: int = 60, seed: int = 0):
    """Groups differ in amplitude: A has small jump, B has large jump."""

    rng = np.random.default_rng(seed)
    group_a = [
        np.r_[rng.normal(-0.5, 0.1, n // 2), rng.normal(0.5, 0.1, n // 2)]
        for _ in range(n_per_group)
    ]
    group_b = [
        np.r_[rng.normal(-5.0, 0.1, n // 2), rng.normal(5.0, 0.1, n // 2)]
        for _ in range(n_per_group)
    ]
    sequences = group_a + group_b
    labels = np.array([0] * n_per_group + [1] * n_per_group)
    return sequences, labels, n


class TestGrouped:
    def test_fit_shapes_and_normalization(self):
        seqs, labels, _ = _make_two_group_sequences()
        est = BayesBreakGroupedClassifier(BayesBreakGaussian(k_max=4)).fit(seqs, labels)
        assert set(est.classes_.tolist()) == {0, 1}
        proba = est.predict_proba(seqs)
        assert proba.shape == (len(seqs), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_predict_proba_favors_true_group_when_groups_differ(self):
        """With visibly different amplitudes, correct group gets highest proba."""

        seqs, labels, _ = _make_two_group_sequences()
        est = BayesBreakGroupedClassifier(BayesBreakGaussian(k_max=4)).fit(seqs, labels)
        proba = est.predict_proba(seqs)
        # The mass on the true class should be >= 0.5 for at least 70% of samples.
        true_class_proba = proba[np.arange(len(labels)), labels]
        assert float(np.mean(true_class_proba >= 0.5)) >= 0.7

    def test_class_prior_uniform(self):
        seqs, labels, _ = _make_two_group_sequences()
        est = BayesBreakGroupedClassifier(BayesBreakGaussian(k_max=4), class_prior="uniform").fit(
            seqs, labels
        )
        proba = est.predict_proba(seqs)
        assert proba.shape == (len(seqs), 2)


class TestMixtureEM:
    def test_fit_runs_and_produces_objective_trace(self):
        """The EM loop runs, populates attributes, and traces an objective."""

        seqs, _, _ = _make_two_group_sequences()
        X = np.stack(seqs)
        est = BayesBreakMixtureClassifier(
            BayesBreakGaussian(k_max=4),
            n_groups=2,
            max_iter=6,
            tol=0.0,
            random_state=0,
        ).fit(X)
        assert est.responsibilities_.shape == (X.shape[0], 2)
        assert np.allclose(est.responsibilities_.sum(axis=1), 1.0, atol=1e-8)
        assert est.pi_.shape == (2,) and est.pi_.sum() == pytest.approx(1.0, abs=1e-8)
        assert len(est.objective_) >= 1
        assert np.all(np.isfinite(est.objective_))

    def test_objective_stays_bounded(self):
        """The objective remains finite across iterations (sanity check)."""

        seqs, _, _ = _make_two_group_sequences()
        X = np.stack(seqs)
        est = BayesBreakMixtureClassifier(
            BayesBreakGaussian(k_max=4),
            n_groups=2,
            max_iter=12,
            tol=0.0,
            random_state=0,
        ).fit(X)
        trace = np.asarray(est.objective_)
        assert np.all(np.isfinite(trace))
        # The range of objective values should be bounded (no runaway).
        assert float(np.ptp(trace)) < 100.0

    def test_predict_proba_row_stochastic(self):
        seqs, _, _ = _make_two_group_sequences()
        X = np.stack(seqs)
        est = BayesBreakMixtureClassifier(
            BayesBreakGaussian(k_max=4),
            n_groups=2,
            max_iter=6,
            random_state=0,
        ).fit(X)
        proba = est.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
