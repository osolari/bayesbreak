"""Checks for the default geometric segment-count prior and uniform alternative."""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian
from bayesbreak.priors import DEFAULT_PRIOR_K_LABEL, resolve_prior_k


def test_default_prior_label_is_geometric_half():
    assert DEFAULT_PRIOR_K_LABEL == "geometric_0.5"


def test_resolve_prior_k_named_and_callable_paths():
    default = resolve_prior_k(None)
    uniform = resolve_prior_k("uniform")
    geo = resolve_prior_k(lambda k: 0.3 ** (k - 1))

    assert default(1) == 1.0 and default(2) == 0.5 and default(3) == 0.25
    assert uniform(1) == uniform(4) == 1.0
    assert geo(2) == pytest.approx(0.3)

    with pytest.raises(ValueError, match="Unknown prior_k"):
        resolve_prior_k("not-a-name")


def test_default_prior_reduces_null_false_positive_rate():
    rng = np.random.default_rng(20260814)
    n = 120
    fp = 0
    for seed in range(30):
        y = np.random.default_rng(seed).normal(0.0, 1.0, n)
        est = BayesBreakGaussian(k_max=8).fit(np.arange(n).reshape(-1, 1), y)
        if int(est.k_map_) > 1:
            fp += 1
    # Null-calibration guard under the declared default model.
    assert fp / 30 < 0.20


def test_uniform_alias_allows_additional_null_segmentation():
    # Regression guard for the difference between uniform and geometric priors.
    rng = np.random.default_rng(261501)
    y = rng.normal(0.0, 1.0, 120)
    est = BayesBreakGaussian(k_max=8, prior_k="uniform").fit(
        np.arange(120).reshape(-1, 1), y
    )
    assert int(est.k_map_) > 1


def test_default_prior_recovers_true_boundaries_on_three_segment_signal():
    """On a canonical truth-k=3 signal the shipped default recovers k=3
    across a high fraction of seeds."""

    n = 120
    correct = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = np.concatenate(
            [rng.normal(0.0, 1.0, 40), rng.normal(2.0, 1.0, 40), rng.normal(-1.0, 1.0, 40)]
        )
        est = BayesBreakGaussian(k_max=8).fit(np.arange(n).reshape(-1, 1), y)
        if int(est.k_map_) == 3:
            correct += 1
    assert correct / 20 >= 0.70
