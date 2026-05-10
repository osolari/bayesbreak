"""Negative-Binomial family closed-form checks (§``sec:nb-block``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayesbreak import BayesBreakNegBin


def _block_evidence_brute(y, r, alpha, beta):
    """Beta–NegBin block log-evidence by direct closed form."""
    n = y.size
    C = float(np.sum(y))
    N = float(np.sum(np.full(n, r)))
    H = float(np.sum([math.lgamma(yi + r) - math.lgamma(yi + 1) - math.lgamma(r) for yi in y]))
    a_post = alpha + N
    b_post = beta + C
    log_B_post = math.lgamma(a_post) + math.lgamma(b_post) - math.lgamma(a_post + b_post)
    log_B_prior = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return H + log_B_post - log_B_prior


def test_block_evidence_matches_brute():
    """Single-block evidence matches the closed form."""

    rng = np.random.default_rng(0)
    n = 25
    r = 4.0
    y = rng.negative_binomial(int(r), 0.5, size=n).astype(float)
    est = BayesBreakNegBin(k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=r).fit(
        np.arange(n).reshape(-1, 1), y
    )
    expected = _block_evidence_brute(y, r, 2.0, 3.0)
    # log_evidence_ = log L_{1,n} − log C_1 + log p(k=1) + ... but with
    # uniform p(k) over k_max=1 the only term is L_{1,n} (C_1 = 1, p(k)=1).
    assert est.log_block_evidence_[0, n] == pytest.approx(expected, abs=1e-9)


def test_observation_mean_target():
    """``A^(1)/A^(0)`` equals ``r_* · b_post / (a_post − 1)``, not the Beta mean of ``p``."""

    rng = np.random.default_rng(1)
    n = 30
    r = 3.0
    y = rng.negative_binomial(int(r), 0.4, size=n).astype(float)
    est = BayesBreakNegBin(k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=r).fit(
        np.arange(n).reshape(-1, 1), y
    )
    a_post = 2.0 + n * r
    b_post = 3.0 + float(np.sum(y))
    expected_obs_mean = r * b_post / (a_post - 1.0)
    # Beta-mean alternative:
    beta_mean = a_post / (a_post + b_post)
    measured = float(est.block_first_moment_[0, n] / np.exp(est.log_block_evidence_[0, n]))
    assert measured == pytest.approx(expected_obs_mean, rel=1e-9)
    assert measured != pytest.approx(beta_mean, rel=1e-2)


def test_r_predict_overrides_segment_mean():
    """Setting ``r_predict`` overrides the per-segment training-mean dispersion."""

    rng = np.random.default_rng(2)
    n = 20
    y = rng.negative_binomial(5, 0.5, size=n).astype(float)
    e_default = BayesBreakNegBin(k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=5.0).fit(
        np.arange(n).reshape(-1, 1), y
    )
    e_overridden = BayesBreakNegBin(
        k_max=1, estimate_hyper=False, alpha=2.0, beta=3.0, r=5.0, r_predict=10.0
    ).fit(np.arange(n).reshape(-1, 1), y)
    a_post = 2.0 + n * 5.0
    b_post = 3.0 + float(np.sum(y))
    base_mean = 5.0 * b_post / (a_post - 1.0)
    overridden_mean = 10.0 * b_post / (a_post - 1.0)
    assert e_default.map_segment_means_[0] == pytest.approx(base_mean, rel=1e-9)
    assert e_overridden.map_segment_means_[0] == pytest.approx(overridden_mean, rel=1e-9)


def test_predictive_logpdf_matches_negbin_compound():
    """Posterior-predictive equals the closed-form Beta-NegBin compound density."""

    from scipy.special import betaln

    rng = np.random.default_rng(3)
    n = 15
    r = 4.0
    alpha, beta = 2.0, 3.0
    y_train = rng.negative_binomial(int(r), 0.5, size=n).astype(float)
    est = BayesBreakNegBin(k_max=1, estimate_hyper=False, alpha=alpha, beta=beta, r=r).fit(
        np.arange(n).reshape(-1, 1), y_train
    )

    a_post = alpha + n * r
    b_post = beta + float(np.sum(y_train))
    y_new = np.array([0.0, 3.0, 7.0])
    log_pred = est.posterior_predictive_logpdf_block(
        a=0, b=n, y_new=y_new, w_new=np.ones_like(y_new)
    )
    expected = (
        np.array([math.lgamma(yi + r) - math.lgamma(yi + 1.0) - math.lgamma(r) for yi in y_new])
        + np.array([betaln(a_post + r, b_post + yi) for yi in y_new])
        - betaln(a_post, b_post)
    )
    assert np.allclose(log_pred, expected, atol=1e-8)
