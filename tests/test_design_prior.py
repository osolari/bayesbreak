"""Length-aware partition-prior tests.

Cover the new ``length_prior`` / ``boundary_coordinates`` / ``prior_k``
plumbing introduced for §``sec:irregular``:

- ``g ≡ 1`` reproduces ``C_k = binom(n - 1, k - 1)`` exactly.
- ``log_C_k`` from ``compute_log_C_k`` matches a brute-force enumeration on
  small ``n``.
- A ``g(ℓ) ∝ ℓ`` length factor shifts boundary marginals toward larger
  segments on an irregular design (matching §``ex:irregular-illustration``).
- The length factor cancels in segment-wise posterior moments
  (Eq. ``segmom-seg``).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from bayesbreak import BayesBreakGaussian
from bayesbreak.dp import compute_log_C_k


def _brute_log_C_k(log_g: np.ndarray, n: int, k: int) -> float:
    """``log Σ_t ∏_q g(Δ(t_{q-1}, t_q))`` by enumerating k-segmentations."""
    if k == 1:
        return float(log_g[0, n])
    log_terms: list[float] = []
    for bps in itertools.combinations(range(1, n), k - 1):
        t = (0, *bps, n)
        s = 0.0
        ok = True
        for a, b in zip(t[:-1], t[1:], strict=False):
            v = float(log_g[a, b])
            if not np.isfinite(v):
                ok = False
                break
            s += v
        if ok:
            log_terms.append(s)
    if not log_terms:
        return -np.inf
    m = max(log_terms)
    return float(m + math.log(sum(math.exp(s - m) for s in log_terms)))


class TestLogCk:
    def test_uniform_matches_combinatorial(self):
        for n in (4, 6, 9):
            log_C = compute_log_C_k(None, n, k_max=min(n, 5))
            for k in range(1, min(n, 5) + 1):
                expected = math.lgamma(n) - math.lgamma(k) - math.lgamma(n - k + 1)
                assert log_C[k] == pytest.approx(expected, abs=1e-12)

    def test_explicit_g_is_one_matches_default(self):
        n = 7
        log_g = np.full((n + 1, n + 1), -np.inf)
        for i in range(n):
            for j in range(i + 1, n + 1):
                log_g[i, j] = 0.0
        log_C_default = compute_log_C_k(None, n, 4)
        log_C_explicit = compute_log_C_k(log_g, n, 4)
        assert np.allclose(log_C_default, log_C_explicit, atol=1e-12)

    def test_length_factor_matches_brute_force(self):
        n = 6
        log_g = np.full((n + 1, n + 1), -np.inf)
        # g(ℓ) ∝ ℓ on a regular grid: log_g[i,j] = log(j - i).
        for i in range(n):
            for j in range(i + 1, n + 1):
                log_g[i, j] = math.log(j - i)
        log_C = compute_log_C_k(log_g, n, k_max=4)
        for k in range(1, 5):
            expected = _brute_log_C_k(log_g, n, k)
            assert log_C[k] == pytest.approx(expected, abs=1e-10)


class TestEstimatorPlumbing:
    def test_irregular_length_prior_shifts_boundary_to_gap(self):
        """A length-aware prior puts more boundary mass at the wide gap.

        Six observations at physical locations (0, 0.2, 0.4, 0.6, 1.6, 1.8):
        a tight cluster of 4 followed by a pair after a wide gap.
        """

        rng = np.random.default_rng(0)
        x = np.array([0.0, 0.2, 0.4, 0.6, 1.6, 1.8])
        y = np.array([0.1, 0.0, -0.05, 0.05, 1.0, 1.05]) + 0.05 * rng.standard_normal(6)

        # Boundary coordinates = midpoints (default in BayesBreakSegmenter).
        idx_uniform = BayesBreakGaussian(k_max=3).fit(x, y)
        length_aware = BayesBreakGaussian(k_max=3, length_prior=lambda d: d).fit(x, y)

        # The candidate boundary at index 4 (the wide gap) has more posterior
        # mass under the length-aware prior than under the index-uniform one.
        bm_uniform = idx_uniform.boundary_marginals_  # length n - 1
        bm_length = length_aware.boundary_marginals_
        assert bm_length[3] > bm_uniform[3]

    def test_g_cancels_in_segment_wise_moments(self):
        """``A^(r)/A^(0)`` is independent of ``g`` (Eq. segmom-seg)."""

        rng = np.random.default_rng(1)
        n = 30
        y = rng.normal(size=n)
        e_uniform = BayesBreakGaussian(k_max=4).fit(np.arange(n).reshape(-1, 1), y)
        e_length = BayesBreakGaussian(k_max=4, length_prior=lambda d: d**1.5).fit(
            np.arange(n).reshape(-1, 1), y
        )

        # Exported MAP segmentation may differ; pick a common block (full range)
        # and verify the per-block posterior mean matches.
        a, b = 0, n
        mu_u = e_uniform._segment_posterior_mean(
            a, b, y, e_uniform.hyper_, e_uniform.sample_weight_
        )
        mu_l = e_length._segment_posterior_mean(a, b, y, e_length.hyper_, e_length.sample_weight_)
        assert mu_u == pytest.approx(mu_l, abs=1e-12)

    def test_prior_k_normalised(self):
        """A custom ``prior_k`` is normalised internally and the posterior sums to 1."""

        rng = np.random.default_rng(2)
        n = 40
        y = rng.normal(size=n)
        # Heavily favour small k.
        e = BayesBreakGaussian(k_max=8, prior_k=lambda k: math.exp(-0.5 * k)).fit(
            np.arange(n).reshape(-1, 1), y
        )
        assert e.k_posterior_.sum() == pytest.approx(1.0, abs=1e-10)
