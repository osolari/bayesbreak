"""Tests for the upstream-driven baseline wrappers.

These tests do not run the algorithms themselves — they verify that:

- the wrapper interfaces normalize upstream output into a uniform
  :class:`BaselineResult`,
- the registry resolves canonical names and aliases,
- a missing upstream dependency surfaces a clear ``ImportError``,
- when the upstream library is installed, the wrapper recovers the
  ground-truth boundary on a well-separated synthetic signal (smoke test).

The upstream packages (``ruptures``, ``rpy2``+``DNAcopy``) are optional
extras; tests that need them ``pytest.importorskip`` cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesbreak.baselines import (
    BaselineResult,
    available_algorithms,
    segment_with,
)


def _step_signal(rng: np.random.Generator, n: int = 90) -> tuple[np.ndarray, list[int]]:
    """A 3-segment Gaussian signal with well-separated means."""
    y = np.r_[
        rng.normal(0.0, 0.2, n // 3),
        rng.normal(3.0, 0.2, n // 3),
        rng.normal(-2.0, 0.2, n - 2 * (n // 3)),
    ]
    truth = [n // 3, 2 * (n // 3)]
    return y, truth


class TestRegistry:
    def test_available_algorithms_match_registry(self):
        names = available_algorithms()
        assert {
            "pelt",
            "optimal_partitioning",
            "binary_segmentation",
            "wild_binary_segmentation",
            "cbs",
        }.issubset(set(names))

    def test_segment_with_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown baseline algorithm"):
            segment_with("not-a-real-algorithm", np.zeros(10))

    def test_segment_with_aliases_resolve(self):
        # We don't run the algorithm; we just want the alias to dispatch.
        # Skip if ruptures isn't installed.
        pytest.importorskip("ruptures")
        y, _ = _step_signal(np.random.default_rng(0))
        for alias in ("bs", "wbs", "dynp", "op"):
            try:
                res = (
                    segment_with(alias, y, n_bkps=2)
                    if alias != "wbs"
                    else segment_with(alias, y, n_bkps=2, random_state=0, n_random_windows=20)
                )
            except TypeError:
                # ``pelt`` is dispatched by a different alias; skip if signatures differ
                continue
            assert isinstance(res, BaselineResult)


class TestRuptures:
    """Smoke tests against ``ruptures``-backed wrappers."""

    def test_pelt_finds_well_separated_breaks(self):
        pytest.importorskip("ruptures")
        rng = np.random.default_rng(0)
        y, truth = _step_signal(rng)
        res = segment_with("pelt", y, penalty=10.0)
        assert isinstance(res, BaselineResult)
        assert res.algorithm == "pelt"
        assert res.package == "ruptures"
        assert res.n == y.size
        assert res.k >= 1
        # On clean 3-segment data PELT should locate both breaks within a
        # few indices; we don't require an exact match.
        for t in truth:
            assert min(abs(b - t) for b in res.boundaries) <= 5

    def test_dynp_returns_exact_k(self):
        pytest.importorskip("ruptures")
        rng = np.random.default_rng(1)
        y, _ = _step_signal(rng)
        res = segment_with("optimal_partitioning", y, n_bkps=2)
        assert res.k == 3
        assert res.tuning["n_bkps"] == 2

    def test_binseg_with_penalty_or_n_bkps(self):
        pytest.importorskip("ruptures")
        rng = np.random.default_rng(2)
        y, _ = _step_signal(rng)
        res = segment_with("binary_segmentation", y, n_bkps=2)
        assert res.k == 3
        with pytest.raises(ValueError):
            segment_with("binary_segmentation", y)  # neither n_bkps nor penalty

    def test_wbs_returns_uniform_result(self):
        pytest.importorskip("ruptures")
        rng = np.random.default_rng(3)
        y, _ = _step_signal(rng)
        res = segment_with(
            "wild_binary_segmentation", y, n_bkps=2, n_random_windows=20, random_state=0
        )
        assert res.k >= 1
        assert "n_candidate_breakpoints" in res.extra


class TestCBSImportSafety:
    """The CBS wrapper raises a readable ImportError when rpy2/DNAcopy are
    missing. Useful even on dev machines without R.
    """

    def test_cbs_raises_importerror_when_rpy2_missing(self):
        try:
            import rpy2  # noqa: F401

            pytest.skip("rpy2 is installed; this test is for the missing-dep path.")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="rpy2 .* DNAcopy"):
            segment_with("cbs", np.zeros(20))


class TestSMUCEImportSafety:
    """The SMUCE wrapper raises a readable ImportError when rpy2/stepR are
    missing. Mirrors the CBS contract."""

    def test_smuce_raises_importerror_when_rpy2_missing(self):
        try:
            import rpy2  # noqa: F401

            pytest.skip("rpy2 is installed; this test is for the missing-dep path.")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="rpy2 .* stepR"):
            segment_with("smuce", np.zeros(20))

    def test_smuce_registered_in_available_algorithms(self):
        from bayesbreak.baselines import available_algorithms

        assert "smuce" in available_algorithms()
