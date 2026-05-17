"""Smoke tests for the fitcache timing extension.

Verifies that :func:`bayesbreak.experiments._fitcache.fit_or_load` records
a wall-clock runtime and a ``ran_at`` timestamp in the cache payload,
backing the §5b reproducibility-expectations note. Old caches that lack
these keys remain loadable.
"""

from __future__ import annotations

import pickle
import time

import pytest

from bayesbreak.experiments._fitcache import fit_or_load


@pytest.fixture
def isolated_fitcache(tmp_path, monkeypatch):
    """Redirect the fitcache to tmp_path via ``BAYESBREAK_FITCACHE``."""
    monkeypatch.setenv("BAYESBREAK_FITCACHE", str(tmp_path))
    return tmp_path


def test_fit_or_load_records_runtime_and_timestamp(isolated_fitcache):
    cache_file = isolated_fitcache / "smoke.fit.pkl"

    def _slow_fit() -> dict[str, int]:
        time.sleep(0.02)  # small but non-zero
        return {"k": 3}

    fit = fit_or_load("smoke.fit.pkl", key="abc", fit_fn=_slow_fit)
    assert fit == {"k": 3}
    with cache_file.open("rb") as fh:
        payload = pickle.load(fh)
    assert payload["key"] == "abc"
    assert payload["fit"] == {"k": 3}
    assert "runtime_s" in payload
    assert isinstance(payload["runtime_s"], float)
    assert payload["runtime_s"] >= 0.0
    assert "ran_at" in payload
    assert isinstance(payload["ran_at"], float)
    assert payload["ran_at"] > 0.0


def test_fit_or_load_returns_cache_on_key_match(isolated_fitcache):
    calls = {"n": 0}

    def _fit():
        calls["n"] += 1
        return {"x": 1}

    fit_or_load("smoke2.fit.pkl", key="xyz", fit_fn=_fit)
    assert calls["n"] == 1
    fit_or_load("smoke2.fit.pkl", key="xyz", fit_fn=_fit)
    # Cache hit: fit_fn must not be called again.
    assert calls["n"] == 1


def test_fit_or_load_recomputes_on_key_change(isolated_fitcache):
    calls = {"n": 0}

    def _fit():
        calls["n"] += 1
        return {"x": calls["n"]}

    fit_or_load("smoke3.fit.pkl", key="v1", fit_fn=_fit)
    fit_or_load("smoke3.fit.pkl", key="v2", fit_fn=_fit)
    assert calls["n"] == 2


def test_fit_or_load_reads_legacy_payload(isolated_fitcache):
    """Caches written without runtime_s / ran_at must still load."""
    cache_file = isolated_fitcache / "legacy.fit.pkl"
    legacy_payload = {"key": "legacy", "fit": {"y": 7}}
    with cache_file.open("wb") as fh:
        pickle.dump(legacy_payload, fh)

    calls = {"n": 0}

    def _should_not_run():
        calls["n"] += 1
        return {"y": 99}

    fit = fit_or_load("legacy.fit.pkl", key="legacy", fit_fn=_should_not_run)
    assert fit == {"y": 7}
    assert calls["n"] == 0  # cache hit, no refit
