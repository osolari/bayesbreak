from __future__ import annotations

import subprocess
import sys

from scripts.run_bounded_test import package_versions, run_bounded_test


def test_bounded_runner_reports_passing_node() -> None:
    result = run_bounded_test("tests/test_utils.py::TestLogsumexp::test_matches_manual", 20.0)
    assert result["status"] == "passed"
    assert result["return_code"] == 0
    assert result["environment"]["packages"]["numpy"] is not None


def test_bounded_runner_reports_timeout(monkeypatch) -> None:
    timeout = subprocess.TimeoutExpired([sys.executable], 0.01, output=b"partial")

    def raise_timeout(*args, **kwargs):
        raise timeout

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    result = run_bounded_test("tests/test_utils.py::test", 0.01)
    assert result["status"] == "unresolved-timeout"
    assert result["return_code"] is None
    assert result["stdout"] == "partial"


def test_missing_optional_packages_are_explicit() -> None:
    versions = package_versions()
    assert set(versions) >= {"numpy", "pytest", "mkdocs"}
    assert versions["numpy"] is not None
