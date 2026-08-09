from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.check_annotations import check_annotations

ROOT = Path(__file__).parents[1]


def test_annotation_completeness_has_one_to_one_verified_records() -> None:
    result = check_annotations()
    assert result["passed"] is True
    assert result["counts"] == {
        "bibliography_keys": 38,
        "manifest_entries": 38,
        "annotation_files": 38,
    }
    for field in (
        "duplicate_bibliography_keys",
        "duplicate_manifest_keys",
        "missing_annotation_files",
        "annotation_hash_mismatches",
        "annotations_without_project_relationship",
        "annotations_without_verification_status",
        "orphaned_annotations",
        "missing_annotations",
    ):
        assert result[field] == []


def test_annotation_checker_cli_returns_machine_readable_success() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_annotations.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["passed"] is True
