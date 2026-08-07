"""Validate Phase 6 bibliography annotation completeness without mutating sources."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


def check_annotations(report_root: Path = REPORT) -> dict[str, object]:
    bibliography = report_root / "shared" / "bibliography" / "references.bib"
    manifest_path = report_root / "shared" / "bibliography" / "annotation_manifest.json"
    annotation_dir = report_root / "shared" / "bibliography" / "annotated_entries"
    keys = re.findall(
        r"@[A-Za-z]+\s*\{\s*([^,\s]+)",
        bibliography.read_text(encoding="utf-8"),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    manifest_keys = [entry["bib_key"] for entry in entries]
    annotation_paths = sorted(annotation_dir.glob("*.tex"))
    annotation_keys = [path.stem for path in annotation_paths]

    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    missing_relationship: list[str] = []
    missing_verification: list[str] = []
    for entry in entries:
        path = report_root / entry["path"]
        if not path.exists():
            missing_files.append(entry["path"])
            continue
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_hash != entry["sha256"]:
            hash_mismatches.append(entry["path"])
        text = path.read_text(encoding="utf-8").lower()
        if not entry.get("explicit_project_relationship") or not (
            "bayesbreak" in text or "this project" in text
        ):
            missing_relationship.append(entry["path"])
        if "verification status:" not in text:
            missing_verification.append(entry["path"])

    duplicate_bibliography = sorted(key for key, count in Counter(keys).items() if count > 1)
    duplicate_manifest = sorted(key for key, count in Counter(manifest_keys).items() if count > 1)
    orphaned_annotations = sorted(set(annotation_keys) - set(keys))
    missing_annotations = sorted(set(keys) - set(annotation_keys))
    passed = bool(
        len(keys) == len(manifest_keys) == len(annotation_keys) == 38
        and set(keys) == set(manifest_keys) == set(annotation_keys)
        and not duplicate_bibliography
        and not duplicate_manifest
        and not missing_files
        and not hash_mismatches
        and not missing_relationship
        and not missing_verification
        and not orphaned_annotations
        and not missing_annotations
    )
    return {
        "passed": passed,
        "counts": {
            "bibliography_keys": len(keys),
            "manifest_entries": len(manifest_keys),
            "annotation_files": len(annotation_keys),
        },
        "duplicate_bibliography_keys": duplicate_bibliography,
        "duplicate_manifest_keys": duplicate_manifest,
        "missing_annotation_files": missing_files,
        "annotation_hash_mismatches": hash_mismatches,
        "annotations_without_project_relationship": missing_relationship,
        "annotations_without_verification_status": missing_verification,
        "orphaned_annotations": orphaned_annotations,
        "missing_annotations": missing_annotations,
    }


def main() -> int:
    result = check_annotations()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
