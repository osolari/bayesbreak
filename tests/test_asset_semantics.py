from __future__ import annotations

import json
from pathlib import Path

from scripts.check_asset_semantics import build_manifest, validate_manifest

ROOT = Path(__file__).parents[1]


def test_all_archived_assets_have_hash_and_semantic_metadata() -> None:
    manifest = build_manifest()
    assert manifest["asset_count"] == len(manifest["assets"]) == 53
    for asset in manifest["assets"]:
        assert asset["status"] == "archived-read-only"
        assert len(asset["sha256"]) == 64
        assert asset["caption_anchor"]
        assert asset["marker_role"]


def test_archived_asset_hashes_exclusions_and_marker_semantics_pass() -> None:
    result = validate_manifest()
    assert result == {
        "passed": True,
        "asset_count": 53,
        "hash_mismatches": [],
        "missing_assets": [],
        "missing_semantics": [],
        "exclusion_checks": {"RES-BB-CMP-002": True, "RES-BB-RD-007Q": True},
        "marker_semantics": True,
        "visual_qa_status": "pass",
    }


def test_written_manifest_matches_generator() -> None:
    written = json.loads(
        (ROOT / "provenance" / "archived-asset-manifest.json").read_text(encoding="utf-8")
    )
    assert written == build_manifest()
