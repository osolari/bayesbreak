from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_environment_lock_hash_matches_test_manifest() -> None:
    manifest = json.loads((ROOT / "provenance" / "test-manifest.json").read_text())
    lock_path = ROOT / manifest["environment_lock"]
    observed = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert observed == manifest["environment_lock_sha256"]


def test_full_profile_terminal_statuses_sum_to_collected() -> None:
    manifest = json.loads((ROOT / "provenance" / "test-manifest.json").read_text())
    profile = manifest["profiles"]["full_package"]
    assert (
        profile["passed"] + profile["skipped"] + profile["failed"] + profile["unresolved"]
        == profile["collected"]
    )
    assert profile["failed"] == 0


def test_bounded_profiles_do_not_count_missing_historical_node_as_pass() -> None:
    manifest = json.loads((ROOT / "provenance" / "test-manifest.json").read_text())
    profiles = manifest["profiles"]
    assert profiles["bounded_ep_historical_node"]["status"] == "unresolved-node-not-collected"
    assert profiles["bounded_ep_convergence_flags"]["status"] == "passed"
    assert profiles["bounded_ep_reference_agreement"]["status"] == "passed"
