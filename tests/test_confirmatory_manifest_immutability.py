"""
Regression test: Stage B rendering must not modify confirmatory experiment artifacts.

Runs ``scripts/figures/render.py --all`` (Stage B, cache-only) and verifies
that every manifest.json under ``results/confirmatory/`` retains the exact
SHA-256 hash it had before rendering.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY_ROOT = ROOT / "results" / "confirmatory"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _collect_manifest_hashes() -> dict[str, str]:
    """Return {relative_path: sha256} for every manifest.json in results/confirmatory/."""
    return {
        str(p.relative_to(ROOT)): _sha256(p)
        for p in sorted(CONFIRMATORY_ROOT.rglob("manifest.json"))
    }


@pytest.mark.slow
def test_stage_b_does_not_modify_confirmatory_manifests(tmp_path) -> None:
    """render.py --all must leave all confirmatory manifest.json files byte-for-byte unchanged."""
    hashes_before = _collect_manifest_hashes()
    assert hashes_before, "No confirmatory manifests found — check CONFIRMATORY_ROOT."

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "figures" / "render.py"), "--all"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"render.py --all failed:\n{result.stdout}\n{result.stderr}"
    )

    hashes_after = _collect_manifest_hashes()

    mutated = [
        path
        for path, digest in hashes_before.items()
        if hashes_after.get(path) != digest
    ]
    added = [path for path in hashes_after if path not in hashes_before]

    failures: list[str] = []
    if mutated:
        failures.append("Manifests mutated by render.py --all:\n  " + "\n  ".join(mutated))
    if added:
        failures.append("New manifests created by render.py --all:\n  " + "\n  ".join(added))

    assert not failures, "\n".join(failures)
