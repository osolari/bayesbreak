from __future__ import annotations

import email
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

import bayesbreak

ROOT = Path(__file__).parents[1]
VERSION = "2.0.0rc3"


def test_runtime_and_project_version_share_one_source() -> None:
    assert bayesbreak.__version__ == VERSION
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = { attr = "bayesbreak._version.__version__" }' in pyproject
    assert "setuptools_scm" not in pyproject


def test_release_workflow_guards_pypi_tag_and_conda_is_validation_only() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    conda = (ROOT / ".github" / "workflows" / "conda-publish.yml").read_text()
    assert 'test "${GITHUB_REF_NAME}" = "v${VERSION}"' in release
    assert "pypa/gh-action-pypi-publish" in release
    assert "anaconda upload" not in conda
    assert "Conda Package Validation" in conda


def test_citation_and_lineage_name_pypi_release_candidate() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    lineage = (ROOT / "docs" / "version_lineage.md").read_text(encoding="utf-8")
    assert citation["version"] == VERSION
    assert citation["url"] == "https://pypi.org/project/bayesbreak/"
    assert VERSION in lineage
    assert "PyPI" in lineage


def test_built_wheel_metadata_uses_canonical_version(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("bayesbreak-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = email.message_from_bytes(archive.read(metadata_name))
    assert metadata["Version"] == VERSION
