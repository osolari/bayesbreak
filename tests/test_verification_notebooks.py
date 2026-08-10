from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
NOTEBOOK_NAMES = (
    "08_core_dp_verification.ipynb",
    "09_family_prediction_verification.ipynb",
    "10_advanced_model_verification.ipynb",
    "11_result_provenance_explorer.ipynb",
)


def test_verification_notebooks_are_valid_stripped_json() -> None:
    for name in NOTEBOOK_NAMES:
        path = ROOT / "tutorials" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
        for cell in notebook["cells"]:
            expected_language = "python" if cell["cell_type"] == "code" else "markdown"
            assert cell["metadata"]["language"] == expected_language
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_documentation_notebooks_are_byte_identical() -> None:
    for name in NOTEBOOK_NAMES:
        assert (ROOT / "tutorials" / name).read_bytes() == (
            ROOT / "docs" / "tutorials" / name
        ).read_bytes()


def test_mkdocs_lists_every_verification_notebook() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for name in NOTEBOOK_NAMES:
        assert f"tutorials/{name}" in mkdocs
