from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
TUTORIAL_DIR = ROOT / "docs" / "tutorials"


def test_tutorial_notebooks_are_valid_stripped_json() -> None:
    paths = sorted(TUTORIAL_DIR.glob("*.ipynb"))
    assert len(paths) == 11
    for path in paths:
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


def test_mkdocs_lists_every_tutorial_notebook() -> None:
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    for path in TUTORIAL_DIR.glob("*.ipynb"):
        assert f"tutorials/{path.name}" in mkdocs


def test_docs_are_the_only_user_facing_source_tree() -> None:
    assert (ROOT / "docs" / "tutorials").is_dir()
    assert (ROOT / "docs" / "manuscript").is_dir()
    for obsolete in ("tutorials", "report", "examples"):
        assert not (ROOT / obsolete).exists()


def test_environment_setup_script_has_canonical_name() -> None:
    assert (ROOT / "setup_env.sh").is_file()
    assert not (ROOT / "create_env.sh").exists()
