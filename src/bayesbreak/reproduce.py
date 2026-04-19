"""Artifact-reproduction orchestration.

``bayesbreak.reproduce.reproduce(target)`` runs the scripts under
``scripts/figures/`` and ``scripts/tables/`` to regenerate the figures and
tables referenced in the report. Output lands under ``results/``.
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys


def _project_root() -> pathlib.Path:
    # ``reproduce.py`` lives at ``src/bayesbreak/reproduce.py``; root is two up.
    return pathlib.Path(__file__).resolve().parents[2]


def _iter_scripts(subdir: str) -> list[pathlib.Path]:
    root = _project_root() / "scripts" / subdir
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.suffix == ".py" and not p.name.startswith("_"))


def _run(path: pathlib.Path) -> None:
    print(f"[reproduce] {path.relative_to(_project_root())}", flush=True)
    # Make project root importable so scripts can `from scripts.figures._style import ...`
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))
    # Each script parses its own argparse on sys.argv; swap it out so the
    # outer CLI's arguments do not leak into the child script.
    saved_argv = sys.argv
    sys.argv = [str(path)]
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = saved_argv


def reproduce(target: str) -> None:
    """Regenerate figures and/or tables.

    Parameters
    ----------
    target : {"figures", "tables", "all"}
    """

    root = _project_root()
    (root / "results").mkdir(parents=True, exist_ok=True)
    # Chdir into project root so scripts that write relative paths land in
    # ``results/`` consistently.
    cwd = os.getcwd()
    try:
        os.chdir(root)
        if target in {"figures", "all"}:
            for p in _iter_scripts("figures"):
                _run(p)
        if target in {"tables", "all"}:
            for p in _iter_scripts("tables"):
                _run(p)
    finally:
        os.chdir(cwd)
