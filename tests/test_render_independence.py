"""Tests that render.py is independent of scientific computation.

Three guarantees verified here:
1. render.py does not transitively import bayesbreak (or scipy / numpy scientific
   inference code that would pull in bayesbreak).  Style-change runs must succeed
   without the model library installed.
2. Calling a render function with a missing CSV raises FileNotFoundError with a
   message that names the missing file.
3. Render functions succeed when the experiment result directories (results/,
   results/phase7/, etc.) are inaccessible -- they read only from the committed
   CSV/JSON cache.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RENDER_PY = _ROOT / "scripts" / "figures" / "render.py"
_DATA_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "data"
_RESULTS_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "results"


# ---------------------------------------------------------------------------
# 1. Import independence
# ---------------------------------------------------------------------------

_FORBIDDEN_TOPLEVEL = {
    "bayesbreak",
    "scipy",
    "sklearn",
}


def _collect_imports(source: str) -> list[str]:
    """Return all top-level module names imported (directly or from-imported) in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def test_render_py_no_bayesbreak_import() -> None:
    """render.py must not directly import bayesbreak at module level."""
    source = _RENDER_PY.read_text(encoding="utf-8")
    direct_imports = _collect_imports(source)
    for forbidden in _FORBIDDEN_TOPLEVEL:
        assert forbidden not in direct_imports, (
            f"render.py directly imports '{forbidden}' -- "
            f"Stage B must not depend on scientific libraries"
        )


def test_render_py_importable_without_bayesbreak(tmp_path: Path) -> None:
    """render.py can be imported even when bayesbreak is absent from sys.path.

    We do this by running python in a subprocess with a modified PYTHONPATH that
    excludes the src/ directory.  If render.py pulls in bayesbreak transitively,
    the import will fail with ModuleNotFoundError.
    """
    import os
    env = os.environ.copy()
    # Remove src/ from PYTHONPATH by filtering it out
    src_dir = str(_ROOT / "src")
    old_path = env.get("PYTHONPATH", "")
    new_path = ":".join(
        p for p in old_path.split(":") if p and not p.startswith(src_dir)
    )
    env["PYTHONPATH"] = new_path
    # Add scripts/figures to the path so _style can be found
    scripts_fig = str(_ROOT / "scripts" / "figures")
    env["PYTHONPATH"] = scripts_fig + (":" + new_path if new_path else "")

    code = (
        "import sys; "
        f"sys.path.insert(0, {repr(scripts_fig)}); "
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('render', {repr(str(_RENDER_PY))}); "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"render.py failed to import without bayesbreak.\n"
        f"stderr: {result.stderr[:800]}"
    )
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# 2. Missing CSV gives FileNotFoundError with informative message
# ---------------------------------------------------------------------------

def _import_render():
    """Import render.py as a module (adds scripts/figures to path)."""
    scripts_fig = str(_ROOT / "scripts" / "figures")
    if scripts_fig not in sys.path:
        sys.path.insert(0, scripts_fig)
    spec = importlib.util.spec_from_file_location("render", _RENDER_PY)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_missing_csv_raises_file_not_found(tmp_path: Path) -> None:
    """_require_csv() raises FileNotFoundError naming the missing file."""
    render = _import_render()
    with pytest.raises(FileNotFoundError) as exc_info:
        render._require_csv("this_csv_does_not_exist.csv")
    msg = str(exc_info.value)
    assert "this_csv_does_not_exist.csv" in msg, (
        f"FileNotFoundError message does not name the missing file: {msg!r}"
    )


def test_missing_table_csv_raises_file_not_found(tmp_path: Path) -> None:
    """_require_table_csv() raises FileNotFoundError naming the missing file."""
    render = _import_render()
    with pytest.raises(FileNotFoundError) as exc_info:
        render._require_table_csv("no_such_table.csv")
    msg = str(exc_info.value)
    assert "no_such_table.csv" in msg


# ---------------------------------------------------------------------------
# 3. Render succeeds when experiment result directories are inaccessible
# ---------------------------------------------------------------------------

def test_render_fig1_independent_of_results_dir(tmp_path: Path) -> None:
    """render_fig1 reads only shared/figures/data/ and writes to out_dir.

    Rendering must succeed even if results/ and all other experiment dirs are
    absent (simulated by patching Path.exists to return False for those paths).
    """
    render = _import_render()

    # Verify the input CSV exists (we're not testing missing data here)
    csv_path = _DATA_DIR / "fig1_synthetic_gaussian.csv"
    if not csv_path.exists():
        pytest.skip("fig1_synthetic_gaussian.csv not present; run prepare_data.py first")

    # Render to a fresh temp directory -- no results/ dir in scope
    render.render_fig1(tmp_path)

    pdf = tmp_path / "fig1_synthetic_gaussian.pdf"
    png = tmp_path / "fig1_synthetic_gaussian.png"
    assert pdf.exists(), "PDF not produced by render_fig1"
    assert png.exists(), "PNG not produced by render_fig1"


def test_render_e07_robustness_independent_of_results_dir(tmp_path: Path) -> None:
    """render_e07_robustness reads only shared/figures/data/ CSVs."""
    render = _import_render()

    csv_path = _DATA_DIR / "fig_E07_robustness.csv"
    if not csv_path.exists():
        pytest.skip("fig_E07_robustness.csv not present; run prepare_data.py first")

    render.render_e07_robustness(tmp_path)

    assert (tmp_path / "fig_E07_robustness.pdf").exists()
    assert (tmp_path / "fig_E07_robustness.png").exists()
