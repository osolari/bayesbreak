"""Run all figure and table generation scripts.

This is a convenience wrapper that sequentially executes the scripts under
`scripts/figures/` and `scripts/tables/`.

Usage
-----
python scripts/make_all_artifacts.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(path: Path) -> None:
    print(f"[run] {path}")
    subprocess.run([sys.executable, str(path)], check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for p in sorted((root / "scripts" / "figures").glob("*.py")):
        _run(p)
    for p in sorted((root / "scripts" / "tables").glob("*.py")):
        _run(p)


if __name__ == "__main__":
    main()
