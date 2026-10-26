"""Data preparation: run all computational figure scripts to refresh CSV caches.

This script re-executes Stage A inference for every figure and table that
requires BayesBreak model fitting or experiment data. It writes plot-ready
CSV and JSON files to the shared data directories so that the cache-only
renderer can reproduce all publication assets without repeating computation.

PUBLICATION RENDERING
---------------------
To regenerate publication figures and tables from cached data use:

    python scripts/figures/render.py --all

That command reads only checked-in CSV/JSON and writes PDF + PNG to
``docs/manuscripts/paper/figures/`` in approximately 8–10 seconds. It is the
single authoritative entry point for publication assets.

DATA PREPARATION (this script)
-------------------------------
Run this script only when experiment results or data sources have changed and
the CSV caches need to be rebuilt. It invokes scripts that run BayesBreak
inference and may take minutes.

Categories rebuilt:
1. Showcase figures (fig1–fig9): scripts under ``scripts/figures/`` that
   fit BayesBreak on simulated or real data and write CSVs + a local figure.
   Local figures go to ``results/figures/`` (not the canonical manuscript dir).
2. Confirmatory figure data (E01, E02): runners under ``scripts/`` that
   re-read frozen ``results/confirmatory/`` metrics and write table CSVs.
3. Table data: scripts under ``scripts/tables/`` that produce CSV inputs.

Usage
-----
uv run --extra plots --extra datasets python scripts/generate_all_figures.py

Optional flags
--------------
--skip-exploratory   Skip scripts/figures/*.py and scripts/tables/*.py.
--skip-confirmatory  Skip E01/E02 data regeneration.
--skip-tables        Skip all table data generation.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
# Exploratory scripts write figures here (staging area, not publication dir).
_STAGING_FIGS = ROOT / "results" / "figures"
MANUSCRIPT_TABLES = ROOT / "docs" / "manuscripts" / "shared" / "tables" / "results"

_CONFIRMATORY_RUNNERS = [
    {
        "script": ROOT / "scripts" / "run_e01_gaussian_calibration.py",
        "exp_dirs": [ROOT / "results" / "confirmatory" / "E01_gaussian_calibration"],
        "expected_tables": ["tab_E01_recovery.tex"],
    },
    {
        "script": ROOT / "scripts" / "run_e02_family_portability.py",
        "exp_dirs": [ROOT / "results" / "confirmatory" / "E02_family_portability"],
        "expected_tables": ["tab_E02_recovery.tex"],
    },
]


def _run(cmd: list[str | Path]) -> None:
    str_cmd = [str(c) for c in cmd]
    logger.info("[run] %s", " ".join(str_cmd))
    subprocess.run(str_cmd, check=True)


def _latest_results_dir(experiment_dirs: list[Path]) -> Path | None:
    """Return the most recently modified subdirectory containing metrics.json."""
    candidates: list[Path] = []
    for exp_dir in experiment_dirs:
        if not exp_dir.is_dir():
            continue
        for sub in exp_dir.iterdir():
            if sub.is_dir() and (sub / "metrics.json").exists():
                candidates.append(sub)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_exploratory_figures() -> None:
    """Run showcase figure scripts to refresh CSV data caches.

    Figures are written to results/figures/ (staging only).
    Run ``python scripts/figures/render.py --all`` afterwards to produce
    the canonical publication PDF/PNG from the refreshed CSVs.
    """
    _STAGING_FIGS.mkdir(parents=True, exist_ok=True)
    # Run fig1–fig9 generation scripts; skip private helpers (_style, _realdata).
    fig_scripts = sorted(p for p in (ROOT / "scripts" / "figures").glob("fig*.py")
                         if not p.name.startswith("_"))
    for script in fig_scripts:
        _run([sys.executable, script, "--outdir", _STAGING_FIGS])


def run_exploratory_tables() -> None:
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)
    table_scripts = sorted((ROOT / "scripts" / "tables").glob("*.py"))
    for script in table_scripts:
        _run([sys.executable, script, "--outdir", MANUSCRIPT_TABLES])


def run_confirmatory(skip_tables: bool = False) -> None:
    """Re-read confirmatory results and refresh table CSV caches."""
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)

    for entry in _CONFIRMATORY_RUNNERS:
        runner_script = entry["script"]
        results_dir = _latest_results_dir(entry["exp_dirs"])
        if results_dir is None:
            logger.warning(
                "No completed results found for %s — skipping. Run the experiment first.",
                runner_script.name,
            )
            continue

        cmd = [sys.executable, runner_script, "--regenerate-figures"]
        if not skip_tables:
            cmd += ["--tables-dest", MANUSCRIPT_TABLES]
        _run(cmd)

        if not skip_tables:
            for fname in entry.get("expected_tables", []):
                dest = MANUSCRIPT_TABLES / fname
                if dest.exists():
                    logger.info("  ✓ %s", fname)
                else:
                    logger.warning("  ✗ %s not found at %s", fname, dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh data caches for all manuscript figures and tables. "
                    "For publication rendering use: python scripts/figures/render.py --all"
    )
    parser.add_argument(
        "--skip-exploratory", action="store_true",
        help="Skip scripts/figures/*.py and scripts/tables/*.py",
    )
    parser.add_argument(
        "--skip-confirmatory", action="store_true",
        help="Skip E01/E02 data regeneration",
    )
    parser.add_argument(
        "--skip-tables", action="store_true",
        help="Skip all table data generation",
    )
    args = parser.parse_args()

    if not args.skip_exploratory:
        logger.info("=== Exploratory figure data ===")
        run_exploratory_figures()
    if not args.skip_exploratory and not args.skip_tables:
        logger.info("=== Exploratory table data ===")
        run_exploratory_tables()
    if not args.skip_confirmatory:
        logger.info("=== Confirmatory table data ===")
        run_confirmatory(skip_tables=args.skip_tables)

    logger.info("Data caches refreshed.")
    logger.info("Run: python scripts/figures/render.py --all")
    logger.info("to regenerate publication figures from updated caches.")


if __name__ == "__main__":
    main()
