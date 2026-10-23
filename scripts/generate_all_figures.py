"""Generate all manuscript figures and tables in one pass.

All figures are written to ``docs/manuscripts/shared/figures/results/``.
All tables are written to ``docs/manuscripts/shared/tables/results/``.

Covers two categories:

1. Exploratory figures (fig1–fig9) and tables (table1–table4): scripts under
   ``scripts/figures/`` and ``scripts/tables/`` that generate their own data
   on the fly.  Each script accepts ``--outdir`` and writes there directly.
   Only the nine main figure scripts (fig1–fig9) are run; supplementary
   scripts (figA1–figA5) are not referenced in the paper or book and are
   excluded.

2. Confirmatory experiment figures and tables (E01, E02): regenerated from
   frozen metrics.json files under ``results/confirmatory/`` without
   re-running experiments.  Each confirmatory runner exposes
   ``--regenerate-figures`` with optional ``--manuscript-dest`` and
   ``--tables-dest`` flags.

Usage
-----
uv run --extra plots --extra datasets python scripts/generate_all_figures.py

Optional flags
--------------
--skip-exploratory   Skip the scripts/figures/*.py and scripts/tables/*.py
                     scripts (faster when you only want to refresh E01/E02).
--skip-confirmatory  Skip E01/E02 figure and table regeneration.
--skip-tables        Skip all table generation (exploratory + confirmatory).
--skip-figures       Skip all figure generation (exploratory + confirmatory).
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
MANUSCRIPT_FIGS = ROOT / "docs" / "manuscripts" / "shared" / "figures" / "results"
MANUSCRIPT_TABLES = ROOT / "docs" / "manuscripts" / "shared" / "tables" / "results"

_CONFIRMATORY_RUNNERS = [
    {
        "script": ROOT / "scripts" / "run_e01_gaussian_calibration.py",
        "exp_dirs": [ROOT / "results" / "confirmatory" / "E01_gaussian_calibration"],
        "expected_figures": [
            "fig_E01_calibration.pdf", "fig_E01_calibration.png",
            "fig_E01_recovery.pdf",    "fig_E01_recovery.png",
        ],
        "expected_tables": ["tab_E01_recovery.tex"],
    },
    {
        "script": ROOT / "scripts" / "run_e02_family_portability.py",
        "exp_dirs": [ROOT / "results" / "confirmatory" / "E02_family_portability"],
        "expected_figures": ["fig_E02_recovery.pdf", "fig_E02_recovery.png"],
        "expected_tables": ["tab_E02_recovery.tex"],
    },
]


def _run(cmd: list[str | Path]) -> None:
    str_cmd = [str(c) for c in cmd]
    logger.info("[run] %s", " ".join(str_cmd))
    subprocess.run(str_cmd, check=True)


def _latest_results_dir(experiment_dirs: list[Path]) -> Path | None:
    """Return the most recently modified subdirectory that contains metrics.json."""
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
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    # Main figure scripts (fig1–fig9); skip private helpers (_style.py, _realdata.py)
    fig_scripts = sorted(p for p in (ROOT / "scripts" / "figures").glob("*.py")
                         if not p.name.startswith("_"))
    for script in fig_scripts:
        _run([sys.executable, script, "--outdir", MANUSCRIPT_FIGS])
    # realdata_tables.py generates realdata_metrics.json validated by test_result_schema
    realdata = ROOT / "scripts" / "tables" / "realdata_tables.py"
    if realdata.exists():
        _run([sys.executable, realdata, "--outdir", MANUSCRIPT_FIGS])


def run_exploratory_tables() -> None:
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)
    table_scripts = sorted((ROOT / "scripts" / "tables").glob("*.py"))
    for script in table_scripts:
        _run([sys.executable, script, "--outdir", MANUSCRIPT_TABLES])


def run_confirmatory(skip_figures: bool = False, skip_tables: bool = False) -> None:
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
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

        cmd = [
            sys.executable, runner_script,
            "--regenerate-figures",
        ]
        if not skip_figures:
            cmd += ["--manuscript-dest", MANUSCRIPT_FIGS]
        if not skip_tables:
            cmd += ["--tables-dest", MANUSCRIPT_TABLES]
        _run(cmd)

        if not skip_figures:
            for fname in entry["expected_figures"]:
                dest = MANUSCRIPT_FIGS / fname
                if dest.exists():
                    logger.info("  ✓ %s", fname)
                else:
                    logger.warning("  ✗ %s not found at %s", fname, dest)

        if not skip_tables:
            for fname in entry["expected_tables"]:
                dest = MANUSCRIPT_TABLES / fname
                if dest.exists():
                    logger.info("  ✓ %s", fname)
                else:
                    logger.warning("  ✗ %s not found at %s", fname, dest)


_MANUSCRIPT_FIGURE_STEMS = {
    # Exploratory
    "fig1_synthetic_gaussian", "fig2_family_showcase", "fig3_boundary_calibration",
    "fig4_latent_groups_cropped", "fig5_runtime_scaling", "fig6_welllog",
    "fig7_cgh", "fig8_spx", "fig9_methylation",
    # Static (not generated by scripts)
    "fig_cgh_probe_agreement", "fig_methylation_heldout", "fig_latent_template_stress",
    # Confirmatory
    "fig_E01_calibration", "fig_E01_recovery", "fig_E02_recovery",
}
# JSON sidecar provenance files validated by test_result_schema.py
_FIGURE_SIDECAR_NAMES = {
    "fig6_welllog.json", "fig7_cgh.json", "fig8_spx.json",
    "fig9_methylation.json", "realdata_metrics.json",
}
_MANUSCRIPT_TABLE_STEMS = {
    "table1_runtime_scaling", "table2_posterior_summary",
    "table3_conjugate_summary", "table4_nonconj_tradeoff",
    "tab_E01_recovery", "tab_E02_recovery",
}


def _prune_output_dirs() -> None:
    """Remove files from the output directories that are not manuscript artifacts."""
    # Keep PDF/PNG for manuscript figures and JSON sidecars required by schema tests
    keep_exts = {".pdf", ".png"}
    for f in MANUSCRIPT_FIGS.iterdir():
        if f.suffix in keep_exts and f.stem in _MANUSCRIPT_FIGURE_STEMS:
            continue
        if f.name in _FIGURE_SIDECAR_NAMES:
            continue
        logger.info("  removing non-manuscript artifact: %s", f.name)
        f.unlink()
    # Keep only referenced .tex tables; prune CSV, MD, JSON, and unreferenced tex
    table_keep_exts = {".tex"}
    for f in MANUSCRIPT_TABLES.iterdir():
        if f.suffix in table_keep_exts and f.stem in _MANUSCRIPT_TABLE_STEMS:
            continue
        logger.info("  removing non-manuscript artifact (tables): %s", f.name)
        f.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all manuscript figures and tables (exploratory + confirmatory)."
    )
    parser.add_argument(
        "--skip-exploratory", action="store_true",
        help="Skip scripts/figures/*.py and scripts/tables/*.py",
    )
    parser.add_argument(
        "--skip-confirmatory", action="store_true",
        help="Skip E01/E02 figure and table regeneration",
    )
    parser.add_argument(
        "--skip-tables", action="store_true",
        help="Skip all table generation (exploratory and confirmatory)",
    )
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="Skip all figure generation (exploratory and confirmatory)",
    )
    args = parser.parse_args()

    if not args.skip_exploratory and not args.skip_figures:
        logger.info("=== Exploratory figures ===")
        run_exploratory_figures()
    if not args.skip_exploratory and not args.skip_tables:
        logger.info("=== Exploratory tables ===")
        run_exploratory_tables()
    if not args.skip_confirmatory:
        logger.info("=== Confirmatory figures and tables ===")
        run_confirmatory(
            skip_figures=args.skip_figures,
            skip_tables=args.skip_tables,
        )

    logger.info("=== Pruning non-manuscript artifacts ===")
    _prune_output_dirs()
    logger.info("Done.")
    logger.info("  Figures → %s", MANUSCRIPT_FIGS)
    logger.info("  Tables  → %s", MANUSCRIPT_TABLES)


if __name__ == "__main__":
    main()
