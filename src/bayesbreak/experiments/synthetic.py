"""Run the §6 synthetic suite.

Equivalent to::

    python -m bayesbreak.experiments.synthetic --all --out OUTDIR

Internally invokes the per-figure / per-table scripts shipped under
``scripts/figures/`` and ``scripts/tables/``. Outputs land in
``OUTDIR/figures`` and ``OUTDIR/tables`` (defaulting to
``docs/report/figures`` and ``docs/report/tables`` so the report build
picks them up directly).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


_SYNTH_FIGS = [
    "fig1_synthetic_gaussian.py",
    "fig2_family_showcase.py",
    "fig3_boundary_calibration.py",
    "fig4_latent_groups.py",
    "fig5_runtime_scaling.py",
]
_SYNTH_TABLES = [
    "table0_metrics_overview.py",
    "table1_runtime_scaling.py",
    "table2_posterior_summary.py",
    "table3_conjugate_summary.py",
    "table4_nonconj_tradeoff.py",
]


def _run_script(script_path: pathlib.Path, argv: list[str]) -> None:
    """Execute a script with controlled ``sys.argv``."""

    print(f"[synthetic] {script_path.relative_to(_project_root())} {argv}", flush=True)
    saved = sys.argv
    sys.argv = [str(script_path), *argv]
    # Make project root importable.
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bayesbreak.experiments.synthetic")
    ap.add_argument("--all", action="store_true", help="run figures and tables")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--include-supplementary",
        action="store_true",
        help="also run scripts/figures/supplementary/*.py",
    )
    args = ap.parse_args(argv)

    do_figs = args.figures or args.all or (not args.figures and not args.tables)
    do_tabs = args.tables or args.all or (not args.figures and not args.tables)

    root = _project_root()
    fig_dir = root / "scripts" / "figures"
    tab_dir = root / "scripts" / "tables"
    out_root = args.out or (root / "docs" / "report")
    out_figs = out_root / "figures"
    out_tabs = out_root / "tables"
    os.makedirs(out_figs, exist_ok=True)
    os.makedirs(out_tabs, exist_ok=True)

    if do_figs:
        for name in _SYNTH_FIGS:
            _run_script(fig_dir / name, ["--outdir", str(out_figs)])
        if args.include_supplementary:
            sup = fig_dir / "supplementary"
            if sup.exists():
                for p in sorted(sup.glob("*.py")):
                    _run_script(p, ["--outdir", str(out_figs)])

    if do_tabs:
        for name in _SYNTH_TABLES:
            _run_script(tab_dir / name, ["--outdir", str(out_tabs)])

    return 0


if __name__ == "__main__":
    sys.exit(main())
