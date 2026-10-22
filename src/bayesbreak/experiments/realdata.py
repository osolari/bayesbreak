"""Run the four real-data illustrations from §6.

Equivalent to::

    python -m bayesbreak.experiments.realdata --dataset welllog --out OUTDIR
    python -m bayesbreak.experiments.realdata --dataset cgh     --out OUTDIR
    python -m bayesbreak.experiments.realdata --dataset spx     --out OUTDIR
    python -m bayesbreak.experiments.realdata --dataset methyl  --out OUTDIR
    python -m bayesbreak.experiments.realdata --dataset all     --out OUTDIR

Each invocation runs the corresponding script in ``scripts/figures/`` and
emits the figure under ``OUTDIR/figures``. The dataset loaders prefer the
real download (``pooch`` cache for the well-log; Bioconductor mirror for
CGH; ``yfinance`` for the S&P 500; the Loyfer 2023 atlas for methylation)
and fall back to a deterministic simulated analog when the download is
unavailable.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys

_SCRIPTS = {
    "welllog": "fig6_welllog.py",
    "cgh": "fig7_cgh.py",
    "spx": "fig8_spx.py",
    "methyl": "fig9_methylation.py",
}


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _run(script_path: pathlib.Path, argv: list[str]) -> None:
    print(f"[realdata] {script_path.relative_to(_project_root())} {argv}", flush=True)
    saved = sys.argv
    sys.argv = [str(script_path), *argv]
    if str(_project_root()) not in sys.path:
        sys.path.insert(0, str(_project_root()))
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bayesbreak.experiments.realdata")
    ap.add_argument(
        "--dataset",
        choices=list(_SCRIPTS.keys()) + ["all"],
        required=True,
        help="dataset to run (or 'all' for every illustration)",
    )
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--simulated",
        action="store_true",
        help="force the deterministic simulated analog",
    )
    ap.add_argument("--csv-path", default=None, help="optional CSV path for methylation")
    args, extra = ap.parse_known_args(argv)

    root = _project_root()
    fig_dir = root / "scripts" / "figures"
    out_root = args.out or (root / "docs" / "report")
    out_figs = out_root / "figures"
    os.makedirs(out_figs, exist_ok=True)

    targets = list(_SCRIPTS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        script = fig_dir / _SCRIPTS[ds]
        forwarded = ["--outdir", str(out_figs)] + list(extra)
        if args.simulated:
            forwarded.append("--simulated")
        if ds == "methyl" and args.csv_path:
            forwarded.extend(["--csv-path", args.csv_path])
        _run(script, forwarded)

    return 0


if __name__ == "__main__":
    sys.exit(main())
