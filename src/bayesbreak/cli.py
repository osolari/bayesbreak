"""Command-line interface for BayesBreak.

Sub-commands:

- ``bayesbreak synthetic [--all|--figures|--tables] [--out DIR]``
  — runs the §6 synthetic suite (delegates to
  :mod:`bayesbreak.experiments.synthetic`).
- ``bayesbreak realdata --dataset {welllog,cgh,spx,methyl,all} [--out DIR]``
  — runs the real-data illustrations (delegates to
  :mod:`bayesbreak.experiments.realdata`).
- ``bayesbreak version``
"""

from __future__ import annotations

import argparse
import sys

from ._version import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayesbreak",
        description="Exact Bayesian segmentation: block evidence + DP.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    synth = sub.add_parser("synthetic", help="run the §6 synthetic suite")
    synth.add_argument("--all", action="store_true")
    synth.add_argument("--figures", action="store_true")
    synth.add_argument("--tables", action="store_true")
    synth.add_argument("--out", default=None)
    synth.add_argument("--include-supplementary", action="store_true")

    real = sub.add_parser("realdata", help="run a real-data illustration")
    real.add_argument(
        "--dataset",
        choices=["welllog", "cgh", "spx", "methyl", "all"],
        required=True,
    )
    real.add_argument("--out", default=None)
    real.add_argument("--simulated", action="store_true")
    real.add_argument("--csv-path", default=None)

    sub.add_parser("version", help="print the installed version")

    args, extra = parser.parse_known_args(argv)

    if args.cmd == "synthetic":
        from .experiments import synthetic as mod

        forwarded = []
        if args.all:
            forwarded.append("--all")
        if args.figures:
            forwarded.append("--figures")
        if args.tables:
            forwarded.append("--tables")
        if args.out:
            forwarded.extend(["--out", args.out])
        if args.include_supplementary:
            forwarded.append("--include-supplementary")
        return mod.main(forwarded + list(extra))

    if args.cmd == "realdata":
        from .experiments import realdata as mod

        forwarded = ["--dataset", args.dataset]
        if args.out:
            forwarded.extend(["--out", args.out])
        if args.simulated:
            forwarded.append("--simulated")
        if args.csv_path:
            forwarded.extend(["--csv-path", args.csv_path])
        return mod.main(forwarded + list(extra))

    if args.cmd == "version":
        print(__version__)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
