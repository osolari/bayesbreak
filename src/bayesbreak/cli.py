"""Command-line interface for BayesBreak.

Exposed as the ``bayesbreak`` entry point. Sub-commands:

- ``bayesbreak reproduce {figures,tables,all}`` — regenerate artifacts under
  ``results/`` used by the report.
- ``bayesbreak version`` — print the installed version.
"""

from __future__ import annotations

import argparse
import sys

from ._version import __version__


def _run_reproduce(target: str) -> int:
    from .reproduce import reproduce

    reproduce(target)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayesbreak",
        description="Exact Bayesian segmentation: block evidence + DP.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    repro = sub.add_parser("reproduce", help="regenerate report figures / tables")
    repro.add_argument(
        "target",
        choices=["figures", "tables", "all"],
        help="which artifacts to regenerate",
    )

    sub.add_parser("version", help="print the installed version")

    args = parser.parse_args(argv)

    if args.cmd == "reproduce":
        return _run_reproduce(args.target)
    if args.cmd == "version":
        print(__version__)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
