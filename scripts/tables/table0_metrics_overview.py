"""Table 0: Overview of metrics reported in the experiments section.

This table is descriptive (it does not depend on random seeds) but we generate
it via a script so that every table appearing in the report has a single source
of truth in the repository.

Outputs
-------
- results/table0_metrics_overview.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("\\begin{tabular}{ll}\\toprule\n")
    lines.append("Evaluation objective & Metric\\\\\\midrule\n")
    lines.append("Boundary recovery & Precision / Recall / F1 at tolerance $\\tau$\\\\\n")
    lines.append("Signal recovery & MSE between estimated and true latent signal\\\\\n")
    lines.append("Posterior calibration & Calibration curve / ECE for $p(b_i=1\\mid y)$\\\\\n")
    lines.append("Model fit quality & $-\\log p(y)$ (marginal evidence) or predictive NLL\\\\\n")
    lines.append("Runtime & Wall-clock time vs. $n$ and $k_{\\max}$\\\\\\bottomrule\n")
    lines.append("\\end{tabular}\n")

    (outdir / "table0_metrics_overview.tex").write_text("".join(lines))


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.outdir)
