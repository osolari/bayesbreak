"""Table 0: Overview of metrics reported in the experiments section.

This table is purely descriptive (it does not depend on random seeds or data)
but we generate it via a script so that every table appearing in the report has
a single source of truth in the repository.

Content
-------
The table lists the five evaluation axes used throughout the paper:

1. **Boundary recovery** — Precision / Recall / F1 within a tolerance
   :math:`\\tau` (number of index positions) around each true changepoint.
2. **Signal recovery** — Mean squared error (MSE) between the estimated
   piecewise-constant function and the true latent signal.
3. **Posterior calibration** — Reliability diagram and expected calibration
   error (ECE) for the marginal boundary probabilities :math:`p(b_i=1\\mid y)`.
4. **Model fit quality** — Negative log marginal evidence :math:`-\\log p(y)`
   or per-observation negative log-likelihood.
5. **Runtime** — Wall-clock time as a function of :math:`n` and
   :math:`k_{\\max}`.

Interpretation
--------------
This table is a reference legend: it does not contain experimental results.
It defines the metrics so that subsequent tables (1–4) can refer to short
column headers (e.g., "F1@τ", "MSE") without repeating definitions.

Outputs
-------
- results/tables/table0_metrics_overview.tex
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
    ap.add_argument("--outdir", type=Path, default=Path("results/tables"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.outdir)
