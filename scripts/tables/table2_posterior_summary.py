r"""Table 2: Posterior summary on a synthetic dataset.

This script fits :class:`bayesbreak.BayesBreakGaussian` on a single synthetic
three-segment Gaussian sequence and reports key summary statistics of the
posterior over the number of segments :math:`k`.

Experiment
----------
A piecewise-constant signal with levels :math:`(0, 1, -0.5)` and equal-length
segments (:math:`n/3` each) is generated with additive Gaussian noise
(:math:`\sigma = 0.25`).  The model is fit with ``k_max`` (default 10).

Quantities reported:

- :math:`\hat{k}` (**selected k**): the maximum-likelihood number of segments
  chosen by the DP.
- :math:`\mathbb{E}[k]` (**posterior mean**): :math:`\sum_k k \cdot P(k \mid y)`.
- :math:`\arg\max_k P(k \mid y)` (**MAP k**).
- :math:`\log p(y)` (**log marginal evidence**): the normalising constant
  computed as a by-product of the forward DP pass.

Interpretation
--------------
This table is a quick "sanity check" for the posterior over :math:`k`:

- For the default data-generating process (3 segments, low noise), we expect
  :math:`\hat{k} = 3` and :math:`\mathbb{E}[k] \approx 3`.
- Deviations after code changes indicate that hyperparameter estimation or
  numerical stability may have been affected.
- The log evidence is useful for comparing model fits across different
  configurations or data sets; larger (less negative) values indicate a better
  fit, penalised by model complexity.

Outputs
-------
- results/tables/table2_posterior_summary.md
- results/tables/table2_posterior_summary.tex

Usage
-----
python scripts/tables/table2_posterior_summary.py [--n 150 --k-max 15]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402


def main(outdir: Path, n: int, k_max: int, seed: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    mu = np.r_[np.zeros(n // 3), np.ones(n // 3), -0.5 * np.ones(n - 2 * (n // 3))]
    y = mu + 0.25 * rng.standard_normal(n)

    m = BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y)).reshape(-1, 1), y)

    C = m.k_posterior_
    k_grid = np.arange(1, C.size + 1)
    ek = float(np.sum(k_grid * C))
    k_map = int(k_grid[np.argmax(C)])

    md = []
    md.append("# Table 2: Posterior summary\n")
    md.append(f"- n = {n}\n")
    md.append(f"- k_max = {k_max}\n")
    md.append("\n")
    md.append("| Quantity | Value |\n")
    md.append("|---|---:|\n")
    md.append(f"| Selected k (k_ml_) | {m.k_map_} |\n")
    md.append(f"| Posterior mean E[k] | {ek:.3f} |\n")
    md.append(f"| MAP k | {k_map} |\n")
    md.append(f"| log evidence log P(y) | {float(m.log_evidence_):.3f} |\n")

    (outdir / "table2_posterior_summary.md").write_text("".join(md))

    # LaTeX tabular (for \input in the paper).
    tex_lines = []
    tex_lines.append("\\begin{tabular}{lr}\\toprule\n")
    tex_lines.append("Quantity & Value\\\\\\midrule\n")
    tex_lines.append(f"Selected $k$ (\\texttt{{k\\_ml\\_}}) & {m.k_map_}\\\\\n")
    tex_lines.append(f"Posterior mean $\\mathbb{{E}}[k]$ & {ek:.3f}\\\\\n")
    tex_lines.append(f"MAP $k$ & {k_map}\\\\\n")
    tex_lines.append(f"$\\log p(y)$ & {float(m.log_evidence_):.3f}\\\\\n")
    tex_lines.append("\\bottomrule\\end{tabular}\n")
    (outdir / "table2_posterior_summary.tex").write_text("".join(tex_lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results/tables"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(outdir=args.outdir, n=args.n, k_max=args.k_max, seed=args.seed)
