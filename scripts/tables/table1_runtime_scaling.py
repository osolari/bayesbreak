r"""Table 1: Simple runtime scaling benchmark.

This script measures wall-clock runtime of the Gaussian BayesBreak fit for a
range of series lengths :math:`n` and a fixed :math:`k_{\max}`.

Experiment
----------
For each :math:`n \in \{50, 100, 200, 400\}` the fit is repeated ``--repeats``
times (default 5).  Each repetition uses a fresh random seed to avoid caching
artefacts.  Mean and standard-deviation of elapsed wall-clock time (via
``time.perf_counter``) are reported.

Interpretation
--------------
Because the core DP algorithm is :math:`O(k_{\max} \cdot n^2)`:

- Doubling :math:`n` should roughly quadruple the runtime.
- The table is intentionally lightweight and is meant for *relative*
  comparisons (e.g., verifying that a code refactor did not introduce a
  performance regression) rather than absolute claims.
- Large standard deviations may indicate GC pauses, memory allocation
  overhead, or thermal throttling.

Outputs
-------
- results/table1_runtime_scaling.csv
- results/table1_runtime_scaling.md
- results/table1_runtime_scaling.tex

Usage
-----
python scripts/tables/table1_runtime_scaling.py [--k-max 20 --repeats 10]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402


def _fit_once(n: int, k_max: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    mu = np.r_[np.zeros(n // 3), np.ones(n // 3), -0.5 * np.ones(n - 2 * (n // 3))]
    y = mu + 0.25 * rng.standard_normal(n)

    t0 = time.perf_counter()
    BayesBreakGaussian(k_max=k_max).fit(y)
    t1 = time.perf_counter()
    return t1 - t0


def main(outdir: Path, k_max: int, repeats: int, seed: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ns = [50, 100, 200, 400]
    rows = []
    for idx, n in enumerate(ns):
        times = [_fit_once(n=n, k_max=k_max, seed=seed + 1000 * idx + r) for r in range(repeats)]
        rows.append(
            (n, k_max, float(np.mean(times)), float(np.std(times, ddof=1)) if repeats > 1 else 0.0)
        )

    csv_path = outdir / "table1_runtime_scaling.csv"
    md_path = outdir / "table1_runtime_scaling.md"
    tex_path = outdir / "table1_runtime_scaling.tex"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("n,k_max,mean_seconds,std_seconds\n")
        for n, km, m, s in rows:
            f.write(f"{n},{km},{m:.6f},{s:.6f}\n")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| n | k_max | mean (s) | std (s) |\n")
        f.write("|---:|-----:|---------:|--------:|\n")
        for n, km, m, s in rows:
            f.write(f"| {n} | {km} | {m:.4f} | {s:.4f} |\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    # LaTeX tabular (for \input in the paper).
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrr}\\toprule\n")
        f.write("n & $k_{\\max}$ & mean (s) & std (s)\\\\\\midrule\n")
        for n, km, m, s in rows:
            f.write(f"{n} & {km} & {m:.4f} & {s:.4f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--k-max", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(outdir=args.outdir, k_max=args.k_max, repeats=args.repeats, seed=args.seed)
