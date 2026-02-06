"""Figure 5: Runtime scaling benchmark.

This script times :class:`bayesbreak.BayesBreakGaussian` for a small grid of
series lengths ``n`` and maximum segment counts ``k_max``. It produces a simple
runtime plot suitable for inclusion in the paper.

Notes
-----
The benchmark is meant for *relative* comparisons (e.g., after code changes)
rather than absolute performance claims.

Outputs
-------
- results/fig5_runtime_scaling.png
- results/fig5_runtime_scaling.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402


def _fit_once(rng: np.random.Generator, n: int, k_max: int) -> float:
    mu = np.r_[np.zeros(n // 3), np.ones(n // 3), -0.5 * np.ones(n - 2 * (n // 3))]
    y = mu + 0.25 * rng.standard_normal(n)

    t0 = time.perf_counter()
    BayesBreakGaussian(k_max=k_max).fit(y)
    t1 = time.perf_counter()
    return t1 - t0


def main(outdir: Path, seed: int, repeats: int) -> None:
    rng = np.random.default_rng(seed)

    ns = [50, 100, 200, 400]
    k_maxs = [10, 20]

    rows = []
    for k_max in k_maxs:
        for n in ns:
            times = [_fit_once(rng, n=n, k_max=k_max) for _ in range(repeats)]
            rows.append(
                (
                    n,
                    k_max,
                    float(np.mean(times)),
                    float(np.std(times, ddof=1)) if repeats > 1 else 0.0,
                )
            )

    outdir.mkdir(parents=True, exist_ok=True)

    # Save CSV for reproducibility.
    csv_path = outdir / "fig5_runtime_scaling.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("n,k_max,mean_seconds,std_seconds\n")
        for n, k_max, m, s in rows:
            f.write(f"{n},{k_max},{m:.6f},{s:.6f}\n")

    # Plot.
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for k_max in k_maxs:
        xs = [n for n, km, _, _ in rows if km == k_max]
        ys = [m for n, km, m, _ in rows if km == k_max]
        es = [s for n, km, _, s in rows if km == k_max]
        ax.errorbar(xs, ys, yerr=es, marker="o", linewidth=1, label=f"k_max={k_max}")

    ax.set_xlabel("series length n")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_title("BayesBreak Gaussian runtime scaling")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "fig5_runtime_scaling.png", dpi=200)
    plt.close(fig)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    main(outdir=args.outdir, seed=args.seed, repeats=args.repeats)
