"""Figure 1: Synthetic Gaussian segmentation.

This script generates a simple piecewise-constant signal corrupted by Gaussian
noise, fits :class:`bayesbreak.BayesBreakGaussian`, and saves a figure showing:

1. The noisy observations and the MAP-like piecewise-constant posterior mean.
2. The posterior probability that each interior index is a change-point.

Outputs
-------
- results/fig1_synthetic_gaussian.png

Usage
-----
python scripts/figures/fig1_synthetic_gaussian.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402


def main(outdir: Path, seed: int, n1: int, n2: int, n3: int, sigma: float) -> None:
    rng = np.random.default_rng(seed)

    mu = np.r_[np.zeros(n1), np.ones(n2), -0.5 * np.ones(n3)]
    y = mu + sigma * rng.standard_normal(mu.size)

    model = BayesBreakGaussian(k_max=10, regression_curve="mix_k").fit(y)
    pc = model.predict()
    d1 = model.get_boundary_posteriors()

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(y, lw=1, label="observed")
    ax[0].plot(pc, lw=2, label="piecewise-constant fit")
    ax[0].set_ylabel("signal")
    ax[0].legend(loc="best")

    # boundary posterior is defined on interior indices 1..n-1
    x = np.arange(1, y.size)
    ax[1].plot(x, d1, lw=1)
    ax[1].set_xlabel("index")
    ax[1].set_ylabel("P(boundary at i | y)")

    fig.tight_layout()
    fig.savefig(outdir / "fig1_synthetic_gaussian.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n1", type=int, default=50)
    p.add_argument("--n2", type=int, default=50)
    p.add_argument("--n3", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.25)
    args = p.parse_args()
    main(args.outdir, args.seed, args.n1, args.n2, args.n3, args.sigma)
