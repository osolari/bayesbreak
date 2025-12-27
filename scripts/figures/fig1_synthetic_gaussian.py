"""Figure 1: Synthetic Gaussian example.

This script generates a simple piecewise-constant latent mean sequence, corrupts
it with Gaussian noise, and fits :class:`bayesbreak.BayesBreakGaussian`.

The resulting figure is designed to match the paper's "single-sequence" results
panel:

1. **Top panel**: marginal posterior boundary probabilities
   :math:`p(b_i = 1 \mid y)` for each interior index ``i``.
2. **Bottom panel**: the observations, the recovered piecewise-constant posterior
   mean under the selected boundaries, and a 90% Normal credible interval for
   the *segment mean* (conditional on the selected partition).

Notes
-----
The credible interval shown here is *not* a fully marginal band over all
segmentations; it conditions on the selected boundaries produced by BayesBreak.
This keeps the script lightweight while still providing an informative
uncertainty visualisation.

Outputs
-------
- results/fig1_synthetic_gaussian.png

Usage
-----
python scripts/figures/fig1_synthetic_gaussian.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402


def _segment_mean_ci_gaussian(
    *,
    boundaries: list[int],
    n: int,
    rho2: float,
    sigma2: float,
    pc_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 90% pointwise CI for the segment mean under selected boundaries.

    For a Normal--Normal conjugate segment model with known ``sigma2`` and prior
    variance ``rho2``, the posterior over the segment mean is Normal with

        Var(mu | segment) = 1 / (1/rho2 + m/sigma2)

    where ``m`` is the segment length (assuming unit observation weights).
    """

    # 90% interval => z_{0.95}
    z = 1.6448536269514722

    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        m = float(b - a)
        post_var = 1.0 / (1.0 / max(rho2, 1e-12) + m / max(sigma2, 1e-12))
        s = z * math.sqrt(max(post_var, 0.0))
        lo[a:b] = pc_fit[a:b] - s
        hi[a:b] = pc_fit[a:b] + s
    return lo, hi


def main(outdir: Path, seed: int, n1: int, n2: int, n3: int, sigma: float) -> None:
    rng = np.random.default_rng(seed)

    mu = np.r_[np.zeros(n1), np.ones(n2), -0.5 * np.ones(n3)]
    y = mu + sigma * rng.standard_normal(mu.size)

    model = BayesBreakGaussian(k_max=10, regression_curve="mix_k").fit(y)
    pc = model.predict()
    d1 = model.get_boundary_posteriors()
    boundaries = model.get_boundaries()

    # Credible interval for the latent segment mean (conditional on selected
    # boundaries).
    hyper = model.hyper_ or {}
    rho2 = float(hyper.get("rho2", 1.0))
    sigma2 = float(hyper.get("sigma2", sigma * sigma))
    lo, hi = _segment_mean_ci_gaussian(boundaries=boundaries, n=y.size, rho2=rho2, sigma2=sigma2, pc_fit=pc)

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Top: boundary posteriors ---
    x_b = np.arange(1, y.size)
    ax[0].plot(x_b, d1, lw=1)
    ax[0].set_ylabel("P(boundary at i | y)")
    # Mark true boundaries for reference.
    for tb in (n1, n1 + n2):
        ax[0].axvline(tb, linestyle="--", linewidth=1)

    # --- Bottom: signal reconstruction + CI ---
    ax[1].plot(y, lw=1, label="observed")
    ax[1].plot(mu, lw=2, linestyle="--", label="true mean")
    ax[1].plot(pc, lw=2, label="segment posterior mean")
    ax[1].fill_between(np.arange(y.size), lo, hi, alpha=0.2, label="90% CI (segment mean)")
    ax[1].set_xlabel("index")
    ax[1].set_ylabel("signal")
    ax[1].legend(loc="best")

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
