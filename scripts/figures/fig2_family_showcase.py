"""Figure 2: Likelihood family showcase.

This script demonstrates BayesBreak on four short synthetic series, one per
supported family:

- Gaussian (Normal--Normal)
- Poisson (Gamma--Poisson)
- Binomial (Beta--Binomial)
- Beta-valued (fractional Beta--Binomial)

The goal is to provide a sanity check and a visual reference for users.

Outputs
-------
- results/fig2_family_showcase.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bayesbreak import (  # noqa: E402
    BayesBreakBeta,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakPoisson,
)


def main(outdir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    # ----------------
    # Gaussian example
    # ----------------
    n = 120
    mu = np.r_[np.zeros(40), 1.5 * np.ones(40), -0.5 * np.ones(40)]
    y_gauss = mu + 0.35 * rng.standard_normal(n)
    m_gauss = BayesBreakGaussian(k_max=10).fit(y_gauss)
    pc_gauss = m_gauss.predict()

    # ----------------
    # Poisson example
    # ----------------
    lam = np.r_[2.0 * np.ones(40), 8.0 * np.ones(40), 3.0 * np.ones(40)]
    y_pois = rng.poisson(lam)
    m_pois = BayesBreakPoisson(k_max=10).fit(y_pois)
    pc_pois = m_pois.predict()

    # ----------------
    # Binomial example
    # ----------------
    n_trials = 20
    p = np.r_[0.1 * np.ones(40), 0.7 * np.ones(40), 0.3 * np.ones(40)]
    y_binom = rng.binomial(n_trials, p)
    m_binom = BayesBreakBinomial(k_max=10, n_trials=n_trials).fit(y_binom)
    pc_binom = m_binom.predict()

    # ----------------
    # Beta-valued example
    # ----------------
    # Draw pseudo-counts from Binomial, map to y in (0,1) as y=s/kappa.
    kappa = 50
    p_beta = np.r_[0.2 * np.ones(40), 0.85 * np.ones(40), 0.4 * np.ones(40)]
    s = rng.binomial(kappa, p_beta)
    y_beta = (s + 0.5) / (kappa + 1.0)  # avoid exact 0/1
    m_beta = BayesBreakBeta(k_max=10, concentration=kappa).fit(y_beta)
    pc_beta = m_beta.predict()

    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(y_gauss, linewidth=1, label="observed")
    axes[0].plot(mu, linewidth=2, linestyle="--", label="true signal")
    axes[0].plot(pc_gauss, linewidth=2, label="BayesBreak fit")
    axes[0].set_title("Gaussian")

    axes[1].plot(y_pois, linewidth=1)
    axes[1].plot(lam, linewidth=2, linestyle="--")
    axes[1].plot(pc_pois, linewidth=2)
    axes[1].set_title("Poisson")

    axes[2].plot(y_binom / n_trials, linewidth=1)
    axes[2].plot(p, linewidth=2, linestyle="--")
    axes[2].plot(pc_binom, linewidth=2)
    axes[2].set_title("Binomial (shown as proportion)")

    axes[3].plot(y_beta, linewidth=1)
    axes[3].plot(p_beta, linewidth=2, linestyle="--")
    axes[3].plot(pc_beta, linewidth=2)
    axes[3].set_title("Beta-valued (fractional Beta--Binomial)")
    axes[3].set_xlabel("Index")

    # One shared legend for all panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(outdir / "fig2_family_showcase.png", dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.outdir, args.seed)
