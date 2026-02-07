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
- results/fig2_family_showcase.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _style import (  # noqa: E402
    COLORS,
    add_panel_label,
    get_figsize,
    save_figure,
    setup_style,
)

from bayesbreak import (  # noqa: E402
    BayesBreakBeta,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakPoisson,
)


def main(outdir: Path, seed: int) -> None:
    # Setup publication style
    setup_style(font_scale=1.1, style="paper")

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
    kappa = 50
    p_beta = np.r_[0.2 * np.ones(40), 0.85 * np.ones(40), 0.4 * np.ones(40)]
    s = rng.binomial(kappa, p_beta)
    y_beta = (s + 0.5) / (kappa + 1.0)
    m_beta = BayesBreakBeta(k_max=10, concentration=kappa).fit(y_beta)
    pc_beta = m_beta.predict()

    outdir.mkdir(parents=True, exist_ok=True)

    # Create 2x2 grid
    figsize = get_figsize("double", aspect=0.75, nrows=2, ncols=2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Panel configurations
    panels = [
        (axes[0], y_gauss, mu, pc_gauss, "Gaussian", "A", "Signal"),
        (axes[1], y_pois, lam, pc_pois, "Poisson", "B", "Count"),
        (axes[2], y_binom / n_trials, p, pc_binom, "Binomial", "C", "Proportion"),
        (axes[3], y_beta, p_beta, pc_beta, "Beta-valued", "D", "Proportion"),
    ]

    for ax, y_data, true_signal, fit, title, label, ylabel in panels:
        x = np.arange(len(y_data))

        # Scatter plot for observations
        ax.scatter(
            x,
            y_data,
            s=20,
            alpha=0.5,
            color=COLORS["grey"],
            edgecolors="none",
            zorder=1,
        )
        # True signal (dashed black)
        ax.plot(
            true_signal,
            lw=2,
            linestyle="--",
            color=COLORS["black"],
            zorder=2,
        )
        # BayesBreak fit (solid blue)
        ax.plot(
            fit,
            lw=2.5,
            color=COLORS["blue"],
            zorder=3,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, len(y_data))
        add_panel_label(ax, label)

    # Set common x-label for bottom row
    axes[2].set_xlabel("Time index")
    axes[3].set_xlabel("Time index")

    # Create custom legend at bottom
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS["grey"],
            markersize=6,
            alpha=0.7,
            label="Observations",
            linestyle="None",
        ),
        Line2D([0], [0], color=COLORS["black"], linestyle="--", lw=2, label="True signal"),
        Line2D([0], [0], color=COLORS["blue"], lw=2.5, label="BayesBreak fit"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    # Save in multiple formats
    save_figure(fig, outdir / "fig2_family_showcase", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.outdir, args.seed)
