r"""Figure 9: Posterior distribution over model complexity :math:`k`.

A distinguishing feature of BayesBreak is that it computes the *full posterior*
:math:`P(k \mid y)` over the number of segments, not just a point estimate.
This figure visualises how that posterior changes as the data become more or
less informative about the true number of changepoints.

Experiment
----------
Three scenarios are compared, all using a Gaussian three-segment sequence
(:math:`n=120`, levels :math:`(0,\;1.5,\;-0.5)`):

1. **Low noise** (:math:`\sigma = 0.15`):  the data are very informative.
   :math:`P(k \mid y)` should concentrate tightly around :math:`k=3`.
2. **Medium noise** (:math:`\sigma = 0.5`):  moderate SNR.  The posterior
   should still peak at :math:`k=3` but with non-negligible mass at
   :math:`k=2` or :math:`k=4`.
3. **High noise** (:math:`\sigma = 1.5`):  the signal is nearly drowned out.
   The posterior should shift towards :math:`k=1` (no changepoints), reflecting
   the model's principled uncertainty.

For each scenario, ``n_rep`` (default 20) random repetitions are drawn and the
posterior is averaged across repetitions to reduce Monte Carlo noise.

The figure has two rows:

* **Top row (A–C)**: bar plot of :math:`P(k \mid y)` for :math:`k = 1 \ldots k_{\max}`
  at each noise level.  A vertical dashed line marks the true :math:`k=3`.
  The bar colour intensity follows the posterior mass.
* **Bottom row (D)**: overlay of the three posteriors on a single axis for
  direct comparison.  Low noise is blue (sharp peak at 3), medium is orange
  (broader), high is red (shifted left).

Interpretation
--------------
- A well-calibrated Bayesian model should produce a posterior that
  *concentrates* at the truth when data are abundant / precise and *spreads*
  when data are scarce / noisy.  This figure tests exactly that.
- If the posterior is always concentrated at :math:`k=1` even at low noise,
  the prior may be too strong (or hyperparameter estimation is too
  conservative).
- If the posterior is always spread across many values, the model evidence
  computation may be inaccurate.
- The averaged posterior smooths out single-seed flukes and gives a more
  representative view.

Outputs
-------
- docs/report/figures/fig9_model_selection.png
- docs/report/figures/fig9_model_selection.pdf

Usage
-----
python scripts/figures/fig9_model_selection.py [--n-rep 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _style import (  # noqa: E402
    COLORS,
    add_panel_label,
    save_figure,
    setup_style,
)

from bayesbreak import BayesBreakGaussian  # noqa: E402


def main(outdir: Path, seed: int, n: int, n_rep: int, k_max: int) -> None:
    setup_style(font_scale=1.0)
    rng = np.random.default_rng(seed)

    levels = np.array([0.0, 1.5, -0.5])
    seg_lens = [n // 3, n // 3, n - 2 * (n // 3)]
    mu = np.repeat(levels, seg_lens)
    k_true = 3

    scenarios = [
        (0.15, "Low noise", COLORS["blue"]),
        (0.50, "Medium noise", COLORS["orange"]),
        (1.50, "High noise", COLORS["red"]),
    ]

    avg_posteriors: list[np.ndarray] = []

    for sigma, _, _ in scenarios:
        Cs: list[np.ndarray] = []
        for _ in range(n_rep):
            y = mu + sigma * rng.standard_normal(n)
            m = BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y)).reshape(-1, 1), y)
            # Pad to k_max if C_ is shorter.
            C = np.array(m.k_posterior_, dtype=float)
            if C.size < k_max:
                C = np.pad(C, (0, k_max - C.size))
            Cs.append(C[:k_max])
        avg_posteriors.append(np.mean(Cs, axis=0))

    outdir.mkdir(parents=True, exist_ok=True)

    # 2-row layout: top = 3 individual posteriors, bottom = overlay comparison
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 1.2],
        hspace=0.45,
        wspace=0.30,
    )

    k_grid = np.arange(1, k_max + 1)

    # Top row: individual posteriors.
    panel_labels = ["A", "B", "C"]
    top_axes = []
    for idx, (sigma, label, colour) in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, idx])
        top_axes.append(ax)
        pk = avg_posteriors[idx]
        ax.bar(
            k_grid,
            pk,
            color=colour,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            width=0.7,
        )
        ax.axvline(k_true, ls="--", lw=1.5, color=COLORS["black"])
        ax.set_xlabel("$k$")
        if idx == 0:
            ax.set_ylabel("$P(k \\mid y)$")
        ax.set_title(f"{label}\n$\\sigma = {sigma}$", fontsize=10)
        ax.set_xlim(0.2, k_max + 0.8)
        # Uniform y-axis across top row for fair comparison
        ax.set_ylim(0, 1.05)
        ax.set_xticks(k_grid)
        ax.grid(True, axis="y", ls="-", alpha=0.12, color=COLORS["grey"])
        add_panel_label(ax, panel_labels[idx])

    # Bottom row (spanning all 3 columns): grouped bar overlay.
    ax = fig.add_subplot(gs[1, :])
    width = 0.22
    offsets = [-width, 0, width]
    for idx, (sigma, label, colour) in enumerate(scenarios):
        ax.bar(
            k_grid + offsets[idx],
            avg_posteriors[idx],
            width=width,
            color=colour,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
            label=f"{label} ($\\sigma$={sigma})",
        )
    ax.axvline(k_true, ls="--", lw=1.5, color=COLORS["black"], label=r"True $k^{\star}=3$")
    ax.set_xlabel("Number of segments $k$")
    ax.set_ylabel("$P(k \\mid y)$")
    ax.set_xlim(0.2, k_max + 0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(k_grid)
    ax.grid(True, axis="y", ls="-", alpha=0.12, color=COLORS["grey"])
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    add_panel_label(ax, "D")

    save_figure(fig, outdir / "fig9_model_selection", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=10)
    args = ap.parse_args()
    main(
        outdir=args.outdir,
        seed=args.seed,
        n=args.n,
        n_rep=args.n_rep,
        k_max=args.k_max,
    )
