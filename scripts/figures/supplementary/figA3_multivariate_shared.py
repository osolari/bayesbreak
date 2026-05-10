r"""Figure 8: Multivariate shared-boundary segmentation.

BayesBreak can segment *vector-valued* time series by assuming conditional
independence across channels given the latent segment parameters, while
enforcing **shared changepoint locations** across all channels.  The joint
block evidence is the *sum* of per-channel log evidences:

.. math::
    \log \mathcal{L}_{ij}^{\text{joint}} =
      \sum_{c=1}^{d} \log \mathcal{L}_{ij}^{(c)}

This allows weak per-channel signals to reinforce each other, producing more
reliable boundary detection than fitting each channel independently.

Experiment
----------
A :math:`d=3`-channel Gaussian time series (:math:`n=200`) is generated.  All
three channels share the **same changepoint positions** but have
*different segment-level means and noise levels*.  Specifically:

- **Channel 1**: levels :math:`(0, 2, -1)`, :math:`\sigma_1 = 0.6`.
- **Channel 2**: levels :math:`(1, -1, 0.5)`, :math:`\sigma_2 = 0.8`.
- **Channel 3**: levels :math:`(−0.5, 0.5, 1.5)`, :math:`\sigma_3 = 1.0`
  (noisiest channel — hard to segment alone).

The true changepoints are at positions 60 and 140.

The figure contains four rows:

* **Rows 1–3**: each channel's observations (grey dots), true signal (dashed
  black), and BayesBreak fit (blue = shared, red = independent).
* **Row 4**: marginal boundary posteriors for the *shared* model (blue) and
  *independent* per-channel models (thin grey lines).  True boundaries are
  shown as dashed vertical lines.

Interpretation
--------------
- The shared model's boundary posterior (blue, bottom row) should show
  **sharper and taller** peaks at the true changepoints than any single
  independent channel (grey lines).  This demonstrates the *statistical
  borrowing of strength* across channels.
- For Channel 3 (the noisiest), the independent fit may fail to detect one or
  both changepoints, while the shared fit succeeds because Channels 1 and 2
  provide supporting evidence.
- If a changepoint is present in only a subset of channels, the shared model
  may produce a weaker peak — this is by design, since the shared assumption
  would be violated.

Outputs
-------
- docs/report/figures/fig8_multivariate_shared.png
- docs/report/figures/fig8_multivariate_shared.pdf

Usage
-----
python scripts/figures/fig8_multivariate_shared.py
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

from bayesbreak import BayesBreakGaussian, SharedBoundaryMultivariateSegmenter  # noqa: E402


def main(outdir: Path, seed: int) -> None:
    setup_style(font_scale=1.0)
    rng = np.random.default_rng(seed)

    n = 200
    true_boundaries = [0, 60, 140, n]
    k_max = 10

    # Three channels, shared changepoints, different levels and noise.
    channel_levels = [
        [0.0, 2.0, -1.0],
        [1.0, -1.0, 0.5],
        [-0.5, 0.5, 1.5],
    ]
    channel_sigmas = [0.6, 0.8, 1.0]
    d = len(channel_levels)

    Y = np.empty((n, d))
    mu_true = np.empty((n, d))
    for c in range(d):
        for seg_idx, (a, b) in enumerate(
            zip(true_boundaries[:-1], true_boundaries[1:], strict=False)
        ):
            mu_true[a:b, c] = channel_levels[c][seg_idx]
        Y[:, c] = mu_true[:, c] + channel_sigmas[c] * rng.standard_normal(n)

    # Fit shared-boundary model.
    base = BayesBreakGaussian(k_max=k_max)
    mv_shared = SharedBoundaryMultivariateSegmenter(base_estimator=base, k_max=k_max).fit(
        np.arange(len(Y)).reshape(-1, 1), Y
    )

    # Fit independent per-channel models.
    from bayesbreak import IndependentMultivariateSegmenter

    mv_indep = IndependentMultivariateSegmenter(base_estimator=base, k_max=k_max).fit(
        np.arange(len(Y)).reshape(-1, 1), Y
    )

    # ---- Plot ----
    outdir.mkdir(parents=True, exist_ok=True)

    # 4-row layout: 3 channel rows + 1 boundary posterior row
    # Use gridspec for unequal height ratios (boundary panel slightly taller)
    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[1, 1, 1, 1.3],
        hspace=0.08,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    # Share x-axis across all panels
    for ax in axes[:-1]:
        ax.sharex(axes[-1])
        plt.setp(ax.get_xticklabels(), visible=False)

    panel_labels = ["A", "B", "C", "D"]
    t = np.arange(n)
    ch_labels = [
        r"Channel 1 ($\sigma$=0.6)",
        r"Channel 2 ($\sigma$=0.8)",
        r"Channel 3 ($\sigma$=1.0)",
    ]

    for c in range(d):
        ax = axes[c]
        ax.scatter(t, Y[:, c], s=6, alpha=0.3, color=COLORS["grey"], edgecolors="none", zorder=1)
        ax.plot(mu_true[:, c], lw=1.5, ls="--", color=COLORS["black"], zorder=2, label="True")
        # Shared fit
        ax.plot(mv_shared.map_curve_[:, c], lw=2, color=COLORS["blue"], label="Shared", zorder=3)
        # Independent fit
        ax.plot(
            mv_indep.map_curve_[:, c],
            lw=1.5,
            color=COLORS["red"],
            ls=":",
            label="Independent",
            zorder=3,
        )
        ax.set_ylabel(ch_labels[c], fontsize=10)
        ax.set_xlim(0, n)
        # Mark true boundaries lightly
        for tb in true_boundaries[1:-1]:
            ax.axvline(tb, color=COLORS["grey"], ls=":", lw=0.8, alpha=0.5)
        if c == 0:
            ax.legend(loc="upper right", fontsize=10, ncol=3)
        add_panel_label(ax, panel_labels[c])

    # Bottom panel: boundary posteriors
    ax = axes[3]
    x_b = np.arange(1, n)
    ax.fill_between(
        x_b,
        0,
        mv_shared.boundary_marginals_,
        alpha=0.35,
        color=COLORS["blue"],
        linewidth=0,
    )
    ax.plot(x_b, mv_shared.boundary_marginals_, lw=2, color=COLORS["blue"], label="Shared")

    # Independent channel boundary posteriors
    ch_colors = [COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for c in range(d):
        est_c = mv_indep.channel_estimators_[c]
        d1_c = est_c.boundary_marginals_
        ax.plot(x_b, d1_c, lw=1.2, color=ch_colors[c], alpha=0.7, label=f"Ch {c + 1} indep.")

    for tb in true_boundaries[1:-1]:
        ax.axvline(
            tb,
            color=COLORS["red"],
            ls="--",
            lw=1.5,
            label="True" if tb == true_boundaries[1] else None,
        )

    ax.set_xlabel("Time index")
    ax.set_ylabel("Boundary probability")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, n)
    ax.legend(loc="upper right", fontsize=10, ncol=3)
    add_panel_label(ax, panel_labels[3])

    save_figure(fig, outdir / "fig8_multivariate_shared", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(outdir=args.outdir, seed=args.seed)
