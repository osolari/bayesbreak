r"""Figure 10: Robustness to missing data via sample weights.

BayesBreak accepts per-observation *sample weights* that modulate the
contribution of each observation to the block evidence.  Setting a weight to
zero effectively removes that observation from the analysis, which provides a
natural mechanism for handling missing data, censored observations, or
domain-specific reliability masks.

Experiment
----------
A three-segment Gaussian sequence (:math:`n=150`, levels :math:`(0, 2, -1)`,
:math:`\sigma=0.3`) is generated.  Then a fraction ``miss_frac`` of the
observations are declared "missing" (their sample weight is set to 0) in three
different patterns:

1. **Random** — observations are removed uniformly at random.
2. **Block** — a contiguous block of ``miss_frac × n`` observations is
   removed, centred on the second changepoint to maximise disruption.
3. **Periodic** — every :math:`k`-th observation is removed (simulates
   systematic sensor drop-outs).

For each pattern, BayesBreak is fit *with* the weight mask (using non-missing
observations) and *without* the mask (i.e., fitting to only the observed
subset, discarding time indices).  The figure also shows a "full data" baseline.

The figure has 3 rows × 2 columns:

* **Column 1**: each missingness pattern visualised — grey dots for observed,
  red ×'s for missing, overlaid with the BayesBreak fit (with weighting).
* **Column 2**: corresponding boundary posteriors.  The "full data" baseline is
  shown as a filled blue curve; the weight-aware fit as a solid line; the
  naïve fit (ignoring time) as a dashed line.

Interpretation
--------------
- The weight-aware fit should produce boundary posteriors that are nearly as
  sharp as the full-data baseline, because the DP correctly accounts for the
  reduced information in segments that contain missing data.
- The naïve approach (discarding time and fitting a shorter series) loses
  boundary position information and may shift or miss changepoints, especially
  when the missing block overlaps a changepoint.
- **Block missingness** is the hardest case: if the missing block straddles a
  changepoint, neither approach has direct information, but the weighted model
  at least preserves the correct time indices.

Outputs
-------
- results/figures/fig10_missing_data.png
- results/figures/fig10_missing_data.pdf

Usage
-----
python scripts/figures/fig10_missing_data.py [--miss-frac 0.3]
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
    get_figsize,
    save_figure,
    setup_style,
)

from bayesbreak import BayesBreakGaussian  # noqa: E402


def main(outdir: Path, seed: int, n: int, miss_frac: float) -> None:
    setup_style(font_scale=0.95)
    rng = np.random.default_rng(seed)

    levels = np.array([0.0, 2.0, -1.0])
    seg_lens = [50, 50, n - 100]
    mu = np.repeat(levels, seg_lens)
    sigma = 0.3
    y_full = mu + sigma * rng.standard_normal(n)
    true_b = [50, 100]
    k_max = 10

    # Full data baseline.
    m_full = BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y_full)).reshape(-1, 1), y_full)
    d1_full = m_full.boundary_marginals_

    # Missingness masks.
    n_miss = int(miss_frac * n)

    # 1) Random
    mask_rand = np.ones(n, dtype=float)
    idx_rand = rng.choice(n, size=n_miss, replace=False)
    mask_rand[idx_rand] = 0.0

    # 2) Block (centred on second changepoint at 100)
    mask_block = np.ones(n, dtype=float)
    start = max(0, 100 - n_miss // 2)
    mask_block[start : start + n_miss] = 0.0

    # 3) Periodic
    mask_periodic = np.ones(n, dtype=float)
    step = max(1, n // n_miss)
    mask_periodic[::step] = 0.0

    masks = [
        (mask_rand, "Random missing"),
        (mask_block, "Block missing"),
        (mask_periodic, "Periodic missing"),
    ]

    outdir.mkdir(parents=True, exist_ok=True)

    figsize = get_figsize("double", aspect=0.65, nrows=3, ncols=2)
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex="col")

    panel_left = ["A", "C", "E"]
    panel_right = ["B", "D", "F"]
    x = np.arange(n)
    x_b = np.arange(1, n)

    for row, (mask, label) in enumerate(masks):
        # Fit with weights.
        m_w = BayesBreakGaussian(k_max=k_max).fit(
            np.arange(len(y_full)).reshape(-1, 1), y_full, sample_weight=mask
        )
        d1_w = m_w.boundary_marginals_
        pc_w = m_w.predict(m_w.x_design_.reshape(-1, 1))

        # --- Left column: signal + fit ---
        ax = axes[row, 0]
        obs_mask = mask > 0
        miss_mask = ~obs_mask

        ax.scatter(
            x[obs_mask],
            y_full[obs_mask],
            s=12,
            alpha=0.5,
            color=COLORS["grey"],
            edgecolors="none",
            zorder=1,
            label="Observed" if row == 0 else None,
        )
        ax.scatter(
            x[miss_mask],
            y_full[miss_mask],
            s=18,
            marker="x",
            alpha=0.5,
            color=COLORS["red"],
            zorder=1,
            label="Missing" if row == 0 else None,
        )
        ax.plot(mu, ls="--", lw=1.5, color=COLORS["black"], zorder=2)
        ax.plot(pc_w, lw=2, color=COLORS["blue"], zorder=3)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlim(0, n)
        if row == 0:
            ax.legend(loc="upper right", fontsize=10, ncol=2)
        if row == 2:
            ax.set_xlabel("Time index")
        add_panel_label(ax, panel_left[row])

        # --- Right column: boundary posteriors ---
        ax = axes[row, 1]
        ax.fill_between(
            x_b,
            0,
            d1_full,
            alpha=0.2,
            color=COLORS["blue"],
            linewidth=0,
            label="Full data" if row == 0 else None,
        )
        ax.plot(
            x_b,
            d1_w,
            lw=2,
            color=COLORS["blue"],
            label="Weighted" if row == 0 else None,
        )
        for tb in true_b:
            ax.axvline(tb, color=COLORS["red"], ls="--", lw=1.2, alpha=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, n)
        if row == 0:
            ax.legend(loc="upper right", fontsize=10)
        if row == 2:
            ax.set_xlabel("Time index")
        ax.set_ylabel("Boundary prob.")
        add_panel_label(ax, panel_right[row])

    save_figure(fig, outdir / "fig10_missing_data", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--miss-frac", type=float, default=0.25)
    args = ap.parse_args()
    main(outdir=args.outdir, seed=args.seed, n=args.n, miss_frac=args.miss_frac)
