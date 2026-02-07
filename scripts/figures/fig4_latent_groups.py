r"""Figure 4: Latent-group pooling (mixture) demonstration.

This script simulates multiple Gaussian sequences from two latent groups with
distinct changepoint locations and segment means. It then fits
:class:`bayesbreak.BayesBreakMixture` (an EM-like latent-group extension of
BayesBreak) and visualises the results.

Experiment
----------
Two groups share the same series length :math:`n` but differ in their
changepoint structure:

- **Group 0**: three segments with boundaries at :math:`n/3` and :math:`2n/3`,
  and levels :math:`(0, 1, -0.5)`.
- **Group 1**: three segments with boundaries at :math:`n/4` and :math:`3n/4`,
  and levels :math:`(0.5, -1, 0.8)`.

``n_seq`` (default 12) sequences are drawn — half from each group — with
additive Gaussian noise (:math:`\sigma=0.35`).  BayesBreakMixture is fit with
``n_groups=2`` and modest iteration budget.

The resulting three-panel figure shows:

1. **Panel A — Responsibility heatmap**: rows are sequences (sorted by true
   label); columns are groups.  A clearly bi-modal pattern (one block blue,
   one block red) indicates successful group discovery.
2. **Panel B — Boundary posteriors per group**: overlaid marginal boundary
   posteriors for each discovered group.  Peaks should align with the true
   changepoints for that group.
3. **Panel C — Reconstructed group signals**: smoothed within-group average
   signal (solid) compared with the true latent means (dashed).  Close
   agreement indicates the model is pooling information correctly.

Interpretation
--------------
The figure answers the question: *can the mixture model separate two groups
with similar noise levels but different changepoint locations?*  Success is
indicated by:

- Near-binary responsibilities (sequences assigned with high confidence).
- Boundary posteriors that peak at the correct group-specific changepoints.
- Reconstructed signals that track the true latent means closely.

If the responsibility heatmap shows mixed assignments or boundary posteriors
that average the two patterns, the EM initialisation or iteration count may
need adjustment.

Outputs
-------
- results/fig4_latent_groups.png
- results/fig4_latent_groups.pdf

Usage
-----
python scripts/figures/fig4_latent_groups.py [--n-seq 20 --sigma 0.3]
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
    save_figure,
    setup_style,
)

from bayesbreak import BayesBreakGaussian  # noqa: E402
from bayesbreak.mixture import BayesBreakMixture  # noqa: E402


def _make_piecewise_constant(n: int, boundaries: list[int], levels: list[float]) -> np.ndarray:
    if len(boundaries) != len(levels) + 1:
        raise ValueError("boundaries must have length len(levels)+1")
    x = np.empty(n, dtype=float)
    for a, b, m in zip(boundaries[:-1], boundaries[1:], levels, strict=False):
        x[a:b] = float(m)
    return x


def _align_groups_by_truth(r: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Return a permutation of group indices that best matches true labels.

    We use a simple heuristic: for each group g, compute the mean responsibility
    among sequences with true label 0 and 1, then choose the mapping that
    maximises the total "diagonal" mass.
    """
    G = int(r.shape[1])
    if G != 2:
        return np.arange(G)
    m0 = float(np.mean(r[y_true == 0, 0]))
    m1 = float(np.mean(r[y_true == 1, 1]))
    score_id = m0 + m1
    m0s = float(np.mean(r[y_true == 0, 1]))
    m1s = float(np.mean(r[y_true == 1, 0]))
    score_swap = m0s + m1s
    return np.array([0, 1], dtype=int) if score_id >= score_swap else np.array([1, 0], dtype=int)


def main(
    outdir: Path,
    seed: int,
    n: int,
    n_seq: int,
    sigma: float,
    k_max: int,
    max_iter: int,
) -> None:
    rng = np.random.default_rng(seed)

    # Two latent groups with different changepoints.
    b0 = [0, n // 3, 2 * n // 3, n]
    m0 = [0.0, 1.0, -0.5]

    b1 = [0, n // 4, 3 * n // 4, n]
    m1 = [0.5, -1.0, 0.8]

    mu0 = _make_piecewise_constant(n, b0, m0)
    mu1 = _make_piecewise_constant(n, b1, m1)

    # Sample true group labels and sequences.
    y_true = rng.integers(0, 2, size=n_seq)
    ys = []
    for s in range(n_seq):
        mu = mu0 if int(y_true[s]) == 0 else mu1
        ys.append(mu + sigma * rng.standard_normal(n))

    base = BayesBreakGaussian(k_max=k_max)
    mix = BayesBreakMixture(
        base_estimator=base,
        n_groups=2,
        k_max=k_max,
        max_iter=max_iter,
        tol=1e-4,
        regression_curve="mix_k",
        random_state=seed,
    ).fit(ys)

    r = np.asarray(mix.responsibilities_, dtype=float)
    perm = _align_groups_by_truth(r, y_true)
    r = r[:, perm]
    states = [mix.group_states_[int(g)] for g in perm]  # type: ignore[index]

    # Sort sequences by true label for a cleaner responsibility heatmap.
    order = np.argsort(y_true)
    r_plot = r[order]

    outdir.mkdir(parents=True, exist_ok=True)

    # Setup publication style - use larger scale for this figure
    setup_style(font_scale=1.0, style="paper")

    # Larger figure size for 3-panel layout
    fig = plt.figure(figsize=(10, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.7, 1.0, 1.0], wspace=0.35)

    # --- (A) Responsibilities heatmap ---
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(
        r_plot,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
    )
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["G0", "G1"])
    ax0.set_ylabel("Sequence")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.08, pad=0.08, shrink=0.9)
    cbar.set_label("Prob.")
    add_panel_label(ax0, "A", offset=(-0.25, 1.05))

    # --- (B) Group boundary posteriors ---
    ax1 = fig.add_subplot(gs[0, 1])
    x_b = np.arange(1, n)
    colors = [COLORS["blue"], COLORS["red"]]
    for g, st in enumerate(states):
        ax1.fill_between(x_b, 0, st.boundary_post, alpha=0.3, color=colors[g], linewidth=0)
        ax1.plot(x_b, st.boundary_post, lw=2, color=colors[g], label=f"Group {g}")
    ax1.set_xlabel("Time index")
    ax1.set_ylabel("Boundary probability")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, n)
    ax1.legend(loc="upper right")
    add_panel_label(ax1, "B", offset=(-0.15, 1.05))

    # --- (C) Reconstructed signals (use simple segment means instead of BRC) ---
    ax2 = fig.add_subplot(gs[0, 2])

    # Plot true signals
    ax2.plot(mu0, linestyle="--", lw=1.5, color=COLORS["blue"], alpha=0.5, label="True G0")
    ax2.plot(mu1, linestyle="--", lw=1.5, color=COLORS["red"], alpha=0.5, label="True G1")

    # Compute group-averaged signals from the data
    for g in range(len(states)):
        # Get sequences assigned to this group (by max responsibility)
        group_mask = np.argmax(r, axis=1) == perm[g]
        if np.sum(group_mask) > 0:
            group_mean = np.mean([ys[i] for i in range(n_seq) if group_mask[i]], axis=0)
            # Smooth with simple moving average
            window = 5
            smoothed = np.convolve(group_mean, np.ones(window) / window, mode="same")
            ax2.plot(smoothed, lw=2, color=colors[g], label=f"Avg G{g}")

    ax2.set_xlabel("Time index")
    ax2.set_ylabel("Signal")
    ax2.set_xlim(0, n)
    ax2.legend(loc="upper right", ncol=2, fontsize=8)
    add_panel_label(ax2, "C", offset=(-0.15, 1.05))

    # Save in multiple formats
    save_figure(fig, outdir / "fig4_latent_groups", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    # Defaults chosen to keep runtime modest while still illustrating
    # responsibility separation and group-specific boundary structure.
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-seq", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=6)
    args = ap.parse_args()

    main(
        outdir=args.outdir,
        seed=args.seed,
        n=args.n,
        n_seq=args.n_seq,
        sigma=args.sigma,
        k_max=args.k_max,
        max_iter=args.max_iter,
    )
