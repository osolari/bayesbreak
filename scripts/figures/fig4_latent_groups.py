r"""Figure 4: Latent-group pooling (mixture) demonstration.

The figure illustrates that :class:`bayesbreak.BayesBreakMixtureClassifier`
recovers two latent populations whose changepoint structure differs even
though their observation noise is identical.

Layout
------
The figure is laid out as a 2x3 grid via a ``GridSpec``:

* **Row 1** -- per-group example sequences.

  * **Panel A** -- a single sequence from Group 0 with the true piecewise
    mean (dashed) and the BayesBreak posterior mean conditional on the
    discovered Group-0 boundaries (solid).
  * **Panel B** -- the same for Group 1.
  * **Panel C** -- per-group marginal boundary probabilities,
    :math:`p(b_i = 1 \mid \text{group } g)`, overlaid with the true
    group-specific changepoints as dashed verticals.

* **Row 2** -- assignment quality.

  * **Panel D** -- a wider responsibility heatmap with the true label as a
    coloured stripe down the side, making mis-assignments instantly
    visible.
  * **Panel E** -- per-group histogram of the maximum responsibility
    :math:`\max_g r_{sg}` per sequence. A right-skewed distribution
    indicates confident assignments; a peak near 0.5 indicates ambiguity.
  * **Panel F** -- group-averaged signals (median across assigned sequences
    with a 25-75 percentile band) compared with the true latent means.

The layout is designed so that the *why-it-works* story reads left-to-right
across each row: row 1 shows that boundary structure is recovered per group,
row 2 shows that sequence-to-group assignment is confident and that the
group-averaged signal closely tracks ground truth.

Outputs
-------
- results/figures/fig4_latent_groups.png
- results/figures/fig4_latent_groups.pdf
- results/figures/fig4_latent_groups_cropped.png  (2-panel version
  used in the running text of section 6)

Usage
-----
python scripts/figures/fig4_latent_groups.py [--n-seq 24 --sigma 0.3]
"""

from __future__ import annotations

import argparse
import logging
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
from matplotlib.patches import Rectangle  # noqa: E402

from bayesbreak import BayesBreakGaussian  # noqa: E402
from bayesbreak.mixture import BayesBreakMixtureClassifier  # noqa: E402

logger = logging.getLogger(__name__)


def _make_piecewise_constant(n: int, boundaries: list[int], levels: list[float]) -> np.ndarray:
    if len(boundaries) != len(levels) + 1:
        raise ValueError("boundaries must have length len(levels)+1")
    x = np.empty(n, dtype=float)
    for a, b, m in zip(boundaries[:-1], boundaries[1:], levels, strict=False):
        x[a:b] = float(m)
    return x


def _align_groups_by_truth(r: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Return a permutation of group indices that best matches true labels."""
    G = int(r.shape[1])
    if G != 2:
        return np.arange(G)
    score_id = float(np.mean(r[y_true == 0, 0])) + float(np.mean(r[y_true == 1, 1]))
    score_swap = float(np.mean(r[y_true == 0, 1])) + float(np.mean(r[y_true == 1, 0]))
    return np.array([0, 1], dtype=int) if score_id >= score_swap else np.array([1, 0], dtype=int)


def _segment_fit(y: np.ndarray, k_max: int) -> tuple[np.ndarray, list[int]]:
    """Run a single-sequence Bayesian fit; return (piecewise mean, boundaries)."""
    m = BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y)).reshape(-1, 1), y)
    return m.predict(m.x_design_.reshape(-1, 1)), list(m.map_boundaries_)


def main(
    outdir: Path,
    seed: int,
    n: int,
    n_seq: int,
    sigma: float,
    k_max: int,
    max_iter: int,
) -> None:
    setup_style(font_scale=1.0)
    rng = np.random.default_rng(seed)

    # Two latent groups with different changepoints.
    b0 = [0, n // 3, 2 * n // 3, n]
    m0 = [0.0, 1.0, -0.5]
    b1 = [0, n // 4, 3 * n // 4, n]
    m1 = [0.5, -1.0, 0.8]

    mu0 = _make_piecewise_constant(n, b0, m0)
    mu1 = _make_piecewise_constant(n, b1, m1)

    # Sample true group labels (balanced) and sequences.
    y_true = np.concatenate(
        [np.zeros(n_seq // 2, dtype=int), np.ones(n_seq - n_seq // 2, dtype=int)]
    )
    rng.shuffle(y_true)
    ys = []
    for s in range(n_seq):
        mu = mu0 if int(y_true[s]) == 0 else mu1
        ys.append(mu + sigma * rng.standard_normal(n))

    base = BayesBreakGaussian(k_max=k_max)
    mix = BayesBreakMixtureClassifier(
        base_estimator=base,
        n_groups=2,
        k_max=k_max,
        max_iter=max_iter,
        tol=1e-4,
        random_state=seed,
    ).fit(np.arange(len(ys)).reshape(-1, 1), ys)

    r = np.asarray(mix.responsibilities_, dtype=float)
    perm = _align_groups_by_truth(r, y_true)
    r = r[:, perm]
    boundary_post_per_group = [mix.get_group_boundary_marginals(int(g)) for g in perm]

    # Hard assignment by max responsibility.
    hard = np.argmax(r, axis=1)
    max_r = r.max(axis=1)

    # Sort sequences by (true label, hard assignment, max responsibility) for the heatmap.
    order = np.lexsort((-max_r, hard, y_true))
    r_plot = r[order]
    y_true_plot = y_true[order]

    # --- Pick representative example sequences (one per group) ---
    idx0 = int(np.argmax((y_true == 0) * max_r))  # most confident G0 sequence
    idx1 = int(np.argmax((y_true == 1) * max_r))  # most confident G1 sequence
    fit0, _ = _segment_fit(ys[idx0], k_max=k_max)
    fit1, _ = _segment_fit(ys[idx1], k_max=k_max)

    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Full 2x3 figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(12.0, 5.4))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.18,
    )

    g_color = {0: COLORS["blue"], 1: COLORS["red"]}

    # ---- Row 1, Panel A: example G0 sequence ----
    axA = fig.add_subplot(gs[0, 0])
    x_t = np.arange(n)
    axA.scatter(
        x_t,
        ys[idx0],
        s=14,
        alpha=0.55,
        color=COLORS["grey"],
        edgecolors="none",
        label="Observations",
        zorder=1,
    )
    axA.plot(mu0, lw=1.8, ls="--", color=COLORS["black"], label="True mean", zorder=3)
    axA.plot(fit0, lw=2.4, color=g_color[0], label="Posterior mean", zorder=4)
    axA.set_xlim(0, n)
    axA.set_xlabel("Time index")
    axA.set_ylabel("Signal")
    axA.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.85)
    add_panel_label(axA, "A", title="  Group 0 example")
    axA.title.set_color(g_color[0])

    # ---- Row 1, Panel B: example G1 sequence ----
    axB = fig.add_subplot(gs[0, 1], sharey=axA)
    axB.scatter(x_t, ys[idx1], s=14, alpha=0.55, color=COLORS["grey"], edgecolors="none", zorder=1)
    axB.plot(mu1, lw=1.8, ls="--", color=COLORS["black"], zorder=3)
    axB.plot(fit1, lw=2.4, color=g_color[1], zorder=4)
    axB.set_xlim(0, n)
    axB.set_xlabel("Time index")
    axB.tick_params(axis="y", labelleft=False)
    add_panel_label(axB, "B", title="  Group 1 example", offset=(-0.06, 1.04))
    axB.title.set_color(g_color[1])

    # ---- Row 1, Panel C: per-group boundary marginals ----
    axC = fig.add_subplot(gs[0, 2])
    x_b = np.arange(1, n)
    true_b_per_group = {0: b0[1:-1], 1: b1[1:-1]}
    for g, bm in enumerate(boundary_post_per_group):
        axC.fill_between(x_b, 0, bm, alpha=0.30, color=g_color[g], linewidth=0)
        axC.plot(x_b, bm, lw=2.0, color=g_color[g], label=f"Group {g}")
        for tb in true_b_per_group[g]:
            axC.axvline(tb, color=g_color[g], ls=":", lw=1.0, alpha=0.65)
    axC.set_ylim(0, 1.05)
    axC.set_xlim(0, n)
    axC.set_xlabel("Time index")
    axC.set_ylabel("Boundary probability")
    axC.legend(loc="center right", fontsize=9, frameon=True, framealpha=0.85)
    add_panel_label(axC, "C", title="  Boundary posterior")

    # ---- Row 2, Panel D: responsibility heatmap with true-label stripe ----
    axD = fig.add_subplot(gs[1, 0])
    # Pad with one extra column for the true-label stripe.
    cmap = plt.get_cmap("RdBu_r")
    im = axD.imshow(
        r_plot,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, 2.0, n_seq - 0.5, -0.5),
    )
    # True-label stripe on the right of the heatmap.
    for s in range(n_seq):
        axD.add_patch(
            Rectangle(
                (2.05, s - 0.5),
                0.35,
                1.0,
                facecolor=g_color[int(y_true_plot[s])],
                edgecolor="none",
                clip_on=False,
            )
        )
    axD.text(2.225, -1.0, "Truth", ha="center", va="bottom", fontsize=9)
    axD.set_xticks([0.5, 1.5])
    axD.set_xticklabels(["G0", "G1"])
    axD.set_xlim(0.0, 2.45)
    axD.set_ylabel("Sequence (by truth)")
    cb = fig.colorbar(im, ax=axD, fraction=0.06, pad=0.10, shrink=0.85)
    cb.set_label("Probability")
    cb.set_ticks([0.0, 0.5, 1.0])
    add_panel_label(axD, "D", title=r"  Responsibilities $r_{sg}$")

    # ---- Row 2, Panel E: per-sequence confidence diagnostic ----
    # Each sequence is one dot, ordered by its true responsibility r_{s, y_s}.
    # Misassigned sequences fall below the 0.5 decision threshold (red shaded).
    axE = fig.add_subplot(gs[1, 1])
    r_truth = np.array([r[s, int(y_true[s])] for s in range(n_seq)], dtype=float)
    correct = hard == y_true
    acc = float(np.mean(correct))
    order_e = np.argsort(r_truth)
    xs_e = np.arange(n_seq)
    r_sorted = r_truth[order_e]
    y_sorted = y_true[order_e]
    correct_sorted = correct[order_e]
    # Shade decision boundary region.
    axE.axhspan(0.0, 0.5, color=COLORS["red"], alpha=0.06, zorder=0)
    axE.axhline(0.5, ls="--", lw=1.1, color=COLORS["grey"], zorder=1)
    axE.text(
        0.02,
        0.52,
        "decision threshold",
        transform=axE.transAxes,
        fontsize=8,
        color=COLORS["grey"],
        va="bottom",
    )
    for g in (0, 1):
        m = y_sorted == g
        axE.scatter(
            xs_e[m],
            r_sorted[m],
            s=42,
            color=g_color[g],
            alpha=0.92,
            edgecolors="white",
            linewidths=0.8,
            label=f"True group {g}",
            zorder=3,
        )
    # Highlight any misassigned sequences with a black ring.
    miss = ~correct_sorted
    if np.any(miss):
        axE.scatter(
            xs_e[miss],
            r_sorted[miss],
            s=110,
            facecolors="none",
            edgecolors=COLORS["black"],
            linewidths=1.4,
            zorder=4,
            label="misassigned",
        )
    # Tighten y-axis to where the dots actually live (truth-responsibility ≥ 0).
    axE.set_ylim(max(0.0, r_truth.min() - 0.10), 1.04)
    axE.set_xlim(-0.5, n_seq - 0.5)
    axE.set_xlabel("Sequence (sorted by truth-responsibility)")
    axE.set_ylabel(r"$r_{s,\,y_s}$")
    axE.text(
        0.98,
        0.04,
        f"acc = {acc * 100:.0f}%",
        transform=axE.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="medium",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": COLORS["lightgrey"],
            "linewidth": 0.8,
        },
    )
    axE.legend(
        loc="lower right", fontsize=8, frameon=True, framealpha=0.9, bbox_to_anchor=(1.0, 0.18)
    )
    add_panel_label(axE, "E", title="  Assignment confidence")

    # ---- Row 2, Panel F: group-averaged reconstructed signals ----
    axF = fig.add_subplot(gs[1, 2])
    Y = np.asarray(ys, dtype=float)
    for g in (0, 1):
        mask = hard == g
        if not np.any(mask):
            continue
        Yg = Y[mask]
        med = np.median(Yg, axis=0)
        q1 = np.percentile(Yg, 25, axis=0)
        q3 = np.percentile(Yg, 75, axis=0)
        axF.fill_between(x_t, q1, q3, color=g_color[g], alpha=0.18, linewidth=0)
        axF.plot(x_t, med, lw=2.2, color=g_color[g], label=f"Group {g} median")
    axF.plot(mu0, lw=1.3, ls="--", color=g_color[0], alpha=0.85)
    axF.plot(mu1, lw=1.3, ls="--", color=g_color[1], alpha=0.85)
    axF.set_xlim(0, n)
    axF.set_xlabel("Time index")
    axF.set_ylabel("Signal")
    axF.legend(loc="lower center", fontsize=9, ncol=2, frameon=True, framealpha=0.85)
    add_panel_label(axF, "F", title="  Group-averaged signal")

    save_figure(fig, outdir / "fig4_latent_groups", formats=("png", "pdf"))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Cropped 2-panel version: responsibilities + boundary marginals.
    # Used in the inline §6 manuscript discussion.
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(8.4, 3.6))
    gs2 = fig2.add_gridspec(1, 2, width_ratios=[0.9, 1.4], wspace=0.35)

    axA2 = fig2.add_subplot(gs2[0, 0])
    im2 = axA2.imshow(
        r_plot,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=(0.0, 2.0, n_seq - 0.5, -0.5),
    )
    for s in range(n_seq):
        axA2.add_patch(
            Rectangle(
                (2.05, s - 0.5),
                0.35,
                1.0,
                facecolor=g_color[int(y_true_plot[s])],
                edgecolor="none",
                clip_on=False,
            )
        )
    axA2.text(2.225, -1.0, "Truth", ha="center", va="bottom", fontsize=9)
    axA2.set_xticks([0.5, 1.5])
    axA2.set_xticklabels(["G0", "G1"])
    axA2.set_xlim(0.0, 2.45)
    axA2.set_ylabel("Sequence")
    cb2 = fig2.colorbar(im2, ax=axA2, fraction=0.05, pad=0.15, shrink=0.85)
    cb2.set_label("Probability")
    cb2.set_ticks([0.0, 0.5, 1.0])
    add_panel_label(axA2, "A", offset=(-0.28, 1.08))

    axB2 = fig2.add_subplot(gs2[0, 1])
    for g, bm in enumerate(boundary_post_per_group):
        axB2.fill_between(x_b, 0, bm, alpha=0.30, color=g_color[g], linewidth=0)
        axB2.plot(x_b, bm, lw=2.0, color=g_color[g], label=f"Group {g}")
        for tb in true_b_per_group[g]:
            axB2.axvline(tb, color=g_color[g], ls=":", lw=1.0, alpha=0.65)
    axB2.set_ylim(0, 1.05)
    axB2.set_xlim(0, n)
    axB2.set_xlabel("Time index")
    axB2.set_ylabel("Boundary probability")
    axB2.legend(loc="upper right", fontsize=9, ncol=2)
    add_panel_label(axB2, "B", offset=(-0.14, 1.08))

    save_figure(fig2, outdir / "fig4_latent_groups_cropped", formats=("png", "pdf"))
    plt.close(fig2)

    logger.info(
        "fig4: n=%d, n_seq=%d, sigma=%.2f, k_max=%d, max_iter=%d -> %s",
        n,
        n_seq,
        sigma,
        k_max,
        max_iter,
        outdir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    ap.add_argument("--seed", type=int, default=0)
    # Larger n_seq makes the heatmap and confidence histogram more informative.
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=8)
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
