r"""Figure 7: Signal recovery as a function of signal-to-noise ratio (SNR).

A key practical question when applying BayesBreak is: *how noisy can the
data be before changepoints become undetectable?* This script sweeps the
noise standard deviation :math:`\sigma` while keeping the step heights
fixed, thereby varying the signal-to-noise ratio

.. math::
    \text{SNR} = \frac{\Delta\mu_{\min}}{\sigma},

where :math:`\Delta\mu_{\min}` is the size of the smallest step in the
latent signal.

Experiment
----------
A three-segment Gaussian sequence (:math:`n=150`) with levels
:math:`(0,\; 1.5,\; -0.5)` is generated for each of 14 :math:`\sigma`
values logarithmically spaced from 0.05 (high SNR) to 3.0 (low SNR).
For each :math:`\sigma`, ``n_rep`` (default 40) random repetitions are
run. For every repetition we record:

* **Boundary F1@tau** -- changepoint detection accuracy.
* **Signal MSE** -- :math:`\frac{1}{n}\sum_i (\hat\mu_i - \mu_i)^2`.
* **Selected k** -- number of segments chosen by the model.

The figure has three panels with consistent x-axis (noise :math:`\sigma`,
log scale):

* **Panel A** -- Boundary F1 with IQR band. The *critical noise*
  :math:`\sigma_{c}` where the median F1 drops below 0.5 is annotated.
* **Panel B** -- Selected :math:`\hat{k}` as a 2D histogram (one column per
  :math:`\sigma`, normalised by column). The horizontal dashed line marks
  :math:`k^{\star}=3`. This panel replaces a redundant MSE-vs-sigma curve
  with the more informative *full empirical distribution* of selected
  segment counts.
* **Panel C** -- Signal MSE (log y-axis) with IQR band, plus a dashed
  reference :math:`\propto \sigma^{2}` line that an oracle would achieve.

Outputs
-------
- docs/report/figures/fig7_snr_sensitivity.png
- docs/report/figures/fig7_snr_sensitivity.pdf
- docs/report/figures/fig7_snr_sensitivity.csv

Usage
-----
python scripts/figures/supplementary/figA2_snr_sensitivity.py [--n-rep 50]
"""

from __future__ import annotations

import argparse
import logging
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

logger = logging.getLogger(__name__)


def _boundary_f1(true_b: list[int], pred_b: list[int], tau: int) -> float:
    """Boundary F1 within tolerance ``tau``."""
    true = sorted(true_b)
    pred = sorted(pred_b)
    matched: set[int] = set()
    tp = 0
    for p in pred:
        cands = [t for t in true if t not in matched and abs(p - t) <= tau]
        if cands:
            matched.add(min(cands, key=lambda t: abs(p - t)))
            tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _critical_sigma(sigmas: np.ndarray, f1_med: np.ndarray, threshold: float = 0.5) -> float | None:
    """Linear interpolation on log(sigma) for the sigma where median F1 = threshold."""
    f1 = np.asarray(f1_med)
    s = np.asarray(sigmas)
    for i in range(1, len(s)):
        if f1[i - 1] >= threshold and f1[i] < threshold:
            x0, x1 = np.log(s[i - 1]), np.log(s[i])
            y0, y1 = f1[i - 1], f1[i]
            if y1 == y0:
                return float(s[i])
            t = (threshold - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0)))
    return None


def main(outdir: Path, seed: int, n: int, n_rep: int, k_max: int, tau: int) -> None:
    setup_style(font_scale=1.05)
    rng = np.random.default_rng(seed)

    levels = np.array([0.0, 1.5, -0.5])
    seg_lens = [n // 3, n // 3, n - 2 * (n // 3)]
    mu = np.repeat(levels, seg_lens)
    true_b = [seg_lens[0], seg_lens[0] + seg_lens[1]]
    delta_min = float(np.min(np.abs(np.diff(levels))))

    sigmas = np.logspace(np.log10(0.05), np.log10(3.0), 14)

    f1_all: list[np.ndarray] = []
    mse_all: list[np.ndarray] = []
    k_all: list[np.ndarray] = []

    for sigma in sigmas:
        f1s, mses, ks = [], [], []
        for _ in range(n_rep):
            y = mu + sigma * rng.standard_normal(n)
            m = BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y)).reshape(-1, 1), y)
            pred_b = m.map_boundaries_[1:-1]
            f1s.append(_boundary_f1(true_b, pred_b, tau=tau))
            mses.append(float(np.mean((m.predict(m.x_design_.reshape(-1, 1)) - mu) ** 2)))
            ks.append(int(m.k_map_))

        f1_all.append(np.asarray(f1s))
        mse_all.append(np.asarray(mses))
        k_all.append(np.asarray(ks, dtype=int))
        logger.info(
            "sigma=%.3f  F1=%.2f  MSE=%.3g  k_med=%.0f",
            sigma,
            float(np.median(f1_all[-1])),
            float(np.median(mse_all[-1])),
            float(np.median(k_all[-1])),
        )

    f1_med = np.array([float(np.median(a)) for a in f1_all])
    f1_q1 = np.array([float(np.percentile(a, 25)) for a in f1_all])
    f1_q3 = np.array([float(np.percentile(a, 75)) for a in f1_all])

    mse_med = np.array([float(np.median(a)) for a in mse_all])
    mse_q1 = np.array([float(np.percentile(a, 25)) for a in mse_all])
    mse_q3 = np.array([float(np.percentile(a, 75)) for a in mse_all])

    sigma_c = _critical_sigma(sigmas, f1_med, threshold=0.5)

    # --- 2D histogram for selected k ---
    k_bins = np.arange(0.5, k_max + 1.5, 1.0)
    H = np.zeros((len(k_bins) - 1, len(sigmas)), dtype=float)
    for j, ks in enumerate(k_all):
        H[:, j], _ = np.histogram(ks, bins=k_bins)
    col_sums = H.sum(axis=0, keepdims=True)
    H_norm = H / np.maximum(col_sums, 1)

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "fig7_snr_sensitivity.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("sigma,snr,f1_med,f1_q1,f1_q3,mse_med,mse_q1,mse_q3,k_med\n")
        for j, s in enumerate(sigmas):
            f.write(
                f"{s:.6f},{delta_min / s:.6f},"
                f"{f1_med[j]:.4f},{f1_q1[j]:.4f},{f1_q3[j]:.4f},"
                f"{mse_med[j]:.6f},{mse_q1[j]:.6f},{mse_q3[j]:.6f},"
                f"{np.median(k_all[j]):.2f}\n"
            )
    logger.info("Wrote %s", csv_path)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    # ---- Panel A: F1 with IQR band + critical-sigma annotation ----
    axA = axes[0]
    axA.fill_between(sigmas, f1_q1, f1_q3, alpha=0.25, color=COLORS["blue"], linewidth=0)
    axA.plot(sigmas, f1_med, "o-", color=COLORS["blue"], markersize=5, lw=2.0)
    axA.axhline(1.0, ls=":", color=COLORS["grey"], lw=1)
    axA.axhline(0.5, ls=":", color=COLORS["grey"], lw=1)
    if sigma_c is not None:
        axA.axvline(sigma_c, ls="--", color=COLORS["red"], lw=1.5, alpha=0.85)
        axA.annotate(
            rf"$\sigma_{{c}} \approx {sigma_c:.2f}$"
            + "\n"
            + rf"(SNR$\approx${delta_min / sigma_c:.1f})",
            xy=(sigma_c, 0.5),
            xytext=(sigma_c * 1.8, 0.72),
            fontsize=10,
            color=COLORS["red"],
            arrowprops={"arrowstyle": "->", "color": COLORS["red"], "lw": 1.2},
        )
    axA.set_xscale("log")
    axA.set_xlabel(r"Noise $\sigma$")
    axA.set_ylabel(r"Boundary F1@$\tau$")
    axA.set_ylim(-0.04, 1.08)
    axA.grid(True, which="major", ls="-", alpha=0.20, color=COLORS["grey"])
    add_panel_label(axA, "A", title="  Detection vs. noise")

    # ---- Panel B: 2D histogram of selected k ----
    axB = axes[1]
    extent = (np.log10(sigmas[0]), np.log10(sigmas[-1]), 0.5, k_max + 0.5)
    im = axB.imshow(
        H_norm,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
    )
    axB.axhline(3.0, ls="--", color="#FFFFFF", lw=1.6, alpha=0.95)
    axB.text(
        np.log10(sigmas[0]) + 0.05,
        3.0,
        r"$k^{\star}=3$",
        color="#FFFFFF",
        fontsize=10,
        fontweight="medium",
        va="bottom",
    )
    tick_sigmas = np.array([0.05, 0.1, 0.3, 1.0, 3.0])
    axB.set_xticks(np.log10(tick_sigmas))
    axB.set_xticklabels([f"{s:g}" for s in tick_sigmas])
    axB.set_yticks(range(1, k_max + 1))
    axB.set_xlabel(r"Noise $\sigma$")
    axB.set_ylabel(r"Selected $\hat{k}$")
    cb = fig.colorbar(im, ax=axB, fraction=0.06, pad=0.04)
    cb.set_label(r"Fraction (per $\sigma$)")
    cb.set_ticks([0.0, 0.5, 1.0])
    add_panel_label(axB, "B", title=r"  Distribution of $\hat{k}$")

    # ---- Panel C: signal MSE with sigma^2 reference ----
    axC = axes[2]
    axC.fill_between(sigmas, mse_q1, mse_q3, alpha=0.25, color=COLORS["red"], linewidth=0)
    axC.plot(sigmas, mse_med, "s-", color=COLORS["red"], markersize=5, lw=2.0, label="BayesBreak")
    avg_len = float(np.mean(seg_lens))
    ref = (sigmas**2) / avg_len
    axC.plot(
        sigmas,
        ref,
        ls="--",
        lw=1.5,
        color=COLORS["black"],
        alpha=0.6,
        label=r"oracle $\sigma^{2}/\bar{m}$",
    )
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"Noise $\sigma$")
    axC.set_ylabel("Signal MSE")
    axC.grid(True, which="major", ls="-", alpha=0.20, color=COLORS["grey"])
    axC.legend(loc="lower right", fontsize=9)
    add_panel_label(axC, "C", title="  Reconstruction error")

    save_figure(fig, outdir / "fig7_snr_sensitivity", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-rep", type=int, default=40)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--tau", type=int, default=2)
    args = ap.parse_args()
    main(
        outdir=args.outdir,
        seed=args.seed,
        n=args.n,
        n_rep=args.n_rep,
        k_max=args.k_max,
        tau=args.tau,
    )
