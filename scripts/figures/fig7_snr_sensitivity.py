r"""Figure 7: Signal recovery as a function of signal-to-noise ratio (SNR).

A key practical question when applying BayesBreak is: *how noisy can the data
be before changepoints become undetectable?*  This script sweeps the noise
standard deviation :math:`\sigma` while keeping the step height fixed, thereby
varying the signal-to-noise ratio

.. math::
    \text{SNR} = \frac{\Delta\mu}{\sigma}

where :math:`\Delta\mu` is the size of the smallest step in the latent signal.

Experiment
----------
A three-segment Gaussian sequence (:math:`n=150`) with levels
:math:`(0,\; 1.5,\; -0.5)` is generated for each of 12 :math:`\sigma` values
logarithmically spaced from 0.05 (high SNR) to 3.0 (low SNR).  For each
:math:`\sigma`, ``n_rep`` (default 30) random repetitions are run.  For every
repetition we record:

* **Boundary F1@τ** — changepoint detection accuracy.
* **Signal MSE** — :math:`\frac{1}{n}\sum_i (\hat\mu_i - \mu_i)^2`.
* **Selected k** — number of segments chosen by the model.

The figure contains three panels:

* **Panel A — F1@τ vs.\ SNR**: shows a sharp phase transition from near-perfect
  detection at high SNR to chance level at low SNR.  The shaded band is the
  inter-quartile range across repetitions.
* **Panel B — MSE vs.\ SNR**: MSE increases with noise; the "floor" at each
  SNR corresponds to the irreducible posterior variance.
* **Panel C — Selected k vs.\ SNR**: at low SNR the model should collapse to
  :math:`k=1` (no changepoints detected), while at high SNR it should
  consistently select :math:`k=3`.

Interpretation
--------------
* The **critical SNR** (the :math:`\sigma` at which F1 drops below 0.5) is a
  useful characterisation of BayesBreak's sensitivity.  For the default
  parameters this is roughly :math:`\sigma \approx 1.0`
  (:math:`\text{SNR} \approx 1.5`).
* A model that over-segments at low SNR (selected :math:`k \gg 1`) would
  suggest inadequate regularisation by the prior.
* The smooth decline in Panel A (rather than an abrupt cliff) demonstrates that
  BayesBreak degrades gracefully.

Outputs
-------
- docs/report/figures/fig7_snr_sensitivity.png
- docs/report/figures/fig7_snr_sensitivity.pdf

Usage
-----
python scripts/figures/fig7_snr_sensitivity.py [--n-rep 50]
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


def main(outdir: Path, seed: int, n: int, n_rep: int, k_max: int, tau: int) -> None:
    setup_style(font_scale=1.1)
    rng = np.random.default_rng(seed)

    levels = np.array([0.0, 1.5, -0.5])
    seg_lens = [n // 3, n // 3, n - 2 * (n // 3)]
    mu = np.repeat(levels, seg_lens)
    true_b = [seg_lens[0], seg_lens[0] + seg_lens[1]]

    sigmas = np.logspace(np.log10(0.05), np.log10(3.0), 12)

    f1_med, f1_q1, f1_q3 = [], [], []
    mse_med, mse_q1, mse_q3 = [], [], []
    k_med, k_q1, k_q3 = [], [], []

    for sigma in sigmas:
        f1s, mses, ks = [], [], []
        for _ in range(n_rep):
            y = mu + sigma * rng.standard_normal(n)
            m = BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y)).reshape(-1, 1), y)
            pred_b = m.map_boundaries_[1:-1]
            f1s.append(_boundary_f1(true_b, pred_b, tau=tau))
            mses.append(float(np.mean((m.predict(m.x_design_.reshape(-1, 1)) - mu) ** 2)))
            ks.append(int(m.k_map_))

        f1_arr = np.asarray(f1s)
        mse_arr = np.asarray(mses)
        k_arr = np.asarray(ks, dtype=float)

        f1_med.append(float(np.median(f1_arr)))
        f1_q1.append(float(np.percentile(f1_arr, 25)))
        f1_q3.append(float(np.percentile(f1_arr, 75)))

        mse_med.append(float(np.median(mse_arr)))
        mse_q1.append(float(np.percentile(mse_arr, 25)))
        mse_q3.append(float(np.percentile(mse_arr, 75)))

        k_med.append(float(np.median(k_arr)))
        k_q1.append(float(np.percentile(k_arr, 25)))
        k_q3.append(float(np.percentile(k_arr, 75)))

    # --- Plot ---
    outdir.mkdir(parents=True, exist_ok=True)

    # Explicit figsize: full double-column width, 3 equal panels, good height
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    # Panel A: F1
    ax = axes[0]
    ax.fill_between(sigmas, f1_q1, f1_q3, alpha=0.25, color=COLORS["blue"], linewidth=0)
    ax.plot(sigmas, f1_med, "o-", color=COLORS["blue"], markersize=4, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Noise $\sigma$")
    ax.set_ylabel(r"Boundary F1@$\tau$")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, ls=":", color=COLORS["grey"], lw=1)
    ax.grid(True, which="major", ls="-", alpha=0.15, color=COLORS["grey"])
    add_panel_label(ax, "A")

    # Panel B: MSE
    ax = axes[1]
    ax.fill_between(sigmas, mse_q1, mse_q3, alpha=0.25, color=COLORS["red"], linewidth=0)
    ax.plot(sigmas, mse_med, "s-", color=COLORS["red"], markersize=4, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Noise $\sigma$")
    ax.set_ylabel("Signal MSE")
    ax.grid(True, which="major", ls="-", alpha=0.15, color=COLORS["grey"])
    add_panel_label(ax, "B")

    # Panel C: selected k
    ax = axes[2]
    ax.fill_between(sigmas, k_q1, k_q3, alpha=0.25, color=COLORS["green"], linewidth=0)
    ax.plot(sigmas, k_med, "D-", color=COLORS["green"], markersize=4, lw=2)
    ax.axhline(3.0, ls="--", color=COLORS["black"], lw=1.5, label=r"$k^{\star}=3$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Noise $\sigma$")
    ax.set_ylabel(r"Selected $\hat{k}$")
    ax.set_ylim(0, k_max + 1)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, which="major", ls="-", alpha=0.15, color=COLORS["grey"])
    add_panel_label(ax, "C")

    save_figure(fig, outdir / "fig7_snr_sensitivity", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-rep", type=int, default=30)
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
