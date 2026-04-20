"""Shared four-panel template for the real-data figures (fig6-fig9).

Layout:

A. raw y + MAP piecewise-constant fit (+ Bayes curve when enabled); MAP
   boundaries overlaid as vertical red dashes.
B. P(b_i = 1 | y) boundary marginals on the same index axis as A.
C. P(k | y) posterior over segment count (independent x-axis in k-space).
D. cumulative held-out posterior-predictive log-density vs a k=1 null
   (independent x-axis in index-space).

A and B share one x-axis (index / design space). C and D have independent
x-axes so C's k-range doesn't compress A and B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from _style import COLORS, add_panel_label, save_figure, setup_style  # noqa: E402

from bayesbreak import BayesBreakSegmenter
from bayesbreak.datasets import DatasetBundle
from bayesbreak.prediction import posterior_predictive_logpdf


def _held_out_trace(estimator: BayesBreakSegmenter, bundle: DatasetBundle) -> np.ndarray:
    per = posterior_predictive_logpdf(estimator, bundle.X, bundle.y, per_sample=True)
    assert isinstance(per, np.ndarray)
    return np.cumsum(per)


def _null_trace(estimator: BayesBreakSegmenter, bundle: DatasetBundle) -> np.ndarray:
    """Cumulative posterior-predictive under a forced one-segment null model."""

    null = estimator.__class__(**{k: v for k, v in estimator.get_params().items() if k != "k_max"})
    null.set_params(k_max=1)
    null.fit(bundle.X, bundle.y, sample_weight=bundle.sample_weight)
    per = posterior_predictive_logpdf(null, bundle.X, bundle.y, per_sample=True)
    assert isinstance(per, np.ndarray)
    return np.cumsum(per)


def make_realdata_figure(
    *,
    estimator: BayesBreakSegmenter,
    bundle: DatasetBundle,
    outdir: Path,
    fig_name: str,
    y_label: str,
    title: str,
    show_null_baseline: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Fit ``estimator`` on ``bundle`` and render the four-panel figure."""

    extra_kwargs = extra_kwargs or {}
    setup_style(font_scale=0.95)

    estimator.fit(bundle.X, bundle.y, sample_weight=bundle.sample_weight, **extra_kwargs)

    n = bundle.y.size
    # Plot everything in index space so panels A and B share a clean axis.
    idx = np.arange(n, dtype=float)

    fig = plt.figure(figsize=(7.8, 8.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.0, 1.0, 1.1], hspace=0.32)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)  # A and B only
    axC = fig.add_subplot(gs[2])
    axD = fig.add_subplot(gs[3])

    # ---- Panel A: raw y + MAP fit --------------------------------------------
    axA.scatter(
        idx,
        bundle.y,
        s=4,
        alpha=0.35,
        color=COLORS["grey"],
        edgecolors="none",
        label="observed",
        zorder=1,
    )
    axA.plot(idx, estimator.predict(bundle.X), lw=1.6, color=COLORS["blue"], label="MAP", zorder=3)
    if estimator.bayes_curve_mean_ is not None:
        axA.plot(
            idx,
            estimator.bayes_curve_mean_,
            lw=1.2,
            color=COLORS["red"],
            alpha=0.9,
            label="Bayes curve",
            zorder=2,
        )
    # MAP-boundary verticals (skip endpoints 0 and n).
    for b in estimator.map_boundaries_[1:-1]:
        axA.axvline(float(b), color=COLORS["red"], ls="--", lw=0.6, alpha=0.7, zorder=4)
    # Ground-truth boundaries (only when known, e.g. simulated bundles).
    for b in bundle.true_boundaries[1:-1]:
        axA.axvline(float(b), color=COLORS["green"], ls=":", lw=0.9, alpha=0.7, zorder=4)
    axA.set_ylabel(y_label)
    axA.set_xlim(0, n)
    # Tighten y-axis to the data range with a 5% pad.
    y_min, y_max = float(np.nanmin(bundle.y)), float(np.nanmax(bundle.y))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    axA.set_ylim(y_min - pad, y_max + pad)
    axA.legend(loc="best", ncol=3, fontsize=7, frameon=False)
    add_panel_label(axA, "A", title)
    plt.setp(axA.get_xticklabels(), visible=False)

    # ---- Panel B: boundary marginals -----------------------------------------
    if estimator.boundary_marginals_ is not None:
        xb = idx[1:]
        axB.fill_between(
            xb, 0, estimator.boundary_marginals_, color=COLORS["blue"], alpha=0.35, linewidth=0
        )
        axB.plot(xb, estimator.boundary_marginals_, color=COLORS["blue"], lw=1.0)
    for b in estimator.map_boundaries_[1:-1]:
        axB.axvline(float(b), color=COLORS["red"], ls="--", lw=0.6, alpha=0.8)
    for b in bundle.true_boundaries[1:-1]:
        axB.axvline(float(b), color=COLORS["green"], ls=":", lw=0.9, alpha=0.6)
    axB.set_ylim(0, 1.05)
    axB.set_xlim(0, n)
    axB.set_xlabel("index")
    axB.set_ylabel(r"$P(b_i = 1 \mid y)$")
    add_panel_label(axB, "B")

    # ---- Panel C: posterior over k (independent x-axis) ----------------------
    k_vals = np.arange(1, estimator.k_posterior_.size + 1)
    axC.bar(k_vals, estimator.k_posterior_, color=COLORS["blue"], edgecolor="none", width=0.75)
    axC.axvline(
        estimator.k_map_,
        color=COLORS["red"],
        ls="--",
        lw=1.0,
        label=rf"$k_{{MAP}} = {estimator.k_map_}$",
    )
    axC.set_xlim(0.3, estimator.k_posterior_.size + 0.7)
    axC.set_xlabel("k")
    axC.set_ylabel(r"$P(k \mid y)$")
    axC.legend(loc="best", fontsize=8, frameon=False)
    add_panel_label(axC, "C")

    # ---- Panel D: cumulative held-out log-likelihood -------------------------
    trace = _held_out_trace(estimator, bundle)
    axD.plot(idx, trace, lw=1.4, color=COLORS["blue"], label="BayesBreak MAP")
    if show_null_baseline:
        try:
            null_tr = _null_trace(estimator, bundle)
            axD.plot(idx, null_tr, lw=1.0, color=COLORS["grey"], ls=":", label="k=1 null")
        except Exception:
            pass
    axD.set_xlim(0, n)
    axD.set_xlabel("index")
    axD.set_ylabel(r"$\sum_{t' \leq t}\log p(y_{t'}\mid \mathcal{M})$")
    axD.legend(loc="best", fontsize=8, frameon=False)
    add_panel_label(axD, "D")

    save_figure(fig, outdir / fig_name, formats=("png", "pdf"))
