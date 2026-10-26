"""Stage B: render all manuscript figures from cached CSV data.

Reads only from docs/manuscripts/shared/figures/data/*.csv and *.json.
Never imports experiment runners or fits models.  Produces PDF and PNG
for every active figure.

Usage
-----
python scripts/figures/render.py --all
python scripts/figures/render.py --only fig1 fig_E01_calibration
python scripts/figures/render.py --all --outdir /tmp/figs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "data"
_OUT_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "results"

sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _nan(v: object) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _require_csv(name: str) -> Path:
    p = _DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {p}\n"
            f"Run: python scripts/figures/prepare_data.py --only <figure_id>"
        )
    return p


def _setup() -> None:
    """Import and apply shared style — safe to call multiple times."""
    from _style import setup_style
    setup_style()


import numpy as np  # noqa: E402


# ---------------------------------------------------------------------------
# fig1: synthetic Gaussian time series
# ---------------------------------------------------------------------------

def render_fig1(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.1)

    rows = _load_csv(_require_csv("fig1_synthetic_gaussian.csv"))
    x = np.array([int(r["x"]) for r in rows])
    y = np.array([_nan(r["y"]) for r in rows])
    true_sig = np.array([_nan(r["true_signal"]) for r in rows])
    mp = np.array([_nan(r["map_prediction"]) for r in rows])
    ci_lo = np.array([_nan(r["ci_lo"]) for r in rows])
    ci_hi = np.array([_nan(r["ci_hi"]) for r in rows])
    bm = np.array([_nan(r["boundary_marginal"]) for r in rows])
    is_mb = np.array([int(r["is_map_boundary"]) for r in rows])
    mb_xs = x[is_mb == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.scatter(x, y, s=10, alpha=0.4, color=COLORS["grey"], edgecolors="none",
                zorder=1, label="Observed")
    ax1.plot(x, true_sig, lw=1.5, ls="--", color=COLORS["black"], zorder=2, label="True signal")
    ax1.plot(x, mp, lw=2.0, color=COLORS["blue"], zorder=3, label="MAP fit")
    ax1.fill_between(x, ci_lo, ci_hi, alpha=0.18, color=COLORS["blue"], zorder=2)
    for bx in mb_xs:
        ax1.axvline(float(bx), color=COLORS["red"], ls="--", lw=0.8, alpha=0.7)
    ax1.set_ylabel("y")
    ax1.legend(fontsize=9, frameon=False, ncol=3)
    add_panel_label(ax1, "A")

    valid_bm = np.isfinite(bm)
    ax2.fill_between(x[valid_bm], 0, bm[valid_bm], alpha=0.35, color=COLORS["blue"])
    ax2.plot(x[valid_bm], bm[valid_bm], lw=1.0, color=COLORS["blue"])
    for bx in mb_xs:
        ax2.axvline(float(bx), color=COLORS["red"], ls="--", lw=0.8, alpha=0.7)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Observation index")
    ax2.set_ylabel(r"$P(b_i=1\mid y)$")
    add_panel_label(ax2, "B")

    fig.tight_layout()
    save_figure(fig, out_dir / "fig1_synthetic_gaussian", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig2: family showcase
# ---------------------------------------------------------------------------

def render_fig2(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, get_figsize, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    setup_style(font_scale=1.1)

    rows = _load_csv(_require_csv("fig2_family_showcase.csv"))
    families = ["gaussian", "poisson", "binomial", "beta"]
    fam_labels = {"gaussian": "Gaussian", "poisson": "Poisson",
                  "binomial": "Binomial", "beta": "Beta-valued"}
    ylabels = {"gaussian": "Signal", "poisson": "Count",
               "binomial": "Proportion", "beta": "Proportion"}
    panel_letters = {"gaussian": "A", "poisson": "B", "binomial": "C", "beta": "D"}

    by_fam: dict[str, dict] = {f: {"x": [], "y": [], "true_signal": [], "fit": [], "mb": []} for f in families}
    for r in rows:
        f = r["family"]
        by_fam[f]["x"].append(int(r["x"]))
        by_fam[f]["y"].append(_nan(r["y"]))
        by_fam[f]["true_signal"].append(_nan(r["true_signal"]))
        by_fam[f]["fit"].append(_nan(r["fit"]))
        by_fam[f]["mb"].append(int(r["is_map_boundary"]))

    figsize = get_figsize("double", aspect=0.75, nrows=2, ncols=2)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    for ax, fam in zip(axes_flat, families):
        d = by_fam[fam]
        x = np.array(d["x"])
        y = np.array(d["y"])
        ts = np.array(d["true_signal"])
        fit = np.array(d["fit"])
        mb_xs = x[np.array(d["mb"]) == 1]

        ax.scatter(x, y, s=20, alpha=0.5, color=COLORS["grey"], edgecolors="none", zorder=1)
        ax.plot(ts, lw=2, ls="--", color=COLORS["black"], zorder=2)
        ax.plot(fit, lw=2.5, color=COLORS["blue"], zorder=3)
        ax.set_ylabel(ylabels[fam])
        ax.set_xlim(0, len(x))
        add_panel_label(ax, panel_letters[fam])

    axes_flat[2].set_xlabel("Time index")
    axes_flat[3].set_xlabel("Time index")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["grey"],
               markersize=6, alpha=0.7, label="Observations", linestyle="None"),
        Line2D([0], [0], color=COLORS["black"], ls="--", lw=2, label="True signal"),
        Line2D([0], [0], color=COLORS["blue"], lw=2.5, label="BayesBreak fit"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    save_figure(fig, out_dir / "fig2_family_showcase", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig3: boundary calibration
# ---------------------------------------------------------------------------

def render_fig3(out_dir: Path) -> None:
    from _style import COLORS, get_figsize, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.1)

    rows = _load_csv(_require_csv("fig3_boundary_calibration.csv"))
    p_bin = np.array([_nan(r["p_bin"]) for r in rows])
    y_bin = np.array([_nan(r["y_bin"]) for r in rows])
    ci_lo = np.array([_nan(r["ci_lo"]) for r in rows])
    ci_hi = np.array([_nan(r["ci_hi"]) for r in rows])
    yerr = np.array([y_bin - ci_lo, ci_hi - y_bin])

    # ECE and Brier from JSON config
    cfg_path = _DATA_DIR / "fig3_boundary_calibration.json"
    ece = brier = float("nan")
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        sm = cfg.get("scalar_metrics", {})
        ece = float(sm.get("ece", float("nan")))
        brier = float(sm.get("brier", float("nan")))

    figsize = get_figsize("single", aspect=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot([0, 1], [0, 1], ls="-", lw=1.5, color=COLORS["grey"], label="Perfect", zorder=1)
    ax.errorbar(p_bin, y_bin, yerr=yerr, marker="o", markersize=8, lw=2,
                color=COLORS["blue"], ecolor=COLORS["blue"], capsize=4, capthick=1.5,
                label="BayesBreak", zorder=2)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    textstr = f"ECE = {ece:.3f}\nBrier = {brier:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.7"})
    save_figure(fig, out_dir / "fig3_boundary_calibration", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig4: latent groups
# ---------------------------------------------------------------------------

def render_fig4(out_dir: Path) -> None:
    from _style import COLORS, _MUTED_PALETTE, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.0)

    assign_path = _require_csv("fig4_assignments.csv")
    bound_path = _require_csv("fig4_boundary.csv")
    ts_path = _require_csv("fig4_timeseries.csv")

    a_rows = _load_csv(assign_path)
    b_rows = _load_csv(bound_path)
    t_rows = _load_csv(ts_path)

    r0 = np.array([_nan(r["r_group0"]) for r in a_rows])
    r1 = np.array([_nan(r["r_group1"]) for r in a_rows])

    bx = np.array([int(r["x"]) for r in b_rows])
    g0_post = np.array([_nan(r.get("group0_posterior", "nan")) for r in b_rows])
    g1_post = np.array([_nan(r.get("group1_posterior", "nan")) for r in b_rows])

    # Per-group time series
    grp_data: dict[int, dict] = {0: {"x": [], "y": [], "fit": [], "true_signal": []},
                                   1: {"x": [], "y": [], "fit": [], "true_signal": []}}
    for r in t_rows:
        g = int(r["group"])
        grp_data[g]["x"].append(int(r["x"]))
        grp_data[g]["y"].append(_nan(r["y"]))
        grp_data[g]["fit"].append(_nan(r["fit"]))
        grp_data[g]["true_signal"].append(_nan(r["true_signal"]))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Row 0: assignment probabilities
    ax = axes[0, 0]
    n_seqs = len(r0)
    ax.bar(range(n_seqs), r0, color=_MUTED_PALETTE[0], alpha=0.8, label="Group 0")
    ax.bar(range(n_seqs), r1, bottom=r0, color=_MUTED_PALETTE[1], alpha=0.8, label="Group 1")
    ax.set_xlabel("Sequence index")
    ax.set_ylabel("Responsibility")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    add_panel_label(ax, "A")

    # Row 0: group 0 boundary posteriors
    ax = axes[0, 1]
    ax.plot(bx, g0_post, color=_MUTED_PALETTE[0], lw=1.2)
    ax.fill_between(bx, 0, g0_post, alpha=0.3, color=_MUTED_PALETTE[0])
    ax.set_ylabel(r"$P(b_i=1\mid y)$")
    ax.set_xlabel("Position")
    ax.annotate("Group 0", xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=9)
    add_panel_label(ax, "B")

    # Row 0: group 1 boundary posteriors
    ax = axes[0, 2]
    ax.plot(bx, g1_post, color=_MUTED_PALETTE[1], lw=1.2)
    ax.fill_between(bx, 0, g1_post, alpha=0.3, color=_MUTED_PALETTE[1])
    ax.set_ylabel(r"$P(b_i=1\mid y)$")
    ax.set_xlabel("Position")
    ax.annotate("Group 1", xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=9)
    add_panel_label(ax, "C")

    # Row 1: example time series per group
    for g_idx, letter in enumerate(["D", "E"]):
        ax = axes[1, g_idx]
        gd = grp_data[g_idx]
        x = np.array(gd["x"])
        y = np.array(gd["y"])
        fit = np.array(gd["fit"])
        ts = np.array(gd["true_signal"])
        ax.scatter(x, y, s=12, alpha=0.4, color=COLORS["grey"], edgecolors="none")
        ax.plot(x, ts, lw=1.5, ls="--", color=COLORS["black"])
        ax.plot(x, fit, lw=2.0, color=_MUTED_PALETTE[g_idx])
        ax.set_xlabel("Index")
        ax.set_ylabel("y")
        ax.annotate(f"Group {g_idx} example", xy=(0.02, 0.97),
                    xycoords="axes fraction", va="top", fontsize=9)
        add_panel_label(ax, letter)

    axes[1, 2].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig4_latent_groups", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig5: runtime scaling
# ---------------------------------------------------------------------------

def render_fig5(out_dir: Path) -> None:
    from _style import COLORS, _MUTED_PALETTE, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.0)

    rows = _load_csv(_require_csv("fig5_runtime_scaling_data.csv"))

    n_sweep = [(int(r["n"]), int(r["k_max"]), _nan(r["mean_seconds"]), _nan(r["std_seconds"]))
               for r in rows if r.get("sweep") == "n"]
    k_sweep = [(int(r["n"]), int(r["k_max"]), _nan(r["mean_seconds"]), _nan(r["std_seconds"]))
               for r in rows if r.get("sweep") == "k"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Panel A: n sweep (log-log)
    if n_sweep:
        k_maxs = sorted(set(r[1] for r in n_sweep))
        palette = [_MUTED_PALETTE[0], _MUTED_PALETTE[2]]
        for i, km in enumerate(k_maxs):
            pts = [(n, m, s) for n, k, m, s in n_sweep if k == km]
            pts.sort(key=lambda t: t[0])
            ns = np.array([p[0] for p in pts])
            ms = np.array([p[1] for p in pts])
            ses = np.array([p[2] for p in pts])
            color = palette[i % len(palette)]
            ax1.loglog(ns, ms, "o-", color=color, lw=1.4, label=f"k_max={km}")
            ax1.fill_between(ns, ms - ses, ms + ses, alpha=0.15, color=color)
        ax1.set_xlabel("n")
        ax1.set_ylabel("Wall-clock time (s)")
        ax1.legend(fontsize=9, frameon=False)
        ax1.grid(True, alpha=0.25)
        add_panel_label(ax1, "A")

    # Panel B: k_max sweep (semilogy)
    if k_sweep:
        ns_fixed = sorted(set(r[0] for r in k_sweep))
        for i, n_val in enumerate(ns_fixed):
            pts = [(k, m, s) for n, k, m, s in k_sweep if n == n_val]
            pts.sort(key=lambda t: t[0])
            ks = np.array([p[0] for p in pts])
            ms = np.array([p[1] for p in pts])
            ax2.semilogy(ks, ms, "s--", color=_MUTED_PALETTE[i % len(_MUTED_PALETTE)],
                         lw=1.4, label=f"n={n_val}")
        ax2.set_xlabel("k_max")
        ax2.set_ylabel("Wall-clock time (s)")
        ax2.legend(fontsize=9, frameon=False)
        ax2.grid(True, alpha=0.25)
        add_panel_label(ax2, "B")
    else:
        ax2.set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_dir / "fig5_runtime_scaling", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Real-data figures (fig6, fig7, fig9) — four-panel template
# ---------------------------------------------------------------------------

def _render_realdata(csv_stem: str, fig_name: str, y_label: str, out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=0.95)

    ts_rows = _load_csv(_require_csv(f"{csv_stem}.csv"))
    kp_path = _DATA_DIR / f"{csv_stem}_kposterior.csv"

    x = np.array([int(r["x"]) for r in ts_rows])
    y = np.array([_nan(r["y"]) for r in ts_rows])
    mp = np.array([_nan(r["map_prediction"]) for r in ts_rows])
    bm = np.array([_nan(r["boundary_marginal"]) for r in ts_rows])
    bc = np.array([_nan(r.get("bayes_curve_mean", "nan")) for r in ts_rows])
    ht = np.array([_nan(r.get("held_out_trace", "nan")) for r in ts_rows])
    nt = np.array([_nan(r.get("null_trace", "nan")) for r in ts_rows])
    is_mb = np.array([int(r["is_map_boundary"]) for r in ts_rows])
    is_tb = np.array([int(r["is_true_boundary"]) for r in ts_rows])
    mb_xs = x[is_mb == 1]
    tb_xs = x[is_tb == 1]

    fig = plt.figure(figsize=(7.8, 8.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.0, 1.0, 1.1], hspace=0.32)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)
    axC = fig.add_subplot(gs[2])
    axD = fig.add_subplot(gs[3])

    # Panel A: data + MAP fit
    axA.scatter(x, y, s=4, alpha=0.35, color=COLORS["grey"], edgecolors="none",
                label="observed", zorder=1)
    axA.plot(x, mp, lw=1.6, color=COLORS["blue"], label="MAP", zorder=3)
    if np.any(np.isfinite(bc)):
        axA.plot(x, bc, lw=1.2, color=COLORS["red"], alpha=0.9, label="Bayes curve", zorder=2)
    for bx in mb_xs:
        axA.axvline(float(bx), color=COLORS["red"], ls="--", lw=0.6, alpha=0.7, zorder=4)
    for tbx in tb_xs:
        axA.axvline(float(tbx), color=COLORS["green"], ls=":", lw=0.9, alpha=0.7, zorder=4)
    axA.set_ylabel(y_label)
    axA.set_xlim(0, len(x))
    axA.legend(loc="best", ncol=3, fontsize=10, frameon=False)
    add_panel_label(axA, "A")
    plt.setp(axA.get_xticklabels(), visible=False)

    # Panel B: boundary marginals
    valid_bm = np.isfinite(bm)
    if np.any(valid_bm):
        xb = x[valid_bm]
        axB.fill_between(xb, 0, bm[valid_bm], color=COLORS["blue"], alpha=0.35, linewidth=0)
        axB.plot(xb, bm[valid_bm], color=COLORS["blue"], lw=1.0)
    for bx in mb_xs:
        axB.axvline(float(bx), color=COLORS["red"], ls="--", lw=0.6, alpha=0.8)
    axB.set_ylim(0, 1.05)
    axB.set_xlim(0, len(x))
    axB.set_xlabel("index")
    axB.set_ylabel(r"$P(b_i = 1 \mid y)$")
    add_panel_label(axB, "B")

    # Panel C: k posterior
    if kp_path.exists():
        kp_rows = _load_csv(kp_path)
        k_vals = np.array([int(r["k"]) for r in kp_rows])
        k_probs = np.array([_nan(r["prob"]) for r in kp_rows])
        k_map = k_vals[np.argmax(k_probs)] if len(k_vals) > 0 else 0
        axC.bar(k_vals, k_probs, color=COLORS["blue"], edgecolor="none", width=0.75)
        axC.axvline(k_map, color=COLORS["red"], ls="--", lw=1.0,
                    label=rf"$k_{{MAP}} = {k_map}$")
        axC.set_xlim(0.3, max(k_vals) + 0.7)
        axC.set_xlabel("k")
        axC.set_ylabel(r"$P(k \mid y)$")
        axC.legend(loc="best", fontsize=10, frameon=False)
    add_panel_label(axC, "C")

    # Panel D: held-out trace
    if np.any(np.isfinite(ht)):
        axD.plot(x, ht, lw=1.4, color=COLORS["blue"], label="BayesBreak MAP")
    if np.any(np.isfinite(nt)):
        axD.plot(x, nt, lw=1.0, color=COLORS["grey"], ls=":", label="k=1 null")
    axD.set_xlim(0, len(x))
    axD.set_xlabel("index")
    axD.set_ylabel(r"$\sum_{t' \leq t}\log p(y_{t'}\mid \mathcal{M})$")
    axD.legend(loc="best", fontsize=10, frameon=False)
    add_panel_label(axD, "D")

    save_figure(fig, out_dir / fig_name, formats=("png", "pdf"))
    plt.close(fig)


def render_fig6(out_dir: Path) -> None:
    _render_realdata("fig6_welllog", "fig6_welllog", "NMR signal", out_dir)


def render_fig7(out_dir: Path) -> None:
    _render_realdata("fig7_cgh", "fig7_cgh", "Copy-number log-ratio", out_dir)


def render_fig9(out_dir: Path) -> None:
    _render_realdata("fig9_methylation", "fig9_methylation", "Methylation fraction", out_dir)


# ---------------------------------------------------------------------------
# E01: calibration panels + recovery
# ---------------------------------------------------------------------------

def render_e01_calibration(out_dir: Path) -> None:
    from _style import COLORS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math as _math
    setup_style()

    rows = _load_csv(_require_csv("fig_E01_calibration.csv"))
    strata = sorted(set(r["stratum"] for r in rows))
    calib: dict[str, dict] = {s: {"pred": [], "obs": [], "lo": [], "hi": [], "ece": 0.0}
                               for s in strata}
    for r in rows:
        s = r["stratum"]
        calib[s]["pred"].append(_nan(r["bin_mean_pred"]))
        calib[s]["obs"].append(_nan(r["bin_mean_obs"]))
        calib[s]["lo"].append(_nan(r["bin_obs_ci_lo"]))
        calib[s]["hi"].append(_nan(r["bin_obs_ci_hi"]))
        calib[s]["ece"] = _nan(r["ece"])

    n_panels = len(strata)
    n_cols = min(n_panels, 4)
    n_rows = _math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.2 * n_rows),
                              constrained_layout=True, squeeze=False)
    ax_flat = axes.flatten()
    for ax in ax_flat[n_panels:]:
        ax.set_visible(False)

    for ax, stratum in zip(ax_flat, strata):
        cv = calib[stratum]
        x = cv["pred"]
        y = cv["obs"]
        lo = cv["lo"]
        hi = cv["hi"]
        ece = cv["ece"]
        x_max = max(x) if x else 1.0
        y_max = max(max(y), max(hi)) if y else 1.0
        xy_max = max(x_max, y_max) * 1.22
        ax.plot([0, xy_max], [0, xy_max], color=COLORS["grey"], lw=1.5, ls="--",
                label="Perfect cal.")
        ax.plot(x, y, "o-", color=COLORS["blue"], ms=7, lw=2.0, label="Observed")
        ax.fill_between(x, lo, hi, alpha=0.20, color=COLORS["blue"])
        ax.set_xlabel("Predicted P(boundary | y)", fontsize=12)
        ax.set_ylabel("Observed frequency", fontsize=12)
        ax.annotate(stratum, xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8, va="top")
        ax.set_xlim(0, xy_max)
        ax.set_ylim(0, xy_max)
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10)
        ax.text(0.04, 0.88, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=11)

    save_figure(fig, out_dir / "fig_E01_calibration", formats=("pdf", "png"))
    plt.close(fig)


def render_e01_recovery(out_dir: Path) -> None:
    from _style import COLORS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    rows = _load_csv(_require_csv("fig_E01_data.csv"))
    primary = [r for r in rows if str(r.get("ablation", "False")).lower() not in ("true", "1")
               and int(r.get("k_true", 0)) > 1]
    if not primary:
        logger.warning("No primary E01 rows found")
        return

    n_values = sorted({int(r["n"]) for r in primary})
    k_values = sorted({int(r["k_true"]) for r in primary})
    k_colors = {k: c for k, c in zip(k_values, [COLORS["blue"], COLORS["red"]])}
    markers = {n: m for n, m in zip(sorted(n_values), ["o", "s"])}

    fig, axes = plt.subplots(1, len(n_values), figsize=(6.5 * len(n_values), 5.5),
                              constrained_layout=True, squeeze=False)
    for ax, n in zip(axes[0], n_values):
        for k in k_values:
            cells = [r for r in primary if int(r["n"]) == n and int(r["k_true"]) == k]
            cells.sort(key=lambda r: float(r["delta"]))
            if not cells:
                continue
            deltas = [float(r["delta"]) for r in cells]
            means = [_nan(r["mean_f1"]) for r in cells]
            lo = [_nan(r["f1_ci_lo"]) for r in cells]
            hi = [_nan(r["f1_ci_hi"]) for r in cells]
            yerr = np.array([[m - l, h_ - m] for m, l, h_ in zip(means, lo, hi)]).T
            ax.errorbar(deltas, means, yerr=yerr,
                        fmt=f"{markers.get(n, 'o')}-",
                        color=k_colors.get(k, COLORS["grey"]),
                        label=f"k={k}", capsize=3)
        ax.set_xlabel(f"Δ/σ  (n = {n})")
        ax.set_ylabel("F1 @ τ=3")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color=COLORS["grey"], lw=0.8, ls="--")
        ax.legend()

    save_figure(fig, out_dir / "fig_E01_recovery", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# E02: family portability recovery
# ---------------------------------------------------------------------------

def render_e02_recovery(out_dir: Path) -> None:
    from _style import COLOR_CYCLE, COLORS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math as _math
    setup_style(font_scale=1.1)

    rows = _load_csv(_require_csv("fig_E02_data.csv"))
    families = ["gaussian", "poisson", "binomial", "negbin", "beta_obs"]
    fam_labels = {"gaussian": "Gaussian", "poisson": "Poisson", "binomial": "Binomial",
                  "negbin": "NegBin", "beta_obs": "Beta-obs"}
    fam_colors = {fam: COLOR_CYCLE[i] for i, fam in enumerate(families)}
    effect_markers = ["o", "^"]

    import matplotlib as _mpl
    _mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.28, wspace=0.32)
    x_pos = np.arange(len(families))

    def _plot_metric(ax, metric_key, ci_lo_key, ci_hi_key, ylabel):
        for i, fam in enumerate(families):
            fam_cells = [r for r in rows if r.get("family") == fam]
            if not fam_cells:
                continue
            x_jitter = x_pos[i] + np.linspace(-0.18, 0.18, len(fam_cells))
            for j, (r, xj) in enumerate(zip(fam_cells, x_jitter)):
                m = _nan(r.get(metric_key, "nan"))
                lo = _nan(r.get(ci_lo_key, "nan"))
                hi = _nan(r.get(ci_hi_key, "nan"))
                if not _math.isfinite(m):
                    continue
                yerr = np.array([[m - lo, hi - m]]).T if _math.isfinite(lo) else None
                marker = effect_markers[j] if j < len(effect_markers) else "o"
                ax.errorbar(xj, m, yerr=yerr, fmt=marker, color=fam_colors[fam],
                            capsize=5, capthick=1.4, elinewidth=1.4,
                            markersize=9, markeredgewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([fam_labels.get(f, f) for f in families],
                           rotation=25, ha="right", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(-0.03, 1.08)
        ax.axhline(1.0, color=COLORS["grey"], lw=1.0, ls="--", zorder=0)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlim(-0.5, len(families) - 0.5)

    _plot_metric(axes[0], "mean_f1", "f1_ci_lo", "f1_ci_hi", "F1 @ τ=3")
    _plot_metric(axes[1], "mean_p_k_true_given_y", "p_k_true_given_y_ci_lo",
                 "p_k_true_given_y_ci_hi", "P(k = 3 | y)")

    save_figure(fig, out_dir / "fig_E02_recovery", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# E03: irregular design
# ---------------------------------------------------------------------------

def render_e03(fig_type: str, out_dir: Path) -> None:
    from _style import PRIOR_COLORS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    rows = _load_csv(_require_csv("fig_E03_data.csv"))
    tmpl_map = {"gap_true": "[60, 120]", "gap_false": "[60, 180]"}
    tmpl_str = tmpl_map.get(fig_type, "[60, 120]")
    tmpl_rows = [r for r in rows if r.get("true_template") == tmpl_str]

    priors = ["uniform", "gap_hazard", "length_cohesion", "combined"]
    prior_labels = {"uniform": "A: Uniform", "gap_hazard": "B: Gap hazard",
                    "length_cohesion": "C: Length cohesion", "combined": "D: Combined"}
    markers = {"uniform": "o", "gap_hazard": "s", "length_cohesion": "^", "combined": "D"}
    colors = PRIOR_COLORS

    gaps = sorted(set(float(r["gap"]) for r in tmpl_rows if r.get("gap")))
    x_ticks = np.arange(len(gaps))
    gap_labels = [f"G={int(g)}" for g in gaps]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax_idx, (metric, ci_lo_k, ci_hi_k, ylabel) in enumerate([
        ("mean_f1", "f1_ci_lo", "f1_ci_hi", f"Mean F1@3 — {tmpl_str}"),
        ("mean_p_b_gap", "p_b_gap_ci_lo", "p_b_gap_ci_hi", "P(boundary at gap)"),
    ]):
        ax = axes[ax_idx]
        for prior in priors:
            y_vals, y_lo, y_hi = [], [], []
            for gap in gaps:
                found = next((r for r in tmpl_rows
                              if abs(float(r.get("gap", 0)) - gap) < 1e-9
                              and r.get("prior") == prior), None)
                y_vals.append(_nan(found.get(metric, "nan")) if found else float("nan"))
                y_lo.append(_nan(found.get(ci_lo_k, "nan")) if found else float("nan"))
                y_hi.append(_nan(found.get(ci_hi_k, "nan")) if found else float("nan"))

            ya = np.array(y_vals)
            yl = np.array(y_lo)
            yh = np.array(y_hi)
            valid = np.isfinite(ya)
            if np.any(valid):
                ax.errorbar(x_ticks[valid], ya[valid],
                            yerr=[ya[valid] - yl[valid], yh[valid] - ya[valid]],
                            label=prior_labels[prior], color=colors[prior],
                            marker=markers[prior], capsize=3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(gap_labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Gap G")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir / f"fig_E03_{fig_type}", formats=("pdf", "png"))
    plt.close(fig)


def render_e03_gap_true(out_dir: Path) -> None:
    render_e03("gap_true", out_dir)


def render_e03_gap_false(out_dir: Path) -> None:
    render_e03("gap_false", out_dir)


# ---------------------------------------------------------------------------
# E04: shared boundary pooling
# ---------------------------------------------------------------------------

def render_e04_pooled(out_dir: Path) -> None:
    from _style import _MUTED_PALETTE, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    rows = _load_csv(_require_csv("fig_E04_data.csv"))
    S_vals = sorted({int(r.get("S", 0)) for r in rows if r.get("S")})
    nfs = sorted({float(r.get("nonshared_fraction", 0)) for r in rows if r.get("nonshared_fraction")})
    nf_labels = {0: "0% non-shared", 0.25: "25% non-shared", 0.5: "50% non-shared"}
    colors = {0: _MUTED_PALETTE[0], 0.25: _MUTED_PALETTE[2], 0.5: _MUTED_PALETTE[1]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x_ticks = np.arange(len(S_vals))
    x_labels = [f"S={s}" for s in S_vals]

    def _get(nf: float, S: int, col: str) -> float:
        r = next((r for r in rows if int(r.get("S", -1)) == S
                  and abs(float(r.get("nonshared_fraction", -1)) - nf) < 1e-9), None)
        return _nan(r.get(col, "nan")) if r else float("nan")

    for nf in nfs:
        color = colors.get(nf, "black")
        label = nf_labels.get(nf, str(nf))
        pf1 = np.array([_get(nf, S, "mean_pooled_f1") for S in S_vals])
        pl = np.array([_get(nf, S, "pooled_f1_ci_lo") for S in S_vals])
        ph = np.array([_get(nf, S, "pooled_f1_ci_hi") for S in S_vals])
        if1 = np.array([_get(nf, S, "mean_indep_f1") for S in S_vals])
        il = np.array([_get(nf, S, "indep_f1_ci_lo") for S in S_vals])
        ih = np.array([_get(nf, S, "indep_f1_ci_hi") for S in S_vals])

        vp = np.isfinite(pf1)
        vi = np.isfinite(if1)
        if np.any(vp):
            axes[0].errorbar(x_ticks[vp] - 0.1, pf1[vp],
                             yerr=[pf1[vp] - pl[vp], ph[vp] - pf1[vp]],
                             label=f"Pooled ({label})", color=color, marker="o", capsize=3)
        if np.any(vi):
            axes[0].errorbar(x_ticks[vi] + 0.1, if1[vi],
                             yerr=[if1[vi] - il[vi], ih[vi] - if1[vi]],
                             label=f"Indep ({label})", color=color, marker="s",
                             linestyle="--", capsize=3)

    for ax in axes:
        ax.set_xlabel("Number of sequences S")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean F1@3")
    axes[0].legend(fontsize=6)
    axes[1].set_visible(False)

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E04_pooled", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# E05: latent template recovery
# ---------------------------------------------------------------------------

def render_e05_templates(out_dir: Path) -> None:
    from _style import save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    rows = _load_csv(_require_csv("fig_E05_data.csv"))
    G_values = sorted({int(r["G"]) for r in rows})
    S_values = sorted({int(r["S"]) for r in rows})
    delta_values = sorted({float(r["Delta"]) for r in rows})

    lookup: dict = {}
    for r in rows:
        lookup[(int(r["G"]), int(r["S"]), float(r["Delta"]))] = r

    n_G = len(G_values)
    n_S = len(S_values)
    fig, axes = plt.subplots(n_G, n_S, figsize=(4.5 * n_S, 3.5 * n_G), squeeze=False)

    for i, G in enumerate(G_values):
        for j, S in enumerate(S_values):
            ax = axes[i][j]
            ax2 = ax.twinx()
            xs = list(delta_values)

            def _vals(col: str) -> list:
                return [_nan(lookup.get((G, S, d), {}).get(col, "nan")) for d in xs]

            accs = _vals("mean_accuracy")
            acc_lo = _vals("accuracy_ci_lo")
            acc_hi = _vals("accuracy_ci_hi")
            aris = _vals("mean_ari")
            ari_lo = _vals("ari_ci_lo")
            ari_hi = _vals("ari_ci_hi")
            dups = _vals("mean_has_duplicate_templates")

            ax.plot(xs, accs, "o-", color="C0", label="Accuracy")
            ax.fill_between(xs, acc_lo, acc_hi, alpha=0.2, color="C0")
            if G > 1:
                ax.plot(xs, aris, "s--", color="C1", label="ARI")
                ax.fill_between(xs, ari_lo, ari_hi, alpha=0.2, color="C1")
            ax2.bar(xs, dups, width=0.15, alpha=0.4, color="C3", label="Dup. freq.")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_ylabel("Dup. template freq.", fontsize=8, color="C3")
            ax2.tick_params(axis="y", colors="C3", labelsize=7)
            ax.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(xs)
            ax.set_xlabel(r"$\Delta$", fontsize=9)
            ax.set_ylabel("Accuracy / ARI", fontsize=8)
            ax.annotate(f"G={G}, S={S}", xy=(0.03, 0.96), xycoords="axes fraction",
                        fontsize=8, va="top")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E05_templates", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# E06: non-conjugate approximation errors
# ---------------------------------------------------------------------------

def render_e06_local_error(out_dir: Path) -> None:
    from _style import METHOD_COLORS, METHOD_LINESTYLES, METHOD_MARKERS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    rows = _load_csv(_require_csv("fig_E06_local_error.csv"))
    p_vals = sorted({float(r["p"]) for r in rows})
    B_vals = sorted({int(r["B"]) for r in rows})

    methods = [("laplace", "Laplace"), ("jj", "JJ"), ("ep", "EP")]

    fig, axes = plt.subplots(1, len(p_vals), figsize=(10, 3.5), sharey=False)
    if len(p_vals) == 1:
        axes = [axes]

    for ax_i, p in enumerate(p_vals):
        ax = axes[ax_i]
        for key, label in methods:
            meds = []
            for B in B_vals:
                r = next((r for r in rows if abs(float(r["p"]) - p) < 0.01 and int(r["B"]) == B), None)
                meds.append(_nan(r.get(f"{key}_median_abs_err", "nan")) if r else float("nan"))
            valid = np.array([math.isfinite(m) for m in meds])
            if np.any(valid):
                B_arr = np.array(B_vals)[valid]
                m_arr = np.array(meds)[valid]
                ax.semilogy(B_arr, m_arr,
                            marker=METHOD_MARKERS.get(key, "o"),
                            linestyle=METHOD_LINESTYLES.get(key, "-"),
                            color=METHOD_COLORS.get(key, "#444444"),
                            label=label)
        ax.set_xlabel("Block size $B$")
        ax.annotate(f"$p = {p}$", xy=(0.97, 0.96), xycoords="axes fraction",
                    fontsize=8, va="top", ha="right")
        ax.legend(fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("Median absolute log-evidence error")

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E06_local_error", formats=("pdf", "png"))
    plt.close(fig)


def render_e06_global_error(out_dir: Path) -> None:
    from _style import METHOD_COLORS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style()

    rows = _load_csv(_require_csv("fig_E06_global_error.csv"))
    methods = ["laplace", "jj", "ep"]
    method_labels = ["Laplace", "JJ", "EP"]

    by_method: dict[str, list] = {m: [] for m in methods}
    for r in rows:
        m = r.get("method", "")
        v = _nan(r.get("max_pk_diff", "nan"))
        if m in by_method and math.isfinite(v):
            by_method[m].append(v)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        vals = by_method[method]
        if vals:
            ax1.violinplot([vals], positions=[i], showmedians=True, widths=0.6)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(method_labels)
    ax1.set_ylabel("Max |p(k|y) difference| vs quadrature reference")

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E06_global_error", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# E07: misspecification robustness
# ---------------------------------------------------------------------------

def render_e07_robustness(out_dir: Path) -> None:
    from _style import SPEC_COLORS, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    setup_style()

    rows = _load_csv(_require_csv("fig_E07_robustness.csv"))
    miss_order = sorted(set(r["missingness"] for r in rows))

    fig, axes = plt.subplots(1, len(miss_order), figsize=(12, 4.5), sharey=True)
    if len(miss_order) == 1:
        axes = [axes]

    for ax, miss in zip(axes, miss_order):
        miss_rows = [r for r in rows if r["missingness"] == miss]
        means = [_nan(r.get("mean_f1", "nan")) for r in miss_rows]
        colors = [SPEC_COLORS["correct"] if int(r.get("is_correct", 0))
                  else SPEC_COLORS["misspecified"] for r in miss_rows]
        labels = [f"{r.get('data_family', '')[:6]}\n{r.get('fit_family', '')}"
                  for r in miss_rows]

        x = np.arange(len(means))
        ax.bar(x, [m if math.isfinite(m) else 0.0 for m in means], color=colors, width=0.7)
        ax.annotate(miss.replace("_", " "), xy=(0.03, 0.97), xycoords="axes fraction",
                    fontsize=8, va="top")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("Mean F1@3")
        ax.axhline(1.0, color="#888888", lw=0.5, ls="--")

    legend_elements = [
        Patch(facecolor=SPEC_COLORS["correct"], label="Correct fit"),
        Patch(facecolor=SPEC_COLORS["misspecified"], label="Misspecified fit"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, out_dir / "fig_E07_robustness", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# E10: computational scaling
# ---------------------------------------------------------------------------

def render_e10_scaling(out_dir: Path) -> None:
    from _style import COLORS, _MUTED_PALETTE, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=0.95)

    rows = _load_csv(_require_csv("fig_E10_data.csv"))
    ok_rows = [r for r in rows if str(r.get("failed", "")).lower() != "true"]
    n_vals = sorted({int(r["n"]) for r in ok_rows})
    k_vals = sorted({int(r["k_max"]) for r in ok_rows})

    def _get(n: int, k: int, col: str) -> float:
        for r in ok_rows:
            if int(r["n"]) == n and int(r["k_max"]) == k:
                return _nan(r.get(col, "nan"))
        return float("nan")

    palette = [_MUTED_PALETTE[0], _MUTED_PALETTE[2], _MUTED_PALETTE[1]]
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for i, k in enumerate(k_vals):
        color = palette[i % len(palette)]
        ns = np.array(n_vals)
        e2e = np.array([_get(n, k, "end_to_end_median_s") for n in ns])
        mem = np.array([_get(n, k, "peak_memory_mb") for n in ns])
        v_e = np.isfinite(e2e)
        v_m = np.isfinite(mem)
        if np.any(v_e):
            ax_t.plot(ns[v_e], e2e[v_e], marker="o", color=color, lw=1.4, label=f"k_max={k}")
        if np.any(v_m):
            ax_m.plot(ns[v_m], mem[v_m], marker="o", color=color, lw=1.4, label=f"k_max={k}")

    for ax, ylabel in [(ax_t, "End-to-end time (s)"), (ax_m, "Peak memory (MB)")]:
        ax.set_xlabel("n")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E10_scaling", formats=("pdf", "png"))
    plt.close(fig)


def render_e10_stage_breakdown(out_dir: Path) -> None:
    from _style import _MUTED_PALETTE, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=0.95)

    rows = _load_csv(_require_csv("fig_E10_data.csv"))
    ok_rows = [r for r in rows if str(r.get("failed", "")).lower() != "true"]
    k_ref = max(int(r["k_max"]) for r in ok_rows)
    k_rows = [r for r in ok_rows if int(r["k_max"]) == k_ref]
    n_vals = sorted({int(r["n"]) for r in k_rows})

    stages = [
        ("block_evidence_median_s", "Block evidence"),
        ("forward_backward_median_s", "Forward-backward"),
        ("posterior_k_median_s", "Posterior k"),
        ("boundary_marginals_median_s", "Boundary marginals"),
        ("map_backtrack_median_s", "MAP backtrack"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, (col, label) in enumerate(stages):
        ns = np.array(n_vals)
        vals_arr = np.array([_nan(next((r.get(col, "nan") for r in k_rows if int(r["n"]) == n), "nan"))
                              for n in n_vals])
        valid = np.isfinite(vals_arr)
        if np.any(valid):
            ax.plot(ns[valid], vals_arr[valid], marker="o",
                    color=_MUTED_PALETTE[i % len(_MUTED_PALETTE)], lw=1.4, label=label)

    ax.set_xlabel("n")
    ax.set_ylabel("Median time (s)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E10_stage_breakdown", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary figures (figA1–figA5)
# ---------------------------------------------------------------------------

def render_figA1(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    setup_style()

    cfg = _load_json(_DATA_DIR / "figA1_mixture_discovery.json")
    examples = _load_csv(_require_csv("figA1_examples.csv"))
    resp_rows = _load_csv(_require_csv("figA1_responsibilities.csv"))
    conf_rows = _load_csv(_require_csv("figA1_confusion.csv"))
    bm_A = _load_csv(_require_csv("figA1_boundary_A.csv"))
    bm_B = _load_csv(_require_csv("figA1_boundary_B.csv"))
    pool_rows = _load_csv(_require_csv("figA1_pooling.csv"))

    n = cfg["n"]
    accuracy = cfg["accuracy"]
    bounds_A = cfg["bounds_A"]
    bounds_B = cfg["bounds_B"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    # Panel A: example sequences
    ax = axes[0, 0]
    for g, color in [(0, COLORS["blue"]), (1, COLORS["red"])]:
        g_examples = sorted([r for r in examples if int(r["group"]) == g],
                             key=lambda r: (int(r["seq_rank"]), int(r["x"])))
        seqs = {}
        for r in g_examples:
            seqs.setdefault(int(r["seq_rank"]), []).append((_nan(r["x"]), _nan(r["y"]), _nan(r["mu"])))
        for rank, pts in seqs.items():
            xs, ys, mus = zip(*pts)
            ax.plot(xs, ys, color=color, alpha=0.4 + 0.2 * rank, lw=0.8)
            if rank == 0:
                ax.plot(xs, mus, color=color, lw=2, ls="--",
                        label=f"Group {'A' if g == 0 else 'B'} mean")
    ax.set_xlabel("Time"); ax.set_ylabel("Value"); ax.set_xlim(0, n)
    ax.legend(loc="lower right", fontsize=9)
    add_panel_label(ax, "A", "Example sequences")

    # Panel B: responsibility matrix
    ax = axes[0, 1]
    n_seq = len(resp_rows)
    R = np.zeros((n_seq, 2))
    for r in resp_rows:
        i = int(r["sort_rank"])
        R[i, 0] = _nan(r["resp_A"]); R[i, 1] = _nan(r["resp_B"])
    n_A = sum(1 for r in resp_rows if int(r["true_label"]) == 0)
    im = ax.imshow(R.T, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
    ax.axvline(n_A - 0.5, color="black", lw=1, ls="--")
    ax.set_xlabel("Sequence (sorted)"); ax.set_ylabel("Group")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["A", "B"])
    fig.colorbar(im, ax=ax, shrink=0.6, label="Responsibility")
    add_panel_label(ax, "B", f"Assignments (acc={accuracy:.0%})")

    # Panel C: confusion matrix
    ax = axes[0, 2]
    cm = np.zeros((2, 2), dtype=int)
    for r in conf_rows:
        cm[int(r["true_group"]), int(r["pred_group"])] = int(r["count"])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["A", "B"]); ax.set_yticklabels(["A", "B"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            col = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=col, fontsize=12, fontweight="bold")
    add_panel_label(ax, "C", "Confusion matrix")

    # Panel D: boundary posterior A
    ax = axes[1, 0]
    pos_A = [int(r["position"]) for r in bm_A]
    prob_A = [_nan(r["prob"]) for r in bm_A]
    ax.bar(pos_A, prob_A, width=1, color=COLORS["blue"], alpha=0.7)
    for b in bounds_A[1:-1]:
        ax.axvline(b, color="black", lw=1.5, ls="--")
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_xlabel("Position"); ax.set_ylabel("Posterior prob.")
    add_panel_label(ax, "D", f"Group A boundaries (k={cfg['k_A']})")

    # Panel E: boundary posterior B
    ax = axes[1, 1]
    pos_B = [int(r["position"]) for r in bm_B]
    prob_B = [_nan(r["prob"]) for r in bm_B]
    ax.bar(pos_B, prob_B, width=1, color=COLORS["red"], alpha=0.7)
    for b in bounds_B[1:-1]:
        ax.axvline(b, color="black", lw=1.5, ls="--")
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_xlabel("Position"); ax.set_ylabel("Posterior prob.")
    add_panel_label(ax, "E", f"Group B boundaries (k={cfg['k_B']})")

    # Panel F: pooling benefit
    ax = axes[1, 2]
    x_pos = np.array([0, 1, 3, 4])
    pool_dict: dict = {}
    for r in pool_rows:
        pool_dict[(r["group"], r["method"])] = _nan(r["mean_error"])
    heights = [pool_dict.get(("A", "independent"), float("nan")),
               pool_dict.get(("A", "pooled"), float("nan")),
               pool_dict.get(("B", "independent"), float("nan")),
               pool_dict.get(("B", "pooled"), float("nan"))]
    bar_colors = [COLORS["grey"], COLORS["blue"], COLORS["grey"], COLORS["red"]]
    ax.bar(x_pos, heights, color=bar_colors, width=0.8)
    ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(["Group A", "Group B"])
    ax.set_ylabel("Mean boundary error (pts)")
    legend_els = [Patch(facecolor=COLORS["grey"], label="Independent"),
                  Patch(facecolor=COLORS["blue"], label="Pooled (A)"),
                  Patch(facecolor=COLORS["red"], label="Pooled (B)")]
    ax.legend(handles=legend_els, loc="upper right", fontsize=9)
    add_panel_label(ax, "F", "Pooling benefit")

    save_figure(fig, out_dir / "figA1_mixture_discovery", formats=("pdf", "png"))
    plt.close(fig)


def render_figA2(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.05)

    cfg = _load_json(_DATA_DIR / "figA2_snr_sensitivity.json")
    curves = _load_csv(_require_csv("figA2_snr_curves.csv"))
    khist = _load_csv(_require_csv("figA2_khist.csv"))

    sigmas = np.array([_nan(r["sigma"]) for r in curves])
    f1_med = np.array([_nan(r["f1_med"]) for r in curves])
    f1_q1 = np.array([_nan(r["f1_q1"]) for r in curves])
    f1_q3 = np.array([_nan(r["f1_q3"]) for r in curves])
    mse_med = np.array([_nan(r["mse_med"]) for r in curves])
    mse_q1 = np.array([_nan(r["mse_q1"]) for r in curves])
    mse_q3 = np.array([_nan(r["mse_q3"]) for r in curves])
    sigma_c = cfg.get("sigma_c")
    delta_min = cfg["delta_min"]
    avg_len = cfg["avg_seg_len"]
    k_max = cfg["k_max"]

    sigma_idxs = sorted({int(r["sigma_idx"]) for r in khist})
    n_sigma = len(sigma_idxs)
    H = np.zeros((k_max, n_sigma))
    for r in khist:
        si = int(r["sigma_idx"])
        ki = int(r["k"]) - 1
        if 0 <= ki < k_max and 0 <= si < n_sigma:
            H[ki, si] = _nan(r["freq"])

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

    axA = axes[0]
    axA.fill_between(sigmas, f1_q1, f1_q3, alpha=0.25, color=COLORS["blue"], linewidth=0)
    axA.plot(sigmas, f1_med, "o-", color=COLORS["blue"], markersize=5, lw=2.0)
    axA.axhline(1.0, ls=":", color=COLORS["grey"], lw=1)
    axA.axhline(0.5, ls=":", color=COLORS["grey"], lw=1)
    if sigma_c is not None:
        axA.axvline(sigma_c, ls="--", color=COLORS["red"], lw=1.5, alpha=0.85)
        axA.annotate(
            rf"$\sigma_{{c}}\approx{sigma_c:.2f}$" + "\n" + rf"SNR$\approx${delta_min/sigma_c:.1f}",
            xy=(sigma_c, 0.5), xytext=(sigma_c * 1.8, 0.72), fontsize=9,
            color=COLORS["red"], arrowprops={"arrowstyle": "->", "color": COLORS["red"], "lw": 1.2})
    axA.set_xscale("log"); axA.set_xlabel(r"Noise $\sigma$")
    axA.set_ylabel(r"Boundary F1@$\tau$"); axA.set_ylim(-0.04, 1.08)
    axA.grid(True, which="major", ls="-", alpha=0.2, color=COLORS["grey"])
    add_panel_label(axA, "A", "Detection vs noise")

    axB = axes[1]
    log_sigmas = np.log10(sigmas)
    extent = (log_sigmas[0], log_sigmas[-1], 0.5, k_max + 0.5)
    im = axB.imshow(H, aspect="auto", interpolation="nearest", cmap="viridis",
                    origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    axB.axhline(3.0, ls="--", color="#FFFFFF", lw=1.6, alpha=0.95)
    axB.text(log_sigmas[0] + 0.05, 3.0, r"$k^{\star}=3$", color="#FFFFFF",
             fontsize=9, va="bottom")
    tick_sigmas = np.array([0.05, 0.1, 0.3, 1.0, 3.0])
    axB.set_xticks(np.log10(tick_sigmas))
    axB.set_xticklabels([f"{s:g}" for s in tick_sigmas])
    axB.set_yticks(range(1, k_max + 1))
    axB.set_xlabel(r"Noise $\sigma$"); axB.set_ylabel(r"Selected $\hat{k}$")
    cb = fig.colorbar(im, ax=axB, fraction=0.06, pad=0.04)
    cb.set_label(r"Fraction (per $\sigma$)"); cb.set_ticks([0.0, 0.5, 1.0])
    add_panel_label(axB, "B", r"Distribution of $\hat{k}$")

    axC = axes[2]
    axC.fill_between(sigmas, mse_q1, mse_q3, alpha=0.25, color=COLORS["red"], linewidth=0)
    axC.plot(sigmas, mse_med, "s-", color=COLORS["red"], markersize=5, lw=2.0, label="BayesBreak")
    ref = (sigmas ** 2) / avg_len
    axC.plot(sigmas, ref, ls="--", lw=1.5, color=COLORS["black"], alpha=0.6,
             label=r"oracle $\sigma^2/\bar{m}$")
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel(r"Noise $\sigma$"); axC.set_ylabel("Signal MSE")
    axC.grid(True, which="major", ls="-", alpha=0.2, color=COLORS["grey"])
    axC.legend(loc="lower right", fontsize=9)
    add_panel_label(axC, "C", "Reconstruction error")

    save_figure(fig, out_dir / "figA2_snr_sensitivity", formats=("pdf", "png"))
    plt.close(fig)


def render_figA3(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.0)

    cfg = _load_json(_DATA_DIR / "figA3_multivariate_shared.json")
    ch_rows = _load_csv(_require_csv("figA3_channels.csv"))
    bd_rows = _load_csv(_require_csv("figA3_boundary.csv"))

    n = cfg["n"]; d = cfg["d"]
    true_boundaries = cfg["true_boundaries"]
    channel_sigmas = cfg["channel_sigmas"]

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.3], hspace=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes[:-1]:
        ax.sharex(axes[-1])
        plt.setp(ax.get_xticklabels(), visible=False)

    ch_labels = [rf"Channel 1 ($\sigma$={channel_sigmas[0]})",
                 rf"Channel 2 ($\sigma$={channel_sigmas[1]})",
                 rf"Channel 3 ($\sigma$={channel_sigmas[2]})"]
    panel_labels = ["A", "B", "C", "D"]

    ch_data: dict[int, dict[str, list]] = {c: {"x": [], "y": [], "mu": [], "sh": [], "ind": []}
                                            for c in range(d)}
    for r in ch_rows:
        c = int(r["channel"])
        ch_data[c]["x"].append(_nan(r["x"]))
        ch_data[c]["y"].append(_nan(r["y"]))
        ch_data[c]["mu"].append(_nan(r["mu_true"]))
        ch_data[c]["sh"].append(_nan(r["shared_fit"]))
        ch_data[c]["ind"].append(_nan(r["indep_fit"]))

    for c in range(d):
        ax = axes[c]
        t = np.array(ch_data[c]["x"])
        ax.scatter(t, ch_data[c]["y"], s=6, alpha=0.3, color=COLORS["grey"],
                   edgecolors="none", zorder=1)
        ax.plot(t, ch_data[c]["mu"], lw=1.5, ls="--", color=COLORS["black"],
                zorder=2, label="True" if c == 0 else None)
        ax.plot(t, ch_data[c]["sh"], lw=2, color=COLORS["blue"],
                label="Shared" if c == 0 else None, zorder=3)
        ax.plot(t, ch_data[c]["ind"], lw=1.5, color=COLORS["red"], ls=":",
                label="Independent" if c == 0 else None, zorder=3)
        ax.set_ylabel(ch_labels[c], fontsize=9)
        ax.set_xlim(0, n)
        for tb in true_boundaries[1:-1]:
            ax.axvline(tb, color=COLORS["grey"], ls=":", lw=0.8, alpha=0.5)
        if c == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=3)
        add_panel_label(ax, panel_labels[c])

    ax = axes[3]
    positions = [int(r["position"]) for r in bd_rows]
    bm_sh = [_nan(r["shared_prob"]) for r in bd_rows]
    ax.fill_between(positions, 0, bm_sh, alpha=0.35, color=COLORS["blue"], linewidth=0)
    ax.plot(positions, bm_sh, lw=2, color=COLORS["blue"], label="Shared")
    ch_colors = [COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for c, col in enumerate(ch_colors):
        ax.plot(positions, [_nan(r[f"ch{c}_prob"]) for r in bd_rows],
                lw=1.2, color=col, alpha=0.7, label=f"Ch {c + 1} indep.")
    for tb in true_boundaries[1:-1]:
        ax.axvline(tb, color=COLORS["red"], ls="--", lw=1.5,
                   label="True" if tb == true_boundaries[1] else None)
    ax.set_xlabel("Time index"); ax.set_ylabel("Boundary prob.")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, n)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    add_panel_label(ax, panel_labels[3])

    save_figure(fig, out_dir / "figA3_multivariate_shared", formats=("pdf", "png"))
    plt.close(fig)


def render_figA4(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.0)

    cfg = _load_json(_DATA_DIR / "figA4_model_selection.json")
    rows = _load_csv(_require_csv("figA4_posteriors.csv"))

    k_max = cfg["k_max"]; k_true = cfg["k_true"]
    scenarios = cfg["scenarios"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["red"]]

    k_grid = np.arange(1, k_max + 1)
    data_by_scenario: dict[int, np.ndarray] = {}
    for r in rows:
        si = int(r["scenario_idx"])
        ki = int(r["k"]) - 1
        if si not in data_by_scenario:
            data_by_scenario[si] = np.zeros(k_max)
        if 0 <= ki < k_max:
            data_by_scenario[si][ki] = _nan(r["prob"])

    y_top = max(d.max() for d in data_by_scenario.values()) * 1.28

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        pk = data_by_scenario.get(idx, np.zeros(k_max))
        k_mode = int(k_grid[int(np.argmax(pk))])
        p_mode = float(pk.max())
        ax.bar(k_grid, pk, color=colors[idx], alpha=0.88, edgecolor="white",
               linewidth=0.6, width=0.78, zorder=3)
        ax.axvline(k_true, ls="--", lw=1.4, color=COLORS["black"], alpha=0.85, zorder=2)
        ax.scatter([k_mode], [p_mode + 0.02 * y_top], marker="v", s=60,
                   color=COLORS["black"], zorder=5)
        ax.text(k_mode, p_mode + 0.08 * y_top, rf"$\hat k = {k_mode}$",
                ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Number of segments $k$")
        if idx == 0:
            ax.set_ylabel(r"$P(k \mid y)$")
        ax.set_xlim(0.4, k_max + 0.6); ax.set_ylim(0.0, y_top)
        ax.set_xticks(k_grid)
        ax.grid(True, axis="y", ls="-", alpha=0.18, color=COLORS["grey"])
        sigma = sc["sigma"]
        ax.text(0.97, 0.92, rf"$\sigma = {sigma}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": COLORS["lightgrey"], "linewidth": 0.8})
        add_panel_label(ax, chr(65 + idx), title=f"  {sc['label']}")
        if idx == 0:
            ax.text(k_true + 0.18, y_top * 0.04, r"true $k^{\star}=3$",
                    fontsize=9, color=COLORS["black"], rotation=90,
                    va="bottom", ha="left", alpha=0.85)

    save_figure(fig, out_dir / "figA4_model_selection", formats=("pdf", "png"))
    plt.close(fig)


def render_figA5(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, get_figsize, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=0.95)

    cfg = _load_json(_DATA_DIR / "figA5_missing_data.json")
    sig_rows = _load_csv(_require_csv("figA5_signal.csv"))
    bd_rows = _load_csv(_require_csv("figA5_boundary.csv"))

    n = cfg["n"]; true_b = cfg["true_boundaries"]
    patterns = [
        ("random", "is_missing_random", "pc_rand", "d1_rand", "Random missing"),
        ("block", "is_missing_block", "pc_block", "d1_block", "Block missing"),
        ("periodic", "is_missing_periodic", "pc_periodic", "d1_periodic", "Periodic missing"),
    ]

    x = np.array([int(r["x"]) for r in sig_rows])
    y_full = np.array([_nan(r["y_full"]) for r in sig_rows])
    mu_true = np.array([_nan(r["mu_true"]) for r in sig_rows])
    d1_full = np.array([_nan(r["d1_full"]) for r in bd_rows])
    positions = np.array([int(r["position"]) for r in bd_rows])

    figsize = get_figsize("double", aspect=0.65, nrows=3, ncols=2)
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex="col")
    panel_left = ["A", "C", "E"]; panel_right = ["B", "D", "F"]

    for row, (pname, miss_col, pc_col, d1_col, label) in enumerate(patterns):
        miss_mask = np.array([int(_nan(r[miss_col])) == 1 for r in sig_rows])
        obs_mask = ~miss_mask
        pc_w = np.array([_nan(r[pc_col]) for r in sig_rows])
        d1_w = np.array([_nan(r[d1_col]) for r in bd_rows])

        ax = axes[row, 0]
        ax.scatter(x[obs_mask], y_full[obs_mask], s=12, alpha=0.5, color=COLORS["grey"],
                   edgecolors="none", zorder=1, label="Observed" if row == 0 else None)
        ax.scatter(x[miss_mask], y_full[miss_mask], s=18, marker="x", alpha=0.5,
                   color=COLORS["red"], zorder=1, label="Missing" if row == 0 else None)
        ax.plot(x, mu_true, ls="--", lw=1.5, color=COLORS["black"], zorder=2)
        ax.plot(x, pc_w, lw=2, color=COLORS["blue"], zorder=3)
        ax.set_ylabel(label, fontsize=9); ax.set_xlim(0, n)
        if row == 0:
            ax.legend(loc="upper right", fontsize=9, ncol=2)
        if row == 2:
            ax.set_xlabel("Time index")
        add_panel_label(ax, panel_left[row])

        ax = axes[row, 1]
        ax.fill_between(positions, 0, d1_full, alpha=0.2, color=COLORS["blue"],
                        linewidth=0, label="Full data" if row == 0 else None)
        ax.plot(positions, d1_w, lw=2, color=COLORS["blue"],
                label="Weighted" if row == 0 else None)
        for tb in true_b:
            ax.axvline(tb, color=COLORS["red"], ls="--", lw=1.2, alpha=0.6)
        ax.set_ylim(0, 1.05); ax.set_xlim(0, n)
        if row == 0:
            ax.legend(loc="upper right", fontsize=9)
        if row == 2:
            ax.set_xlabel("Time index")
        ax.set_ylabel("Boundary prob.")
        add_panel_label(ax, panel_right[row])

    save_figure(fig, out_dir / "figA5_missing_data", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Static figure: latent template stress
# ---------------------------------------------------------------------------

def render_latent_stress(out_dir: Path) -> None:
    from _style import COLORS, add_panel_label, save_figure, setup_style
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(font_scale=1.0)

    rows = _load_csv(_require_csv("fig_latent_template_stress.csv"))
    G_vals = sorted({int(r["G"]) for r in rows})
    S_vals = sorted({int(r["S"]) for r in rows})
    Delta_vals = sorted({_nan(r["Delta"]) for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]
    markers = ["o", "s", "D"]

    for ax_idx, (y_col, ci_lo_col, ci_hi_col, ylabel) in enumerate([
        ("mean_accuracy", "accuracy_ci_lo", "accuracy_ci_hi", "Accuracy"),
        ("mean_dup_template", "dup_template_ci_lo", "dup_template_ci_hi", "Dup. template rate"),
    ]):
        ax = axes[ax_idx]
        for gi, G in enumerate(G_vals):
            for si, S in enumerate(S_vals):
                sub = sorted([r for r in rows if int(r["G"]) == G and int(r["S"]) == S],
                             key=lambda r: _nan(r["Delta"]))
                if not sub:
                    continue
                deltas = [_nan(r["Delta"]) for r in sub]
                ys = [_nan(r[y_col]) for r in sub]
                ci_lo = [_nan(r.get(ci_lo_col, "nan")) for r in sub]
                ci_hi = [_nan(r.get(ci_hi_col, "nan")) for r in sub]
                label = f"G={G} S={S}"
                ci_arr = np.array(list(zip(
                    np.array(ys) - np.array(ci_lo),
                    np.array(ci_hi) - np.array(ys)
                ))).T
                ax.errorbar(deltas, ys, yerr=ci_arr,
                            color=colors[gi % len(colors)],
                            marker=markers[si % len(markers)],
                            lw=1.2, markersize=5, label=label, capsize=3)
        if ax_idx == 0:
            ax.axhline(0.5, ls="--", color=COLORS["grey"], lw=1)
        ax.set_xlabel(r"$\Delta$ (template separation)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        add_panel_label(ax, chr(65 + ax_idx))

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_latent_template_stress", formats=("pdf", "png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table renderers: generate LaTeX table bodies from cached CSVs
# ---------------------------------------------------------------------------

_TABLES_DATA_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "tables" / "data"
_TABLES_RESULTS_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "tables" / "results"


def _require_table_csv(name: str) -> Path:
    p = _TABLES_DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required table CSV not found: {p}\n"
            f"Run: python scripts/figures/prepare_data.py --only tables tab_extra"
        )
    return p


def _fmt_ci(mean_val: float, lo: float, hi: float, prec: int = 3) -> str:
    if not math.isfinite(mean_val):
        return "---"
    fmt = f"{{:.{prec}f}}"
    s = fmt.format(mean_val)
    if math.isfinite(lo) and math.isfinite(hi):
        s += f" [{fmt.format(lo)},{fmt.format(hi)}]"
    return s


def render_tab_E01_recovery(out_dir: Path) -> None:
    rows = _load_csv(_require_table_csv("tab_E01_recovery.csv"))
    geom_rows = [r for r in rows if "geometric" in r.get("prior", "")]
    unif_rows = [r for r in rows if "uniform" in r.get("prior", "")]

    lines: list[str] = []
    lines.append(r"\begin{tabular}{ccccccccc}")
    lines.append(r"\toprule")
    lines.append(r"$n$ & $k_{\rm true}$ & $\Delta/\sigma$ & Prior & F1@3 & "
                 r"P($k$=true) & Count MAE & Brier & Signal MSE \\")
    lines.append(r"\midrule")

    def _emit(subset: list[dict], prior_label: str) -> None:
        n_vals = sorted({int(r["n"]) for r in subset})
        for nv in n_vals:
            n_rows = sorted([r for r in subset if int(r["n"]) == nv],
                            key=lambda r: (_nan(r["k_true"]), _nan(r["delta"])))
            for r in n_rows:
                kt = int(float(r["k_true"]))
                delta = _nan(r["delta"])
                f1 = _nan(r["mean_f1"])
                f1_lo = _nan(r.get("f1_ci_lo", "nan"))
                f1_hi = _nan(r.get("f1_ci_hi", "nan"))
                pk = _nan(r["mean_p_k_true_given_y"])
                pk_lo = _nan(r.get("p_k_true_given_y_ci_lo", "nan"))
                pk_hi = _nan(r.get("p_k_true_given_y_ci_hi", "nan"))
                mae = _nan(r.get("mean_count_abs_error", "nan"))
                mae_lo = _nan(r.get("count_abs_error_ci_lo", "nan"))
                mae_hi = _nan(r.get("count_abs_error_ci_hi", "nan"))
                brier = _nan(r.get("mean_brier", "nan"))
                br_lo = _nan(r.get("brier_ci_lo", "nan"))
                br_hi = _nan(r.get("brier_ci_hi", "nan"))
                mse = _nan(r.get("mean_signal_mse", "nan"))
                mse_lo = _nan(r.get("signal_mse_ci_lo", "nan"))
                mse_hi = _nan(r.get("signal_mse_ci_hi", "nan"))

                if kt == 1 and delta == 0.0:
                    fp = _nan(r.get("mean_n_false_at_3", "nan"))
                    f1_str = f"FP={fp:.3f}"
                else:
                    f1_str = _fmt_ci(f1, f1_lo, f1_hi, 3)

                lines.append(
                    f"{nv} & {kt} & {delta:.1f} & {prior_label} & "
                    f"{f1_str} & {_fmt_ci(pk, pk_lo, pk_hi, 3)} & "
                    f"{_fmt_ci(mae, mae_lo, mae_hi, 2)} & "
                    f"{_fmt_ci(brier, br_lo, br_hi, 3)} & "
                    f"{_fmt_ci(mse, mse_lo, mse_hi, 3)} \\\\"
                )
            lines.append(r"\midrule")

    ablation_false_geom = [r for r in geom_rows if str(r.get("ablation", "")).lower() == "false"]
    ablation_true_geom = [r for r in geom_rows if str(r.get("ablation", "")).lower() == "true"]
    _emit(ablation_false_geom, "Geom.")
    _emit(ablation_true_geom, "Unif.")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    _TABLES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _TABLES_RESULTS_DIR / "tab_E01_recovery.tex"
    caption = (r"E01: Gaussian recovery (primary + ablation). Geometric-decay $p(k)$ prior. "
               r"Mean [95\% bootstrap CI] over 100 independent datasets per cell. "
               r"F1 at boundary tolerance $\tau=3$; null cells report expected false-boundary count.")
    label = "tab:E01-recovery-paper"
    _write_latex_table(out, lines, caption, label, "footnotesize")
    logger.info("Rendered tab_E01_recovery → %s", out)


def render_tab_E02_recovery(out_dir: Path) -> None:
    rows = _load_csv(_require_table_csv("tab_E02_recovery.csv"))
    family_order = ["gaussian", "poisson", "binomial", "negbin", "beta_obs"]
    family_labels = {
        "gaussian": "Gaussian", "poisson": "Poisson",
        "binomial": "Binomial", "negbin": "NegBin", "beta_obs": "Beta-obs",
    }

    lines: list[str] = []
    lines.append(r"\begin{tabular}{llccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Family & Param & $n$ & F1@3 & P($k$=3) & Count MAE & Signal MSE & Brier \\")
    lines.append(r"\midrule")

    current_family: str | None = None
    for fam in family_order:
        fam_rows = sorted([r for r in rows if r.get("family", "") == fam or
                           (fam == "beta_obs" and r.get("family", "") in ("beta", "beta_obs"))],
                          key=lambda r: _nan(r.get("value", r.get("param", "0"))))
        if not fam_rows:
            continue
        if current_family is not None:
            lines.append(r"\midrule")
        current_family = fam
        for r in fam_rows:
            param_val = r.get("param", r.get("value", ""))
            try:
                param_str = f"$\\Delta$={float(param_val):.2f}" if fam == "gaussian" else \
                            f"$q$={float(param_val):.2f}" if fam == "poisson" else \
                            f"$p_{{\\rm mid}}$={float(param_val):.2f}" if fam in ("binomial",) else \
                            f"$\\mu_{{\\rm mid}}$={float(param_val):.2f}" if fam in ("negbin", "beta_obs") else \
                            str(param_val)
            except (ValueError, TypeError):
                param_str = str(param_val)
            n = int(float(r["n"]))
            f1 = _nan(r.get("mean_f1", "nan"))
            f1_lo = _nan(r.get("f1_ci_lo", "nan")); f1_hi = _nan(r.get("f1_ci_hi", "nan"))
            pk = _nan(r.get("mean_p_k_true_given_y", "nan"))
            pk_lo = _nan(r.get("p_k_true_given_y_ci_lo", "nan"))
            pk_hi = _nan(r.get("p_k_true_given_y_ci_hi", "nan"))
            mae = _nan(r.get("mean_count_abs_error", "nan"))
            mae_lo = _nan(r.get("count_abs_error_ci_lo", "nan"))
            mae_hi = _nan(r.get("count_abs_error_ci_hi", "nan"))
            mse = _nan(r.get("mean_signal_mse", "nan"))
            mse_lo = _nan(r.get("signal_mse_ci_lo", "nan"))
            mse_hi = _nan(r.get("signal_mse_ci_hi", "nan"))
            brier = _nan(r.get("mean_brier", "nan"))
            br_lo = _nan(r.get("brier_ci_lo", "nan")); br_hi = _nan(r.get("brier_ci_hi", "nan"))
            lines.append(
                f"{family_labels.get(fam, fam)} & {param_str} & {n} & "
                f"{_fmt_ci(f1, f1_lo, f1_hi, 3)} & {_fmt_ci(pk, pk_lo, pk_hi, 3)} & "
                f"{_fmt_ci(mae, mae_lo, mae_hi, 2)} & "
                f"{_fmt_ci(mse, mse_lo, mse_hi, 3)} & "
                f"{_fmt_ci(brier, br_lo, br_hi, 3)} \\\\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = _TABLES_RESULTS_DIR / "tab_E02_recovery.tex"
    caption = (r"E02: Observation-family portability. $k_{\rm true}=3$, exact equal segment lengths, "
               r"geometric-decay $p(k)$ prior, $k_{\max}=10$. Mean [95\% bootstrap CI] over "
               r"100 (conjugate) or 50 (Beta-obs) independent datasets per cell. F1 at boundary "
               r"tolerance $\tau=3$.")
    _write_latex_table(out, lines, caption, "tab:E02-recovery-paper", "footnotesize")
    logger.info("Rendered tab_E02_recovery → %s", out)


def render_tab_E05_latent(out_dir: Path) -> None:
    rows = _load_csv(_require_table_csv("tab_E05_latent.csv"))
    rows_sorted = sorted(rows, key=lambda r: (int(float(r["G"])), int(float(r["S"])),
                                               _nan(r["Delta"])))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{@{}lccccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"$G$ & $S$ & $\Delta$ & $n$ & Accuracy (95\,\% CI) & "
                 r"ARI (95\,\% CI) & Dup.\ freq. & Mon.\ fail \\")
    lines.append(r"\midrule")

    for r in rows_sorted:
        G = int(float(r["G"])); S = int(float(r["S"]))
        Delta = _nan(r["Delta"]); n_rep = int(float(r.get("n_replicates", 50)))
        acc = _nan(r.get("mean_accuracy", "nan"))
        acc_lo = _nan(r.get("accuracy_ci_lo", "nan"))
        acc_hi = _nan(r.get("accuracy_ci_hi", "nan"))
        ari = _nan(r.get("mean_ari", "nan"))
        ari_lo = _nan(r.get("ari_ci_lo", "nan"))
        ari_hi = _nan(r.get("ari_ci_hi", "nan"))
        dup = _nan(r.get("mean_has_duplicate_templates", "nan"))
        mon = int(float(r.get("n_monotone_failures", 0)))

        acc_str = _fmt_ci(acc, acc_lo, acc_hi, 3)
        ari_str = "--" if not math.isfinite(ari) else _fmt_ci(ari, ari_lo, ari_hi, 3)
        dup_str = f"{dup:.3f}" if math.isfinite(dup) else "---"

        lines.append(
            f"${G}$ & ${S}$ & ${Delta:.1f}$ & ${n_rep}$ & "
            f"${acc_str}$ & ${ari_str}$ & ${dup_str}$ & ${mon}$ \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = _TABLES_RESULTS_DIR / "tab_E05_latent.tex"
    caption = (r"E05 latent-template score clustering. Accuracy is label-permutation-invariant "
               r"hard assignment accuracy; ARI is adjusted Rand index (undefined for $G=1$, "
               r"shown as --). Bootstrap 95\% percentile intervals from 2000 resamples of "
               r"independently generated datasets. Dup.\ freq.\ is the fraction of replicates "
               r"where two groups recovered identical templates.")
    _write_latex_table(out, lines, caption, "tab:E05-latent", "small")
    logger.info("Rendered tab_E05_latent → %s", out)


def render_tab_E10_scaling(out_dir: Path) -> None:
    rows = _load_csv(_require_table_csv("tab_E10_scaling.csv"))
    ok = [r for r in rows if str(r.get("failed", "")).lower() != "true"]
    ok_sorted = sorted(ok, key=lambda r: (int(r["n"]), int(r["k_max"])))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{rrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$n$ & $k_{\max}$ & $k_{\rm map}$ & Block (s) & "
                 r"Fwd-bwd (s) & E2E (s) & Mem (MB)\\")
    lines.append(r"\midrule")

    for r in ok_sorted:
        n = int(r["n"]); km = int(r["k_max"]); kmap = int(float(r["k_map"]))
        be = _nan(r.get("block_evidence_median_s", "nan"))
        fb = _nan(r.get("forward_backward_median_s", "nan"))
        e2e = _nan(r.get("end_to_end_median_s", "nan"))
        mem = _nan(r.get("peak_memory_mb", "nan"))
        lines.append(
            f"{n} & {km} & {kmap} & "
            f"{be:.4f} & {fb:.4f} & {e2e:.4f} & "
            f"{mem:.1f}\\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = _TABLES_RESULTS_DIR / "tab_E10_scaling.tex"
    _write_latex_table(out, lines, "", "tab:E10-scaling-paper", size_cmd=None)
    logger.info("Rendered tab_E10_scaling → %s", out)


def render_tab_E07_robustness(out_dir: Path) -> None:
    rows = _load_csv(_require_table_csv("tab_E07_robustness.csv"))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\hline")
    lines.append(r"Data family & Fit & Missingness & F1@3 & Count exact & "
                 r"$p(k^\star|y)$ & Brier$_k$ \\")
    lines.append(r"\hline")

    for r in rows:
        df = r.get("data_family", "").replace("_", " ")
        ff = r.get("fit_family", "")
        is_correct = str(r.get("is_correct", "1")).lower() not in ("0", "false")
        miss = r.get("missingness", "none").replace("_", " ")
        f1 = _nan(r.get("mean_f1", "nan"))
        ci_col = r.get("f1_ci_lo", ""); ci_hi_col = r.get("f1_ci_hi", "")
        n_reps = int(float(r.get("n_reps", r.get("n_replicates", 100))))
        # Approx CI from the stored mean_f1 (±0 if no CI columns)
        f1_str = f"{f1:.3f}"

        # Need count_exact and brier from the wider CSV data
        count_exact_col = r.get("mean_count_exact", r.get("count_exact", "nan"))
        brier_col = r.get("mean_brier", r.get("brier", "nan"))
        pk_col = r.get("mean_p_k_true_given_y", r.get("p_k_true_given_y", "nan"))
        dag = "" if is_correct else r"$\dag$"
        lines.append(
            f"{df} & {ff}{dag} & {miss} & "
            f"{_nan(count_exact_col):.3f} & {_nan(pk_col):.3f} & "
            f"{_nan(brier_col):.4f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\multicolumn{7}{l}{\small$\dag$ misspecified fit.} \\")
    lines.append(r"\end{tabular}")

    out = _TABLES_RESULTS_DIR / "tab_E07_robustness.tex"
    caption = (r"E07 misspecification and missing data: boundary recovery F1@3 (mean $\pm$ 95\% CI), "
               r"posterior count accuracy, and Brier score for $k=k_\mathrm{true}$.")
    _write_latex_table(out, lines, caption, "tab:E07-robustness", "footnotesize")
    logger.info("Rendered tab_E07_robustness → %s", out)


def _write_latex_table(out: Path, body_lines: list[str], caption: str, label: str,
                       size_cmd: str | None = "footnotesize") -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [r"\begin{table}[!htbp]", r"\centering"]
    if size_cmd:
        parts.append(f"\\{size_cmd}")
    if caption:
        parts.append(f"\\caption{{{caption}}}")
    parts.append(f"\\label{{{label}}}")
    parts.extend(body_lines)
    parts.append(r"\end{table}")
    with open(out, "w") as f:
        f.write("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, object] = {
    "fig1": render_fig1,
    "fig2": render_fig2,
    "fig3": render_fig3,
    "fig4": render_fig4,
    "fig5": render_fig5,
    "fig6": render_fig6,
    "fig7": render_fig7,
    "fig9": render_fig9,
    "fig_E01_calibration": render_e01_calibration,
    "fig_E01_recovery": render_e01_recovery,
    "fig_E02_recovery": render_e02_recovery,
    "fig_E03_gap_true": render_e03_gap_true,
    "fig_E03_gap_false": render_e03_gap_false,
    "fig_E04_pooled": render_e04_pooled,
    "fig_E05_templates": render_e05_templates,
    "fig_E06_local_error": render_e06_local_error,
    "fig_E06_global_error": render_e06_global_error,
    "fig_E07_robustness": render_e07_robustness,
    "fig_E10_scaling": render_e10_scaling,
    "fig_E10_stage_breakdown": render_e10_stage_breakdown,
    "figA1": render_figA1,
    "figA2": render_figA2,
    "figA3": render_figA3,
    "figA4": render_figA4,
    "figA5": render_figA5,
    "fig_latent_stress": render_latent_stress,
    "tab_E01_recovery": render_tab_E01_recovery,
    "tab_E02_recovery": render_tab_E02_recovery,
    "tab_E05_latent": render_tab_E05_latent,
    "tab_E10_scaling": render_tab_E10_scaling,
    "tab_E07_robustness": render_tab_E07_robustness,
}

_ALL_ORDER = [
    "fig1", "fig2", "fig3", "fig4", "fig5",
    "fig6", "fig7", "fig9",
    "fig_E01_calibration", "fig_E01_recovery",
    "fig_E02_recovery",
    "fig_E03_gap_true", "fig_E03_gap_false",
    "fig_E04_pooled",
    "fig_E05_templates",
    "fig_E06_local_error", "fig_E06_global_error",
    "fig_E07_robustness",
    "fig_E10_scaling", "fig_E10_stage_breakdown",
    "figA1", "figA2", "figA3", "figA4", "figA5",
    "fig_latent_stress",
    "tab_E01_recovery", "tab_E02_recovery",
    "tab_E05_latent", "tab_E10_scaling", "tab_E07_robustness",
]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Render all manuscript figures from cached CSV data. "
                    "Zero scientific computation."
    )
    ap.add_argument("--all", action="store_true", help="Render all figures")
    ap.add_argument("--only", nargs="+", metavar="ID",
                    help="Render specific figure(s) by registry key")
    ap.add_argument("--outdir", type=Path, default=_OUT_DIR,
                    help="Output directory for PDF/PNG files")
    args = ap.parse_args(argv)

    if not args.all and not args.only:
        ap.error("Specify --all or --only <IDs>")

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _ALL_ORDER if args.all else args.only
    t_start = time.perf_counter()
    n_ok = n_fail = 0

    for name in targets:
        fn = _REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown figure ID %r — skipping", name)
            continue
        logger.info("Rendering %s ...", name)
        t0 = time.perf_counter()
        try:
            fn(out_dir)  # type: ignore[call-arg]
            elapsed = time.perf_counter() - t0
            logger.info("  OK  %s  (%.2fs)", name, elapsed)
            n_ok += 1
        except FileNotFoundError as exc:
            logger.error("  MISS %s: %s", name, exc)
            n_fail += 1
        except Exception as exc:
            logger.error("  FAIL %s: %s", name, exc, exc_info=True)
            n_fail += 1

    total = time.perf_counter() - t_start
    logger.info("Done: %d rendered, %d failed — total %.2fs", n_ok, n_fail, total)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
