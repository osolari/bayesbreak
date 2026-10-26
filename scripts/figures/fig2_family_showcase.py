r"""Figure 2: Likelihood family showcase.

This script demonstrates BayesBreak on four short synthetic series, one per
supported conjugate family:

- **Gaussian** (Normal--Normal conjugate):  continuous real-valued observations
  with known noise variance and a normal prior on the segment mean.
- **Poisson** (Gamma--Poisson conjugate):  non-negative count observations
  with a Gamma prior on the segment rate :math:`\lambda`.
- **Binomial** (Beta--Binomial conjugate):  success counts out of a known
  number of trials with a Beta prior on the segment probability.
- **Beta-valued** (fractional Beta--Binomial):  continuous :math:`(0,1)`-valued
  observations treated as fractional pseudo-counts of a Beta--Binomial.

Experiment
----------
For each family a three-segment piecewise-constant latent parameter sequence
(:math:`n=120`, segments of 40 observations each) is generated.  Observations
are drawn from the corresponding likelihood, and BayesBreak is fit with
``k_max=10``.  No hyperparameter tuning is performed; defaults are used.

The resulting 2×2 grid shows, for every family:

* **Grey dots** – raw observations.
* **Dashed black line** – the *true* latent parameter sequence.
* **Solid blue line** – the BayesBreak piecewise-constant posterior estimate.

Interpretation
--------------
If BayesBreak is working correctly, the blue fit should closely overlay the
dashed true signal and should *not* introduce spurious changepoints.  The
panel titles indicate the family, so each sub-plot doubles as a visual
regression test: any systematic bias (e.g., the Poisson fit consistently
under-estimating high rates) would indicate a problem in the corresponding
family's block-evidence computation.

Outputs
-------
- results/figures/fig2_family_showcase.png
- results/figures/fig2_family_showcase.pdf

Usage
-----
python scripts/figures/fig2_family_showcase.py
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
    setup_style(font_scale=1.1)

    rng = np.random.default_rng(seed)

    # ----------------
    # Gaussian example
    # ----------------
    n = 120
    mu = np.r_[np.zeros(40), 1.5 * np.ones(40), -0.5 * np.ones(40)]
    y_gauss = mu + 0.35 * rng.standard_normal(n)
    m_gauss = BayesBreakGaussian(k_max=10).fit(np.arange(len(y_gauss)).reshape(-1, 1), y_gauss)
    pc_gauss = m_gauss.predict(m_gauss.x_design_.reshape(-1, 1))

    # ----------------
    # Poisson example
    # ----------------
    lam = np.r_[2.0 * np.ones(40), 8.0 * np.ones(40), 3.0 * np.ones(40)]
    y_pois = rng.poisson(lam)
    m_pois = BayesBreakPoisson(k_max=10).fit(np.arange(len(y_pois)).reshape(-1, 1), y_pois)
    pc_pois = m_pois.predict(m_pois.x_design_.reshape(-1, 1))

    # ----------------
    # Binomial example
    # ----------------
    n_trials = 20
    p = np.r_[0.1 * np.ones(40), 0.7 * np.ones(40), 0.3 * np.ones(40)]
    y_binom = rng.binomial(n_trials, p)
    m_binom = BayesBreakBinomial(k_max=10, n_trials=n_trials).fit(
        np.arange(len(y_binom)).reshape(-1, 1), y_binom
    )
    pc_binom = m_binom.predict(m_binom.x_design_.reshape(-1, 1))

    # ----------------
    # Beta-valued example
    # ----------------
    kappa = 50
    p_beta = np.r_[0.2 * np.ones(40), 0.85 * np.ones(40), 0.4 * np.ones(40)]
    s = rng.binomial(kappa, p_beta)
    y_beta = (s + 0.5) / (kappa + 1.0)
    m_beta = BayesBreakBeta(k_max=10, concentration=kappa).fit(
        np.arange(len(y_beta)).reshape(-1, 1), y_beta
    )
    pc_beta = m_beta.predict(m_beta.x_design_.reshape(-1, 1))

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

    save_figure(fig, outdir / "fig2_family_showcase", formats=("png", "pdf"))
    plt.close(fig)

    import json as _json
    _data = {
        "seed": seed,
        "gaussian": {"y": y_gauss.tolist(), "true_signal": mu.tolist(), "fit": pc_gauss.tolist(),
                     "map_boundaries": list(m_gauss.map_boundaries_)},
        "poisson": {"y": y_pois.tolist(), "true_signal": lam.tolist(), "fit": pc_pois.tolist(),
                    "map_boundaries": list(m_pois.map_boundaries_)},
        "binomial": {"y": (y_binom / n_trials).tolist(), "true_signal": p.tolist(),
                     "fit": pc_binom.tolist(), "map_boundaries": list(m_binom.map_boundaries_)},
        "beta": {"y": y_beta.tolist(), "true_signal": p_beta.tolist(), "fit": pc_beta.tolist(),
                 "map_boundaries": list(m_beta.map_boundaries_)},
    }
    with open(outdir / "fig2_family_showcase_data.json", "w") as _f:
        _json.dump(_data, _f)

    # Write canonical plot-ready CSV to shared data dir.
    import csv as _csv
    _canonical = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "data"
    _canonical.mkdir(parents=True, exist_ok=True)
    with open(_canonical / "fig2_family_showcase_data.json", "w") as _f2:
        _json.dump(_data, _f2)
    _fam_order = ["gaussian", "poisson", "binomial", "beta"]
    _csv_rows = []
    for _fam in _fam_order:
        _fd = _data[_fam]
        _mb_set = set(list(_fd["map_boundaries"])[1:-1])
        for _x in range(len(_fd["y"])):
            _csv_rows.append({
                "family": _fam, "x": _x, "y": _fd["y"][_x],
                "true_signal": _fd["true_signal"][_x], "fit": _fd["fit"][_x],
                "is_map_boundary": 1 if _x in _mb_set else 0,
            })
    with open(_canonical / "fig2_family_showcase.csv", "w", newline="") as _cf:
        _w = _csv.DictWriter(_cf, fieldnames=list(_csv_rows[0].keys()))
        _w.writeheader()
        _w.writerows(_csv_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("results/figures"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.outdir, args.seed)
