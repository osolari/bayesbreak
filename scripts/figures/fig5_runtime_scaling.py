r"""Figure 5: Runtime scaling benchmark.

Two panels characterise the empirical runtime of
:class:`bayesbreak.BayesBreakGaussian`:

* **Panel A** -- log-log time vs. series length :math:`n` for two values of
  :math:`k_{\max}`. A dashed reference line shows the theoretical
  :math:`O(k_{\max}\, n^2)` curve, anchored to the smallest-:math:`n`
  measurement. The empirically fit slope is annotated.
* **Panel B** -- semilogy time vs. :math:`k_{\max}` at a fixed :math:`n`,
  showing the (expected) linear scaling in the segment-budget parameter.

The benchmark records wall-clock times via ``time.perf_counter`` and is
intended for *relative* comparisons across code changes; absolute timings
depend on hardware and Python build.

Outputs
-------
- results/figures/fig5_runtime_scaling.png
- results/figures/fig5_runtime_scaling.pdf
- results/figures/fig5_runtime_scaling.csv

Usage
-----
python scripts/figures/fig5_runtime_scaling.py [--repeats 10]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
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

logger = logging.getLogger(__name__)


def _fit_once(rng: np.random.Generator, n: int, k_max: int) -> float:
    mu = np.r_[np.zeros(n // 3), np.ones(n // 3), -0.5 * np.ones(n - 2 * (n // 3))]
    y = mu + 0.25 * rng.standard_normal(n)

    t0 = time.perf_counter()
    BayesBreakGaussian(k_max=k_max).fit(np.arange(len(y)).reshape(-1, 1), y)
    t1 = time.perf_counter()
    return t1 - t0


def _bench_grid(
    rng: np.random.Generator,
    ns: list[int],
    k_maxs: list[int],
    repeats: int,
) -> list[tuple[int, int, float, float]]:
    rows: list[tuple[int, int, float, float]] = []
    for k_max in k_maxs:
        for n in ns:
            times = [_fit_once(rng, n=n, k_max=k_max) for _ in range(repeats)]
            mean = float(np.mean(times))
            std = float(np.std(times, ddof=1)) if repeats > 1 else 0.0
            rows.append((n, k_max, mean, std))
            logger.info("n=%4d k_max=%3d mean=%.4fs std=%.4fs", n, k_max, mean, std)
    return rows


def _log_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Slope of a linear fit on log-log scale."""
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main(
    outdir: Path,
    seed: int,
    repeats: int,
    n_for_kmax: int,
) -> None:
    rng = np.random.default_rng(seed)

    setup_style(font_scale=1.05)

    # --- Panel A: time vs n at two k_max values ---
    ns = [50, 100, 200, 400, 800]
    k_maxs = [10, 20]
    rows_n = _bench_grid(rng, ns=ns, k_maxs=k_maxs, repeats=repeats)

    # --- Panel B: time vs k_max at fixed n ---
    ks = [5, 10, 20, 40, 80]
    rows_k: list[tuple[int, int, float, float]] = []
    for k in ks:
        times = [_fit_once(rng, n=n_for_kmax, k_max=k) for _ in range(repeats)]
        mean = float(np.mean(times))
        std = float(np.std(times, ddof=1)) if repeats > 1 else 0.0
        rows_k.append((n_for_kmax, k, mean, std))
        logger.info("n=%4d k_max=%3d mean=%.4fs std=%.4fs", n_for_kmax, k, mean, std)

    outdir.mkdir(parents=True, exist_ok=True)

    # Save CSV for reproducibility.
    csv_path = outdir / "fig5_runtime_scaling.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("sweep,n,k_max,mean_seconds,std_seconds\n")
        for n, k_max, m, s in rows_n:
            f.write(f"n,{n},{k_max},{m:.6f},{s:.6f}\n")
        for n, k_max, m, s in rows_k:
            f.write(f"k,{n},{k_max},{m:.6f},{s:.6f}\n")
    logger.info("Wrote %s", csv_path)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    colors = [COLORS["blue"], COLORS["red"]]
    markers = ["o", "s"]

    # --- Panel A: n sweep on log-log ---
    axA = axes[0]
    for idx, k_max in enumerate(k_maxs):
        xs = np.array([n for n, km, _, _ in rows_n if km == k_max], dtype=float)
        ys = np.array([m for n, km, m, _ in rows_n if km == k_max], dtype=float)
        es = np.array([s for n, km, _, s in rows_n if km == k_max], dtype=float)
        slope = _log_slope(xs, ys)
        axA.errorbar(
            xs,
            ys,
            yerr=es,
            marker=markers[idx],
            markersize=7,
            linewidth=2.0,
            color=colors[idx],
            ecolor=colors[idx],
            capsize=3.5,
            capthick=1.3,
            label=rf"$k_{{\max}}={k_max}$  (slope={slope:.2f})",
            zorder=3,
        )

    # Theoretical O(n^2) reference, anchored to the lowest-(n,k_max) point.
    anchor_x = float(min(ns))
    anchor_idx = next(i for i, (n, km, _, _) in enumerate(rows_n) if n == ns[0] and km == k_maxs[0])
    anchor_y = rows_n[anchor_idx][2]
    n_ref = np.array(ns, dtype=float)
    ref = anchor_y * (n_ref / anchor_x) ** 2
    axA.plot(
        n_ref,
        ref,
        ls="--",
        lw=1.5,
        color=COLORS["black"],
        alpha=0.55,
        label=r"reference $\propto n^{2}$",
        zorder=2,
    )

    axA.set_xscale("log", base=2)
    axA.set_yscale("log")
    axA.set_xticks(ns)
    axA.set_xticklabels([str(n) for n in ns])
    axA.set_xlabel("Series length $n$")
    axA.set_ylabel("Time (seconds)")
    axA.grid(True, which="major", linestyle="-", alpha=0.25, color=COLORS["grey"])
    axA.grid(True, which="minor", linestyle=":", alpha=0.15, color=COLORS["grey"])
    axA.legend(loc="upper left", fontsize=9)
    add_panel_label(axA, "A", title="  Runtime vs. series length")

    # --- Panel B: k_max sweep on semi-log y ---
    axB = axes[1]
    xs_k = np.array([k for _, k, _, _ in rows_k], dtype=float)
    ys_k = np.array([m for _, _, m, _ in rows_k], dtype=float)
    es_k = np.array([s for _, _, _, s in rows_k], dtype=float)
    slope_k = _log_slope(xs_k, ys_k)
    axB.errorbar(
        xs_k,
        ys_k,
        yerr=es_k,
        marker="D",
        markersize=7,
        linewidth=2.0,
        color=COLORS["green"],
        ecolor=COLORS["green"],
        capsize=3.5,
        capthick=1.3,
        label=rf"empirical (slope={slope_k:.2f})",
        zorder=3,
    )
    # Linear reference: t ~ k.
    anchor_k = xs_k[0]
    anchor_yk = ys_k[0]
    axB.plot(
        xs_k,
        anchor_yk * (xs_k / anchor_k),
        ls="--",
        lw=1.5,
        color=COLORS["black"],
        alpha=0.55,
        label=r"reference $\propto k_{\max}$",
        zorder=2,
    )
    axB.set_xscale("log", base=2)
    axB.set_yscale("log")
    axB.set_xticks(ks)
    axB.set_xticklabels([str(k) for k in ks])
    axB.set_xlabel(r"Segment budget $k_{\max}$")
    axB.set_ylabel("Time (seconds)")
    axB.grid(True, which="major", linestyle="-", alpha=0.25, color=COLORS["grey"])
    axB.grid(True, which="minor", linestyle=":", alpha=0.15, color=COLORS["grey"])
    axB.legend(loc="upper left", fontsize=9)
    add_panel_label(axB, "B", title=rf"  Runtime vs. $k_{{\max}}$  ($n={n_for_kmax}$)")

    save_figure(fig, outdir / "fig5_runtime_scaling", formats=("png", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--n-for-kmax", type=int, default=200, help="Fixed n used for the k_max sweep panel."
    )
    args = ap.parse_args()

    main(
        outdir=args.outdir,
        seed=args.seed,
        repeats=args.repeats,
        n_for_kmax=args.n_for_kmax,
    )
