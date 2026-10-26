"""Figure E10: computational scaling benchmark.

Loads from docs/manuscripts/shared/figures/data/fig_E10_data.csv (committed).
Does not re-run the benchmark.

Outputs
-------
- docs/manuscripts/shared/figures/results/fig_E10_scaling.{pdf,png}
- docs/manuscripts/shared/figures/results/fig_E10_stage_breakdown.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "figures"))

import numpy as np  # noqa: E402
from _style import (  # noqa: E402
    COLORS,
    _MUTED_PALETTE,
    setup_style,
    save_figure,
)

_DATA_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "data"
_OUT_DIR = _ROOT / "docs" / "manuscripts" / "shared" / "figures" / "results"


def _load_data(data_path: Path) -> list[dict]:
    with open(data_path) as f:
        rows = []
        for row in csv.DictReader(f):
            d = {k: (float(v) if v not in ("", "nan", "None", "True", "False") else v)
                 for k, v in row.items()}
            rows.append(d)
    return rows


def make_figures(rows: list[dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    setup_style(font_scale=0.95)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_values = sorted(set(int(r["n"]) for r in rows if r.get("failed") != "True"))
    k_max_values = sorted(set(int(r["k_max"]) for r in rows if r.get("failed") != "True"))

    def _get(n: int, k: int, col: str) -> float:
        for r in rows:
            if int(r["n"]) == n and int(r["k_max"]) == k:
                v = r.get(col, float("nan"))
                return float(v) if isinstance(v, (int, float)) and not isinstance(v, str) else float("nan")
        return float("nan")

    # --- Figure 1: E2E and memory vs n for each k_max ---
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax_t, ax_m = axes

    palette = [_MUTED_PALETTE[0], _MUTED_PALETTE[2], _MUTED_PALETTE[1]]
    for i, k in enumerate(k_max_values):
        color = palette[i % len(palette)]
        ns = np.array([n for n in n_values])
        e2e = np.array([_get(n, k, "end_to_end_median_s") for n in ns])
        mem = np.array([_get(n, k, "peak_memory_mb") for n in ns])
        valid = np.isfinite(e2e)
        if np.any(valid):
            ax_t.plot(ns[valid], e2e[valid], marker="o", color=color, lw=1.4,
                      label=f"k_max={k}")
        valid_m = np.isfinite(mem)
        if np.any(valid_m):
            ax_m.plot(ns[valid_m], mem[valid_m], marker="o", color=color, lw=1.4,
                      label=f"k_max={k}")

    ax_t.set_xlabel("n")
    ax_t.set_ylabel("End-to-end time (s)")
    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.legend(fontsize=8, frameon=False)
    ax_t.grid(True, alpha=0.25)

    ax_m.set_xlabel("n")
    ax_m.set_ylabel("Peak memory (MB)")
    ax_m.set_xscale("log", base=2)
    ax_m.set_yscale("log")
    ax_m.legend(fontsize=8, frameon=False)
    ax_m.grid(True, alpha=0.25)

    fig.tight_layout()
    save_figure(fig, out_dir / "fig_E10_scaling", formats=("pdf", "png"))
    plt.close(fig)

    # --- Figure 2: stage breakdown at k_max=20 ---
    k_ref = max(k_max_values)
    stages = [
        ("block_evidence_median_s", "Block evidence"),
        ("forward_backward_median_s", "Forward-backward"),
        ("posterior_k_median_s", "Posterior k"),
        ("boundary_marginals_median_s", "Boundary marginals"),
        ("map_backtrack_median_s", "MAP backtrack"),
    ]
    fig2, ax2 = plt.subplots(figsize=(7.0, 3.6))
    ns = np.array(n_values)
    for i, (col, label) in enumerate(stages):
        vals = np.array([_get(n, k_ref, col) for n in ns])
        valid = np.isfinite(vals)
        if np.any(valid):
            ax2.plot(ns[valid], vals[valid], marker="o",
                     color=_MUTED_PALETTE[i % len(_MUTED_PALETTE)], lw=1.4, label=label)
    ax2.set_xlabel("n")
    ax2.set_ylabel("Median time (s)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.legend(fontsize=8, frameon=False, ncol=2)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    save_figure(fig2, out_dir / "fig_E10_stage_breakdown", formats=("png",))
    plt.close(fig2)


def main(data_path: Path, out_dir: Path) -> None:
    rows = _load_data(data_path)
    make_figures(rows, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=_DATA_DIR / "fig_E10_data.csv")
    ap.add_argument("--outdir", type=Path, default=_OUT_DIR)
    args = ap.parse_args()
    main(args.data, args.outdir)
