"""Figure 4: Latent-group pooling (mixture) demonstration.

This script simulates multiple Gaussian sequences from two latent groups with
distinct changepoint locations and segment means. It then fits
:class:`bayesbreak.BayesBreakMixture` (an EM-like latent-group extension of
BayesBreak) and visualises:

1. **Left panel**: posterior responsibilities (group membership probabilities)
   for each sequence.
2. **Right panels**: group-level marginal boundary posteriors and group-level
   Bayesian regression curves.

Outputs
-------
- results/fig4_latent_groups.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakGaussian  # noqa: E402
from bayesbreak.mixture import BayesBreakMixture  # noqa: E402


def _make_piecewise_constant(n: int, boundaries: list[int], levels: list[float]) -> np.ndarray:
    if len(boundaries) != len(levels) + 1:
        raise ValueError("boundaries must have length len(levels)+1")
    x = np.empty(n, dtype=float)
    for a, b, m in zip(boundaries[:-1], boundaries[1:], levels):
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

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])

    # --- Responsibilities heatmap ---
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(r_plot, aspect="auto", interpolation="nearest")
    ax0.set_xlabel("group")
    ax0.set_ylabel("sequence (sorted)")
    ax0.set_title("Responsibilities")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # --- Group boundary posteriors ---
    ax1 = fig.add_subplot(gs[0, 1])
    x_b = np.arange(1, n)
    for g, st in enumerate(states):
        ax1.plot(x_b, st.boundary_post, lw=1, label=f"group {g}")
    ax1.set_title("Group boundary posterior")
    ax1.set_xlabel("index")
    ax1.set_ylabel("P(boundary | y)")
    ax1.legend(loc="best")

    # --- Group Bayesian regression curves ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(mu0, linestyle="--", lw=1, label="true group 0")
    ax2.plot(mu1, linestyle="--", lw=1, label="true group 1")
    for g, st in enumerate(states):
        if st.brc is not None:
            ax2.plot(st.brc, lw=2, label=f"Bayes curve (group {g})")
    ax2.set_title("Group Bayesian regression curves")
    ax2.set_xlabel("index")
    ax2.set_ylabel("signal")
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig4_latent_groups.png", dpi=200)
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
