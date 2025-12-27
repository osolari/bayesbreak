"""Figure 3: Calibration of boundary posterior probabilities.

This script tests whether the marginal posterior boundary probabilities
``p(b_i=1 | y)`` produced by BayesBreak are empirically calibrated under a
synthetic data-generating process (Gaussian segments).

Procedure
---------
1) Simulate many piecewise-constant Gaussian sequences with random changepoints.
2) Fit :class:`bayesbreak.BayesBreakGaussian` to each sequence.
3) Collect predicted probabilities and binary "is-boundary" labels.
4) Bin predictions and plot empirical frequency vs predicted probability.

Outputs
-------
- results/fig3_boundary_calibration.png
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


def _sample_boundaries(rng: np.random.Generator, n: int, k_true: int, min_len: int) -> list[int]:
    """Sample strictly increasing boundaries with a minimum segment length."""
    if k_true < 1:
        raise ValueError("k_true must be >= 1")
    if k_true == 1:
        return [0, n]

    # Sample segment lengths from a Dirichlet then round, enforcing min_len.
    # We retry until constraints are satisfied.
    for _ in range(10_000):
        props = rng.dirichlet(np.ones(k_true))
        lens = np.maximum(min_len, np.floor(props * n).astype(int))
        # Adjust to sum to n.
        diff = int(n - np.sum(lens))
        if diff > 0:
            # Add remaining length to the largest segments.
            for _ in range(diff):
                j = int(np.argmax(lens))
                lens[j] += 1
        elif diff < 0:
            # Remove from the largest segments while respecting min_len.
            for _ in range(-diff):
                j = int(np.argmax(lens))
                if lens[j] > min_len:
                    lens[j] -= 1
                else:
                    break
        if int(np.sum(lens)) != n:
            continue
        if np.all(lens >= min_len):
            b = [0]
            c = 0
            for L in lens:
                c += int(L)
                b.append(c)
            if b[-1] == n and len(b) == k_true + 1:
                return b
    raise RuntimeError("Failed to sample boundaries that satisfy constraints")


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Expected calibration error for binary labels."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        p_hat = float(np.mean(p[mask]))
        y_hat = float(np.mean(y[mask]))
        ece += float(np.mean(mask)) * abs(y_hat - p_hat)
    return float(ece)


def main(
    outdir: Path,
    seed: int,
    n: int,
    n_seq: int,
    k_max: int,
    sigma: float,
    min_seg_len: int,
    n_bins: int,
) -> None:
    rng = np.random.default_rng(seed)

    probs: list[float] = []
    labels: list[int] = []

    for s in range(n_seq):
        k_true = int(rng.integers(3, 7))  # 3..6 segments
        b = _sample_boundaries(rng, n=n, k_true=k_true, min_len=min_seg_len)
        # Latent means.
        mus = rng.normal(loc=0.0, scale=1.0, size=k_true)

        mu = np.empty(n, dtype=float)
        for q, (a, c) in enumerate(zip(b[:-1], b[1:])):
            mu[a:c] = float(mus[q])

        y = mu + sigma * rng.standard_normal(n)
        m = BayesBreakGaussian(k_max=k_max).fit(y)
        d1 = m.get_boundary_posteriors()  # length n-1, for i=1..n-1

        true_set = set(b[1:-1])  # interior true boundaries
        for i in range(1, n):
            probs.append(float(d1[i - 1]))
            labels.append(1 if i in true_set else 0)

    p = np.asarray(probs, dtype=float)
    yb = np.asarray(labels, dtype=int)

    # Bin and aggregate.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    p_bin = []
    y_bin = []
    n_bin = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        p_bin.append(float(np.mean(p[mask])))
        y_bin.append(float(np.mean(yb[mask])))
        n_bin.append(int(np.sum(mask)))

    ece = _ece(p, yb.astype(float), n_bins=n_bins)
    brier = float(np.mean((p - yb) ** 2))

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1, label="perfect calibration")
    ax.plot(p_bin, y_bin, marker="o", linewidth=1, label="BayesBreak")
    ax.set_xlabel("Predicted boundary probability")
    ax.set_ylabel("Empirical boundary frequency")
    ax.set_title("Boundary posterior calibration")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    ax.text(
        0.02,
        0.98,
        f"n_seq={n_seq}\nECE={ece:.3f}\nBrier={brier:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    fig.tight_layout()
    fig.savefig(outdir / "fig3_boundary_calibration.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-seq", type=int, default=80)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--min-seg-len", type=int, default=10)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()
    main(
        outdir=args.outdir,
        seed=args.seed,
        n=args.n,
        n_seq=args.n_seq,
        k_max=args.k_max,
        sigma=args.sigma,
        min_seg_len=args.min_seg_len,
        n_bins=args.n_bins,
    )
