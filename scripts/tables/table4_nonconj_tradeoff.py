"""Table 4: Non-conjugate approximation trade-offs (logistic-normal).

BayesBreak supports non-conjugate likelihoods via per-block evidence
approximations. For the Bernoulli / Binomial case with a logistic-normal
segment prior, the implementation offers several options:

* ``quadrature`` (Gauss--Hermite reference)
* ``laplace`` (Laplace approximation)
* ``jj`` (Jaakkola--Jordan variational bound)
* ``ep`` (Expectation Propagation)
* ``pg_vb`` (Polya--Gamma variational Bayes)

This script compares these methods on a small synthetic Bernoulli sequence:

1) The maximum absolute discrepancy in per-block log evidence relative to the
   quadrature reference.
2) End-to-end runtime.
3) Boundary F1 within tolerance ``tau``.

Outputs
-------
- results/table4_nonconj_tradeoff.csv
- results/table4_nonconj_tradeoff.md
- results/table4_nonconj_tradeoff.tex
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import BayesBreakLogisticNormal  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _boundary_f1(true_b: Iterable[int], pred_b: Iterable[int], tau: int) -> float:
    true = sorted(int(t) for t in true_b)
    pred = sorted(int(p) for p in pred_b)
    matched_true = set()
    tp = 0
    for p in pred:
        candidates = [t for t in true if t not in matched_true and abs(p - t) <= tau]
        if not candidates:
            continue
        t_best = min(candidates, key=lambda t: abs(p - t))
        matched_true.add(t_best)
        tp += 1
    fp = len(pred) - tp
    fn = len(true) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def main(outdir: Path, seed: int, n: int, k_max: int, tau: int, gh_points: int) -> None:
    rng = np.random.default_rng(seed)

    # Synthetic Bernoulli sequence with piecewise-constant log-odds.
    b_true = [0, n // 3, 2 * n // 3, n]
    theta_levels = np.array([-2.0, 1.2, -0.7], dtype=float)
    theta = np.repeat(theta_levels, [n // 3, n // 3, n - 2 * (n // 3)])
    p = _sigmoid(theta)
    y = rng.binomial(1, p).astype(float)

    # Reference (quadrature).
    t0 = time.perf_counter()
    ref = BayesBreakLogisticNormal(k_max=k_max, approx="quadrature", gh_points=gh_points).fit(y)
    t_ref = time.perf_counter() - t0
    lA0_ref = ref.lA0_
    if lA0_ref is None:
        raise RuntimeError("Reference fit did not store lA0_")

    true_interior = b_true[1:-1]

    methods = [
        ("quadrature", {"approx": "quadrature", "gh_points": gh_points}),
        ("laplace", {"approx": "laplace"}),
        ("jj", {"approx": "jj"}),
        ("ep", {"approx": "ep"}),
        ("pg_vb", {"approx": "pg_vb"}),
    ]

    rows = []

    # Record the reference row first.
    pred_b = ref.get_boundaries()[1:-1]
    f1 = _boundary_f1(true_interior, pred_b, tau=tau)
    rows.append(("quadrature", 0.0, t_ref, f1, int(ref.k_ml_)))

    # Compare approximations.
    for name, kwargs in methods[1:]:
        t0 = time.perf_counter()
        m = BayesBreakLogisticNormal(k_max=k_max, **kwargs).fit(y)
        t_fit = time.perf_counter() - t0

        lA0 = m.lA0_
        if lA0 is None:
            raise RuntimeError(f"{name} fit did not store lA0_")

        mask = np.isfinite(lA0_ref) & np.isfinite(lA0)
        max_abs = float(np.max(np.abs(lA0[mask] - lA0_ref[mask]))) if np.any(mask) else float("nan")

        pred_b = m.get_boundaries()[1:-1]
        f1 = _boundary_f1(true_interior, pred_b, tau=tau)
        rows.append((name, max_abs, t_fit, f1, int(m.k_ml_)))

    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "table4_nonconj_tradeoff.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("method,max_abs_block_log_evidence_error,fit_seconds,boundary_f1,k_sel\n")
        for name, max_abs, t_fit, f1, k_sel in rows:
            f.write(f"{name},{max_abs:.6f},{t_fit:.4f},{f1:.4f},{k_sel}\n")

    md_path = outdir / "table4_nonconj_tradeoff.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Method | max |Δ log A0| | fit time (s) | F1@tau | k_sel |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for name, max_abs, t_fit, f1, k_sel in rows:
            f.write(f"| {name} | {max_abs:.3f} | {t_fit:.3f} | {f1:.3f} | {k_sel} |\n")

    tex_path = outdir / "table4_nonconj_tradeoff.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\\toprule\n")
        f.write("Method & $\\max|\\Delta \\log A^0|$ & time (s) & F1@$\\tau$ & $\\hat{k}$\\\\\\midrule\n")
        for name, max_abs, t_fit, f1, k_sel in rows:
            f.write(f"{name} & {max_abs:.3f} & {t_fit:.3f} & {f1:.3f} & {k_sel}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--gh-points", type=int, default=80)
    args = ap.parse_args()

    main(outdir=args.outdir, seed=args.seed, n=args.n, k_max=args.k_max, tau=args.tau, gh_points=args.gh_points)
