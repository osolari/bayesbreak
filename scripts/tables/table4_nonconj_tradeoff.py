r"""Table 4: Non-conjugate approximation trade-offs (logistic-normal).

BayesBreak supports non-conjugate likelihoods via per-block evidence
approximations.  For the Bernoulli / Binomial case with a logistic-normal
segment prior, the implementation offers several options:

* ``quadrature`` — Gauss–Hermite numerical integration (high accuracy, slower).
* ``laplace`` — second-order Laplace approximation around the posterior mode.
* ``jj`` — Jaakkola–Jordan variational lower bound.
* ``ep`` — Expectation Propagation.
* ``pg_vb`` — Pólya–Gamma variational Bayes.

Experiment
----------
A synthetic Bernoulli sequence (:math:`n=80`, 3 segments) with piecewise-
constant log-odds :math:`\theta \in \{-2.0, 1.2, -0.7\}` is generated.  Each
approximation method is used to fit :class:`bayesbreak.BayesBreakLogisticNormal`
with ``k_max=10``.  The ``quadrature`` method (80 Gauss–Hermite points) serves
as the reference.

This script compares the methods on three axes:

1. **Block-evidence accuracy** — maximum absolute discrepancy in per-block log
   evidence :math:`\max_{(i,j)} |\log A^0_{ij} - \log A^0_{ij,\text{ref}}|`
   relative to the quadrature reference.  Smaller means the approximation
   introduces less distortion in the DP table.
2. **End-to-end runtime** — wall-clock fit time in seconds.
3. **Boundary F1@τ** — whether the downstream segmentation is affected by the
   approximation error.

Interpretation
--------------
- The ``quadrature`` row always shows zero discrepancy (it *is* the reference).
  Its runtime is the cost of high-accuracy integration.
- ``laplace`` and ``jj`` are typically the fastest methods.  If their F1
  matches quadrature, they are "good enough" for practical use.
- A method with large block-evidence error but unchanged F1 suggests the DP
  is robust to that level of noise in the evidence table.
- A method that changes :math:`\hat{k}` relative to quadrature is shifting the
  posterior over model complexity and should be treated with caution.

Outputs
-------
- docs/report/tables/table4_nonconj_tradeoff.csv
- docs/report/tables/table4_nonconj_tradeoff.md
- docs/report/tables/table4_nonconj_tradeoff.tex

Usage
-----
python scripts/tables/table4_nonconj_tradeoff.py [--gh-points 120]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from pathlib import Path

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


def _reachable_blocks_mask(n: int, k_max: int) -> np.ndarray:
    """Mask of blocks ``(i, j]`` reachable by some k-segmentation with ``k <= k_max``.

    Mirrors the §`prop:stability` reachability convention: a block is
    reachable iff ``i + (n - j) + 1 <= k_max`` so that there is room for at
    most one head segment, this block, and at most one tail segment.
    """

    mask = np.zeros((n + 1, n + 1), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if i + (n - j) + 1 <= k_max:
                mask[i, j] = True
    return mask


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
    ref = BayesBreakLogisticNormal(k_max=k_max, approx="quadrature", gh_points=gh_points).fit(
        np.arange(len(y)).reshape(-1, 1), y
    )
    t_ref = time.perf_counter() - t0
    lA0_ref = ref.log_block_evidence_
    if lA0_ref is None:
        raise RuntimeError("Reference fit did not store lA0_")

    true_interior = b_true[1:-1]
    reachable = _reachable_blocks_mask(n, k_max)

    methods = [
        ("quadrature", {"approx": "quadrature", "gh_points": gh_points}),
        ("laplace", {"approx": "laplace"}),
        ("jj", {"approx": "jj"}),
        ("ep", {"approx": "ep"}),
        ("pg-vb", {"approx": "pg_vb"}),
    ]

    rows = []

    # Reference row.
    pred_b = ref.map_boundaries_[1:-1]
    f1 = _boundary_f1(true_interior, pred_b, tau=tau)
    rows.append(("quadrature", 0.0, 0.0, 0.0, t_ref, f1, int(ref.k_map_)))

    for name, kwargs in methods[1:]:
        t0 = time.perf_counter()
        m = BayesBreakLogisticNormal(k_max=k_max, **kwargs).fit(np.arange(len(y)).reshape(-1, 1), y)
        t_fit = time.perf_counter() - t0

        lA0 = m.log_block_evidence_
        if lA0 is None:
            raise RuntimeError(f"{name} fit did not store log_block_evidence_")

        finite = np.isfinite(lA0_ref) & np.isfinite(lA0)
        mask = reachable & finite
        diffs = np.abs(lA0[mask] - lA0_ref[mask])
        if diffs.size:
            max_abs = float(np.max(diffs))
            q95 = float(np.quantile(diffs, 0.95))
            median = float(np.median(diffs))
        else:
            max_abs = q95 = median = float("nan")

        pred_b = m.map_boundaries_[1:-1]
        f1 = _boundary_f1(true_interior, pred_b, tau=tau)
        rows.append((name, max_abs, q95, median, t_fit, f1, int(m.k_map_)))

    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "table4_nonconj_tradeoff.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "method,reachable_max_abs,reachable_q95_abs,reachable_median_abs,"
            "fit_seconds,boundary_f1,k_sel\n"
        )
        for name, mx, q95, md, t_fit, f1, k_sel in rows:
            f.write(f"{name},{mx:.6f},{q95:.6f},{md:.6f},{t_fit:.4f},{f1:.4f},{k_sel}\n")

    md_path = outdir / "table4_nonconj_tradeoff.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(
            "| Method | reachable max |Δ log A0| | reachable q95 |Δ log A0| "
            "| reachable median |Δ log A0| | fit time (s) | F1@tau | k_sel |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, mx, q95, md, t_fit, f1, k_sel in rows:
            f.write(
                f"| {name} | {mx:.3f} | {q95:.3f} | {md:.3f} | {t_fit:.3f} | {f1:.3f} | {k_sel} |\n"
            )

    def _tex_escape(s: str) -> str:
        return s.replace("_", r"\_")

    tex_path = outdir / "table4_nonconj_tradeoff.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\\toprule\n")
        f.write(
            "Method & $\\max|\\Delta \\log A^0|$ & $q_{95}|\\Delta \\log A^0|$ & "
            "median $|\\Delta \\log A^0|$ & time (s) & F1@$\\tau$ & $\\hat{k}$"
            "\\\\\\midrule\n"
        )
        for name, mx, q95, md, t_fit, f1, k_sel in rows:
            f.write(
                f"{_tex_escape(name)} & {mx:.3f} & {q95:.3f} & {md:.3f} & "
                f"{t_fit:.3f} & {f1:.3f} & {k_sel}\\\\\n"
            )
        f.write("\\bottomrule\\end{tabular}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("docs/report/tables"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--tau", type=int, default=2)
    ap.add_argument("--gh-points", type=int, default=80)
    args = ap.parse_args()

    main(
        outdir=args.outdir,
        seed=args.seed,
        n=args.n,
        k_max=args.k_max,
        tau=args.tau,
        gh_points=args.gh_points,
    )
