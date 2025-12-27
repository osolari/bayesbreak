"""Table 3: Quantitative synthetic summary across conjugate families.

The paper's results section references a compact quantitative check that the
core BayesBreak inference behaves sensibly across conjugate exponential-family
models.

This script runs a small synthetic benchmark for four built-in conjugate
families:

* Gaussian (Normal--Normal)
* Poisson (Gamma--Poisson)
* Binomial (Beta--Binomial)
* Beta-valued (fractional Beta--Binomial)

For each family we simulate repeated sequences with known changepoints, fit the
corresponding BayesBreak model, and report:

* boundary F1 within a tolerance ``tau``
* mean absolute boundary error (MAE)
* mean squared error (MSE) on the latent segment parameter
* negative log evidence per observation: ``-log p(y)/n``

Outputs
-------
- results/table3_conjugate_summary.csv
- results/table3_conjugate_summary.md
- results/table3_conjugate_summary.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

# Allow running the script from a source checkout without installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from bayesbreak import (  # noqa: E402
    BayesBreakBeta,
    BayesBreakBinomial,
    BayesBreakGaussian,
    BayesBreakPoisson,
)


def _boundary_f1(true_b: Iterable[int], pred_b: Iterable[int], tau: int) -> float:
    true = sorted(int(t) for t in true_b)
    pred = sorted(int(p) for p in pred_b)
    # Greedy matching within tolerance.
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


def _boundary_mae(true_b: Iterable[int], pred_b: Iterable[int], n: int) -> float:
    true = np.asarray(sorted(int(t) for t in true_b), dtype=int)
    pred = np.asarray(sorted(int(p) for p in pred_b), dtype=int)
    if true.size == 0:
        return 0.0
    if pred.size == 0:
        return float(n)
    d = []
    for t in true:
        d.append(int(np.min(np.abs(pred - t))))
    return float(np.mean(d))


def _summarise(values: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def main(outdir: Path, seed: int, n_rep: int, tau: int) -> None:
    rng = np.random.default_rng(seed)

    n = 120
    boundaries_true = [0, n // 3, 2 * n // 3, n]
    true_b = boundaries_true[1:-1]
    k_true = len(boundaries_true) - 1

    rows = []

    # ----------------
    # Gaussian
    # ----------------
    mu_levels = np.array([0.0, 1.0, -0.5], dtype=float)
    sigma = 0.25
    f1s = []
    maes = []
    mses = []
    nlls = []
    k_sels = []
    for r in range(n_rep):
        mu = np.repeat(mu_levels, [n // 3, n // 3, n - 2 * (n // 3)])
        y = mu + sigma * rng.standard_normal(n)
        m = BayesBreakGaussian(k_max=12).fit(y)
        pred_b = m.get_boundaries()[1:-1]
        f1s.append(_boundary_f1(true_b, pred_b, tau=tau))
        maes.append(_boundary_mae(true_b, pred_b, n=n))
        mses.append(float(np.mean((m.predict() - mu) ** 2)))
        nlls.append(float(-m.score() / n))
        k_sels.append(int(m.k_ml_))

    rows.append(
        (
            "Gaussian",
            n,
            k_true,
            int(np.median(k_sels)),
            *_summarise(np.asarray(f1s)),
            *_summarise(np.asarray(maes)),
            *_summarise(np.asarray(mses)),
            *_summarise(np.asarray(nlls)),
        )
    )

    # ----------------
    # Poisson
    # ----------------
    lam_levels = np.array([2.0, 8.0, 3.0], dtype=float)
    f1s = []
    maes = []
    mses = []
    nlls = []
    k_sels = []
    lam_true = np.repeat(lam_levels, [n // 3, n // 3, n - 2 * (n // 3)])
    for _ in range(n_rep):
        y = rng.poisson(lam_true)
        m = BayesBreakPoisson(k_max=12).fit(y)
        pred_b = m.get_boundaries()[1:-1]
        f1s.append(_boundary_f1(true_b, pred_b, tau=tau))
        maes.append(_boundary_mae(true_b, pred_b, n=n))
        mses.append(float(np.mean((m.predict() - lam_true) ** 2)))
        nlls.append(float(-m.score() / n))
        k_sels.append(int(m.k_ml_))
    rows.append(
        (
            "Poisson",
            n,
            k_true,
            int(np.median(k_sels)),
            *_summarise(np.asarray(f1s)),
            *_summarise(np.asarray(maes)),
            *_summarise(np.asarray(mses)),
            *_summarise(np.asarray(nlls)),
        )
    )

    # ----------------
    # Binomial
    # ----------------
    n_trials = 20
    p_levels = np.array([0.1, 0.7, 0.3], dtype=float)
    p_true = np.repeat(p_levels, [n // 3, n // 3, n - 2 * (n // 3)])
    f1s = []
    maes = []
    mses = []
    nlls = []
    k_sels = []
    for _ in range(n_rep):
        y = rng.binomial(n_trials, p_true)
        m = BayesBreakBinomial(k_max=12, n_trials=n_trials).fit(y)
        pred_b = m.get_boundaries()[1:-1]
        f1s.append(_boundary_f1(true_b, pred_b, tau=tau))
        maes.append(_boundary_mae(true_b, pred_b, n=n))
        mses.append(float(np.mean((m.predict() - p_true) ** 2)))
        nlls.append(float(-m.score() / n))
        k_sels.append(int(m.k_ml_))
    rows.append(
        (
            "Binomial",
            n,
            k_true,
            int(np.median(k_sels)),
            *_summarise(np.asarray(f1s)),
            *_summarise(np.asarray(maes)),
            *_summarise(np.asarray(mses)),
            *_summarise(np.asarray(nlls)),
        )
    )

    # ----------------
    # Beta-valued (fractional Beta--Binomial)
    # ----------------
    kappa = 50
    p_levels = np.array([0.2, 0.85, 0.4], dtype=float)
    p_true = np.repeat(p_levels, [n // 3, n // 3, n - 2 * (n // 3)])
    f1s = []
    maes = []
    mses = []
    nlls = []
    k_sels = []
    for _ in range(n_rep):
        s = rng.binomial(kappa, p_true)
        y = (s + 0.5) / (kappa + 1.0)  # avoid exact 0/1
        m = BayesBreakBeta(k_max=12, concentration=float(kappa)).fit(y)
        pred_b = m.get_boundaries()[1:-1]
        f1s.append(_boundary_f1(true_b, pred_b, tau=tau))
        maes.append(_boundary_mae(true_b, pred_b, n=n))
        mses.append(float(np.mean((m.predict() - p_true) ** 2)))
        nlls.append(float(-m.score() / n))
        k_sels.append(int(m.k_ml_))
    rows.append(
        (
            "Beta-valued",
            n,
            k_true,
            int(np.median(k_sels)),
            *_summarise(np.asarray(f1s)),
            *_summarise(np.asarray(maes)),
            *_summarise(np.asarray(mses)),
            *_summarise(np.asarray(nlls)),
        )
    )

    # ---- Write outputs ----
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "table3_conjugate_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "family,n,k_true,k_sel_med,f1_mean,f1_std,mae_mean,mae_std,mse_mean,mse_std,nll_mean,nll_std\n"
        )
        for row in rows:
            fam, n, k_true, k_sel, f1m, f1s, maem, maes, msem, mses, nllm, nlls = row
            f.write(
                f"{fam},{n},{k_true},{k_sel},{f1m:.4f},{f1s:.4f},{maem:.3f},{maes:.3f},{msem:.4f},{mses:.4f},{nllm:.4f},{nlls:.4f}\n"
            )

    md_path = outdir / "table3_conjugate_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(
            "| Family | n | k_true | k_sel (median) | F1@tau | MAE | MSE | -log p(y)/n |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            fam, n, k_true, k_sel, f1m, _, maem, _, msem, _, nllm, _ = row
            f.write(
                f"| {fam} | {n} | {k_true} | {k_sel} | {f1m:.3f} | {maem:.2f} | {msem:.4f} | {nllm:.3f} |\n"
            )

    tex_path = outdir / "table3_conjugate_summary.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\\toprule\n")
        f.write(
            "Family & n & $k^\star$ & $\hat{k}$ & F1@$\\tau$ & MAE & MSE & $-\\log p(y)/n$\\\\\\midrule\n"
        )
        for row in rows:
            fam, n, k_true, k_sel, f1m, _, maem, _, msem, _, nllm, _ = row
            f.write(
                f"{fam} & {n} & {k_true} & {k_sel} & {f1m:.3f} & {maem:.2f} & {msem:.4f} & {nllm:.3f}\\\\\n"
            )
        f.write("\\bottomrule\\end{tabular}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rep", type=int, default=25)
    ap.add_argument("--tau", type=int, default=2)
    args = ap.parse_args()
    main(outdir=args.outdir, seed=args.seed, n_rep=args.n_rep, tau=args.tau)
