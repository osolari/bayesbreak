"""Generate demonstration notebooks directly under docs/tutorials/.

Each notebook is authored programmatically so that updating the demo set
is a single-file change. Running this script populates:

- docs/tutorials/03_real_data_showcase.ipynb
- docs/tutorials/04_diagnostics.ipynb
- docs/tutorials/05_baselines.ipynb
- docs/tutorials/06_sliding_window.ipynb
- docs/tutorials/07_latent_groups.ipynb

The notebooks are NOT executed at generation time (mkdocs-jupyter is
configured with execute=false). Users run them interactively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"


def md(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": text.splitlines(keepends=True),
    }


def code(src: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def write_notebook(name: str, cells: list[dict[str, Any]]) -> None:
    for index, cell in enumerate(cells):
        cell["id"] = f"{name}-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10+",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = TUTORIALS / f"{name}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------- 03
nb03 = [
    md(
        "# Real-data showcase\n\n"
        "Four real-data case studies fit by BayesBreak, mirroring §6 of the manuscript:\n\n"
        "1. **Well-log NMR** geology (Gaussian block, length 4050 NMR series).\n"
        "2. **Coriell array-CGH** copy number (heteroscedastic multi-subject Gaussian).\n"
        "3. **S&P 500 squared returns** volatility regimes (Gaussian on `log r_t^2`).\n"
        "4. **CpG-atlas methylation** (Beta-response block with per-CpG precision).\n\n"
        "Each loader falls back to a deterministic simulated analog when the network"
        " download is unavailable, so this notebook runs end-to-end on a fresh checkout"
        " without any datasets installed."
    ),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from bayesbreak import BayesBreakGaussian, BayesBreakBetaObs\n"
        "from bayesbreak.datasets import (\n"
        "    load_welllog,\n"
        "    load_cgh,\n"
        "    load_spx,\n"
        "    load_methylation,\n"
        ")\n"
    ),
    md("## 1. Well-log NMR (Gaussian block)"),
    code(
        "bundle = load_welllog()\n"
        "print('source:', bundle.source, ' n:', bundle.y.size)\n"
        "\n"
        "# Subsample 4050 -> ~500 for a fast demo.\n"
        "y = bundle.y[::8]\n"
        "X = np.arange(y.size).reshape(-1, 1)\n"
        "\n"
        "est = BayesBreakGaussian(k_max=40, regression_curve='none').fit(X, y)\n"
        "print(f'k_map={est.k_map_}, log p(y)={est.log_evidence_:.1f}')"
    ),
    code(
        "fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)\n"
        "axes[0].plot(X.ravel(), y, color='grey', lw=0.5)\n"
        "axes[0].plot(X.ravel(), est.map_curve_, color='C0', lw=1.5, label='MAP')\n"
        "for b in est.map_boundaries_[1:-1]:\n"
        "    axes[0].axvline(b, color='C3', ls='--', lw=0.6, alpha=0.7)\n"
        "axes[0].set_ylabel('NMR (standardised)')\n"
        "axes[0].legend()\n"
        "\n"
        "axes[1].fill_between(np.arange(1, y.size), 0, est.boundary_marginals_, color='C0', alpha=0.4)\n"
        "axes[1].set_xlabel('index')\n"
        "axes[1].set_ylabel(r'$P(b_i=1 | y)$')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 2. Array-CGH (multi-subject pooled Gaussian)"),
    code(
        "from bayesbreak import SharedBoundaryReplicatesSegmenter\n"
        "\n"
        "cgh = load_cgh()\n"
        "print('source:', cgh.source, ' shape:', cgh.y.shape)\n"
        "\n"
        "y_cgh = cgh.y if cgh.y.ndim == 2 else cgh.y[:, None]\n"
        "w_cgh = cgh.sample_weight\n"
        "X_cgh = np.arange(y_cgh.shape[0]).reshape(-1, 1)\n"
        "\n"
        "rep = SharedBoundaryReplicatesSegmenter(\n"
        "    BayesBreakGaussian(k_max=15)\n"
        ").fit(X_cgh, y_cgh, sample_weight=w_cgh)\n"
        "print(f'pooled k_map={rep.k_map_}, log p(y)={rep.log_evidence_:.1f}')"
    ),
    md("## 3. S&P 500 volatility regimes (Gaussian on log squared returns)"),
    code(
        "spx = load_spx()\n"
        "print('source:', spx.source, ' n:', spx.y.size)\n"
        "\n"
        "y_spx = spx.y[::4]              # stride-4 subsample for speed\n"
        "X_spx = np.arange(y_spx.size).reshape(-1, 1)\n"
        "est_spx = BayesBreakGaussian(k_max=50, regression_curve='none').fit(X_spx, y_spx)\n"
        "print(f'k_map={est_spx.k_map_}, log p(y)={est_spx.log_evidence_:.1f}')\n"
        "\n"
        "plt.figure(figsize=(8, 2.5))\n"
        "plt.plot(X_spx.ravel(), y_spx, color='grey', lw=0.4)\n"
        "for b in est_spx.map_boundaries_[1:-1]:\n"
        "    plt.axvline(b, color='C3', ls='--', lw=0.5, alpha=0.6)\n"
        "plt.title('SPX volatility-regime MAP boundaries'); plt.tight_layout(); plt.show()"
    ),
    md("## 4. Methylation (Beta-response block with per-CpG precision)"),
    code(
        "meth = load_methylation()\n"
        "print('source:', meth.source, ' n:', meth.y.size)\n"
        "\n"
        "phi = meth.sample_weight if meth.sample_weight is not None else 50.0\n"
        "est_meth = BayesBreakBetaObs(k_max=15, phi=phi, regression_curve='mix_k').fit(meth.X, meth.y)\n"
        "print(f'k_map={est_meth.k_map_}, log p(y)={est_meth.log_evidence_:.1f}')"
    ),
    md(
        "## What's next?\n\n"
        "- See [Diagnostics walkthrough](04_diagnostics.ipynb) for the TV-bound, prior-sensitivity, and G-selection diagnostics.\n"
        "- See [Baselines comparison](05_baselines.ipynb) for PELT / BS / WBS comparisons on the same data.\n"
        "- See [Sliding-window for large n](06_sliding_window.ipynb) for handling sequences far longer than the exact DP can fit."
    ),
]
write_notebook("03_real_data_showcase", nb03)


# ----------------------------------------------------------------------- 04
nb04 = [
    md(
        "# Diagnostics walkthrough\n\n"
        "All four `bayesbreak.diagnostics` routines on one synthetic Bernoulli-logistic dataset."
    ),
    code(
        "import numpy as np\n"
        "from bayesbreak import (\n"
        "    BayesBreakGaussian,\n"
        "    BayesBreakLogisticNormal,\n"
        "    run_dp_diagnostics,\n"
        "    run_non_conjugate_diagnostics,\n"
        "    run_prior_sensitivity,\n"
        "    select_n_groups_by_holdout,\n"
        ")\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 80\n"
        "y_g = np.r_[rng.normal(0, 0.3, 30), rng.normal(2, 0.3, 30), rng.normal(-1, 0.3, 20)]\n"
        "X = np.arange(n).reshape(-1, 1)"
    ),
    md(
        "## 1. DP invariants (`run_dp_diagnostics`)\n"
        "Checks `∑ P(k) = 1`, forward/backward agreement (`prop:fb-duality`),\n"
        "`∑_i P(b_i | y, k_map) = k_map − 1`, and MAP backtrack consistency\n"
        "(`thm:map-correctness`)."
    ),
    code(
        "est = BayesBreakGaussian(k_max=8).fit(X, y_g)\n"
        "report = run_dp_diagnostics(est)\n"
        "print(report.summary)\n"
        "for c in report.checks:\n"
        "    print(f'  - {c.name}: passed={c.passed}  detail={c.detail}')"
    ),
    md(
        "## 2. Non-conjugate diagnostics (`run_non_conjugate_diagnostics`)\n"
        "Measures the empirical `ε` of `ass:uniform-block-error` and the\n"
        "worst-case TV bound `exp(2 k_max ε) − 1` from\n"
        "`cor:probability-error-conversion`. Reference fit uses high-Q\n"
        "Gauss–Hermite quadrature."
    ),
    code(
        "theta = np.r_[np.full(20, -1.0), np.full(20, 1.0), np.full(20, -0.5)]\n"
        "p = 1.0 / (1.0 + np.exp(-theta))\n"
        "y_b = rng.binomial(1, p).astype(float)\n"
        "X_b = np.arange(y_b.size).reshape(-1, 1)\n"
        "\n"
        "ref = BayesBreakLogisticNormal(k_max=8, approx='quadrature', gh_points=80).fit(X_b, y_b)\n"
        "lap = BayesBreakLogisticNormal(k_max=8, approx='laplace').fit(X_b, y_b)\n"
        "\n"
        "diag = run_non_conjugate_diagnostics(lap, ref)\n"
        "print('block_error_max:', diag.extra['block_error_max'])\n"
        "print('pk_tv_empirical:', diag.extra['pk_tv_empirical'])\n"
        "print('pk_tv_upper_bound:', diag.extra['pk_tv_upper_bound'])\n"
        "print('theoretical_rate:', diag.extra['theoretical_rate'])\n"
        "print('theoretical_rate_violated:', diag.extra['theoretical_rate_violated'])"
    ),
    md(
        "## 3. Prior-sensitivity (`run_prior_sensitivity`)\n"
        "Reruns the DP under perturbations of `p(k)` and the length factor\n"
        "`g(ℓ)`; reports `Δ p(k|y)` and `Δ P(b_i|y, k_map)` per variant.\n"
        "This is the §5b *partition-prior sensitivity* diagnostic."
    ),
    code(
        "sens = run_prior_sensitivity(est)\n"
        "for v in sens.extra['variants']:\n"
        "    print(f\"  {v['variant']}: Δ p(k|y) max={v['delta_pk_max']:.3f}, \"\n"
        "          f\"TV={v['delta_pk_tv']:.3f}; Δ P(b|y) L1={v['delta_bm_l1']:.3f}\")"
    ),
    md(
        "## 4. Held-out G-selection (`select_n_groups_by_holdout`)\n"
        "K-fold marginal log-likelihood over the sequence axis for the latent-\n"
        "template mixture. Mitigates `rem:teicher-overspec` (overspecified-G\n"
        "redundancy)."
    ),
    code(
        "# Two visibly different templates: smooth jump up vs sharp drop.\n"
        "seqs_a = [np.r_[rng.normal(0, 0.2, 30), rng.normal(2, 0.2, 30)] for _ in range(6)]\n"
        "seqs_b = [np.r_[rng.normal(0, 0.2, 15), rng.normal(-2, 0.2, 45)] for _ in range(6)]\n"
        "sequences = seqs_a + seqs_b\n"
        "\n"
        "sel = select_n_groups_by_holdout(\n"
        "    BayesBreakGaussian(k_max=4), sequences, g_grid=(1, 2, 3), n_folds=3,\n"
        ")\n"
        "print('best_g =', sel.extra['best_g'])\n"
        "for g, m, s in zip(sel.extra['g_grid'], sel.extra['mean_test_loglik'], sel.extra['std_test_loglik']):\n"
        "    print(f'  G={g}: mean held-out log p(y) = {m:.2f}  (±{s:.2f})')"
    ),
]
write_notebook("04_diagnostics", nb04)


# ----------------------------------------------------------------------- 05
nb05 = [
    md(
        "# Baselines comparison\n\n"
        "Run BayesBreak alongside the upstream-driven baseline wrappers in\n"
        "`bayesbreak.baselines`. We do **not** re-implement these algorithms —\n"
        "each call is a thin wrapper around the canonical package.\n\n"
        "This notebook covers the pure-Python wrappers (`ruptures` + the\n"
        "`fearnhead_exact` reference). R-backed wrappers (`cbs`, `smuce`,\n"
        "`rjmcmc`) are listed at the end with install hints."
    ),
    code(
        "import numpy as np\n"
        "from bayesbreak import BayesBreakGaussian\n"
        "from bayesbreak.baselines import segment_with, available_algorithms\n"
        "\n"
        "print('available:', available_algorithms())\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 200\n"
        "true_b = [70, 130]\n"
        "y = np.r_[\n"
        "    rng.normal(0.0, 0.3, 70),\n"
        "    rng.normal(2.0, 0.3, 60),\n"
        "    rng.normal(-1.0, 0.3, 70),\n"
        "]\n"
        "X = np.arange(n).reshape(-1, 1)"
    ),
    md("## BayesBreak"),
    code(
        "bb = BayesBreakGaussian(k_max=8).fit(X, y)\n"
        "print(f'BayesBreak: k_map={bb.k_map_}, boundaries={bb.map_boundaries_[1:-1]}')"
    ),
    md("## Ruptures (PELT, Optimal Partitioning, BS, WBS)"),
    code(
        "for name, kwargs in [\n"
        "    ('pelt',                 dict(penalty=10.0)),\n"
        "    ('optimal_partitioning', dict(n_bkps=2)),\n"
        "    ('binary_segmentation',  dict(n_bkps=2)),\n"
        "    ('wild_binary_segmentation',\n"
        "                             dict(n_bkps=2, random_state=0, n_random_windows=30)),\n"
        "]:\n"
        "    res = segment_with(name, y, **kwargs)\n"
        "    print(f'{name:>26s}: k={res.k}, boundaries={res.boundaries.tolist()}')"
    ),
    md(
        "## Fearnhead-exact-DP reference\n"
        "Drives BayesBreak's own DP at the Fearnhead-2006 prior choice\n"
        "(geometric `p(k)`, optional length-aware cohesion). Labelled\n"
        "reference comparator — no standalone third-party implementation."
    ),
    code(
        "res = segment_with('fearnhead_exact', y, k_max=8, geometric_rate=0.3)\n"
        "print(f'fearnhead_exact: k={res.k}, boundaries={res.boundaries.tolist()}')\n"
        "print(f'  provenance: {res.package} v{res.package_version}')\n"
        "print(f'  extra: {res.extra}')"
    ),
    md(
        "## R-backed baselines (install hints)\n\n"
        "These require `pip install bayesbreak[baselines-r]` plus the R packages.\n"
        "See [Installation](../installation.md):\n\n"
        "- **CBS** (Olshen et al. 2004) via `DNAcopy::segment`:\n"
        "  ```python\n"
        "  res = segment_with('cbs', y_log2ratio, alpha=0.01, nperm=10_000)\n"
        "  ```\n"
        "- **SMUCE** (Frick, Munk & Sieling 2014) via `stepR::stepFit`:\n"
        "  ```python\n"
        "  res = segment_with('smuce', y, alpha=0.05, family='gauss')\n"
        "  ```\n"
        "- **RJMCMC-style MCMC** (Lindeløv 2020) via `mcp::mcp` + JAGS:\n"
        "  ```python\n"
        "  res = segment_with('rjmcmc', y, n_segments=3, n_iter=3000, n_chains=2)\n"
        "  ```\n"
    ),
]
write_notebook("05_baselines", nb05)


# ----------------------------------------------------------------------- 06
nb06 = [
    md(
        "# Sliding-window decomposition for large n\n\n"
        "The exact DP is `Θ(k_max n²)` time and at least `Θ(k_max n)` memory\n"
        "(`prop:bb-complexity`). At `n ≳ 10^5` this becomes impractical.\n"
        "§5b *Computational regime* proposes a sliding-window decomposition\n"
        "that splits the sequence into overlapping windows, runs the exact DP\n"
        "on each, and stitches the per-window outputs.\n\n"
        "This is an **approximation** — stitched results do not inherit\n"
        "`thm:dp-correctness` exactly and the boundary-event-sum identity holds\n"
        "only per window."
    ),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import time\n"
        "from bayesbreak import BayesBreakGaussian, SlidingWindowSegmenter\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "# A longer signal than the exact DP would normally chew on.\n"
        "n = 3000\n"
        "k_true = 10\n"
        "bps = sorted(rng.choice(np.arange(50, n - 50), size=k_true - 1, replace=False).tolist())\n"
        "means = rng.normal(0, 1.5, size=k_true)\n"
        "y = np.empty(n)\n"
        "starts = [0, *bps, n]\n"
        "for i in range(k_true):\n"
        "    y[starts[i]:starts[i + 1]] = rng.normal(means[i], 0.4, size=starts[i + 1] - starts[i])\n"
        "X = np.arange(n).reshape(-1, 1)\n"
        "print('n =', n, 'true segments =', k_true)"
    ),
    md("## Fit with the sliding-window decomposition"),
    code(
        "t0 = time.perf_counter()\n"
        "sw = SlidingWindowSegmenter(\n"
        "    BayesBreakGaussian(k_max=8),\n"
        "    window_size=500,\n"
        "    overlap=100,\n"
        ").fit(X, y)\n"
        "dt = time.perf_counter() - t0\n"
        "print(f'sliding-window fit: n_windows={len(sw.windows_)}, total runtime={dt:.2f}s')\n"
        "print(f'  k_hat = {sw.k_hat_}, |boundaries| = {len(sw.map_boundaries_) - 2}')\n"
        "print(f'  approximate log p(y) = {sw.log_evidence_:.1f}')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 3))\n"
        "ax.plot(X.ravel(), y, color='grey', lw=0.3)\n"
        "ax.plot(X.ravel(), sw.map_curve_, color='C0', lw=1.4, label='sliding-window MAP')\n"
        "for b in sw.map_boundaries_[1:-1]:\n"
        "    ax.axvline(b, color='C3', ls='--', lw=0.5, alpha=0.6)\n"
        "for b in bps:\n"
        "    ax.axvline(b, color='C2', ls=':', lw=0.7, alpha=0.7)\n"
        "ax.set_xlabel('index'); ax.set_ylabel('y')\n"
        "ax.legend(); plt.tight_layout(); plt.show()"
    ),
    md(
        "## When to prefer the exact DP\n\n"
        "Use the sliding-window when `n` would make `prop:bb-complexity` time or\n"
        "memory prohibitive (typically `n ≳ 10^5`). For smaller `n`, the exact\n"
        "DP gives you the boundary-event sum identity, the forward-backward\n"
        "duality, and Corollary `cor:probability-error-conversion` exactly."
    ),
]
write_notebook("06_sliding_window", nb06)


# ----------------------------------------------------------------------- 07
nb07 = [
    md(
        "# Latent-group EM\n\n"
        "Subjects come from `G` unknown groups, each with its own boundary template.\n"
        "`BayesBreakMixtureClassifier` alternates E-step (exact responsibilities) and\n"
        "M-step (responsibility-weighted max-sum DP), optimising the finite\n"
        "template-mixture objective `ℓ_⋆` (`thm:em-monotone`)."
    ),
    code(
        "import numpy as np\n"
        "from bayesbreak import BayesBreakGaussian, BayesBreakMixtureClassifier\n"
        "\n"
        "rng = np.random.default_rng(0)\n"
        "n = 60\n"
        "# Group A: jump at index 30 (up).\n"
        "group_a = [\n"
        "    np.r_[rng.normal(-1.0, 0.2, 30), rng.normal(2.0, 0.2, 30)] for _ in range(8)\n"
        "]\n"
        "# Group B: jump at index 15 (down).\n"
        "group_b = [\n"
        "    np.r_[rng.normal(0.0, 0.2, 15), rng.normal(-2.0, 0.2, 45)] for _ in range(8)\n"
        "]\n"
        "sequences = group_a + group_b\n"
        "labels_true = np.array([0]*8 + [1]*8)"
    ),
    md("## Fit the mixture with G=2 (correct G)"),
    code(
        "mix = BayesBreakMixtureClassifier(\n"
        "    BayesBreakGaussian(k_max=4),\n"
        "    n_groups=2,\n"
        "    max_iter=20,\n"
        "    random_state=0,\n"
        ").fit(sequences)\n"
        "\n"
        "print('canonical permutation:', mix.canonical_permutation_)\n"
        "print('pi:', mix.pi_)\n"
        "for g, state in enumerate(mix.group_states_):\n"
        "    print(f'  group {g}: k_g={state.k_g}, template={state.template}')"
    ),
    md(
        "## Verify identifiability anchoring (`prop:latent-identifiability`)\n\n"
        "The canonical anchor sorts groups by `k_g`, then lexicographically by\n"
        "`t^{(g)}`. Two restarts converging to the same template multiset must\n"
        "report them in identical order."
    ),
    code(
        "mix_alt = BayesBreakMixtureClassifier(\n"
        "    BayesBreakGaussian(k_max=4), n_groups=2, max_iter=20, random_state=42\n"
        ").fit(sequences)\n"
        "\n"
        "set_orig = {(s.k_g, tuple(s.template)) for s in mix.group_states_}\n"
        "set_alt  = {(s.k_g, tuple(s.template)) for s in mix_alt.group_states_}\n"
        "print('matched templates:', set_orig == set_alt)\n"
        "if set_orig == set_alt:\n"
        "    keys_orig = [(s.k_g, tuple(s.template)) for s in mix.group_states_]\n"
        "    keys_alt  = [(s.k_g, tuple(s.template)) for s in mix_alt.group_states_]\n"
        "    print('same canonical order:', keys_orig == keys_alt)"
    ),
    md(
        "## Held-out G selection (`select_n_groups_by_holdout`)\n\n"
        "Mitigates `rem:teicher-overspec` overspecification: at G > G*, two\n"
        "distinct (π, τ) tuples can produce identical mixture densities.\n"
        "Held-out marginal log-likelihood is the §5b recommended response."
    ),
    code(
        "from bayesbreak import select_n_groups_by_holdout\n"
        "\n"
        "sel = select_n_groups_by_holdout(\n"
        "    BayesBreakGaussian(k_max=4), sequences,\n"
        "    g_grid=(1, 2, 3),\n"
        "    n_folds=4,\n"
        "    random_state=0,\n"
        ")\n"
        "print('best_g =', sel.extra['best_g'])\n"
        "for g, m, s in zip(sel.extra['g_grid'], sel.extra['mean_test_loglik'], sel.extra['std_test_loglik']):\n"
        "    print(f'  G={g}: mean held-out log p(y) = {m:.2f}  (±{s:.2f})')"
    ),
]
write_notebook("07_latent_groups", nb07)
