# BayesBreak

[![CI](https://github.com/osolari/bayesbreak/actions/workflows/ci.yml/badge.svg)](https://github.com/osolari/bayesbreak/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.14681-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2603.14681)

**Exact Bayesian segmentation with a scikit-learn compatible API.**

BayesBreak turns the two-layer design of the accompanying report into a small,
reusable library:

- **Block evidence** — a family-specific integrated single-segment marginal
  likelihood `A^0_{ij}` (Gaussian, Poisson, Binomial, Bernoulli, Beta,
  Beta-observation, NegBin, Logistic-Normal).
- **Dynamic programming** — a distribution-agnostic engine that delivers, from
  that block matrix alone, the marginal evidence `p(y)`, the segment-count
  posterior `P(k|y)`, the conditional boundary-event marginal
  `P(b_i = 1 | y, k_map)`, the joint MAP segmentation
  (max-sum + backtracking), and the Bayesian regression curve.
- **Design-aware partition prior** — pass `length_prior=g(Δ)` and let the DP
  thread the segment-cohesion factor through every recursion (the
  `C_k = Σ_t ∏_q g(Δ_x)` normalizer is computed automatically).

Every estimator inherits from `sklearn.base.BaseEstimator` and can live inside
an `sklearn.pipeline.Pipeline`.

## Install

```bash
pip install bayesbreak                # runtime + core deps
pip install "bayesbreak[plots]"       # + matplotlib/seaborn
pip install "bayesbreak[datasets]"    # + real-data loaders (pooch / rdata / requests)
pip install "bayesbreak[dev]"         # + test/lint/type toolchain
```

Editable dev setup:

```bash
bash create_env.sh                    # conda env "bayesbreak" with Python 3.11
bash create_env.sh --venv             # or a plain python -m venv
```

## Quickstart

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from bayesbreak import BayesBreakGaussian

rng = np.random.default_rng(0)
n = 200
X = np.arange(n).reshape(-1, 1)
y = np.r_[rng.normal(0, 0.3, 80), rng.normal(2, 0.3, 60), rng.normal(-1, 0.3, 60)]

pipe = Pipeline([("scale", StandardScaler()), ("seg", BayesBreakGaussian(k_max=10))])
pipe.fit(X, y)

seg = pipe.named_steps["seg"]
print("MAP segment count:", seg.k_map_)
print("MAP boundaries    :", seg.map_boundaries_)
print("log p(y)          :", seg.log_evidence_)
print("predictive score  :", pipe.score(X, y))  # mean posterior-predictive log-density
```

Key fitted attributes (trailing underscore):

| Attribute | Meaning |
|---|---|
| `k_map_` | Posterior-mode segment count |
| `map_boundaries_` | Joint MAP boundary vector `(0, t_1, …, n)` |
| `map_segment_means_` | Posterior mean per MAP segment |
| `k_posterior_` | `P(k \| y)` as an array of length `k_max` |
| `boundary_marginals_` | `P(b_i = 1 \| y, k_map_)` (the §6 calibration target) |
| `log_evidence_` | `log p(y)` of the training sequence |
| `log_C_k_` | Partition-prior normalizer `log C_k` (matches the `length_prior`) |
| `boundary_coordinates_` | Candidate boundary coordinates `u_0 < … < u_n` |
| `bayes_curve_mean_` | Posterior-mean latent signal (when `regression_curve != "none"`) |

## Design-aware partition priors

Irregular designs and physical segment-length cohesions enter through the
constructor:

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

x = np.array([0.0, 0.2, 0.4, 0.6, 1.6, 1.8])  # tight cluster + wide gap + pair
y = np.array([0.1, 0.0, -0.05, 0.05, 1.0, 1.05])

# Length-aware prior g(Δ) ∝ Δ shifts boundary mass toward the wide gap.
seg = BayesBreakGaussian(k_max=3, length_prior=lambda d: d).fit(x, y)
```

`prior_k=callable` and `boundary_coordinates=ndarray` are also exposed; on a
regular grid pass `boundary_coordinates=np.arange(n + 1)` to fix `u_i = i`.

## Supported families

```python
from bayesbreak import (
    BayesBreakGaussian,        # Normal-Normal (weighted)
    BayesBreakPoisson,         # Poisson-Gamma (exposure)
    BayesBreakBinomial,        # Beta-Binomial
    BayesBreakBernoulli,       # Beta-Bernoulli
    BayesBreakBeta,            # fractional Beta-Binomial for y in (0,1)
    BayesBreakBetaObs,         # Beta likelihood + Beta prior (per-obs φ_t, 1-D quadrature)
    BayesBreakNegBin,          # Beta-NegBin overdispersed counts (observation-mean target)
    BayesBreakLogisticNormal,  # Bernoulli + Normal prior on log-odds
    make_bayesbreak,           # factory: make_bayesbreak("poisson", k_max=20)
)
```

## Multivariate, replicates, groups, latent groups

- `SharedBoundaryMultivariateSegmenter` — single segmentation across vector-
  valued response channels.
- `IndependentMultivariateSegmenter` — per-channel fit.
- `SharedBoundaryReplicatesSegmenter` — exact **boundary-posterior pooling**
  for multi-subject 1-D sequences (Theorem `multisubject`); subject-specific
  segment posteriors recovered conditionally on the pooled boundary vector.
- `BayesBreakGroupedClassifier` — known-group classification; per-group
  exemplar fits + exported-MAP scoring of new sequences.
- `BayesBreakMixtureClassifier` — latent-group **template-mixture EM**
  (§`latent-em`), maximizing the finite-template mixture objective `ℓ_⋆`.

## Prediction layer (§`sec:prediction`)

```python
from bayesbreak.prediction import (
    posterior_predictive_logpdf,    # Case A: pointwise
    Unit, unit_responsibilities,    # Case B: set-valued / multivariate units
    predict_group, predict_map_signal,  # algorithms predict-group / predict-map
    pit_residuals,                  # PIT diagnostic for closed-CDF families
    held_out_log_likelihood_trace,  # cumulative HLL diagnostic
)
```

Outputs are exactly the five interfaces of `tab:prediction-outputs`:
`ℓ_g`, `P(g | new)`, `f^MAP_g(X)`, `f^Bayes_g(X)`, and unit responsibilities.

## Diagnostics

```python
from bayesbreak import run_dp_diagnostics, run_non_conjugate_diagnostics
```

`run_dp_diagnostics(estimator)` checks the four §4.2 invariants
(`Σ P(k) = 1`, `L_kn = R_k0`, `Σ P(b) = k − 1` per Corollary
`cor:boundary-event-sum`, max-sum vs. terminal score).
`run_non_conjugate_diagnostics(approx, reference)` reports the max / 95th /
median block error over **reachable** blocks, the posterior-sensitivity
summaries, and the absolute-probability TV bound
`exp(2·k_max·ε) − 1` of Corollary `cor:abs-prob` (fields
`pk_tv_empirical`, `pk_tv_upper_bound`). Every check is tagged with a
`failure_mode` matching the rewritten §4 approximation-validation
checklist.

`bayesbreak.diagnostics.run_prior_sensitivity(estimator)` is the planned
diagnostic from §6 (paragraph 6-C1): it reruns the DP on the existing
`log_block_evidence_` under perturbations of `p(k)` and the length factor
`g`, and reports the resulting variation of `P(k|y)` and the fixed-`k_map`
boundary marginals.

## Baselines (frequentist comparators)

`bayesbreak.baselines` exposes thin wrappers around the canonical
upstream packages — we do **not** re-implement these algorithms. PELT,
optimal partitioning, BS, and WBS are driven through
[`ruptures`](https://github.com/deepcharles/ruptures); CBS is driven
through Bioconductor `DNAcopy` via `rpy2`. Coverage matches the §6
planned-baseline list (5-A1, 6-E3) plus the `ruptures`/`changepoint`
positioning paragraph (1-D2/G-1).

```bash
pip install bayesbreak[baselines]      # ruptures (Python only)
pip install bayesbreak[baselines-r]    # rpy2; you also need R + DNAcopy
```

```python
from bayesbreak.baselines import segment_with

res = segment_with("pelt", y, penalty=10.0)
res.boundaries          # array of interior changepoint indices
res.k                   # number of segments
res.package, res.package_version, res.tuning
```

Missing upstream packages raise a single readable `ImportError` rather
than an opaque attribute error. Each :class:`BaselineResult` records the
package name, version, and the tuning kwargs that were used.

Every fitted segmenter exposes `admissibility_mask_` — the boolean mask
of finite cells in `log_block_evidence_`, materializing the
§`sec:setup` "Block-score contract" that the DP and `compute_log_C_k`
share the same admissibility convention. Each family also declares a
`MOMENT_SIGN_CONTRACT` class attribute (`"signed"` for Gaussian,
`"nonneg"` for the others) per §5 paragraph 5-C1.

## How it maps to the paper

| Paper equation / theorem | Implementation |
|---|---|
| Block evidence `A^0_{ij}` (Theorem `ef-integral`) | `_compute_block_evidence` per family in `bayesbreak.families` |
| Forward / backward sum-product (eq. `LR`) | `bayesbreak.dp.forward_backward` |
| `P(k \| y)`, `log p(y)` (eq. `post-k`) | `bayesbreak.dp.posterior_over_k` |
| Conditional boundary-event marginal (eq. `boundary-event`) | `bayesbreak.dp.boundary_event_marginals_fixed_k` |
| Length-prior absorption + `C_k` (eq. `Atilde`, `Ck-general`) | `compute_log_C_k` + `log_g_table` everywhere |
| Joint MAP via max-sum + backtracking (eq. `joint-map-k`) | `bayesbreak.dp.max_sum_segmentation` |
| Bayes regression curve (eq. `segmom`) | `bayesbreak.dp.bayes_regression_curve_{fixed,mixed}_k` |
| Theorem `multisubject` (replicates) | `bayesbreak.replicates.SharedBoundaryReplicatesSegmenter` |
| Algorithm `multi-em` (latent-template EM) | `bayesbreak.mixture.BayesBreakMixtureClassifier` |
| Posterior-predictive (Prop. `ef-predictive`) | `bayesbreak.prediction.posterior_predictive_logpdf` |
| Algorithms `predict-group` / `predict-map` | `bayesbreak.prediction.{predict_group, predict_map_signal}` |
| Non-conjugate block approximations (§`sec:nonconj`) | `BayesBreakLogisticNormal(approx=...)`, `BayesBreakBetaObs(phi=φ_t)` |
| Stability bound (Prop. `stability`) | `run_non_conjugate_diagnostics` reports reachable-block error |
| §6 sanity checks + §`prediction-diagnostics` | `bayesbreak.run_dp_diagnostics`, `pit_residuals`, `held_out_log_likelihood_trace` |

## Real-data loaders

`bayesbreak.datasets` ships four loaders — `load_welllog()`, `load_cgh()`,
`load_spx()`, `load_methylation()` — that return a uniform `DatasetBundle`
(`X`, `y`, `sample_weight`, `true_boundaries`, `source`, `description`):

| Loader | Real source | Falls back to |
|---|---|---|
| `load_welllog()` | TCPD `well_log.txt` (n=4050) | seed-pinned NMR analog |
| `load_cgh()` | `cran/ecp` `ACGH.RData` (2215 × 43, multi-subject) | seed-pinned single-subject analog |
| `load_spx()` | `yfinance` `^GSPC` log-squared returns | GARCH-like regime analog |
| `load_methylation()` | `methylKit` chr21 example (n=1904, per-CpG coverage as `φ_t`) | Beta-fraction analog |

All loaders accept `simulated=True` to force the analog and emit a banner
recording provenance.

## Reproducing figures and tables

```bash
# §6 synthetic suite (figures + tables, archived completed artifacts):
python -m bayesbreak.experiments.synthetic --all

# Real-data illustrations (placeholder mode by default — watermark + sidecar JSON):
python -m bayesbreak.experiments.realdata --dataset welllog
python -m bayesbreak.experiments.realdata --dataset cgh
python -m bayesbreak.experiments.realdata --dataset spx
python -m bayesbreak.experiments.realdata --dataset methyl

# After the author has explicitly verified a finalized real-data run:
python -m bayesbreak.experiments.realdata --dataset cgh --verified
# (or: BAYESBREAK_VERIFIED=1)
```

The same entry points are exposed via the `bayesbreak` CLI:

```bash
bayesbreak synthetic --all
bayesbreak realdata --dataset cgh --verified
```

Real-data figures **default to placeholder mode** with a translucent
"PLACEHOLDER" watermark and a sidecar JSON (`<fig>.json`) containing the
raw-data hash, preprocessing hash, fit hyperparameters, and DP diagnostics.
Per the report's §6 placeholder convention, only `--verified` (or
`BAYESBREAK_VERIFIED=1`) clears the watermark — and only after author
approval. Simulated-fallback bundles are *always* placeholders regardless
of the flag.

## Citing

If you use BayesBreak in your work, please cite the accompanying arXiv
preprint: [arXiv:2603.14681](https://arxiv.org/abs/2603.14681).

```bibtex
@article{solari2026bayesbreak,
  title   = {Generalized Hierarchical Bayesian Segmentation with Irregular Designs,
             Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs},
  author  = {Solari, Omid Shams},
  journal = {arXiv preprint arXiv:2603.14681},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.14681},
}
```

See also [`CITATION.cff`](CITATION.cff) and the full report at
[`docs/report/bayesbreak.pdf`](docs/report/bayesbreak.pdf).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The test suite (`pytest tests/`)
includes 133 conceptual-correctness tests — brute-force DP comparisons,
closed-form predictive checks, length-prior plumbing, EM monotonicity,
Beta-NegBin moment scale, replicates pooling, stability-bound bounds, PIT
uniformity, and the diagnostics module.
