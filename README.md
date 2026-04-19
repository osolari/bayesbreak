# BayesBreak

[![CI](https://github.com/osolari/bayesbreak/actions/workflows/ci.yml/badge.svg)](https://github.com/osolari/bayesbreak/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Exact Bayesian segmentation with a scikit-learn compatible API.**

BayesBreak turns the two-layer design of the accompanying report into a small,
reusable library:

- **Block evidence** — a family-specific integrated single-segment marginal
  likelihood `A^0_{ij}` (Gaussian, Poisson, Binomial, Bernoulli, Beta,
  Beta-observation, Logistic-Normal).
- **Dynamic programming** — a distribution-agnostic engine that delivers, from
  that block matrix alone, the marginal evidence `p(y)`, the segment-count
  posterior `P(k|y)`, boundary-event marginals, the joint MAP segmentation
  (max-sum + backtracking), and the Bayesian regression curve.

Every estimator inherits from `sklearn.base.BaseEstimator` and can live inside
an `sklearn.pipeline.Pipeline`.

## Install

```bash
pip install bayesbreak                # runtime + core deps
pip install "bayesbreak[plots]"       # + matplotlib/seaborn
pip install "bayesbreak[datasets]"    # + real-data loaders
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
| `boundary_marginals_` | `P(b_i=1 \| y)` for interior indices |
| `log_evidence_` | `log p(y)` of the training sequence |
| `bayes_curve_mean_` | Posterior-mean latent signal (when `regression_curve != "none"`) |

## Supported families

```python
from bayesbreak import (
    BayesBreakGaussian,        # Normal-Normal (weighted)
    BayesBreakPoisson,         # Poisson-Gamma (exposure)
    BayesBreakBinomial,        # Beta-Binomial
    BayesBreakBernoulli,       # Beta-Bernoulli
    BayesBreakBeta,            # fractional Beta-Binomial for y in (0,1)
    BayesBreakBetaObs,         # Beta likelihood + Beta prior (1-D quadrature)
    BayesBreakLogisticNormal,  # Bernoulli + Normal prior on log-odds
    make_bayesbreak,           # factory: make_bayesbreak("poisson", k_max=20)
)
```

## Multivariate and hierarchical wrappers

- `SharedBoundaryMultivariateSegmenter` — single segmentation across channels.
- `IndependentMultivariateSegmenter` — per-channel fit.
- `BayesBreakGroupedClassifier` — supervised group classification with
  group-specific block families (`predict_proba` returns `(n_sequences, G)`).
- `BayesBreakMixtureClassifier` — latent-group EM mixture (see §4.7 of the
  report for the template-mixture objective).

## How it maps to the paper

| Paper equation / theorem | Implementation |
|---|---|
| Block evidence `A^0_{ij}` | `_compute_block_evidence` per family in `bayesbreak.families` |
| Forward / backward sum-product (§4.3) | `bayesbreak.dp.forward_backward` |
| `P(k \| y)`, `log p(y)` | `bayesbreak.dp.posterior_over_k` |
| Boundary-event marginals | `bayesbreak.dp.boundary_event_marginals` |
| Joint MAP via max-sum + backtracking (§4.4) | `bayesbreak.dp.max_sum_segmentation` |
| Bayes regression curve (§4.3) | `bayesbreak.dp.bayes_regression_curve_{fixed,mixed}_k` |
| Posterior-predictive (§8, Prop. `ef-predictive`) | `bayesbreak.prediction.posterior_predictive_logpdf` |
| Latent-group EM (§4.7) | `bayesbreak.mixture.BayesBreakMixtureClassifier` |
| Non-conjugate block approximations (§5) | `BayesBreakLogisticNormal(approx=...)`, `BayesBreakBetaObs` |

## Reproducing figures and tables

```bash
bayesbreak reproduce figures   # runs scripts/figures/*.py → docs/report/{figures,tables}
bayesbreak reproduce tables    # runs scripts/tables/*.py  → docs/report/{figures,tables}
bayesbreak reproduce all
```

## Citing

See [`CITATION.cff`](CITATION.cff) and `docs/report/bayesbreak.pdf`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The test suite (`pytest tests/`)
includes conceptual-correctness tests — brute-force DP comparisons,
closed-form predictive checks, EM convergence, and the full scikit-learn
contract.
