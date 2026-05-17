# BayesBreak

Exact Bayesian segmentation with a scikit-learn compatible API.

BayesBreak separates two concerns and keeps them composable:

- **Block evidence** — a family-specific integrated single-segment marginal
  likelihood on every candidate block `(i, j]`. Closed-form for the conjugate
  exponential-family branch (Gaussian, Poisson, Binomial, Bernoulli, Beta,
  Beta-observation, Negative-Binomial), plus deterministic approximations for
  non-conjugate GLM blocks (Laplace, Jaakkola–Jordan, Pólya–Gamma mean field,
  expectation propagation, one-dimensional Gauss–Hermite quadrature).
- **Dynamic programming** — a distribution-agnostic engine that consumes the
  triangular block-evidence matrix and produces

  - the marginal evidence `log p(y)` and segment-count posterior `P(k | y)`,
  - boundary-event marginals `P(b_i = 1 | y)`,
  - the **joint** MAP segmentation (max-sum DP + backtracking — distinct from
    marginal-top-`k` summaries),
  - the Bayesian regression curve (posterior mean of the latent signal).

## Highlights

| Feature | Reference |
|---|---|
| Closed-form EF–conjugate block evidence | §4 `prop:gaussian-block`–`prop:beta-block` |
| Exact DP over partitions | §4 `thm:dp-correctness`, `prop:fb-duality` |
| Joint MAP via max-sum + backtracking | §4 `thm:map-correctness` |
| Length-aware partition prior for irregular designs | §4 `cor:inherited-partition-invariance` |
| Exact shared-boundary replicate pooling | §4 `thm:multisubject`, `prop:shared-boundary-identifiability` |
| Latent-template EM with deterministic anchor | §4 `prop:latent-identifiability`, `rem:teicher-overspec` |
| Non-conjugate block approximations | §4 `prop:stability`, `prop:uniform-bounds` |
| TV bound on `P(k|y)` | §4 `cor:probability-error-conversion` |
| Sliding-window decomposition for large `n` | §5b *Computational regime* |
| External-baseline wrappers (PELT, BS, WBS, CBS, SMUCE, RJMCMC) | §6 planned-comparator slot |

## At a glance

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

rng = np.random.default_rng(0)
n = 200
y = np.r_[rng.normal(0, 0.3, 70),
          rng.normal(2, 0.3, 60),
          rng.normal(-1, 0.3, 70)]
X = np.arange(n).reshape(-1, 1)

est = BayesBreakGaussian(k_max=10, regression_curve="fixed_k").fit(X, y)
est.k_map_                 # posterior-mode segment count
est.map_boundaries_        # joint-MAP boundary vector
est.boundary_marginals_    # P(b_i = 1 | y, k_map)
est.k_posterior_           # P(k | y)
est.bayes_curve_mean_      # posterior-mean latent signal
```

## Where to go next

- **[Quickstart](quickstart.md)** — install, fit, predict.
- **[Concepts](concepts.md)** — the block-evidence / DP separation, the
  joint-MAP-vs-marginal-mode distinction, the §5b limitations.
- **[Model families](models.md)** — conjugate and non-conjugate options.
- **[Tutorials](tutorials/01_quickstart.ipynb)** — runnable Jupyter notebooks.
- **[Diagnostics](diagnostics.md)** — TV bound, prior sensitivity, G-selection.
- **[API reference](api.md)** — full method list + manuscript cross-references.
- **[Manuscript PDF](report.md)** — the technical report this implementation
  matches one-to-one.
