# Quickstart

A five-minute introduction. For the conceptual map first, see
[Concepts](concepts.md); for the full method list, see [API
reference](api.md).

## Install

```bash
pip install bayesbreak
```

Optional extras — see [Installation](installation.md) for the full list:

```bash
pip install "bayesbreak[plots,datasets,baselines]"
```

## A three-segment Gaussian fit

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

rng = np.random.default_rng(0)
n = 150
mu = np.r_[np.zeros(50), np.ones(50), -0.5 * np.ones(50)]
y = mu + 0.25 * rng.standard_normal(n)
X = np.arange(n).reshape(-1, 1)

est = BayesBreakGaussian(k_max=10, regression_curve="mix_k").fit(X, y)

est.k_map_                 # posterior-mode segment count
est.map_boundaries_        # joint-MAP boundary vector
est.boundary_marginals_    # P(b_i = 1 | y, k_map)
est.k_posterior_           # P(k | y)
est.bayes_curve_mean_      # posterior-mean latent signal
est.score(X, y)            # mean posterior-predictive log-density
```

The model is a strict scikit-learn estimator: `fit`, `predict`, `score`,
`transform`, `get_params`, `clone`, and `Pipeline`/`GridSearchCV`
plumbing all work as expected.

## Built-in diagnostics

```python
from bayesbreak import run_dp_diagnostics

report = run_dp_diagnostics(est)
print(report.summary)          # "4/4 checks passed"
print(report.to_json(indent=2))
```

`run_dp_diagnostics` verifies the four §4 invariants on every fit:
$\sum_k P(k\mid y)=1$, forward-backward total-evidence identity
(`prop:fb-duality`), boundary-event sum-to-$k-1$, and MAP backtrack
consistency (`thm:map-correctness`). See [Diagnostics](diagnostics.md)
for the full diagnostic catalogue.

## Counts and proportions

The same DP backend works across families — only the per-block routine
changes:

=== "Poisson"

    ```python
    from bayesbreak import BayesBreakPoisson

    y = rng.poisson(np.r_[3 * np.ones(60), 10 * np.ones(60)])
    est = BayesBreakPoisson(k_max=6).fit(np.arange(y.size).reshape(-1, 1), y)
    print(est.map_boundaries_)
    ```

=== "Binomial"

    ```python
    from bayesbreak import BayesBreakBinomial

    p = np.r_[0.2 * np.ones(50), 0.6 * np.ones(50)]
    y = rng.binomial(20, p)
    est = BayesBreakBinomial(k_max=4, n_trials=20).fit(np.arange(y.size).reshape(-1, 1), y)
    print(est.map_boundaries_)
    ```

=== "Beta proportions"

    ```python
    from bayesbreak import BayesBreakBetaObs

    mu = np.r_[0.2 * np.ones(50), 0.8 * np.ones(50)]
    y = np.clip(mu + 0.05 * rng.standard_normal(mu.size), 0.01, 0.99)
    est = BayesBreakBetaObs(k_max=4, phi=50.0).fit(np.arange(y.size).reshape(-1, 1), y)
    print(est.map_boundaries_)
    ```

## scikit-learn Pipeline / GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler()),
                 ("seg",   BayesBreakGaussian())])

grid = GridSearchCV(pipe,
                    {"seg__k_max": [4, 8, 16]},
                    cv=TimeSeriesSplit(3))
grid.fit(X, y)
print(grid.best_params_)
```

The default `score` is the per-sample posterior-predictive log-density
under the §4 prediction interface — a sensible CV criterion.

## Factory helper

```python
from bayesbreak import make_bayesbreak

model = make_bayesbreak("binomial", k_max=8, n_trials=50)
```

Available family strings: `"gaussian"` (`"normal"`), `"poisson"`
(`"count"`), `"binomial"`, `"bernoulli"` (`"binary"`), `"beta"`
(`"fractional"`), `"beta-obs"`, `"logistic-normal"`, `"negbin"`
(`"negative-binomial"`).

## Next steps

- [Concepts](concepts.md) — the block-evidence / DP separation, why
  joint MAP ≠ vector of marginal modes, and the §5b limitations as
  first-class diagnostics.
- [Tutorials](tutorials/01_quickstart.ipynb) — runnable notebooks
  covering quickstart, multivariate, real-data showcase, diagnostics,
  baselines comparison, large-$n$, and latent-group EM.
- [Results](results.md) — the §6 figures and numbers embedded inline.
