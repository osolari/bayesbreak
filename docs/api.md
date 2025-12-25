# API reference

This page highlights the most commonly used objects and methods.

## Base estimator

`bayesbreak.base.BayesBreakBase`

- `fit(X, y=None, sample_weight=None)`
- `predict(X=None)`
- `score(...)` returns the log-evidence `log P(y)`.

Attributes created by `fit`:

- `n_`: number of observations
- `hyper_`: fitted/used hyperparameters dictionary
- `C_`: posterior over number of segments, shape `(k_max,)`
- `k_ml_`: selected number of segments
- `boundary_post_`: marginal posterior probability that each interior index is a changepoint
- `boundaries_`: selected boundaries `[0, ..., n]`
- `pc_fit_`: piecewise-constant posterior mean signal
- `brc_`: Bayes regression curve (optional)

## Factory

`bayesbreak.make_bayesbreak(family: str, **kwargs)`

Constructs the right estimator for a family name:

- `"gaussian"`
- `"poisson"`
- `"binomial"`
- `"bernoulli"`
- `"beta"`

## Wrappers and group prediction

- `bayesbreak.multivariate.BayesBreakMultivariate`: shared-boundary segmentation
  for vector-valued observations `y.shape == (n, d)`.
- `bayesbreak.groups.BayesBreakGrouped`: group-membership scoring (posterior over
  known groups) and MAP signal evaluation under group-specific models.

Example:

```python
from bayesbreak import make_bayesbreak
m = make_bayesbreak("poisson", k_max=25).fit(y)
```
