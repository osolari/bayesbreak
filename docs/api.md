# API reference

This page highlights the most commonly used objects and methods.

## Base estimator

`bayesbreak.base.BayesBreakBase`

- `fit(X, y=None)`
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
- `"beta"`

Example:

```python
from bayesbreak import make_bayesbreak
m = make_bayesbreak("poisson", k_max=25).fit(y)
```
