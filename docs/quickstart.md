# Quickstart

## Gaussian segmentation

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

# Synthetic piecewise constant signal
rng = np.random.default_rng(0)
mu = np.r_[np.zeros(50), np.ones(50), -0.5*np.ones(50)]
y = mu + 0.25 * rng.standard_normal(mu.size)

m = BayesBreakGaussian(k_max=10, regression_curve="mix_k").fit(y)

# piecewise-constant posterior mean on the MAP-like segmentation
pc = m.predict()

# optional Bayesian regression curve
brc = m.get_regression_curve()

print("k* =", m.get_segment_count())
print("boundaries =", m.get_boundaries())
print("log evidence =", m.score())
```

## Poisson counts segmentation

```python
import numpy as np
from bayesbreak import BayesBreakPoisson

rng = np.random.default_rng(0)
lam = np.r_[3*np.ones(60), 10*np.ones(60)]
y = rng.poisson(lam)

m = BayesBreakPoisson(k_max=6).fit(y)
print("boundaries =", m.get_boundaries())
```

## Factory helper

```python
from bayesbreak import make_bayesbreak
m = make_bayesbreak("binomial", k_max=8, n_trials=50)
```
