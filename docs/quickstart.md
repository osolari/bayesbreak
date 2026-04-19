# Quickstart

## Strict sklearn API

Every segmenter takes `fit(X, y, sample_weight=...)`, where `X` is a 2-D design
matrix (or 1-D indices that are reshaped to `(n, 1)`) and `y` is the ordered
response sequence.

## Gaussian segmentation

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

rng = np.random.default_rng(0)
mu = np.r_[np.zeros(50), np.ones(50), -0.5 * np.ones(50)]
y = mu + 0.25 * rng.standard_normal(mu.size)
X = np.arange(mu.size).reshape(-1, 1)

model = BayesBreakGaussian(k_max=10, regression_curve="mix_k").fit(X, y)

print("k_map           :", model.k_map_)
print("MAP boundaries  :", model.map_boundaries_)
print("log p(y)        :", model.log_evidence_)
print("score (held-out):", model.score(X, y))

pc_fit = model.predict(X)             # MAP piecewise-constant curve
bayes_curve = model.bayes_curve_mean_ # posterior mean latent signal
seg_index = model.transform(X)        # per-point segment label
```

## Poisson counts

```python
import numpy as np
from bayesbreak import BayesBreakPoisson

rng = np.random.default_rng(0)
lam = np.r_[3 * np.ones(60), 10 * np.ones(60)]
y = rng.poisson(lam)
X = np.arange(y.size).reshape(-1, 1)

model = BayesBreakPoisson(k_max=6).fit(X, y)
print(model.map_boundaries_)
```

## Scikit-learn Pipeline and GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from bayesbreak import BayesBreakGaussian

pipe = Pipeline([("scale", StandardScaler()), ("seg", BayesBreakGaussian())])
grid = GridSearchCV(pipe, {"seg__k_max": [4, 8, 16]}, cv=TimeSeriesSplit(3))
grid.fit(X, y)
print(grid.best_params_)
```

## Factory helper

```python
from bayesbreak import make_bayesbreak

model = make_bayesbreak("binomial", k_max=8, n_trials=50)
```
