# BayesBreak

BayesBreak implements Bayesian piecewise-constant regression (segmentation) via
dynamic programming. The package is **distribution-aware** through a small set
of conjugate likelihood families (Gaussian, Poisson, Binomial, Bernoulli, and
Beta-valued fractional Beta--Binomial).

The public API follows scikit-learn conventions (`fit`, `predict`, `score`) and
exposes additional posterior objects such as boundary posteriors and Bayesian
regression curves. It also supports:

- Per-observation `sample_weight` across all conjugate families (e.g., to model
  heteroscedasticity, exposures, or missingness via zero weights).
- A multivariate wrapper that performs **shared-boundary** segmentation for
  vector-valued observations (independent channels under a shared partition).
- A grouped interface for **group-membership scoring** and MAP signal evaluation.

## Installation (editable)

```bash
pip install -e .
```

SciPy is optional but recommended for faster special functions:

```bash
pip install scipy
```

## Quickstart

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

# synthetic piecewise-constant signal
rng = np.random.default_rng(0)
y = np.r_[rng.normal(0.0, 0.4, 80), rng.normal(2.0, 0.4, 60), rng.normal(-1.0, 0.4, 70)]

m = BayesBreakGaussian(k_max=15, regression_curve="mix_k")
m.fit(y)

print("k*:", m.get_segment_count())
print("boundaries:", m.get_boundaries())

pc = m.predict()                   # MAP-like piecewise-constant estimate
brc = m.get_regression_curve()     # optional Bayesian regression curve
```

## Sample weights

All families accept `sample_weight` in `fit`. A common pattern is to encode
heteroscedasticity (larger weight = higher confidence) or missingness (weight
0.0) without changing array shapes.

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

y = np.array([0.0, 0.1, 4.0, 4.2, 4.1])
w = np.array([1.0, 1.0, 0.2, 0.2, 0.2])  # downweight the high segment

m = BayesBreakGaussian(k_max=10)
m.fit(y, sample_weight=w)
print(m.get_boundaries())
```

## Bernoulli (Beta--Bernoulli)

For binary sequences use `BayesBreakBernoulli`:

```python
import numpy as np
from bayesbreak import BayesBreakBernoulli

y = np.r_[np.zeros(40), np.ones(40)]
m = BayesBreakBernoulli(k_max=10)
m.fit(y)
print(m.get_boundaries())
```

## Multivariate wrapper

Use `BayesBreakMultivariate` to segment vector-valued observations with shared
boundaries under an independent-channel likelihood:

```python
import numpy as np
from bayesbreak import BayesBreakGaussian, BayesBreakMultivariate

rng = np.random.default_rng(0)
n = 120
y = np.c_[np.r_[np.zeros(60), np.ones(60)], np.r_[np.zeros(60), 2*np.ones(60)]]
y = y + 0.2 * rng.standard_normal(y.shape)

mv = BayesBreakMultivariate(BayesBreakGaussian(k_max=15))
mv.fit(y)
print(mv.get_boundaries())
```

## Group membership scoring and MAP signal evaluation

`BayesBreakGrouped` trains group-specific hyperparameters from labeled
sequences, scores new sequences by group marginal likelihood, and can produce
group-conditional MAP fits:

```python
import numpy as np
from bayesbreak import BayesBreakGaussian, BayesBreakGrouped

rng = np.random.default_rng(0)
X = [rng.normal(loc=-2.0, scale=0.5, size=80), rng.normal(loc=+2.0, scale=0.5, size=80)]
y = np.array(["A", "B"], dtype=object)

clf = BayesBreakGrouped(BayesBreakGaussian(k_max=10))
clf.fit(X, y)

test = rng.normal(loc=+2.0, scale=0.5, size=80)
print("p(g|y):", clf.predict_proba([test])[0])
print("pred:", clf.predict([test])[0])
fit = clf.map_signal([test])[0]
```

## Reproducing figures and tables

All scripts for figures and tables live under `scripts/`:

- `scripts/figures/` produces PNG/PDF figures into `artifacts/figures/`.
- `scripts/tables/` produces CSV/Markdown tables into `artifacts/tables/`.

Run everything:

```bash
python scripts/make_all_artifacts.py
```

## Documentation

Markdown documentation intended for MkDocs lives in `docs/`.

