# Model families

BayesBreak is implemented as a distribution-agnostic dynamic program in
`BayesBreakBase`. Model families subclass the base and provide closed-form
segment integrals.

All families share the public API:

- `fit(X, y=None)`
- `predict(X=None)`
- `score(X=None, y=None)` (log-evidence)

and expose the same posterior objects:

- `get_segment_count()` (selected `k`)
- `get_boundaries()`
- `get_boundary_posteriors()`
- `get_regression_curve()`

## Gaussian (`BayesBreakGaussian`)

- Likelihood: Normal with segment mean
- Prior: Normal prior on segment mean

Hyperparameters:

- `nu`: prior mean
- `rho2`: prior variance of segment means
- `sigma2`: observation variance

When `estimate_hyper=True`, these are estimated via moment-type estimators; user-provided values override estimates.

## Poisson (`BayesBreakPoisson`)

- Likelihood: Poisson with rate `λ`
- Prior: Gamma prior on `λ` (shape/rate)

Hyperparameters: `alpha` (shape), `beta` (rate).

## Binomial (`BayesBreakBinomial`)

- Likelihood: Binomial with trials `n_i` and success probability `p`
- Prior: Beta prior on `p`

Hyperparameters: `alpha`, `beta`.

`n_trials` can be a scalar (same trials per observation) or an array-like of length `n`.

## Beta-valued (`BayesBreakBeta`)

Targets real-valued `y ∈ (0,1)` by introducing pseudo-counts using a concentration parameter `kappa`.

Hyperparameters: `alpha`, `beta`, plus `concentration=kappa`.

This family is useful when the data represent noisy rates or proportions.
