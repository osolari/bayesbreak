# Model families

Every family subclasses `BayesBreakSegmenter` and implements three hooks:
`_estimate_hyperparameters`, `_compute_block_evidence`, and
`_segment_posterior_mean` (plus `posterior_predictive_logpdf_block` for the
prediction layer).

All families expose the full scikit-learn contract (`fit`, `predict`, `score`,
`transform`) and the same posterior attributes.

## Conjugate families

### `BayesBreakGaussian`

Weighted Normal-Normal: `y_i | μ_q ~ N(μ_q, σ² / w_i)`, `μ_q ~ N(ν, ρ²)`.

Hyperparameters: `nu`, `rho2`, `sigma2`. With `estimate_hyper=True` the code
uses moment-type estimators (adjacent covariance for `rho2` by default).

### `BayesBreakPoisson`

`y_i | λ_q ~ Poisson(λ_q w_i)`, `λ_q ~ Gamma(α, β)`.

Hyperparameters: `alpha`, `beta` (shape, rate).

### `BayesBreakBinomial`

`y_i | p_q ~ Binomial(n_i, p_q)`, `p_q ~ Beta(α, β)`.

Hyperparameters: `alpha`, `beta`, `n_trials` (scalar or array).

### `BayesBreakBernoulli`

Special case of Binomial with `n_trials ≡ 1`.

### `BayesBreakBeta`

Fractional Beta-Binomial on `y ∈ (0, 1)` — maps each `y_i` to pseudo-counts
`(κ y_i, κ (1 - y_i))` and reuses Beta-Binomial conjugacy.

Hyperparameters: `alpha`, `beta`, `concentration` (`κ`).

## Non-conjugate families

### `BayesBreakBetaObs`

`y_i | μ_q ~ Beta(φ μ_q, φ (1 - μ_q))`, `μ_q ~ Beta(α, β)`. The segment-mean
integral is computed via 1-D Gauss-Legendre quadrature on `(0, 1)`.

### `BayesBreakLogisticNormal`

`y_i | η_q ~ Bernoulli(sigmoid(η_q))`, `η_q ~ N(ν, ρ²)`. The segment-log-odds
integral is not closed-form; pick one of:

- `approx="laplace"` — Newton mode + Gaussian correction.
- `approx="jj"` — Jaakkola-Jordan quadratic bound.
- `approx="pg_vb"` — Pólya-Gamma variational Bayes (equivalent formulation).
- `approx="ep"` — expectation propagation (Gauss-Hermite moment matching).
- `approx="quadrature"` — high-accuracy Gauss-Hermite reference.

See §5 of the report for the stability bound `|Δ log odds| ≤ (k + k') ε`.
