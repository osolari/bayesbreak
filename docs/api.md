# API reference

## Core segmenter

`bayesbreak.base.BayesBreakSegmenter` is the abstract, scikit-learn compatible
base class. Every concrete family inherits from it.

### Methods

- `fit(X, y, *, sample_weight=None)` — strict sklearn contract.
- `predict(X, *, mode="map")` — piecewise-constant curve at query points.
  `mode="bayes"` returns the posterior-mean curve (requires
  `regression_curve != "none"` at fit time).
- `score(X, y, sample_weight=None)` — mean posterior-predictive log-density
  (§8 of the report; use this as the sklearn CV target).
- `transform(X)` — segment indices `(0, …, k_map - 1)` at query points
  (BayesBreak acts as a featurizer in a Pipeline).
- `get_map_segmentation()` — `(k_map, map_boundaries, map_segment_means)` tuple
  for consumers that want the classical change-point output.
- `get_log_evidence()` — `log p(y)` on the training sequence.

### Fitted attributes

| Attribute | Meaning |
|---|---|
| `n_` | Training sequence length |
| `x_design_` | `(n,)` stored design points |
| `hyper_` | Family-specific hyperparameter dict |
| `log_block_evidence_` | `(n+1, n+1)` triangular `log A^0_{ij}` |
| `block_first_moment_` | `(n+1, n+1)` linear `A^1_{ij}` |
| `log_left_`, `log_right_` | Sum-product DP tables |
| `log_evidence_` | `log p(y)` |
| `k_posterior_` | `P(k | y)` |
| `k_map_` | `argmax_k P(k | y)` |
| `boundary_marginals_` | `P(b_i = 1 | y)` |
| `boundary_location_posterior_` | `P(t_p = h | y, k_map)` per boundary |
| `map_boundaries_` | Joint MAP boundary vector |
| `map_segment_means_` | Posterior mean per MAP segment |
| `map_curve_` | Piecewise-constant fit on training indices |
| `bayes_curve_mean_` | Posterior mean latent signal (optional) |
| `sample_weight_` | Weights used at fit time |

## Families

- `BayesBreakGaussian`
- `BayesBreakPoisson`
- `BayesBreakBinomial`
- `BayesBreakBernoulli`
- `BayesBreakBeta`
- `BayesBreakBetaObs`
- `BayesBreakLogisticNormal`

## Wrappers

- `SharedBoundaryMultivariateSegmenter(base_estimator, k_max=None)` —
  single MAP segmentation across channels.
- `IndependentMultivariateSegmenter(base_estimator, k_max=None)` —
  per-channel fit.
- `BayesBreakGroupedClassifier(base_estimator, class_prior="empirical")` —
  supervised group classifier; `predict_proba` returns `(n_sequences, G)`.
- `BayesBreakMixtureClassifier(base_estimator, n_groups, max_iter, ...)` —
  latent-group EM.

## Factory

```python
bayesbreak.make_bayesbreak(family, **kwargs)
```

Valid `family` strings: `gaussian`, `poisson`, `binomial`, `bernoulli`, `beta`,
`beta-obs`, `logistic-normal` (plus a few aliases).

## Standalone DP primitives

`bayesbreak.dp` exposes the DP layer directly so it can be composed with custom
block routines:

- `forward_backward(log_block_evidence, n, k_max)`
- `posterior_over_k(log_left, n, k_max)`
- `boundary_event_marginals(log_left, log_right, log_posterior_k, n, k_max)`
- `boundary_location_posterior(log_left, log_right, n, k)`
- `max_sum_segmentation(log_block_evidence, k, *, log_length_prior=None)`
- `bayes_regression_curve_fixed_k(...)`, `bayes_regression_curve_mixed_k(...)`

## Posterior predictive

`bayesbreak.prediction`:

- `posterior_predictive_logpdf(estimator, X_new, y_new, per_sample=False)`
- `held_out_log_likelihood_trace(estimator, X_new, y_new, prefix_fractions=None)`
