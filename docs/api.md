# API reference

## Core segmenter

`bayesbreak.base.BayesBreakSegmenter` is the abstract, scikit-learn compatible
base class. Every concrete family inherits from it.

### Methods

- `fit(X, y, *, sample_weight=None)` — strict sklearn contract.
- `predict(X, *, mode="map", extrapolation="error")` — piecewise-constant
  curve at query points with explicit coordinate-support handling.
  `mode="bayes"` returns the posterior-mean curve (requires
  `regression_curve != "none"` at fit time).
- `score(X, y, sample_weight=None)` — mean posterior-predictive log-density
  (§8 of the report; use this as the sklearn CV target).
- `transform(X, *, extrapolation="error")` — segment indices
  `(0, …, k_map - 1)` at query points
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
| `map_curve_` | Piecewise-constant fit on training indices — the *exported segmentation* in the manuscript sense (`def:exported-segmentation`) |
| `bayes_curve_mean_` | Posterior mean latent signal (optional) — the *Bayes curve* (`def:bayes-curve`) |
| `sample_weight_` | Weights used at fit time |
| `admissibility_mask_` | Boolean `(n+1, n+1)` mask of finite cells in `log_block_evidence_`; the DP and `compute_log_C_k` share this mask (the admissibility contract of §`sec:setup`). |

### Class attributes

- `MOMENT_SIGN_CONTRACT: str` — `"signed"` (Gaussian) or `"nonneg"` (all
  other families). Declares whether `block_first_moment_` can change
  sign on admissible cells; §5 signed-moment storage guidance.

### Manuscript cross-references

- Standing assumption: `ass:standing-offline` (offline segmentation
  support and block independence).
- Time and space complexity: `prop:bb-complexity` (`Θ(n²)` block
  precomputation, `Θ(k_max n²)` DP, `Θ(k_max n)` working memory for
  evidences and `Θ(k_max n²)` when retaining backpointers).
- DP invariants: `prop:fb-duality` (forward/backward total-evidence
  identity — exercised by `run_dp_diagnostics`),
  `prop:block-covering-decomposition` (Bayes-curve moments — implemented
  by `bayes_regression_curve_*`).
- Joint MAP correctness: `thm:map-correctness` (max-sum + backtrack
  returns the joint MAP at the chosen `k`; ties broken deterministically
  by the back-pointer convention).
- Block-evidence per family: `prop:gaussian-block`, `prop:poisson-block`,
  `prop:binomial-block`, `prop:negbin-block`, `prop:beta-block`.
- Non-conjugate approximations:
  `ass:uniform-block-error` (uniform per-block log-evidence error `ε`),
  `prop:stability` (posterior-odds stability),
  `obl:routine-certification-paper` (routine-specific error control remains
  a proof obligation),
  and `cor:probability-error-conversion` (TV bound on `P(k|y)` —
  reported as `pk_tv_upper_bound` in `run_non_conjugate_diagnostics`).
- Shared-boundary replicates:
  `ass:cond-indep-subjects` (common-grid + conditional independence),
  `prop:shared-boundary-identifiability` (when the identifying-block
  hypothesis `rem:identifying-block` holds), implemented by
  `SharedBoundaryReplicatesSegmenter`.
- Latent-group mixture: `prop:latent-identifiability` (identifiable up
  to label permutation), `ex:label-switch-counterexample`,
  `rem:teicher-overspec` (overspecified-`G` redundancy — mitigate by
  held-out `G` selection).
- Inherited partitions / irregular designs:
  `cor:inherited-partition-invariance` (`g(Δ)` interacts with prior
  weight only, not with the likelihood).
- Prediction: `def:prediction-cases` (Case A pointwise — see
  `posterior_predictive_logpdf`; Case B set-valued — see
  `prediction.Unit` + `predict_group`; Case C vector-valued — see
  `SharedBoundaryMultivariateSegmenter` / `IndependentMultivariateSegmenter`),
  `def:segment-assignment-map` (the carry from a new design point to its
  containing MAP segment — `_assign_to_map_blocks`),
  `ass:prediction-independence` (per-unit conditional independence
  given the exported group model).
- Definitions of empirical metrics:
  `def:metric-boundary-f1`, `def:metric-boundary-mae`,
  `def:metric-ece-boundary`, `def:metric-loglik` (boundary F1, MAE,
  ECE, held-out predictive log-likelihood — reported by
  `bayesbreak.diagnostics` and `bayesbreak.prediction`).

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

The latent-group estimator optimizes the finite-template objective stated in
the report; it is not a normalized finite-mixture likelihood. Successful fits
expose `objective_trace_`, `final_objective_`, `selected_restart_`, and one
`RestartDiagnostic` per attempted restart. Nonfinite, stale, or nonmonotone
traces are excluded from restart selection; objective ties select the earliest
seeded restart before canonical template-label ordering. `objective_` remains a
compatibility alias for `objective_trace_`.

For exact shared-boundary pooling, `bayesbreak.groups.shared_fit(...)` fits a
`SharedBoundaryReplicatesSegmenter`. Subject block log evidences are combined by
`bayesbreak.replicates.aggregate_block_log_evidence(...)` without exponentiating
unbounded values; `-inf` remains structural zero support and `NaN`/`+inf` are
rejected. The fitted model exposes subject-specific MAP-segment means and a
bounded `block_posterior_mean_` average diagnostic. It does not construct a
single pooled first moment for distinct subject parameters.

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

## Partition priors

`bayesbreak.priors.PartitionPriorConfig` declares segment cohesion separately
from the interior-boundary hazard:

- `log_cohesion(start, stop, x, config)` scores the complete segment. Supported
  modes are uniform, physical-span power, and explicit per-segment factors;
  minimum segment length/span and maximum span can assign zero support.
- `bayesbreak.design_prior.log_boundary_hazard(boundary, x, config)` scores an
  interior endpoint only. Supported modes are uniform, explicit factors, and
  fixed-count Poisson interval occupancy.
- `partition_log_prior(boundaries, x, config)` combines one cohesion per segment
  with one hazard per nonterminal boundary.

For the Poisson occupancy mode, candidate interval `j` uses local odds
`exp(Lambda_j) - 1`, where `Lambda_j` is its integrated intensity. It does not
use occupancy probability `1 - exp(-Lambda_j)`, and it does not reinterpret a
physical-span cohesion as a Poisson-process prior.

Every observation-family estimator accepts `partition_prior=config`. The same
local prior table is used by the partition normalizer, sum-product posterior
recursions, max-sum joint-MAP recursion, and Bayes curves. The legacy
`length_prior` argument remains supported as an additional segment cohesion.

## Posterior predictive

`bayesbreak.prediction`:

- `posterior_predictive_logpdf(estimator, X_new, y_new, per_sample=False)`
- `held_out_log_likelihood_trace(estimator, X_new, y_new, prefix_fractions=None)`

All public prediction wrappers use `assign_to_partition(...)` and default to
`extrapolation="error"`. Named alternatives are `"clip"` for legacy clipping
on both sides, `"left_endpoint"` for left-only endpoint extension, and
`"right_endpoint"` for right-only extension. Exact fitted endpoints are
in-support, and unsorted query order is preserved. Each call records
`prediction_metadata_` and `prediction_provenance_` with the selected policy
and fitted coordinate support.

`BayesBreakBetaObs` overrides the generic fallback with its fitted observation
family. For each MAP block it integrates
`Beta(y_new | phi_new * mu, phi_new * (1 - mu))` over the fitted quadrature
posterior for `mu`. In this family, prediction `sample_weight` values are the
known positive `phi_new` descriptors; training likelihood-power weights remain
separate in the fitted estimator. Values outside the open interval `(0, 1)`
receive log density `-inf`.

## Diagnostics

`bayesbreak.diagnostics`:

- `run_dp_diagnostics(estimator)` — checks the four §4 invariants
  (`Σ P(k) = 1`, forward/backward total-evidence identity per
  `prop:fb-duality`, `Σ P(b_i|y,k) = k − 1` stated inline in the DP
  correctness theorem, MAP backtrack score matching
  `thm:map-correctness`).
- `run_non_conjugate_diagnostics(estimator, reference)` — reachable-block
  error quantiles (the empirical `ε` of `ass:uniform-block-error`),
  posterior-sensitivity summaries, and the worst-case total-variation
  bound on `P(k|y)` from `cor:probability-error-conversion` (fields
  `pk_tv_empirical`, `pk_tv_upper_bound`; check `pk_tv_bound_check`).
  Each check carries a `failure_mode` tag aligned with the §4
  approximation-validation checklist and §5b limitations on the
  non-conjugate approximation regime.
- `bayesbreak.nonconjugate.evaluate_reachable_segment_error(...)` — records a
  hash of shared reachable block coordinates, reference discrepancy,
  optimization residual, tail bound, quadrature error, and explicit
  verified/unverifiable/failed status. `propagate_partition_bounds(...)`
  returns conditional global bounds using the maximum reachable error and
  caps total variation at one.
- `run_prior_sensitivity(estimator, *, pk_perturbations=None,
  g_variants=("uniform", "length-proportional"))` — partition-prior
  sensitivity diagnostic from §5b limitations. Reruns the DP on the
  existing `log_block_evidence_` under perturbed `p(k)` and `g`, and
  reports `Δ p(k|y)` (max / TV) and `Δ P(b_i|y, k_map)` (max / L1) per
  variant.
- `select_n_groups_by_holdout(base_estimator, sequences, *,
  g_grid=(1,2,3,4,5), n_folds=5, random_state=0, ...)` — K-fold
  held-out log-likelihood `G`-selection for the latent-template mixture.
  Mitigates the overspecified-`G` redundancy of `rem:teicher-overspec`
  per the §5b "Identifiability failures (named)" guidance, using the
  per-sequence marginal `log p(y^{(s)})` of `def:metric-loglik` as the
  scoring rule (`BayesBreakMixtureClassifier.sequence_log_likelihood`).

## Result provenance

`bayesbreak.provenance` provides versioned result sidecars without rewriting
historical assets:

- `ResultRecord` separates execution status, scientific interpretation, and
  original/corrected lineage. Executed records require data, configuration,
  code, and environment SHA-256 hashes.
- `validate_result_record(record, release_mode=True)` rejects invalid result
  identifiers, missing corrected-result parent links, missing corrected output
  hashes, and non-repository-relative artifact paths.
- `write_sidecar(path, record)` writes deterministic schema-versioned JSON.
- `read_sidecar(path, migration_manifest=...)` reads current records or applies
  a hash-verified, in-memory path migration to an immutable legacy sidecar.

The JSON Schema 2020-12 contracts are under `schemas/`. Corrected reruns must
use a new result ID and identify their historical parent; reading a legacy
record does not authorize changing its archived bytes or interpretation.

## Boundary metrics

`bayesbreak.metrics.match_boundaries_one_to_one(predicted, reference,
tolerance)` first maximizes the number of eligible matches and then minimizes
their total absolute distance under deterministic input ordering. No predicted
or reference boundary is reused. `boundary_metrics(...)` returns precision,
recall, F1, matched MAE (`None` when no pair matches), axis names, reference
type, matching rule, and metric version; `to_dict()` follows the versioned
boundary-metric schema.

Comparisons with BayesBreak MAP boundaries must use a reference type such as
`bayesbreak-map-agreement`; they are not external-truth accuracy. The cached
baseline table generator now uses this canonical metric owner.

## Comparator validation

`bayesbreak.comparators.ComparatorInputSchema` validates comparator values,
coordinate axes, task type, source provenance, and an explicit `TuningBudget`
before algorithm dispatch or metric computation. Multisequence requests require
an unflattened sequence-by-coordinate raw observation matrix and an axis whose
length matches the matrix observation dimension. `scripts/run_comparators.py`
constructs this validated raw route from `.npy` inputs.

Fitted curves and cached map traces cannot stand in for raw multisequence data.
The historical CGH cached route is rejected as `FAIL-BB-002` and retained only
as a diagnostic record; no comparator is dispatched from that artifact.

## Baselines

`bayesbreak.baselines` exposes wrappers around upstream baseline
libraries (no re-implementation). Algorithms are dispatched through

```python
bayesbreak.baselines.segment_with(algorithm, y, **kwargs) -> BaselineResult
```

| Name | Upstream | Reference |
|---|---|---|
| `pelt` | `ruptures.Pelt` | Killick, Fearnhead & Eckley (2012) |
| `optimal_partitioning` (`op`, `dynp`) | `ruptures.Dynp` | Jackson et al. (2005) |
| `binary_segmentation` (`bs`) | `ruptures.Binseg` | classical BS |
| `wild_binary_segmentation` (`wbs`) | `ruptures.Binseg` + random windows | Fryzlewicz (2014) |
| `cbs` | `DNAcopy::segment` via `rpy2` | Olshen et al. (2004) |
| `smuce` | `stepR::stepFit` via `rpy2` | Frick, Munk & Sieling (2014) |
| `rjmcmc` (`mcp`) | `mcp::mcp` via `rpy2` + JAGS | Lindeløv (2020) |
| `fearnhead_exact` (`fearnhead`) | `bayesbreak.dp` at Fearnhead-2006 prior config (labelled reference) | Fearnhead (2006) |

Install extras:

- `pip install bayesbreak[baselines]` — `ruptures`.
- `pip install bayesbreak[baselines-r]` — `rpy2`; also requires an R
  install with the Bioconductor `DNAcopy` package.

Each :class:`BaselineResult` records `boundaries`, `k`, `algorithm`,
upstream `package` + `package_version`, and the full `tuning` dict.
