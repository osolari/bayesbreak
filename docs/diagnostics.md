# Diagnostics

`bayesbreak.diagnostics` collects the practical checks that go alongside
every fit: DP invariants, non-conjugate approximation error, prior
sensitivity, and held-out `G`-selection for the latent-template mixture.

## Quick reference

| Routine | What it answers | Returns |
|---|---|---|
| `run_dp_diagnostics(est)` | "Did the DP solve itself correctly?" | `DiagnosticReport` with 4 invariants |
| `run_non_conjugate_diagnostics(est, ref)` | "How much does my non-conjugate approximation perturb the posterior?" | block-error quantiles, TV bound, theoretical-rate annotation |
| `run_prior_sensitivity(est)` | "How much does the chosen `p(k)` and `g(ℓ)` affect my conclusions?" | `Δ p(k|y)` and `Δ P(b_i|y, k_map)` under prior perturbations |
| `select_n_groups_by_holdout(base, sequences)` | "How many latent groups should I use?" | K-fold held-out marginal log-likelihood per `G` |

Every diagnostic returns a `DiagnosticReport` with a list of
`DiagnosticCheck` items. Each check has a name, a pass/fail flag, a
human-readable detail, an optional `failure_mode` tag (aligned with
§5b limitations), and a `measured` value. Reports serialise to JSON.

## `run_dp_diagnostics`

Checks the four §4 invariants on a fitted segmenter:

1. `∑_k P(k | y) = 1` — segment-count posterior normalisation.
2. `L̃[k_map, n] = R̃[k_map, 0]` — forward / backward total-evidence
   identity (Proposition `prop:fb-duality`).
3. `∑_i P(b_i = 1 | y, k_map) = k_map − 1` — boundary-event normalisation
   identity, stated inline in the DP correctness theorem.
4. The backtracked max-sum score matches the stored `log_joint_map_`
   (Theorem `thm:map-correctness`).

```python
from bayesbreak import BayesBreakGaussian, run_dp_diagnostics

est = BayesBreakGaussian(k_max=8).fit(X, y)
report = run_dp_diagnostics(est)
print(report.summary)            # e.g. "4/4 checks passed"
print(report.to_json(indent=2))  # full JSON record
```

## `run_non_conjugate_diagnostics`

Compares a non-conjugate fit to a higher-accuracy reference (typically a
high-node Gauss–Hermite quadrature fit on the same data) and reports:

- `block_error_max`, `block_error_q95`, `block_error_median` over the
  reachable-block mask (the empirical uniform `ε` named in
  Assumption `ass:uniform-block-error`);
- `k_posterior_l1` and `boundary_marginal_l1` between the two posteriors;
- `pk_tv_upper_bound = exp(2 k_max ε_max) − 1`, the worst-case TV bound
  from Corollary `cor:probability-error-conversion`;
- `pk_tv_empirical = 0.5 · |P̂(k|y) − P_ref(k|y)|_1`;
- per-routine `theoretical_rate` (e.g. `"O(n^{-1}) on reachable blocks"`
  for Laplace / JJ / PG; `"O(Q^{-2r})"` for Gauss–Hermite;
  `"EP: not uniformly bounded"` for true EP);
- `theoretical_rate_violated` — `True` when the empirical ε exceeds the
  routine's expected rate by an order of magnitude, or when EP failed to
  converge on at least one block.

```python
from bayesbreak import BayesBreakLogisticNormal, run_non_conjugate_diagnostics

ref = BayesBreakLogisticNormal(approx="quadrature", gh_points=120).fit(X, y)
fit = BayesBreakLogisticNormal(approx="laplace").fit(X, y)

report = run_non_conjugate_diagnostics(fit, ref)
print(report.extra["block_error_max"])
print(report.extra["pk_tv_upper_bound"])
print(report.extra["theoretical_rate"])
print(report.extra["theoretical_rate_violated"])
```

## `run_prior_sensitivity`

Reruns the DP under perturbations of `p(k)` (uniform, geometric `0.8^k`,
`1/k`) and the length factor `g(ℓ)` (uniform, length-proportional). Each
variant reports

- `delta_pk_max`, `delta_pk_tv` — max-absolute and total-variation
  change in `P(k|y)` from the fitted estimator;
- `delta_bm_max`, `delta_bm_l1` — max-absolute and `L^1` change in the
  fixed-`k_map` boundary marginals.

Use it before drawing prior-sensitive conclusions from segment counts
or boundary positions — the §5b *Partition-prior sensitivity* paragraph
explicitly calls this out.

## `select_n_groups_by_holdout`

K-fold held-out marginal log-likelihood selection of `G` for the latent
template-mixture. Mitigates the overspecified-`G` redundancy of
`rem:teicher-overspec`: the saturated-`G` identifiability of
Proposition `prop:latent-identifiability` does not prevent two distinct
parameter tuples from inducing identical mixture densities when `G > G*`.

```python
from bayesbreak import BayesBreakGaussian, select_n_groups_by_holdout

report = select_n_groups_by_holdout(
    BayesBreakGaussian(k_max=4),
    sequences,             # list of 1-D arrays, all length n
    g_grid=(1, 2, 3, 4),
    n_folds=5,
    random_state=0,
)
print(report.extra["best_g"])
print(report.extra["mean_test_loglik"])
```

The scoring rule is the per-sequence marginal log-likelihood
`log p(y^{(s)}) = log ∑_g π_g · S_g(y^{(s)}; τ_g)` exposed as
`BayesBreakMixtureClassifier.sequence_log_likelihood` — the same
quantity named `def:metric-loglik` in §6.

## Failure-mode tags

Every `DiagnosticCheck` may carry a `failure_mode` tag:

| Tag | Meaning |
|---|---|
| `"short-segment-laplace"` | Laplace expansion inaccuracy on short blocks |
| `"mf-vb-variance"` | mean-field variational variance underestimation |
| `"method-sensitivity"` | posterior summary depends heavily on the choice of approximation method |
| `"ep-nonconvergence"` | EP iteration failed to converge or oscillates |
| `"tv-bound"` | total-variation bound on `P(k|y)` from `cor:probability-error-conversion` |
| `"prior-sensitivity"` | partition-prior perturbation diagnostic |
| `"teicher-overspec"` | overspecified-`G` mixture redundancy (`rem:teicher-overspec`) |
| `"abs-prob-tv-bound"` | (legacy alias for `"tv-bound"`) |

These align with the §4 approximation-validation checklist and §5b
limitations. Use them to filter reports programmatically:

```python
report = run_non_conjugate_diagnostics(fit, ref)
failures = [c for c in report.checks if c.failure_mode == "ep-nonconvergence"]
```
