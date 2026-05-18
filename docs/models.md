# Model families

BayesBreak supplies a block routine per likelihood family; the DP layer
is unchanged across families (`thm:dp-correctness`). Adding a new
likelihood is a local block-routine change.

Every family subclasses `BayesBreakSegmenter` and implements three hooks:

- `_estimate_hyperparameters(y, sample_weight)` — empirical-Bayes
  defaults; pass explicit hyperparameter overrides on the constructor
  for a fully-specified run.
- `_compute_block_evidence(y, hyper, sample_weight)` — returns the
  triangular `(log A^(0)_{ij}, A^(1)_{ij})` arrays.
- `_segment_posterior_mean(a, b, y, hyper, sample_weight)` — posterior
  mean of the observation-scale parameter on block `(a, b]`.

All families inherit the full scikit-learn contract (`fit`, `predict`,
`score`, `transform`, `get_params`, `clone`) and the standard fitted
attributes (`k_map_`, `map_boundaries_`, `boundary_marginals_`,
`bayes_curve_mean_`, `log_evidence_`, `k_posterior_`, …).

## Conjugate families (closed-form block evidence)

The integrated single-segment evidence has a closed-form ratio of
log-partition functions; see `prop:gaussian-block` / `prop:poisson-block`
/ `prop:binomial-block` / `prop:negbin-block` / `prop:beta-block` in §4.

### `BayesBreakGaussian`

Weighted Normal–Normal:

$$y_i \mid \mu_q \sim \mathcal{N}(\mu_q,\, \sigma^2 / w_i),\qquad
\mu_q \sim \mathcal{N}(\nu,\, \rho^2).$$

Hyperparameters: `nu`, `rho2`, `sigma2`. With `estimate_hyper=True` the
code estimates `(ν, ρ²)` via adjacent-pair moment-matching and `σ²` via
the within-segment residual variance (`rho_estimation="cov"` by default;
`"var"` is the alternative moment estimator).

`MOMENT_SIGN_CONTRACT = "signed"` — the segment mean can be negative;
moment numerators use signed-linear storage per §5.

### `BayesBreakPoisson`

$y_i \mid \lambda_q \sim \text{Poisson}(\lambda_q\, w_i)$,
$\lambda_q \sim \text{Gamma}(\alpha, \beta)$.

Hyperparameters: `alpha`, `beta` (shape, rate).

### `BayesBreakBinomial`

$y_i \mid p_q \sim \text{Binomial}(n_i, p_q)$,
$p_q \sim \text{Beta}(\alpha, \beta)$.

Hyperparameters: `alpha`, `beta`, `n_trials` (scalar or array of
trial counts).

### `BayesBreakBernoulli`

Convenience subclass for `n_trials ≡ 1`.

### `BayesBreakBeta`

Fractional Beta-Binomial on $y \in (0, 1)$: maps each `y_i` to
pseudo-counts $(\kappa y_i,\, \kappa (1-y_i))$ and reuses Beta-Binomial
conjugacy. Hyperparameters: `alpha`, `beta`, `concentration` ($\kappa$).

### `BayesBreakNegBin`

Negative-Binomial with fixed dispersion $r$ and Beta prior on the
success probability $p_q$:

$$y_i \mid p_q \sim \text{NegBin}(r,\, p_q),\qquad
p_q \sim \text{Beta}(\alpha, \beta).$$

Hyperparameters: `alpha`, `beta`, `r` (fixed dispersion), `r_predict`
(optional dispersion used for observation-mean outputs).

## Non-conjugate families (deterministic approximations)

### `BayesBreakBetaObs`

Beta-response with known per-observation precision $\phi_t$:

$$y_i \mid \mu_q \sim \text{Beta}(\phi_i \mu_q,\, \phi_i (1-\mu_q)),\qquad
\mu_q \sim \text{Beta}(\alpha, \beta).$$

The integral in $\mu$ is one-dimensional and is computed by
Gauss–Legendre quadrature on $(0, 1)$ (default 32 nodes,
`quadrature_points` kwarg).

### `BayesBreakLogisticNormal`

Bernoulli observations with a Normal prior on the log-odds:

$$y_i \mid \eta_q \sim \text{Bernoulli}(\sigma(\eta_q)),\qquad
\eta_q \sim \mathcal{N}(\nu, \rho^2).$$

The segment-evidence integral is not closed-form. Choose one
approximation via `approx=`:

| `approx=` | Block routine | Uniform-$\varepsilon$ rate (`prop:uniform-bounds`) |
|---|---|---|
| `"laplace"` | 1-D Newton + Laplace expansion | $O(n^{-1})$ on reachable blocks |
| `"jj"` | Jaakkola–Jordan variational lower bound | $O(n^{-1})$ on reachable blocks |
| `"pg_vb"` | Pólya–Gamma mean-field | $O(n^{-1})$ on reachable blocks |
| `"ep"` | True Minka EP with accumulated site normalizers | not uniformly bounded; convergence-conditional |
| `"gh"` / `"quadrature"` | 1-D Gauss–Hermite (low / high node count) | $O(Q^{-2r})$ for $C^{2r}$ integrands |

Use `run_non_conjugate_diagnostics(est, ref)` to measure the empirical
$\varepsilon$ against a high-Q reference; the report emits a
`theoretical_rate_violated` flag when the empirical $\varepsilon$ exceeds
the routine's expected rate by an order of magnitude, or when EP fails
to converge on at least one block (`prop:uniform-bounds (v)`).

The stability theorem (`prop:stability`) bounds the propagated effect on
the segmentation posterior: $\bigl|\Delta \log\frac{P(k\mid y)}{P(k'\mid y)}\bigr|
\le (k+k')\,\varepsilon$ for the segment-count odds and
$\bigl|\Delta \log\frac{P(b_i\mid y, k)}{P(b_{i'}\mid y, k)}\bigr|
\le 2k\,\varepsilon$ for the fixed-$k$ boundary-event odds. The total-
variation bound on $P(k\mid y)$ is $\exp(2 k_{\max} \varepsilon) - 1$
(Corollary `cor:probability-error-conversion`).

## Multi-subject and grouped wrappers

| Class | Manuscript reference | Use case |
|---|---|---|
| `SharedBoundaryReplicatesSegmenter(base)` | `thm:multisubject`, `prop:shared-boundary-identifiability` | Multiple subjects on a common grid, **shared** changepoints. |
| `BayesBreakGroupedClassifier(base, class_prior=...)` | known-groups DP | Supervised classification when each sequence's group label is observed. |
| `BayesBreakMixtureClassifier(base, n_groups=G)` | `thm:em-monotone`, `prop:latent-identifiability` | Unknown group memberships; latent-template EM with canonical permutation anchor. |
| `SharedBoundaryMultivariateSegmenter(base)` | §4 multivariate extension | Vector-valued response, single shared segmentation across channels. |
| `IndependentMultivariateSegmenter(base)` | §4 multivariate extension | Vector-valued response, per-channel fits. |
| `SlidingWindowSegmenter(base, window_size, overlap)` | §5b *Computational regime* | Approximate DP for $n \gtrsim 10^5$. |
