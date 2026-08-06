# Concepts

This page introduces the mental model that makes BayesBreak's API natural.
The reference treatment is the [Phase 6 report package](report.md); here we
summarise the four ideas that recur throughout.

## 1. Block evidence and dynamic programming are separate

BayesBreak deliberately factors the work in two:

```
data y  →  block-evidence array log A^(0)_{ij}  →  dynamic-programming engine  →  posteriors
            └────── per family ─────────┘        └──── distribution-agnostic ────┘
```

The block routine is the only family-specific piece. It returns

- `log_block_evidence[i, j] = log A^(0)_{ij}` on every admissible block,
  `-inf` on every inadmissible block (the admissibility contract of
  §`sec:setup`);
- `block_first_moment[i, j] = A^(0)_{ij} · E[μ | (i, j]]` for downstream
  Bayes-curve moments.

The DP engine consumes this triangular array (and an optional length-aware
cohesion `g(Δ)`) to produce:

- the segment-count posterior `P(k | y)` (`prop:fb-duality` invariant
  `L̃[k, n] = R̃[k, 0]`);
- the boundary-event marginals `P(b_i = 1 | y, k)` with the identity
  `∑_i P(b_i = 1 | y, k) = k − 1`;
- the joint MAP segmentation by max-sum + backtracking
  (`thm:map-correctness`);
- the Bayesian regression curve
  `E[μ(x) | y]` (`prop:block-covering-decomposition`).

Once block evidence is supplied, the DP works the same way regardless of
the family. Adding a new likelihood is a local block-routine change.

## 2. Joint MAP ≠ vector of marginal modes

The boundary-event marginals `P(b_i = 1 | y, k)` and the joint MAP
boundary vector `argmax_t P(t | y, k)` are **different posterior
summaries**. The vector of indices that maximises each marginal
independently is not, in general, the joint mode (cf. the §4
counterexample), because the ordering constraint `t_1 < t_2 < ... <
t_{k-1}` couples coordinates.

BayesBreak computes both:

- `est.boundary_marginals_` — `P(b_i = 1 | y, k_map)` per interior index;
- `est.map_boundaries_` — the joint MAP boundary vector.

Use boundary marginals for calibration and uncertainty visualisation; use
the joint MAP for downstream tasks that need a single boundary set (region
labelling, set-valued prediction).

## 3. Exact and surrogate scores share the DP

For non-conjugate likelihoods (e.g. Bernoulli-logistic with a Normal prior
on log-odds, in `BayesBreakLogisticNormal`), the closed-form block
integral does not exist. BayesBreak then approximates the block evidence
locally — pick one of:

| `approx=` | Block routine | Certification status |
|---|---|---|
| `"laplace"` | 1-D Newton + Laplace expansion | routine-specific uniform rate unresolved |
| `"jj"` | Jaakkola–Jordan variational lower bound | routine-specific uniform rate unresolved |
| `"pg_vb"` | Pólya–Gamma mean-field | routine-specific uniform rate unresolved |
| `"ep"` | Real Minka EP with site normalizers | convergence and uniform rate unresolved |
| `"gh"` / `"quadrature"` | 1-D Gauss–Hermite (low / high node count) | tail and quadrature error must be established |

The DP layer is unchanged: it consumes whatever score matrix the block
routine produced, and posterior odds are perturbed by a controlled amount
under the stability theorem. The conditional total-variation bound
`min(1, exp(2 k_max ε) − 1)` on `P(k | y)` (Corollary
`cor:probability-error-conversion`) gives the absolute-probability
counterpart to the odds-level guarantee.

`bayesbreak.run_non_conjugate_diagnostics` measures the empirical uniform
ε against a reference fit and emits a structured support/error record. It
does not infer a routine-wide convergence rate from the approximation name.

## 4. Limitations are first-class

§5b of the manuscript names eight failure modes — computational regime,
block-model misspecification, non-conjugate-approximation regime,
latent-group identifiability, boundary semantics, partition-prior
sensitivity, identifiability failures, and out-of-scope settings. The
implementation surfaces each as a diagnostic:

| §5b failure mode | Diagnostic |
|---|---|
| Computational regime (large `n`) | `bayesbreak.SlidingWindowSegmenter` (approximation) |
| Block-model misspecification | `bayesbreak.prediction.pit_residuals`, `held_out_log_likelihood_trace` |
| Non-conjugate approximation regime | `run_non_conjugate_diagnostics` + `segment_error_record` |
| Latent-group identifiability (label switching) | `BayesBreakMixtureClassifier.canonical_permutation_` |
| Latent-group identifiability (overspec G) | `select_n_groups_by_holdout` |
| Boundary semantics | `est.boundary_marginals_` + `est.map_boundaries_` exposed separately |
| Partition-prior sensitivity | `run_prior_sensitivity` |
| Out-of-scope settings | documented in `bayesbreak.SlidingWindowSegmenter`'s docstring |

If you only remember one thing from this page: BayesBreak is exact under
the conjugate-block + product-partition-prior hypotheses, controlled
under stated approximation regimes, and explicit about everything else.
