# Math notes

A short glossary mapping the report's notation to the implementation. See
`docs/report/bayesbreak.pdf` for full derivations.

## Block evidence

For every candidate block `(i, j]`:

- `A^0_{ij} = ∫ ∏_{t ∈ (i, j]} p(y_t | θ) π(θ) dθ` — single-segment evidence.
- `A^1_{ij} = A^0_{ij} · E[μ(θ) | y_{(i,j]}]` — first moment numerator used by
  the regression curve.

Stored in `log_block_evidence_` (log) and `block_first_moment_` (linear).

## Sum-product DP

Forward / prefix table:

```
L[k, j] = log Σ_{0 = t_0 < t_1 < ... < t_k = j} Π_q A^0_{t_{q-1}, t_q}
```

Backward / suffix table `R[k, i]` analogously on the suffix `y_{i+1:n}`.

Under a uniform prior over `k` and over boundary vectors given `k`,

```
log P(k | y) ∝ L[k, n] − log C(n-1, k-1) − log k_max,
log p(y) = log Σ_k exp log P(k | y).
```

Implemented in `bayesbreak.dp.{forward_backward, posterior_over_k}`.

## Boundary posteriors

Per-boundary location posterior

```
P(t_p = h | y, k) = exp( L[p, h] + R[k − p, h] − L[k, n] ).
```

Per-index boundary-event marginal (sums to `k − 1`)

```
P(b_i = 1 | y) = Σ_k P(k | y) · Σ_{p=1..k−1} P(t_p = i | y, k).
```

Implemented in `bayesbreak.dp.{boundary_location_posterior, boundary_event_marginals}`.

## Joint MAP segmentation

The joint MAP `argmax_t p(t | y, k)` is **not** the marginal-top-`k−1` summary.
It is recovered by max-sum DP with backtracking:

```
M[q, j] = max_{h < j} (M[q-1, h] + log A^0_{hj}),
t_q = argmax_h (...).
```

Implemented in `bayesbreak.dp.max_sum_segmentation`.

## Bayesian regression curve

Difference-array trick that accumulates per-block contributions in `O(n²)`:

- `bayes_regression_curve_fixed_k(L, R, lA0, A1, n, k)`
- `bayes_regression_curve_mixed_k(..., posterior_k)` averages over `k`.

## Posterior predictive (§8)

Per MAP block `B` with posterior hyperparameters `(α_B, β_B)`:

```
log p(y_new_B | M, t) = H^new_B
                     + log Z(α_B + S^new_B, β_B + W^new_B)
                     - log Z(α_B, β_B).
```

Implemented in `bayesbreak.prediction.posterior_predictive_logpdf` via the
family-specific `posterior_predictive_logpdf_block` method.

## Partition prior

By default the prior is index-uniform: `p(t | k) = 1 / C(n-1, k-1)`. A design-
aware prior `p(t | k) ∝ Π_q g(x_{t_q} − x_{t_{q-1}})` can be added as an
additive `log_length_prior` matrix to the block evidences in
`max_sum_segmentation`.
