# Phase 6 formal and numerical verification

This independent check evaluates representative closed-form segment marginal likelihoods, finite dynamic-programming identities, the fixed-count Poisson occupancy calculation, numerical-error propagation, and the latent-template Jensen decomposition. It is a numerical verification record, not a replacement for the proofs in the manuscript.

Overall status: **passed**.

| ID | Check | Status | Tolerance |
|---|---|---:|---|
| `MATH-BB-001` | Gaussian-known-variance segment marginal likelihood versus direct integration | PASS | relative error < 2e-11 |
| `MATH-BB-002` | Gamma-Poisson segment marginal likelihood versus direct integration | PASS | relative error < 2e-11 |
| `MATH-BB-003` | Beta-Binomial segment marginal likelihood versus direct integration | PASS | relative error < 2e-11 |
| `MATH-BB-004` | Beta-negative-binomial segment marginal likelihood versus direct integration | PASS | relative error < 2e-11 |
| `MATH-BB-005` | Beta-observation one-dimensional Gauss-Legendre segment integration versus adaptive quadrature | PASS | 128-node relative error < 2e-9; no monotonicity assertion |
| `MATH-BB-006` | Sum-product fixed-count partition evidence versus exhaustive enumeration | PASS | absolute log-evidence error < 2e-12 |
| `MATH-BB-007` | Max-sum recursion and backtracking versus exhaustive joint MAP search | PASS | objective gap < 2e-12 and identical partition |
| `MATH-BB-008` | Ordered-boundary and boundary-event marginals versus exhaustive posterior enumeration | PASS | maximum absolute error < 2e-12 |
| `MATH-BB-009` | Posterior mean and second-moment curves from block-cover decomposition versus exhaustive averaging | PASS | maximum absolute error < 3e-12 |
| `MATH-BB-010` | Shared-boundary pooling by products of sequence-specific segment marginal likelihoods | PASS | absolute log-evidence error < 2e-12 |
| `MATH-BB-011` | Conditional propagation of uniform segment log-score error through dynamic-programming evidences and normalized finite distributions | PASS | state error <= k epsilon; probability ratios within exp(+-2 eta); TV <= min(1, exp(2 eta)-1) |
| `MATH-BB-012` | Fixed-count Poisson interval occupancy gives local odds exp(Lambda_j)-1 after conditioning | PASS | odds formula error < 2e-15 and occupancy-probability alternative demonstrably differs |
| `MATH-BB-013` | Jensen minorizer tightness and responsibility-weighted latent-template score decomposition | PASS | both absolute gaps < 2e-13 |

## Detailed metrics

### MATH-BB-001

```json
{
  "formula": 0.0032767495902880734,
  "quadrature": 0.003276749590288074,
  "quad_abs_error": 4.6040722964667666e-14,
  "relative_error": 4.336808689942018e-19
}
```

### MATH-BB-002

```json
{
  "formula": 0.0001329113758617592,
  "quadrature": 0.00013291137586175913,
  "quad_abs_error": 9.415824914591544e-15,
  "relative_error": 8.131516293641283e-20
}
```

### MATH-BB-003

```json
{
  "formula": 0.0139162307255416,
  "quadrature": 0.013916230725541585,
  "quad_abs_error": 2.096410594218364e-15,
  "relative_error": 1.5612511283791264e-17
}
```

### MATH-BB-004

```json
{
  "formula": 0.00012029487042101353,
  "quadrature": 0.00012029487042101373,
  "quad_abs_error": 1.3355413488570916e-18,
  "relative_error": 2.0328790734103208e-19
}
```

### MATH-BB-005

```json
{
  "reference": 1.3022547944210567,
  "quad_scaled_abs_error": 2.2681157267340995e-15,
  "node_counts": [
    8,
    16,
    32,
    64,
    128
  ],
  "approximations": [
    1.283688662844132,
    1.3022566977372931,
    1.3022547944210539,
    1.3022547944210594,
    1.3022547944210496
  ],
  "relative_errors": [
    0.01425691167079075,
    1.4615522728374437e-06,
    2.2166014488037984e-15,
    2.046093645049656e-15,
    5.456249720132427e-15
  ]
}
```

### MATH-BB-006

```json
{
  "max_absolute_log_evidence_error": 4.440892098500626e-16,
  "n": 7,
  "kmax": 4
}
```

### MATH-BB-007

```json
{
  "max_objective_gap": 0.0,
  "all_partitions_identical": "True"
}
```

### MATH-BB-008

```json
{
  "ordered_boundary_max_abs_error": 1.1102230246251565e-16,
  "boundary_event_max_abs_error": 1.1102230246251565e-16
}
```

### MATH-BB-009

```json
{
  "mean_curve_max_abs_error": 2.220446049250313e-16,
  "second_moment_curve_max_abs_error": 4.440892098500626e-16
}
```

### MATH-BB-010

```json
{
  "sequences": 4,
  "max_absolute_log_evidence_error": 2.220446049250313e-16
}
```

### MATH-BB-011

```json
{
  "epsilon": 0.018,
  "maximum_state_error_as_fraction_of_k_epsilon": 0.8737401983190273,
  "eta_generic": 0.20683838098157126,
  "minimum_probability_ratio": 0.8022823519856507,
  "maximum_probability_ratio": 1.196351352406565,
  "total_variation": 0.05541730626442494,
  "total_variation_bound": 0.5123681929419355
}
```

### MATH-BB-012

```json
{
  "max_abs_error_odds_formula": 1.1102230246251565e-16,
  "max_abs_gap_if_occupancy_probabilities_used_instead": 0.1882129075646054,
  "candidate_intervals": 5,
  "occupied_intervals": 2
}
```

### MATH-BB-013

```json
{
  "jensen_tightness_gap": 0.0,
  "template_decomposition_gap": 0.0,
  "effective_group_weight_n_g": 2.396787252382345
}
```
