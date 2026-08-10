# Required theory and results for the technical route

## Equations and definitions to consider

- Segment marginal likelihood: `M_ij = integral p(y_i:j | theta_ij, d_i:j) p(theta_ij) d theta_ij`.
- Local partition factor: segment marginal likelihood times segment cohesion and, for an interior endpoint, boundary hazard.
- Fixed-count partition marginal likelihood and segment-count posterior.
- Forward/backward sum-product recursions and changepoint marginals.
- Max-sum recursion with backpointers for the joint MAP partition.
- Shared-boundary pooling: `log M_ij^shared = sum_s log M_ij^(s)` under aligned conditional independence.
- Finite latent-group criterion, auxiliary weights, and responsibility-weighted template update.
- Conditional propagation of a uniform segment log-marginal-likelihood error to partition quantities.

## Established results that must be represented accurately

- Exact conjugate segment calculation under stated regularity and propriety conditions.
- Exact sum-product posterior calculation for the declared factorized model.
- Correctness of max-sum backtracking for the joint MAP partition.
- Exact common-partition pooling for aligned conditionally independent sequences.
- Finite-candidate posterior concentration under the positive average expected-score-gap assumptions.
- Monotone minorization--maximization for the stated finite latent-group criterion, distinct from a normalized mixture model.
- Conditional approximation-error propagation theorem.

## Empirical results that may be shown

- Gaussian calibration: ECE approximately 0.010 and Brier approximately 0.011 for the archived design.
- Finite latent-group simulation: 96% hard assignment accuracy for the stated configuration.
- Finite-range timing: 0.0455 seconds at n=50 and 1.0585 seconds at n=800 for k_max=20; do not infer asymptotically linear complexity.
- Observation-family examples and the four descriptive applications.

## Required limitations

- One routine-specific nonconjugate rate remains a proof obligation.
- The latent-group final-objective implementation path requires repair.
- Seventeen bounded-run tests remain unresolved.
- The CGH comparator and methylation predictive historical calculations are excluded from their named conclusions.
- The applications lack independent reference changepoints.
