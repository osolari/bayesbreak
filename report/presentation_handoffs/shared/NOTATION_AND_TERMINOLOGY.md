# Notation and terminology

## Core notation

| Symbol | Meaning |
|---|---|
| `y_1:n` | Ordered observations. |
| `x_1 < ... < x_n` | Optional regular or irregular design coordinates. |
| `tau = (t_0,...,t_k)` | Ordered partition with `t_0=0` and `t_k=n`. |
| `B_ij` | Candidate segment from index `i` through `j`. |
| `M_ij` | Segment marginal likelihood after integrating the segment parameter. |
| `c(i,j)` | Segment-cohesion prior factor. |
| `h(j)` | Interior-boundary-hazard prior factor. |
| `p(k | y)` | Posterior distribution of the number of segments. |
| `S_s(tau_g)` | Positive sequence-specific score used in the finite latent-group criterion. It is not a normalized sampling density. |
| `r_sg` | Auxiliary allocation weight in the Jensen minorization. |

## Required terminology

Use: exponential-family observation model, sufficient statistic, natural parameter, log-partition function, proper conjugate prior, segment marginal likelihood, product-partition model, partition prior, sum-product recursion, max-sum recursion, backtracking, joint MAP partition, posterior changepoint probability, hierarchical Bayesian model, common changepoints, known groups, finite latent-group criterion, Jensen minorization, minorization--maximization, posterior-predictive distribution, coordinate support, one-to-one boundary matching, approximation error, numerical tolerance, and computational complexity.

## Terms not permitted as substitutes for mathematics

Do not replace defined statistical or computational terms with branding or project-management vocabulary. Name the actual probability model, prior factor, dynamic-programming recursion, optimization objective, metric, or numerical check.

## Generality statement

The partition recursions are independent of the observation family once finite segment marginal likelihoods are supplied. Exact analytic segment integration applies to the supported regular exponential-family models with proper conjugate priors and finite normalizing constants. Numerical segment integration yields approximate posterior quantities unless an explicit segmentwise error bound is established.
