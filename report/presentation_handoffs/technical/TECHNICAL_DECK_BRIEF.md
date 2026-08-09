# Technical presentation brief

## Intended audience

Statisticians, probabilists, Bayesian-computation researchers, machine-learning researchers, numerical-method specialists, and scientific-software engineers familiar with likelihoods, priors, posterior inference, and dynamic programming.

## Technical objective

Explain the generalized hierarchical Bayesian segmentation method, its exact and approximate regimes, the distinction between posterior marginalization and joint MAP optimization, its irregular-design and multi-sequence extensions, the finite latent-group criterion, the current mathematical results, and the executed empirical studies with their limitations.

## Required narrative sequence

1. Multiple-changepoint problem, ordered partitions, and inferential targets.
2. Exponential-family segment model, sufficient statistics, proper conjugate prior, and segment marginal likelihood.
3. Factorized partition prior, including distinct segment-cohesion and boundary-hazard factors.
4. Sum-product recursions for marginal likelihoods and posterior quantities.
5. Max-sum recursion and backtracking for the joint MAP partition.
6. Irregular designs, shared changepoints across aligned sequences, and known groups.
7. Finite latent-group criterion and Jensen minorization.
8. Numerical segment integration and conditional approximation-error propagation.
9. Synthetic and applied results, with exact interpretation limits.
10. Failure cases, implementation defects, unresolved tests, and proof obligation.

## Prohibited content

The technical presentation must not contain project-development, staffing, repository, publication, resource-allocation, milestone, or decision roadmaps. It may show algorithmic flow, theorem dependencies, experiment design, and failure boundaries because these are part of the scientific argument.

Do not prescribe slide layouts in this handoff. A later presentation workflow should determine pacing and visual composition while preserving the required content and scientific status.
