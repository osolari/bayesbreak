# Executive presentation brief

## Intended audience

Research leadership, technical program leadership, principal investigators, scientific-software leadership, and decision-makers responsible for implementation, validation, publication, and release resources.

## Executive objective

Communicate the statistical problem, principal method, current mathematical and empirical status, implementation corrections, major risks, resource categories, staged validation, and decisions required for a defensible research release.

## Required content

- Exact title and generalized hierarchical Bayesian segmentation narrative.
- Short explanation of segment marginal likelihoods plus dynamic programming over ordered partitions.
- Irregular-design, multi-sequence, known-group, and finite latent-group extensions.
- Established mathematical results and the one remaining proof obligation.
- Selected real executed results with design-specific interpretation.
- Two excluded historical calculations and why they cannot support the corresponding conclusions.
- Current bounded implementation verification: 179 collected, 173 passed, five skipped, one EP logistic-normal timeout under a 20-second per-test cap, and zero failed; retain the historical 157/5/17 record as provenance.
- Minimum corrective cycle versus expanded validation and release program.
- Required roles: statistical-method review, scientific software, empirical execution, and independent final review.
- Open decisions on comparator scope, nonconjugate theory, latent-group future work, venue, repository, and artifact location.

## Approved roadmaps

The executive route may reuse only:

1. `shared/figures/tikz/roadmaps/executive_implementation_sequence.tex`, approved in executive Section 5 as Figure `fig:exec-implementation-sequence`.
2. `shared/figures/tikz/roadmaps/executive_release_decisions.tex`, approved in executive Section 6 as Figure `fig:exec-release-decisions`.

Do not invent a third roadmap or change stage and decision semantics without author approval.
