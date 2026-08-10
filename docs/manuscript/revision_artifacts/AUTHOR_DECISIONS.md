# Author decisions

## Binding scientific decisions

1. The title is fixed exactly as:

   **Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

2. The principal narrative is generalized hierarchical Bayesian segmentation. Irregular designs, multi-sequence hierarchies, known groups, and grouped or latent-group designs are central method components.
3. Generality is stated mathematically. Partition recursions accept segment marginal likelihoods from supported observation models. Analytic exactness applies to regular exponential-family models with proper conjugate priors and finite normalizing constants; numerical segment integration yields explicitly approximate posterior quantities.
4. Scientific prose uses established terminology from statistics, probability, Bayesian computation, machine learning, and optimization. Probability models, sufficient statistics, priors, recursions, optimization criteria, metrics, and numerical-error statements replace vague branding.
5. The SAIM Unified Professional Template v1.0, Computer Modern Sans, governs book and paper-form artifacts for this session.
6. Every populated archived numerical output is a real execution and remains read-only. Scientific interpretation is separate from execution provenance.

## Phase 6 mathematical decisions

- Structural partition support is defined independently of the realized segment likelihood values.
- A reported segment functional that depends on exposure or trial structure is evaluated at a declared reference descriptor, `m_star(theta)=m(theta;d_star)`.
- Sum-product posterior inference and max-sum joint-MAP recovery are separate results.
- Fixed-count Poisson interval occupancy contributes local odds `exp(Lambda_j)-1`.
- The latent-group Jensen criterion uses the implementation-scale group weight, including `n_g log gamma_u(i,j)`.
- Nonconjugate global perturbation results are conditional on a uniform admissible-segment log-score bound; routine-specific rates remain unestablished without separate proofs.
- Changepoint matching is maximum-cardinality, minimum-total-distance, and one-to-one; matched-distance summaries are `NA` when no match exists.

## Communication and implementation decisions

- The executive summary derives from the book and journal paper and introduces no new scientific claim.
- Only the two approved executive diagrams may express implementation and release decisions.
- The main journal paper and technical presentation handoff contain no project, publication, staffing, repository, milestone, resource-allocation, or decision roadmap.
- This workflow generates presentation-source handoffs, not slides.
- The repository skeleton remains visibly incomplete until scientific modules and tests are implemented and verified.

## Result-specific decisions

- `RES-BB-CMP-002` remains a historical executed computation but is excluded from comparator conclusions because the compared objects use incompatible coordinate axes.
- `RES-BB-RD-007Q` remains a historical executed computation but is excluded from posterior-predictive conclusions because the Beta-observation path used a Gaussian predictive calculation and implicit endpoint assignment.
- Corrected computations require new result identifiers, parent-result links, data/configuration hashes, and explicit interpretation records.
- `RES-BB-QA-002` is the immutable historical bounded-test record. `RES-BB-QA-003` is the Phase 6 bounded rerun and contains one unresolved EP timeout, not a test failure.

## Open scientific and implementation decisions

- A normalized probabilistic latent-mixture model remains reserve work and would require a distinct model, proofs, implementation, and reruns.
- Routine-specific nonconjugate error rates remain a proof obligation unless complete assumptions and proofs are supplied.
- The latent-group returned-objective/restart-selection defect requires a package-code change and regression tests.
- Corrected array-CGH comparator and methylation posterior-predictive computations require implementation changes and reruns.
- One EP logistic-normal test remains unresolved under the Phase 6 20-second cap.
- Independent external changepoint annotations remain unavailable.
- Final journal venue, permanent repository, and data-release locations remain unset.
