# BayesBreak research dossier — Phase 6 final state

## Protected title and scientific thesis

**Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

BayesBreak is a generalized hierarchical Bayesian method for multiple-changepoint segmentation of ordered observations. For each admissible segment, the selected observation model supplies a segment marginal likelihood and, when required, posterior moments of a declared segment functional. Sum-product dynamic programming gives marginal likelihoods and posterior summaries over contiguous partitions; a separate max-sum recursion with backtracking gives the joint MAP partition.

For supported regular exponential-family models with proper conjugate priors and finite normalizing constants, segment calculations are analytic functions of cumulative sufficient statistics. When analytic integration is unavailable, named numerical integration or approximation methods may supply segment quantities, and resulting posterior quantities are explicitly approximate. The hierarchical constructions include irregular design points, common or group-specific changepoints across multiple sequences, known groups, and finite latent-group allocations.

## Source hierarchy

1. Integrity and the author's latest direct instructions.
2. Editable manuscript, implementation, tests, result assets, and cached-fit records.
3. Mathematical corrections that preserve the title, main narrative, and method.
4. Verified primary literature.
5. Explicitly identified inference.

## Established contribution structure

1. A generalized Bayesian segmentation formulation in which observation-model segment marginal likelihoods feed inference over ordered partitions.
2. Analytic segment marginal likelihoods and posterior moments for supported conjugate exponential-family models using cumulative sufficient statistics.
3. Sum-product recursions for partition marginal likelihoods, segment-count probabilities, changepoint probabilities, segment-cover probabilities, and posterior signal summaries.
4. A separate max-sum recursion with backtracking for the joint MAP partition.
5. Irregular-coordinate priors with distinct segment-cohesion and interior-boundary-hazard factors.
6. Hierarchical multiple-sequence models with common changepoints and known group structure.
7. A finite latent-group segmentation criterion optimized by Jensen minorization and coordinate ascent, without a normalized finite-mixture likelihood claim.
8. Conditional propagation of a uniform admissible-segment log-marginal-likelihood error to partition marginal likelihoods, posterior odds, and normalized finite distributions.
9. Family-specific posterior prediction with explicit coordinate-support rules and one-to-one changepoint matching.
10. Executed synthetic and applied computations interpreted only under their recorded model, data, coordinate system, metric, and implementation conditions.

## Phase 6 mathematical verification

Phase 6 corrected the partition-support definition, the descriptor-indexed reporting functional, the fixed-count Poisson interval prior, sum-product and max-sum theorem separation, output/storage complexity, loss-function conditions, latent-group score scaling, nonconjugate perturbation assumptions and total-variation bound, and changepoint matching definition. Thirteen independent finite-case or numerical checks passed. These checks supplement but do not replace the in-place proofs.

## Formal status

The technical book has 28 established theorem-like statements with 28 immediately following proofs and one proof obligation. The journal paper has 12 established statements with 12 immediately following proofs and one proof obligation. Routine-specific nonconjugate rates remain unestablished unless method-specific assumptions and proofs are supplied.

## Literature position

The work is situated in exponential-family conjugate analysis, product-partition models, Bayesian multiple-changepoint inference, exact dynamic programming for partitions, hierarchical changepoint models for related sequences, design-dependent priors, finite latent allocation, and nonconjugate Bayesian computation. Exact changepoint dynamic programming, product-partition priors, and conjugate exponential-family integration are established foundations. BayesBreak's contribution is the generalized hierarchical construction combining observation-family segment calculations with irregular designs, multi-sequence hierarchies, and grouped or latent-group models.

The final bibliography contains 38 unique records and 38 project-specific annotations. Phase 6 corrected or standardized metadata where independent publisher, institutional, or archival checks found discrepancies; it did not infer results beyond the assumptions of the cited work.

## Numerical results and implementation state

All 53 archived result assets retain their recorded SHA-256 values. No research experiment was rerun and no populated archived numerical value was changed.

- `RES-BB-CMP-002` is excluded from comparator conclusions because the comparator output and BayesBreak reference do not share a coordinate axis.
- `RES-BB-RD-007Q` is excluded from posterior-predictive conclusions because the executed path used the wrong observation-family predictive calculation and an implicit endpoint rule.
- Comparator changepoint scores without external annotations measure agreement with BayesBreak MAP changepoints, not independent accuracy.
- Common-changepoint and separately fitted array-CGH marginal likelihoods correspond to different factorizations and are not a direct Bayes factor.
- Reported runtime slopes describe the measured range and do not replace worst-case complexity.

The historical bounded test record `RES-BB-QA-002` remains 179 collected, 157 passed, five skipped, and 17 unresolved. The independent Phase 6 bounded rerun `RES-BB-QA-003` records 179 collected, 173 passed, five skipped, one EP timeout under a 20-second per-test cap, and zero failed. The original scientific implementation was not modified.

## Final artifact state

The SAIM project contains a 168-page technical book, 35-page two-column journal paper, 42-page single-column review paper, and 12-page executive summary. It also contains five required book appendices, 38 annotated references, a synchronized coding handoff with 16 tasks and 15 experiment protocols, an explicitly incomplete repository interface skeleton, 15 presentation-source handoff files, two approved executive diagrams, and no slides. All 257 pages passed visual inspection.
