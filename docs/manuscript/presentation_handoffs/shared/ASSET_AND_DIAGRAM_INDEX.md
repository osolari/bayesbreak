# Asset and diagram index

## Editable scientific diagrams

| Scientific question | Canonical TikZ source | Approved uses | Notes |
|---|---|---|---|
| How do family-specific segment calculations feed posterior and MAP partition inference? | `shared/figures/tikz/workflows/segment_to_partition.tex` | Book, main paper, executive summary, technical presentation | Preserve the distinction between sum-product and max-sum routes. |
| What is the distinction between segment cohesion and boundary hazard? | `shared/figures/tikz/models/cohesion_boundary_hazard.tex` | Book, main paper, technical presentation | Do not present a Poisson-gap interpretation unless it matches the specified prior. |
| How are aligned sequences pooled under a common partition? | `shared/figures/tikz/models/executive_shared_boundary.tex` and `shared/figures/tikz/models/shared_boundary_plate.tex` | Executive or technical route | State conditional independence and aligned candidate coordinates. |
| How is the finite latent-group criterion optimized? | `shared/figures/tikz/models/latent_group_templates.tex` | Book, main paper, executive summary, technical presentation | Do not call the criterion a normalized finite-mixture likelihood. |
| How does segment approximation error propagate? | `shared/figures/tikz/theory/segment_error_propagation.tex` | Book, main paper, technical presentation appendix | State the uniform reachable-segment error assumption. |
| What are the principal theorem dependencies? | `shared/figures/tikz/theory/theorem_dependency_map.tex` | Book and technical presentation | Routine-specific rates remain a proof obligation. |
| What are the boundaries of the method? | `shared/figures/tikz/workflows/scope_and_failure_map.tex` | Book, main paper, technical presentation appendix | Preserve failure cases and exclusions. |
| What implementation stages are approved? | `shared/figures/tikz/roadmaps/executive_implementation_sequence.tex` | Executive summary and later executive presentation only | Do not use in a technical presentation. |
| What release decisions follow failed criteria? | `shared/figures/tikz/roadmaps/executive_release_decisions.tex` | Executive summary and later executive presentation only | Reuse the approved logic; do not invent a new decision sequence. |

## Archived empirical figures

| Asset | Source | Permitted use |
|---|---|---|
| Observation-family examples | `shared/figures/results/fig2_family_showcase.pdf` | Illustrate family-specific segment models with common partition inference. |
| Boundary calibration | `shared/figures/results/fig3_boundary_calibration.pdf` | Report the archived Gaussian calibration design and its limits. |
| Latent-group simulation | `shared/figures/results/fig4_latent_groups_cropped.pdf` | Show behavior under the stated finite criterion; no normalized-mixture claim. |
| Runtime scaling | `shared/figures/results/fig5_runtime_scaling.pdf` | Show finite-range timing only. |
| Well-log case study | `shared/figures/results/fig6_welllog.pdf` | Descriptive fitted MAP changepoints; no external-event accuracy. |
| Array-CGH case study | `shared/figures/results/fig7_cgh.pdf` | Descriptive shared-boundary fit; exclude invalid comparator rows. |
| SPX case study | `shared/figures/results/fig8_spx.pdf` | Descriptive Gaussian and Bernoulli analyses. |
| Methylation case study | `shared/figures/results/fig9_methylation.pdf` | Descriptive segmentation; do not present the excluded held-out score. |

All numerical assets remain read-only. Empirical plots should be regenerated only from versioned result records after the corresponding implementation tasks are complete.
