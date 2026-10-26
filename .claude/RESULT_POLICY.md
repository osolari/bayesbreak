# Current paper results: permitted use and work needed

These decisions concern the **existing paper-facing figures and tables**, not new E00-E10 experimental outcomes. Published numerical assets are retained under the stated observation model unless a clear defect precludes the claimed interpretation. Retaining an executed output does not mean it has been independently replicated. Never modify a reported number manually; rerun with corrected code and verified data, then regenerate the complete target figure/table.

| Paper result | Current treatment | Required follow-up |
|---|---|---|
| Gaussian simulation and segment count | Retain for that particular unit-weight example; an extra short fitted segment remains visible | E00 exactness oracle COMPLETE (03a4d1c38bfe08c1). E01 calibration COMPLETE (1e84a2673bc5b214). |
| Gaussian, Poisson, Binomial, Beta observation-family showcase | Retain as limited illustration; results across observation scales are not direct performance rankings | E02 COMPLETE (621e2f1812b22ffa). Rerun any weighted/exposure-dependent configuration if needed. |
| Boundary probability calibration | Retain provisionally; independence is by **sequence**, not by boundary position | E01 calibration prevalence and sequence-level intervals COMPLETE. |
| Two-group assignment illustration | Retain for the stated finite-template setting, not proof of universal identifiability | E05 COMPLETE (d33f1c5e067d38ff). |
| Latent-template stress evaluation | Retain provisionally as completed exploratory computation | E05 COMPLETE with objective monotonicity and collapse checks. |
| Logistic-normal approximation table | E06 COMPLETE (9739f69c790aae6f). Manuscript documents that pg_vb aliases jj (shared code path, not independently benchmarked). Four methods reported: quadrature (reference), Laplace (max err 0.038), JJ (max err 0.977, rigorous lower bound), EP (max err 30.001, not a lower bound). | None -- pg_vb aliasing correctly documented; no independent PG-VB implementation required for current reporting. |
| Gaussian finite-range runtime chart | Retain as observed wall times for its stated workloads | E10 COMPLETE (bece422c9d90093f): memory O(n^2) slope 1.98; forward-backward O(k_max*n^1.14) in practice on Apple Silicon / numpy 2.4.6. Do not cite n^1.14 as an asymptotic improvement over O(n^2). |
| Well-log subset | Retain as descriptive fitted segmentation, not externally scored stratigraphy | E09 COMPLETE for welllog (0f5281c46ee8976e): k_map=83 (83 segments, 82 internal changepoints) with geometric_decay_log4 prior. Do not report "percentage of structural information" language. |
| Array-CGH shared-boundary figure and metrics | E09 COMPLETE (0f5281c46ee8976e). CGH single_subject0: k_map=71 (resolved) with correct per-probe sample_weight. CGH pooled (43 subjects): truncated at k_max=60, P(k=60|y)=1.0. Report pooled as truncated; true MAP k unknown, k>60 certain. | Do not claim arm-level CNV events without genomic annotation. Do not compare truncated pooled to resolved single-subject as equivalent fits. |
| Array-CGH probe-aligned comparator agreement | Retain provisionally as agreement with the **model-derived** reference segmentation, not biological accuracy | Redo if corrected CGH reference changes; E09 independent-truth/predictive comparisons unavailable (R baselines not installed). |
| S&P 500 regimes | Excluded from E09 (yfinance not installed). Do not label simulated fallback data as real SPX data. | SPX excluded; simulated fallback not a substitute for observational data. |
| Methylation segmentation | Methylation persistently truncated at k_max=120, P(k=120|y)~1.0. True MAP k unknown. Report as truncated; do not extrapolate or claim "possibly 200+" segments. | E09 methylation: extend k_max well above 120 or report truncation explicitly. |
| Methylation in-support held-out Beta scoring | Retain provisionally under the stated split and dependence limitations | E08 COMPLETE (a23d9fe92c02724c, supersedes cc26eda02dc82357 which was contaminated by likelihood_power_ defects). |

Excluded analyses are not repository results: do not recreate a flattened-axis CGH comparison, Gaussian-fallback methylation predictive score, or simulated example labeled as independent real data. The source archive with historical records is outside this active distribution.

**Confirmatory experiments (E00-E10):** All executed. gate_pass=True for all. Scientific-completeness gaps documented in `CURRENT_WORK.md`. Results are immutable; do not rerun confirmatory experiments to change reported numbers.

**Supplemental experiments:**
- `results/supplemental/E09_ruptures_baselines/1aa2500bdfe56fb9/`: ruptures PELT/Dynp/BinSeg/BottomUp comparisons on welllog and CGH single_subject0.
- `results/supplemental/E09_boundary_marginals/dc8432589c3fa528/`: exact boundary-event marginals for 6 resolved E09 configurations.
