# Current paper results: permitted use and work needed

These decisions concern the **existing paper-facing figures and tables**, not new E00-E10 experimental outcomes. Published numerical assets are retained under the stated observation model unless a clear defect precludes the claimed interpretation. Retaining an executed output does not mean it has been independently replicated. Never modify a reported number manually; rerun with corrected code and verified data, then regenerate the complete target figure/table.

| Paper result | Current treatment | Required follow-up |
|---|---|---|
| Gaussian simulation and segment count | Retain for that particular unit-weight example; an extra short fitted segment remains visible | E00 exactness oracle, E01 repeated recovery. |
| Gaussian, Poisson, Binomial, Beta observation-family showcase | Retain as limited illustration; results across observation scales are not direct performance rankings | E02, and rerun any weighted/exposure-dependent configuration. |
| Boundary probability calibration | Retain provisionally; independence is by **sequence**, not by boundary position | E01 calibration prevalence and sequence-level intervals. |
| Two-group assignment illustration | Retain for the stated finite-template setting, not proof of universal identifiability | E05 independent grouping and restart diagnostics. |
| Latent-template stress evaluation | Retain provisionally as completed exploratory computation | Check objective monotonicity, collapse, seeds and failures; E05. |
| Logistic-normal approximation table | **Rebuild**, not a comparison of five independent methods: JJ and PG-VB share a code path | E06 with independent quadrature oracle and method identity audit. |
| Gaussian finite-range runtime chart | Retain as observed wall times for its stated workloads | E10 standardization and stage-specific profiling. |
| Well-log subset | Retain as descriptive fitted segmentation, not externally scored stratigraphy | E09 independent annotations on the correct coordinate axis. |
| Array-CGH shared-boundary figure and metrics | **Regenerate** after correcting Gaussian precision semantics; old numbers do not validate the corrected sampling model | E09, extended segment-count support and independent/separate fits. |
| Array-CGH probe-aligned comparator agreement | Retain provisionally as agreement with the **model-derived** reference segmentation, not biological accuracy | Redo if corrected CGH reference changes; E09 independent-truth/predictive comparisons. |
| S&P 500 regimes | Retain as descriptive analysis under its precise response transforms | E08/E09 forward-time testing; no claim of causal event detection. |
| Methylation segmentation | Retain provisionally under the working Beta-response precision model; selected count reaches the imposed cap | E09 quadrature and count-limit sensitivity; evaluate read-count likelihood where available. |
| Methylation in-support held-out Beta scoring | Retain provisionally under the stated split and dependence limitations | E08/E09 independent-block/subject scoring and calibrated uncertainty. |

Excluded analyses are not repository results: do not recreate a flattened-axis CGH comparison, Gaussian-fallback methylation predictive score, or simulated example labeled as independent real data. The source archive with historical records is outside this active distribution.

**Prospective experiments:** E00-E10 are unexecuted and the related manuscript result slots remain Planned. The current experiment configuration and metric definitions live exclusively under `.claude/skills/bayesbreak-release-audit/`.
