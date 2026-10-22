# Figure and table registry, current and planned

## Current scientific results, preserve until verified rerun
Figures `fig:gaussian-result-paper`, `fig:family-result-paper`, `fig:calibration-template-paper`, `fig:latent-stress-corrected-paper`, `fig:runtime-result-paper`, `fig:welllog-paper`, `fig:cgh-paper`, `fig:cgh-agreement-corrected-paper`, `fig:spx-paper`, `fig:methylation-paper`, `fig:methyl-predictive-corrected-paper`. These correspond to previously executed results. Do not relabel them planned, nor treat a historical benchmark agreement score as external truth. Main paper does not display result lineage; see `.claude/RESULT_POLICY.md` and full run manifests.

## Planned figure slots
| Figure label | Protocols | Required panels | Files to generate |
|---|---|---|---|
| `fig:planned-calibration-paper` | E01 | (a) boundary reliability by k/SNR with bin counts and sequence bootstrap, (b) F1 vs jump size with 95% CIs | `fig_E01_calibration.pdf/png`, `fig_E01_recovery.pdf/png` |
| `fig:planned-irregular-paper` | E03 | (a) posterior boundary mass at the true long-gap boundary, (b) false-positive boundary mass when the true changepoint is elsewhere | `fig_E03_gap_true.pdf/png`, `fig_E03_gap_false.pdf/png` |
| `fig:planned-hierarchical-paper` | E04,E05 | (a) pooled F1 vs S with misspecification strata, (b) ARI and collapsed template rate vs separation | `fig_E04_pooling.pdf/png`, `fig_E05_templates.pdf/png` |
| `fig:planned-nonconjugate-paper` | E06 | (a) log evidence errors by block length and success probability, (b) count/boundary posterior discrepancy by method | `fig_E06_local_error.pdf/png`, `fig_E06_global_error.pdf/png` |
| `fig:planned-external-paper` | E09 | (a) boundary accuracy vs independent reference OR agreement explicitly labeled, (b) count-posterior tail vs kmax | `fig_E09_comparators.pdf/png`, `fig_E09_kmax.pdf/png` |
| `fig:planned-scaling-paper` | E10 | (a) component time curves with IQR, (b) peak RSS memory | `fig_E10_times.pdf/png`, `fig_E10_memory.pdf/png` |

## Planned table slots
| Table label | Protocols | Required data columns | File to generate |
|---|---|---|---|
| `tab:planned-oracles-paper` | E00 | number of enumerated cases, max discrepancy in partition sums, moments and family evidence, reference tolerance | `tab_P0_oracles.tex` |
| `tab:planned-recovery-paper` | E01,E02 | cell, family, n, ktrue, seed count, mean F1 + CI, count accuracy, Brier/log score | `tab_P1_recovery.tex` |
| `tab:planned-nonconjugate-paper` | E06 | method, block sizes, median/max absolute log evidence error, failure rate, timing 95% CI | `tab_P2_nonconjugate.tex` |
| `tab:planned-robustness-paper` | E07,E08 | condition, F1 + CI, mean heldout log predictive, prediction interval coverage | `tab_P3_robustness.tex` |
| `tab:planned-external-paper` | E09 | dataset hash, model, kmax, tail, independent/reference type, F1 CI if meaningful, heldout log score | `tab_P4_external.tex` |

The paper intentionally contains the literal `TBD` in body cells for planned tables. Replace only when a versioned run produces *all* required values, including failure counts. Captions must then be revised to remove “Planned”; do not leave stale figure text or placeholder rows. Keep provisional plots separate from main-paper figures until validated.
