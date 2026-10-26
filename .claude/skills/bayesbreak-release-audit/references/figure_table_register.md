# Figure and table registry

## Current scientific results

All E00-E10 experiments have been executed with gate_pass=True. The tables below distinguish the originally planned output filenames (from the prospective protocol) from the actual output filenames committed to the repository. Where the paper-facing filename differs from the protocol's planned name, the actual file takes precedence.

Accepted results and regeneration rules are in `.claude/RESULT_POLICY.md`.

### Executed figure outputs

| Protocol | Actual output files | Notes |
|---|---|---|
| E01 | `fig_E01_calibration.pdf/png`, `fig_E01_recovery.pdf/png` | Complete |
| E02 | `fig_E02_recovery.pdf/png` | Complete; Beta-obs quadrature corrected |
| E03 | `fig_E03_gap_true.pdf/png`, `fig_E03_gap_false.pdf/png` | Complete |
| E04 | `fig_E04_pooled.pdf/png` | Complete |
| E05 | `fig_E05_templates.pdf/png`, `fig_latent_template_stress.pdf/png` | Complete |
| E06 | `fig_E06_local_error.pdf/png`, `fig_E06_global_error.pdf/png` | Complete; pg_vb/jj aliasing documented |
| E07 | `fig_E07_robustness.pdf/png` | Complete |
| E08 | Incorporated in real-data figures | Complete; supersedes cc26eda02dc82357 |
| E09 | `fig6_welllog.pdf/png`, `fig7_cgh.pdf/png`, `fig9_methylation.pdf/png` | Partially complete; see CURRENT_WORK.md for E09 gaps |
| E10 | `fig_E10_scaling.pdf/png`, `fig_E10_stage_breakdown.pdf/png` | Complete |

### Executed table outputs

| Protocol | Actual output files | Notes |
|---|---|---|
| E00 | `tab_E01_recovery.tex` (via E01 runner) | E00 oracle checks passed; 380/380 |
| E01, E02 | `tab_E01_recovery.tex`, `tab_E02_recovery.tex` | Complete |
| E05 | `tab_E05_latent.tex` | Complete |
| E07 | `tab_E07_robustness.tex` | Complete |
| E10 | `tab_E10_scaling.tex` | Complete |

## Original protocol planned names (historical reference)

The prospective protocols in `experiment_protocols.md` used these planned filenames before execution. They are preserved here for traceability. Do not create new files with these names.

| Original planned label | Protocol | Corresponding executed output |
|---|---|---|
| `fig:planned-calibration-paper` | E01 | `fig_E01_calibration`, `fig_E01_recovery` |
| `fig:planned-irregular-paper` | E03 | `fig_E03_gap_true`, `fig_E03_gap_false` |
| `fig:planned-hierarchical-paper` | E04, E05 | `fig_E04_pooled`, `fig_E05_templates` |
| `fig:planned-nonconjugate-paper` | E06 | `fig_E06_local_error`, `fig_E06_global_error` |
| `fig:planned-external-paper` | E09 | `fig6_welllog`, `fig7_cgh`, `fig9_methylation` (R baselines unavailable) |
| `fig:planned-scaling-paper` | E10 | `fig_E10_scaling`, `fig_E10_stage_breakdown` |
| `tab:planned-oracles-paper` | E00 | Included in E01 recovery table |
| `tab:planned-recovery-paper` | E01, E02 | `tab_E01_recovery.tex`, `tab_E02_recovery.tex` |
| `tab:planned-nonconjugate-paper` | E06 | `tab_E05_latent.tex` (E06 error data in figure) |
| `tab:planned-robustness-paper` | E07, E08 | `tab_E07_robustness.tex` |
| `tab:planned-external-paper` | E09 | Results in manuscript text; independent comparisons unavailable |
