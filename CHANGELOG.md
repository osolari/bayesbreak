# Changelog

## Unreleased

### Confirmatory experiment execution (E03-E10)

- Executed E03 irregular-design experiment: `results/confirmatory/E03_irregular_design/e0d11782dcbd95ff/`, gate_pass=True.
- Executed E04 shared-boundary experiment: `results/confirmatory/E04_shared_boundary/0d135a601bdf32b1/`, gate_pass=True.
- Executed E05 latent-templates experiment: `results/confirmatory/E05_latent_templates/d33f1c5e067d38ff/`, gate_pass=True.
- Executed E06 nonconjugate approximation experiment: `results/confirmatory/E06_nonconjugate/9739f69c790aae6f/`, gate_pass=True. Note: `pg_vb` aliases `jj` in the estimator code path; it is not benchmarked as an independent method. Manuscript table must reflect this before reporting pg_vb and jj as distinct methods.
- Executed E07 misspecification experiment: `results/confirmatory/E07_misspecification/a038ac8e7e4a17b9/`, gate_pass=True.
- Executed E08 posterior-prediction experiment (corrected): `results/confirmatory/E08_posterior_prediction/a23d9fe92c02724c/`, gate_pass=True. Supersedes contaminated run `cc26eda02dc82357` (affected by `likelihood_power_` defects in BetaObs and Binomial families).
- Executed E09 real-data experiment: `results/confirmatory/E09_real_data/0f5281c46ee8976e/`, gate_pass=True (execution), scientific-completeness INCOMPLETE. 31 configurations, 0 execution failures, 25 truncated fits. Welllog full: k_map=83 (geometric_decay_log4), resolved. CGH single_subject0: k_map=71, resolved. CGH pooled and methylation: persistently truncated; true MAP k unknown.
- Executed E10 scaling benchmark: `results/confirmatory/E10_scaling/bece422c9d90093f/`, gate_pass=True. 15 cells (5n x 3k_max), 0 failures. Memory O(n^2) slope 1.98; forward-backward O(k_max * n^1.14) on Apple Silicon / numpy 2.4.6 in practice.

### Supplemental analyses

- Added `scripts/supplemental_e09_ruptures_baselines.py`: ruptures v1.1.10 PELT/Dynp/BinSeg/BottomUp baselines on welllog and CGH single_subject0. Results at `results/supplemental/E09_ruptures_baselines/1aa2500bdfe56fb9/`. Penalty chosen by BIC criterion; BayesBreak MAP boundaries not used as ground truth.
- Added `scripts/supplemental_e09_boundary_marginals.py`: exact boundary-event marginals for 6 resolved E09 configurations via `_dp.boundary_event_marginals_fixed_k`. DP invariant verified (sum = k_map-1 for all configs). Results at `results/supplemental/E09_boundary_marginals/dc8432589c3fa528/`.

### Bug fixes (production code)

- `src/bayesbreak/families/beta_obs.py`: removed `likelihood_power_` scaling of block log-likelihoods; BetaObs now computes ordinary marginal likelihoods. The E08 contaminated run (`cc26eda02dc82357`) was produced with the defective code.
- `src/bayesbreak/families/binomial.py`: same `likelihood_power_` correction.
- `src/bayesbreak/base.py` (`_min_len_prior`): fixed to return `1.0` for valid-length segments (so `log(1.0)=0.0` is added) and `0.0` for invalid segments (fails `value > 0` check and is assigned -inf). Previously returned `0.0` for valid segments, causing all cells to fail in E09 welllog runs before the fix.

### E09 documentation

- Added `docs/manuscripts/revision_artifacts/E09_GATE_RESOLUTION.md`: corrected gate resolution for E09, separating execution PASS from scientific INCOMPLETE status. Removes overclaims present in the initial version (no "arm-level CNV" claim, no "percentage of structural information" language, no comparison of truncated pooled CGH to resolved single-subject, no "possibly 200+" methylation extrapolation, no claim that MCMC is required for boundary uncertainty).
- Added `docs/manuscripts/revision_artifacts/E09_SCIENTIFIC_STATUS.md`: 12-objective tracking document with COMPLETE/INCOMPLETE/UNAVAILABLE/PENDING/NOT PERFORMED statuses for all E09 scientific goals.

### E10 documentation and figures

- Added `docs/manuscripts/revision_artifacts/E10_GATE_RESOLUTION.md`: full E10 gate resolution with per-cell timing table, empirical vs theoretical scaling analysis, and hardware/software environment record.
- Added `docs/manuscripts/shared/tables/defs/tab_E10_scaling.tex` and `results/tab_E10_scaling.tex`: LaTeX timing table for manuscript.
- Generated `docs/manuscripts/shared/figures/results/fig_E10_scaling.png/pdf`: 2-panel E10 figure (E2E runtime and peak memory vs n) with O(n^2) reference curve.
- Added `scripts/run_e10_scaling.py`: E10 benchmark runner with BLAS thread pinning, staged DP timings, and tracemalloc memory measurement.

### Figure and table pipeline

- Added `scripts/generate_all_figures.py`: unified one-command runner that
  regenerates all manuscript figures and tables (exploratory fig1–fig9 plus
  confirmatory E01/E02) and prunes non-manuscript artifacts from the output
  directories.
- Added `scripts/run_e01_gaussian_calibration.py`: confirmatory E01 runner
  with `--regenerate-figures` / `--manuscript-dest` / `--tables-dest` flags.
  Produces `fig_E01_calibration` (2×4 calibration grid) and `fig_E01_recovery`
  using the Paul Tol "muted" palette via `_style.py`.
- Added `scripts/run_e02_family_portability.py`: confirmatory E02 runner for
  the five-family portability experiment. Produces `fig_E02_recovery` with
  per-family colours and circle/triangle markers for weak/strong effect sizes,
  legend positioned outside the plots.
- All exploratory figure scripts (fig1–fig9) and supplementary figure scripts
  now route their output through `_style.py` (seaborn "ticks" context, Paul Tol
  "muted" colour palette, 300 DPI, full four-spine frames).

### Confirmatory experiment results

- Added `results/confirmatory/E01_gaussian_calibration/` and
  `results/confirmatory/E02_family_portability/` with frozen `metrics.json`,
  figure artefacts, and LaTeX tables.

### Manuscript figure and table artefacts

- `docs/manuscripts/shared/figures/results/`: now contains exactly the figures
  referenced by `docs/manuscripts/paper/` and `docs/manuscripts/book/`.
  Deleted stale/supplementary files: `fig3_boundary_calibration_E01`,
  `fig4_latent_groups` (non-cropped), `fig6_mixture_discovery`,
  `fig7_snr_sensitivity`, `fig8_multivariate_shared`, `fig9_model_selection`,
  `fig10_missing_data`, `example_basic_usage`, and CSV/JSON sidecars that were
  not provenance records.
- `docs/manuscripts/shared/tables/results/`: pruned to exactly the six tables
  referenced in the paper/book (removed `table0_metrics_overview` and all
  CSV/MD auxiliary files).

### Bug fixes

- `src/bayesbreak/experiments/_placeholder.py` (`write_run_record`): sidecar
  JSON files now store a repo-relative `figure_path` (e.g.
  `docs/manuscripts/shared/figures/results/fig6_welllog.pdf`) instead of an
  absolute machine-specific path, so `test_result_schema.py` passes on any
  checkout.
- `docs/manuscripts/paper/sections/09-results.tex`: corrected stale
  `\includegraphics` reference from `fig3_boundary_calibration_E01.png` to
  `fig_E01_calibration.png`.
