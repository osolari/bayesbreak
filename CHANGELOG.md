# Changelog

## Unreleased

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
