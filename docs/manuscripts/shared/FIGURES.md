# Figure Generation: Unified Style Guide

All statistical plots, TikZ diagrams, and LaTeX tables use a single visual
language defined in two files:

- **Python**: `scripts/figures/_style.py`
- **LaTeX**: `docs/manuscripts/shared/components/colors.tex` +
  `docs/manuscripts/shared/components/tikz.tex`

---

## Seaborn theme

Every Python figure script calls `setup_style()` (or its alias
`apply_paper_style()`) exactly once before drawing:

```python
from _style import setup_style
setup_style()            # paper context, font_scale=1.0
setup_style(font_scale=0.95)   # slightly smaller for dense panels
```

The function applies `seaborn.set_theme(style="ticks", context="paper",
palette=_MUTED_PALETTE)`. Do not call `sns.set_theme` elsewhere.

---

## Semantic color dictionaries

Import semantic dictionaries rather than hardcoded hex values:

```python
from _style import (
    COLORS,           # named generic colors: blue, red, teal, grey, ...
    METHOD_COLORS,    # by approximation method
    FAMILY_COLORS,    # by observation family
    PRIOR_COLORS,     # by boundary prior type
    SPEC_COLORS,      # correct / misspecified
    POOL_COLORS,      # pooled / independent
    METHOD_MARKERS,   # marker per method
    METHOD_LINESTYLES,# linestyle per method
)
```

### Method colors (E06)

| Key | Hex | Meaning |
|-----|-----|---------|
| `quadrature` | `#4477AA` | reference (blue) |
| `laplace` | `#44AA99` | analytic approximation (teal) |
| `jj` / `pg_vb` | `#DDCC77` | variational lower bound (sand) |
| `ep` | `#CC6677` | expectation propagation (rose) |

### Prior colors (E03)

| Key | Hex | Meaning |
|-----|-----|---------|
| `uniform` | `#888888` | uniform count prior (grey) |
| `geometric` | `#4477AA` | geometric count prior (blue) |
| `gap_hazard` | `#44AA99` | gap-dependent hazard (teal) |
| `length_cohesion` | `#EE7733` | length cohesion (orange) |
| `combined` | `#CC6677` | combined hazard + cohesion (rose) |

### Misspecification colors (E07)

| Key | Hex |
|-----|-----|
| `correct` | `#4477AA` |
| `misspecified` | `#CC6677` |

---

## LaTeX color correspondence

`docs/manuscripts/shared/components/colors.tex` is included in every paper
preamble. It defines `\definecolor{bbXxx}{HTML}{RRGGBB}` names that match the
Python dictionaries exactly.

| LaTeX name | Python key | Hex |
|------------|-----------|-----|
| `bbBlue` | `COLORS["blue"]` | `4477AA` |
| `bbRose` | `COLORS["red"]` | `CC6677` |
| `bbTeal` | `COLORS["green"]` | `44AA99` |
| `bbSand` | `COLORS["yellow"]` | `DDCC77` |
| `bbOrange` | `COLORS["orange"]` | `EE7733` |
| `bbPurple` | `COLORS["purple"]` | `AA4499` |
| `bbGrey` | `COLORS["grey"]` | `888888` |
| `bbMethodQuadrature` | `METHOD_COLORS["quadrature"]` | `4477AA` |
| `bbMethodLaplace` | `METHOD_COLORS["laplace"]` | `44AA99` |
| `bbMethodJJ` | `METHOD_COLORS["jj"]` | `DDCC77` |
| `bbMethodEP` | `METHOD_COLORS["ep"]` | `CC6677` |
| `bbFamilyGaussian` | `FAMILY_COLORS["gaussian"]` | `4477AA` |
| `bbFamilyPoisson` | `FAMILY_COLORS["poisson"]` | `44AA99` |
| `bbFamilyBinomial` | `FAMILY_COLORS["binomial"]` | `DDCC77` |
| `bbFamilyNegbin` | `FAMILY_COLORS["negbin"]` | `EE7733` |
| `bbFamilyBeta` | `FAMILY_COLORS["beta"]` | `AA4499` |

---

## No embedded titles

Figures must not carry embedded titles (`fig.suptitle`, centered
`ax.set_title`). Panel identity is conveyed by:

1. A bold letter outside the axes box via `add_panel_label(ax, "A")`.
2. A left-aligned panel descriptor via `add_panel_label(ax, "A", "descriptor text")`.
3. Annotations inside the panel via `ax.annotate(text, xy=(0.02, 0.97), xycoords='axes fraction', va='top')`.

Captions and figure numbers are in the LaTeX wrapper, not the plot.

---

## TikZ diagram architecture

### Source files

TikZ source lives in `docs/manuscripts/shared/figures/tikz/` organized by
category:

```
tikz/
  models/      -- plate diagrams, DAG models
  workflows/   -- algorithm flow, inference routes
  experiments/ -- (if any experiment-specific diagrams)
  theory/      -- mathematical illustrations
  roadmaps/    -- planning or survey figures
```

Each `.tex` file in these directories contains only the `tikzpicture`
environment, nothing else.

### Wrapper files

Each TikZ figure has a separate wrapper in
`docs/manuscripts/shared/figures/defs/` named `fig_<name>.tex`:

```latex
\begin{figure}[!htbp]
  \centering
  \resizebox{0.96\columnwidth}{!}{\input{shared/figures/tikz/<category>/<name>.tex}}
  \caption{Caption text here.}
  \label{fig:<name>-paper}
\end{figure}
```

Sections include diagrams via `\input{shared/figures/defs/fig_<name>.tex}`,
never with an inline `\begin{figure}` block.

### Node style conventions

All TikZ nodes use the styles defined in
`docs/manuscripts/shared/components/tikz.tex`:

| Style | Fill | Text | Use |
|-------|------|------|-----|
| `bbstage` | `bbStageFill` (`4477AA`) | white | primary pipeline stage |
| `bbnode` | `bbNodeFill` (`E8EFF7`) | `bbNodeText` | supporting information |
| `bbaccent` | `bbAccentFill` (`CC6677`) | white | highlighted result |
| `bbneutral` | `bbLightGrey` | `bbNodeText` | secondary or inactive node |

Nodes are:
- **Square-cornered** (`rounded corners=0pt`)
- **Borderless** (`draw=none`)
- **Solid fill** (no gradients)

### Relative positioning

Use `positioning` library and `above=of`, `right=of`, `below=of` etc.
Avoid absolute coordinates for node placement; only use them for edge
control points when necessary.

---

## Two-stage rendering architecture

All active data-driven figures and generated tables operate under a strict
two-stage model. Stage A runs scientific computation and writes plot-ready
CSVs; Stage B reads only those CSVs and produces PDF + PNG. Changing a plot
style requires only Stage B, which takes ~8 seconds across all figures.

### Stage A: prepare data (scientific computation)

```bash
# Prepare data for every figure and table
python scripts/figures/prepare_data.py --all

# Prepare only specific figures
python scripts/figures/prepare_data.py --only fig_E07 figA2
```

Outputs go to:
- `docs/manuscripts/shared/figures/data/` (CSV + JSON for figures)
- `docs/manuscripts/shared/tables/data/` (CSV + JSON for tables)

### Stage B: render (zero scientific computation)

```bash
# Render all figures and tables from cached data
python scripts/figures/render.py --all          # ~8 seconds

# Render only specific figures
python scripts/figures/render.py --only fig_E07 fig6 tab_E01_recovery
```

Outputs go to:
- `docs/manuscripts/shared/figures/results/` (PDF + PNG)
- `docs/manuscripts/shared/tables/results/` (LaTeX .tex bodies)

### Inventory validator

Verify that every active manuscript figure and table has full two-stage
coverage (CSV inputs, JSON config, PDF/PNG outputs, registered render
function):

```bash
python scripts/figures/validate_inventory.py            # exits 0 on pass
python scripts/figures/validate_inventory.py --verbose  # show all checks
```

The validator discovers assets by recursively following `\input{}` references
from `paper/main.tex`. Figures with `provenance_status: UNVERIFIED` in their
JSON config produce warnings rather than failures.

### Adding a new figure

1. Write a `prepare_<name>()` function in `scripts/figures/prepare_data.py`
   that writes CSV(s) and a JSON config to `shared/figures/data/`.
2. Register it in `prepare_data.py`'s `_REGISTRY` and `_ALL_ORDER`.
3. Write a `render_<name>(out_dir)` function in `scripts/figures/render.py`
   that reads only those CSVs/JSONs and calls `save_figure()`.
4. Register it in `render.py`'s `_REGISTRY` and `_ALL_ORDER`.
5. Add an entry to `validate_inventory.py`'s `_FIGURE_MANIFEST`.
6. Run `prepare_data.py --only <name>` then `render.py --only <name>` and
   commit all new CSV, JSON, PDF, and PNG files.

### Adding a new generated table

Same flow as figures but:
- CSVs go to `shared/tables/data/`
- Output is a `.tex` file in `shared/tables/results/`
- Add the entry to `_TABLE_MANIFEST` in `validate_inventory.py`

---

## Committed figure-input data

Every figure has its numerical input data committed in
`docs/manuscripts/shared/figures/data/`. These files are the source of truth
for figure regeneration. Style changes never require re-running experiments.

| Data file | Figure(s) | Format |
|-----------|-----------|--------|
| `fig1_synthetic_gaussian_data.json` | fig1 | JSON: y, map_prediction, boundary_marginals, ci_lo/hi |
| `fig2_family_showcase_data.json` | fig2 | JSON: per-family y, true_signal, fit, map_boundaries |
| `fig3_boundary_calibration_data.json` | fig3 | JSON: p_bin, y_bin, n_bin, ece, brier |
| `fig4_latent_groups_data.json` | fig4 | JSON: responsibilities, boundary_post_per_group, example sequences |
| `fig5_runtime_scaling_data.csv` | fig5 | CSV: n, k_max, runtime columns |
| `fig6_welllog_panel_data.json` | fig6 | JSON: y, map_prediction, boundary_marginals, k_posterior, trace |
| `fig7_cgh_panel_data.json` | fig7 | JSON: same panel structure |
| `fig9_methylation_panel_data.json` | fig9 | JSON: same panel structure |
| `fig_E01_data.csv` | E01 figs | CSV: cell_summaries from canonical artifact |
| `fig_E02_data.csv` | E02 figs | CSV: cell_summaries |
| `fig_E03_data.csv` | E03 figs | CSV: gap, prior, true_template, mean_f1, mean_p_b_gap |
| `fig_E04_data.csv` | E04 figs | CSV: S, nonshared_fraction, pooled/indep metrics |
| `fig_E05_data.csv` | E05 figs | CSV: G, S, delta, template recovery metrics |
| `fig_E06_metrics.csv` | E06 figs | CSV: per-block BlockMetric fields |
| `fig_E06_seg_results.csv` | E06 figs | CSV: per-replicate SegResult fields |
| `fig_E07_data.csv` | E07 figs | CSV: per-replicate misspecification results |
| `fig_E08_data.csv` | E08 figs | CSV: posterior prediction metrics |
| `fig_E10_data.csv` | E10 figs | CSV: timing/memory cells from metrics.json |

### Re-generating figures from committed data

Use the canonical regeneration script for experiment figures:

```bash
python scripts/figures/regenerate_from_data.py              # all E01-E07
python scripts/figures/regenerate_from_data.py --experiments E03 E06
```

For showcase and real-data figures, data is saved automatically when scripts
run. To regenerate the PDFs only (no refit, no re-simulation needed -- TODO):
once each script has a `--from-data` flag, run:

```bash
python scripts/figures/fig1_synthetic_gaussian.py --from-data
python scripts/figures/fig6_welllog.py --simulated --from-data
```

Until `--from-data` is implemented, these scripts re-run quickly (<2 min each
in simulated mode) and re-save the data file as a byproduct.

## Figure generation commands

Statistical figures are regenerated from committed artifact CSV/JSON files
(in `results/`). Run from the repo root:

```bash
# Individual experiment figures
python scripts/run_e01_gaussian_calibration.py --regenerate-figures
python scripts/run_e02_family_portability.py   --regenerate-figures
python scripts/run_e03_irregular_design.py     --regenerate-figures
python scripts/run_e04_shared_boundary.py      --regenerate-figures
python scripts/run_e05_latent_templates.py     --regenerate-figures
python scripts/run_e06_nonconjugate.py         --regenerate-figures
python scripts/run_e07_misspecification.py     --regenerate-figures

# Showcase figures (freestanding scripts)
python scripts/figures/fig1_synthetic_gaussian.py
python scripts/figures/fig2_family_showcase.py
python scripts/figures/fig3_boundary_calibration.py
python scripts/figures/fig6_welllog.py --simulated
python scripts/figures/fig7_cgh.py     --simulated
python scripts/figures/fig9_methylation.py --simulated
```

Output goes to `results/figures/` for showcase figures and to
`results/e0X/figures/` for experiment figures. Copy final PDFs to
`docs/manuscripts/shared/figures/results/` before committing.

---

## Architectural requirement

Raw numerical input data (CSV, JSON) for every figure must be committed
to the repository so that style changes can be applied by running only the
figure-generation functions -- without re-running experiments. This is the
prerequisite for fast style iteration.
