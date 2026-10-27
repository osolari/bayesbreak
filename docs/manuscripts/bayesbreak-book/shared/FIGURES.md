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

1. A bold label outside the axes box via `add_panel_label(ax, "(a)")`.
2. Annotations inside the panel via `ax.annotate(text, xy=(0.02, 0.97), xycoords='axes fraction', va='top')`.

Panel labels use lowercase with parentheses: `"(a)"`, `"(b)"`, `"(c)"`, etc.
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
- `docs/manuscripts/paper/figures/` (PDF + PNG — canonical output directory)
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

Every figure's numerical input data is committed in
`docs/manuscripts/shared/figures/data/`. These files are the canonical
source for `render.py` — style changes never require re-running
experiments. Authoritative filenames are the `csvs` entries in
`scripts/figures/validate_inventory.py`'s `_FIGURE_MANIFEST`.

To list all currently committed data files:

```bash
ls docs/manuscripts/shared/figures/data/
ls docs/manuscripts/shared/tables/data/
```

### Modifying the shared palette

1. Edit `scripts/figures/_style.py` (Python) and
   `docs/manuscripts/shared/components/colors.tex` (LaTeX) in tandem.
2. Run `python scripts/figures/render.py --all` to regenerate all PDFs and PNGs.
3. Verify with `python scripts/figures/validate_inventory.py`.

PDF and PNG are always generated together by `save_figure()`.

### Paper and book sharing

Both `docs/manuscripts/paper/main.tex` and `docs/manuscripts/book/main.tex`
reference figures from `paper/figures/` (the canonical output directory).
Regenerating with `render.py --all` updates both publications simultaneously.
