# Available templates and build targets

## Governing template

The current book and all paper-form artifacts use the attached **SAIM Unified Professional Template v1.0, Computer Modern Sans**. Its shared components are under `shared/saim/`.

No slide source is created in this project. A later presentation workflow may add technical or executive slide targets only after it reads these handoffs and preserves the exact scientific title, terminology, result status, and route-specific roadmap rules.

## Current LaTeX targets

| Target | Command | Output |
|---|---|---|
| Technical book | `make book` | `build/bayesbreak-technical-book.pdf` |
| Two-column journal paper | `make paper` | `build/paper/bayesbreak-main-paper.pdf` |
| Single-column review paper | `make paper-single` | `build/paper-single/bayesbreak-main-paper-single.pdf` |
| Executive summary | `make executive` | `build/executive/bayesbreak-executive-summary.pdf` |
| Current report validation | `make validate-phase6` | Builds all current documents and runs handoff, presentation, and mathematical checks. Historical release checks are preserved in the signed Phase 6 records. |

## Reusable sources

- `shared/metadata.tex`: exact title, author, and keywords.
- `shared/components/`: mathematical and visual components.
- `shared/figures/tikz/`: editable scientific diagrams.
- `shared/figures/results/`: archived real result figures.
- `shared/tables/results/`: archived real result tables.
- `shared/bibliography/references.bib`: canonical bibliography metadata.
