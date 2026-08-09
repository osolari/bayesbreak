# BayesBreak unified SAIM LaTeX project

This project contains the technical book, two journal-paper layouts, executive summary, implementation handoff, repository interface skeleton, and presentation-source handoffs for:

**Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

The **SAIM Unified Professional Template v1.0, Computer Modern Sans** governs the book and all paper-form artifacts.

## Scientific scope

BayesBreak is a generalized hierarchical Bayesian method for multiple-changepoint segmentation of ordered observations. For an admissible segment, the selected observation model supplies a segment marginal likelihood and, when needed, posterior moments of segment-specific parameters. These quantities enter sum-product dynamic-programming recursions for marginal likelihoods and posterior summaries and a separate max-sum recursion with backtracking for the joint MAP partition.

For supported regular exponential-family observation models with proper conjugate priors and finite normalizing constants, segment calculations are analytic functions of cumulative sufficient statistics. Numerical segment integration can be used when analytic integration is unavailable; the resulting posterior quantities are then approximate and retain the stated numerical-error assumptions. The hierarchical constructions include irregular design points, common or group-specific changepoints across multiple sequences, known groups, and finite latent-group allocations.

## Build and validation

From the project root:

```bash
make book                 # technical book
make paper                # two-column journal paper
make paper-single         # single-column review paper
make executive            # executive summary
make handoff-check        # canonical coding-handoff synchronization
make skeleton-check       # repository-interface and explicit-status checks
make presentation-check   # presentation-source handoffs; no slides
make validate-phase6      # build every target and run the final verification suite
```

Generated PDFs:

```text
build/bayesbreak-technical-book.pdf
build/paper/bayesbreak-main-paper.pdf
build/paper-single/bayesbreak-main-paper-single.pdf
build/executive/bayesbreak-executive-summary.pdf
```

## Canonical sources

- `shared/metadata.tex`: author-approved title and project metadata.
- `book/`: canonical technical exposition and five required appendices.
- `paper/`: shared scientific source and two journal layouts.
- `executive/`: decision-oriented summary derived from the book and paper; it introduces no new scientific claims.
- `shared/bibliography/references.bib`: 38 bibliography records.
- `shared/bibliography/annotated_entries/`: one project-specific annotation for each book reference.
- `shared/handoffs/coding_agent_handoff.json`: canonical implementation specification rendered into the book appendix and standalone Markdown.
- `coding/repository_skeleton/`: typed interface blueprint with `scientific_implementation_complete=false`.
- `presentation_handoffs/`: technical and executive content handoffs; no slide sources are generated.
- `shared/figures/tikz/`: editable scientific diagrams and the two executive release diagrams.
- `shared/figures/results/` and `shared/tables/results/`: archived executed numerical assets, verified unchanged from the corrective Phase 4R source.

## Numerical results and interpretation

Every populated archived numerical value is a real executed computation and is read-only in this manuscript phase. Interpretation is restricted by the recorded model, coordinate system, metric, and implementation state.

- `RES-BB-CMP-002` is excluded from comparator conclusions because the comparator output and the BayesBreak reference use incompatible coordinate axes.
- `RES-BB-RD-007Q` is excluded from posterior-predictive conclusions because the archived computation used the wrong observation-family predictive calculation and an implicit endpoint rule.
- A corrected computation must receive a new result identifier, parent-result link, configuration and data hashes, and a new interpretation record.

No research experiment or scientific package implementation is performed by this LaTeX release project.
