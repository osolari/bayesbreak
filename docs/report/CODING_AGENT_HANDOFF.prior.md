## project overview

BayesBreak is a LaTeX manuscript and implementation specification for Bayesian segmentation with exact dynamic programming, conjugate block scores, non-conjugate approximations, multi-sequence shared-boundary extensions, latent-template grouping, and posterior-predictive scoring. The revision preserves the manuscript scope and distinguishes completed archived diagnostics from planned quantitative-comparator outputs according to the approved provenance manifest.

Primary source root: `bayesbreak.tex`.
Primary sections: `sections/0-abstract.tex` through `sections/8.appendix.tex`.
Bibliography: `reference/cite.bib`.
Figures: `figures/` and `assets/`.
Tables: `tables/` plus inline tables in section files.

## build instructions

Use the documented LaTeX build recipe:

```bash
pdflatex -interaction=nonstopmode -halt-on-error bayesbreak.tex
bibtex bayesbreak
pdflatex -interaction=nonstopmode -halt-on-error bayesbreak.tex
pdflatex -interaction=nonstopmode -halt-on-error bayesbreak.tex
```

In the revision environment, `/usr/bin/bibtex` was a broken symlink. Use the available executable `/usr/bin/bibtex.original` if the default `bibtex` command fails. A clean final pass produced a 100-page PDF with zero TeX errors, zero undefined references, zero undefined citations, zero overfull hboxes, and 28 underfull hboxes.

Do not require new LaTeX packages unless strictly necessary. The project style, class file, local macros, section structure, and table/figure paths should remain stable.

## implementation tasks

- Implement the block-routine interface documented in the algorithms section, including signed-log handling for likelihood scores and moment numerators.
- Implement exact conjugate block-score routines for the exponential-family cases already derived in the manuscript.
- Implement log-space prefix and suffix dynamic programs with the documented boundary conventions and finite-support checks.
- Implement posterior summaries: fixed-k partition posterior, joint MAP segmentation, boundary marginals, Bayes-curve moments, and posterior over the number of segments.
- Implement shared-boundary multi-sequence scoring as a product or sum of subject-specific block scores on a common grid.
- Implement latent-template EM with exact E-step responsibilities, M-step mixture weights, pseudocount or floor options, monotonicity diagnostics, and label-ordering diagnostics.
- Implement non-conjugate approximation backends: Laplace, variational Jaakkola-Jordan logistic, expectation propagation with site-normalization constants, Polya-Gamma-based approximation, and one-dimensional quadrature.
- Implement prediction-time export objects, segment assignment, posterior-predictive scoring, group posterior normalization, and set-valued or vector-valued input scoring.
- Add unit tests for admissible-support behavior, recurrence indexing, prefix/suffix identity checks, posterior normalization, boundary marginal ranges, Bayes-curve moment consistency, and approximation error accounting.

## experiment plan

Completed archived diagnostics are already represented by the current figures and populated tables. Future coding work should reproduce those outputs and then extend the planned real-data quantitative comparison tables.

Synthetic diagnostics:

- Single Gaussian recovery and posterior-summary table.
- Family showcase across conjugate likelihood families.
- Boundary-calibration diagnostic.
- Latent-group diagnostic.
- Non-conjugate approximation tradeoff table.
- Runtime scaling figure and companion table.

Real-data pipelines:

- Well-log processed diagnostic window.
- Array-CGH selected chromosome arm or panel.
- S&P 500 equity-return volatility segmentation.
- CpG-atlas methylation selected region and held-out region extensions.

Every pipeline should save inputs, preprocessing metadata, random seeds, configuration files, source hashes when available, intermediate caches, figure data, and final table data.

## figures and tables to generate

Current real figure assets are present and should be reproducible from code:

- `figures/fig1_synthetic_gaussian.*`
- `figures/fig2_family_showcase.*`
- `figures/fig3_boundary_calibration.*`
- `figures/fig4_latent_groups_cropped.*`
- `figures/fig5_runtime_scaling.*`
- `figures/fig6_welllog.*`
- `figures/fig7_cgh.*`
- `figures/fig8_spx.*`
- `figures/fig9_methylation.*`

Planned quantitative tables to generate:

- `tab:real_welllog`: accuracy, tolerance, log score, competitor, and preprocessing-window metadata.
- `tab:real_cgh`: selected array-CGH comparison metrics and source-label verification.
- `tab:real_spx`: rolling-origin or held-out financial-volatility comparison metrics.
- `tab:real_methylation`: per-region and per-cell-type held-out scoring and atlas-alignment metrics.

Theoretical tables require no source CSV: notation, family summary, prediction outputs, metrics, and complexity accounting.

## projected or expected results

Do not overwrite or alter existing populated numerical values without explicit author authorization. The expected future outputs are planned quantitative-comparator tables, not replacements for the completed figures. Planned real-data tables should report reproducible comparator metrics and ablation results while preserving the current figure-level narratives unless a rerun uncovers a documented error.

Expected qualitative outcomes to test:

- Synthetic recovery remains stable under the archived small diagnostic settings.
- Runtime scaling follows the documented dynamic-programming complexity regimes.
- Real-data figures remain reproducible from saved preprocessing and model configurations.
- Planned quantitative tables distinguish primary BayesBreak results from comparator baselines and ablations.

## theory-to-code connections

- The admissible-support definitions control which candidate blocks and partitions enter the DP; zero-support candidates must be excluded rather than silently treated as low-scoring evidence.
- The algebraic DP recurrences define the mathematical objects; the implementation must use log-space `logsumexp` forms.
- Boundary marginals and Bayes-curve moments come from block-covering decompositions, not from the joint MAP segmentation alone.
- Shared-boundary inference is obtained by adding subject-specific log block scores on the common grid.
- Latent-template EM responsibilities must use exact DP marginal likelihoods for each template and preserve label-switching diagnostics.
- Non-conjugate backends must return comparable approximate log scores and moment outputs, along with convergence or failure metadata.
- Prediction scoring must use the exported segmentation object, posterior summaries, and segment-assignment map documented in the prediction layer.

## open technical questions

- Verify bibliography metadata and add or correct externally sourced entries during Phase Four only.
- Decide whether calibration annotations should be numerically reconciled with a rerun; the manuscript currently preserves all archived numerical content.
- Decide whether runtime table values and plotted runtime data should be regenerated from a single source artifact or kept as companion archived diagnostics.
- Finalize preprocessing-window metadata for the well-log diagnostic.
- Verify array-CGH source labels and comparator definitions.
- Specify the exact rolling-origin protocol for S&P 500 volatility scoring.
- Specify held-out region and cell-type selection rules for CpG-atlas methylation scoring.
- Decide whether very-large-n approximations will be implemented as windowed, sparse-candidate, or multiresolution approximations.

## files changed or added

Changed manuscript source files:

- `sections/0-abstract.tex`
- `sections/1.intro.tex`
- `sections/2.problem.tex`
- `sections/3.setup.tex`
- `sections/4.method.tex`
- `sections/5.algorithms.tex`
- `sections/5b.limitations.tex`
- `sections/6.evaluation.tex`
- `sections/7.conclusion.tex`
- `sections/8.appendix.tex`

Added deliverables:

- `CHANGELOG.md`
- `CODING_AGENT_HANDOFF.md`
- final compiled PDF
- clean Overleaf source ZIP

Prior-pass files were renamed during Phase One as `CHANGELOG.prior.md` and `CODING_AGENT_HANDOFF.prior.md` and should not be treated as current deliverables.

## do-not-change constraints

- Do not change any populated numerical values, strings, or categorical labels in tables without explicit author approval.
- Do not infer placeholder status from small sample size or cautious caption language; use only the provenance manifest rules.
- Do not mark real figures as planned.
- Do not remove conceptual threads, derivations, or sections from the manuscript.
- Do not compress proofs or derivations into summaries when expanding rigor is possible.
- Do not change document class, theme, package ecosystem, or macro structure unless no existing tool supports a necessary correction.
- Do not add Phase Four literature or bibliography changes without external-source verification.
- Do not paste manuscript content into chat during revision workflows.
- Do not replace planned quantitative real-data tables with fabricated values.
- Do not modify prior-pass artifacts except to keep them clearly distinguished from current deliverables.
