# CODING_AGENT_HANDOFF

This handoff is for the next coding agent that will implement BayesBreak experiments, regenerate
figures and tables, verify data sources, and prepare the next publication-development iteration.
It intentionally distinguishes completed manuscript theory and completed synthetic artifacts from
planned experiments, expected outputs, projected outputs, and placeholder real-data artifacts. Do
not treat expected or projected results as observed results until the corresponding scripts have
been run and audited.

This version reflects the consolidated Phase Two edit plan, the Phase Four independent audit,
and a post-delivery CG integration pass that cross-compared the working copy against an external
ChatGPT-revised draft. Relative to the previous handoff, the principal updates are: the project
now compiles end to end with a full TeX toolchain; thirteen verified bibliography entries were
added (twelve in Phase Three plus `fearnhead2003particle` in the CG pass); two new corollaries
(`cor:abs-prob` in Phase Three, `cor:boundary-event-sum` in the CG pass), four remarks (including
the CG-pass `rem:score-matrix-exactness`), one exploratory appendix subsection, an explicit
setup-section block-score contract, and parallel limitations paragraphs in the introduction and
conclusion were inserted; two minor Phase Four polish edits were applied; the well-log appendix
recipe was resolved using the verified `changepoint.influence::welldata` object name; and the
methylation atlas GitHub repository attribution was corrected to point at the actual companion
software for the 2023 atlas (`nloyfer/wgbs_tools` and `nloyfer/UXM_deconv`) instead of the
older `nloyfer/meth_atlas` repository, which implements the unrelated 2018 Moss et al.\ method.
A set of explicit author-verification tasks surfaced by the literature investigation remains
recorded in Section 8.

## 1. Project Overview

BayesBreak is a modular offline Bayesian segmentation framework. The manuscript separates local
block evidence calculations from global dynamic-programming inference over ordered partitions,
then extends the same interface to irregular designs, replicated and grouped sequences,
latent-template mixture models, non-conjugate GLM approximations, and posterior-predictive
scoring.

Current manuscript status:

- Theoretical and methodological sections are written as completed manuscript contributions. The
  pass added one proposed corollary (`cor:abs-prob`, an absolute-probability-error bound for the
  segment-count posterior, derived from assumptions already in force), three clarifying remarks,
  and one explicitly labeled exploratory appendix subsection
  (`app:latent-template-positioning`).
- The bundled synthetic figures and tables support the current synthetic recovery, family
  showcase, calibration, latent-group, non-conjugate approximation, and runtime demonstrations.
- The well-log, array-CGH, S&P 500 volatility, and CpG methylation case studies are planned
  real-data evaluations with placeholder figures and tables and pipeline descriptions. Their
  placeholder figure and table captions are now prefixed "(PLANNED)".
- External baseline comparisons, extended ablations, larger robustness studies, a planned
  prior-sensitivity diagnostic, and FPOP/SNIP baseline comparisons are planned
  publication-development work.

## 2. Build Instructions

Root file: `bayesbreak.tex`.

Document class and local style files:

- `saim.cls`
- `natbib.sty`
- `fancyhdr.sty`
- `plainnat.bst`
- `iclr2026_conference.bst` (vestigial; not used by the active `\bibliographystyle{plainnat}`)
- `math_commands.tex`

Bibliography file: `reference/cite.bib`.

Bibliography engine: BibTeX with the `plainnat` style.

Build sequence:

```bash
pdflatex bayesbreak.tex
bibtex bayesbreak
pdflatex bayesbreak.tex
pdflatex bayesbreak.tex
```

Build status: the project compiles end to end with zero errors, zero undefined references,
zero undefined citations, and zero multiply-defined labels. The revised PDF has 98 pages
following the post-delivery CG integration pass (the original compiled to 92 pages; Phase Three
produced 96, Phase Four 97, and the CG integration 98). The increase reflects one new corollary
with proof (`cor:abs-prob`), three Phase Three remarks, one Phase Three exploratory appendix
subsection, four expanded annotated-literature rows, two Phase Four polish edits, and the CG
additions (one new remark `rem:score-matrix-exactness`, one new corollary `cor:boundary-event-sum`
with proof, the setup-section block-score contract paragraph, two limitations paragraphs in the
introduction and conclusion, and the resolved well-log appendix recipe). The bibliography has
48 entries (47 after Phase Four, plus the verified `fearnhead2003particle` entry added in CG-4).

Portability shims: `saim.cls` and `bayesbreak.tex` carry reversible, commented compatibility
shims (tagged `[Build-A1]`) so the project compiles on minimal TeX installs. Specifically,
`microtype` is loaded with `expansion=false`; the unused `latin` Babel option is dropped;
`lmodern` and `bbm` are loaded through `\IfFileExists` guards with graceful fallbacks. On a full
TeX installation these shims are no-ops. If the intended `lmodern` fonts are present, the
remaining sub-9-point overfull boxes (see below) are expected to shrink further.

Known warnings: 15 overfull and 50 underfull `\hbox` warnings. The two largest overfull boxes
present at baseline were fixed in this pass; the remaining overfull boxes are all under 9 points
and are font-substitution artifacts in headings and run-in paragraph headers. These are layout
warnings, not unresolved-reference or missing-citation errors.

Figure directory: `figures/`. Table directory: `tables/`. Auxiliary assets: `assets/`.

## 3. Implementation Tasks

### IMP-01: Core block data model

- Related manuscript sections: 2, 3, 4.1, 5.
- Description: Implement a block-score interface that stores block evidence, optional moment
  numerators, admissibility status, and any length-prior adjustment.
- Expected input: ordered observations, weights/exposures/trials where applicable, design
  coordinates, family hyperparameters, admissible block constraints.
- Expected output: block matrices for evidence and requested moment numerators.
- Dependencies: agreed observation-weight convention and physical block-length convention.
- Priority: high.
- Completion criteria: all admissible blocks populated; invalid blocks represented consistently
  as zero evidence or negative-infinite log score; prefix-sum and direct computation agree on
  small cases.

### IMP-02: Conjugate family engines

- Related manuscript sections: 4.1, 4.7, Appendix A.1.
- Description: Implement Gaussian, Poisson, Binomial, and Negative-Binomial block routines with
  evidence and moment tests.
- Expected input: observations, weights/exposures/trials, hyperparameters.
- Expected output: block evidences and observation-scale moment numerators where supported.
- Dependencies: family-specific domain checks.
- Priority: high.
- Completion criteria: closed-form routines match direct small-block numerical checks or
  independently computed reference values. Note (per edit 5-C1): the conjugate-family moment
  numerators are strictly positive and may be stored directly in log space; the signed-log path
  is required only for sign-changing moment targets such as a centered Gaussian mean. Implement a
  per-family assertion of moment-numerator sign rather than a global assumption.

### IMP-03: Beta-response quadrature and precision variants

- Related manuscript sections: 4.7, 6, Appendix A.12.
- Description: Implement fixed-precision Beta-response quadrature and decide whether to implement
  the planned observation-specific precision extension for methylation.
- Expected input: response values in `(0,1)`, precision parameter(s), prior hyperparameters,
  quadrature settings.
- Expected output: deterministic block log-evidence and moment estimates.
- Dependencies: node-refinement checks and endpoint handling.
- Priority: high for the methylation pipeline; medium otherwise.
- Completion criteria: quadrature estimates stable under node refinement; methylation
  configuration documented as fixed-precision or observation-specific precision.

### IMP-04: Partition-prior normalizers

- Related manuscript sections: 3, 4.2, 4.3, 5, Appendix A.2.
- Description: Implement `C_k` computation under index-uniform, length-aware, segment-cohesion,
  and renewal-style priors.
- Expected input: `n`, admissible `k`, boundary coordinates, cohesion function, block
  admissibility constraints.
- Expected output: log normalizers and diagnostic tables.
- Dependencies: physical block-length convention.
- Priority: high.
- Completion criteria: regular-grid index-uniform case matches the combinatorial normalizer;
  irregular-grid cases pass brute-force enumeration for small `n`. Note (per edit App-B1): for
  general irregular designs the factorized prior is a valid product-partition prior but is not an
  i.i.d. renewal law; do not rely on a renewal interpretation outside the translation-invariant
  case.

### IMP-05: Sum-product DP core

- Related manuscript sections: 4.2, 5.
- Description: Implement forward/backward DP in log space for fixed `k` and for the posterior
  over `k`.
- Expected input: block log-evidence matrix, log normalizers, `p(k)`, admissibility mask.
- Expected output: prefix/suffix evidence tables, posterior over segment count, boundary
  marginals, segment-membership weights.
- Dependencies: IMP-01 and IMP-04.
- Priority: high.
- Completion criteria: small-`n` brute-force enumeration agrees with DP evidence, boundary
  marginals, and posterior over `k`.

### IMP-06: Max-sum MAP backtracking

- Related manuscript sections: 4.2, 5, Appendix A.7.
- Description: Implement fixed-`k` and across-`k` MAP objectives with deterministic tie handling.
- Expected input: block log scores, count prior, normalizers, chosen `k` mode.
- Expected output: ordered boundary vector and terminal max-sum score.
- Dependencies: IMP-05.
- Priority: high.
- Completion criteria: backtracked score equals stored terminal value; brute-force MAP agrees on
  small cases. Note (per edit 4-B2): independently maximizing the boundary marginals recovers the
  joint MAP only in the degenerate product-measure case; the max-sum recursion is required in
  general.

### IMP-07: Posterior moments and Bayes curves

- Related manuscript sections: 4.1, 4.2, 4.7, 4.9.
- Description: Compute segment-level moments under exported segmentations and Bayes-curve moments
  averaged over posterior segmentations.
- Expected input: moment numerator matrices and posterior block weights.
- Expected output: pointwise Bayes curves and pointwise uncertainty summaries.
- Dependencies: valid moment targets by family.
- Priority: high.
- Completion criteria: moment ratios agree with direct enumeration on small synthetic cases.

### IMP-08: Shared-boundary replicate pooling

- Related manuscript sections: 4.4, 6, Appendix A.12.
- Description: Implement pooled block evidence across subjects with shared boundaries and
  subject-specific block parameters.
- Expected input: aligned or union-grid replicate data, subject-specific weights/missingness,
  hyperparameters.
- Expected output: shared-boundary posterior and subject-level posterior summaries.
- Dependencies: common-grid convention and missingness handling.
- Priority: medium-high.
- Completion criteria: pooled inference reduces to single-sequence inference when there is one
  subject and to independent evidence multiplication under shared boundaries.

### IMP-09: Known-group inference

- Related manuscript sections: 4.5, 6.
- Description: Implement group-specific DP inference when labels are observed.
- Expected input: group labels, observations, group-specific `k` settings, priors.
- Expected output: group-specific boundary posteriors and summaries.
- Dependencies: IMP-05 and label indexing.
- Priority: medium.
- Completion criteria: group factorization tests pass, and known-label results provide a
  supervised baseline for latent-group experiments.

### IMP-10: Latent-template EM

- Related manuscript sections: 4.6, 6, Appendix A.5, Appendix A.10.
- Description: Implement finite-template EM with exact responsibility updates and
  responsibility-weighted template updates under the stated objective.
- Expected input: sequences, number of latent groups, candidate `k` values, priors, restart
  settings.
- Expected output: fitted templates, responsibilities, mixture weights, objective trace,
  convergence diagnostics.
- Dependencies: IMP-05 and IMP-06.
- Priority: high.
- Completion criteria: objective is non-decreasing under deterministic tie handling; empty groups
  handled as specified; restarts are reproducible.

### IMP-11: Non-conjugate approximation routines

- Related manuscript sections: 4.8, 6, Appendix A.8, Appendix A.9.
- Description: Implement or wrap Laplace, Jaakkola--Jordan, Polya--Gamma mean-field, EP-style, and
  quadrature reference routines where applicable.
- Expected input: GLM family, design/covariates if any, response data, weights/trials, prior,
  convergence settings.
- Expected output: approximate block log-evidences and moment estimates where available.
- Dependencies: validated block-error reference on selected blocks.
- Priority: medium-high.
- Completion criteria: diagnostics report max/quantile block errors against a high-accuracy
  reference on selected problems. Note (per edit 4-C1): the approximation-validation checklist in
  Section 4.8 now ties each diagnostic to a specific failure mode; implement the checklist so it
  reports the uniform block error used by the stability proposition, MAP-path errors, and
  posterior sensitivity, and report the measured error alongside the posterior margin.

### IMP-12: Prediction and scoring layer

- Related manuscript sections: 2, 4.9, 6.
- Description: Implement posterior-predictive scoring under exported MAP segmentation, Bayes
  curves, and optional resegmentation.
- Expected input: fitted posterior summaries, new sequence or set-valued unit, family-specific
  predictive routine.
- Expected output: log scores, group posteriors, responsibilities, predicted signal summaries,
  pointwise uncertainty.
- Dependencies: IMP-05, IMP-07, and family predictive formulas.
- Priority: medium.
- Completion criteria: scoring modes are unit tested and produce consistent outputs on synthetic
  data. Note (per edit 2-C1): the optional resegmentation scoring mode runs the DP on the new
  data at `O(k_max m^2)` per group; budget for this cost where the resegmentation mode is used.

### IMP-13: Diagnostics and reporting

- Related manuscript sections: 4.2, 5, 6.
- Description: Implement diagnostic checks for posterior normalization, forward/backward
  equality, boundary marginal sums, MAP score consistency, block-error sensitivity, and
  reproducibility metadata.
- Expected input: fitted model output and run metadata.
- Expected output: diagnostic report per experiment.
- Dependencies: all core engines.
- Priority: high.
- Completion criteria: every experiment emits a machine-readable report with pass/fail checks.

### IMP-14: Plotting and table generation

- Related manuscript sections: 6.
- Description: Generate publication figures and tables from experiment outputs with stable file
  names matching manuscript labels.
- Expected input: experiment result files and plotting scripts.
- Expected output: updated files in `figures/` and `tables/`.
- Dependencies: experiment outputs.
- Priority: high for the final publication iteration.
- Completion criteria: placeholder figures and tables replaced only after verified outputs are
  available. When a placeholder is populated with verified output, remove the "(PLANNED)" prefix
  from its caption.

### IMP-15: Real-data pipeline implementation

- Related manuscript sections: 6, Appendix A.12.
- Description: Implement and verify the well-log, array-CGH, S&P 500, and methylation pipelines.
- Expected input: external datasets or download scripts, preprocessing settings, model
  hyperparameters.
- Expected output: finalized figures, tables, run logs, and verification notes.
- Dependencies: external data access and source verification (see Section 8).
- Priority: high for publication submission.
- Completion criteria: each pipeline produces a reproducible artifact bundle with data hashes,
  generated figures and tables, and documented comparison labels. Note (per edits 6-A1 and
  App-A1): the well-log dataset object name in the `changepoint` R package family must be
  resolved against the installed version; the recipe in Appendix A.12 uses a placeholder object
  name `<welllog_object>` for this reason. Do not use `Lai2005fig4` for the well-log series --
  that object is the array-CGH example of Lai et al. (2005).

## 4. Experiment Plan

### EXP-01: Single-sequence Gaussian recovery sweep

- Status: the current manuscript includes one completed archived illustration; an extended sweep
  is planned.
- Setup: simulate Gaussian piecewise-constant sequences across sample sizes, segment lengths,
  noise levels, and jump sizes.
- Baselines: exact BayesBreak index-uniform prior, length-aware prior where applicable, and
  frequentist segmentation baselines after citation and package verification.
- Metrics: boundary F1 with tolerance, boundary MAE, signal MSE, posterior segment-count
  calibration, runtime.
- Expected output: updated figure and table summarizing recovery and uncertainty.
- Success criteria: DP diagnostics pass; posterior summaries track recoverability trends under
  controlled changes.

### EXP-02: Likelihood-family portability benchmark

- Status: the current manuscript includes a compact archived family showcase; a broader Monte
  Carlo benchmark is planned.
- Setup: Gaussian, Poisson, Binomial, Negative-Binomial, and Beta-response simulations under
  matched changepoint structures.
- Metrics: boundary recovery, signal reconstruction, calibration, runtime, family-specific
  diagnostics.
- Expected output: expanded family summary table and optional supplemental figure.
- Success criteria: the common DP backend works across all implemented block engines.

### EXP-03: Boundary posterior calibration study

- Status: the current manuscript includes a completed Gaussian calibration illustration; an
  expanded calibration study is planned.
- Setup: repeated simulations across signal-to-noise ratios and segment-count priors.
- Metrics: calibration curves, ECE, reliability by posterior-probability bins.
- Expected output: calibration figure and diagnostic table.
- Success criteria: calibration improves or degrades in interpretable ways as information
  changes.

### EXP-04: Runtime and memory scaling

- Status: the current manuscript includes archived runtime scaling over a limited range; extended
  scaling is planned.
- Setup: vary `n`, `k_max`, family, prior, and block-engine type.
- Metrics: preprocessing time, DP time, memory footprint, total runtime.
- Expected output: runtime curves and complexity table.
- Success criteria: empirical scaling is consistent with theoretical complexity over the tested
  regimes.

### EXP-05: Irregular-design prior ablation

- Status: planned.
- Setup: simulate irregularly spaced sequences with large gaps and known physical changepoints.
- Comparisons: index-uniform prior, boundary-coordinate prior, segment-cohesion prior,
  renewal-style prior.
- Metrics: boundary posterior calibration, MAP boundary error in physical coordinates, posterior
  over `k`, prior predictive boundary distribution.
- Expected output: irregular-design ablation figure and table.
- Success criteria: length-aware priors improve physical-coordinate calibration when the design
  geometry is informative.

### EXP-06: Shared-boundary pooling ablation

- Status: planned.
- Setup: multi-subject simulations with shared boundaries, subject-specific means,
  heteroscedasticity, and missing observations.
- Comparisons: pooled shared-boundary inference versus independent per-subject inference.
- Metrics: shared-boundary F1, subject-specific signal MSE, calibration, runtime.
- Expected output: pooled-versus-independent ablation table.
- Success criteria: pooling improves boundary recovery when shared-boundary assumptions hold and
  degrades gracefully when they are violated.

### EXP-07: Known-group versus latent-group comparison

- Status: planned.
- Setup: grouped sequences with controlled group separation and known labels for an oracle
  comparison.
- Comparisons: known-group DP, latent-template EM, independent sequence segmentation.
- Metrics: label recovery, template boundary F1, responsibility entropy, objective monotonicity.
- Expected output: known-versus-latent summary table.
- Success criteria: latent inference approaches known-group performance as separation and sample
  size increase.

### EXP-08: Latent-template robustness study

- Status: planned.
- Setup: vary the number of groups, restarts, initialization, noise, sequence count, and group
  imbalance.
- Metrics: objective trace, recovery rate, empty-group frequency, label stability after
  deterministic ordering.
- Expected output: robustness plots and convergence diagnostics.
- Success criteria: the restart strategy and tie handling produce stable templates under
  identifiable settings.

### EXP-09: Non-conjugate approximation validation

- Status: the current manuscript includes a small archived trade-off illustration; broader
  validation is planned.
- Setup: Bernoulli/logistic and other GLM block models with quadrature or high-accuracy
  references on selected blocks.
- Metrics: block-error max/quantiles, posterior over `k` sensitivity, boundary-marginal
  differences, MAP-path stability, moment-error diagnostics, runtime.
- Expected output: approximation diagnostic table and sensitivity figure.
- Success criteria: approximation choices are interpretable through block-error and
  posterior-sensitivity diagnostics. Corollary `cor:abs-prob` gives the absolute-probability
  consequence of a measured uniform block error; the validation should report the measured error
  in the form the corollary consumes.

### EXP-10: Prediction evaluation

- Status: planned.
- Setup: held-out units or held-out sequence regions from the synthetic and real-data pipelines.
- Comparisons: exported MAP scoring, Bayes-curve scoring, and optional resegmentation scoring.
- Metrics: predictive log score, negative log-likelihood, calibration diagnostics, group
  posterior accuracy where labels are known.
- Expected output: prediction table and calibration plot.
- Success criteria: scoring modes are reproducible and their trade-offs are clearly documented.

### EXP-11: Failure-case and robustness analysis

- Status: planned.
- Setup: weak jumps, close changepoints, family misspecification, heavy-tailed noise,
  missingness, irregular gaps, and incorrect priors.
- Metrics: boundary recovery, posterior concentration, false-positive rate, posterior over `k`,
  runtime, diagnostic failures.
- Expected output: failure-case appendix table.
- Success criteria: limitations are characterized without overstating robustness.

### EXP-12: External baseline comparison

- Status: planned; requires literature and package verification.
- Comparator categories: frequentist changepoint methods (including PELT and the
  functional-pruning algorithms FPOP and SNIP, per edit 6-E3), Bayesian offline alternatives,
  online Bayesian changepoint methods, penalized-likelihood approaches, and domain-specific tools
  for the real-data examples.
- Metrics: boundary recovery, predictive score, calibration when available, runtime, and
  usability constraints.
- Expected output: baseline-comparison table and discussion.
- Success criteria: comparisons are reproducible and cite verified implementations or papers.

### EXP-13: Prior-sensitivity diagnostic (new, planned)

- Status: planned; added in this pass (edit 6-C1).
- Setup: perturb the partition prior `p(k)` and the length factor `g`, holding data fixed, across
  the synthetic and planned real-data benchmarks.
- Metrics: variation of `P(k|y)` and of the boundary marginals under prior perturbation.
- Expected output: a prior-sensitivity row or panel accompanying the planned `p(k)` and `g`
  ablations.
- Success criteria: posterior summaries that are reported as headline results are shown to be
  stable, or their sensitivity is documented, under reasonable prior perturbation.

## 5. Figures and Tables to Generate

### Existing completed synthetic artifacts to preserve unless regenerated deliberately

- `fig:single_synth`: single-sequence Gaussian example.
- `tab:posterior_summary`: posterior summary for the Gaussian example.
- `fig:family_showcase`: likelihood-family showcase.
- `tab:single_quant`: family-specific synthetic summary.
- `fig:calibration`: boundary posterior calibration.
- `fig:latent_groups`: latent-group synthetic diagnostics.
- `tab:nonconj_tradeoff`: non-conjugate approximation trade-off.
- `fig:runtime`: runtime scaling figure.
- `tab:runtime_scaling`: runtime scaling table.

### Placeholder or planned real-data artifacts

All placeholder figure and table captions below now carry a "(PLANNED)" prefix; remove the prefix
only when the artifact is populated with verified output.

- `fig:welllog` and `tab:real_welllog`: replace after a verified well-log pipeline run.
- `fig:cgh` and `tab:real_cgh`: replace after a verified array-CGH pipeline run and an
  annotation-label check.
- `fig:spx` and `tab:real_spx`: replace after a verified S&P 500 data pull, transformation, and
  event-alignment protocol.
- `fig:methylation` and `tab:real_methylation`: replace after a verified methylation dataset,
  atlas/proxy labels, held-out split, and precision model.
- `tab:realdata-status`: update when each planned case study is completed.

### Candidate additional artifacts for the next iteration

- Irregular-design ablation figure/table for EXP-05.
- Pooled-versus-independent replicate ablation table for EXP-06.
- Known-versus-latent group comparison table for EXP-07.
- Latent-template robustness plot for EXP-08.
- Approximation-sensitivity diagnostic table for EXP-09.
- Prediction evaluation table/plot for EXP-10.
- Failure-case appendix table for EXP-11.
- External-baseline comparison table for EXP-12, including FPOP/SNIP.
- Prior-sensitivity diagnostic panel for EXP-13.

## 6. Projected or Expected Results

The manuscript and the planned evaluation program include expected or projected outcomes. These
are not observed results until the corresponding scripts have been run, checked, and archived.
The coding agent must not replace placeholders with guessed values.

Guidelines for replacing expected or projected material:

1. Run the planned experiment using versioned code and recorded seeds.
2. Save raw outputs, processed summaries, generated figures, and generated tables.
3. Record data-source hashes or access details for real-data pipelines.
4. Compare outputs against the manuscript's expectations.
5. Replace expected or projected language with observed-result language only after the author
   approves the verified outputs, and remove the "(PLANNED)" caption prefix at that point.
6. If observed results differ from expectations, update the narrative to match the data rather
   than modifying the data to fit the expectations.

## 7. Theory-to-Code Connections

- Exponential-family block theorem: validate with family-specific block-evidence unit tests and
  numerical references.
- Segment-factorized partition prior: validate with small-`n` brute-force normalizer checks.
- Forward/backward DP exactness: validate against brute-force enumeration for evidence and
  posterior summaries.
- Boundary marginal identity: test that fixed-`k` boundary probabilities sum to the number of
  interior boundaries.
- MAP backtracking: test terminal-score equality and deterministic tie handling; recall that
  marginal-mode selection equals joint MAP only in the degenerate product-measure case
  (`rem:marg-eq-joint`).
- Bayes curves and posterior moments: test moment ratios against enumeration on small cases.
- Irregular-design priors: test physical-coordinate consistency and prior-predictive boundary
  distributions; the renewal interpretation is exact only under translation invariance
  (`rem:renewal-scope`).
- Shared-boundary replicates: test that pooled evidence equals the product of subject-specific
  block evidences under shared boundaries.
- Known-group factorization: test that group-specific outputs factorize conditional on labels.
- Latent-template EM: monitor monotone objective traces under deterministic tie handling. The
  exploratory appendix subsection `app:latent-template-positioning` records a planned head-to-head
  comparison against BASIC-style and JRPM-style multi-sequence inference.
- Non-conjugate stability: the posterior-odds stability proposition (`prop:stability`), the
  boundary-ranking corollary (`cor:ranking`), and the new absolute-probability corollary
  (`cor:abs-prob`) together connect a measured uniform block error to segment-count and
  boundary-posterior error. Compute block-error diagnostics and posterior-sensitivity summaries
  in the form these results consume.
- Prediction layer: test exported MAP, Bayes-curve, and resegmentation scoring on controlled
  held-out examples; budget `O(k_max m^2)` per group for resegmentation.

## 8. Open Technical Questions

### Bibliography author-verification tasks (surfaced by the Phase Two literature investigation)

- Confirm the venue of Bleakley & Vert (2011), "The Group Fused Lasso for Multiple Change-Point
  Detection." The entry is currently recorded as an arXiv preprint (arXiv:1106.4199); the
  originally cited Annals of Applied Statistics venue could not be verified and is likely
  incorrect.
- Confirm which Denison, Mallick & Smith (1998) paper is intended; the title and venue should be
  reconciled (there is a JRSS-B "Automatic Bayesian curve fitting" paper and a separate JASA
  paper).
- Confirm the venue for Punskaya et al. (2002): the IEEE Transactions on Signal Processing
  journal version (currently used) versus the IEEE ICASSP conference version.
- Confirm which Rigaill "pruned dynamic programming" artifact is intended: the 2010 arXiv
  preprint (arXiv:1004.0887, currently used) or the 2015 Journal de la Societe Francaise de
  Statistique version.
- Confirm the end page of Muller, Quintana & Rosner (2011) in JCGS (260--277 versus 260--278).
- Confirm the well-log dataset object name in the installed version of the `changepoint` R
  package family; the appendix recipe uses the placeholder `<welllog_object>`.
- Confirm the methylation atlas distribution channel against the Loyfer et al. (2023)
  data-availability statement: the GitHub repository path (`nloyfer/meth_atlas`) and the GEO
  accession (reported as `GSE186458`).
- Decide whether the planned well-log analysis uses the cleaned or the raw version of the series
  (the manuscript currently states the cleaned version, following Fearnhead & Rigaill 2019).

### Implementation and evaluation decisions

- Confirm the final intended Git repository, branch, and commit-hash policy for manuscript
  artifacts.
- Verify external dataset source names, download commands, package object names, and access
  constraints for all real-data pipelines.
- Decide whether the methylation pipeline uses fixed precision, observation-specific precision,
  or an empirical precision proxy.
- Decide which external baselines are required for the final submission, including whether FPOP
  and SNIP are run, and verify the corresponding citations and licenses.
- Decide the tolerance window and physical-coordinate metric for irregular-grid and real-data
  boundary comparisons.
- Decide whether external real-data labels are ground truth, proxy annotations, expert
  annotations, or qualitative alignment targets.
- Decide how to store result artifacts: CSV tables, parquet/feather outputs, JSON diagnostics,
  PDF/PNG figures, and run logs.
- Decide whether to add an automated CI build for LaTeX.
- Decide whether the conservative `k_max` bound in Corollary `cor:abs-prob` should be tightened
  to a per-state `k` bound in a later iteration.
- Decide whether the exploratory appendix subsection `app:latent-template-positioning` should be
  developed into a full BASIC/JRPM comparison in the main text or kept as an appendix note.

## 9. Files Changed or Added

Updated deliverable files:

- `CHANGELOG.md`
- `CODING_AGENT_HANDOFF.md`

Changed manuscript source files:

- `bayesbreak.tex` (portability shim comments)
- `saim.cls` (portability shim comments)
- `reference/cite.bib` (6 corrected entries, 12 new entries)
- `sections/0-abstract.tex`
- `sections/1.intro.tex`
- `sections/2.problem.tex`
- `sections/4.method.tex`
- `sections/5.algorithms.tex`
- `sections/6.evaluation.tex` (Phase Three edits; Phase Four fix P4-H1; CG-4 well-log body citation)
- `sections/8.appendix.tex` (Phase Three edits; Phase Four fix P4-H2; CG-4 well-log recipe and methylation GitHub correction)
- `sections/3.setup.tex` (CG-3 block-score contract paragraph)
- `sections/1.intro.tex` (Phase Three edits; CG-4 limitations paragraph)
- `sections/7.conclusion.tex` (CG-4 limitations paragraph)
- `sections/4.method.tex` (Phase Three edits; CG-1 score-matrix-exactness remark; CG-2 boundary-event-sum corollary)

Manuscript source files reviewed and unchanged in this pass:

- `sections/3.setup.tex`
- `sections/7.conclusion.tex`
- `math_commands.tex`
- files under `figures/`, `tables/`, and `assets/`

New bibliography entries (12): `oruanaidh1996numerical`, `lai2005comparative`,
`snijders2001assembly`, `loyfer2023atlas`, `killick2014changepoint`, `hartigan1990partition`,
`chib1998estimation`, `wyse2011approximate`, `maidstone2017optimal`, `fearnhead2019changepoint`,
`jewell2022testing`, `truong2018ruptures`. No new LaTeX package, macro, figure file, or table
file was added. The two portability shims reference `lmodern` and `bbm` only through
`\IfFileExists` guards with fallbacks, so no new hard package dependency was introduced.

New manuscript labels (8): `rem:marg-eq-joint`, `cor:abs-prob`, `rem:abs-prob-status`,
`rem:renewal-scope`, `app:latent-template-positioning`, and the labels associated with the two
new annotated-literature rows and the displayed equations introduced by the overfull-box fixes.

## 10. Do-Not-Change Constraints

Do not remove or silently replace these elements without explicit author approval:

- the BayesBreak theoretical direction;
- the modular block-evidence plus DP architecture;
- the irregular-design prior framework;
- the shared-boundary, known-group, and latent-template extensions;
- the non-conjugate approximation interface;
- the prediction and scoring layer;
- the planned real-data experiments;
- expected or projected results language, unless verified outputs replace it;
- placeholder real-data figures and tables, and their "(PLANNED)" caption prefixes, until
  finalized artifacts exist;
- implementation plans and planned baseline comparisons;
- the current LaTeX template, document class, local style files, bibliography style, labels,
  figure paths, and table environments;
- the proposed corollary `cor:abs-prob`, the three new remarks, and the exploratory appendix
  subsection `app:latent-template-positioning`, which were added under the approved Phase Two
  plan and should be treated as part of the current manuscript.

Do not invent completed results, benchmark scores, citations, p-values, observed empirical
findings, or figure/table values. All planned and expected material must remain clearly
distinguished from observed results until the relevant experiment is completed and verified.
