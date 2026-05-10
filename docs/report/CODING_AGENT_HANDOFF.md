# CODING_AGENT_HANDOFF

This handoff is for the next coding agent that will implement BayesBreak experiments, regenerate figures and tables, verify data sources, and prepare the next publication-development iteration. It intentionally distinguishes completed manuscript theory and completed synthetic artifacts from planned experiments, expected outputs, projected outputs, and placeholder real-data artifacts. Do not treat expected or projected results as observed results until the corresponding scripts have been run and audited.

## 1. Project Overview

BayesBreak is a modular offline Bayesian segmentation framework. The manuscript separates local block evidence calculations from global dynamic-programming inference over ordered partitions, then extends the same interface to irregular designs, replicated and grouped sequences, latent-template mixture models, non-conjugate GLM approximations, and posterior-predictive scoring.

Current manuscript status:

- Theoretical and methodological sections are written as completed manuscript contributions.
- The bundled synthetic figures/tables support the current synthetic recovery, family showcase, calibration, latent-group, non-conjugate approximation, and runtime demonstrations.
- The well-log, array-CGH, S&P 500 volatility, and CpG methylation case studies are planned real-data evaluations with placeholder figures/tables and pipeline descriptions.
- External baseline comparisons, extended ablations, and larger robustness studies are planned publication-development work.

## 2. Build Instructions

Root file: `bayesbreak.tex`.

Document class and local style files:

- `saim.cls`
- `natbib.sty`
- `fancyhdr.sty`
- `plainnat.bst`
- `iclr2026_conference.bst`
- `math_commands.tex`

Bibliography file: `reference/cite.bib`.

Inferred bibliography engine: BibTeX-compatible workflow with `plainnat`.

Preferred build sequence when a TeX toolchain is installed:

```bash
pdflatex bayesbreak.tex
bibtex bayesbreak
pdflatex bayesbreak.tex
pdflatex bayesbreak.tex
```

If the local system has a broken `bibtex` symlink but a working original binary, use the working BibTeX-compatible executable and record it in the build log.

Current execution-environment caveat: the handoff pass was run in a container without `latexmk`, `pdflatex`, `bibtex`, `biber`, `xelatex`, `lualatex`, or `tectonic`. Static checks were therefore used in this pass, and the supplied PDF is the latest available compiled render of the packaged manuscript source. Static checks found present inputs, present graphics, unique labels, resolved source-level references, and citation keys present in `reference/cite.bib`.

Known warning class from prior successful builds: underfull-box layout warnings in long tables, algorithms, and formatted appendix material. These are layout warnings, not unresolved-reference or missing-citation errors.

Figure directory: `figures/`.

Table directory: `tables/`.

Auxiliary assets: `assets/`.

## 3. Implementation Tasks

### IMP-01: Core block data model

- Related manuscript sections: 2, 3, 4.1, 5.
- Description: Implement a block-score interface that stores block evidence, optional moment numerators, admissibility status, and any length-prior adjustment.
- Expected input: ordered observations, weights/exposures/trials where applicable, design coordinates, family hyperparameters, admissible block constraints.
- Expected output: block matrices for evidence and requested moment numerators.
- Dependencies: agreed observation-weight convention and physical block-length convention.
- Priority: high.
- Completion criteria: all admissible blocks populated; invalid blocks represented consistently as zero evidence or negative-infinite log score; prefix-sum and direct computation agree on small cases.

### IMP-02: Conjugate family engines

- Related manuscript sections: 4.1, 4.7, Appendix A.1.
- Description: Implement Gaussian, Poisson, Binomial, and Negative-Binomial block routines with evidence and moment tests.
- Expected input: observations, weights/exposures/trials, hyperparameters.
- Expected output: block evidences and observation-scale moment numerators where supported.
- Dependencies: family-specific domain checks.
- Priority: high.
- Completion criteria: closed-form routines match direct small-block numerical checks or independently computed reference values.

### IMP-03: Beta-response quadrature and precision variants

- Related manuscript sections: 4.7, 6, Appendix A.12.
- Description: Implement fixed-precision Beta-response quadrature and decide whether to implement the planned observation-specific precision extension for methylation.
- Expected input: response values in `(0,1)`, precision parameter(s), prior hyperparameters, quadrature settings.
- Expected output: deterministic block log-evidence and moment estimates.
- Dependencies: node-refinement checks and endpoint handling.
- Priority: high for methylation pipeline; medium otherwise.
- Completion criteria: quadrature estimates stable under node refinement; methylation configuration documented as fixed-precision or observation-specific precision.

### IMP-04: Partition-prior normalizers

- Related manuscript sections: 3, 4.2, 4.3, 5, Appendix A.2.
- Description: Implement `C_k` computation under index-uniform, length-aware, segment-cohesion, and renewal-style priors.
- Expected input: `n`, admissible `k`, boundary coordinates, cohesion function, block admissibility constraints.
- Expected output: log normalizers and diagnostic tables.
- Dependencies: physical block-length convention.
- Priority: high.
- Completion criteria: regular-grid index-uniform case matches the combinatorial normalizer; irregular-grid cases pass brute-force enumeration for small `n`.

### IMP-05: Sum-product DP core

- Related manuscript sections: 4.2, 5.
- Description: Implement forward/backward DP in log space for fixed `k` and for posterior over `k`.
- Expected input: block log-evidence matrix, log normalizers, `p(k)`, admissibility mask.
- Expected output: prefix/suffix evidence tables, posterior over segment count, boundary marginals, segment-membership weights.
- Dependencies: IMP-01 and IMP-04.
- Priority: high.
- Completion criteria: small-`n` brute-force enumeration agrees with DP evidence, boundary marginals, and posterior over `k`.

### IMP-06: Max-sum MAP backtracking

- Related manuscript sections: 4.2, 5, Appendix A.7.
- Description: Implement fixed-`k` and across-`k` MAP objectives with deterministic tie handling.
- Expected input: block log scores, count prior, normalizers, chosen `k` mode.
- Expected output: ordered boundary vector and terminal max-sum score.
- Dependencies: IMP-05.
- Priority: high.
- Completion criteria: backtracked score equals stored terminal value; brute-force MAP agrees on small cases.

### IMP-07: Posterior moments and Bayes curves

- Related manuscript sections: 4.1, 4.2, 4.7, 4.9.
- Description: Compute segment-level moments under exported segmentations and Bayes-curve moments averaged over posterior segmentations.
- Expected input: moment numerator matrices and posterior block weights.
- Expected output: pointwise Bayes curves and pointwise uncertainty summaries.
- Dependencies: valid moment targets by family.
- Priority: high.
- Completion criteria: moment ratios agree with direct enumeration on small synthetic cases.

### IMP-08: Shared-boundary replicate pooling

- Related manuscript sections: 4.4, 6, Appendix A.12.
- Description: Implement pooled block evidence across subjects with shared boundaries and subject-specific block parameters.
- Expected input: aligned or union-grid replicate data, subject-specific weights/missingness, hyperparameters.
- Expected output: shared-boundary posterior and subject-level posterior summaries.
- Dependencies: common-grid convention and missingness handling.
- Priority: medium-high.
- Completion criteria: pooled inference reduces to single-sequence inference when there is one subject and to independent evidence multiplication under shared boundaries.

### IMP-09: Known-group inference

- Related manuscript sections: 4.5, 6.
- Description: Implement group-specific DP inference when labels are observed.
- Expected input: group labels, observations, group-specific `k` settings, priors.
- Expected output: group-specific boundary posteriors and summaries.
- Dependencies: IMP-05 and label indexing.
- Priority: medium.
- Completion criteria: group factorization tests pass, and known-label results provide a supervised baseline for latent-group experiments.

### IMP-10: Latent-template EM

- Related manuscript sections: 4.6, 6, Appendix A.5, Appendix A.10.
- Description: Implement finite-template EM with exact responsibility updates and responsibility-weighted template updates under the stated objective.
- Expected input: sequences, number of latent groups, candidate `k` values, priors, restart settings.
- Expected output: fitted templates, responsibilities, mixture weights, objective trace, convergence diagnostics.
- Dependencies: IMP-05 and IMP-06.
- Priority: high.
- Completion criteria: objective is non-decreasing under deterministic tie handling; empty groups handled as specified; restarts are reproducible.

### IMP-11: Non-conjugate approximation routines

- Related manuscript sections: 4.8, 6, Appendix A.8, Appendix A.9.
- Description: Implement or wrap Laplace, Jaakkola--Jordan, Pólya--Gamma mean-field, EP-style, and quadrature reference routines where applicable.
- Expected input: GLM family, design/covariates if any, response data, weights/trials, prior, convergence settings.
- Expected output: approximate block log-evidences and moment estimates where available.
- Dependencies: validated block-error reference on selected blocks.
- Priority: medium-high.
- Completion criteria: diagnostics report max/quantile block errors against a high-accuracy reference on selected problems.

### IMP-12: Prediction and scoring layer

- Related manuscript sections: 2, 4.9, 6.
- Description: Implement posterior-predictive scoring under exported MAP segmentation, Bayes curves, and optional resegmentation.
- Expected input: fitted posterior summaries, new sequence or set-valued unit, family-specific predictive routine.
- Expected output: log scores, group posteriors, responsibilities, predicted signal summaries, pointwise uncertainty.
- Dependencies: IMP-05, IMP-07, and family predictive formulas.
- Priority: medium.
- Completion criteria: scoring modes are unit tested and produce consistent outputs on synthetic data.

### IMP-13: Diagnostics and reporting

- Related manuscript sections: 4.2, 5, 6.
- Description: Implement diagnostic checks for posterior normalization, forward/backward equality, boundary marginal sums, MAP score consistency, block-error sensitivity, and reproducibility metadata.
- Expected input: fitted model output and run metadata.
- Expected output: diagnostic report per experiment.
- Dependencies: all core engines.
- Priority: high.
- Completion criteria: every experiment emits a machine-readable report with pass/fail checks.

### IMP-14: Plotting and table generation

- Related manuscript sections: 6.
- Description: Generate publication figures and tables from experiment outputs with stable file names matching manuscript labels.
- Expected input: experiment result files and plotting scripts.
- Expected output: updated files in `figures/` and `tables/`.
- Dependencies: experiment outputs.
- Priority: high for final publication iteration.
- Completion criteria: placeholder figures/tables replaced only after verified outputs are available.

### IMP-15: Real-data pipeline implementation

- Related manuscript sections: 6, Appendix A.12.
- Description: Implement and verify the well-log, array-CGH, S&P 500, and methylation pipelines.
- Expected input: external datasets or download scripts, preprocessing settings, model hyperparameters.
- Expected output: finalized figures, tables, run logs, and verification notes.
- Dependencies: external data access and source verification.
- Priority: high for publication submission.
- Completion criteria: each pipeline produces a reproducible artifact bundle with data hashes, generated figures/tables, and documented comparison labels.

## 4. Experiment Plan

### EXP-01: Single-sequence Gaussian recovery sweep

- Status: current manuscript includes one completed archived illustration; extended sweep is planned.
- Setup: simulate Gaussian piecewise-constant sequences across sample sizes, segment lengths, noise levels, and jump sizes.
- Baselines: exact BayesBreak index-uniform prior, length-aware prior when applicable, and frequentist segmentation baselines after citation/package verification.
- Metrics: boundary F1 with tolerance, boundary MAE, signal MSE, posterior segment-count calibration, runtime.
- Expected output: updated figure/table summarizing recovery and uncertainty.
- Success criteria: DP diagnostics pass; posterior summaries track recoverability trends under controlled changes.

### EXP-02: Likelihood-family portability benchmark

- Status: current manuscript includes a compact archived family showcase; broader Monte Carlo benchmark is planned.
- Setup: Gaussian, Poisson, Binomial, Negative-Binomial, and Beta-response simulations under matched changepoint structures.
- Metrics: boundary recovery, signal reconstruction, calibration, runtime, family-specific diagnostics.
- Expected output: expanded family summary table and optional supplemental figure.
- Success criteria: common DP backend works across all implemented block engines.

### EXP-03: Boundary posterior calibration study

- Status: current manuscript includes a completed Gaussian calibration illustration; expanded calibration study is planned.
- Setup: repeated simulations across signal-to-noise ratios and segment-count priors.
- Metrics: calibration curves, ECE, reliability by posterior-probability bins.
- Expected output: calibration figure and diagnostic table.
- Success criteria: calibration improves or degrades in interpretable ways as information changes.

### EXP-04: Runtime and memory scaling

- Status: current manuscript includes archived runtime scaling over a limited range; extended scaling is planned.
- Setup: vary `n`, `k_max`, family, prior, and block-engine type.
- Metrics: preprocessing time, DP time, memory footprint, total runtime.
- Expected output: runtime curves and complexity table.
- Success criteria: empirical scaling is consistent with theoretical complexity over tested regimes.

### EXP-05: Irregular-design prior ablation

- Status: planned.
- Setup: simulate irregularly spaced sequences with large gaps and known physical changepoints.
- Comparisons: index-uniform prior, boundary-coordinate prior, segment-cohesion prior, renewal-style prior.
- Metrics: boundary posterior calibration, MAP boundary error in physical coordinates, posterior over `k`, prior predictive boundary distribution.
- Expected output: irregular-design ablation figure and table.
- Success criteria: length-aware priors improve physical-coordinate calibration when design geometry is informative.

### EXP-06: Shared-boundary pooling ablation

- Status: planned.
- Setup: multi-subject simulations with shared boundaries, subject-specific means, heteroscedasticity, and missing observations.
- Comparisons: pooled shared-boundary inference versus independent per-subject inference.
- Metrics: shared-boundary F1, subject-specific signal MSE, calibration, runtime.
- Expected output: pooled-versus-independent ablation table.
- Success criteria: pooling improves boundary recovery when shared-boundary assumptions hold and degrades gracefully when violated.

### EXP-07: Known-group versus latent-group comparison

- Status: planned.
- Setup: grouped sequences with controlled group separation and known labels for oracle comparison.
- Comparisons: known-group DP, latent-template EM, independent sequence segmentation.
- Metrics: label recovery, template boundary F1, responsibility entropy, objective monotonicity.
- Expected output: known-versus-latent summary table.
- Success criteria: latent inference approaches known-group performance as separation and sample size increase.

### EXP-08: Latent-template robustness study

- Status: planned.
- Setup: vary number of groups, restarts, initialization, noise, sequence count, and group imbalance.
- Metrics: objective trace, recovery rate, empty-group frequency, label stability after deterministic ordering.
- Expected output: robustness plots and convergence diagnostics.
- Success criteria: restart strategy and tie handling produce stable templates under identifiable settings.

### EXP-09: Non-conjugate approximation validation

- Status: current manuscript includes a small archived trade-off illustration; broader validation is planned.
- Setup: Bernoulli/logistic and other GLM block models with quadrature or high-accuracy references on selected blocks.
- Metrics: block-error max/quantiles, posterior over `k` sensitivity, boundary-marginal differences, MAP-path stability, moment-error diagnostics, runtime.
- Expected output: approximation diagnostic table and sensitivity figure.
- Success criteria: approximation choices are interpretable through block-error and posterior-sensitivity diagnostics.

### EXP-10: Prediction evaluation

- Status: planned.
- Setup: held-out units or held-out sequence regions from synthetic and real-data pipelines.
- Comparisons: exported MAP scoring, Bayes-curve scoring, and optional resegmentation scoring.
- Metrics: predictive log score, negative log-likelihood, calibration diagnostics, group posterior accuracy where labels are known.
- Expected output: prediction table and calibration plot.
- Success criteria: scoring modes are reproducible and their trade-offs are clearly documented.

### EXP-11: Failure-case and robustness analysis

- Status: planned.
- Setup: weak jumps, close changepoints, family misspecification, heavy-tailed noise, missingness, irregular gaps, and incorrect priors.
- Metrics: boundary recovery, posterior concentration, false positive rate, posterior over `k`, runtime, diagnostic failures.
- Expected output: failure-case appendix table.
- Success criteria: limitations are characterized without overstating robustness.

### EXP-12: External baseline comparison

- Status: planned and requires literature/package verification.
- Comparator categories: frequentist changepoint methods, Bayesian offline alternatives, online Bayesian changepoint methods, penalized likelihood approaches, and domain-specific tools for real-data examples.
- Metrics: boundary recovery, predictive score, calibration when available, runtime, and usability constraints.
- Expected output: baseline-comparison table and discussion.
- Success criteria: comparisons are reproducible and cite verified implementations or papers.

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

- `fig:welllog` and `tab:real_welllog`: replace after verified well-log pipeline run.
- `fig:cgh` and `tab:real_cgh`: replace after verified array-CGH pipeline run and annotation-label check.
- `fig:spx` and `tab:real_spx`: replace after verified S&P 500 data pull, transformation, and event-alignment protocol.
- `fig:methylation` and `tab:real_methylation`: replace after verified methylation dataset, atlas/proxy labels, held-out split, and precision model.
- `tab:realdata-status`: update when each planned case study is completed.

### Candidate additional artifacts for the next iteration

- Irregular-design ablation figure/table for EXP-05.
- Pooled-versus-independent replicate ablation table for EXP-06.
- Known-versus-latent group comparison table for EXP-07.
- Latent-template robustness plot for EXP-08.
- Approximation-sensitivity diagnostic table for EXP-09.
- Prediction evaluation table/plot for EXP-10.
- Failure-case appendix table for EXP-11.
- External-baseline comparison table for EXP-12.

## 6. Projected or Expected Results

The manuscript and planned evaluation program include expected or projected outcomes. These are not observed results until the corresponding scripts have been run, checked, and archived. The coding agent must not replace placeholders with guessed values.

Guidelines for replacing expected/projected material:

1. Run the planned experiment using versioned code and recorded seeds.
2. Save raw outputs, processed summaries, generated figures, and generated tables.
3. Record data-source hashes or access details for real-data pipelines.
4. Compare outputs against manuscript expectations.
5. Replace expected/projected language with observed-result language only after the author approves the verified outputs.
6. If observed results differ from expectations, update the narrative to match the data rather than modifying the data to fit expectations.

## 7. Theory-to-Code Connections

- Exponential-family block theorem: validate with family-specific block evidence unit tests and numerical references.
- Segment-factorized partition prior: validate with small-`n` brute-force normalizer checks.
- Forward/backward DP exactness: validate against brute-force enumeration for evidence and posterior summaries.
- Boundary marginal identity: test that fixed-`k` boundary probabilities sum to the number of interior boundaries.
- MAP backtracking: test terminal score equality and deterministic tie handling.
- Bayes curves and posterior moments: test moment ratios against enumeration on small cases.
- Irregular-design priors: test physical-coordinate consistency and prior-predictive boundary distributions.
- Shared-boundary replicates: test pooled evidence equals product of subject-specific block evidences under shared boundaries.
- Known-group factorization: test that group-specific outputs factorize conditional on labels.
- Latent-template EM: monitor monotone objective traces under deterministic tie handling.
- Non-conjugate stability: compute block-error diagnostics and posterior-sensitivity summaries.
- Prediction layer: test exported MAP, Bayes-curve, and resegmentation scoring on controlled held-out examples.

## 8. Open Technical Questions

- Confirm the final intended Git repository, branch, and commit-hash policy for manuscript artifacts.
- Verify external dataset source names, download commands, package object names, and access constraints for all real-data pipelines.
- Decide whether methylation uses fixed precision, observation-specific precision, or an empirical precision proxy.
- Decide which external baselines are required for the final submission and verify corresponding citations and licenses.
- Decide the tolerance window and physical-coordinate metric for irregular-grid and real-data boundary comparisons.
- Decide whether external real-data labels are ground truth, proxy annotations, expert annotations, or qualitative alignment targets.
- Decide how to store result artifacts: CSV tables, parquet/feather outputs, JSON diagnostics, PDF/PNG figures, and run logs.
- Decide whether to add an automated CI build for LaTeX once a TeX toolchain is available.
- Decide whether to include additional appendix figures/tables or keep some outputs handoff-only for the next iteration.

## 9. Files Changed or Added in This Pass

Added or updated deliverable files:

- `CHANGELOG.md`
- `CODING_AGENT_HANDOFF.md`

Manuscript files reviewed and preserved in the package:

- `bayesbreak.tex`
- `sections/0-abstract.tex`
- `sections/1.intro.tex`
- `sections/2.problem.tex`
- `sections/3.setup.tex`
- `sections/4.method.tex`
- `sections/5.algorithms.tex`
- `sections/6.evaluation.tex`
- `sections/7.conclusion.tex`
- `sections/8.appendix.tex`
- `reference/cite.bib`
- files under `figures/`, `tables/`, and `assets/`

No new LaTeX package, macro, figure file, table file, or bibliography entry was added during the handoff pass.

## 10. Do-Not-Change Constraints

Do not remove or silently replace these elements without explicit author approval:

- the BayesBreak theoretical direction;
- the modular block-evidence plus DP architecture;
- the irregular-design prior framework;
- shared-boundary, known-group, and latent-template extensions;
- non-conjugate approximation interface;
- prediction/scoring layer;
- planned real-data experiments;
- expected or projected results language unless verified outputs replace it;
- placeholder real-data figures and tables until finalized artifacts exist;
- implementation plans and planned baseline comparisons;
- current LaTeX template, document class, local style files, bibliography style, labels, figure paths, and table environments.

Do not invent completed results, benchmark scores, citations, p-values, observed empirical findings, or figure/table values. All planned and expected material must remain clearly distinguished from observed results until the relevant experiment is completed and verified.
