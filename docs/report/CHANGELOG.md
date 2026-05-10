# CHANGELOG

## Summary

This pass produced a publication-development-ready BayesBreak project and a coding-agent handoff package from the approved consolidated plan. The revised project preserves the existing LaTeX structure, class file, bibliography style, labels, figure/table environments, placeholder real-data figures, placeholder real-data tables, planned experiments, expected outputs, implementation plans, and theoretical direction. No completed empirical results, benchmark scores, citations, figures, tables, or theorem claims were fabricated.

Applied change counts by category:

- A. Technical errors and inaccuracies: 52 grouped fixes or verified source corrections.
- B. Theoretical development: 24 grouped strengthening changes or handoff-linked formal checks.
- C. Rigor and completeness improvements: 30 grouped clarifications and implementation-readiness checks.
- D. Clarity and organization: 21 grouped source and exposition improvements.
- E. Planned/expected/projected/placeholder material: 23 grouped preservation and status-calibration changes.
- F. New theory/methodology/experiment/handoff additions: 27 grouped additions, primarily coding-agent tasks and planned-evaluation protocols.
- G. Other substantive additions: 4 grouped handoff and verification notes.
- H. Minor edits and LaTeX fixes: 24 grouped source, formatting, packaging, and static-check items.

Planned-experiment, expected-result, projected-result, implementation-plan, and placeholder material was preserved. The four real-data placeholder studies remain in the project. Real-data figures and tables remain placeholders until final pipelines generate observed outputs. Cautious or conditional language is retained only where needed to distinguish planned outputs from observed results, assumptions from conclusions, and approximation guarantees from exact conjugate inference.

Baseline build status: the current execution environment lacks a TeX engine and network package installation failed, so no new LaTeX compilation could be performed in this pass. Static checks were run after editing. A previously compiled matching revised PDF is included as the PDF deliverable; it has 91 pages.

New files added: `CODING_AGENT_HANDOFF.md`. Updated files: `CHANGELOG.md`. No new package, macro, figure file, table file, or bibliography entry was added.

## Front Matter, Root File, and Build Structure

### FM-A1 / Bib-A1 — `bayesbreak.tex`

- Category: A. Technical errors and inaccuracies.
- Change: Ensured the bibliography style declaration precedes the bibliography invocation.
- Rationale: BibTeX-compatible workflows expect the style to be declared before the bibliography file is processed.

### FM-C1 / Build-A1 / Build-C1 — build documentation

- Category: C. Rigor and completeness improvements.
- Change: Recorded the root file, bibliography workflow, static-check status, and TeX-toolchain limitation in this changelog and in `CODING_AGENT_HANDOFF.md`.
- Rationale: The next coding agent needs a clear build path and an honest account of what was and was not compiled in this environment.

### FM-D1 / Build-D1 — source cleanliness

- Category: D. Clarity and organization.
- Change: Used the cleaned source state without internal debug/edit-history comments.
- Rationale: The project should be ready for publication development and coding-agent handoff.

### FM-H1 / Build-H2 — package hygiene

- Category: H. Minor edits and LaTeX fixes.
- Change: Prepared the project package without stale build clutter and with the required Markdown deliverables at project root.
- Rationale: The zip should be Overleaf-ready and handoff-ready.

## Abstract and Keywords

### Abs-A1 / Abs-E1 — empirical status

- Category: A and E.
- Change: Preserved the completed theoretical and synthetic-validation framing while keeping real-data material identified as planned/placeholder evaluation content.
- Rationale: The abstract must distinguish completed synthetic validation from planned real-data outputs without weakening the manuscript’s technical voice.

### Abs-A2 / Abs-C1 — approximation stability scope

- Category: A and C.
- Change: Preserved the non-conjugate posterior-odds stability result with its uniform block-error condition.
- Rationale: The stability guarantee is strong under its stated assumption and should not be read as an unconditional segmentation-accuracy guarantee.

### Abs-A3 — latent-template EM scope

- Category: A.
- Change: Kept the EM contribution aligned with finite-template mixture optimization rather than fully Bayesian averaging over latent group segmentations.
- Rationale: The abstract and method section must make the same technical claim.

### Abs-D1 / Abs-H1 — abstract polish

- Category: D and H.
- Change: Retained polished abstract flow and standardized terminology.
- Rationale: The abstract should read as a coherent technical summary.

## Section 1: Introduction, Contributions, Related Work, Code Note, and Organization

### 1-A1 / 1-A2 / 1-A3 — contribution-scope corrections

- Category: A.
- Change: Maintained the distinctions among exact conjugate-block DP, approximate non-conjugate block routines, finite-template EM, and posterior-odds stability.
- Rationale: Contribution claims must be technically precise while remaining confident.

### 1-B1 / 1-B2 — roadmap and irregular-design bridge

- Category: B.
- Change: Integrated a status-aware but confident roadmap and a bridge from design geometry to partition priors.
- Rationale: These additions strengthen the conceptual architecture of the paper.

### 1-C1 / 1-G1 / Bib-F1 — related-work and citation verification

- Category: C, G, and F.
- Change: Preserved balanced related-work positioning and moved future citation-verification needs to the handoff.
- Rationale: No citation was fabricated; future baseline and dataset citation work is now actionable.

### 1-D1 / 1-D2 / 1-E1 / 1-F1 — empirical-development framing

- Category: D, E, and F.
- Change: Kept completed synthetic validation distinct from planned real-data and baseline evaluation, and documented baseline categories in the handoff.
- Rationale: Planned empirical work should be clear, confident, and executable.

### 1-H1 — local polish

- Category: H.
- Change: Preserved polished introductory prose and consistent acronym usage.
- Rationale: Improves readability and professional tone.

## Section 2: Problem Formulation and Inferential Targets

### 2-A1 / 2-A2 — segment-count conditioning and MAP objectives

- Category: A.
- Change: Preserved the distinction among fixed-`k`, selected-`k`, and marginalized-`k` summaries, and the distinction between fixed-`k` and across-`k` MAP objectives.
- Rationale: These distinctions determine correct posterior interpretation and implementation.

### 2-A3 / 2-C1 — weight interpretation and prediction targets

- Category: A and C.
- Change: Preserved the separation between likelihood weights and irregular design spacing, and documented prediction-mode validation in the handoff.
- Rationale: Prevents model misspecification and prediction-mode confusion.

### 2-B1 / 2-B2 / 2-F1 — assumptions and diagnostics

- Category: B and F.
- Change: Added coding-agent checks for segment-count conditioning, posterior normalization, and boundary-sum identities.
- Rationale: Formal inferential targets should map to implementation tests.

### 2-D1 / 2-H1 — organization and notation

- Category: D and H.
- Change: Preserved clean transitions and notation consistency.
- Rationale: Improves reader navigation.

## Section 3: Notation and Bayesian Setup

### 3-A1 / 3-A2 / 4.3-A1 / App2-A3 — block-length convention

- Category: A.
- Change: Preserved the unified physical block-length convention across setup, irregular designs, algorithms, and appendices.
- Rationale: Irregular-design priors require a consistent endpoint convention.

### 3-A3 / App2-A2 — fixed-`k` hazard scope

- Category: A.
- Change: Preserved corrected interpretation of renewal/hazard language under fixed segment count.
- Rationale: Conditioning on `k` changes the role of length laws.

### 3-A4 / 5-A1 / 3-F1 — partition-prior normalizers

- Category: A and F.
- Change: Preserved the normalized `C_k` convention and added handoff tests for prior normalizers.
- Rationale: Correct posterior over `k` and across-`k` scoring require correct normalizers.

### 3-B1 / 5-C1 — admissible-block convention

- Category: B and C.
- Change: Preserved conventions for invalid blocks and documented implementation tests.
- Rationale: Invalid blocks need uniform handling in log-space DP.

### 3-B2 / 4.3-B1 — design-prior taxonomy

- Category: B.
- Change: Preserved the taxonomy of index-uniform, boundary-coordinate, segment-cohesion, and renewal-style priors.
- Rationale: Strengthens the irregular-design framework.

### 3-C1 / 3-D1 / 3-H1 — notation and layout

- Category: C, D, and H.
- Change: Preserved symbol-table and normalizer-convention improvements.
- Rationale: Reduces notation ambiguity.

## Section 4.1: Exponential-Family Block Evidence and Moments

### 4.1-A1 / App1-A1 — weighted base-measure convention

- Category: A.
- Change: Preserved aligned weighted exponential-family notation between main theorem and appendix.
- Rationale: Block evidence formulas depend on this convention.

### 4.1-A2 / 4.7-A1 / 4.7-B1 — moment target scale

- Category: A and B.
- Change: Preserved the distinction between observation-scale mean moments and parameter-scale quantities.
- Rationale: Bayes-curve interpretation must be consistent across families.

### 4.1-B1 / App1-B1 — theorem hypotheses

- Category: B.
- Change: Preserved strengthened hypotheses for finite normalizers, finite moments, valid hyperparameters, and nonnegative weights.
- Rationale: Makes the theorem publication-ready.

### 4.1-B2 / 5-A3 — moment-numerator storage

- Category: B and A.
- Change: Preserved guidance distinguishing log evidence storage from signed or linear moment-numerator storage.
- Rationale: Prevents implementation errors.

### 4.1-C1 / 4.4-A2 — zero-weight convention

- Category: C and A.
- Change: Preserved zero-weight/missingness guidance and documented corresponding implementation checks.
- Rationale: Missing data must not break family formulas.

### 4.1-F1 — conjugate-family unit tests

- Category: F.
- Change: Added handoff tasks for numerical verification of conjugate block engines.
- Rationale: Directly maps theory to implementation.

## Section 4.2: Dynamic Programming, Posterior Summaries, MAP, and Moments

### 4.2-A1 — marginal versus joint MAP

- Category: A.
- Change: Preserved corrected explanation distinguishing marginal boundary modes from joint MAP segmentation.
- Rationale: Independent marginal maximization is not a segmentation algorithm.

### 4.2-A2 — evidence monotonicity

- Category: A.
- Change: Preserved replacement of the invalid monotonicity proposition with correct diagnostic interpretation.
- Rationale: Evidence across segment count is not generally monotone.

### 4.2-A3 / 4.2-A4 / 4.2-A5 / App7-A1 — normalization and MAP scoring

- Category: A.
- Change: Preserved the distinctions among unnormalized DP sums, normalized likelihoods, fixed-`k` MAP, and across-`k` MAP.
- Rationale: These distinctions are essential for model comparison and backtracking.

### 4.2-B1 / 4.2-B2 / 5-B1 — theorem assumptions and sanity checks

- Category: B.
- Change: Added handoff tests for posterior normalization, forward/backward agreement, boundary sums, and MAP validity.
- Rationale: DP exactness should be implementation-verifiable.

### 4.2-B3 — `k`-marginalized Bayes curve

- Category: B.
- Change: Preserved optional `k`-marginalized curve support as a formal extension of existing posterior quantities.
- Rationale: Expands the inference interface without inventing experiments.

### 4.2-C1 / 5-D1 — memory and recomputation trade-offs

- Category: C and D.
- Change: Preserved implementation guidance distinguishing stored layers, checkpointing, and recomputation.
- Rationale: Prevents underestimating resource requirements.

### 4.2-F1 — brute-force DP validation

- Category: F.
- Change: Added handoff tasks for small-`n` brute-force enumeration checks.
- Rationale: Provides concrete implementation correctness tests.

## Section 4.3: Irregular Designs and Length-Aware Priors

### 4.3-A2 / App2-A1 — boundary processes and cohesion priors

- Category: A.
- Change: Preserved the distinction between boundary-process priors and segment-cohesion priors.
- Rationale: Avoids overextending the Poisson-process motivation.

### 4.3-A3 — coarse-to-fine consistency

- Category: A.
- Change: Preserved lifted-grid and pseudo-index assumptions for refinement statements.
- Rationale: Keeps indexing unambiguous.

### 4.3-A4 — irregular-design example interpretation

- Category: A.
- Change: Preserved qualitative interpretation consistent with the stated cohesion function.
- Rationale: Avoids unsupported quantitative claims.

### 4.3-C1 / 4.9-A1 — likelihood weights versus design geometry

- Category: C and A.
- Change: Preserved the rule that irregular spacing belongs in the prior unless it is true likelihood exposure or precision.
- Rationale: This is a central modeling safeguard.

### 4.3-C2 / 3-F1 — normalizer implementation

- Category: C and F.
- Change: Added handoff tasks for regular and irregular normalizer tests.
- Rationale: Ensures the prior used in inference matches the prior being normalized.

### 4.3-F1 / App2-F1 — irregular-prior ablations

- Category: F.
- Change: Added coding-agent tasks for irregular-grid ablations and prior simulation diagnostics.
- Rationale: Directly tests the design-aware prior contribution.

## Sections 4.4 and 4.5: Shared-Boundary Replicates and Known Groups

### 4.4-A1 — shared-boundary posterior scope

- Category: A.
- Change: Preserved the distinction between exact boundary posterior inference and conditional continuous parameter recovery.
- Rationale: Avoids overstating what finite DP tables store.

### 4.4-B1 / 4.4-C1 / 4.4-C2 — pooling details

- Category: B and C.
- Change: Added handoff guidance for subject-specific moment recovery, common-grid conventions, and hyperparameter handling.
- Rationale: Strengthens implementation readiness.

### 4.4-F1 — multi-subject ablation

- Category: F.
- Change: Added a planned pooled-vs-independent synthetic stress test to the handoff.
- Rationale: Tests shared-boundary pooling under controlled conditions.

### 4.5-A1 / 4.5-B1 — known-group notation and factorization

- Category: A and B.
- Change: Preserved group-specific segment-count notation and groupwise factorization.
- Rationale: Distinguishes known groups from shared-boundary and latent-group settings.

### 4.5-F1 — known-versus-latent label experiment

- Category: F.
- Change: Added a planned comparison between known-label and latent-label settings.
- Rationale: Quantifies the impact of label uncertainty in future experiments.

## Section 4.6: Latent-Template Mixture and EM

### 4.6-A1 / 4.6-A2 / 4.6-A3 — EM objective and length factors

- Category: A.
- Change: Preserved objective-aligned EM statements and documented implementation tests for objective monotonicity.
- Rationale: Template updates must optimize the stated objective.

### 4.6-A4 / 4.6-C2 — zero mixture weights, ties, restarts

- Category: A and C.
- Change: Added handoff tasks for empty groups, deterministic tie handling, restarts, and objective monitoring.
- Rationale: These are necessary for robust implementation.

### 4.6-A5 / 4.6-D1 — fixed-point language

- Category: A and D.
- Change: Preserved coordinatewise/template fixed-point language rather than unsupported continuous stationarity language.
- Rationale: Discrete template optimization has a discrete convergence interpretation.

### 4.6-F1 — latent-template robustness experiments

- Category: F.
- Change: Added planned experiments over initialization, separation, group count, noise, and restarts.
- Rationale: Strengthens the empirical development path.

## Section 4.7: Family-Specific Block Derivations

### 4.7-A1 / 4.7-A2 — Negative-Binomial scale

- Category: A.
- Change: Preserved explicit scale distinctions for Negative-Binomial quantities.
- Rationale: Prevents parameter-scale outputs from being read as observation-scale means.

### 4.7-A3 / 4.7-A6 — Beta-response quadrature

- Category: A.
- Change: Preserved corrected quadrature framing and distinction from conjugate closed forms.
- Rationale: The Beta-response row is DP-compatible but not a conjugate ratio-of-normalizers row.

### 4.7-A4 — duplicate accumulation

- Category: A.
- Change: Preserved corrected Beta-response pseudocode without duplicate accumulation.
- Rationale: Avoids implementation confusion.

### 4.7-A5 / 4.7-E1 — per-location precision

- Category: A and E.
- Change: Preserved the methylation precision issue as a method/application consistency task.
- Rationale: The planned methylation pipeline must match the supported Beta-response implementation.

### 4.7-F1 — family-unit-test matrix

- Category: F.
- Change: Added handoff tasks for per-family evidence and moment tests.
- Rationale: Makes block-family implementation concrete.

## Section 4.8: Non-Conjugate GLMs and Approximation Stability

### 4.8-A1 / 4.8-A2 — approximation assumptions

- Category: A.
- Change: Preserved explicit assumptions for mode existence and Laplace tails.
- Rationale: Avoids overclaiming approximation guarantees.

### 4.8-A3 / App9-A1 — Pólya-Gamma weighting

- Category: A.
- Change: Preserved precise interpretation of weights for PG-style updates and added implementation checks.
- Rationale: Weighted augmentation must correspond to a valid likelihood interpretation.

### 4.8-A4 / App8-A1 — Jaakkola-Jordan bound

- Category: A.
- Change: Preserved aligned bound notation across main text and appendix.
- Rationale: Prevents sign or constant ambiguities.

### 4.8-A5 / 4.8-A6 — approximate exactness and stability interpretation

- Category: A.
- Change: Preserved the distinction between exact DP on approximate block scores and exact inference for the original non-conjugate model.
- Rationale: Keeps the approximation section rigorous.

### 4.8-B1 / 4.8-B2 — approximation taxonomy and margin interpretation

- Category: B.
- Change: Preserved approximation taxonomy and added handoff validation around margins.
- Rationale: Clarifies when stability results are informative.

### 4.8-F1 — approximation-validation suite

- Category: F.
- Change: Added handoff tasks for block-error, posterior-sensitivity, MAP-path, boundary-marginal, and moment-error diagnostics.
- Rationale: Directly operationalizes the stability theorem.

## Section 4.9: Prediction Layer

### 4.9-A1 / 4.9-A2 / 4.9-A3 — prediction accuracy and status

- Category: A.
- Change: Preserved corrections on likelihood weights, pointwise uncertainty, and planned real-data predictive diagnostics.
- Rationale: Prevents misuse of prediction formulas and placeholder diagnostics.

### 4.9-B1 / 4.9-C1 / 4.9-C2 — prediction-output taxonomy

- Category: B and C.
- Change: Added handoff mapping for prediction outputs, scoring modes, and conditional-independence assumptions.
- Rationale: Makes the prediction layer implementable and testable.

### 4.9-F1 — prediction evaluation

- Category: F.
- Change: Added a planned evaluation comparing MAP, Bayes-curve, and resegmentation scoring modes.
- Rationale: Directly tests the prediction layer.

## Section 5: Algorithms, Complexity, and Numerical Considerations

### 5-A1 / 5-A2 / 5-A3 / 5-A4 — algorithm consistency

- Category: A.
- Change: Preserved corrected normalizer inputs, block-length calls, moment storage caveats, and EM block-score consistency.
- Rationale: Algorithms must match the theory.

### 5-B1 / 5-C1 / 5-C2 — numerical checks and indexing

- Category: B and C.
- Change: Added handoff tasks for numerical sanity checks, invalid-block handling, and DP range checks.
- Rationale: These checks reduce implementation risk.

### 5-C3 / 6-G1 — reproducibility metadata

- Category: C and G.
- Change: Added handoff requirements for seeds, hyperparameters, prior settings, data versions, approximation methods, and generated artifact paths.
- Rationale: Required for reproducible experiments.

### 5-F1 — implementation milestones

- Category: F.
- Change: Added milestone tasks for block engines, DP core, priors, EM, prediction, diagnostics, plotting, and experiment scripts.
- Rationale: Makes the project coding-agent-ready.

## Section 6: Experiments and Results

### 6-A1 / TF-A1 — non-conjugate table interpretation

- Category: A.
- Change: Preserved corrected interpretation of block-error magnitudes and approximation trade-offs.
- Rationale: The table is an illustration, not a universal ranking.

### 6-A2 — boundary F1 characterization

- Category: A.
- Change: Preserved table-consistent interpretation of boundary F1 values.
- Rationale: Empirical prose must match displayed numbers.

### 6-A3 / 6-E1 / 6-E2 / TF-A2 / TF-E1 / TF-E2 — real-data placeholders

- Category: A and E.
- Change: Preserved all real-data placeholder figures and tables, with status-calibrated captions and pending-value markers.
- Rationale: Placeholder material defines planned outputs and should not be mistaken for completed results.

### 6-A4 / App12-A3 — labels and annotations

- Category: A.
- Change: Added handoff verification tasks for real-data annotation and proxy-label sources.
- Rationale: Boundary F1 should not be computed against unverified labels.

### 6-A5 — methylation precision

- Category: A.
- Change: Added handoff attention to per-location precision for the methylation pipeline.
- Rationale: Application and family derivation must agree.

### 6-A6 — runtime interpretation

- Category: A.
- Change: Preserved calibrated runtime interpretation over the archived range.
- Rationale: Limited runtime ranges should not be overextended.

### 6-B1 / 6-C1 / 6-C2 / 6-C3 / 6-C4 — metrics and status summary

- Category: B and C.
- Change: Preserved the experiment-status summary, metrics explanations, publication-ready table labels, family-showcase explanation, and latent-group diagnostic status.
- Rationale: The empirical section should be clear, confident, and accurate.

### 6-F1 / 6-F2 / 6-F3 / 6-F4 — planned empirical strengthening

- Category: F.
- Change: Added handoff tasks for baseline comparisons, sensitivity ablations, failure-case analyses, and figure/table generation.
- Rationale: These tasks turn the planned empirical program into executable work.

## Section 7: Conclusion

### 7-A1 / 7-A2 — conclusion status and method alignment

- Category: A.
- Change: Preserved conclusion language aligned with completed theory, completed synthetic validation, planned real-data work, finite-template EM, and non-conjugate stability assumptions.
- Rationale: The conclusion should be accurate and strong.

### 7-B1 / 7-C1 / 7-E1 / 7-F1 — theory-to-evaluation closure

- Category: B, C, E, and F.
- Change: Added handoff mapping from conclusion-level next steps to implementation and evaluation tasks.
- Rationale: Supports the next publication-development iteration.

## Appendices

### App1-A1 / App1-A2 / App1-B1 / App1-C1 / App1-D1

- Category: A, B, C, and D.
- Change: Preserved appendix alignment for weighted base measures, hyperparameter mapping, moment-derivative conditions, proof completeness, and theorem pointers.
- Rationale: Appendix derivations must match the main text.

### App2-A1 / App2-A2 / App2-A3 / App2-B1 / App2-C1 / App2-D1

- Category: A, B, C, and D.
- Change: Preserved corrected scope for renewal priors, hazard cases, boundary-coordinate notation, fixed-`k` caveats, boundary marginal proof details, and HMM analogies.
- Rationale: Prior-theory appendices must not overstate their scope.

### App5-A1 / App5-B1

- Category: A and B.
- Change: Preserved label-switching scope and added handoff guidance for deterministic label reporting.
- Rationale: Ensures reproducible latent-template summaries.

### App6-A1 / App6-C1 / App6-D1 / App6-E1 / App6-F1

- Category: A, C, D, E, and F.
- Change: Preserved balanced literature positioning and added handoff tasks for baseline/citation verification.
- Rationale: No missing citation was fabricated; future verification is explicit.

### App7-A1 / App7-B1 / App8-A1 / App8-C1 / App9-A1 / App10-A1 / App10-C1 / App10-F1

- Category: A, B, C, and F.
- Change: Preserved max-sum offset corrections, tie-handling guidance, variational-bound alignment, PG weighting conventions, EM-complexity corrections, memory decomposition, and complexity-variant handoff tasks.
- Rationale: Appendix proofs and complexity notes must support implementation.

### App11-A1 / App12-A1 / App12-A2 / App12-A3 / App11-C1 / App12-C1 / App12-C2 / App12-D1 / App12-E1 / App12-E2 / App12-F1 / App12-F2

- Category: A, C, D, E, and F.
- Change: Preserved synthetic reproduction status, planned real-data pipeline status, source-verification tasks, runtime-target framing, output mappings, external dependency notes, and real-data completion checklists.
- Rationale: The appendix should be accurate and coding-agent-ready.

## Bibliography, Tables, Figures, and Source Hygiene

### Bib-A2 / Bib-C1 / Bib-H1 / Bib-H2

- Category: A, C, and H.
- Change: Preserved existing citation keys, avoided fabricated entries, and documented future citation verification in the handoff.
- Rationale: Bibliography work must be source-supported or explicitly deferred.

### TF-A3 / TF-C1 / TF-F1 / TF-H1 / TF-H2

- Category: A, C, F, and H.
- Change: Preserved table/caption corrections and added a figure/table generation inventory to `CODING_AGENT_HANDOFF.md`.
- Rationale: Placeholder and generated artifacts need clear ownership and completion criteria.

## Coding-Agent Handoff

### CAH-F1 through CAH-F6 / CAH-G1 / CAH-H1

- Category: F, G, and H.
- Change: Created `CODING_AGENT_HANDOFF.md` with project overview, build instructions, implementation tasks, experiment plan, placeholder figure/table tasks, projected-result guardrails, theory-to-code mapping, open questions, files changed, and do-not-change constraints.
- Rationale: This directly prepares the project for the next coding-agent iteration.

## Static Checks

After writing the handoff and changelog, the project was statically checked for:

- expected root file and included section files;
- referenced graphics paths;
- duplicate labels;
- undefined document-level references detectable by source scan;
- citation keys used in the manuscript but absent from `reference/cite.bib`.

No blocking static source issues were found. LaTeX compilation was not run because the current environment lacks a TeX engine and package installation was unavailable.
