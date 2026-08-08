---
name: bayesbreak-phase6-validation
description: 'Audit or rerun integrated BayesBreak Phase 6 validation and Gate E release checks. Use for the completed Gate D evidence, corrected-result hashes, review PR readiness, merge gating, v2.0.0rc3 tag readiness, or PyPI verification.'
argument-hint: '[plan|execute|status|gate-e|release-status]'
user-invocable: true
disable-model-invocation: true
---

# BayesBreak Phase 6 Integrated Validation

Validate the adopted release from source, code, data, and result lineage. A successful command
does not override a scientific-status failure, and an unresolved check is never counted as a
pass.

## Current Verified Baseline

- Integrated Gate D validation is complete; inspect before rerunning it.
- Canonical corrected results: `RES-BB-SYN-005`, `RES-BB-CMP-003`, and
  `RES-BB-RD-008Q`, each with a current sidecar and validated artifact hashes.
- Canonical result count: 20 unique records. `RES-BB-CMP-002` and
  `RES-BB-RD-007Q` remain excluded from their original intended conclusions.
- Terminal profile: 324/324 tests passed, 76 percent package coverage, repository-pinned
  all-file hooks passed, mypy passed 53 package files, strict MkDocs passed under `/tmp`,
  and the `2.0.0rc3` sdist/wheel passed Twine and metadata checks.
- Evidence sources: `provenance/test-manifest.json`, corrected result sidecars, canonical
  registries, and `report/revision_artifacts/adoption/ADOPTION_LEDGER.md`.

For Gate E, first verify that the evidence still matches the review head. Rerun only checks
made stale by later changes; never rerun expensive scientific experiments for release review.

## Preconditions

1. Read [the adoption orchestrator](../bayesbreak-phase6-adoption/SKILL.md).
2. Require explicit Gate D approval for a new integrated validation execution. Status and
  Gate E evidence audits are read-only and use the completed terminal profile.
3. Record local changes and preserve unrelated/user edits.
4. Load the adoption ledger, canonical task/claim/experiment/result registries, and all
   generated sync manifests.
5. Resolve commands from the adopted repository and report Makefile rather than assuming the
   archive's environment.

## Validation Order

Run cheap, discriminating checks first. Stop dependent checks after a blocking failure while
continuing independent checks that can provide useful evidence.

### 1. Static Integrity

- Validate archive and adopted-artifact SHA-256 records.
- Reject absolute paths, traversal, symlinks, duplicate/case-colliding paths, caches, build
  products in source packages, and untracked generated artifacts.
- Parse all JSON, TOML, YAML, BibTeX, and schema files with structured tools.
- Verify skill/customization frontmatter and repository links.
- Verify one canonical package version, citation target, report root, and generated handoff
  source.

### 2. Scientific Invariants

- Check the exact title and generalized hierarchical narrative across all entry points.
- Check exact-conjugate versus approximate-nonconjugate qualification.
- Check sum-product posterior versus max-sum joint-MAP terminology.
- Check structural partition support, reference descriptor reporting, fixed-count Poisson
  odds, latent-group group weight, conditional approximation bounds, and one-to-one matching.
- Confirm 20 scientific result records and the current interpretation of each, unless a newer
  canonical registry intentionally changes the count with validated lineage.
- Confirm `RES-BB-CMP-002` and `RES-BB-RD-007Q` remain excluded for their intended claims.
- Confirm no claim uses BayesBreak MAP agreement as independent ground truth.
- Confirm unresolved proof, EP timeout, external annotation, venue, repository, and data
  location states remain explicit unless supported by newer evidence.

### 3. Python Quality and Behavior

Run, using the project's configured environment:

1. focused tests for every changed `CODE-BB-*` task;
2. schema, provenance, historical read-only, and corrected-lineage regression tests;
3. exact finite-case/property tests against exhaustive enumeration;
4. Beta predictive and coordinate-support tests;
5. latent objective/restart and CGH axis-rejection tests;
6. full `pytest` with recorded collection/pass/skip/fail/timeout counts;
7. Ruff format/lint checks and mypy according to `pyproject.toml`;
8. package build, install into a clean environment, import, CLI smoke test, and examples.

Do not require historical count `179` from a changed suite. Preserve `RES-BB-QA-003` as its
historical execution and create a new QA record for a new run.

### 4. Statistical and Numerical Checks

- Re-run the 13 supplied finite-case/numerical checks or their adopted equivalents.
- Verify stable log-domain aggregation and no overflow-induced ranking change.
- Verify approximation records tighten under nested tolerances or return explicit failures.
- Verify corrected experiments follow registered repetitions, uncertainty, split, tuning,
  metric, support, and abort rules.
- Verify null, reversed, failed, and unresolved outcomes are retained.

### 5. Document and Artifact Builds

- Run handoff synchronization, bibliography, presentation, skeleton/status, source-content,
  and Phase 6 validators.
- Clean-build book, two-column paper, single-column paper, and executive summary.
- At initial source adoption, compare with the released 168/35/42/12-page artifacts and
  checksums or approved pixel-equivalent baselines.
- After approved corrected results change content, establish new versioned build baselines;
  never rewrite Phase 6 release checksums.
- Render every page for visual QA and inspect contact sheets plus pages affected by changes.
- Validate figure/table source hashes, status metadata, caption anchors, and links.
- Build MkDocs/site output and check internal links without treating generated `site/` as
  authoritative source.

### 6. Result and Provenance Audit

- Compare all historical populated result assets with pre-adoption and Phase 4R hashes.
- Validate every corrected result ID, unique parent link, and data/config/code/environment
  hash.
- Verify coordinate axis, observation family, prior, split, support/extrapolation policy,
  metric version, tuning budget, seeds, commands, and environment are recorded where relevant.
- Verify generated figures/tables reference the corrected ID while preserving parent history.
- Reject release of any corrected result with missing lineage or hashes (`FAIL-BB-006`).

### 7. Clean Reproduction

In a fresh temporary checkout or exported source tree:

1. create the documented environment;
2. install the package and test dependencies;
3. run required package tests and validators;
4. rebuild documents and generated handoffs;
5. reproduce declared lightweight artifacts;
6. compare hashes or document legitimate nondeterminism and tolerances;
7. confirm no author-local path or undeclared file is required.

Do not rerun expensive scientific experiments during clean reproduction unless separately
authorized; verify their immutable, hashed artifacts and reproducible command records.

## Release Decision

Classify each check as `pass`, `fail`, `unresolved`, `not applicable`, or `not run`. Include
command, environment, evidence path, and impact. Release is blocked by:

- changed historical numerical values;
- missing or invalid corrected-result lineage;
- failed scientific invariant, schema, package, focused regression, or document build;
- comparator axis mismatch or unsupported predictive behavior entering a conclusion;
- generated/canonical handoff drift;
- an unresolved item represented as complete.

An approved release may retain explicitly scoped nonblocking unresolved work only when it is
not required for a claimed conclusion and its status is visible.

## Gate E Promotion Checks

Before recommending merge:

1. Confirm the PR head contains the terminal Gate D commit or a newer fully revalidated head.
2. Confirm required PR checks and review approvals pass and unresolved limitations remain
  visible in the ledger/release notes.
3. Classify merge readiness separately from publication readiness.

Before tagging or publication:

1. Require explicit tag/publication approval after merge.
2. Confirm the merged default-branch version is exactly `2.0.0rc3` and no conflicting tag or
  PyPI version exists.
3. Confirm `v2.0.0rc3` will point to the approved merged commit. Never move an existing tag.
4. After push, verify the release workflow, trusted-publishing result, PyPI version, and
  published artifact metadata/hashes. Record failure states; do not retry by mutating history.

The current verified decision is review-ready, not published. Opening a PR is reversible;
merge and tag remain separate approval boundaries.

## Final Deliverables

1. Machine-readable validation summary and human-readable release report.
2. Completed adoption ledger and remaining-work registry.
3. New QA result record for this validation run.
4. Historical hash comparison and corrected-result lineage report.
5. Build, test, static-analysis, visual-QA, and clean-reproduction evidence.
6. Release notes separating manuscript corrections, implementation changes, new executions,
   excluded historical computations, and unresolved work.
7. A final `go`, `conditional go`, or `no-go` decision with reasons.
