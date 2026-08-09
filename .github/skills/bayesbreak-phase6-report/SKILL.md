---
name: bayesbreak-phase6-report
description: 'Adopt and validate the authoritative BayesBreak Phase 6 technical book, journal paper, executive summary, registries, and scientific narrative. Use after explicit approval for the report/source phase of the Phase 6 adoption.'
argument-hint: '[plan|execute]'
user-invocable: true
disable-model-invocation: true
---

# BayesBreak Phase 6 Report Adoption

Import the final manuscript sources without weakening scientific status, changing archived
numbers, or creating competing canonical report trees.

## Preconditions

1. Read and follow [the adoption orchestrator](../bayesbreak-phase6-adoption/SKILL.md).
2. Require explicit approval for the scientific source phase.
3. Confirm Phase 0 archive and checksum checks passed.
4. Inspect local changes under `report/`, `docs/`, `README.md`, `CHANGELOG.md`,
   `CITATION.cff`, `mkdocs.yml`, and `pyproject.toml`; never overwrite uncommitted user work.
5. Record current populated result-asset hashes and current report build behavior.

## Canonical Inputs

Use `BayesBreak_Phase6_Unified_Overleaf_Project.zip` as the canonical editable source. Use
the outer package PDFs and Markdown reports as signed release artifacts and verification
evidence, not as substitutes for editable source.

Authoritative inner paths include:

- `shared/metadata.tex`;
- `book/`, `paper/`, and `executive/`;
- `shared/bibliography/` and `shared/figures/tikz/`;
- `shared/handoffs/coding_agent_handoff.json`;
- `shared/metadata/claim_traceability.json`;
- `shared/metadata/experiment_protocols.json`;
- `shared/metadata/result_interpretation.json`;
- `presentation_handoffs/`;
- `revision_artifacts/phase6/` and release validators.

Treat `shared/figures/results/` and `shared/tables/results/` as read-only archived results.

## Adoption Procedure

### 1. Stage and Audit

1. Extract the unified source into a temporary directory outside the repository.
2. Reject unsafe archive members and verify the clean-source audit before copying anything.
3. Run the supplied synchronization, skeleton, bibliography, presentation, and Phase 6
   validators in staging.
4. Build all four targets in staging and compare their SHA-256 values with the release
   manifest: 168-page book, 35-page two-column paper, 42-page single-column paper, and
   12-page executive summary.
5. Produce a path-level comparison with the current `report/` tree.

### 2. Establish One Canonical Report Tree

1. Adopt the staged unified project as the canonical content of `report/`.
2. Do not keep two active manuscript source trees. Use git history for superseded sources.
3. Before removing a current report file, classify it as superseded, migrated, generated,
   or uniquely retained in the adoption ledger.
4. Migrate uniquely retained local material only when its role is still valid and it does
   not conflict with canonical Phase 6 content.
5. Keep compiled release PDFs in a clearly versioned release-artifact location; do not
   confuse them with generated working files.
6. Exclude LaTeX auxiliaries, caches, rendered page images, and local absolute paths.

### 3. Adopt Scientific Metadata and Narrative

1. Apply the exact title and generalized hierarchical segmentation description to package,
   README, documentation, citation, site, and report entry points.
2. Replace narrow unqualified claims such as "Exact Bayesian segmentation" with language
   that qualifies exact conjugate and approximate nonconjugate regimes.
3. Preserve the distinction between posterior boundary marginals and the joint MAP
   partition in every affected document.
4. Preserve all 14 scientific result records: 12 usable with stated limitations and two
   excluded for their intended conclusions.
5. State that no archived numerical result changed during manuscript revision.
6. Keep the unresolved routine-specific nonconjugate rate as a proof obligation.
7. Keep the Phase 6 EP test as one unresolved timeout, not a failure or pass.

### 4. Adopt Canonical Registries and Generated Views

1. Install the canonical JSON handoff and metadata registries in stable report/provenance
   locations.
2. Generate Markdown and TeX handoff views from the canonical JSON; do not hand-edit
   generated copies.
3. Adopt the implementation task registry, experiment protocols, claim traceability,
   result interpretation, and synchronization manifest together.
4. Preserve all IDs and reject orphaned or duplicate claims, results, tasks, protocols, and
   bibliography annotations.
5. Adopt presentation handoffs only as source constraints. Do not generate slides.

### 5. Reconcile Repository Documentation

Update only statements affected by Phase 6 in:

- `README.md`, `CHANGELOG.md`, and `CITATION.cff`;
- `docs/` scientific, report, result, and reproducibility pages;
- `pyproject.toml` description, keywords, and report links;
- report build instructions and contribution guidance;
- generated site content only through the documented site build.

Do not rewrite unrelated documentation or silently publish unset journal, repository, or
data-release locations.

## Focused Validation

Run the first executable check immediately after the first report-source edit. Prefer the
supplied synchronization or Phase 6 validator for the touched source.

Then require:

1. all supplied Phase 6 source validators pass;
2. all four documents build from a clean directory;
3. initial adopted builds match released checksums or documented pixel-equivalent baselines;
4. title and narrative checks pass across book, paper, executive summary, handoffs, and
   presentation constraints;
5. bibliography has 38 unique keys, 38 annotations, and 38 manifest entries;
6. archived result assets match the pre-adoption and Phase 4R hashes;
7. handoff views and registries are synchronized;
8. no slides, author-local paths, build products, or cache files enter distributable source;
9. links from root documentation resolve to the adopted report and release artifacts.

## Stop Conditions

Stop this phase and request a decision if:

- canonical and author-provided sources conflict;
- local uncommitted report work would be overwritten;
- any archived numerical asset hash changes;
- a generated view cannot be reproduced from canonical data;
- a released document cannot be rebuilt or materially differs from its verified artifact;
- adoption would require inventing a journal venue or release location.

## Phase Output

Provide the report-tree diff, migration/removal ledger, build and validator evidence,
archived-result hash comparison, and remaining incomplete items. Obtain Gate B approval
before package implementation.
