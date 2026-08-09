---
name: bayesbreak-phase6-adoption
description: 'Plan, coordinate, review, or release the BayesBreak Phase 6 adoption. Use for adoption status, Gate E review PRs, merge approval, v2.0.0rc3 release gating, or historical Phase 0-4 workflow. Requires explicit approval before merge, tagging, publication, or rerunning scientific experiments.'
argument-hint: '[plan|status|review-pr|execute <phase>|release-status]'
user-invocable: true
disable-model-invocation: true
---

# BayesBreak Phase 6 Adoption

Coordinate adoption of the Phase 6 scientific release into the existing repository. This
skill is the entry point; use the specialized phase skills for implementation details.

## Invocation Contract

- `plan`: inspect and report only. Do not modify repository content.
- `status`: compare completed evidence with the gates below. Do not infer completion.
- `execute <phase>`: modify only the named phase after explicit user confirmation in the
  current conversation.
- `review-pr`: prepare or open a Gate E review PR from `phase6-adoption-gate-b`; this does
  not authorize merging, tagging, or publication.
- `release-status`: inspect merge, tag, workflow, and PyPI state without changing them.
- A general request to inspect, explain, or plan is not execution confirmation.
- Confirmation for one phase does not authorize later phases or scientific reruns.

## Current Verified State

- Gates A-D and integrated Phase 4 validation are complete on
  `phase6-adoption-gate-b`; use the adoption ledger and test manifest as evidence rather
  than repeating completed scientific runs.
- Corrected results are `RES-BB-SYN-005`, `RES-BB-CMP-003`, and
  `RES-BB-RD-008Q`. Their historical parents remain immutable, and the two invalid parent
  interpretations remain excluded.
- The canonical registry contains 20 unique results. The terminal package profile records
  324 collected and passed tests, all corrected artifact hashes, strict MkDocs, package
  build/Twine checks, and repository-pinned all-file hooks.
- The next gate is Gate E review and release promotion. No `v2.0.0rc3` tag or PyPI
  publication exists unless newer evidence proves otherwise.

## Authority Order

Resolve conflicts in this order:

1. The author's latest explicit instruction.
2. [BayesBreak_Chatbot_Change_Guide.md](../../../BayesBreak_Chatbot_Change_Guide.md).
3. `BayesBreak_PHASE_6_AUTHOR_DECISIONS.md` in the final release package.
4. Canonical sources inside `BayesBreak_Phase6_Unified_Overleaf_Project.zip`, especially
   `shared/handoffs/coding_agent_handoff.json` and `shared/metadata/*.json`.
5. Generated Markdown/TeX handoffs and the repository skeleton.
6. Existing repository prose and behavior.

Treat the repository skeleton as an interface and schema blueprint. It is explicitly
incomplete and must never replace functioning package code wholesale.

## Scientific Invariants

- Preserve the exact title: **Generalized Hierarchical Bayesian Segmentation with
  Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**.
- State that the methodology and research direction did not change.
- Present irregular designs, multi-sequence hierarchies, and known/latent groups as central.
- Distinguish exact conjugate inference from approximate numerical segment integration.
- Distinguish sum-product posterior quantities from max-sum joint-MAP recovery.
- Use established statistical, probabilistic, computational, and optimization terminology.
- Preserve every populated archived numerical value and its execution provenance.
- Keep `RES-BB-CMP-002` excluded from comparator conclusions.
- Keep `RES-BB-RD-007Q` excluded from posterior-predictive conclusions.
- Keep unresolved proofs, tests, computations, annotations, and release decisions explicit.
- Never call agreement with BayesBreak MAP boundaries independent ground-truth accuracy.

## Phase Sequence

### Phase 0: Read-Only Intake

1. Record `git status` without changing or reverting existing work.
2. Compute the outer archive SHA-256 and verify the supplied checksum manifest.
3. Inventory both nested archives and reject path traversal, absolute paths, symlinks, and
   case-colliding destinations before extraction.
4. Read the manifest, author decisions, status handoff, canonical coding handoff, claim map,
   experiment registry, result interpretation registry, and validation records.
5. Produce a collision and destination map for `report/`, `src/`, `tests/`, `scripts/`,
   `results/`, documentation, configuration, schemas, and provenance.
6. Record hashes for current populated result assets before any adoption edit.

**Gate A:** stop and report archive integrity, local modifications, collisions, proposed
deletions, and any conflict among authoritative sources. Obtain explicit approval for
Phase 1.

### Phase 1: Scientific Source Adoption

Invoke [bayesbreak-phase6-report](../bayesbreak-phase6-report/SKILL.md).

**Gate B:** require exact source rebuilds, synchronized registries, unchanged archived-result
hashes, and an approved diff before package changes.

### Phase 2: Package and Interface Convergence

Invoke [bayesbreak-phase6-code](../bayesbreak-phase6-code/SKILL.md).

**Gate C:** require focused tests for each completed `CODE-BB-*` task, full package checks,
and explicit remaining incomplete states before any corrected experiment is run.

### Phase 3: Corrected Scientific Reruns

Invoke [bayesbreak-phase6-reruns](../bayesbreak-phase6-reruns/SKILL.md).

**Gate D:** approve each experiment separately. A code-change approval is not permission to
spend compute or publish a replacement result.

### Phase 4: Integrated Release Validation

Invoke [bayesbreak-phase6-validation](../bayesbreak-phase6-validation/SKILL.md).

**Gate E:** release only when all blocking checks pass and every nonblocking unresolved item
is identified with status, evidence, owner or reason, and scope of impact.

### Phase 5: Gate E Review and Release Promotion

1. Open a review PR from `phase6-adoption-gate-b` to the protected default branch and attach
  the Gate D evidence; PR creation does not imply approval.
2. Obtain explicit author/reviewer approval for the merge and release candidate.
3. Merge only after required checks and approvals pass. Do not discard the adoption branch's
  result or provenance history.
4. Obtain separate explicit approval before creating or pushing `v2.0.0rc3`; pushing the tag
  triggers the PyPI release workflow.
5. Verify the tag matches `src/bayesbreak/_version.py`, the workflow succeeds, PyPI exposes
  `2.0.0rc3`, and published artifacts match the validated build metadata.
6. Record release URLs, workflow run, artifact hashes, and any failure without rewriting the
  historical Gate D evidence.

Merge approval is not tag approval. Tag approval is not permission to repair or rerun a
scientific result. Never retry publication by moving or replacing an existing release tag.

## Required Adoption Ledger

Maintain a version-controlled ledger during execution with one row per adopted artifact or
task. Record:

- source archive path and SHA-256;
- destination path and action (`add`, `merge`, `replace`, `retain`, or `exclude`);
- authority and rationale;
- implementation or result IDs affected;
- validation command and evidence path;
- status (`planned`, `approved`, `implemented`, `validated`, or `blocked`);
- unresolved conflict or follow-up.

Do not mark a row validated from file presence alone.

## Completion Report

Report phase-by-phase:

- adopted files and intentional removals;
- tests, builds, validators, and result-hash checks actually run;
- corrected results created, with parent links and hashes;
- historical results retained and excluded interpretations preserved;
- remaining proof obligation, EP timeout, missing external annotations, and unset release
  destinations unless later evidence resolves them;
- deviations from the canonical handoff, with author approval where required.
