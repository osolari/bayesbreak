# Phase 6 Adoption Ledger

Date: 2026-08-05

Phase: scientific report and release-source adoption

## Authority

1. `BayesBreak_Chatbot_Change_Guide.md`
2. `report/releases/phase6/BayesBreak_PHASE_6_AUTHOR_DECISIONS.md`
3. Canonical sources and registries under `report/`
4. Generated handoffs and the explicitly incomplete repository skeleton

The exact title is **Generalized Hierarchical Bayesian Segmentation with Irregular
Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**.

## Intake Integrity

- Outer package SHA-256:
  `9719c05f4a4eec4c9db5968d8ed251aae14adf277a356a1f7cec264888de8d79`
- Outer ZIP integrity: passed.
- All 22 entries in `BayesBreak_PHASE_6_DELIVERABLE_SHA256SUMS.txt`: passed.
- Nested repository, unified-source, and presentation ZIP integrity: passed.
- Unsafe paths: none.
- Symbolic links: none.
- Case collisions: one, documented below.

## Adopted Artifacts

| Source | Destination | Action | Status |
|---|---|---|---|
| Unified Overleaf project | `report/` | Added 319 normalized canonical files | Adopted |
| Outer release package | `report/releases/phase6/` | Added all 23 signed deliverables | Adopted |
| Uppercase annotation key list | `report/releases/phase6/source-collision/ANNOTATION_MANIFEST.json` | Preserved byte-for-byte at a noncolliding path | Adopted |
| Legacy manuscript | `docs/report/` | Removed all 84 superseded files after result verification | Adopted |
| Root and site metadata | `README.md`, `CITATION.cff`, `pyproject.toml`, `mkdocs.yml`, `docs/` | Merged Phase 6 title, scope, links, and result status | Adopted |
| Artifact generators | `scripts/`, `examples/`, `src/bayesbreak/experiments/synthetic.py` | Redirected new outputs to `results/` | Adopted |

The repository skeleton under `report/coding/repository_skeleton/` is an interface
blueprint. It was not copied over the functioning package implementation.

## Numerical Assets

- Canonical read-only assets checked: 53.
- Phase 4R hash mismatches after adoption: 0.
- Pre-adoption legacy assets present: 53.
- Fifty-one legacy files were byte-identical to the Phase 4R baseline.
- `realdata_metrics.json` and `realdata_metrics.txt` differed only in Unicode encoding
  and terminology. All JSON numeric paths and values were identical.
- Pre-adoption legacy result-manifest SHA-256:
  `ba5d02300737ab9a79247936c4ee1f03e16773e5fa866cc4cd96ff95de1fdb47`.

`RES-BB-CMP-002` remains excluded from comparator conclusions.
`RES-BB-RD-007Q` remains excluded from posterior-predictive conclusions.

## Validation Evidence

- Canonical source comparison: all 319 normalized files match the signed unified-source
  ZIP byte-for-byte.
- Release payload: all 22 checksum-manifest entries passed; all 23 outer deliverables are
  retained under `report/releases/phase6/`.
- Canonical handoff synchronization: passed (16 tasks, 15 experiments, 14 claims,
  17 result records, six failure states, seven stages).
- Repository-skeleton checks: passed; two tests passed; scientific implementation
  remains explicitly incomplete.
- Presentation handoff checks: passed (15 files, two approved roadmaps, no slides).
- Independent mathematical checks: 13 of 13 passed.
- Phase 6 source checks independent of compiled output: passed.
- Signed PDFs: all four match the release SHA-256 manifest.
- Four immutable historical sidecars contain author-local `figure_path` values. Their
  hashes remain unchanged, and repository-relative consumer mappings are recorded in
  `LEGACY_ABSOLUTE_PATH_MIGRATIONS.json`.
- Strict MkDocs build: passed.
- Ruff checks for `src/`, `scripts/`, `examples/`, and `tests/`: passed.
- Full package test suite with `PYTHONPATH=src`: 176 passed, three skipped.
- Git whitespace validation: passed.
- Local source rebuild: not run because `latexmk`, pdfLaTeX, and Poppler tools are not
  installed. Tectonic cannot satisfy the source's explicit pdfLaTeX requirement.

## Phase 2 Package Convergence

Package baseline on 2026-08-05: 174 passed and five optional `ruptures` tests skipped
under Python 3.11.14. No baseline test failed.

| Source and SHA-256 | Destination and action | Authority and rationale | Task | Validation evidence | Status | Follow-up |
|---|---|---|---|---|---|---|
| `report/shared/handoffs/coding_agent_handoff.json` (`4a778128eb66cd2bce96497e2176ac8e9615d3efbf2e6faf11f97dcdb17c4f97`) | `src/bayesbreak/provenance.py`; `schemas/*.schema.json`; `tests/test_result_schema.py`; `docs/api.md` (add) | Canonical `CODE-BB-010`; version new result records, reject invalid release lineage and paths, and migrate declared legacy paths in memory without changing archived bytes | `CODE-BB-010` | Focused: `python -m pytest -q tests/test_result_schema.py` (14 passed). Ruff: touched Python files passed. All three schemas parsed as JSON. Production-module Pylance diagnostics: none. Strict MkDocs build passed with output under `/tmp`. Post-edit full suite: 188 passed, five optional `ruptures` skips, zero failed. | Validated | Integrate these records into new experiment outputs under later tasks; do not rewrite archived sidecars. |
| Canonical handoff above; `src/bayesbreak/priors.py` (`9639d147e356509ab0fe3be6f326c697f616820162a4e6ef9959ef7c8d2dc112`); `src/bayesbreak/design_prior.py` (`0d0ad49fbb5bca1b12d87d3eeb66a793a28b0a6f61ab11c6858c7be0bf0e5dda`) | `src/bayesbreak/priors.py`; `src/bayesbreak/design_prior.py`; `tests/test_priors.py`; `docs/api.md` (add) | Canonical `CODE-BB-001` and adopted `REV-BB-004`; represent complete-segment cohesion separately from interior-boundary hazard and use fixed-count Poisson occupancy odds `exp(Lambda_j)-1` | `CODE-BB-001` | Focused: `python -m pytest -q tests/test_priors.py` (nine passed). Relevant slice with existing design-prior regressions: 15 passed. Ruff and production-module Pylance diagnostics passed. Strict MkDocs build passed with output under `/tmp`. | Validated | Integrated into both inference semirings under `CODE-BB-002`; retain archived prior-sensitive values unchanged. |
| Canonical handoff above; `src/bayesbreak/base.py` (`40414ad217ece5e594cd7aa4487ba161e00f97d668f9f3eebda3735a0478a0a3`); `tests/test_dp_priors.py` (`97bb4cc81effd71efdde34aa5e767d28091e0ee3d17c7d7b1d760161e8a417a8`) | `src/bayesbreak/base.py`; all observation-family constructors; `src/bayesbreak/design_prior.py`; `tests/test_dp_priors.py`; `tests/test_api.py`; `docs/api.md` (merge) | Canonical `CODE-BB-002` and adopted `REV-BB-005`; route one local cohesion-plus-interior-hazard table through prior normalization, sum-product, max-sum, diagnostics, and Bayes curves while retaining `length_prior` compatibility | `CODE-BB-002` | Focused randomized suite: five tests with 1,000 seeded exhaustive finite cases passed. Complete prior/DP/API slice: 67 passed. All eight families store the config unchanged; seven clone with it. Ruff and editor diagnostics passed. Post-edit full suite: 202 passed, five optional `ruptures` skips, zero failed. Strict MkDocs build passed under `/tmp`; git whitespace validation passed. | Validated | Pre-existing logistic-normal sklearn cloning remains blocked because its constructor modifies `approx`, not because of `partition_prior`. |
| Canonical handoff above; `src/bayesbreak/replicates.py` (`4f9e32ffc2d502840786564b12ead562a683e567318754c2d56edcbea184f3f7`); `src/bayesbreak/groups.py` (`37c729331a761b78a721bbdc469f18d468092408ca05efd7d900699a8903c1b8`) | `src/bayesbreak/replicates.py`; `src/bayesbreak/groups.py`; `tests/test_replicates_stability.py`; `docs/api.md` (merge) | Canonical `CODE-BB-004` and adopted `REV-BB-006`/`REV-BB-012`; accurately sum aligned finite log evidence, preserve support intersections, reject invalid scores, and avoid exponentiating unbounded pooled evidence | `CODE-BB-004` | Focused adversarial suite: six passed. Relevant replicate/grouped slice: 17 passed. Ruff and editor diagnostics passed; unstable exponentiation pattern absent. Post-edit full suite: 208 passed, five optional `ruptures` skips, zero failed. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | `block_posterior_mean_` replaces the former numerically unsafe pseudo-pooled `block_first_moment_` diagnostic. |
| Canonical handoff above; `src/bayesbreak/mixture.py` (`bff7847c1a8a2616174a57405730b5d2bd746a6067ec4ac616f1c4608b93e022`) | `src/bayesbreak/mixture.py`; `tests/test_mixture_restarts.py`; `docs/api.md` (merge) | Canonical `CODE-BB-003` and adopted `REV-BB-007`; return the final finite-template objective on every successful exit, reject invalid restart traces, rank by final objective, and retain deterministic tie/label ordering | `CODE-BB-003` | Focused exit/restart suite: ten passed. Complete mixture/diagnostics slice before final two cases: 26 passed. Ruff and editor diagnostics passed. Post-edit full suite: 218 passed, five optional `ruptures` skips, zero failed. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | The archived latent-group result remains unchanged and requires a separately approved rerun. |

## Remaining Gates

- Package and interface convergence remains incomplete for all tasks not marked validated
  in the Phase 2 ledger.
- The latent-group, array-CGH, and methylation corrected reruns have not been executed.
- The routine-specific nonconjugate rate proof obligation and bounded EP timeout remain
  unresolved as recorded by Phase 6.
- Journal venue, permanent repository, data-release locations, and independent external
  changepoint annotations remain unset.
