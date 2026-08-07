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
| Canonical handoff above; `src/bayesbreak/replicates.py` (`43c303ee7c59cd37f00f9d2651e6a9e91396470cfcdd496ab25abe995819a259`); `src/bayesbreak/groups.py` (`37c729331a761b78a721bbdc469f18d468092408ca05efd7d900699a8903c1b8`) | `src/bayesbreak/replicates.py`; `src/bayesbreak/groups.py`; `tests/test_replicates_stability.py`; `docs/api.md` (merge) | Canonical `CODE-BB-004` and adopted `REV-BB-006`/`REV-BB-012`; accurately sum aligned finite log evidence, preserve support intersections, reject invalid scores, and avoid exponentiating unbounded pooled evidence | `CODE-BB-004` | Focused adversarial suite: six passed. Relevant replicate/grouped slice: 17 passed. Ruff and editor diagnostics passed; unstable exponentiation pattern absent. Post-edit full suite: 208 passed, five optional `ruptures` skips, zero failed. Strict MkDocs passed under `/tmp`; whitespace passed. Final Gate C mypy cleanup preserved behavior. | Validated | `block_posterior_mean_` replaces the former numerically unsafe pseudo-pooled `block_first_moment_` diagnostic. |
| Canonical handoff above; `src/bayesbreak/mixture.py` (`bff7847c1a8a2616174a57405730b5d2bd746a6067ec4ac616f1c4608b93e022`) | `src/bayesbreak/mixture.py`; `tests/test_mixture_restarts.py`; `docs/api.md` (merge) | Canonical `CODE-BB-003` and adopted `REV-BB-007`; return the final finite-template objective on every successful exit, reject invalid restart traces, rank by final objective, and retain deterministic tie/label ordering | `CODE-BB-003` | Focused exit/restart suite: ten passed. Complete mixture/diagnostics slice before final two cases: 26 passed. Ruff and editor diagnostics passed. Post-edit full suite: 218 passed, five optional `ruptures` skips, zero failed. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | The archived latent-group result remains unchanged and requires a separately approved rerun. |
| Canonical handoff above; `src/bayesbreak/nonconjugate/error_bounds.py` (`c6c123509639c8f5c01a1a9e76f4a91ca03dd5909d91f3bedc6e738129f9bcf9`); `schemas/approximation_error_record.schema.json` (`b7c641856b83a9f3ce3a460c919f26d2d1142928d83ae82972a9801249a230ca`) | `src/bayesbreak/nonconjugate/`; `src/bayesbreak/diagnostics.py`; approximation-error schema; focused tests; current docs (add/merge) | Canonical `CODE-BB-005` and adopted `REV-BB-008`; compare identical reachable block coordinates, retain separate residual/tail/quadrature fields, return explicit failure states, and propagate only a conditional maximum-error bound capped at TV one | `CODE-BB-005` | Focused record/diagnostics/stability slice: 16 passed. Ruff, editor diagnostics, and schema parsing passed. Post-edit full suite: 224 passed, five optional `ruptures` skips, zero failed. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | Routine-wide convergence rates remain an unresolved proof obligation and are no longer inferred from method names. |
| Canonical handoff above; `src/bayesbreak/families/beta_obs.py` (`32727a9ece5e24ec8aca677e94dd899fa3e2ad2eab2e6cdd81a90c4d44f3d343`) | `src/bayesbreak/families/beta_obs.py`; `tests/test_beta_obs_prediction.py`; current API/model docs (merge) | Canonical `CODE-BB-006` and adopted `REV-BB-009`; integrate the new-observation Beta density over the fitted segment posterior, keep precision separate from training power weights, enforce open support, and prohibit the Gaussian fallback | `CODE-BB-006` | Focused reference/normalization/support suite: six passed. Relevant Beta/family/prediction slice before the precise cleanup: 40 passed. Ruff and editor diagnostics passed after cleanup. Post-edit full suite: 230 passed, five optional `ruptures` skips, zero failed. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | `RES-BB-RD-007Q` remains unchanged and excluded; no corrected methylation run is authorized. |
| Canonical handoff above; `src/bayesbreak/prediction.py` (`ee24c8a76e392a081e2d7ff1e69ac5cfac9ed5d5b92803431f78a8f862356476`) | Core and wrapper prediction APIs; `tests/test_prediction_support.py`; API docs (merge) | Canonical `CODE-BB-007` and adopted `REV-BB-009`; reject unsupported coordinates by default, make endpoint policies explicit, preserve in-range assignment, and record policy/support metadata | `CODE-BB-007` | Focused policy/wrapper suite: ten passed. Complete API/prediction/PIT/replicate/multivariate/sliding slice: 69 passed. Ruff and editor diagnostics passed. Full suite passed with five optional `ruptures` skips; sklearn grid-search folds emitted expected nonfinite-score warnings where test coordinates exceeded fold support. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | Historical endpoint-clipped outputs remain unchanged. CV workflows must explicitly choose and record an extrapolation model or policy rather than relying on silent clipping. |
| Canonical handoff above; `src/bayesbreak/metrics.py` (`026e739204e880a0ef538abee4a7164a7388702dafcd5035af4218e39a692835`); boundary schema (`a36c6d7a8071fe7a1a7a1e08fe28be0ddec224ff7b982d5b0665a0d8683f5b36`) | `src/bayesbreak/metrics.py`; `tests/test_boundary_matching.py`; cached baseline table script; API docs (add) | Canonical `CODE-BB-008` and adopted `REV-BB-009`/`REV-BB-010`; maximize match cardinality before minimizing total distance, prevent reuse, return NA MAE for zero matches, and record axes/reference type/version | `CODE-BB-008` | Focused hand/tie/duplicate/empty/symmetry suite: ten passed. Metric plus baseline regressions: 22 passed and five optional skips. Ruff and editor diagnostics passed. Full suite passed with five optional skips and the recorded CV support warnings. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | Remaining figure/table-local greedy metrics are handled during semantic asset regeneration, not by rewriting archived results. |
| Canonical handoff above; `src/bayesbreak/comparators.py` (`2a54c0184fb75affc8a9a50e5ba104e45ce5036b7fcdbc96a42e73dafb5ebbf1`); `scripts/run_comparators.py` (`298dc9621cb8e8f6d5fce31d11f0b570f4d5ab57ecb0932b66015fab019f3ecc`) | Comparator schema/API; raw-request CLI; cached baseline CGH rejection; focused tests; API docs (add/merge) | Canonical `CODE-BB-009` and adopted `REV-BB-010`; require raw unflattened multisequence matrices, matching coordinate axes, explicit task/tuning strata, and deterministic pre-dispatch rejection of cached fitted traces | `CODE-BB-009` | Focused accepted/rejected/schema/script suite: ten passed. Complete comparator/metric/baseline slice: 32 passed and five optional skips. Ruff and editor diagnostics passed. Full suite passed with five optional skips and recorded CV support warnings. Strict MkDocs passed under `/tmp`; whitespace passed. | Validated | `RES-BB-CMP-002` remains unchanged and excluded; no corrected CGH comparator was run. |
| Canonical handoff above; `src/bayesbreak/datasets/base.py` (`46189ed043e8dbc08d07b337df697763cf05ee21f4c9cd3a4c9f758df9f048ca`); data-card index (`d2784477732c728130c7381d03d5b7af565c3d7d0979cc58d6ec34441ca07810`) | Provenance-aware loader API; four static data cards; MkDocs navigation; offline tests (add/merge) | Canonical `CODE-BB-011` and adopted `REV-BB-011`; synchronize post-stride dimensions/hashes, source and family declarations, descriptor roles, coordinate conventions, and external-annotation status | `CODE-BB-011` | Focused offline card suite: ten passed. Complete card/loader slice: 22 passed. Ruff, strict MkDocs under `/tmp`, and whitespace checks passed. Full suite passed with five optional skips and recorded CV support warnings. | Validated | Source dates remain explicitly unrecorded where not established; no external annotations or corrected real-data execution were invented. |
| Canonical JSON (`4a778128eb66cd2bce96497e2176ac8e9615d3efbf2e6faf11f97dcdb17c4f97`); renderer (`26723b180386f60d8a483a5d504926beb1ef74706048f0804505fd275e7080cb`) | Generated Markdown, detailed TeX appendix, task-registry TeX, and sync manifest (retain synchronized views) | Canonical `CODE-BB-014` and adopted `REV-BB-002`/`REV-BB-012`; preserve stable IDs and one canonical handoff source | `CODE-BB-014` | `report/scripts/check_sync.py` passed: 16 tasks, 15 experiments, 14 claims, 17 results, six failure states, seven stages. Skeleton check passed with matching registries and explicit incomplete status. | Validated | Generated views remain owned by the canonical JSON renderer; no manual copy was edited. |
| Canonical bibliography manifest and annotations; checker (`8b08eefa02ddcc962d4bb696e444440e22793c02925762ccc5f80bde05d725ff`) | `scripts/check_annotations.py`; `tests/test_annotations.py` (add) | Canonical `CODE-BB-016`, `APP-BB-A`, and adopted `REV-BB-012`; enforce one annotation per key, manifest hashes, project relationships, and verification status | `CODE-BB-016` | Focused API/CLI tests: two passed. Standalone checker passed with 38 bibliography keys, 38 manifest entries, and 38 annotation files; every duplicate/missing/orphan/hash/relationship/verification list was empty. Ruff and editor diagnostics passed. | Validated | Full Phase 6 PDF validator was not rerun because local compiled targets/toolchain are unavailable; signed validation records remain unchanged. |
| Environment lock (`c6ec18e35f980334b10680dcf646b90ed01dfd41144e328fb537757d4d48abd6`); test manifest (`0fb2b2f26e4f336f9e0b60f36338d5b0df065d304caa414422e91043d1e6f1ba`) | `provenance/environment-lock.json`; `provenance/test-manifest.json`; bounded-test runner and integrity tests (add) | Canonical `CODE-BB-012` and adopted `REV-BB-012`; report pass/skip/fail/unresolved separately and bind evidence to a recorded environment | `CODE-BB-012` | Final Gate C suite: 294 collected, 289 passed, five optional `ruptures` skips, zero failed, 74% coverage, with explicit-support CV warnings recorded. Current EP successor nodes passed the 20-second cap at 6.26s and 5.18s. Runner/manifest integrity tests passed; Ruff/editor/JSON checks passed. | Validated | Historical `tests/test_families.py::test_ep_logistic_normal` is unresolved because the node no longer collects; `RES-BB-QA-003` is not rewritten. MkDocs is a separate Python 3.12 profile. |
| Git/version/CI audit on 2026-08-05 | Retain current version and publication files pending author decision | Canonical `CODE-BB-013` and adopted `REV-BB-012`; one version source and one release target are required, but Phase 6 forbids inventing either | `CODE-BB-013` | `git describe` returned `aba2464-dirty`; imported/tracked `_version.py` reported `1.0.0.dev50+g8a548bcfe.d20260508`; changelogs declare unreleased `2.0.0-rc3`; setuptools-scm fallback is `1.0.0.dev0`; PyPI and Conda publication workflows both exist. | Blocked | Author must select the canonical version/release target and decide whether the generated tracked `_version.py` remains. Journal venue and permanent release locations are also unset. |
| Phase 4R hash baseline; asset manifest (`569dc00d26c72e01b02c4515301a489002e262c9eb82c4d501fd6e801eddd753`); checker (`8a652ff97bcf3921bb0c3cb2ba065c3a177d7d65d968d7f62850e1902b7b677d`) | `provenance/archived-asset-manifest.json`; `scripts/check_asset_semantics.py`; focused tests (add) | Canonical `CODE-BB-015` and adopted `REV-BB-001`/`REV-BB-010`/`REV-BB-011`; validate archived source hashes, interpretation links, caption anchors, marker roles, exclusions, and visual QA without mutating numerical assets | `CODE-BB-015` | Read-only checker and three focused tests passed: 53 assets, zero missing/hash/semantic mismatches, both exclusion checks true, marker semantics true, signed visual QA pass. Ruff/editor checks passed. | Blocked | Historical-asset validation is complete. Regenerating corrected figures/tables requires separately approved experiments, new result IDs/parent links, and renewed visual QA; no such rerun was authorized. |

### Gate C Terminal State

- Validated tasks: `CODE-BB-001` through `CODE-BB-012`, `CODE-BB-014`, and
  `CODE-BB-016`.
- Blocked tasks: `CODE-BB-013` pending an author version/release-target decision;
  `CODE-BB-015` pending separately approved corrected experiments and renewed
  visual QA. Its read-only historical-asset validation is complete.
- Final package suite: 294 collected, 289 passed, five optional `ruptures`
  skips, zero failed, and 74 percent package coverage.
- Static/package checks: Ruff passed for `src`, `scripts`, `tests`, and
  `examples`; mypy passed all 53 package source files; strict MkDocs passed with
  generated output under `/tmp`; git whitespace validation passed.
- Registry/asset checks: handoff synchronization, skeleton synchronization,
  bibliography annotations, and all 53 archived asset hashes/semantics passed.
- Explicit incomplete states: the historical EP node no longer collects and
  remains unresolved lineage; default sklearn CV warns and yields nonfinite
  fold scores when validation coordinates exceed training support unless a
  named extrapolation policy is selected; package version/release target and
  permanent publication locations remain unset; local PDF rebuild was not run
  because the required TeX/Poppler toolchain is unavailable.
- Corrected scientific reruns executed: none. `RES-BB-CMP-002` and
  `RES-BB-RD-007Q` remain immutable and excluded from their stated conclusions.

## Remaining Gates

- Package and interface convergence remains incomplete for all tasks not marked validated
  in the Phase 2 ledger.
- The latent-group, array-CGH, and methylation corrected reruns have not been executed.
- The routine-specific nonconjugate rate proof obligation and bounded EP timeout remain
  unresolved as recorded by Phase 6.
- Journal venue, permanent repository, data-release locations, and independent external
  changepoint annotations remain unset.
