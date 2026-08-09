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
| Environment lock (`c6ec18e35f980334b10680dcf646b90ed01dfd41144e328fb537757d4d48abd6`); test manifest (`7fa47e3bf3c273e2652eb58fe71b0bc5141eaed75cf14d0591ec444938cec9b7`) | `provenance/environment-lock.json`; `provenance/test-manifest.json`; bounded-test runner and integrity tests (add) | Canonical `CODE-BB-012` and adopted `REV-BB-012`; report pass/skip/fail/unresolved separately and bind evidence to a recorded environment | `CODE-BB-012` | Version-checkpoint suite: 298 collected, 293 passed, five optional `ruptures` skips, zero failed, 74% coverage, with explicit-support CV warnings recorded. Current EP successor nodes passed the 20-second cap at 6.26s and 5.18s. Runner/manifest integrity tests passed; Ruff/editor/JSON checks passed. | Validated | Historical `tests/test_families.py::test_ep_logistic_normal` is unresolved because the node no longer collects; `RES-BB-QA-003` is not rewritten. MkDocs is a separate Python 3.12 profile. |
| Author decision on 2026-08-06; canonical version source (`49a98a0fe94456c35286c86f09b0558fc78ebd2d281c7e7b672830789dffcdd6`); PyPI workflow (`ea9d04a07695b42125b22fa31cc041f1b6d7c481b82422446a030b9e6d363af3`) | Static `src/bayesbreak/_version.py`; dynamic setuptools attribute; Conda build validation; guarded PyPI publication; citation/changelog/lineage docs (merge) | Canonical `CODE-BB-013` and adopted `REV-BB-012`; the author selected PEP 440 version `2.0.0rc3` and PyPI as the sole publication target | `CODE-BB-013` | Four focused lineage checks passed: runtime import, citation/lineage, workflow guards, and built wheel metadata all report `2.0.0rc3`. Ruff and mypy passed; strict MkDocs passed under `/tmp`; full suite passed with five optional skips and declared CV warnings; whitespace passed. | Validated | No tag or PyPI publication was created. Publication remains gated on explicit `v2.0.0rc3`, trusted publishing, and successful workflow checks. Conda remains build validation only. Journal venue and permanent data release remain unset. |
| Phase 4R hash baseline; asset manifest (`569dc00d26c72e01b02c4515301a489002e262c9eb82c4d501fd6e801eddd753`); checker (`8a652ff97bcf3921bb0c3cb2ba065c3a177d7d65d968d7f62850e1902b7b677d`) | `provenance/archived-asset-manifest.json`; `scripts/check_asset_semantics.py`; focused tests (add) | Canonical `CODE-BB-015` and adopted `REV-BB-001`/`REV-BB-010`/`REV-BB-011`; validate archived source hashes, interpretation links, caption anchors, marker roles, exclusions, and visual QA without mutating numerical assets | `CODE-BB-015` | Read-only checker and three focused tests passed: 53 assets, zero missing/hash/semantic mismatches, both exclusion checks true, marker semantics true, signed visual QA pass. Ruff/editor checks passed. | Blocked | Historical-asset validation is complete. Regenerating corrected figures/tables requires separately approved experiments, new result IDs/parent links, and renewed visual QA; no such rerun was authorized. |

### Gate C Terminal State

- Validated tasks: `CODE-BB-001` through `CODE-BB-014` except `CODE-BB-015`,
  plus `CODE-BB-016`.
- Blocked task: `CODE-BB-015` pending separately approved corrected experiments and renewed
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

- Gate D implementation, corrected reruns, and integrated validation are complete. Gate E
  review PR #8 merged to `main` at commit
  `9ef9c554014d3957a831460c3de2aa5149931a88`; `main` is now the default and active
  development branch. Release promotion still awaits explicit approval; no `v2.0.0rc3`
  tag or PyPI publication has been performed.
- Nonblocking scientific limitations remain explicit: routine-specific nonconjugate rate
  proofs are not established; Beta-observation PIT calibration is unavailable; independent
  external changepoint annotations are unavailable; and the removed historical EP node
  remains unresolved lineage even though its current successor tests pass.
- Default sklearn cross-validation still produces nonfinite fold scores when validation
  coordinates exceed fitted support unless a named extrapolation policy or model is selected.
- Journal venue, permanent repository, and permanent data-release locations remain unset.

## Gate D Corrected Reruns

| Result | Parent | Protocol | Execution and evidence | Interpretation | Status |
|---|---|---|---|---|---|
| `RES-BB-SYN-005` | `RES-BB-SYN-002` | `EPR-BB-005` | Pilot: 0.19 s, 182 MB peak RSS, projected 75 s. Full run: 400 seeded fits across eight predeclared cells in 84.4 s, 293 MB peak RSS. All 400 traces monotone; every final objective equals `trace[-1]`; all 1,200 restarts valid; 400 unique dataset hashes. Sidecar and all artifact hashes validate; 53 historical asset hashes remain unchanged; canonical registry sync passes with 18 results. | Archived-design mean hard accuracy 0.9742 (95% interval 0.9536 to 0.9948), mean ARI 0.9183. Low-separation and duplicate-template cells show expected non-identifiability/collapse. Valid for the stated finite score-clustering simulations with limitations; not normalized-mixture identifiability. | Validated |
| `RES-BB-CMP-003` | `RES-BB-CMP-002` | `EPR-BB-010`; `EPR-BB-013` | Final pilot: 4.01 s, 222 MB peak RSS, projected 97.8 min. Full raw 2,215-probe by 43-subject run: 785.1 s, 2.27 GB peak RSS. Source SHA-256 `b82da97ffe6b5c431a60c3f811ee5c339708562126ed4a3d0b0344f2f2e09a63`; parsed matrix SHA-256 `1551547d50564227ac020c196e56cae0b29c76ccf76eb4140d47628019cdb9a8`. Shared and independent BayesBreak fits reproduce archived values; comparator axes, sidecar, artifact hashes, 54 focused post-run tests, and all 53 historical asset hashes validate. | Common-axis agreement F1 at tolerance 3: PELT 0.8000, Dynp 0.9286, binary segmentation 0.7143, WBS 0.7857. Dynp, binary segmentation, and WBS returned the exact 14-boundary target; the predeclared PELT grid returned the closest candidate with 11 boundaries and was not retuned. Valid for model-derived MAP agreement with limitations; not external biological accuracy or predictive superiority. | Validated |
| `RES-BB-RD-008Q` | `RES-BB-RD-007Q` | `EPR-BB-012` | Pilot: one 152-CpG block in 5.87 s, 368 MB peak RSS, projected 32.3 s. Full run: ten disjoint stratified interior blocks, 1,520 held-out CpGs, eleven total fits in 32.84 s, 376 MB peak RSS. Source SHA-256 `f823f0eebd6ec44994c28882c1b7d16ea21eaf32ee49c93a1a149c5096b5b54e`; split SHA-256 `76c7f52af33b45e9d9385fe368bfe0d621e93bdde89ae4a9cfe494918e8a1941`. Every score is finite; coverage is positive `phi_new`; all predictions record `extrapolation=error`; the full-data fit reproduces the archived segmentation; sidecar/artifact hashes and 53 historical assets validate. | Total family-correct log predictive score `-23605.6749` on denominator 1,520; pooled mean `-15.5300`; split-mean 95% t interval `[-23.1445, -7.9156]`; mean boundary-stability F1@3 `0.8786`. Valid for the declared in-support chromosome-block evaluation with limitations. Blocks are not independent biological samples; PIT calibration and external accuracy are unavailable; the score is not comparable to the excluded parent because family and split changed. | Validated |

The parent result remains immutable. New artifacts are under
`results/phase6/RES-BB-SYN-005/` and were generated from clean code commit
`734ea3b241f0c0ae0ecbc30ad2ae144a2a2f3750`.

### Gate D Terminal State

- Corrected results validated: `RES-BB-SYN-005`, `RES-BB-CMP-003`, and
  `RES-BB-RD-008Q`; each has a distinct parent link, execution/configuration/data/code/
  environment hashes, versioned artifacts, and a current result sidecar.
- Final package suite at commit `28a04c209614911bb06dd0551d73190df97aa871`:
  324 collected, 324 passed, zero skipped, zero failed, 76 percent package coverage.
  The ten warnings are the declared sklearn cross-validation support-policy behavior.
- Repository checks: all-file pinned pre-commit hooks passed; Ruff lint passed; the
  repository-pinned Ruff 0.4.10 formatter passed; mypy passed 53 package source files;
  strict MkDocs passed under the separate Python 3.12 profile with output under `/tmp`.
- Release validation: sdist and wheel built under `/tmp`; both passed Twine; wheel
  metadata reports `2.0.0rc3`; PyPI tag/workflow guards passed. No tag or publication
  was created.
- Integrity checks: canonical handoff synchronization passed with 20 unique results;
  all 38 bibliography annotations passed; all 53 historical asset hashes and semantics
  passed; all corrected sidecar/artifact hashes passed; `RES-BB-CMP-002` and
  `RES-BB-RD-007Q` remain unchanged and excluded from their original conclusions.
- The Gate C environment lock remains a preserved historical snapshot. Gate D added
  `ruptures==1.1.10` and `rdata==1.1.0` for the authorized executions; each corrected
  sidecar records its execution-specific environment hash.
