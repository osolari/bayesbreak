# BayesBreak Phase 6 final manifest

## Authoritative scientific sources

| Path | Role | Status |
|---|---|---|
| `shared/metadata.tex` | Exact title, author, keywords, and scientific description | authoritative |
| `book/main.tex` and `book/chapters/` | Canonical technical exposition | authoritative |
| `book/appendices/` | Annotated literature, implementation plan, coding handoff, experiment protocols, and traceability | authoritative views of shared sources |
| `paper/main.tex` | Shared scientific source for both journal layouts | authoritative |
| `paper/main-two-column.tex` | Primary journal layout | authoritative layout |
| `paper/main-single-column.tex` | Review layout | authoritative layout |
| `executive/main.tex` and `executive/sections/` | Executive summary derived from the book and paper | authoritative executive source |
| `shared/bibliography/references.bib` | Bibliographic metadata | authoritative |
| `shared/bibliography/annotated_entries/` | One project-specific annotation per book reference | authoritative |
| `shared/handoffs/coding_agent_handoff.json` | Canonical implementation specification | authoritative |
| `shared/metadata/claim_traceability.json` | Claim status and location map | authoritative |
| `shared/metadata/experiment_protocols.json` | Experiment protocol registry | authoritative |
| `shared/metadata/result_interpretation.json` | Result interpretation registry | authoritative |
| `shared/figures/tikz/` | Editable scientific and executive diagrams | authoritative |
| `shared/figures/results/` and `shared/tables/results/` | Archived executed-result assets | read-only |
| `presentation_handoffs/` | Technical and executive presentation-source constraints | authoritative communication handoff |

## Generated synchronized views

| Path | Generated from |
|---|---|
| `shared/handoffs/coding_agent_handoff.tex` | `shared/handoffs/coding_agent_handoff.json` |
| `shared/handoffs/implementation_task_registry.tex` | `shared/handoffs/coding_agent_handoff.json` |
| `shared/handoffs/SYNC_MANIFEST.json` | Canonical handoff and rendered views |

## Compiled targets

| Path | Pages |
|---|---:|
| `build/bayesbreak-technical-book.pdf` | 168 |
| `build/paper/bayesbreak-main-paper.pdf` | 35 |
| `build/paper-single/bayesbreak-main-paper-single.pdf` | 42 |
| `build/executive/bayesbreak-executive-summary.pdf` | 12 |

## Phase 6 verification sources

| Path | Role |
|---|---|
| `scripts/verify_phase6_math.py` | Independent finite-case and numerical checks |
| `scripts/check_sync.py` | Canonical handoff synchronization check |
| `scripts/check_presentation_handoffs.py` | Presentation-source and no-slides check |
| `revision_artifacts/phase6/INDEPENDENT_SCIENTIFIC_VERIFICATION.md` | Scientific verification and incomplete-item record |
| `revision_artifacts/phase6/FORMAL_NUMERICAL_VERIFICATION.json` | Machine-readable numerical verification |
| `revision_artifacts/phase6/BIBLIOGRAPHY_VERIFICATION.json` | Bibliography and annotation verification |
| `revision_artifacts/phase6/VISUAL_QA.json` | Full-document rendering and inspection record |
| `revision_artifacts/phase6/DISTRIBUTABLE_SOURCE_AUDIT.json` | Clean-source content check |
| `revision_artifacts/phase6/SOURCE_REBUILD_REGRESSION.md` | Empty-directory rebuild and rendered comparison |
| `revision_artifacts/phase6/BayesBreak_PHASE_6_VALIDATION.json` | Final machine-readable validation output |
| `revision_artifacts/phase5/PHASE4R_READONLY_HASHES.json` | Read-only numerical-asset baseline |

## Numerical-result rule

A populated archived numerical value is never overwritten. A corrected computation receives a new result identifier and an explicit parent-result link.
