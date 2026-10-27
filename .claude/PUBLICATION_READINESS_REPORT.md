# BayesBreak 2.1.0 — Publication Readiness Report

Package: `bayesbreak` | Version: `2.1.0` | Date: 2026-10-27
HEAD: `aedc7eadf883f466512c7ba59fce2fe118b4fe6c`

---

## 1. Package Metadata Audit

| Item | Status | Notes |
|---|---|---|
| Package name (`bayesbreak`) | PASS | PyPI name available; no prior public releases |
| Version (`2.1.0`) | PASS | `src/bayesbreak/_version.py` and `pyproject.toml` consistent |
| Author name | DECISION | `LICENSE` says "Omid Solari"; `pyproject.toml`/`CITATION.cff` say "Omid Shams Solari" — must resolve canonical form |
| Author email | PASS | `omid.shams@gmail.com` in `pyproject.toml` |
| License (`MIT`) | PASS | `LICENSE` file present; classifier declared |
| License classifier (PEP 639) | DECISION | Deprecated form in use (`License :: OSI Approved :: MIT License` + `license = { file = "LICENSE" }`); setuptools warns. Migrate to SPDX or defer — author decision |
| `Paper` URL (`pyproject.toml`) | PASS | Fixed from broken PDF link to `https://arxiv.org/abs/2603.14681` (pending commit) |
| `Homepage` URL | PASS | `https://osolari.github.io/bayesbreak` |
| `Repository` URL | PASS | `https://github.com/osolari/bayesbreak` |
| Python requires (`>=3.10`) | PASS | Declared in `pyproject.toml` |
| Python 3.11 classifier | PASS | Present |
| Python 3.13 classifier | DECISION | Not declared; tests run on 3.11 only. Add only after CI verification |
| `bayesbreak/experiments/` in wheel | DECISION | Included (serves `bayesbreak synthetic` / `bayesbreak realdata` CLI). Confirm intentional |
| `py.typed` declared | PASS | Declared in `pyproject.toml` under `[tool.setuptools.package-data]` |
| `py.typed` file present | PASS | Created in `src/bayesbreak/py.typed` (pending commit; was missing) |
| `CHANGELOG.md` has release entry | PASS | `## [2.1.0] — 2026-10-26 — First public release` added (pending commit) |
| `CITATION.cff` software/paper distinction | PASS | Rewritten with `preferred-citation:` block for paper (pending commit) |
| `CITATION.cff` ORCID | DECISION | Placeholder `0000-0000-0000-0000`; must replace with real ORCID before commit |
| `CITATION.cff` abstract, license, keywords | PASS | Added (pending commit) |

---

## 2. Build Artifacts Verification

| Item | Status | Notes |
|---|---|---|
| `python -m build` produces wheel | PASS | `bayesbreak-2.1.0-py3-none-any.whl` |
| `python -m build` produces sdist | PASS | `bayesbreak-2.1.0.tar.gz` |
| `twine check dist/*` | PASS | PASSED for both wheel and sdist |
| Wheel contains `py.typed` | PASS | Present after pending commit |
| Wheel contains `LICENSE` | PASS | Present |
| Wheel excludes experiment artifacts | PASS | No result JSONs, no manuscripts, no credentials |
| Wheel excludes manuscripts | PASS | Confirmed |
| Python source files in wheel | PASS | 62 Python source files across all subpackages |
| Subpackages | PASS | `families`, `datasets`, `baselines`, `experiments`, `nonconjugate`, + top-level modules |

### Smoke tests from installed wheel

| Test | Result |
|---|---|
| `BayesBreakGaussian(k_max=5).fit(X, y)` → `k_map=3` | PASS |
| `boundary_marginals_` accessible | PASS |
| `bayes_curve_mean_` accessible | PASS |
| `posterior_predictive_logpdf` returns float | PASS |
| `bayesbreak version` CLI → `2.1.0` | PASS |

---

## 3. Test Suite

| Item | Status | Notes |
|---|---|---|
| Tests passed | PASS | 963 passed |
| Tests failed | PASS | 0 failed |
| E00 oracle immutability fix | PASS | Fixture now writes to `tmp_path_factory` temp dir (pending commit) |
| E00 canonical oracle consistency check | PASS | Added to `tests/test_confirmatory_E00_oracles.py` (pending commit) |

---

## 4. Documentation

| Item | Status | Notes |
|---|---|---|
| `mkdocs build --strict` | PASS | Clean (no warnings, no broken links) |
| Tutorial count | PASS | "Ten" (was "Seven" in README/index.md — fixed, pending commit) |
| `pg_vb` alias description | PASS | Updated to "Alias for 'jj' — invokes the same Jaakkola–Jordan routine" in `docs/concepts.md` (pending commit) |
| Algorithm terminology in README | PASS | "EM" → "Jensen minorization-maximization" (pending commit) |
| Manuscript source paths in docs | PASS | Canonical paths (`bayesbreak-paper/paper/`, `bayesbreak-book/book/`) replacing symlink references (pending commit) |
| `docs/reproducibility.md` paths | PASS | Updated to canonical (pending commit) |
| GitHub Pages deployment | PASS | `gh-pages` branch at `1bd5ec4`; `https://osolari.github.io/bayesbreak` live |
| Post-release badge updates | PENDING | PyPI, Zenodo DOI, arXiv badges not yet added — requires live DOI from Phase 3 |

---

## 5. Confirmatory Experiments (E00–E10)

| Experiment | gate_pass | Notes |
|---|---|---|
| E00 — Exact oracle checks | True | 380/380 passed; artifact `03a4d1c38bfe08c1` |
| E01 — Gaussian calibration | True | ECE 0.00044–0.0081; F1@3 > 0.93 at Δ/σ=2 |
| E02 — Family portability | True | F1@3 0.88–1.00 at strong effect across 5 families |
| E03 | True | |
| E04 | True | |
| E05 | True | |
| E06 | True | `pg_vb = jj` alias documented |
| E07 | True | |
| E08 — Posterior prediction | True | A1/A2/PIT confirmed; artifact `cc26eda02dc82357` |
| E09 — Real data | True (gate) | INCOMPLETE scientifically: 25/31 cells truncated at k_max; well-log k_map=83 resolved; CGH single k_map=71 resolved; CGH pooled truncated; methylation truncated; SPX excluded (yfinance ToS) |
| E10 — Computational scaling | True | 15 cells; O(n²) memory; FB O(k_max × n^1.14) in practice; E2E 3.25s at n=2048 k_max=20, 500 MB; artifact `bece422c9d90093f` |

---

## 6. Manuscript Source (arXiv:2603.14681 replacement)

| Item | Status | Notes |
|---|---|---|
| `docs/manuscripts/bayesbreak-paper/paper/main.tex` | PASS | Present |
| Section `.tex` files (01–11 + appendix) | PASS | All present |
| `paper/figures/` (`.pdf` + `.png`) | PASS | All referenced figures present |
| `shared/bibliography/references.bib` | PASS | Present |
| `saim-paper.cls` | PASS | Present |
| LaTeX compilation (local) | NOT VERIFIED | No LaTeX compiler available locally; must compile in Overleaf before arXiv submission |
| PDF visual inspection | NOT VERIFIED | Must be done in Overleaf |
| E09 limitations section | PASS | 25/31 truncated cells documented; SPX exclusion documented |
| `pg_vb = jj` in E06 section | PASS | Documented |
| Code/data availability statement | PASS | Present; references GitHub repo |
| Zenodo DOI in availability statement | PENDING | DOI not yet minted (Phase 3 required first) |
| arXiv source package (no symlinks) | PENDING | Must use `--dereference` when packaging; symlinks (`paper`, `book`, `shared`) cannot be submitted to arXiv |

---

## 7. Zenodo Metadata (`.zenodo.json`)

| Item | Status | Notes |
|---|---|---|
| `.zenodo.json` present | PASS | Created (pending commit) |
| Title | PASS | Software-specific title |
| Version | PASS | `2.1.0` |
| License | PASS | MIT |
| Authors list | PASS | Present |
| ORCID in `.zenodo.json` | DECISION | Placeholder `0000-0000-0000-0000`; must replace with real ORCID |
| Zenodo GitHub integration enabled | PENDING | Must be enabled at `https://zenodo.org/account/settings/github/` before creating GitHub Release |
| Zenodo DOI | PENDING | Will be minted upon GitHub Release published event |

---

## 8. GitHub Actions Workflow

| Item | Status | Notes |
|---|---|---|
| `.github/workflows/release.yml` | PASS | Created (pending commit) |
| Trigger | PASS | `on: release: types: [published]` |
| PyPI environment | PASS | `environment: pypi` |
| PyPI Trusted Publishing (OIDC) | PENDING | Must be configured on PyPI after first manual upload creates the project |
| First release upload method | MANUAL | First upload via `twine upload` with API token; OIDC for subsequent releases |

---

## 9. GitHub Release Preparation

| Item | Status | Notes |
|---|---|---|
| Git tag `v2.1.0` | PENDING | Not yet created; created in Phase 1b |
| Tag pushed to origin | PENDING | Phase 1c |
| GitHub Release published | PENDING | Phase 3b |
| Release notes drafted | PASS | Content prepared in release plan |
| No prior GitHub Releases | PASS | First release |

---

## 10. Data Preservation

| Item | Status | Notes |
|---|---|---|
| E00–E10 confirmatory artifacts in repo | PASS | All manifest.json, oracle_checks.json, metrics.json committed |
| Manuscript figure PDFs/PNGs in repo | PASS | `shared/figures/results/` committed |
| Manuscript LaTeX tables in repo | PASS | `shared/tables/results/` committed |
| Raw data files in repo | NOT APPLICABLE | Accessed via `pooch` data loaders; not committed (size/license) |
| Fitted estimator pickles | NOT APPLICABLE | Gitignored (`*.fit.pkl`) by design |
| Well-log (TCPD) redistribution license | DECISION | Publicly available; exact license for redistribution not verified by author |
| Array-CGH (Bioconductor) redistribution | PASS | Public R package data |
| Methylation (methylKit) redistribution | PASS | Public R package data |
| SPX data redistribution | EXCLUDED | yfinance terms of service prohibit redistribution; simulated fallback only |
| Separate Zenodo data archive | NOT REQUIRED | All confirmatory manifests and CSVs are in the public GitHub repo |

---

## 11. Outstanding Author Decisions

The following items cannot be resolved by tooling and require explicit author action before or shortly after the first commit:

1. **ORCID** — Replace placeholder `0000-0000-0000-0000` in both `CITATION.cff` and `.zenodo.json` with the real ORCID identifier before committing.

2. **Canonical author name** — Decide between "Omid Solari" (as in `LICENSE`) and "Omid Shams Solari" (as in `pyproject.toml` and `CITATION.cff`). Apply consistently to all files.

3. **License classifier (PEP 639)** — The current `pyproject.toml` form triggers a setuptools deprecation warning. Either migrate to the SPDX form (`license = "MIT"`, removing the trove classifier and file-based form) or explicitly defer the migration.

4. **Python 3.13 classifier** — Do not add until CI is run on Python 3.13.

5. **`experiments/` in wheel** — Confirm that including the `bayesbreak/experiments/` subpackage in the public wheel is intentional (it currently serves the `bayesbreak synthetic` / `bayesbreak realdata` CLI subcommands).

6. **Well-log (TCPD) redistribution license** — Verify the exact license for the TCPD dataset before claiming redistribution rights.

7. **E09 scientific completeness** — Decide whether to (a) complete the R comparator study before journal submission, (b) explicitly reframe E09 as exploratory (not comparative), or (c) submit to JOSS first and defer the methodology paper.

8. **Zenodo GitHub integration** — Must be manually enabled at `https://zenodo.org/account/settings/github/` before the GitHub Release is published.

9. **PyPI Trusted Publishing** — Must be configured at `https://pypi.org/manage/project/bayesbreak/settings/publishing/` after the first manual upload.

---

## 12. Diff Summary — Pending Uncommitted Changes

All changes below exist in the working tree and are not yet committed. They are blocked only by the ORCID and author-name decisions (items 1 and 2 above).

| File | Change type | Summary |
|---|---|---|
| `tests/test_confirmatory_E00_oracles.py` | Modified | Fixture uses `tmp_path_factory`; canonical oracle consistency check added |
| `README.md` | Modified | "EM" → "Jensen minorization-maximization"; symlink paths → canonical; "Seven" → "Ten" tutorials; latent-template table entry corrected |
| `docs/concepts.md` | Modified | `pg_vb` alias description updated |
| `docs/index.md` | Modified | Tutorial count Seven → Ten |
| `docs/reproducibility.md` | Modified | Manuscript paths → canonical |
| `src/bayesbreak/py.typed` | Created | Declared in `pyproject.toml` but was missing from source tree |
| `pyproject.toml` | Modified | Broken `Report` URL replaced with `Paper = "https://arxiv.org/abs/2603.14681"` |
| `CITATION.cff` | Modified | Rewritten: software/paper distinction, `preferred-citation:`, abstract, license, keywords, ORCID (placeholder) |
| `CHANGELOG.md` | Modified | `## [2.1.0] — 2026-10-26 — First public release` section added |
| `.github/workflows/release.yml` | Created | GitHub Actions workflow: PyPI Trusted Publishing via OIDC, triggered on Release published |
| `.zenodo.json` | Created | Zenodo software deposit metadata |

---

## 13. Final GO / NO-GO Statement

**Current status: CONDITIONAL GO**

The package is technically ready to release:
- Test suite is clean (963/0)
- Build artifacts pass `twine check`
- Smoke tests pass from installed wheel
- Documentation builds cleanly
- All E00–E10 experiments have `gate_pass=True`
- All mechanical issues (broken URL, missing `py.typed`, missing CHANGELOG entry, missing workflow, missing `.zenodo.json`) are fixed in the working tree

**Conditions that must be satisfied before the commit and tag are created:**

- [ ] Author resolves ORCID placeholder (Decision 1)
- [ ] Author resolves canonical name discrepancy (Decision 2)
- [ ] Author confirms `experiments/` wheel inclusion is intentional (Decision 5)

**Conditions that do not block the commit but must be resolved before specific downstream phases:**

- [ ] Zenodo GitHub integration enabled (blocks Phase 3 Zenodo DOI)
- [ ] PyPI Trusted Publishing configured (blocks automated OIDC releases; first release can be manual)
- [ ] Overleaf LaTeX compilation verified clean (blocks Phase 4 arXiv submission)
- [ ] E09 strategy decided (blocks Phase 6 journal submission)
