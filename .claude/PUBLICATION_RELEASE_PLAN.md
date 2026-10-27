# BayesBreak 2.1.0 — Publication Release Plan

Package: `bayesbreak` | Version: `2.1.0` | arXiv: `2603.14681`
HEAD at plan creation: `aedc7eadf883f466512c7ba59fce2fe118b4fe6c`

---

## Phase 0 — Pre-commit corrections (author action required)

These changes exist in the working tree and are not yet committed. Some require author input before the commit is created.

### 0a. Required author actions before any commit

**ORCID placeholder** — both `CITATION.cff` and `.zenodo.json` contain `0000-0000-0000-0000`. Replace with the real ORCID before committing.

```
# Locate the placeholder in both files, then edit:
grep -n "0000-0000-0000-0000" CITATION.cff .zenodo.json
# Edit both files with the real ORCID, then proceed.
```

**Canonical author name** — `LICENSE` uses `Omid Solari`; `pyproject.toml` and `CITATION.cff` use `Omid Shams Solari`. Decide on one canonical form and apply it consistently across all files before committing.

**License classifier** — `pyproject.toml` uses the deprecated PEP 639 pattern (`License :: OSI Approved :: MIT License` alongside `license = { file = "LICENSE" }`). This generates a setuptools deprecation warning. Options:
- Keep the current form (warning is non-fatal, backward-compatible).
- Migrate to SPDX: replace `license = { file = "LICENSE" }` with `license = "MIT"` and remove the trove classifier (PEP 639 style). Requires testing with the version of setuptools/build in use.

**Python 3.13 classifier** — tests were run on Python 3.11 only. If 3.13 support is intended, add CI verification before declaring the classifier in `pyproject.toml`.

**`experiments/` in wheel** — `bayesbreak/experiments/` is included in the wheel and serves the `bayesbreak synthetic` / `bayesbreak realdata` CLI commands. Confirm this is intentional before public release.

### 0b. Pending uncommitted changes (ready once 0a is resolved)

| File | Change |
|---|---|
| `tests/test_confirmatory_E00_oracles.py` | Fixture writes to `tmp_path_factory`; added canonical oracle consistency check |
| `README.md` | "EM" → "Jensen minorization-maximization"; symlink paths → canonical; "Seven" → "Ten" tutorials; latent-template table entry corrected |
| `docs/concepts.md` | `pg_vb` described as alias for `jj` |
| `docs/index.md` | Tutorial count Seven → Ten |
| `docs/reproducibility.md` | Manuscript paths → canonical |
| `src/bayesbreak/py.typed` | Created (was declared in `pyproject.toml` but missing) |
| `pyproject.toml` | Broken `Report` URL → `Paper = "https://arxiv.org/abs/2603.14681"` |
| `CITATION.cff` | Rewritten: software vs paper distinction, `preferred-citation:` block, abstract, license, keywords, ORCID |
| `CHANGELOG.md` | Added `## [2.1.0] — 2026-10-26 — First public release` section |
| `.github/workflows/release.yml` | New: GitHub Actions release workflow (PyPI Trusted Publishing, OIDC, environment `pypi`) |
| `.zenodo.json` | New: Zenodo software metadata |

### 0c. GO/NO-GO gate after Phase 0

- [ ] Real ORCID substituted in `CITATION.cff` and `.zenodo.json`
- [ ] Canonical author name resolved and applied
- [ ] License classifier decision made and applied (or explicitly deferred)
- [ ] `experiments/` wheel inclusion confirmed as intentional
- [ ] All pending changes reviewed and approved by author

---

## Phase 1 — Commit, tag, and push

Execute after all Phase 0 author decisions are resolved and applied.

### 1a. Stage and commit

```bash
cd /Users/Omid.Solari/workspace/projects/bayesbreak

# Stage all pending changes
git add tests/test_confirmatory_E00_oracles.py
git add README.md
git add docs/concepts.md docs/index.md docs/reproducibility.md
git add src/bayesbreak/py.typed
git add pyproject.toml
git add CITATION.cff
git add CHANGELOG.md
git add .github/workflows/release.yml
git add .zenodo.json

# Review staged diff before committing
git diff --staged

# Commit
git commit -m "Release 2.1.0: package metadata, py.typed, workflow, Zenodo, citation"
```

### 1b. Tag

```bash
git tag -a v2.1.0 -m "BayesBreak 2.1.0 — first public release"
git log --oneline -3          # confirm tag is on the correct commit
```

### 1c. Push branch and tag

```bash
git push origin main
git push origin v2.1.0
```

### 1d. GO/NO-GO gate after Phase 1

- [ ] `git push` succeeded for both branch and tag
- [ ] Tag `v2.1.0` visible on GitHub at `https://github.com/osolari/bayesbreak/releases/tag/v2.1.0`
- [ ] GitHub Actions tab shows no unexpected workflow failures (the release workflow will not trigger yet — it fires on Release published, not on tag push)

---

## Phase 2 — PyPI first publish (manual, before Trusted Publishing)

Trusted Publishing (OIDC) cannot be configured until the PyPI project exists. The first upload must be manual with an API token. After Phase 2 completes, configure Trusted Publishing for all future releases.

### 2a. Build distribution artifacts

```bash
cd /Users/Omid.Solari/workspace/projects/bayesbreak

pip install --upgrade build twine
python -m build

# Verify artifacts
ls dist/
twine check dist/*
```

Expected output: `bayesbreak-2.1.0-py3-none-any.whl` and `bayesbreak-2.1.0.tar.gz`. `twine check` must report PASSED for both.

### 2b. Test upload to TestPyPI (recommended dry run)

```bash
twine upload --repository testpypi dist/*
# Enter TestPyPI credentials when prompted
# Verify at: https://test.pypi.org/project/bayesbreak/
```

Install from TestPyPI and run the smoke test:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bayesbreak==2.1.0
python -c "
import numpy as np
from bayesbreak import BayesBreakGaussian
X = np.linspace(0, 10, 200)
y = np.concatenate([np.random.normal(0,1,80), np.random.normal(3,1,70), np.random.normal(0,1,50)])
m = BayesBreakGaussian(k_max=5).fit(X, y)
print('k_map:', m.k_map_, 'version:', __import__('bayesbreak').__version__)
"
bayesbreak version
```

Expected: `k_map` between 2–4; `version: 2.1.0`.

### 2c. Upload to PyPI

```bash
twine upload dist/*
# Enter PyPI credentials (API token, not password) when prompted
```

Verify: `https://pypi.org/project/bayesbreak/2.1.0/`

### 2d. Configure PyPI Trusted Publishing (manual account action)

After the project exists on PyPI, go to:
`https://pypi.org/manage/project/bayesbreak/settings/publishing/`

Add a publisher with the following configuration:

| Field | Value |
|---|---|
| Repository owner | `osolari` |
| Repository name | `bayesbreak` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

This cannot be done from the command line. Once configured, subsequent releases via GitHub Release will use OIDC and no API token is needed.

### 2e. GO/NO-GO gate after Phase 2

- [ ] `twine check` PASSED for both wheel and sdist
- [ ] TestPyPI smoke test passed
- [ ] `https://pypi.org/project/bayesbreak/2.1.0/` is live and shows correct metadata
- [ ] Trusted Publishing configured on PyPI (blocks Trusted Publishing in Phase 3+, but does not block Phase 3 itself)

---

## Phase 3 — GitHub Release and Zenodo

GitHub Release triggers `.github/workflows/release.yml`. Zenodo is connected via the GitHub webhook and will mint a DOI upon the Release published event.

### 3a. Verify Zenodo GitHub integration

Before creating the Release, confirm the Zenodo webhook is active:
1. Go to `https://zenodo.org/account/settings/github/`
2. Confirm the `osolari/bayesbreak` repository is listed and the toggle is ON.
3. If not listed: click "Sync" to refresh the repository list, then enable the toggle.

This is a manual account action — it cannot be done from the command line.

### 3b. Create GitHub Release

```bash
gh release create v2.1.0 \
  --title "BayesBreak 2.1.0 — First public release" \
  --notes "$(cat <<'EOF'
## BayesBreak 2.1.0

First public release of BayesBreak, a Python package for Bayesian change-point detection via Dirichlet-process segmentation with conjugate and non-conjugate exponential families.

### What's in this release
- Conjugate families: Gaussian, Poisson, Binomial, NegativeBinomial, Beta, BetaObs, Bernoulli, LogisticNormal
- Nonconjugate error bounds (Jaakkola–Jordan variational)
- Dirichlet process prior with geometric and truncated-geometric variants
- E00–E10 confirmatory experiment suite: all gate_pass=True
- Full test suite: 963 tests, 0 failures
- Companion preprint: https://arxiv.org/abs/2603.14681

### Installation
\`\`\`
pip install bayesbreak==2.1.0
\`\`\`

### Notes
- E09 real-data scientific completeness is INCOMPLETE (25/31 cells truncated at k_max); documented in manuscript.
- SPX dataset excluded (yfinance terms of service).
EOF
)"
```

Verify: `https://github.com/osolari/bayesbreak/releases/tag/v2.1.0`

### 3c. Confirm GitHub Actions release workflow

```bash
gh run list --workflow=release.yml --limit=5
gh run view <run-id>        # substitute run ID from above
```

If the workflow used Trusted Publishing and succeeded, the wheel was re-uploaded by the workflow. If Trusted Publishing was not yet configured (Phase 2d incomplete), the workflow will fail with an OIDC error — this is acceptable on the first release since Phase 2c already uploaded the wheel manually. Configure Trusted Publishing (Phase 2d) before the next release.

### 3d. Retrieve Zenodo DOI

After the Release is published, Zenodo mints a DOI within a few minutes. Go to:
`https://zenodo.org/search?q=bayesbreak`

Or check the Zenodo GitHub integration page. The DOI will have the form `10.5281/zenodo.XXXXXXX`.

Record the DOI — it is needed for Phase 5 badge updates.

### 3e. GO/NO-GO gate after Phase 3

- [ ] GitHub Release at `v2.1.0` is publicly visible
- [ ] Zenodo record created and DOI minted
- [ ] Zenodo record metadata (title, authors, version, license) matches `.zenodo.json`
- [ ] Zenodo ORCID in the record matches the author's real ORCID

---

## Phase 4 — arXiv replacement

Submit a replacement for arXiv:2603.14681 incorporating all E00–E10 results and Phase 3 manuscript revisions.

### 4a. Pre-submission manuscript checklist

The following must be verified before packaging the arXiv source:

- [ ] LaTeX compilation succeeds cleanly in Overleaf (no errors, no overfull hbox warnings in result sections)
- [ ] All referenced figures are present in `docs/manuscripts/bayesbreak-paper/paper/figures/` as `.pdf`
- [ ] All referenced tables are present as `.tex` files
- [ ] E09 limitations section explicitly states 25/31 cells truncated and SPX excluded
- [ ] `pg_vb = jj` alias documented in the E06 section
- [ ] Code and data availability statement is present and references the GitHub repo and Zenodo DOI from Phase 3d
- [ ] Bibliography `references.bib` includes the Zenodo software citation entry with the minted DOI

### 4b. Prepare arXiv source package

arXiv requires direct directory structure — symlinks are not followed. Use the canonical paths:

```bash
cd /Users/Omid.Solari/workspace/projects/bayesbreak/docs/manuscripts/bayesbreak-paper

# Package source without symlinks
tar czf /tmp/bayesbreak-arxiv-replacement.tar.gz \
  --dereference \
  paper/main.tex \
  paper/saim-paper.cls \
  paper/01-introduction.tex \
  paper/02-*.tex \
  paper/03-*.tex \
  paper/04-*.tex \
  paper/05-*.tex \
  paper/06-*.tex \
  paper/07-*.tex \
  paper/08-*.tex \
  paper/09-*.tex \
  paper/10-*.tex \
  paper/11-conclusion.tex \
  paper/appendix*.tex \
  paper/figures/ \
  shared/bibliography/references.bib

# Inspect the archive
tar tzf /tmp/bayesbreak-arxiv-replacement.tar.gz | head -40
```

Adjust the file list to match the actual section filenames present in `paper/`.

### 4c. Submit replacement on arXiv

1. Log in to `https://arxiv.org`
2. Navigate to `https://arxiv.org/abs/2603.14681` → "Replace submission"
3. Upload `/tmp/bayesbreak-arxiv-replacement.tar.gz`
4. Verify the compiled PDF preview in the arXiv submission interface before finalizing
5. Submit

### 4d. GO/NO-GO gate after Phase 4

- [ ] arXiv replacement submitted and in moderation queue
- [ ] arXiv PDF preview confirmed correct (figures render, tables render, no compilation errors)
- [ ] Zenodo DOI from Phase 3d included in the data availability statement

---

## Phase 5 — Documentation and citation sync

Update GitHub Pages, README badges, and citation files to reflect live PyPI, Zenodo, and arXiv replacement.

### 5a. Update README badges

After Phase 2–4 complete, the following badges can be made live in `README.md`:

```markdown
[![PyPI version](https://badge.fury.io/py/bayesbreak.svg)](https://pypi.org/project/bayesbreak/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-2603.14681-b31b1b.svg)](https://arxiv.org/abs/2603.14681)
```

Replace `10.5281/zenodo.XXXXXXX` with the DOI from Phase 3d.

```bash
# Edit README.md with the real DOI, then:
git add README.md
git commit -m "Add PyPI, Zenodo, and arXiv badges post-release"
git push origin main
```

### 5b. Update CITATION.cff with Zenodo DOI

```bash
# Edit CITATION.cff: add identifiers block with the Zenodo DOI
# Example entry to add under `identifiers:`:
#   - type: doi
#     value: 10.5281/zenodo.XXXXXXX
#     description: "Zenodo software archive"

git add CITATION.cff
git commit -m "Add Zenodo DOI to CITATION.cff"
git push origin main
```

### 5c. Rebuild and deploy GitHub Pages

```bash
cd /Users/Omid.Solari/workspace/projects/bayesbreak
mkdocs build --strict          # must be clean
mkdocs gh-deploy --force       # deploys to gh-pages branch
```

Verify: `https://osolari.github.io/bayesbreak` loads correctly, installation page shows `pip install bayesbreak`, version is `2.1.0`.

### 5d. GO/NO-GO gate after Phase 5

- [ ] README badges all resolve to live pages
- [ ] `CITATION.cff` contains the real Zenodo DOI
- [ ] `https://osolari.github.io/bayesbreak` live and correct
- [ ] `gh-pages` branch updated (check `git log origin/gh-pages`)

---

## Phase 6 — Journal submission

### 6a. Resolve E09 before submission

**Primary risk**: reviewers at Statistics and Computing and JCGS will ask for R-based comparators (CBS via `DNAcopy`, SMUCE) for the CGH and methylation datasets. E09 currently has 25/31 cells truncated and no R comparator study.

Options:
1. Complete the R comparator study (implement CBS/SMUCE baseline via `rpy2` or separate R script) and re-run E09 with increased `k_max` where scientifically justified.
2. Explicitly scope E09 out in the paper: reframe as "our method applied to real data with known limitations" rather than a comparative study. Remove language that implies superiority over R baselines.
3. Submit only to JOSS first (software paper, no comparative claims required), and defer the methodology paper until E09 is complete.

**Minimum for Statistics and Computing or JCGS**: at minimum, the truncated cells should either be resolved or clearly scoped with scientific rationale that reviewers will accept.

### 6b. Prepare cover letter

The cover letter should address:
- The Jensen minorization-maximization algorithm (not EM): explain the distinction upfront
- E09 limitations: be forthright about truncation and SPX exclusion
- `pg_vb = jj` alias: disclose that the variational bound uses the Jaakkola–Jordan approximation
- All E00–E10 experiments are computationally reproducible from the public GitHub repo and Zenodo archive

### 6c. Submission commands (venue-specific)

**Statistics and Computing** (Springer Editorial Manager):
- Submission via `https://www.editorialmanager.com/stco/`
- File types: PDF for review, LaTeX source for accepted manuscripts
- No command-line submission tool

**JCGS** (Taylor & Francis ScholarOne):
- Submission via `https://mc.manuscriptcentral.com/jcgs`

**JOSS**:
```bash
# JOSS submission is via GitHub issue at https://github.com/openjournals/joss-reviews/issues/new
# Requires a short paper (250–1000 words) in docs/joss-paper/paper.md
# JOSS will build the PDF automatically from the markdown
```

### 6d. GO/NO-GO gate for journal submission

- [ ] E09 either resolved or scoped with scientific rationale approved by author
- [ ] arXiv replacement (Phase 4) live (reviewers will check)
- [ ] Zenodo DOI (Phase 3) live (cited in the paper)
- [ ] Cover letter drafted and reviewed

---

## Manual account actions summary

The following actions cannot be performed from the command line and require browser-based interaction with external services:

| Action | Service | URL | When |
|---|---|---|---|
| Replace ORCID placeholder | ORCID registry | `https://orcid.org` | Before Phase 0 commit |
| Enable Zenodo GitHub integration | Zenodo | `https://zenodo.org/account/settings/github/` | Before Phase 3 |
| Configure PyPI Trusted Publishing | PyPI | `https://pypi.org/manage/project/bayesbreak/settings/publishing/` | After Phase 2c |
| Create GitHub Release | GitHub | `https://github.com/osolari/bayesbreak/releases/new` | Phase 3 (or use `gh` CLI as shown) |
| Submit arXiv replacement | arXiv | `https://arxiv.org/abs/2603.14681` | Phase 4 |
| Submit journal manuscript | Venue | Venue-specific | Phase 6 |
