# Development and contribution

## Environment setup

The recommended development environment is the `bayesbreak` conda environment
(Python 3.11):

```bash
bash setup_env.sh          # creates conda env "bayesbreak"
bash setup_env.sh --venv   # or a plain venv
```

Install in editable mode with all dev dependencies:

```bash
pip install -e ".[dev]"
```

## Running the test suite

```bash
pytest tests/              # full suite with coverage
pytest tests/ -x -q        # fast, stop on first failure
pytest tests/ -m "not slow"  # skip long-running tests
```

The suite includes conceptual-correctness tests (brute-force DP comparisons,
closed-form predictive checks, length-prior plumbing, EM monotonicity), numerical
regression tests for E00–E10 confirmatory artifacts, and manuscript-asset
immutability checks.

**Required condition for release:** zero unexplained test failures.

## Building the documentation

```bash
mkdocs build --strict      # build site to site/
mkdocs serve               # local preview at http://127.0.0.1:8000
```

The site is excluded from version control (`site/` is gitignored). The
`docs/` source is committed on `main`; the built site is deployed to
`gh-pages`.

## Updating API and tutorial pages

The API reference (`docs/api.md`) is generated from docstrings via
`mkdocstrings`. Run `mkdocs build` to verify that all referenced symbols
resolve. Tutorial notebooks in `docs/tutorials/` are executed on demand by
`mkdocs-jupyter`; set `execute: false` in `mkdocs.yml` to skip execution
during builds.

## Regenerating figures

Publication figures use a two-stage pipeline:

```bash
# Stage B (fast, ~10 s): render from cached CSV/JSON
python scripts/figures/render.py --all

# Render a specific figure
python scripts/figures/render.py --only fig1

# Stage A (slow): rebuild CSV caches from experiment artifacts
python scripts/figures/prepare_data.py --all
```

Stage B reads from `docs/manuscripts/shared/figures/data/` and writes to
`docs/manuscripts/paper/figures/`. It performs no inference and never modifies
`results/confirmatory/`.

## Updating the manuscript

The paper is in `docs/manuscripts/bayesbreak-paper/paper/` (symlinked as
`docs/manuscripts/paper/`). It is maintained in Overleaf and mirrored here.
Always run `overleaf_sync.sh compare bayesbreak-paper` before pushing. See
`docs/guides/overleaf_runbook.md`.

Any numerical update must come from a verified run record following the frozen
E00–E10 protocols. Do not adjust reported numbers manually.

## Publishing GitHub Pages

GitHub Pages is published from the `gh-pages` branch. To deploy an update:

```bash
# 1. Build from main
mkdocs build --strict

# 2. Deploy to gh-pages using mike (preserves branch history)
#    or use the worktree procedure:
git worktree add /tmp/gh-pages gh-pages
rsync -av --delete --exclude='.git' site/ /tmp/gh-pages/
cd /tmp/gh-pages
git add -A
git commit -m "Deploy site from $(git -C /path/to/repo rev-parse --short HEAD)"
git push origin gh-pages
git worktree remove /tmp/gh-pages
```

Do not force-push `gh-pages`. Do not modify `_legacy/` if it exists.

## Validating a release

Before committing to `main`:

1. All tests pass (`pytest tests/`).
2. `mkdocs build --strict` completes without errors.
3. Confirmatory manifests are unmodified (`pytest tests/test_confirmatory_manifest_immutability.py`).
4. Figure and table inventory validates (`python scripts/figures/validate_inventory.py`).
5. Git diff contains no unintended scientific artifact changes.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/osolari/bayesbreak/blob/main/CONTRIBUTING.md)
in the repository root. Bug reports and feature requests: GitHub Issues.
