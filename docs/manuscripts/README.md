# BayesBreak manuscripts

Two self-contained manuscript packages, each mirrored to its own Overleaf project:

| Package | Entry point | Overleaf project |
|---|---|---|
| `bayesbreak-paper/` | `paper/main.tex` | SAIM v1.1 single-column journal paper |
| `bayesbreak-book/` | `book/main.tex` | Long-form technical book |

Each package carries its own `shared/` tree (macros, figures, tables, bibliography) so it compiles independently on Overleaf without cross-project `\input` paths.

## Building locally

```bash
make -C docs/manuscripts paper   # builds bayesbreak-paper → build/bayesbreak-paper.pdf
make -C docs/manuscripts book    # builds bayesbreak-book  → build/bayesbreak-book.pdf
make -C docs/manuscripts         # builds both
make -C docs/manuscripts clean   # removes build/
```

Requires `latexmk` and a standard TeX Live / MacTeX installation.

## Overleaf sync

Manuscripts are kept in sync with Overleaf via `scripts/overleaf_sync.sh`.
Always compare before pushing — push uses `--delete` and will remove
Overleaf-only files without warning.

```bash
bash scripts/overleaf_sync.sh compare all          # diff local vs Overleaf
bash scripts/overleaf_sync.sh push all -m "msg"    # push local → Overleaf
bash scripts/overleaf_sync.sh pull all             # pull Overleaf → local
bash scripts/overleaf_sync.sh doctor               # diagnose setup issues
```

Project IDs and subpath mappings are in `overleaf-projects.tsv`.
Auth comes from the git credential helper for `git.overleaf.com`; no token is
stored in this repo.

## Structure

```
docs/manuscripts/
  bayesbreak-paper/
    paper/          LaTeX source (sections, figures, saim-paper.cls)
    shared/         Macros, components, bibliography, tables, figures
    latexmkrc       Build configuration
  bayesbreak-book/
    book/           LaTeX source (chapters, appendices, saim-book.sty)
    shared/         Macros, components, bibliography, tables, figures
    latexmkrc       Build configuration
  Makefile          Local build targets
  overleaf-projects.tsv   Overleaf project registry
```

## Key shared macros

Project-wide metadata (title, author, emails, URLs, DOI) is defined in
`<package>/shared/metadata.tex` and used by both `main.tex` files.
Edit metadata there; do not hardcode values in section files.
