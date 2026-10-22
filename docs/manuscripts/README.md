# BayesBreak manuscripts

`paper/main.tex` is the only maintained paper, in the SAIM v1.1 single-column style. `book/main.tex` is the companion long-form technical book. Both share `shared/` LaTeX macros, figures, tables and references.

Build from the repo root using `make -C docs/manuscripts paper` or `make -C docs/manuscripts book`. The source is organized to compile from `docs/manuscripts/` so LaTeX can resolve the common `shared/` paths. Book-wide scientific reconciliation with the latest paper is still outstanding; see `.claude/CURRENT_WORK.md`. The new E00-E10 experiments are planned rather than executed, with full designs in `.claude/skills/bayesbreak-release-audit/`.

Do not recreate other paper formats, authoring copies, development histories, or LaTeX-generated coding-agent instructions here.
