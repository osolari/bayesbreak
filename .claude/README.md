# BayesBreak working instructions

This directory is the only location for active coding-agent instructions. Do not search for older edit plans, archived run histories or auxiliary handoffs. Follow `CURRENT_WORK.md`, `RESULT_POLICY.md`, and `skills/bayesbreak-release-audit/SKILL.md`.

## Canonical sources

- Single-column paper: `docs/manuscripts/paper/main.tex`.
- Technical book: `docs/manuscripts/book/main.tex` (additional scientific synchronization remains).
- Shared LaTeX resources: `docs/manuscripts/shared/`.
- Experiment definitions: `skills/bayesbreak-release-audit/protocols/experiments.json` with its `references/experiment_protocols.md`.
- Pending scientific tasks and publication gates: `CURRENT_WORK.md`.
- Retention / regeneration decisions for existing paper results: `RESULT_POLICY.md`.

## Methodological invariants

Preserve the generalized exponential-family framework, proper block marginalization, normalized partition priors, sum-product posterior calculations, max-sum joint MAP, irregular-design priors, shared-boundary pooling, and the **finite-template score-clustering/MM** objective. Do not substitute generative EM without explicit approval. Enforce family-specific observation descriptors and do not mix ordinary precision/exposure with arbitrary likelihood powers. Keep proofs adjacent to theorem statements. Do not invent results. Both publications use the same verified results.

Run `python .claude/skills/bayesbreak-release-audit/scripts/validate_protocols.py`, `python .claude/skills/bayesbreak-release-audit/scripts/check_active_repository.py`, and both LaTeX build targets after changing sources. Document actual failures in the current task report, not in new phase/history files.
