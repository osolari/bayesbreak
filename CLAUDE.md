# BayesBreak: one active research workflow

Read only the following authoritative files at the start of a Claude Code session:

1. `.claude/README.md` for navigation and fixed methodological requirements.
2. `.claude/CURRENT_WORK.md` for outstanding code, manuscript and experiment tasks.
3. `.claude/RESULT_POLICY.md` before changing any reported number, figure or table.
4. `.claude/skills/bayesbreak-release-audit/SKILL.md` and its E00-E10 protocol references when implementing or running experiments.

The only maintained paper is `docs/manuscripts/paper/main.tex` (single column). The companion book is `docs/manuscripts/book/main.tex`. Both share `docs/manuscripts/shared/`. Do not introduce an alternate paper, a history directory, phase-specific plans or competing experiment definitions. Preserve all existing verified results under their stated models and leave unrun figures/tables labeled Planned. No new experiment is completed until actually executed and checked.

Source code is in `src/bayesbreak/`; tests are in `tests/`. Work on the current code and E00-E10 specifications, not on hypothetical migrations or historical tasks.
