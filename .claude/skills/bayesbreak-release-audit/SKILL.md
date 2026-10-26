---
name: bayesbreak-release-audit
description: Implement and validate BayesBreak's current E00-E10 statistical experiments, numerical likelihood corrections, manuscript result regeneration, and single-column paper/book consistency using only the current repo's canonical protocols. Use in Claude Code for development, experiment execution, reproducibility tests and scientific release QA.
---

# BayesBreak scientific implementation and experiment skill

## Start here

Read `CLAUDE.md`, `.claude/CURRENT_WORK.md`, `.claude/RESULT_POLICY.md`, then this skill's `references/experiment_protocols.md`, `references/metrics_and_statistics.md`, `references/figure_table_register.md`, and `protocols/experiments.json`. These are the only active execution specifications. There is no separate stage or history-driven plan.

## Invariants

1. Protect the generalized conjugate-exponential-family segmentation framework, normalized partition prior, exact posterior messages, max-sum joint MAP and finite-template score-clustering/MM model. Never silently switch the latter to generative EM.
2. Implement Gaussian precision, Poisson exposure and Binomial trials as their actual sampling-model descriptors; do not treat nonunit powers as ordinary evidence. Keep an explicitly separate API if generalized-Bayes powers are scientifically needed.
3. Show all theorem and proposition proofs directly below their statements in the **one-column** SAIM v1.1 paper. No status badges, development history or run IDs in scientific prose.
4. E00-E10 have all been executed and passed their execution gate (gate_pass=True for all). E00-E08 and E10 are scientifically complete. E09 passed its execution gate but has incomplete scientific evaluation — CGH pooled and methylation posteriors are truncated at the tested k_max, R baselines are unavailable, and SPX was excluded from the authentic real-data experiment. These are documented in `.claude/CURRENT_WORK.md` and `.claude/RESULT_POLICY.md`. For any *new* experiment or re-run: treat it as planned until an actual run produces checked metrics, manuscript cells stay `TBD`, and figures retain their Planned label until the acceptance gate passes.
5. Retain only the bounded interpretation of existing paper results recorded in `.claude/RESULT_POLICY.md`. Regenerate numerical results from original inputs after any likelihood, scoring, inference or model change. Never edit old numbers manually.

## Procedure

1. Run both checks before editing:
   `python .claude/skills/bayesbreak-release-audit/scripts/validate_protocols.py`
   and `python .claude/skills/bayesbreak-release-audit/scripts/check_active_repository.py`.
2. Write one reproducible code change and its mathematical oracle tests at a time. Test unit and nonunit sampling descriptors, all-missing blocks, posterior normalization, known small-case posteriors and impossibility states.
3. Reject synthetic fallback for any benchmark declared real-data; verify source/coordinate/weight/split hashes, and document the actual family and likelihood support.
4. Freeze protocol JSON, seed policy and comparator tuning before inspecting confirmatory outcomes. Run the exact design in `references/experiment_protocols.md` without substituting convenient parameters.
5. Store new verified runs only under `results/confirmatory/<experiment_id>/<config_sha256>/` with `manifest.json`, per-replicate metrics, failure logs, figure data and LaTeX fragments. An invalid or interrupted cell is not silently omitted or retried under a different method.
6. Use correct independent replication units for uncertainty, proper predictive scoring on the same observations, and independently sourced boundary truth. A fitted BayesBreak segmentation may be a comparison reference only when labeled **agreement**.
7. Fill the paper's Planned figures and tables only after every required cell passes. Then update the book using the **same** validated result records. Do not change scientific claims to match missing experiments.
8. Build: `make -C docs/manuscripts paper` and `make -C docs/manuscripts book`. Inspect equation references, citations, floating figures, chapter pages and unresolved layout warnings.

## Completion rules

Use `references/figure_table_register.md` for target labels and filenames. Run the independent E00 oracle, family-correct predictive tests, appropriate integration convergence checks, full package tests, and build/visual QA. The release remains incomplete while any required scientific gate fails; record the observed failure under `.claude/CURRENT_WORK.md`, not in a new revision or historical workflow directory.
