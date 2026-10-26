# Reproducibility and experiments

The single-column paper source is `docs/manuscripts/paper/main.tex` and the companion technical book is `docs/manuscripts/book/main.tex`. Build commands:

```bash
make -C docs/manuscripts paper
make -C docs/manuscripts book
```

## Experiment execution status

All controlled E00-E10 experiments have been executed and verified. Status, artifact hashes, and scientific-completeness notes are in `.claude/CURRENT_WORK.md`. Scientific limitations and outstanding E09 gaps are documented there and in `.claude/RESULT_POLICY.md`.

The experiment protocols were written prospectively in `.claude/skills/bayesbreak-release-audit/protocols/experiments.json` and `references/experiment_protocols.md`. Those files define exact design and integrity requirements; they must not be changed after a run begins. Run the validators before any new execution:

```bash
python .claude/skills/bayesbreak-release-audit/scripts/check_active_repository.py
python .claude/skills/bayesbreak-release-audit/scripts/validate_protocols.py
pytest -q
```

## Adding new results

New outputs belong under `results/confirmatory/<experiment_id>/<config_sha256>/` with `manifest.json`, per-replicate metrics, failure logs, figure data, and LaTeX fragments. Every manifest must record model family, descriptor semantics, dataset and split hashes, coordinate axis, priors, tuning policy, software version, seeds, numerical method, convergence states, and output hashes.

For real-data evaluation, disable simulated fallbacks and confirm source provenance. Missing validation remains explicitly outstanding and must not be labeled complete.

## Figure and table pipeline

Publication figures reside in `docs/manuscripts/paper/figures/`. Regenerate them from cached CSV data with:

```bash
python scripts/figures/render.py --all
```

This runs in approximately 10 seconds and performs no scientific computation. See `docs/manuscripts/shared/FIGURES.md` for the full figure-cache architecture.

Accepted results and regeneration rules are in `.claude/RESULT_POLICY.md`. Do not modify a reported number manually; rerun with corrected code and verified data.
