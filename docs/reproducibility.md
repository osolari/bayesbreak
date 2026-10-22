# Reproducibility and experiments

This page documents the **current execution path**. The single-column paper source is `docs/manuscripts/paper/main.tex` and the technical book is `docs/manuscripts/book/main.tex`. Their build commands are:

```bash
make -C docs/manuscripts paper
make -C docs/manuscripts book
```

The controlled, prospective experiments are E00-E10, specified by `.claude/skills/bayesbreak-release-audit/protocols/experiments.json` and its `references/experiment_protocols.md`. Run the validators before execution:

```bash
python .claude/skills/bayesbreak-release-audit/scripts/check_active_repository.py
python .claude/skills/bayesbreak-release-audit/scripts/validate_protocols.py
pytest -q
```

Experiment manifests must record model family, descriptor semantics, dataset/source and split hashes, coordinate axis, priors, tuning policy, software version, seeds, numerical method, convergence/failure states, time and output hashes. Use the independent sequence or biologically appropriate group as the unit of replication, as declared in the protocol. Write new outputs under `results/confirmatory/<experiment_id>/<config_sha256>/`, never into an existing paper figure or table by hand. Only a completed, validated output may replace a Planned slot.

For real-data evaluation, disable simulated fallbacks and confirm data source provenance. The experiment protocol is the authority for which comparisons use external truth, held-out predictive scores or mere segmentation agreement. Missing validation must remain explicitly outstanding.

Existing figures in `docs/manuscripts/shared/figures/results/` remain reader-facing assets from earlier completed experiments, subject to `.claude/RESULT_POLICY.md`. This repo does not ship obsolete runs or historical test profiles as current evidence.
