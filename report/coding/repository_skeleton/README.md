# BayesBreak repository skeleton

**Status:** interface blueprint only. The statistical algorithms are not implemented in this skeleton.

The skeleton exposes the module boundaries, typed records, configuration keys, result schemas, task IDs, failure states, and tests that an implementation must satisfy for:

**Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

The scientific source of truth is `../../shared/handoffs/coding_agent_handoff.json`. The generated standalone handoff is `../CODING_AGENT_HANDOFF.md`.

## Use restriction

Do not publish, benchmark, or cite this directory as a completed BayesBreak implementation. Public functions that require scientific code raise `NotImplementedError` with the governing `CODE-BB-*` task. A completed implementation must preserve archived populated results, create new IDs for corrected computations, and satisfy the task-level acceptance tests.

## Skeleton checks

From the unified project root:

```bash
python coding/repository_skeleton/scripts/check_skeleton.py
python -m pytest -q coding/repository_skeleton/tests
```

These commands verify interface presence and explicit incomplete states. They do not validate the BayesBreak statistical algorithms.
