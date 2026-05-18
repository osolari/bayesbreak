# Manuscript

<div class="saim-cite" markdown>
> **Solari, O. S.** (2026). *Generalized Hierarchical Bayesian
> Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and
> Grouped/Latent-Group Designs.* sAIm Labs.
> [arXiv:2603.14681](https://arxiv.org/abs/2603.14681){target=_blank} ·
> [PDF](https://github.com/osolari/bayesbreak/blob/master/docs/report/bayesbreak.pdf){target=_blank}
</div>

The technical report (PDF) and source LaTeX are bundled in
`docs/report/` and reproduced verbatim in `docs/report.zip`.

- [`docs/report/bayesbreak.pdf`](https://github.com/osolari/bayesbreak/blob/master/docs/report/bayesbreak.pdf) — the rendered manuscript.
- [`docs/report.zip`](https://github.com/osolari/bayesbreak/blob/master/docs/report.zip) — Overleaf-ready source tree (sections, figures, tables, bibliography, class files).
- [`docs/report/CHANGELOG.md`](https://github.com/osolari/bayesbreak/blob/master/docs/report/CHANGELOG.md) — manuscript change log (separate from the repo CHANGELOG).
- [`docs/report/CODING_AGENT_HANDOFF.md`](https://github.com/osolari/bayesbreak/blob/master/docs/report/CODING_AGENT_HANDOFF.md) — author-verification checklist mirrored in the repo.

## Sections

The manuscript is organised into:

1. **§1 Introduction** — contributions, related work, paper organisation.
2. **§2 Problem formulation** — inferential targets, the standing assumption
   `ass:standing-offline`, the segmentation-space and posterior-summary
   definitions.
3. **§3 Setup and notation** — the Bayesian segmentation model, partition
   priors, weighted exponential-family blocks.
4. **§4 Method** — block evidence, DP, irregular designs, pooling, latent
   groups, families, non-conjugate blocks, prediction.
5. **§5 Algorithms** — implementation-centric pseudocode, complexity
   (`prop:bb-complexity`), max-sum correctness (`thm:map-correctness`).
6. **§5b Limitations** — named failure modes, the assumption-to-failure-mode
   map, the decision flowchart, identifiability failures.
7. **§6 Experiments and results** — synthetic suite, four real-data case
   studies, planned external-comparator agenda.
8. **§7 Conclusion** — recap and planned next iteration.
9. **§8 Appendix** — proofs, annotated literature review, real-data
   reproduction pipelines, code-and-reproduction notes.

## Code-to-manuscript cross-reference

Every public surface in `bayesbreak` cites the manuscript label it
implements. The `docs/api.md` page lists the canonical map. Highlights:

- `bayesbreak.dp.forward_backward` ↔ `prop:fb-duality`, `eq:LR`.
- `bayesbreak.dp.max_sum_segmentation` ↔ `thm:map-correctness`.
- Each family's `_compute_block_evidence` ↔ `prop:gaussian-block` etc.
- `bayesbreak.run_non_conjugate_diagnostics` ↔ `ass:uniform-block-error`,
  `prop:stability`, `prop:uniform-bounds`,
  `cor:probability-error-conversion`.
- `bayesbreak.SharedBoundaryReplicatesSegmenter` ↔ `thm:multisubject`,
  `prop:shared-boundary-identifiability`.
- `bayesbreak.BayesBreakMixtureClassifier` ↔ `thm:em-monotone`,
  `prop:latent-identifiability`, `rem:teicher-overspec`,
  `ex:label-switch-counterexample`.
- `bayesbreak.SlidingWindowSegmenter` ↔ §5b *Computational regime*.

## Reproducing manuscript figures and tables

```bash
PYTHONPATH=src python -m bayesbreak.reproduce all
```

This regenerates every figure and table under `scripts/figures/` and
`scripts/tables/` into `results/` (the figure scripts also write into
`docs/report/figures/` for the manuscript). See
[Reproducibility](reproducibility.md) for full details.
