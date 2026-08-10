# BayesBreak main journal paper

This paper is derived from the original manuscript and the corrected technical book. Its title and scientific emphasis are fixed by the author:

> **Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

The manuscript presents BayesBreak as a generalized hierarchical Bayesian multiple-changepoint method. Exact segment integration is developed for regular exponential-family observation models with proper conjugate priors; the partition recursions then compute posterior quantities over contiguous segmentations. The paper also treats irregular designs, shared and group-specific changepoints across multiple sequences, finite latent groups, numerical segment integration for nonconjugate models, posterior prediction, and the archived synthetic and real-data results.

## Build targets

```bash
make -C docs/manuscript paper          # two-column journal route
make -C docs/manuscript paper-single   # single-column review route
make -C docs/manuscript paper-all      # both routes
make -C docs/manuscript validate-phase6
```

The SAIM Unified Professional template controls document typography and layout only. The title, terminology, model statements, contribution claims, and interpretation of results are determined by the manuscript and the author's decisions.

All populated archived numbers are real executed computations. `RES-BB-CMP-002` is excluded from comparator conclusions because its arrays use incompatible axes. `RES-BB-RD-007Q` is excluded from posterior-predictive conclusions because the archived calculation used the wrong observation-family routine and an unstated endpoint assignment. Corrected calculations require new result identifiers and explicit links to the archived outputs.
