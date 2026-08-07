# RES-BB-CMP-003 corrected array-CGH comparator

Parent: `RES-BB-CMP-002`. Protocols: `EPR-BB-010`, `EPR-BB-013`.
Scientific execution commit: `4a5a53a5a5ec6ad829d394e213cbe0f3a63402b2`.

The exact hashed CRAN ecp ACGH matrix contains 2,215 probes and 43 subjects. The shared BayesBreak fit reproduced 15 MAP segments and the archived pooled log evidence (76359.7995869515). Comparators ran on the unflattened raw matrix and were scored on the same probe-index axis.

| Algorithm | Boundaries (target 14) | Count status | F1@3 | Matched MAE@3 | Exact Jaccard |
|---|---:|---|---:|---:|---:|
| PELT | 11 | closest-grid-count-mismatch | 0.800 | 0.300 | 0.389 |
| Optimal partitioning | 14 | exact-matched-k | 0.929 | 0.538 | 0.400 |
| Binary segmentation | 14 | exact-matched-k | 0.714 | 0.900 | 0.217 |
| Wild binary segmentation | 14 | exact-matched-k | 0.786 | 0.455 | 0.333 |

Dynp, binary segmentation, and WBS use the shared BayesBreak MAP boundary count. The predeclared eight-value PELT penalty grid did not attain 14 boundaries; its closest candidate returned 11 and is reported as a count mismatch without post-hoc retuning.

These are agreement diagnostics against a model-derived MAP reference, not external biological accuracy or evidence of predictive superiority. No independently verified external changepoint annotations are available.
