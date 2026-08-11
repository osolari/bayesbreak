# RES-BB-SYN-006 misspecification and negative-control suite

Protocol: `EPR-BB-015`. Code commit: `96464039e12b43207735835b004b0a59a9966b57`.
Scientific interpretation: **pending independent review**.

All 400 predeclared datasets were retained: 50 in each of eight cells. No top-level cell failed. All 50 EP fits reached the predeclared 20-second fit-only timeout; no EP diagnostics were imputed.

## Predeclared failure indicators

| Cell | Failure indicator | Rate | 95% interval |
|---|---|---:|---:|
| `null-gaussian` | `false_positive_dataset_rate` | 0.680 | 0.542 to 0.792 |
| `heavy-tail-gaussian` | `one_minus_complete_boundary_recovery_rate` | 0.500 | 0.366 to 0.634 |
| `zero-inflated-poisson` | `map_saturation_rate` | 1.000 | 0.929 to 1.000 |
| `dense-gaussian` | `map_saturation_rate` | 1.000 | 0.929 to 1.000 |
| `short-segment-gaussian` | `one_minus_complete_boundary_recovery_rate` | 0.920 | 0.812 to 0.968 |
| `prior-conflict-gaussian` | `missed_change_rate` | 1.000 | 1.000 to 1.000 |
| `shared-boundary-heterogeneity` | `subject_deviation_selected_rate` | 1.000 | 0.929 to 1.000 |
| `logistic-approximation-failure` | `ep_timeout_rate` | 1.000 | 0.929 to 1.000 |

## Key observations

- Null Gaussian false-positive dataset rate: 0.680.
- Zero-inflated Poisson and dense Gaussian MAP saturation rates: 1.000 and 1.000.
- Short-segment exact recovery rate: 0.080.
- Prior-conflict fits assigned zero posterior mass to unsupported segment counts and missed the truth-compatible boundaries in every dataset.
- Shared mean subject F1 was 0.491; mean independent F1 was 0.621.
- Mean empirical posterior TV was 0.016 for quadrature-40 and 0.031 for Laplace. Large maximum block errors and smaller posterior TV must be reported together.

These outcomes map the declared failure regimes. They are not evidence of universal robustness, model superiority, or external-truth accuracy. Acceptance for manuscript conclusions remains pending independent scientific review.
