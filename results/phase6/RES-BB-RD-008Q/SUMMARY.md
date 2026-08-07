# RES-BB-RD-008Q corrected methylation posterior prediction

Parent: `RES-BB-RD-007Q`. Protocol: `EPR-BB-012`.
Scientific execution commit: `314f46a62c56a12f96724fafecad8a2972f24af6`.

The exact hashed methylKit chromosome-21 source contains 1,904 ordered CpGs. Ten predeclared, disjoint, stratified interior blocks each hold out 152 CpGs while retaining both global fitted-support endpoints. The fitted family is `BayesBreakBetaObs`; training coverage is `phi`, held-out coverage is positive `phi_new`, and every prediction uses `extrapolation=error`.

Across 1,520 held-out CpGs, total log predictive score was -23605.675 and pooled mean score was -15.530. The mean of the ten split means was -15.530 (95% t interval -23.144 to -7.916). Mean boundary-stability F1@3 against the model-derived full-data MAP was 0.879 (95% t interval 0.810 to 0.947). All split fits selected 15 segments.

| Split | Test indices | Mean log predictive | MAP segments | Stability F1@3 |
|---:|---:|---:|---:|---:|
| 1 | 28:180 | -19.885 | 15 | 0.786 |
| 2 | 202:354 | -17.587 | 15 | 0.786 |
| 3 | 382:534 | -4.708 | 15 | 1.000 |
| 4 | 586:738 | -9.052 | 15 | 0.929 |
| 5 | 790:942 | -25.057 | 15 | 0.929 |
| 6 | 966:1118 | -0.611 | 15 | 1.000 |
| 7 | 1171:1323 | -2.175 | 15 | 0.857 |
| 8 | 1341:1493 | -28.881 | 15 | 0.714 |
| 9 | 1545:1697 | -19.499 | 15 | 0.857 |
| 10 | 1743:1895 | -27.847 | 15 | 0.929 |

This corrected result is not numerically comparable to the excluded parent score: the observation-family predictive distribution and split definition both changed. The ten blocks are regions of one chromosome, not independent biological samples. No certified Beta-observation PIT helper or external biological changepoint truth is available, so calibration and external accuracy are not reported.
