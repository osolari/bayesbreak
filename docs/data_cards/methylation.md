# Methylation data card

| Field | Declaration |
|---|---|
| Dataset ID | `methylation` |
| Source | methylKit `test1.myCpG`, chromosome 21; source date not recorded |
| Observation family | Beta observation with latent segment mean (`BayesBreakBetaObs`) |
| Coordinate axis | CpG genomic coordinate |
| Descriptor | Per-CpG read coverage as known Beta precision `phi_t` |
| Executed case study | `n=1904`; descriptive segmentation |
| External annotations | None independently verified |

## Notes on descriptor semantics

Coverage is an observation-family precision descriptor, not a
likelihood-power weight. The Beta precision `phi_t` enters the
within-segment integrated likelihood, not as a generalized-Bayes
power-weighting of an otherwise specified likelihood.

## Limitations

Both methylation protocol cells (Dirichlet-process count prior and
uniform count prior) were **truncated at $k_{\max}=120$**. The unrestricted
MAP segment counts are unknown; sensitivity of the MAP segment count to
$k_{\max}$ was not characterized for this dataset.

Predictive scores computed on endpoint-clipped or out-of-support observations
remain excluded from predictive conclusions. Only family-correct, in-support
held-out scores may support predictive claims.
