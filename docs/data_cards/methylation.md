# Methylation data card

| Field | Declaration |
|---|---|
| Dataset ID | `methylation` |
| Source | methylKit `test1.myCpG`, chromosome 21; source date not recorded |
| Observation family | Beta observation with latent segment mean |
| Coordinate axis | CpG genomic coordinate |
| Descriptor | Per-CpG coverage as known Beta precision `phi_t` |
| Executed case study | `n=1904`; descriptive segmentation |
| External annotations | None independently verified |

Coverage is an observation-family precision descriptor, not a likelihood-power
weight. Only family-correct, in-support held-out scores may support predictive conclusions.
and endpoint clipping; it remains excluded from predictive conclusions.
