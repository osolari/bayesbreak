# Array-CGH data card

| Field | Declaration |
|---|---|
| Dataset ID | `cgh` |
| Source | CRAN `ecp::ACGH`; source date not recorded |
| Observation family | Gaussian shared-boundary, subject-specific parameters |
| Coordinate axis | Probe index, length 2215 |
| Data shape | 43 subjects × 2215 probes for multi-subject analysis |
| Descriptor | Per-probe precision weights (see below) |
| External annotations | None independently verified |

## Precision weights

The single-subject CGH analysis uses per-probe precision weights derived
from rolling median absolute deviation (MAD) of log₂ ratio differences.
The resulting relative inverse-variance values are passed as `sample_weight`
to `BayesBreakGaussian`, which treats them as known per-observation precision
scale factors (not likelihood power weights).

These weights are data-derived rather than physically calibrated: the
rolling-MAD specification is a heuristic choice. Inference under a
uniform-precision or platform-calibrated noise model has not been compared.

## Limitations

- The pooled multi-subject analysis (S=43) was truncated at $k_{\max}=60$;
  the unrestricted MAP segment count is unknown.
- Fitted MAP boundaries are not validated against independent cytogenetic
  annotations.
- Multisequence comparators require the raw observation matrix and probe
  axis; fitted curves are not valid substitutes.
