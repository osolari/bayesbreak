# Array-CGH data card

| Field | Declaration |
|---|---|
| Dataset ID | `cgh` |
| Source | CRAN `ecp::ACGH`; source date not recorded |
| Observation family | Gaussian shared-boundary, subject-specific parameters |
| Coordinate axis | Probe index, length 2215 |
| Data shape | 43 subjects by 2215 probes for comparator requests |
| Descriptor | Per-probe/subject Gaussian precision |
| External annotations | None independently verified |

Multisequence comparators require the raw matrix and probe axis. Fitted curves are not valid substitutes for raw observations in external comparator evaluations. Comparator studies must use the original observation matrix and matching coordinate axis.
