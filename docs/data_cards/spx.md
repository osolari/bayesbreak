# SPX data card

| Field | Declaration |
|---|---|
| Dataset ID | `spx` |
| Source | Yahoo Finance `^GSPC`, 2015-01-01 through 2023-12-31 |
| Response | Daily log-squared return |
| Observation family | Gaussian for the primary fit; Bernoulli for the declared threshold fit |
| Coordinate axis | Trading-day index |
| Archived case-study stride | 4 (`n=566`) |
| External annotations | None; event overlays are contextual only |

Evidence values from different transformed responses or observation families
are not directly comparable as Bayes factors.
