# Baselines

`bayesbreak.baselines` wraps upstream changepoint libraries so they can be
called through a uniform Python API. **We do not re-implement any of these
algorithms** — each wrapper drives the canonical upstream package and
normalises its output into a `BaselineResult`.

This is the §6 planned-external-comparator suite of the manuscript.

## Available algorithms

| Name | Aliases | Upstream | Reference |
|---|---|---|---|
| `pelt` | — | `ruptures.Pelt` | Killick, Fearnhead & Eckley (2012) |
| `optimal_partitioning` | `op`, `dynp` | `ruptures.Dynp` | Jackson et al. (2005) |
| `binary_segmentation` | `bs`, `binseg` | `ruptures.Binseg` | classical BS |
| `wild_binary_segmentation` | `wbs` | `ruptures.Binseg` + random windows | Fryzlewicz (2014) |
| `cbs` | — | `DNAcopy::segment` via `rpy2` | Olshen et al. (2004) |
| `smuce` | — | `stepR::stepFit` via `rpy2` | Frick, Munk & Sieling (2014) |
| `rjmcmc` | `mcp` | `mcp::mcp` via `rpy2` + JAGS | Lindeløv (2020) |
| `fearnhead_exact` | `fearnhead` | `bayesbreak.dp` at Fearnhead-2006 prior config (labelled reference) | Fearnhead (2006) |

## Usage

```python
from bayesbreak.baselines import segment_with

result = segment_with("pelt", y, penalty=10.0)
result.boundaries           # interior changepoint indices
result.k                    # number of segments
result.package              # "ruptures"
result.package_version      # the installed ruptures version
result.tuning               # full tuning dict — reproducible
```

Each wrapper returns a `BaselineResult` dataclass:

```python
@dataclass
class BaselineResult:
    algorithm: str
    package: str
    package_version: str
    n: int
    boundaries: np.ndarray
    tuning: dict
    extra: dict
```

## Install extras

| Extra | What you get |
|---|---|
| `pip install bayesbreak[baselines]` | `ruptures` (covers PELT, optimal partitioning, BS, WBS). |
| `pip install bayesbreak[baselines-r]` | `rpy2`. You also need an R install with `DNAcopy`, `stepR`, `mcp`, `rjags` — and the JAGS binary for `mcp`. |

Missing upstream deps raise a single readable `ImportError` pointing to
the right install command:

```text
ImportError: ruptures is required for the PELT / Dynp / Binseg / WBS
wrappers; install with `pip install bayesbreak[baselines]` or `pip
install ruptures`.
```

## Notes per algorithm

### `pelt`

PELT is the standard exact-DP-with-pruning baseline. The expected linear
cost holds under the changepoint-density condition; worst case stays
`O(n^2)`. We pass through `penalty`, `cost_model`, `min_size`, `jump`
unchanged to `ruptures.Pelt.predict`.

### `optimal_partitioning` (`op`, `dynp`)

Fixed-`k` optimal partitioning. Pass `n_bkps` to set the segment count.
Useful when you want to compare against BayesBreak's fixed-`k` solution
at a known `k`.

### `binary_segmentation` (`bs`)

Classical binary segmentation. Pass either `n_bkps` (fixed count) or
`penalty` (BIC-style stopping). Cheaper than PELT, less accurate on
closely-spaced changepoints — useful as a baseline for "would BS have
caught this?".

### `wild_binary_segmentation` (`wbs`)

WBS adds random-window sampling on top of BS to escape its known weakness
on multiple close changepoints. We sample windows uniformly, fit BS in
each, take the union of candidate boundaries, then rescore on the full
signal. `n_random_windows` controls the sampling intensity;
`random_state` seeds the RNG.

### `cbs`

Circular Binary Segmentation via Bioconductor `DNAcopy`. The canonical
array-CGH baseline. Inputs are log-2 copy-number ratios; optional
`chromosome` and `position` columns route through `DNAcopy::CNA`.
`alpha`, `nperm`, `undo_splits`, `smooth` map to the corresponding
`DNAcopy::segment` arguments.

### `smuce`

Multi-scale changepoint inference (Frick, Munk & Sieling 2014) via R
`stepR`. Modern releases expose `stepFit`; older ones expose `smuceR`
— the wrapper falls back automatically. `alpha` controls the
confidence level; `family` chooses the noise model.

### `rjmcmc` (`mcp`)

Bayesian-MCMC multi-changepoint fit via R `mcp` (Lindeløv 2020) with a
JAGS backend. The wrapper fits the intercept-only segment formula
`[y ~ 1, ~ 1, …, ~ 1]` at a user-specified `n_segments`, then returns
the posterior-mean changepoint locations.

`mcp` is not strictly Green-1995 RJMCMC (it fixes `n_segments` per fit);
trans-dimensional selection is done externally via `mcp::loo`. For the
§5b RJMCMC slot, fit at several candidate `n_segments` and compare by
held-out predictive log-likelihood — the same scoring rule used by
`bayesbreak.select_n_groups_by_holdout`.

### `fearnhead_exact` (`fearnhead`)

There is no widely-distributed standalone Fearnhead-2006 implementation
on PyPI or CRAN. The cleanest reproducible reference comparator is
BayesBreak's own DP at the Fearnhead-2006 prior choice (geometric
`p(k)` plus optional length-aware cohesion). The wrapper drives
`bayesbreak.make_bayesbreak` with that configuration and clearly
labels the returned `BaselineResult.package` as
`"bayesbreak (fearnhead2006 config)"` so downstream tables flag the
provenance honestly.

If you need a genuinely third-party Fearnhead-2006 implementation, the
options are: (i) compile Paul Fearnhead's original Fortran/MATLAB code
(not packaged); (ii) `R changepoint::SegNeigh`; (iii) the JSFdS 2015
pruned-DP code referenced in `rigaill2010pruned`.
