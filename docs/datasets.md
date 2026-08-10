# Real-data loaders

`bayesbreak.datasets.load_with_provenance(dataset_id, config)` returns the
normalized `DatasetBundle` together with a versioned `DatasetCard`. The card
hashes the post-preprocessing coordinate/response arrays and any family
descriptor, records source kind/date/URI, stride, coordinate semantics, family,
and sequence count, and keeps external annotations separate from model-derived
MAP markers. See the [data cards](data_cards/index.md) for the four application
sources and their unresolved provenance limitations.

`bayesbreak.datasets` ships four loaders, one per real-data case study
in §6 of the manuscript. Every loader returns a `DatasetBundle` with a
common schema and falls back to a deterministic simulated analog when
the upstream download is unavailable — the test suite and CI stay
reproducible offline.

```python
from bayesbreak import BayesBreakGaussian
from bayesbreak.datasets import load_welllog

bundle = load_welllog()                    # tries real download; falls back to sim
est = BayesBreakGaussian(k_max=20).fit(
    bundle.X, bundle.y, sample_weight=bundle.sample_weight
)
print(bundle.source, est.k_map_, est.map_boundaries_[:5])
```

## Available loaders

| Loader | Upstream | Block model | §6 case study |
|---|---|---|---|
| `load_welllog()` | Ó Ruanaidh well-log NMR series (4050 points; bundled by R `changepoint.influence::welldata`) | Gaussian with known variance | Well-log geology |
| `load_cgh()` | Coriell array-CGH cell-line panel (Snijders et al. 2001; via `DNAcopy` or a CRAN `ecp` mirror) | Heteroscedastic Gaussian, multi-subject pooled | Array-CGH copy number |
| `load_spx()` | S&P 500 daily closes via `yfinance` → $\log r_t^2$ | Gaussian with known variance | Equity-return volatility regimes |
| `load_methylation()` | Loyfer 2023 CpG methylation atlas (companion code at `nloyfer/wgbs_tools` / `nloyfer/UXM_deconv`; GEO `GSE186458`); local fallback uses the `methylKit` chr21 test region | Beta-response with per-CpG precision $\phi_t$ | CpG-atlas methylation |

!!! note "Author-verification caveat"
    Two loader docstrings carry verified-fact notes (May 2026): (i) the
    well-log `welldata` object lives in R `changepoint.influence`, **not**
    in the `Lai2005fig4` slot of the older `changepoint` package
    (`Lai2005fig4` is an unrelated array-CGH dataset); (ii) the Loyfer
    2023 methylation atlas is at `nloyfer/wgbs_tools` +
    `nloyfer/UXM_deconv`, **not** the older `nloyfer/meth_atlas`
    (which implements the Moss 2018 array deconvolution).
    See the relevant `datasets/welllog.py` and `datasets/methylation.py`
    docstrings.

## `DatasetBundle` fields

```python
@dataclass
class DatasetBundle:
    X: np.ndarray                     # (n, 1) design matrix
    y: np.ndarray                     # (n,)  response or (n, S) for multi-subject CGH
    sample_weight: np.ndarray | None  # optional per-observation weight
    true_boundaries: list[int]        # ground truth (populated only for simulated bundles)
    name: str                         # "welllog", "cgh", "spx", "methylation"
    source: str                       # "downloaded" or "simulated"
    description: str
    metadata: dict
```

`bundle.is_simulated` is the convenience predicate the figure scripts
check before rendering provenance badges.

## Caching and offline policy

- Cache directory defaults to `~/.cache/bayesbreak`; override via
  `$BAYESBREAK_DATA`.
- Real downloads require `pip install "bayesbreak[datasets]"`
  (adds `pooch`, `pandas`). Live S&P prices require
  `pip install "bayesbreak[datasets-live]"` (adds `yfinance`).
- When the dependency is missing, the download fails, or parsing fails,
  the loader prints a one-line banner
  `[bayesbreak.datasets] <name>: falling back to simulated analog (...)`
  and returns the deterministic simulated bundle. The simulated bundles
  have known `true_boundaries`, which is what the test suite checks.
- Pass `simulated=True` to force the fallback path (used by tests and
  CI to keep figure regressions hermetic).

## Headline numbers from the real-data fits

Archived in `docs/manuscript/shared/figures/results/realdata_metrics.json`; see
[Results](results.md) for interpretation and exclusions.

| Case study | Fit | Outcome |
|---|---|---|
| Well-log NMR (stride-8, $n=507$) | `BayesBreakGaussian(k_max=40)` | $\widehat k = 23$, $\log p(y) = -4989.28$ |
| Coriell array-CGH (43 subjects, $n_{\mathrm{probes}}=2215$) | `SharedBoundaryReplicatesSegmenter(BayesBreakGaussian(k_max=15))` | $\widehat k = 15$, pooled $\log p(y) = 76\,359.8$ |
| S&P 500 (stride-4, $n=566$) | `BayesBreakGaussian(k_max=50)` | $\widehat k = 29$, $\log p(y) = -1296.65$ |
| Methylation (chr21 region, $n=1904$) | `BayesBreakBetaObs(k_max=15, phi=coverage)` | $\widehat k = 15$; historical predictive score excluded |

The methylation segmentation remains a real result. The historical held-out score
`RES-BB-RD-007Q` is excluded from posterior-predictive conclusions because it used a
Gaussian predictive calculation for Beta observations and an implicit endpoint rule.
