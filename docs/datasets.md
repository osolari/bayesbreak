# Real-data loaders

`bayesbreak.datasets` ships four loaders used by the real-data figures in the
report (fig6–fig9). Every loader returns a `DatasetBundle` with the same
schema — fits directly into the sklearn `fit(X, y)` contract:

```python
from bayesbreak import BayesBreakGaussian
from bayesbreak.datasets import load_welllog

bundle = load_welllog()            # tries real download, falls back to simulation
est = BayesBreakGaussian(k_max=20).fit(bundle.X, bundle.y,
                                       sample_weight=bundle.sample_weight)
print(bundle.source, est.map_boundaries_)
```

## Available loaders

| Loader | Real source | Fallback |
|---|---|---|
| `load_welllog()` | TCPD mirror of the Ó Ruanaidh NMR well-log (n=4050) via `pooch` | piecewise-Gaussian simulation with 10 true boundaries |
| `load_cgh(csv_path=...)` | User-provided CSV of log2 ratios | simulated CGH with amplifications + deletion |
| `load_spx()` | Daily `^GSPC` close via `yfinance`; log-squared returns | GARCH-like regime simulation |
| `load_methylation(csv_path=...)` | User-provided CSV of methylation fractions | Beta-response plateau simulation |

Two of the four pull from the network; `load_cgh` and `load_methylation` accept
a local CSV, since we do not redistribute a public mirror.

## `DatasetBundle` fields

```python
@dataclass
class DatasetBundle:
    X: np.ndarray                     # (n, 1) design matrix
    y: np.ndarray                     # (n,) response
    sample_weight: np.ndarray | None  # optional per-observation weight
    true_boundaries: list[int]        # ground truth (always for simulated)
    name: str                         # "welllog", "cgh", "spx", "methylation"
    source: str                       # "downloaded" or "simulated"
    description: str
    metadata: dict
```

`bundle.is_simulated` is a convenience for the figure scripts.

## Caching & offline policy

- Cache directory defaults to `~/.cache/bayesbreak`; override with
  `$BAYESBREAK_DATA`.
- `pooch` must be installed for real downloads (ships in the
  `bayesbreak[datasets]` extra). `yfinance` for live S&P prices ships in
  `bayesbreak[datasets-live]`.
- When the dependency is missing, the download fails, or parsing fails, the
  loader prints a one-line `[bayesbreak.datasets] <name>: falling back …`
  banner and returns the deterministic simulated analog. This keeps the test
  suite and CI reproducible offline.
- Pass `simulated=True` to force the fallback (useful for tests and
  deterministic reports).

## Example: running real-data experiments

```bash
pip install "bayesbreak[datasets,datasets-live]"
bayesbreak reproduce figures     # figures 6–9 use real data when available
```

Each real-data figure follows the same four-panel template:

1. raw `y` + MAP piecewise-constant fit (and Bayes curve when enabled),
2. boundary-event marginals `P(b_i = 1 | y)`,
3. segment-count posterior `P(k | y)`,
4. cumulative held-out posterior-predictive log-density vs a `k = 1` null.

The figure caption tags whether the panel was produced from real data or the
simulated fallback so the provenance is always obvious.
