# Installation

BayesBreak runs on Python ≥ 3.10 (3.10, 3.11, 3.12 are tested).

## Minimal install

```bash
pip install bayesbreak
```

This pulls in `numpy`, `scipy`, and `scikit-learn` only. Everything in
`bayesbreak.dp`, `bayesbreak.families`, `bayesbreak.diagnostics`,
`bayesbreak.mixture`, `bayesbreak.replicates`, and `bayesbreak.prediction`
works under this minimal install.

## Optional extras

The package ships several `pip` extras, each with a focused purpose. You can
combine them, e.g. `pip install bayesbreak[plots,datasets,baselines]`.

| Extra | Pulls in | Use case |
|---|---|---|
| `plots` | `matplotlib`, `seaborn` | The figure-rendering scripts under `scripts/figures/` and the notebook tutorials. |
| `datasets` | `pooch`, `pandas` | Cached real-data loaders for the well-log, array-CGH, and methylation case studies. |
| `datasets-live` | `yfinance` | Live S&P 500 download path used by `load_spx()`. |
| `baselines` | `ruptures` | External-baseline wrappers for PELT, optimal partitioning, BS, WBS. |
| `baselines-r` | `rpy2` (plus a working R install) | R-side baselines: CBS via `DNAcopy`, SMUCE via `stepR`, RJMCMC-style MCMC via `mcp`. |
| `notebooks` | `jupyter`, `jupyterlab`, `ipykernel` | Run the tutorials locally. |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Build this site locally. |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `nbstripout`, `codespell` | Development workflow. |

## R-side dependencies

The `[baselines-r]` extra installs the Python `rpy2` bridge, but the R
packages themselves are not installed by pip. After `pip install
bayesbreak[baselines-r]`:

```r
# CBS via DNAcopy (Bioconductor):
if (!requireNamespace("BiocManager")) install.packages("BiocManager")
BiocManager::install("DNAcopy")

# SMUCE via stepR (CRAN):
install.packages("stepR", repos = "https://cloud.r-project.org")

# RJMCMC-style MCMC via mcp (CRAN; requires JAGS):
install.packages(c("mcp", "rjags"), repos = "https://cloud.r-project.org")
```

You also need the JAGS binary for `mcp`: `brew install jags` on macOS,
`apt install jags` on Debian/Ubuntu.

## Development install

```bash
git clone https://github.com/osolari/bayesbreak.git
cd bayesbreak
pip install -e ".[dev,plots,datasets,docs,notebooks]"
pre-commit install
pytest
```

The `pre-commit` hooks run `ruff`, `ruff-format`, `codespell`, `nbstripout`,
and a few other safety checks. They run automatically on `git commit`.

## Verifying the install

```python
import bayesbreak as bb

print(bb.__version__)
print(sorted(bb.__all__))
```

You should see a version string and a list of public surfaces including
`BayesBreakGaussian`, `BayesBreakMixtureClassifier`,
`SharedBoundaryReplicatesSegmenter`, `SlidingWindowSegmenter`,
`run_dp_diagnostics`, `run_non_conjugate_diagnostics`,
`run_prior_sensitivity`, and `select_n_groups_by_holdout`.
