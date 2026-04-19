# Reproducibility

## CLI

After installing, use the `bayesbreak` command:

```bash
bayesbreak reproduce figures   # scripts/figures/*.py  → docs/report/figures or docs/report/tables
bayesbreak reproduce tables    # scripts/tables/*.py   → docs/report/figures or docs/report/tables
bayesbreak reproduce all
bayesbreak version
```

## Figures

Each figure is driven by one script under [`scripts/figures/`](../scripts/figures):

| Script | Report figure |
|---|---|
| `fig1_synthetic_gaussian.py` | Synthetic Gaussian segmentation + boundary posterior |
| `fig2_family_showcase.py` | Gaussian / Poisson / Binomial / Beta showcase |
| `fig3_boundary_calibration.py` | Boundary-event calibration (ECE) |
| `fig4_latent_groups.py` | Latent-group EM responsibilities |
| `fig5_runtime_scaling.py` | Runtime scaling over `(n, k_max)` |
| `fig6_mixture_discovery.py` | Mixture-template discovery |
| `fig7_snr_sensitivity.py` | SNR sensitivity curves |
| `fig8_multivariate_shared.py` | Shared-boundary multivariate recovery |
| `fig9_model_selection.py` | Model-selection demo |
| `fig10_missing_data.py` | Missing-data handling demo |

Each script writes both `.png` and `.pdf` to `docs/report/figures/`.

## Tables

`scripts/tables/`:

| Script | Report table |
|---|---|
| `table0_metrics_overview.py` | Metrics definitions (§6) |
| `table1_runtime_scaling.py` | Runtime vs `(n, k_max)` |
| `table2_posterior_summary.py` | Posterior summary at fixed `n` |
| `table3_conjugate_summary.py` | Per-family F1 / MAE / evidence |
| `table4_nonconj_tradeoff.py` | Non-conjugate Laplace / JJ / EP / quadrature tradeoff |

Outputs land in `docs/report/tables/` as `.md`, `.tex`, and `.csv`.

## Reproducibility checklist

1. `bash create_env.sh`
2. `pip install -e ".[dev,plots,docs,notebooks]"`
3. `pytest tests/` (96+ tests, conceptual-correctness + sklearn contract)
4. `bayesbreak reproduce all`
5. `cd docs/report && latexmk -pdf bayesbreak.tex` (optional LaTeX build)
