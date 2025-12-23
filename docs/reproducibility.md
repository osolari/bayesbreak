# Reproducibility

This repository includes simple, script-based generators for the figures and
tables referenced in the documentation.

## Figures

All figure scripts live under `scripts/figures/` and save outputs to `results/`.

- `fig1_synthetic_gaussian.py`: synthetic Gaussian segmentation + posterior
  boundary probabilities.
- `fig2_family_showcase.py`: short showcase of Gaussian/Poisson/Binomial/Beta
  families on family-appropriate synthetic data.

Run all figure scripts:

```bash
python scripts/make_all_figures.py
```

## Tables

All table scripts live under `scripts/tables/` and save outputs to `results/`.

- `table1_runtime_scaling.py`: runtime scaling over `(n, k_max)`.

Run all table scripts:

```bash
python scripts/make_all_tables.py
```
