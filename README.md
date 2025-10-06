# BayesBreak: Hierarchical Bayes Process Segmentation of Mixtures of Random Variables

This directory holds reproducible Conda environment definitions.

## Typical workflow

```bash
# (1) Update lock files and sync pyproject.toml
bash env/pin.sh

# (2) Create environment from the platform-specific lock
bash env/create_env.sh

# (3) Run tests
bash scripts/pytest.sh