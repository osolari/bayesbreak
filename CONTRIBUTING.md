# Contributing to BayesBreak

Thank you for your interest in contributing! The codebase is small and
organized around a two-layer design — block evidence (per family) + generic
DP (`bayesbreak.dp`) — so most changes touch one of those layers.

## Development setup

```bash
git clone https://github.com/osolari/bayesbreak.git
cd bayesbreak
bash setup_env.sh               # conda env "bayesbreak", Python 3.11
conda activate bayesbreak
pre-commit install
```

`setup_env.sh --venv` uses `python -m venv` instead of conda.

## Tooling

- **ruff** (`ruff check`, `ruff format`) — lint + format.
- **mypy** — type checks on `src/bayesbreak/`.
- **pytest** — test suite under `tests/`, including conceptual-correctness
  tests (brute-force DP, closed-form predictive checks, sklearn contract).
- **pre-commit** — enforces the above on every commit.

Run everything at once:

```bash
pre-commit run --all-files
mypy src/bayesbreak
pytest tests/
```

## Code style

- Type-hint public functions and classes; NumPyDoc docstrings.
- Follow the scikit-learn estimator contract: store constructor args untouched,
  validate inside `fit`, trailing-underscore fitted attributes.
- No backwards-compatibility shims. Breaking changes are documented in
  `CHANGELOG.md`.
- Tests must live alongside new functionality. Prefer conceptual-correctness
  tests (closed-form comparisons, brute-force DP on small `n`) over pure
  smoke tests.

## Pull request checklist

1. Create a feature branch.
2. Implement your change with tests.
3. `pre-commit run --all-files` passes.
4. `pytest tests/` passes (`-m "not network"` for offline runs).
5. Update `CHANGELOG.md`.
6. Open the PR with a short motivation and a test-plan section.

## Reporting issues

- GitHub Issues for bug reports and feature requests.
- Include a minimal reproducer (`X`, `y`, estimator call), Python version, and
  scikit-learn / numpy versions.
