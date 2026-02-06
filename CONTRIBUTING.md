# Contributing to BayesBreak

Thank you for your interest in contributing to BayesBreak! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/bayesbreak.git
cd bayesbreak
```

### 2. Set Up Development Environment

```bash
cd env
bash create_env.sh
conda activate bayesbreak
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run before each commit.

## Development Workflow

### Code Style

We use:
- **Black** for code formatting (100 character line length)
- **Ruff** for linting
- **MyPy** for type checking

These are configured in `pyproject.toml` and enforced via pre-commit hooks.

Format code before committing:

```bash
black src/ tests/
ruff check --fix src/ tests/
```

### Testing

Run tests with coverage:

```bash
pytest --cov=src/bayesbreak tests/
```

Ensure new code has corresponding tests.

### Type Hints

Add type hints to new functions and classes:

```python
def compute_segment_stats(
    y: np.ndarray, sample_weight: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Compute segment statistics.

    Parameters
    ----------
    y : np.ndarray
        Observations
    sample_weight : Optional[np.ndarray]
        Per-observation weights

    Returns
    -------
    Tuple[float, float]
        Mean and variance
    """
```

### Documentation

- Add docstrings in NumPy format to all public functions and classes
- Update relevant documentation in `docs/` when adding features
- Add examples to docstrings for key functions

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with tests
3. Run tests and code checks: `pytest` and `pre-commit run --all-files`
4. Commit with clear messages
5. Push to your fork and open a pull request
6. Respond to feedback

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproducible examples when possible
- Specify Python version and environment details

## Questions?

Feel free to open a discussion or issue for questions.
