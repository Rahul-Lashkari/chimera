# Contributing to CHIMERA

Thank you for your interest in contributing to CHIMERA! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   ```bash
   git clone https://github.com/Rahul-Lashkari/chimera.git
   cd chimera
   ```

3. **Add the upstream remote**:

   ```bash
   git remote add upstream https://github.com/Rahul-Lashkari/chimera.git
   ```

## Development Setup

1. **Create a virtual environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install development dependencies**:

   ```bash
   make install-dev
   ```

3. **Set up pre-commit hooks**:

   ```bash
   pre-commit install
   ```

4. **Verify installation**:

   ```bash
   make test
   ```

## Making Changes

1. **Create a new branch** for your feature or bugfix:

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```

2. **Make your changes**, following the [style guidelines](#style-guidelines)

3. **Add tests** for any new functionality

4. **Run the test suite**:

   ```bash
   make check
   ```

5. **Commit your changes** with a clear, descriptive message:

   ```bash
   git commit -m "feat: add calibration binning strategy for ECE computation"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/modifications
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

## Pull Request Process

1. **Update your branch** with the latest upstream changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your branch** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - A clear title following conventional commits
   - A description of what changes were made and why
   - Reference to any related issues

4. **Address review feedback** by pushing additional commits

5. **Squash and merge** once approved

## Style Guidelines

### Python Code

- **Formatter**: Black with 100 character line length
- **Import sorting**: isort with black profile
- **Linting**: Ruff
- **Type hints**: Required for all public functions

```python
def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        confidences: Model confidence scores (0-1).
        accuracies: Binary correctness indicators.
        n_bins: Number of bins for calibration.

    Returns:
        Expected Calibration Error score.

    Raises:
        ValueError: If inputs have mismatched lengths.
    """
    ...
```

### Docstrings

We use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """Short description of function.

    Longer description if needed, explaining the function's
    behavior in more detail.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When invalid arguments are provided.

    Example:
        >>> function_name("hello", 42)
        True
    """
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run without coverage (faster)
make test-fast

# Run specific test file
pytest tests/test_metrics/test_calibration.py -v

# Run tests matching a pattern
pytest -k "calibration" -v
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use descriptive test names: `test_ece_returns_zero_for_perfect_calibration`
- Use fixtures from `conftest.py` where appropriate
- Mark slow tests with `@pytest.mark.slow`

```python
import pytest
from chimera.metrics.calibration import compute_ece

class TestComputeECE:
    def test_returns_zero_for_perfect_calibration(self):
        """ECE should be 0 when confidence matches accuracy exactly."""
        confidences = np.array([0.0, 0.5, 1.0])
        accuracies = np.array([0, 0, 1])  # Matches confidence
        assert compute_ece(confidences, accuracies) == pytest.approx(0.0)

    def test_raises_on_mismatched_lengths(self):
        """Should raise ValueError for mismatched input lengths."""
        with pytest.raises(ValueError, match="mismatched lengths"):
            compute_ece(np.array([0.5]), np.array([0, 1]))
```

## Documentation

### Building Docs

```bash
make docs       # Build documentation
make serve-docs # Serve locally at http://localhost:8000
```

### Writing Docs

- API documentation is auto-generated from docstrings
- Add conceptual guides in `docs/concepts/`
- Add tutorials in `docs/tutorials/`
- Use Markdown with MkDocs-Material syntax

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Reach out to maintainers for guidance

Thank you for contributing to CHIMERA! 🎉
