.PHONY: help install install-dev test lint format type-check clean docs serve-docs build publish

help:
	@echo "CHIMERA Benchmark Development Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install package in production mode"
	@echo "  make install-dev   Install package with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests with coverage"
	@echo "  make test-fast     Run tests without coverage"
	@echo "  make lint          Run all linters"
	@echo "  make format        Format code with black and isort"
	@echo "  make type-check    Run mypy type checking"
	@echo "  make check         Run all checks (lint + type-check + test)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Build documentation"
	@echo "  make serve-docs    Serve documentation locally"
	@echo ""
	@echo "Build & Publish:"
	@echo "  make build         Build distribution packages"
	@echo "  make publish       Publish to PyPI (requires credentials)"
	@echo "  make clean         Remove build artifacts"
	@echo ""
	@echo "Benchmark:"
	@echo "  make run           Run benchmark with default config"
	@echo "  make run-quick     Run quick benchmark (subset of tasks)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=chimera --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v --no-cov

test-unit:
	pytest tests/ -v -m "unit" --no-cov

test-integration:
	pytest tests/ -v -m "integration" --no-cov

# Linting and formatting
lint:
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/chimera

# Run all checks
check: lint type-check test

# Documentation
docs:
	mkdocs build

serve-docs:
	mkdocs serve

# Build
build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Benchmark commands
run:
	python -m chimera.cli.main run --config configs/default.yaml

run-quick:
	python -m chimera.cli.main run --config configs/default.yaml --max-samples 10

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Version bump helpers
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major
