"""Pytest configuration and shared fixtures for tests."""

import os
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Path Fixtures


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory."""
    return project_root / "configs"


# Environment Fixtures


@pytest.fixture(autouse=True)
def clean_environment() -> Generator[None, None, None]:
    """Ensure clean environment for each test."""
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_api_key() -> str:
    """Return a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def env_with_api_keys(mock_api_key: str) -> Generator[None, None, None]:
    """Set up environment with mock API keys."""
    os.environ["GOOGLE_API_KEY"] = mock_api_key
    os.environ["OPENAI_API_KEY"] = mock_api_key
    yield


# Mock Fixtures


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Return a mock LLM response structure."""
    return {
        "text": "The answer is 42.",
        "confidence": 0.85,
        "reasoning": "Based on my analysis...",
        "tokens_used": 150,
        "latency_ms": 245,
    }


@pytest.fixture
def mock_model() -> MagicMock:
    """Return a mock LLM model."""
    model = MagicMock()
    model.name = "mock-model"
    model.generate.return_value = {
        "text": "Mock response",
        "confidence": 0.75,
    }
    return model


# Sample Data Fixtures


@pytest.fixture
def sample_calibration_task() -> dict[str, Any]:
    """Return a sample calibration task."""
    return {
        "id": "cal-001",
        "question": "What is the capital of France?",
        "correct_answer": "Paris",
        "difficulty": "easy",
        "category": "geography",
        "metadata": {
            "source": "synthetic",
            "verified": True,
        },
    }


@pytest.fixture
def sample_error_detection_task() -> dict[str, Any]:
    """Return a sample error detection task."""
    return {
        "id": "err-001",
        "question": "Solve: 15 + 27 = ?",
        "initial_response": "The answer is 41.",  # Wrong answer
        "correct_answer": "42",
        "error_type": "arithmetic",
        "error_location": "final_answer",
    }


@pytest.fixture
def sample_boundary_task() -> dict[str, Any]:
    """Return a sample knowledge boundary task."""
    return {
        "id": "bound-001",
        "question": "What was the exact temperature in London at 3:47 PM on March 15, 1847?",
        "is_answerable": False,
        "expected_behavior": "abstain",
        "category": "unanswerable_specific",
    }


@pytest.fixture
def sample_confidence_scores() -> list[float]:
    """Return sample confidence scores for calibration testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.fixture
def sample_accuracy_scores() -> list[int]:
    """Return sample accuracy scores (binary) for calibration testing."""
    return [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]


# Configuration Fixtures


@pytest.fixture
def default_config() -> dict[str, Any]:
    """Return default benchmark configuration."""
    return {
        "benchmark": {
            "name": "chimera-test",
            "version": "0.1.0",
        },
        "tracks": {
            "calibration": {"enabled": True, "max_samples": 100},
            "error_detection": {"enabled": True, "max_samples": 100},
            "knowledge_boundary": {"enabled": True, "max_samples": 100},
            "self_correction": {"enabled": True, "max_samples": 100},
        },
        "model": {
            "name": "gemini-2.0-flash",
            "temperature": 0.0,
            "max_tokens": 1024,
        },
        "evaluation": {
            "n_bins": 15,
            "bootstrap_samples": 1000,
        },
        "output": {
            "results_dir": "results",
            "save_raw_responses": True,
        },
    }


# Pytest Configuration


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "requires_api: mark test as requiring API access")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add unit marker to tests in test_* directories
        if "test_" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Skip API tests unless explicitly enabled
        if "requires_api" in item.keywords and not config.getoption(
            "--run-api-tests", default=False
        ):
            item.add_marker(pytest.mark.skip(reason="API tests disabled (use --run-api-tests)"))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API access",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
