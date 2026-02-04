"""Unit tests for CHIMERA model interfaces.

Tests cover:
- BaseModelInterface contract
- GeminiModel and OpenAIModel configuration
- ModelCapabilities
- Mock-based interface testing
"""

from unittest.mock import patch

import pytest

from chimera.interfaces.base import (
    BaseModelInterface,
    GenerationResult,
    ModelCapabilities,
    ModelConfig,
)
from chimera.interfaces.gemini import GeminiConfig, GeminiModel
from chimera.interfaces.openai import OpenAIConfig, OpenAIModel
from chimera.models.task import Task, TrackType


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ModelConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.top_p == 1.0
        assert config.max_retries == 3

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ModelConfig(
            model_name="custom-model",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.95,
            top_k=40,
        )
        assert config.model_name == "custom-model"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.top_k == 40

    def test_temperature_bounds(self) -> None:
        """Test temperature value bounds."""
        # Valid temperature
        config = ModelConfig(model_name="test", temperature=1.5)
        assert config.temperature == 1.5

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed."""
        config = ModelConfig(
            model_name="test",
            custom_field="custom_value",
        )
        assert config.model_extra.get("custom_field") == "custom_value"


class TestModelCapabilities:
    """Tests for ModelCapabilities."""

    def test_default_capabilities(self) -> None:
        """Test default capability values."""
        caps = ModelCapabilities()
        assert caps.supports_system_prompt is True
        assert caps.supports_streaming is True
        assert caps.max_context_length == 8192

    def test_custom_capabilities(self) -> None:
        """Test custom capability values."""
        caps = ModelCapabilities(
            supports_system_prompt=False,
            supports_vision=True,
            max_context_length=128_000,
        )
        assert caps.supports_system_prompt is False
        assert caps.supports_vision is True
        assert caps.max_context_length == 128_000


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_basic_result(self) -> None:
        """Test basic generation result."""
        result = GenerationResult(
            text="Hello, world!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=100.0,
        )
        assert result.text == "Hello, world!"
        assert result.finish_reason == "stop"
        assert result.total_tokens == 15

    def test_minimal_result(self) -> None:
        """Test minimal generation result."""
        result = GenerationResult(text="Response")
        assert result.text == "Response"
        assert result.finish_reason is None
        assert result.prompt_tokens == 0


class TestGeminiConfig:
    """Tests for GeminiConfig."""

    def test_default_config(self) -> None:
        """Test default Gemini configuration."""
        config = GeminiConfig()
        assert config.model_name == "gemini-2.0-flash"

    def test_custom_config(self) -> None:
        """Test custom Gemini configuration."""
        config = GeminiConfig(
            model_name="gemini-1.5-pro",
            api_key="test-key",
            temperature=0.5,
        )
        assert config.model_name == "gemini-1.5-pro"
        assert config.api_key == "test-key"

    def test_get_api_key_from_config(self) -> None:
        """Test getting API key from config."""
        config = GeminiConfig(api_key="my-api-key")
        assert config.get_api_key() == "my-api-key"

    def test_get_api_key_missing_raises(self) -> None:
        """Test that missing API key raises error."""
        config = GeminiConfig()
        # Clear environment
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="No API key"),
        ):
            config.get_api_key()

    def test_get_api_key_from_env(self) -> None:
        """Test getting API key from environment."""
        config = GeminiConfig()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}):
            assert config.get_api_key() == "env-key"


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_default_config(self) -> None:
        """Test default OpenAI configuration."""
        config = OpenAIConfig()
        assert config.model_name == "gpt-4o"

    def test_custom_config(self) -> None:
        """Test custom OpenAI configuration."""
        config = OpenAIConfig(
            model_name="gpt-4-turbo",
            api_key="test-key",
            organization="org-123",
        )
        assert config.model_name == "gpt-4-turbo"
        assert config.organization == "org-123"

    def test_get_api_key_from_config(self) -> None:
        """Test getting API key from config."""
        config = OpenAIConfig(api_key="my-api-key")
        assert config.get_api_key() == "my-api-key"

    def test_get_api_key_from_env(self) -> None:
        """Test getting API key from environment."""
        config = OpenAIConfig()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            assert config.get_api_key() == "env-key"


class TestGeminiModelCapabilities:
    """Tests for Gemini model capabilities detection."""

    def test_flash_capabilities(self) -> None:
        """Test capabilities for Gemini Flash models."""
        with (
            patch("google.generativeai.configure"),
            patch("google.generativeai.GenerativeModel"),
        ):
            config = GeminiConfig(
                model_name="gemini-2.0-flash",
                api_key="test-key",
            )
            try:
                model = GeminiModel(config)
                caps = model.get_capabilities()
                assert caps.supports_vision is True
                assert caps.max_context_length >= 1_000_000
            except ImportError:
                pytest.skip("google-generativeai not installed")

    def test_pro_capabilities(self) -> None:
        """Test capabilities for Gemini Pro models."""
        with (
            patch("google.generativeai.configure"),
            patch("google.generativeai.GenerativeModel"),
        ):
            config = GeminiConfig(
                model_name="gemini-1.5-pro",
                api_key="test-key",
            )
            try:
                model = GeminiModel(config)
                caps = model.get_capabilities()
                assert caps.supports_function_calling is True
            except ImportError:
                pytest.skip("google-generativeai not installed")


class TestOpenAIModelCapabilities:
    """Tests for OpenAI model capabilities detection."""

    def test_gpt4o_capabilities(self) -> None:
        """Test capabilities for GPT-4o."""
        with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
            config = OpenAIConfig(
                model_name="gpt-4o",
                api_key="test-key",
            )
            try:
                model = OpenAIModel(config)
                caps = model.get_capabilities()
                assert caps.supports_vision is True
                assert caps.supports_json_mode is True
                assert caps.max_context_length == 128_000
            except ImportError:
                pytest.skip("openai not installed")

    def test_gpt35_capabilities(self) -> None:
        """Test capabilities for GPT-3.5."""
        with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
            config = OpenAIConfig(
                model_name="gpt-3.5-turbo",
                api_key="test-key",
            )
            try:
                model = OpenAIModel(config)
                caps = model.get_capabilities()
                assert caps.supports_vision is False
                assert caps.max_context_length == 16_385
            except ImportError:
                pytest.skip("openai not installed")

    def test_o1_capabilities(self) -> None:
        """Test capabilities for o1 model."""
        with patch("openai.OpenAI"), patch("openai.AsyncOpenAI"):
            config = OpenAIConfig(
                model_name="o1",
                api_key="test-key",
            )
            try:
                model = OpenAIModel(config)
                caps = model.get_capabilities()
                # o1 has limited system prompt support
                assert caps.supports_system_prompt is False
            except ImportError:
                pytest.skip("openai not installed")


class MockModelInterface(BaseModelInterface):
    """Mock implementation of BaseModelInterface for testing."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.generate_calls: list[tuple[str, str | None]] = []
        self.mock_response = "Answer: Test\nConfidence: 80%"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        self.generate_calls.append((prompt, system_prompt))
        return GenerationResult(
            text=self.mock_response,
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=100.0,
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        return self.generate(prompt, system_prompt)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    def close(self) -> None:
        pass


class TestBaseModelInterface:
    """Tests for BaseModelInterface using mock implementation."""

    @pytest.fixture
    def mock_model(self) -> MockModelInterface:
        """Create a mock model interface."""
        config = ModelConfig(model_name="mock-model")
        return MockModelInterface(config)

    def test_generate_for_task(self, mock_model: MockModelInterface) -> None:
        """Test generate_for_task method."""
        task = Task(
            track=TrackType.CALIBRATION,
            question="What is 2+2?",
            correct_answer="4",
        )
        response = mock_model.generate_for_task(task)

        assert response.task_id == task.id
        assert response.raw_text == mock_model.mock_response
        assert len(mock_model.generate_calls) == 1

    def test_generate_batch(self, mock_model: MockModelInterface) -> None:
        """Test generate_batch method."""
        tasks = [
            Task(
                track=TrackType.CALIBRATION,
                question=f"Question {i}?",
                correct_answer=f"Answer {i}",
            )
            for i in range(3)
        ]
        responses = mock_model.generate_batch(tasks)

        assert len(responses) == 3
        assert len(mock_model.generate_calls) == 3

    def test_default_system_prompt(self, mock_model: MockModelInterface) -> None:
        """Test default system prompt generation."""
        prompt = mock_model.get_default_system_prompt()
        assert "confidence" in prompt.lower()
        assert "answer" in prompt.lower()

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        config = ModelConfig(model_name="test")
        with MockModelInterface(config) as model:
            result = model.generate("Test prompt")
            assert result.text is not None

    def test_validate_connection(self, mock_model: MockModelInterface) -> None:
        """Test connection validation."""
        mock_model.mock_response = "OK"
        assert mock_model.validate_connection() is True

        mock_model.mock_response = "Hello"
        assert mock_model.validate_connection() is False


class TestModelInterfaceIntegration:
    """Integration-style tests for model interfaces."""

    def test_task_to_response_flow(self) -> None:
        """Test complete flow from task to parsed response."""
        config = ModelConfig(model_name="test")
        model = MockModelInterface(config)
        model.mock_response = """
        Answer: Paris

        Confidence: 95%

        Reasoning: Paris is the capital city of France and has been since
        the late 10th century.
        """

        task = Task(
            track=TrackType.CALIBRATION,
            question="What is the capital of France?",
            correct_answer="Paris",
        )

        response = model.generate_for_task(task)

        # Verify response structure
        assert response.task_id == task.id
        assert response.parsed_answer is not None
        assert "Paris" in response.parsed_answer.raw_answer
        assert abs(response.confidence.numeric - 0.95) < 0.05
        assert response.metadata is not None
        assert response.metadata.model_name == "test"
