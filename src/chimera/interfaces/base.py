"""Base model interface for CHIMERA benchmark.

This module defines the abstract interface that all model adapters must implement,
providing a consistent API for interacting with different LLMs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chimera.models.response import ModelResponse, ResponseMetadata
from chimera.models.task import Task


class ModelConfig(BaseModel):
    """Base configuration for model interfaces.

    Attributes:
        model_name: The model identifier (e.g., "gemini-2.0-flash")
        api_key: API key for authentication (if not using environment variable)
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter (if supported)
        timeout_seconds: Request timeout
        max_retries: Maximum retry attempts on failure
        retry_delay_seconds: Delay between retries
    """

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(description="Model identifier")
    api_key: str | None = Field(
        default=None,
        description="API key (uses environment variable if not provided)",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum response tokens",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter",
    )
    timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Delay between retries",
    )


@dataclass
class ModelCapabilities:
    """Describes what a model can do.

    Attributes:
        supports_system_prompt: Whether model accepts system prompts
        supports_streaming: Whether model supports streaming responses
        supports_json_mode: Whether model can output structured JSON
        supports_function_calling: Whether model supports tool/function calls
        max_context_length: Maximum context window size
        supports_vision: Whether model can process images
    """

    supports_system_prompt: bool = True
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_function_calling: bool = False
    max_context_length: int = 8192
    supports_vision: bool = False


@dataclass
class GenerationResult:
    """Raw result from model generation.

    Attributes:
        text: The generated text
        finish_reason: Why generation stopped
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        total_tokens: Total tokens used
        latency_ms: Generation latency in milliseconds
        raw_response: The raw API response object
    """

    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_response: Any = field(default=None, repr=False)


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces.

    All model adapters (Gemini, OpenAI, etc.) must implement this interface
    to work with the CHIMERA benchmark.

    Example:
        >>> class MyModel(BaseModelInterface):
        ...     def generate(self, prompt: str) -> GenerationResult:
        ...         # Implementation here
        ...         pass
        ...
        >>> model = MyModel(config)
        >>> result = model.generate("What is 2+2?")
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model interface.

        Args:
            config: Model configuration.
        """
        self.config = config
        self._client: Any = None

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Generate a response for a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            GenerationResult containing the response.
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Asynchronously generate a response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            GenerationResult containing the response.
        """
        pass

    def generate_for_task(
        self,
        task: Task,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Generate a response for a CHIMERA task.

        This is the main method for benchmark evaluation. It handles
        prompt construction, response generation, and parsing.

        Args:
            task: The CHIMERA task to respond to.
            system_prompt: Optional system prompt override.

        Returns:
            Parsed ModelResponse with confidence and answer.
        """
        from chimera.interfaces.parsers import ResponseParser

        # Use default system prompt if not provided
        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()

        # Generate response
        result = self.generate(task.question, system_prompt)

        # Parse the response
        parser = ResponseParser()
        response = parser.parse(
            task_id=task.id,
            raw_text=result.text,
            metadata=ResponseMetadata(
                model_name=self.config.model_name,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                temperature=self.config.temperature,
            ),
        )

        return response

    async def generate_for_task_async(
        self,
        task: Task,
        system_prompt: str | None = None,
    ) -> ModelResponse:
        """Asynchronously generate a response for a CHIMERA task.

        Args:
            task: The CHIMERA task to respond to.
            system_prompt: Optional system prompt override.

        Returns:
            Parsed ModelResponse with confidence and answer.
        """
        from chimera.interfaces.parsers import ResponseParser

        if system_prompt is None:
            system_prompt = self.get_default_system_prompt()

        result = await self.generate_async(task.question, system_prompt)

        parser = ResponseParser()
        response = parser.parse(
            task_id=task.id,
            raw_text=result.text,
            metadata=ResponseMetadata(
                model_name=self.config.model_name,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
                temperature=self.config.temperature,
            ),
        )

        return response

    def generate_batch(
        self,
        tasks: list[Task],
        system_prompt: str | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for multiple tasks.

        Args:
            tasks: List of tasks to process.
            system_prompt: Optional system prompt for all tasks.

        Returns:
            List of ModelResponse objects.
        """
        responses = []
        for task in tasks:
            response = self.generate_for_task(task, system_prompt)
            responses.append(response)
        return responses

    async def generate_batch_async(
        self,
        tasks: list[Task],
        system_prompt: str | None = None,
        max_concurrent: int = 5,
    ) -> list[ModelResponse]:
        """Asynchronously generate responses for multiple tasks.

        Args:
            tasks: List of tasks to process.
            system_prompt: Optional system prompt for all tasks.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of ModelResponse objects.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(task: Task) -> ModelResponse:
            async with semaphore:
                return await self.generate_for_task_async(task, system_prompt)

        responses = await asyncio.gather(*[process_with_semaphore(task) for task in tasks])
        return list(responses)

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Get the capabilities of this model.

        Returns:
            ModelCapabilities describing what the model can do.
        """
        pass

    def get_default_system_prompt(self) -> str:
        """Get the default system prompt for CHIMERA evaluation.

        Returns:
            System prompt string.
        """
        return """You are a helpful AI assistant participating in an evaluation.
When answering questions:
1. Provide your answer clearly
2. Express your confidence level as a percentage (0-100%)
3. If you're unsure, it's okay to say so
4. Be honest about what you know and don't know

Format your response as:
Answer: [your answer]
Confidence: [X]%
Reasoning: [brief explanation if helpful]"""

    def validate_connection(self) -> bool:
        """Validate that the model connection is working.

        Returns:
            True if connection is valid, False otherwise.
        """
        try:
            result = self.generate("Hello, please respond with 'OK'.")
            return "ok" in result.text.lower()
        except Exception:
            return False

    @abstractmethod
    def close(self) -> None:
        """Close the model connection and clean up resources."""
        pass

    def __enter__(self) -> "BaseModelInterface":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
