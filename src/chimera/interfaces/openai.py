"""OpenAI model interface for CHIMERA benchmark.

This module provides an adapter for OpenAI's API (GPT-4, etc.),
handling authentication, request formatting, and response parsing.
"""

import os
import time
from typing import Any

from pydantic import Field

from chimera.interfaces.base import (
    BaseModelInterface,
    GenerationResult,
    ModelCapabilities,
    ModelConfig,
)


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI model interface.

    Attributes:
        model_name: OpenAI model identifier
        api_key: OpenAI API key
        organization: Optional organization ID
        base_url: Optional custom API base URL
        response_format: Optional response format (e.g., {"type": "json_object"})
    """

    model_name: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier",
    )
    organization: str | None = Field(
        default=None,
        description="OpenAI organization ID",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL",
    )
    response_format: dict[str, str] | None = Field(
        default=None,
        description="Response format specification",
    )

    def get_api_key(self) -> str:
        """Get the API key from config or environment.

        Returns:
            The API key.

        Raises:
            ValueError: If no API key is available.
        """
        if self.api_key:
            return self.api_key

        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            return env_key

        raise ValueError(
            "No API key provided. Set OPENAI_API_KEY environment variable, "
            "or pass api_key in config."
        )


class OpenAIModel(BaseModelInterface):
    """Adapter for OpenAI's API.

    This class provides a consistent interface for interacting with
    OpenAI models (GPT-4, GPT-4o, etc.) within the CHIMERA benchmark.

    Example:
        >>> config = OpenAIConfig(model_name="gpt-4o")
        >>> model = OpenAIModel(config)
        >>> result = model.generate("What is 2+2?")
        >>> print(result.text)
    """

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        """Initialize the OpenAI model interface.

        Args:
            config: OpenAI configuration. Uses defaults if None.
        """
        if config is None:
            config = OpenAIConfig()

        super().__init__(config)
        self.config: OpenAIConfig = config
        self._client: Any = None
        self._async_client: Any = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI, OpenAI

            api_key = self.config.get_api_key()

            client_kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
            }

            if self.config.organization:
                client_kwargs["organization"] = self.config.organization
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self._client = OpenAI(**client_kwargs)
            self._async_client = AsyncOpenAI(**client_kwargs)

        except ImportError as err:
            raise ImportError(
                "openai package is required. " "Install with: pip install openai"
            ) from err

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Generate a response using OpenAI.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            GenerationResult containing the response.
        """
        start_time = time.perf_counter()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }

            if self.config.response_format:
                request_kwargs["response_format"] = self.config.response_format

            # Make the request
            response = self._client.chat.completions.create(**request_kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract response data
            text = ""
            if response.choices:
                text = response.choices[0].message.content or ""

            finish_reason = None
            if response.choices:
                finish_reason = response.choices[0].finish_reason

            prompt_tokens = 0
            completion_tokens = 0
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                text=f"Error: {str(e)}",
                finish_reason="error",
                latency_ms=latency_ms,
            )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Asynchronously generate a response using OpenAI.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            GenerationResult containing the response.
        """
        start_time = time.perf_counter()

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            request_kwargs: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }

            if self.config.response_format:
                request_kwargs["response_format"] = self.config.response_format

            response = await self._async_client.chat.completions.create(**request_kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000

            text = ""
            if response.choices:
                text = response.choices[0].message.content or ""

            finish_reason = None
            if response.choices:
                finish_reason = response.choices[0].finish_reason

            prompt_tokens = 0
            completion_tokens = 0
            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            return GenerationResult(
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return GenerationResult(
                text=f"Error: {str(e)}",
                finish_reason="error",
                latency_ms=latency_ms,
            )

    def get_capabilities(self) -> ModelCapabilities:
        """Get OpenAI model capabilities.

        Returns:
            ModelCapabilities for this OpenAI model.
        """
        model_name = self.config.model_name.lower()

        # GPT-4o and variants
        if "gpt-4o" in model_name:
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_function_calling=True,
                max_context_length=128_000,
                supports_vision=True,
            )
        # GPT-4 Turbo
        elif "gpt-4-turbo" in model_name or "gpt-4-1106" in model_name:
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_function_calling=True,
                max_context_length=128_000,
                supports_vision="vision" in model_name,
            )
        # GPT-4
        elif "gpt-4" in model_name:
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_function_calling=True,
                max_context_length=8_192,
                supports_vision=False,
            )
        # GPT-3.5
        elif "gpt-3.5" in model_name:
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_function_calling=True,
                max_context_length=16_385,
                supports_vision=False,
            )
        # o1 models
        elif "o1" in model_name:
            return ModelCapabilities(
                supports_system_prompt=False,  # o1 doesn't support system prompts the same way
                supports_streaming=False,
                supports_json_mode=False,
                supports_function_calling=False,
                max_context_length=128_000,
                supports_vision="o1" in model_name,  # o1 supports vision
            )
        else:
            # Default capabilities
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_function_calling=False,
                max_context_length=4_096,
                supports_vision=False,
            )

    def close(self) -> None:
        """Close the OpenAI connection."""
        if self._client:
            self._client.close()
        if self._async_client:
            # Async client cleanup
            pass
        self._client = None
        self._async_client = None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.config.model_name,
            "provider": "openai",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "capabilities": self.get_capabilities().__dict__,
        }

    def list_available_models(self) -> list[str]:
        """List available OpenAI models.

        Returns:
            List of model IDs.
        """
        try:
            models = self._client.models.list()
            return [model.id for model in models.data]
        except Exception:
            return []
