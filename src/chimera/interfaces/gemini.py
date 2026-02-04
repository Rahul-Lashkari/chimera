"""Google Gemini model interface for CHIMERA benchmark.

This module provides an adapter for Google's Gemini API,
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


class GeminiConfig(ModelConfig):
    """Configuration for Gemini model interface.

    Attributes:
        model_name: Gemini model identifier
        api_key: Google AI API key
        safety_settings: Optional safety filter settings
        generation_config: Additional generation parameters
    """

    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model identifier",
    )
    safety_settings: dict[str, str] | None = Field(
        default=None,
        description="Safety filter settings",
    )
    generation_config: dict[str, Any] | None = Field(
        default=None,
        description="Additional generation parameters",
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

        env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if env_key:
            return env_key

        raise ValueError(
            "No API key provided. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
            "or pass api_key in config."
        )


class GeminiModel(BaseModelInterface):
    """Adapter for Google's Gemini API.

    This class provides a consistent interface for interacting with
    Gemini models within the CHIMERA benchmark.

    Example:
        >>> config = GeminiConfig(model_name="gemini-2.0-flash")
        >>> model = GeminiModel(config)
        >>> result = model.generate("What is 2+2?")
        >>> print(result.text)
    """

    def __init__(self, config: GeminiConfig | None = None) -> None:
        """Initialize the Gemini model interface.

        Args:
            config: Gemini configuration. Uses defaults if None.
        """
        if config is None:
            config = GeminiConfig()

        super().__init__(config)
        self.config: GeminiConfig = config
        self._client: Any = None
        self._model: Any = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai

            api_key = self.config.get_api_key()
            genai.configure(api_key=api_key)

            # Build generation config
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            if self.config.top_k is not None:
                generation_config["top_k"] = self.config.top_k

            if self.config.generation_config:
                generation_config.update(self.config.generation_config)

            # Initialize the model
            self._model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,  # type: ignore[arg-type]
                safety_settings=self.config.safety_settings,  # type: ignore[arg-type]
            )
            self._client = genai

        except ImportError as err:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            ) from err

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Generate a response using Gemini.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            GenerationResult containing the response.
        """
        start_time = time.perf_counter()

        try:
            # Combine system prompt with user prompt if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Generate response
            response = self._model.generate_content(full_prompt)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract text
            text = ""
            if response.text:
                text = response.text

            # Extract token counts if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            # Get finish reason
            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

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
            # Return error as text for now
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
        """Asynchronously generate a response using Gemini.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.

        Returns:
            GenerationResult containing the response.
        """
        start_time = time.perf_counter()

        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Use async generation
            response = await self._model.generate_content_async(full_prompt)

            latency_ms = (time.perf_counter() - start_time) * 1000

            text = ""
            if response.text:
                text = response.text

            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

            finish_reason = None
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)

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
        """Get Gemini model capabilities.

        Returns:
            ModelCapabilities for this Gemini model.
        """
        # Capabilities vary by model
        model_name = self.config.model_name.lower()

        if "flash" in model_name:
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_function_calling=True,
                max_context_length=1_000_000,  # 1M context for Gemini 2.0
                supports_vision=True,
            )
        elif "pro" in model_name:
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_function_calling=True,
                max_context_length=2_000_000,  # 2M context for Gemini Pro
                supports_vision=True,
            )
        else:
            # Default capabilities
            return ModelCapabilities(
                supports_system_prompt=True,
                supports_streaming=True,
                supports_json_mode=False,
                supports_function_calling=False,
                max_context_length=32_000,
                supports_vision=False,
            )

    def close(self) -> None:
        """Close the Gemini connection."""
        self._model = None
        self._client = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.
        """
        if self._model is None:
            return 0

        try:
            result = self._model.count_tokens(text)
            return int(result.total_tokens)
        except Exception:
            # Rough estimation if counting fails
            return len(text) // 4

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.config.model_name,
            "provider": "google",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "capabilities": self.get_capabilities().__dict__,
        }
