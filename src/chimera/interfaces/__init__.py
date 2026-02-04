"""Model interface adapters for CHIMERA benchmark.

This module provides abstract interfaces and concrete implementations
for interacting with various LLM APIs:

- BaseModelInterface: Abstract base class defining the contract
- GeminiModel: Adapter for Google's Gemini API
- OpenAIModel: Adapter for OpenAI's API
- ConfidenceParser: Utilities for extracting confidence from responses
- ResponseParser: Utilities for parsing model responses
"""

from chimera.interfaces.base import (
    BaseModelInterface,
    ModelCapabilities,
    ModelConfig,
)
from chimera.interfaces.gemini import GeminiConfig, GeminiModel
from chimera.interfaces.openai import OpenAIConfig, OpenAIModel
from chimera.interfaces.parsers import (
    ConfidenceParser,
    ParserConfig,
    ResponseParser,
)

__all__ = [
    # Base interface
    "BaseModelInterface",
    "ModelConfig",
    "ModelCapabilities",
    # Gemini
    "GeminiModel",
    "GeminiConfig",
    # OpenAI
    "OpenAIModel",
    "OpenAIConfig",
    # Parsers
    "ConfidenceParser",
    "ResponseParser",
    "ParserConfig",
]
