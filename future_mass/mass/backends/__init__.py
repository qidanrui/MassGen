"""
MASS Backends - LLM provider abstractions for the Multi-Agent Scaling System.

This module provides a unified interface for different LLM providers (OpenAI, Anthropic, Mock)
while keeping the MassAgent class clean and focused on collaboration logic.
"""

from .base import LLMBackend, StreamChunk
from .openai_backend import OpenAIBackend
from .anthropic_backend import AnthropicBackend
from .mock_backend import MockBackend

__all__ = [
    "LLMBackend", 
    "StreamChunk",
    "OpenAIBackend", 
    "AnthropicBackend", 
    "MockBackend"
]