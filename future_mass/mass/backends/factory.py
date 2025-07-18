"""
Backend factory for creating the appropriate LLM backend based on model name.
"""

import os

from .anthropic_backend import AnthropicBackend
from .base import LLMBackend
from .grok_backend import GrokBackend
from .mock_backend import MockBackend
from .openai_backend import OpenAIBackend


def create_backend(model_name: str, api_key: str | None = None) -> LLMBackend:
    """
    Create the appropriate backend based on model name and API availability.

    Args:
        model_name: Name of the model (e.g., "gpt-3.5-turbo", "claude-3-haiku-20240307")
        api_key: Optional API key override

    Returns:
        LLMBackend: The appropriate backend instance
    """
    model_lower = model_name.lower()

    # Check availability first
    backends = get_available_backends()

    # OpenAI models
    if any(keyword in model_lower for keyword in ["gpt", "openai"]):
        if backends["openai"]["available"] or api_key:
            return OpenAIBackend(api_key=api_key)
        else:
            print(
                f"Warning: OpenAI API not available for '{model_name}', using mock backend"
            )
            return MockBackend(api_key=api_key)

    # Anthropic models
    elif any(keyword in model_lower for keyword in ["claude", "anthropic"]):
        if backends["anthropic"]["available"] or api_key:
            return AnthropicBackend(api_key=api_key)
        else:
            print(
                f"Warning: Anthropic API not available for '{model_name}', using mock backend"
            )
            return MockBackend(api_key=api_key)

    # Grok/xAI models
    elif any(keyword in model_lower for keyword in ["grok", "xai"]):
        if backends["grok"]["available"] or api_key:
            return GrokBackend(api_key=api_key)
        else:
            print(
                f"Warning: Grok API not available for '{model_name}', using mock backend"
            )
            return MockBackend(api_key=api_key)

    # Mock models
    elif "mock" in model_lower:
        return MockBackend(api_key=api_key)

    # Default to mock for unknown models
    else:
        print(f"Warning: Unknown model '{model_name}', using mock backend")
        return MockBackend(api_key=api_key)


def get_available_backends() -> dict:
    """
    Get information about available backends and their API key status.

    Returns:
        dict: Backend availability information
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")

    return {
        "openai": {
            "available": bool(openai_key and not openai_key.startswith("your-")),
            "models": [
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
            ],
        },
        "anthropic": {
            "available": bool(anthropic_key and not anthropic_key.startswith("your-")),
            "models": [
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
            ],
        },
        "grok": {
            "available": bool(xai_key and not xai_key.startswith("your-")),
            "models": ["grok-beta", "grok-vision-beta"],
        },
        "mock": {
            "available": True,
            "models": ["mock-gpt", "mock-claude", "mock-grok", "mock-test"],
        },
    }
