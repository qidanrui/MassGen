MODEL_MAPPINGS = {
    "openai": [
        # GPT-4 variants
        "gpt-4o",  # -> gpt-4o-2024-08-06
        "gpt-4o-2024-11-20",
        # GPT-4 Mini variants
        "gpt-4o-mini",  # -> gpt-4o-mini-2024-07-18
        # o1
        "o1",  # -> o1-2024-12-17
        # o3
        "o3",
        "o3-low",
        "o3-medium",
        "o3-high",
        # o3 mini
        "o3-mini",
        "o3-mini-low",
        "o3-mini-medium",
        "o3-mini-high",
        # o4 mini
        "o4-mini",
        "o4-mini-low",
        "o4-mini-medium",
        "o4-mini-high",
        # gpt-4.1
        "gpt-4.1-mini",
        "gpt-4.1",
    ],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro"],
    "grok": [
        "grok-3",
        "grok-4",
    ],
}


def get_agent_type_from_model(model_name: str) -> str:
    """
    Get the agent type from a model name.

    Args:
        model_name: The specific model name (e.g., "gpt-4o", "gemini-2.5-flash")

    Returns:
        The agent type ("openai", "gemini", "grok")

    Raises:
        ValueError: If the model name is not found
    """
    for agent_type, models in MODEL_MAPPINGS.items():
        if model_name in models:
            return agent_type

    # Create a flat list of all available models for error message
    all_models = []
    for models in MODEL_MAPPINGS.values():
        all_models.extend(models)

    raise ValueError(f"Unknown model '{model_name}'. Available models: {', '.join(all_models)}")


def get_available_models() -> list:
    """Get a flat list of all available model names."""
    all_models = []
    for models in MODEL_MAPPINGS.values():
        all_models.extend(models)
    return all_models
