"""Agent configuration for MASS framework."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .message_templates import MessageTemplates


@dataclass
class AgentConfig:
    """Configuration for MassAgent with backend and framework settings.

    Args:
        backend_params: Settings passed directly to LLM provider (model, api_key, etc.)
        provider_tools: Provider tools to enable (None=all, []=none, [list]=specific)
        user_tools: Custom user-provided tools (collision-protected)
        message_templates: Custom message templates (None=default)
    """

    backend_params: dict[str, Any] = field(default_factory=dict)
    provider_tools: list[str] | None = None
    user_tools: list[dict[str, Any]] = field(default_factory=list)
    message_templates: Optional["MessageTemplates"] = None

    @classmethod
    def create_gpt_config(
        cls,
        model: str = "gpt-4o-mini",
        provider_tools: list[str] | None = None,
        **kwargs,
    ) -> "AgentConfig":
        """Create OpenAI GPT configuration."""
        return cls(
            backend_params={"model": model, **kwargs}, provider_tools=provider_tools
        )

    @classmethod
    def create_claude_config(
        cls,
        model: str = "claude-3-sonnet-20240229",
        provider_tools: list[str] | None = None,
        **kwargs,
    ) -> "AgentConfig":
        """Create Anthropic Claude configuration."""
        return cls(
            backend_params={"model": model, **kwargs}, provider_tools=provider_tools
        )

    @classmethod
    def create_grok_config(
        cls,
        model: str = "grok-beta",
        provider_tools: list[str] | None = None,
        **kwargs,
    ) -> "AgentConfig":
        """Create Grok configuration."""
        return cls(
            backend_params={"model": model, **kwargs}, provider_tools=provider_tools
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "backend_params": self.backend_params,
            "provider_tools": self.provider_tools,
            "user_tools": self.user_tools,
        }

        # Handle message_templates serialization
        if self.message_templates is not None:
            try:
                if hasattr(self.message_templates, "_template_overrides"):
                    overrides = self.message_templates._template_overrides
                    if all(not callable(v) for v in overrides.values()):
                        result["message_templates"] = overrides
                    else:
                        result["message_templates"] = "<contains_callable_functions>"
                else:
                    result["message_templates"] = "<custom_message_templates>"
            except (AttributeError, TypeError):
                result["message_templates"] = "<non_serializable>"

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create from dictionary (for deserialization)."""
        # Extract basic fields
        backend_params = data.get("backend_params", {})
        provider_tools = data.get("provider_tools")
        user_tools = data.get("user_tools", [])

        # Handle message_templates
        message_templates = None
        template_data = data.get("message_templates")
        if isinstance(template_data, dict):
            from .message_templates import MessageTemplates

            message_templates = MessageTemplates(**template_data)

        return cls(
            backend_params=backend_params,
            provider_tools=provider_tools,
            user_tools=user_tools,
            message_templates=message_templates,
        )
