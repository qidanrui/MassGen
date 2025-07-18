from __future__ import annotations

"""
Anthropic backend implementation.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any

from .base import LLMBackend, StreamChunk


class AnthropicBackend(LLMBackend):
    """Anthropic backend using the Claude API."""

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    async def stream_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        has_called_update_summary: bool = False,
        has_voted: bool = False,
        pending_notifications: int = 0,
        agent_ids: list[str] = None,
        provider_tools: list[str] | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using Anthropic API."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            # Convert tools to Anthropic format
            anthropic_tools = []
            for tool in tools:
                anthropic_tools.append(
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "input_schema": tool["function"]["parameters"],
                    }
                )

            # Extract max_tokens and temperature from kwargs
            max_tokens = kwargs.get("max_tokens", 1000)
            temperature = kwargs.get("temperature", 0.7)

            async with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                tools=anthropic_tools,
            ) as stream:
                content = ""
                tool_calls = []

                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            pass  # Ready to receive text
                        elif event.content_block.type == "tool_use":
                            tool_calls.append(
                                {
                                    "id": event.content_block.id,
                                    "type": "function",
                                    "name": event.content_block.name,
                                    "arguments": {},
                                }
                            )
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            content += event.delta.text
                            yield StreamChunk(type="content", content=event.delta.text)
                        elif event.delta.type == "input_json_delta":
                            # Tool input is being streamed
                            pass
                    elif event.type == "content_block_stop":
                        if event.content_block.type == "tool_use":
                            # Tool call is complete
                            for tc in tool_calls:
                                if tc["id"] == event.content_block.id:
                                    tc["arguments"] = event.content_block.input

                # If tool calls were made, yield them
                if tool_calls:
                    yield StreamChunk(
                        type="tool_calls", content=content, tool_calls=tool_calls
                    )
                else:
                    # No tool calls, LLM is done
                    yield StreamChunk(type="done")

        except Exception as e:
            yield StreamChunk(type="error", error=str(e))

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "Anthropic"

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimate: 4 characters per token for English text
        return len(text) // 4

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Calculate cost for Anthropic token usage (2024-2025 pricing).

        Pricing last updated: December 2024
        Source: https://www.anthropic.com/pricing
        """
        model_lower = model.lower()

        # Anthropic pricing (as of 2024-2025)
        if "claude-4" in model_lower:
            if "opus" in model_lower:
                # Claude-4 Opus
                input_cost = input_tokens * 0.015 / 1000  # $0.015 per 1K tokens
                output_cost = output_tokens * 0.075 / 1000  # $0.075 per 1K tokens
            elif "sonnet" in model_lower:
                # Claude-4 Sonnet
                input_cost = input_tokens * 0.003 / 1000  # $0.003 per 1K tokens
                output_cost = output_tokens * 0.015 / 1000  # $0.015 per 1K tokens
            else:
                # Default Claude-4 pricing (Sonnet)
                input_cost = input_tokens * 0.003 / 1000
                output_cost = output_tokens * 0.015 / 1000
        elif "claude-3" in model_lower:
            if "opus" in model_lower:
                # Claude-3 Opus
                input_cost = input_tokens * 0.015 / 1000  # $0.015 per 1K tokens
                output_cost = output_tokens * 0.075 / 1000  # $0.075 per 1K tokens
            elif "sonnet" in model_lower:
                if "3.5" in model_lower or "3.7" in model_lower:
                    # Claude-3.5/3.7 Sonnet
                    input_cost = input_tokens * 0.003 / 1000  # $0.003 per 1K tokens
                    output_cost = output_tokens * 0.015 / 1000  # $0.015 per 1K tokens
                else:
                    # Claude-3 Sonnet
                    input_cost = input_tokens * 0.003 / 1000  # $0.003 per 1K tokens
                    output_cost = output_tokens * 0.015 / 1000  # $0.015 per 1K tokens
            elif "haiku" in model_lower:
                if "3.5" in model_lower:
                    # Claude-3.5 Haiku
                    input_cost = input_tokens * 0.0008 / 1000  # $0.0008 per 1K tokens
                    output_cost = output_tokens * 0.004 / 1000  # $0.004 per 1K tokens
                else:
                    # Claude-3 Haiku
                    input_cost = input_tokens * 0.00025 / 1000  # $0.00025 per 1K tokens
                    output_cost = (
                        output_tokens * 0.00125 / 1000
                    )  # $0.00125 per 1K tokens
            else:
                # Default Claude-3 pricing (Sonnet)
                input_cost = input_tokens * 0.003 / 1000
                output_cost = output_tokens * 0.015 / 1000
        else:
            # Default Anthropic pricing (use Claude-3 Sonnet rates)
            input_cost = input_tokens * 0.003 / 1000
            output_cost = output_tokens * 0.015 / 1000

        return input_cost + output_cost

    def get_supported_builtin_tools(self) -> list[str]:
        """Get list of builtin tools supported by Anthropic."""
        return ["code_execution"]  # Anthropic supports code execution (beta)
