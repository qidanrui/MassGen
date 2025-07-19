from __future__ import annotations

"""
OpenAI backend implementation.
"""

import os
from collections.abc import AsyncGenerator
from typing import Any, Union, Optional, List, Dict

from ..utils.api_tracer import trace_api_call
from .base import LLMBackend, StreamChunk


class OpenAIBackend(LLMBackend):
    """OpenAI backend using the Response API."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def stream_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        has_called_update_summary: bool = False,
        has_voted: bool = False,
        pending_notifications: int = 0,
        agent_ids: List[str] = None,
        provider_tools: Union[List[str], None] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using OpenAI Response API."""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            # Determine call type for tracing
            call_type = self._determine_call_type(
                messages, has_called_update_summary, has_voted, pending_notifications
            )

            # Filter provider tools to supported ones
            filtered_provider_tools = self.filter_provider_tools(provider_tools)

            # Prepare request data for tracing
            request_data = {
                "model": model,
                "input": messages,
                "tools": tools,
                "provider_tools": filtered_provider_tools,
                "stream": True,
                "max_tokens": kwargs.get("max_tokens"),
                "temperature": kwargs.get("temperature"),
            }

            # Trace the API call
            trace_api_call(
                provider="OpenAI",
                model=model,
                call_type=call_type,
                request_data=request_data,
                agent_id=kwargs.get("agent_id"),
                session_id=kwargs.get("session_id"),
            )

            # Use the Response API with streaming, tool calling, and configurable builtin tools
            api_params = {
                "model": model,
                "input": messages,
                "tools": tools,
                "stream": True,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["max_tokens", "temperature"]
                },
            }

            # Add provider tools if any are configured and supported
            if filtered_provider_tools:
                api_params["builtin_tools"] = filtered_provider_tools

            stream = await client.responses.create(**api_params)

            content = ""
            tool_calls = []

            async for chunk in stream:
                # Handle content streaming
                if hasattr(chunk, "content") and chunk.content:
                    content += chunk.content
                    yield StreamChunk(type="content", content=chunk.content)

                # Handle tool calls
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        tool_calls.append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "name": tc.function.name,
                                "arguments": self._safe_json_loads(
                                    tc.function.arguments
                                ),
                            }
                        )

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
        return "OpenAI"

    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by OpenAI."""
        return ["web_search", "code_interpreter"]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimate: 4 characters per token for English text
        return len(text) // 4

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Calculate cost for OpenAI token usage (2024-2025 pricing).

        Pricing last updated: December 2024
        Source: https://openai.com/api/pricing/
        """
        model_lower = model.lower()

        # OpenAI pricing (as of 2024-2025)
        if "gpt-4" in model_lower:
            if "turbo" in model_lower:
                # GPT-4 Turbo (128k context)
                input_cost = input_tokens * 0.01 / 1000  # $0.01 per 1K tokens
                output_cost = output_tokens * 0.03 / 1000  # $0.03 per 1K tokens
            elif "4o-mini" in model_lower:
                # GPT-4o-mini (much cheaper!)
                input_cost = input_tokens * 0.00015 / 1000  # $0.00015 per 1K tokens
                output_cost = output_tokens * 0.0006 / 1000  # $0.0006 per 1K tokens
            elif "4o" in model_lower:
                # GPT-4o
                input_cost = input_tokens * 0.005 / 1000  # $0.005 per 1K tokens
                output_cost = output_tokens * 0.020 / 1000  # $0.020 per 1K tokens
            elif "4.1" in model_lower:
                # GPT-4.1 (estimated similar to 4o)
                input_cost = input_tokens * 0.005 / 1000  # $0.005 per 1K tokens
                output_cost = output_tokens * 0.020 / 1000  # $0.020 per 1K tokens
            else:
                # GPT-4 (8k context)
                input_cost = input_tokens * 0.03 / 1000  # $0.03 per 1K tokens
                output_cost = output_tokens * 0.06 / 1000  # $0.06 per 1K tokens
        elif "gpt-3.5" in model_lower:
            # GPT-3.5 Turbo (current pricing)
            input_cost = input_tokens * 0.0005 / 1000  # $0.0005 per 1K tokens
            output_cost = output_tokens * 0.0015 / 1000  # $0.0015 per 1K tokens
        else:
            # Default OpenAI pricing (use GPT-3.5 rates)
            input_cost = input_tokens * 0.0005 / 1000
            output_cost = output_tokens * 0.0015 / 1000

        return input_cost + output_cost

    def _determine_call_type(
        self,
        messages: List[Dict[str, Any]],
        has_called_update_summary: bool,
        has_voted: bool,
        pending_notifications: int,
    ) -> str:
        """Determine the type of API call for tracing purposes."""
        if not messages:
            return "initial"

        last_message = messages[-1].get("content", "")

        if "You were selected as the representative" in last_message:
            return "final"
        elif "System instruction:" in last_message or pending_notifications > 0:
            return "restart"
        elif any("tool_calls" in msg for msg in messages if isinstance(msg, dict)):
            return "tool_response"
        else:
            return "initial"

    def _safe_json_loads(self, json_str):
        """Safely parse JSON string."""
        try:
            import json

            return json.loads(json_str)
        except:
            return {}
