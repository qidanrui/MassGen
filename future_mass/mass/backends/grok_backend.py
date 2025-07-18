from __future__ import annotations

"""
Grok/xAI backend implementation using OpenAI-compatible API.
"""

import os
from typing import Dict, List, Any, AsyncGenerator, Optional
from .base import LLMBackend, StreamChunk
from ..utils.api_tracer import trace_api_call


class GrokBackend(LLMBackend):
    """Grok backend using xAI's OpenAI-compatible API."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = "https://api.x.ai/v1"
    
    async def stream_with_tools(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                              has_called_update_summary: bool = False, has_voted: bool = False,
                              pending_notifications: int = 0, agent_ids: List[str] = None, 
                              provider_tools: Optional[List[str]] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using xAI's OpenAI-compatible API."""
        try:
            import openai
            
            # Use OpenAI client with xAI base URL
            client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Extract parameters
            max_tokens = kwargs.get("max_tokens", 1000)
            temperature = kwargs.get("temperature", 0.7)
            
            # Determine call type for tracing
            call_type = self._determine_call_type(messages, has_called_update_summary, has_voted, pending_notifications)
            
            # Filter provider tools to supported ones
            filtered_provider_tools = self.filter_provider_tools(provider_tools)
            
            # Prepare request data for tracing
            request_data = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "provider_tools": filtered_provider_tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "base_url": self.base_url
            }
            
            # Trace the API call
            trace_api_call(
                provider="Grok",
                model=model,
                call_type=call_type,
                request_data=request_data,
                agent_id=kwargs.get("agent_id"),
                session_id=kwargs.get("session_id")
            )
            
            # Use chat completions endpoint (xAI uses this, not Response API)
            api_params = {
                "model": model,
                "messages": messages,
                "tools": tools if tools else None,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            # Add provider tools if any are configured and supported  
            if filtered_provider_tools:
                api_params["builtin_tools"] = filtered_provider_tools
            
            stream = await client.chat.completions.create(**api_params)
            
            content = ""
            tool_calls = []
            
            async for chunk in stream:
                try:
                    # Handle content streaming
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        
                        # Handle content delta
                        if hasattr(choice, 'delta') and choice.delta:
                            if hasattr(choice.delta, 'content') and choice.delta.content:
                                content_chunk = choice.delta.content
                                content += content_chunk
                                yield StreamChunk(type="content", content=content_chunk)
                            
                            # Handle tool calls
                            if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                                for tool_call in choice.delta.tool_calls:
                                    if hasattr(tool_call, 'function'):
                                        # Convert to our standard format
                                        tool_calls.append({
                                            "id": getattr(tool_call, 'id', f"call_{len(tool_calls)}"),
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments
                                        })
                        
                        # Check for finish reason
                        if hasattr(choice, 'finish_reason') and choice.finish_reason:
                            if choice.finish_reason == "tool_calls" and tool_calls:
                                yield StreamChunk(type="tool_calls", tool_calls=tool_calls)
                            elif choice.finish_reason in ["stop", "length"]:
                                yield StreamChunk(type="done")
                            return
                
                except Exception as chunk_error:
                    yield StreamChunk(type="error", error=f"Chunk processing error: {chunk_error}")
                    continue
            
            # Signal completion if we exit the loop without explicit finish
            yield StreamChunk(type="done")
            
        except Exception as e:
            yield StreamChunk(type="error", error=f"Grok API error: {e}")
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "Grok"
    
    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by Grok."""
        return ["web_search"]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        return int(len(text.split()) * 1.3)  # Rough estimate
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        # xAI pricing (as of 2024) - approximate
        # Input: $5 per 1M tokens, Output: $15 per 1M tokens
        input_cost = (input_tokens / 1_000_000) * 5.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost
    
    def _determine_call_type(self, messages: List[Dict[str, Any]], has_called_update_summary: bool, 
                           has_voted: bool, pending_notifications: int) -> str:
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