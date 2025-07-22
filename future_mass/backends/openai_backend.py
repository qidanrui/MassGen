from __future__ import annotations

"""
OpenAI backend implementation using Response API.
Standalone implementation optimized for OpenAI's Response API format.
"""

import os
from typing import Dict, List, Any, AsyncGenerator, Optional
from .base import LLMBackend, StreamChunk


class OpenAIBackend(LLMBackend):
    """OpenAI backend using the Response API."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    def convert_tools_to_response_api_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools from Chat Completions format to Response API format if needed.
        
        Chat Completions format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        Response API format: {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        if not tools:
            return tools
            
        converted_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                # Chat Completions format - convert to Response API format
                func = tool["function"]
                converted_tools.append({
                    "type": "function",
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func.get("parameters", {})
                })
            else:
                # Already in Response API format or non-function tool
                converted_tools.append(tool)
        
        return converted_tools
    
    async def stream_with_tools(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                              **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream response using OpenAI Response API."""
        try:
            import openai
            
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Extract provider tool settings
            enable_web_search = kwargs.get("enable_web_search", False)
            enable_code_interpreter = kwargs.get("enable_code_interpreter", False)
            
            # Response API parameters (uses 'input', not 'messages')
            api_params = {
                "model": model,
                "input": messages,
                "stream": True
            }
                
            # Add max_output_tokens if specified (o-series models don't support this)
            max_tokens = kwargs.get("max_tokens")
            if max_tokens and not model.startswith("o"):
                api_params["max_output_tokens"] = max_tokens
            
            # Add framework tools (convert to Response API format)
            if tools:
                converted_tools = self.convert_tools_to_response_api_format(tools)
                api_params["tools"] = converted_tools
            
            # Add provider tools (web search, code interpreter) if enabled
            provider_tools = []
            if enable_web_search:
                provider_tools.append({"type": "web_search"})
            
            if enable_code_interpreter:
                provider_tools.append({
                    "type": "code_interpreter",
                    "container": {"type": "auto"}
                })
            
            if provider_tools:
                if "tools" not in api_params:
                    api_params["tools"] = []
                api_params["tools"].extend(provider_tools)
            
            stream = await client.responses.create(**api_params)
            
            content = ""
            tool_calls = []
            
            async for chunk in stream:
                # Handle Responses API streaming format
                if hasattr(chunk, 'type'):
                    if chunk.type == 'response.output_text.delta' and hasattr(chunk, 'delta'):
                        content += chunk.delta
                        yield StreamChunk(type="content", content=chunk.delta)
                    elif chunk.type == 'response.web_search_call.in_progress':
                        yield StreamChunk(type="content", content=f"\n🔍 [Provider Tool: Web Search] Starting search...\n")
                    elif chunk.type == 'response.web_search_call.searching':
                        yield StreamChunk(type="content", content=f"🔍 [Provider Tool: Web Search] Searching...\n")
                    elif chunk.type == 'response.web_search_call.completed':
                        yield StreamChunk(type="content", content=f"✅ [Provider Tool: Web Search] Search completed\n")
                    elif chunk.type == 'response.code_interpreter_call.in_progress':
                        yield StreamChunk(type="content", content=f"\n💻 [Provider Tool: Code Interpreter] Starting execution...\n")
                    elif chunk.type == 'response.code_interpreter_call.executing':
                        yield StreamChunk(type="content", content=f"💻 [Provider Tool: Code Interpreter] Executing...\n")
                    elif chunk.type == 'response.code_interpreter_call.completed':
                        yield StreamChunk(type="content", content=f"✅ [Provider Tool: Code Interpreter] Execution completed\n")
                    elif chunk.type == 'response.output_item.done':
                        # Get search query or executed code details
                        if hasattr(chunk, 'item') and chunk.item:
                            if hasattr(chunk.item, 'type') and chunk.item.type == 'web_search_call':
                                if hasattr(chunk.item, 'action') and hasattr(chunk.item.action, 'query'):
                                    search_query = chunk.item.action.query
                                    if search_query:
                                        yield StreamChunk(type="content", content=f"🔍 [Search Query] '{search_query}'\n")
                            elif hasattr(chunk.item, 'type') and chunk.item.type == 'code_interpreter_call':
                                if hasattr(chunk.item, 'code') and chunk.item.code:
                                    code_snippet = chunk.item.code[:100] + "..." if len(chunk.item.code) > 100 else chunk.item.code
                                    yield StreamChunk(type="content", content=f"💻 [Code Executed] {code_snippet}\n")
                    elif chunk.type == 'response.completed':
                        # Check for tool calls in final response
                        if hasattr(chunk, 'response') and hasattr(chunk.response, 'output'):
                            for output_item in chunk.response.output:
                                if hasattr(output_item, 'name') and hasattr(output_item, 'arguments'):
                                    # Parse arguments if string
                                    arguments = output_item.arguments
                                    if isinstance(arguments, str):
                                        try:
                                            import json
                                            arguments = json.loads(arguments)
                                        except:
                                            arguments = {}
                                    
                                    tool_calls.append({
                                        "id": getattr(output_item, 'id', f"call_{len(tool_calls)}"),
                                        "name": output_item.name,
                                        "arguments": arguments
                                    })
                        
                        # Yield tool calls if found
                        if tool_calls:
                            yield StreamChunk(type="tool_calls", tool_calls=tool_calls)
            
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
        return len(text) // 4
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for OpenAI token usage (2024-2025 pricing)."""
        model_lower = model.lower()
        
        if "gpt-4" in model_lower:
            if "4o-mini" in model_lower:
                input_cost = input_tokens * 0.00015 / 1000
                output_cost = output_tokens * 0.0006 / 1000
            elif "4o" in model_lower:
                input_cost = input_tokens * 0.005 / 1000
                output_cost = output_tokens * 0.020 / 1000
            else:
                input_cost = input_tokens * 0.03 / 1000
                output_cost = output_tokens * 0.06 / 1000
        elif "gpt-3.5" in model_lower:
            input_cost = input_tokens * 0.0005 / 1000
            output_cost = output_tokens * 0.0015 / 1000
        else:
            input_cost = input_tokens * 0.0005 / 1000
            output_cost = output_tokens * 0.0015 / 1000
        
        return input_cost + output_cost