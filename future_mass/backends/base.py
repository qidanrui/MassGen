from __future__ import annotations

"""
Base backend interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, AsyncGenerator, Optional
from dataclasses import dataclass


@dataclass
class StreamChunk:
    """Standardized chunk format for streaming responses."""
    type: str  # "content", "tool_calls", "done", "error"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


@dataclass
class TokenUsage:
    """Token usage and cost tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0


class LLMBackend(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.config = kwargs
        self.token_usage = TokenUsage()
    
    @abstractmethod
    async def stream_with_tools(self, model: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], 
                              **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a response with tool calling support.
        
        Args:
            model: Model identifier
            messages: Conversation messages
            tools: Available tools schema
            **kwargs: Additional provider-specific parameters
            
        Yields:
            StreamChunk: Standardized response chunks
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass
    
    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for token usage."""
        pass
    
    def update_token_usage(self, messages: List[Dict[str, Any]], response_content: str, model: str):
        """Update token usage tracking."""
        # Estimate input tokens from messages
        input_text = str(messages)
        input_tokens = self.estimate_tokens(input_text)
        
        # Estimate output tokens from response
        output_tokens = self.estimate_tokens(response_content)
        
        # Update totals
        self.token_usage.input_tokens += input_tokens
        self.token_usage.output_tokens += output_tokens
        
        # Calculate cost
        cost = self.calculate_cost(input_tokens, output_tokens, model)
        self.token_usage.estimated_cost += cost
    
    def get_token_usage(self) -> TokenUsage:
        """Get current token usage."""
        return self.token_usage
    
    def reset_token_usage(self):
        """Reset token usage tracking."""
        self.token_usage = TokenUsage()
    
    def get_supported_builtin_tools(self) -> List[str]:
        """Get list of builtin tools supported by this provider."""
        return []