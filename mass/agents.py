"""
MassAgent implementations that wrap the existing agent backends.

This module provides MassAgent-compatible wrappers for the existing
OpenAI, Gemini, and Grok agent implementations.
"""

import os
import sys
from typing import Callable, Union, Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

# Add agents directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from .agent import AgentResponse, MassAgent

# Try to import function_to_json, but make it optional
try:
   from .utils import function_to_json
except ImportError:
    # Fallback if agents.util is not available
    def function_to_json(func):
        return {
            "type": "function",
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }


class OpenAIMassAgent(MassAgent):
    """MassAgent wrapper for OpenAI agent implementation."""

    def __init__(
        self, 
        agent_id: int, 
        orchestrator=None, 
        model: str = "o4-mini", 
        tools: List[str] = None,
        max_retries: int = 3,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = None,
        processing_timeout: float = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
        **kwargs
    ):
        # Pass all configuration to parent
        super().__init__(
            agent_id=agent_id,
            orchestrator=orchestrator,
            model=model,
            tools=tools,
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            processing_timeout=processing_timeout,
            stream=stream,
            stream_callback=stream_callback,
            **kwargs
        )
        self._client = None  # Store client for cleanup

        # Import the OpenAI process_message function
        try:
            from .backends.oai import process_message as oai_process_message

            self._process_message_impl = oai_process_message
        except ImportError:
            self._process_message_impl = None

    def cleanup(self):
        """Clean up HTTP client resources."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._client = None

    def process_message(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AgentResponse:
        """Process message using OpenAI backend."""
        if self._process_message_impl is None:
            return AgentResponse(
                text="OpenAI agent implementation not available (missing dependencies)",
                code=[],
                citations=[],
                function_calls=[],
            )

        # Merge instance configuration with any overrides from kwargs
        # Use the exact parameter names expected by the backend
        config = {
            "model": self.model,
            "tools": self.tools,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_key": None,  # Let backend use environment variable
            "processing_timeout": self.processing_timeout,
            "stream": self.stream,
            "stream_callback": self.stream_callback,
            **self.kwargs,  # Instance kwargs
            **kwargs,  # Override kwargs
        }
        
        # Handle parameter name mapping for overrides
        if 'timeout' in config:
            config['processing_timeout'] = config.pop('timeout')
        
        # Remove None values to avoid overriding backend defaults
        config = {k: v for k, v in config.items() if v is not None}

        try:
            result = self._process_message_impl(
                messages=messages,
                **config
            )

            return AgentResponse(
                text=result.get("text", ""),
                code=result.get("code", []),
                citations=result.get("citations", []),
                function_calls=result.get("function_calls", []),
            )
        except Exception as e:
            # Return error response
            return AgentResponse(
                text=f"Error in OpenAI agent processing: {str(e)}",
                code=[],
                citations=[],
                function_calls=[],
            )


class GeminiMassAgent(MassAgent):
    """MassAgent wrapper for Gemini agent implementation."""

    def __init__(
        self,
        agent_id: int,
        orchestrator=None,
        model: str = "gemini-2.5-flash",
        tools: List[str] = None,
        max_retries: int = 3,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = None,
        processing_timeout: float = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
        **kwargs,
    ):
        # Pass all configuration to parent
        super().__init__(
            agent_id=agent_id,
            orchestrator=orchestrator,
            model=model,
            tools=tools,
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            processing_timeout=processing_timeout,
            stream=stream,
            stream_callback=stream_callback,
            **kwargs
        )
        self._session = None  # Store session for cleanup

        # Import the Gemini process_message function
        try:
            from .backends.gemini import process_message as gemini_process_message

            self._process_message_impl = gemini_process_message
        except ImportError:
            import traceback

            self._process_message_impl = None
            print("Gemini agent implementation not available (missing dependencies)")
            traceback.print_exc()
            exit()

    def cleanup(self):
        """Clean up HTTP session resources."""
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._session = None

    def process_message(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AgentResponse:
        """Process message using Gemini backend."""
        if self._process_message_impl is None:
            return AgentResponse(
                text="Gemini agent implementation not available (missing dependencies)",
                code=[],
                citations=[],
                function_calls=[],
            )

        # Merge instance configuration with any overrides from kwargs
        # Use the exact parameter names expected by the backend
        config = {
            "model": self.model,
            "tools": self.tools,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_key": None,  # Let backend use environment variable
            "processing_timeout": self.processing_timeout,
            "stream": self.stream,
            "stream_callback": self.stream_callback,
            **self.kwargs,  # Instance kwargs
            **kwargs,  # Override kwargs
        }
        
        # Handle parameter name mapping for overrides
        if 'timeout' in config:
            config['processing_timeout'] = config.pop('timeout')
        
        # Remove None values to avoid overriding backend defaults
        config = {k: v for k, v in config.items() if v is not None}

        try:
            result = self._process_message_impl(
                messages=messages,
                **config
            )

            return AgentResponse(
                text=result.get("text", ""),
                code=result.get("code", []),
                citations=result.get("citations", []),
                function_calls=result.get("function_calls", []),
            )
        except Exception as e:
            # Return error response
            return AgentResponse(
                text=f"Error in Gemini agent processing: {str(e)}",
                code=[],
                citations=[],
                function_calls=[],
            )


class GrokMassAgent(MassAgent):
    """MassAgent wrapper for Grok agent implementation."""

    def __init__(
        self, 
        agent_id: int, 
        orchestrator=None, 
        model: str = "grok-4", 
        tools: List[str] = None,
        max_retries: int = 3,
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = None,
        processing_timeout: float = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
        **kwargs
    ):
        # Pass all configuration to parent
        super().__init__(
            agent_id=agent_id,
            orchestrator=orchestrator,
            model=model,
            tools=tools,
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            processing_timeout=processing_timeout,
            stream=stream,
            stream_callback=stream_callback,
            **kwargs
        )
        self._client = None  # Store client for cleanup

        # Import the Grok process_message function
        try:
            from .backends.grok import process_message as grok_process_message

            self._process_message_impl = grok_process_message
        except ImportError:
            import traceback

            self._process_message_impl = None
            print("Grok agent implementation not available (missing dependencies)")
            traceback.print_exc()
            exit()

    def cleanup(self):
        """Clean up HTTP client resources."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass  # Ignore cleanup errors
            self._client = None

    def process_message(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AgentResponse:
        """Process message using Grok backend."""
        if self._process_message_impl is None:
            return AgentResponse(
                text="Grok agent implementation not available (missing dependencies)",
                code=[],
                citations=[],
                function_calls=[],
            )

        # Merge instance configuration with any overrides from kwargs
        # Use the exact parameter names expected by the backend
        config = {
            "model": self.model,
            "tools": self.tools,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_key": None,  # Let backend use environment variable
            "processing_timeout": self.processing_timeout,
            "stream": self.stream,
            "stream_callback": self.stream_callback,
            **self.kwargs,  # Instance kwargs
            **kwargs,  # Override kwargs
        }
        
        # Handle parameter name mapping for overrides
        if 'timeout' in config:
            config['processing_timeout'] = config.pop('timeout')
        
        # Remove None values to avoid overriding backend defaults
        config = {k: v for k, v in config.items() if v is not None}

        try:
            result = self._process_message_impl(
                messages=messages,
                **config
            )

            return AgentResponse(
                text=result.get("text", ""),
                code=result.get("code", []),
                citations=result.get("citations", []),
                function_calls=result.get("function_calls", []),
            )
        except Exception as e:
            # Return error response
            return AgentResponse(
                text=f"Error in Grok agent processing: {str(e)}",
                code=[],
                citations=[],
                function_calls=[],
            )


# Agent factory function
def create_agent(agent_type: str, agent_id: int, orchestrator=None, **kwargs) -> MassAgent:
    """
    Factory function to create agents of different types.

    Args:
        agent_type: Type of agent to create ("openai", "gemini", "grok")
        agent_id: Unique identifier for the agent
        orchestrator: Reference to the orchestrator
        **kwargs: Additional arguments passed to the agent constructor

    Returns:
        MassAgent instance of the specified type
    """
    agent_classes = {
        "openai": OpenAIMassAgent,
        "gemini": GeminiMassAgent,
        "grok": GrokMassAgent,
    }

    if agent_type.lower() not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(agent_classes.keys())}")

    agent_class = agent_classes[agent_type.lower()]
    return agent_class(agent_id=agent_id, orchestrator=orchestrator, **kwargs)
