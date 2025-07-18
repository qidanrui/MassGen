"""
MassAgent implementations that wrap the existing agent backends.

This module provides MassAgent-compatible wrappers for the existing
OpenAI, Gemini, and Grok agent implementations.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Add agents directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from mass_agent import AgentResponse, MassAgent

# Try to import function_to_json, but make it optional
try:
    from agents.util import function_to_json
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

    def __init__(self, agent_id: int, orchestration_system=None, model: str = "o4-mini", **kwargs):
        super().__init__(agent_id, orchestration_system)
        self.model = model
        self.kwargs = kwargs
        self._client = None  # Store client for cleanup

        # Import the OpenAI process_message function
        try:
            from agents.oai import process_message as oai_process_message

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
        messages: list[dict[str, str]],
        tools: list[str] = None,
        temperature: float = 0.7,
        timeout: float = None,
        stream: bool = False,
        stream_callback: callable | None = None,
        **kwargs,
    ) -> AgentResponse:
        """Process message using OpenAI backend."""
        if tools is None:
            tools = self._get_available_tools()

        # Merge kwargs with instance defaults
        merged_kwargs = {**self.kwargs, **kwargs}

        if self._process_message_impl is None:
            return AgentResponse(
                text="OpenAI agent implementation not available (missing dependencies)",
                code=[],
                citations=[],
                function_calls=[],
            )

        try:
            result = self._process_message_impl(
                messages=messages,
                model=self.model,
                tools=tools,
                temperature=temperature,
                processing_timeout=timeout,
                stream=stream,
                stream_callback=stream_callback,
                **merged_kwargs,
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
        orchestration_system=None,
        model: str = "gemini-2.5-flash",
        **kwargs,
    ):
        super().__init__(agent_id, orchestration_system)
        self.model = model
        self.kwargs = kwargs
        self._session = None  # Store session for cleanup

        # Import the Gemini process_message function
        try:
            from agents.gemini import process_message as gemini_process_message

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
        messages: list[dict[str, str]],
        tools: list[str] = None,
        temperature: float = 0.7,
        timeout: float = None,
        stream: bool = False,
        stream_callback: callable | None = None,
        **kwargs,
    ) -> AgentResponse:
        """Process message using Gemini backend."""
        if tools is None:
            tools = self._get_available_tools()

        # Merge kwargs with instance defaults
        merged_kwargs = {**self.kwargs, **kwargs}

        if self._process_message_impl is None:
            return AgentResponse(
                text="Gemini agent implementation not available (missing dependencies)",
                code=[],
                citations=[],
                function_calls=[],
            )

        try:
            result = self._process_message_impl(
                messages=messages,
                model=self.model,
                tools=tools,
                temperature=temperature,
                processing_timeout=timeout,
                stream=stream,
                stream_callback=stream_callback,
                **merged_kwargs,
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

    def __init__(self, agent_id: int, orchestration_system=None, model: str = "grok-4", **kwargs):
        super().__init__(agent_id, orchestration_system)
        self.model = model
        self.kwargs = kwargs
        self._client = None  # Store client for cleanup

        # Import the Grok process_message function
        try:
            from agents.grok import process_message as grok_process_message

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
        messages: list[dict[str, str]],
        tools: list[str] = None,
        temperature: float = 0.7,
        timeout: float = None,
        stream: bool = False,
        stream_callback: callable | None = None,
        **kwargs,
    ) -> AgentResponse:
        """Process message using Grok backend."""
        if tools is None:
            tools = self._get_available_tools()

        # Merge kwargs with instance defaults
        merged_kwargs = {**self.kwargs, **kwargs}

        if self._process_message_impl is None:
            return AgentResponse(
                text="Grok agent implementation not available (missing dependencies)",
                code=[],
                citations=[],
                function_calls=[],
            )

        try:
            result = self._process_message_impl(
                messages=messages,
                model=self.model,
                tools=tools,
                temperature=temperature,
                processing_timeout=timeout,
                stream=stream,
                stream_callback=stream_callback,
                **merged_kwargs,
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
def create_agent(agent_type: str, agent_id: int, orchestration_system=None, **kwargs) -> MassAgent:
    """
    Factory function to create agents of different types.

    Args:
        agent_type: Type of agent to create ("openai", "gemini", "grok")
        agent_id: Unique identifier for the agent
        orchestration_system: Reference to the orchestration system
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
    return agent_class(agent_id=agent_id, orchestration_system=orchestration_system, **kwargs)
