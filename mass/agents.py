"""
MassAgent implementations that wrap the existing agent backends.

This module provides MassAgent-compatible wrappers for the existing
OpenAI, Gemini, and Grok agent implementations.
"""

import os
import sys
import copy
import time
from typing import Callable, Union, Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

from .agent import MassAgent
from .types import ModelConfig, TaskInput
from .tools import register_tool

class OpenAIMassAgent(MassAgent):
    """MassAgent wrapper for OpenAI agent implementation."""

    def __init__(
        self, 
        agent_id: int, 
        orchestrator=None, 
        model_config: Optional[ModelConfig] = None,
        stream_callback: Optional[Callable] = None,
        **kwargs
    ):
           
        # Pass all configuration to parent, including agent_type
        super().__init__(
            agent_id=agent_id,
            orchestrator=orchestrator,
            model_config=model_config,
            stream_callback=stream_callback,
            **kwargs
        )
    
    def work_on_task(self, task: TaskInput):
        """
        Work on the task using the OpenAI backend. Take the task and current conversation history (messages)
        Return the continued conversation history (messages) as output.
        """
        curr_round = 0
        
        # Initialize conversation
        messages = []
        system_instruction = self.get_instruction_based_on_status("system")
        messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": task.question})
        
        # Get available tools (system tools + built-in tools + custom tools)
        all_tools = self._get_available_tools()
        
        # Start the task solving loop
        while curr_round < self.max_rounds:
            try:
                # Call LLM
                result = self.process_message(messages=messages, tools=all_tools)
                
                # Add assistant response
                if result.text:
                    messages.append({"role": "assistant", "content": result.text})
                    
                    # Check if this is the final presentation by the representative agent
                    self._check_if_final_presentation(result.text)
                
                # Execute function calls if any
                if result.function_calls:
                    self._execute_function_calls(result.function_calls, messages)
                else:
                    # No function calls - check if we should continue or wait
                    if self.state.status == "voted":
                        # Agent has voted, exit the work loop
                        break
                    else:
                        # Continue working or waiting for updates
                        curr_round += 1
                        continue
                        
                curr_round += 1
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {curr_round}: {e}")
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                break
            

class GeminiMassAgent(OpenAIMassAgent):
    """MassAgent wrapper for Gemini agent implementation."""

    def __init__(
        self,
        agent_id: int,
        orchestrator=None,
        model_config: Optional[ModelConfig] = None,
        stream_callback: Optional[Callable] = None,
        **kwargs,
    ):

        # Pass all configuration to parent, including agent_type
        super().__init__(
            agent_id=agent_id,
            orchestrator=orchestrator,
            model_config=model_config,
            stream_callback=stream_callback,
            **kwargs
        )
    
    # Inherits work_on_task from OpenAIMassAgent


class GrokMassAgent(OpenAIMassAgent):
    """MassAgent wrapper for Grok agent implementation."""

    def __init__(
        self,
        agent_id: int,
        orchestrator=None,
        model_config: Optional[ModelConfig] = None,
        stream_callback: Optional[Callable] = None,
        **kwargs,
    ):

        # Pass all configuration to parent, including agent_type
        super().__init__(
            agent_id=agent_id,
            orchestrator=orchestrator,
            model_config=model_config,
            stream_callback=stream_callback,
            **kwargs
        )
    
    # Inherits work_on_task from OpenAIMassAgent


def create_agent(agent_type: str, agent_id: int, orchestrator=None, model_config: Optional[ModelConfig] = None, **kwargs) -> MassAgent:
    """
    Factory function to create agents of different types.
    
    Args:
        agent_type: Type of agent ("openai", "gemini", "grok")
        agent_id: Unique identifier for the agent
        orchestrator: Reference to the MassOrchestrator
        model_config: Model configuration
        **kwargs: Additional arguments
        
    Returns:
        MassAgent instance of the specified type
    """
    agent_classes = {
        "openai": OpenAIMassAgent,
        "gemini": GeminiMassAgent,
        "grok": GrokMassAgent
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}. Available types: {list(agent_classes.keys())}")
        
    return agent_classes[agent_type](
        agent_id=agent_id,
        orchestrator=orchestrator,
        model_config=model_config,
        **kwargs
    )

