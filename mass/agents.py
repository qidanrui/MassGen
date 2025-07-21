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
    
    def work_on_task(self, task: TaskInput, messages: List[Dict[str, str]], restart_instruction: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Work on the task using the OpenAI backend with conversation continuation.
        
        NOTE:
        OpenAI's function call can not have id, otherwise it will require related reasoning items which we did not maintain.
        You must provide function call and function call output in consecutive messages.
        
        Args:
            task: The task to work on
            messages: Current conversation history
            restart_instruction: Optional instruction for restarting work (e.g., updates from other agents)
            
        Returns:
            Updated conversation history including agent's work
        """
        curr_round = 0
        
        # Start with provided messages
        working_messages = messages.copy()
        
        # Add restart instruction if provided
        if restart_instruction:
            working_messages.append({"role": "user", "content": restart_instruction})
        
        # Get available tools (system tools + built-in tools + custom tools)
        all_tools = self._get_available_tools()
        
        # Start the task solving loop
        while curr_round < self.max_rounds and self.state.status == "working":
            try:
                # Call LLM with current conversation
                result = self.process_message(messages=working_messages, tools=all_tools)
                
                # Add assistant response
                if result.text:
                    working_messages.append({"role": "assistant", "content": result.text})
                
                # Execute function calls if any
                if result.function_calls:
                    function_outputs = self._execute_function_calls(result.function_calls)
                    # Add function call results to conversation
                    for function_call, function_output in zip(result.function_calls, function_outputs):
                        # [OpenAI] Remove id to avoid the requirements of related reasoning items
                        clean_function_call = copy.deepcopy(function_call)
                        del clean_function_call['id'] 
                        working_messages.extend([clean_function_call, function_output])
                else:
                    # No function calls - check if we should continue or stop
                    if self.state.status == "voted":
                        # Agent has voted, exit the work loop
                        break
                    else:
                        # Continue working
                        curr_round += 1
                        continue
                        
                curr_round += 1
                
                # Check if agent voted or failed
                if self.state.status in ["voted", "failed"]:
                    break
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {curr_round}: {e}")
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                break
        
        return working_messages


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
    
    def work_on_task(self, task: TaskInput, messages: List[Dict[str, str]], restart_instruction: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Work on the task using the Gemini backend with conversation continuation.
        
        NOTE:
        Gemini's does not support built-in tools and function call at the same time.
        Therefore, we provide them interchangedly in different rounds.
        The way the conversation is constructed is also different from OpenAI.
        You can provide consecutive user messages to represent the function call results.
        
        Args:
            task: The task to work on
            messages: Current conversation history
            restart_instruction: Optional instruction for restarting work (e.g., updates from other agents)
            
        Returns:
            Updated conversation history including agent's work
        """
        curr_round = 0
        
        # Start with provided messages
        working_messages = messages.copy()
        
        # Add restart instruction if provided
        if restart_instruction:
            working_messages.append({"role": "user", "content": restart_instruction})
        
        # Get available tools (system tools + built-in tools + custom tools)
        all_tools = self._get_available_tools()
        
        # Start the task solving loop
        while curr_round < self.max_rounds and self.state.status == "working":
            try:
                # Call LLM with current conversation
                result = self.process_message(messages=working_messages, tools=all_tools)
                
                # Add assistant response
                if result.text:
                    working_messages.append({"role": "assistant", "content": result.text})
                
                # Execute function calls if any
                if result.function_calls:
                    function_outputs = self._execute_function_calls(result.function_calls)
                    # Add function call results to conversation (Gemini format)
                    for function_call, function_output in zip(result.function_calls, function_outputs):
                        # Add function call result as user message
                        working_messages.append({
                            "role": "user",
                            "content": (
                                f"You have called function `{function_call['name']}` with arguments: {function_call['arguments']}.\n"
                                f"The return is:\n{function_output['output']}"
                            )
                        })
                    
                    # Important: Increment round after processing function calls
                    curr_round += 1
                    
                    # Check if agent voted or failed after function calls
                    if self.state.status in ["voted", "failed"]:
                        print(f"🔄 Agent {self.agent_id} stopping work loop - status changed to {self.state.status}")
                        break
                else:
                    # No function calls - check if we should continue or stop
                    if self.state.status == "voted":
                        # Agent has voted, exit the work loop
                        print(f"🔄 Agent {self.agent_id} stopping work loop - already voted")
                        break
                    else:
                        # Continue working - increment round to prevent infinite loop
                        curr_round += 1
                        # Add a small pause to prevent rapid cycling
                        if curr_round >= self.max_rounds - 2:
                            print(f"⚠️ Agent {self.agent_id} approaching max rounds ({curr_round}/{self.max_rounds})")
                        continue
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {curr_round}: {e}")
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                break
        
        print(f"🏁 Agent {self.agent_id} finished work loop after {curr_round} rounds with status: {self.state.status}")
        return working_messages


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
    
    def work_on_task(self, task: TaskInput, messages: List[Dict[str, str]], restart_instruction: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Work on the task using the Grok backend with conversation continuation.
        
        NOTE:
        Grok is much flexible than OpenAI and Gemini.
        You can provide built-in tools and function call at the same time.
        You can have consecutive user messages to represent the function call results.
        
        Args:
            task: The task to work on
            messages: Current conversation history
            restart_instruction: Optional instruction for restarting work (e.g., updates from other agents)
            
        Returns:
            Updated conversation history including agent's work
        """
        curr_round = 0
        
        # Start with provided messages
        working_messages = messages.copy()
        
        # Add restart instruction if provided
        if restart_instruction:
            working_messages.append({"role": "user", "content": restart_instruction})
        
        # Get available tools (system tools + built-in tools + custom tools)
        all_tools = self._get_available_tools()
        
        # Start the task solving loop
        while curr_round < self.max_rounds and self.state.status == "working":
            try:
                # Call LLM with current conversation
                result = self.process_message(messages=working_messages, tools=all_tools)
                
                # Add assistant response
                if result.text:
                    working_messages.append({"role": "assistant", "content": result.text})
                
                # Execute function calls if any
                if result.function_calls:
                    function_outputs = self._execute_function_calls(result.function_calls)
                    # Add function call results to conversation
                    for function_call, function_output in zip(result.function_calls, function_outputs):
                        # Add function call result as user message
                        working_messages.append({
                            "role": "user",
                            "content": (
                                f"You have called function `{function_call['name']}` with arguments: {function_call['arguments']}.\n"
                                f"The return is:\n{function_output['output']}"
                            )
                        })
                else:
                    # No function calls - check if we should continue or stop
                    if self.state.status == "voted":
                        # Agent has voted, exit the work loop
                        break
                    else:
                        # Continue working
                        curr_round += 1
                        continue
                        
                curr_round += 1
                
                # Check if agent voted or failed
                if self.state.status in ["voted", "failed"]:
                    break
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {curr_round}: {e}")
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                break
        
        return working_messages


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

