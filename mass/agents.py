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
        Work on the task.
        
        This method should be implemented by concrete agent classes.
        """
        curr_round = 0
        
        messages = []
        system_instruction = self.get_instruction_based_on_phase("system")
        
        # Add system instruction to current messages
        messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": task.question})
        
        # Find three types of tools to use:
        system_tools = self._get_system_tools()
        built_in_tools = []
        custom_tools = []
        for tool in self.tools:
            if tool in ["live_search", "code_execution"]:
                built_in_tools.append(tool)
            elif tool in register_tool.keys():
                custom_tools.append(tool)
            else:
                raise ValueError(f"Unregistered tool: {tool}")
        all_tools = built_in_tools + custom_tools + system_tools
        
        # Execute the tools with the tool mappings
        tool_mappings = {
            "update_summary": self.update_summary,
            "vote": self.vote,
        } 
        for tool in register_tool.keys():
            tool_mappings[tool] = register_tool[tool]
        
        # Add system tools to current messages
        messages.append({"role": "system", "content": system_tools})
        
        # Start the task solving loop
        while curr_round < self.max_rounds:
            result = self.process_message(
                messages=messages,
                tools=all_tools,
            )
            
            # Always add the assistant response to the messages
            if result['text']:
                messages.append({"role": "assistant", "content": result['text']})
            
            # If there are function calls, execute them with the tool mappings
            if result['function_calls']:
                function_outputs = execute_function_calls(result['function_calls'], tool_mappings)
                # Add function call results to conversation
                for function_call, function_output in zip(result['function_calls'], function_outputs):
                    # [OpenAI] Remove id to avoid the requirements of related reasoning items
                    clean_function_call = copy.deepcopy(function_call)
                    del clean_function_call['id'] 
                    messages.extend([clean_function_call, function_output])
            else: # If no function calls, we add notification based on the system status
                has_updates = bool(self.check_updates())
                if self.agent.status == "voted" and not has_updates:
                    wait()
                elif self.agent.status == "voted" and has_updates:
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("notification")})
                elif self.orchestrator.system_state.phase == "debate":
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("debate")})
                elif self.orchestrator.system_state.phase == "consensus":
                    if self.orchestrator.system_state.representative_agent_id == self.agent_id:
                        messages.append({"role": "user", "content": self.get_instruction_based_on_status("presentation")})
                    else:
                        break
                else:
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("notification")})
                
            curr_round += 1
        
        return messages
            

class GeminiMassAgent(MassAgent):
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
    
    def work_on_task(self, task: TaskInput):
        """
        Work on the task.
        
        This method should be implemented by concrete agent classes.
        """
        curr_round = 0
        
        messages = []
        system_instruction = self.get_instruction_based_on_phase("system")
        
        # Add system instruction to current messages
        messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": task.question})
        
        # Find three types of tools to use:
        system_tools = self._get_system_tools()
        built_in_tools = []
        custom_tools = []
        for tool in self.tools:
            if tool in ["live_search", "code_execution"]:
                built_in_tools.append(tool)
            elif tool in register_tool.keys():
                custom_tools.append(tool)
            else:
                raise ValueError(f"Unregistered tool: {tool}")
            
        # Gemini can not use built-in tools and custom tools at the same time
        all_tools = built_in_tools
        
        # Execute the tools with the tool mappings
        tool_mappings = {
            "update_summary": self.update_summary,
            "vote": self.vote,
        } 
        for tool in register_tool.keys():
            tool_mappings[tool] = register_tool[tool]
        
        # Add system tools to current messages
        messages.append({"role": "system", "content": system_tools})
        
        # Start the task solving loop
        while curr_round < self.max_rounds:
            result = self.process_message(
                messages=messages,
                tools=all_tools,
            )
            
            # Always add the assistant response to the messages
            if result['text']:
                messages.append({"role": "assistant", "content": result['text']})
                # If have response, use system tools to update the summary
                all_tools = system_tools + custom_tools
            
            # If there are function calls, execute them with the tool mappings
            if result['function_calls']:
                function_outputs = execute_function_calls(result['function_calls'], tool_mappings)
                # Add function call results to conversation (Gemini format)
                for function_call, function_output in zip(result['function_calls'], function_outputs):
                    # Add function call result as user message
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have called function `{function_call['name']}` with arguments: {function_call['arguments']}.\n"
                            f"The return is:\n{function_output['output']}"
                        )
                    })
                # If have used our custom tools, we need to switch back to the built-in tools
                all_tools = built_in_tools
            else: # If no function calls, we add notification based on the system status
                has_updates = bool(self.check_updates())
                if self.agent.status == "voted" and not has_updates:
                    wait()
                elif self.agent.status == "voted" and has_updates:
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("notification")})
                elif self.orchestrator.system_state.phase == "debate":
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("debate")})
                elif self.orchestrator.system_state.phase == "consensus":
                    if self.orchestrator.system_state.representative_agent_id == self.agent_id:
                        messages.append({"role": "user", "content": self.get_instruction_based_on_status("presentation")})
                    else:
                        break
                else:
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("notification")})
                
            curr_round += 1
        
        return messages


class GrokMassAgent(MassAgent):
    """MassAgent wrapper for Grok agent implementation."""

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
        Work on the task.
        
        This method should be implemented by concrete agent classes.
        """
        curr_round = 0
        
        messages = []
        system_instruction = self.get_instruction_based_on_phase("system")
        
        # Add system instruction to current messages
        messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": task.question})
        
        # Find three types of tools to use:
        system_tools = self._get_system_tools()
        built_in_tools = []
        custom_tools = []
        for tool in self.tools:
            if tool in ["live_search", "code_execution"]:
                built_in_tools.append(tool)
            elif tool in register_tool.keys():
                custom_tools.append(tool)
            else:
                raise ValueError(f"Unregistered tool: {tool}")
        all_tools = built_in_tools + custom_tools + system_tools
        
        # Execute the tools with the tool mappings
        tool_mappings = {
            "update_summary": self.update_summary,
            "vote": self.vote,
        } 
        for tool in register_tool.keys():
            tool_mappings[tool] = register_tool[tool]
        
        # Add system tools to current messages
        messages.append({"role": "system", "content": system_tools})
        
        # Start the task solving loop
        while curr_round < self.max_rounds:
            result = self.process_message(
                messages=messages,
                tools=all_tools,
            )
            
            # Always add the assistant response to the messages
            if result['text']:
                messages.append({"role": "assistant", "content": result['text']})
            
            # If there are function calls, execute them with the tool mappings
            if result['function_calls']:
                function_outputs = execute_function_calls(result['function_calls'], tool_mappings)
                # Add function call results to conversation
                for function_call, function_output in zip(result['function_calls'], function_outputs):
                    # Add function call result as user message
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have called function `{function_call['name']}` with arguments: {function_call['arguments']}.\n"
                            f"The return is:\n{function_output['output']}"
                        )
                    })
            else: # If no function calls, we add notification based on the system status
                has_updates = bool(self.check_updates())
                if self.agent.status == "voted" and not has_updates:
                    wait()
                elif self.agent.status == "voted" and has_updates:
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("notification")})
                elif self.orchestrator.system_state.phase == "debate":
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("debate")})
                elif self.orchestrator.system_state.phase == "consensus":
                    if self.orchestrator.system_state.representative_agent_id == self.agent_id:
                        messages.append({"role": "user", "content": self.get_instruction_based_on_status("presentation")})
                    else:
                        break
                else:
                    messages.append({"role": "user", "content": self.get_instruction_based_on_status("notification")})
                
            curr_round += 1
        
        return messages


# Agent factory function
def create_agent(agent_type: str, agent_id: int, orchestrator=None, model_config: Optional[ModelConfig] = None, **kwargs) -> MassAgent:
    """
    Factory function to create agents of different types.

    Args:
        agent_type: Type of agent to create ("openai", "gemini", "grok")
        agent_id: Unique identifier for the agent
        orchestrator: Reference to the orchestrator
        model_config: Model configuration object containing model parameters
        **kwargs: Additional arguments passed to the agent constructor (like stream_callback)

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
    return agent_class(agent_id=agent_id, orchestrator=orchestrator, model_config=model_config, **kwargs)

