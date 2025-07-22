"""
MassAgent implementations that wrap the existing agent backends.

This module provides MassAgent-compatible wrappers for the existing
OpenAI, Gemini, and Grok agent implementations.
"""

import os
import sys
import copy
import time
import traceback
from typing import Callable, Union, Optional, List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

from .agent import MassAgent, SYSTEM_INSTRUCTION, AGENT_ANSWER_MESSAGE, AGENT_ANSWER_AND_VOTE_MESSAGE
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
    
    def _get_builtin_tools(self) -> List[Dict[str, Any]]:
        """Return the built-in tools that are available to OpenAI models. 
        live_search and code_execution are supported right now.
        """
        return ["live_search", "code_execution"]
    
    def work_on_task(self, task: TaskInput) -> List[Dict[str, str]]:
        """
        Work on the task using the OpenAI backend with conversation continuation.
        
        NOTE:
        OpenAI's function call can not have id, otherwise it will require related reasoning items which we did not maintain.
        You must provide function call and function call output in consecutive messages.
        
        Args:
            task: The task to work on
            messages: Starting conversation history
            
        Returns:
            Updated conversation history including agent's work
        """    
        
        # Get available tools (system tools + built-in tools + custom tools)
        all_tools = []
        all_tools.extend(self._get_builtin_tools())
        all_tools.extend(self._get_registered_tools())
        all_tools.extend(self._get_system_tools())
        
        # Initialize working messages
        curr_round = 0
        working_messages = self._get_task_input_messages(task)
        
        # Start the task solving loop
        while curr_round < self.max_rounds and self.state.status == "working":
            try:
                # Call LLM with current conversation
                result = self.process_message(messages=working_messages, 
                                              tools=all_tools)
                
                # Add assistant response
                if result.text:
                    working_messages.append({"role": "assistant", "content": result.text})
                
                # Execute function calls if any
                if result.function_calls:
                    # Deduplicate function calls by their name
                    result.function_calls = self.deduplicate_function_calls(result.function_calls)
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
                        # Check if there is any update from other agents that are unseen by this agent
                        has_update = self.check_update()
                        if has_update: 
                            # Renew the conversation within the loop
                            working_messages = self._get_task_input_messages(task)
                        else: # Continue the current conversation and prompting checkin
                            working_messages.append({"role": "user", "content": "Please use either `add_answer` or `vote` tool once your analysis is done."})
                 
                curr_round += 1
                self.state.chat_round += 1        
                
                # Check if agent voted or failed
                if self.state.status in ["voted", "failed"]:
                    break
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {self.state.chat_round}: {e}")                   
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                self.state.chat_round += 1
                # DEBUGGING
                with open("errors.txt", "a") as f:
                    f.write(f"Agent {self.agent_id} error in round {self.state.chat_round}: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n\n")
                           
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
    
    def _get_builtin_tools(self) -> List[Dict[str, Any]]:
        """Return the built-in tools that are available to Gemini models. 
        live_search and code_execution are supported right now.
        However, the built-in tools and function call are not supported at the same time.
        """
        return ["live_search", "code_execution"]
    
    def work_on_task(self, task: TaskInput) -> List[Dict[str, str]]:
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
    
        # Get available tools (system tools + built-in tools + custom tools)
        system_tools = self._get_system_tools()
        built_in_tools = self._get_builtin_tools()
        custom_tools = self._get_registered_tools()
        
        # Gemini does not support built-in tools and function call at the same time.
        # If built-in tools are provided, we will switch to them in the next round.
        tool_switch = bool(built_in_tools)
        
        # We provide built-in tools in the first round, and then custom tools in the next round.
        if tool_switch:
            function_call_enabled = False
            available_tools = built_in_tools
        else:
            function_call_enabled = True
            available_tools = system_tools + custom_tools
        
        # Initialize working messages
        curr_round = 0
        working_messages = self._get_task_input_messages(task)
        
        # Start the task solving loop
        while curr_round < self.max_rounds and self.state.status == "working":
            try:
                # If function call is enabled or not, add a notification to the user
                if working_messages[-1].get("role", "") == "user":
                    if not function_call_enabled:
                        working_messages[-1]["content"] += "\n\n" + "Note that the `new_answer` and `vote` tools are not enabled now. Please prioritize using the built-in tools to analyze the task first."
                    else:
                        working_messages[-1]["content"] += "\n\n" + "Note that the `new_answer` and `vote` tools are enabled now."
                
                # Call LLM with current conversation
                result = self.process_message(messages=working_messages, tools=available_tools)
                
                # Add assistant response
                if result.text:
                    working_messages.append({"role": "assistant", "content": result.text})
                
                # Execute function calls if any
                if result.function_calls:
                    # Deduplicate function calls by their name
                    result.function_calls = self.deduplicate_function_calls(result.function_calls)
                    function_outputs = self._execute_function_calls(result.function_calls)
                    # Add function call results to conversation (Gemini format)
                    for function_call, function_output in zip(result.function_calls, function_outputs):
                        working_messages.extend([function_call, function_output])
                    
                    # If we have used custom tools, switch to built-in tools in the next round
                    if tool_switch:
                        available_tools = built_in_tools
                        function_call_enabled = False
                        print(f"🔄 Agent {self.agent_id} (Gemini) switching to built-in tools in the next round")
                    
                    # Check if agent voted or failed after function calls
                    if self.state.status in ["voted", "failed"]:
                        print(f"🔄 Agent {self.agent_id} stopping work loop - status changed to {self.state.status}")
                        break
                else:
                    # No function calls - check if we should continue or stop
                    if self.state.status == "voted":
                        # Agent has voted, exit the work loop
                        break
                    else:
                        # Check if there is any update from other agents that are unseen by this agent
                        has_update = self.check_update()
                        if has_update: 
                            # Renew the conversation within the loop
                            working_messages = self._get_task_input_messages(task)
                        else: # Continue the current conversation and prompting checkin
                            working_messages.append({"role": "user", "content": "Please use either `add_answer` or `vote` tool once your analysis is done."})

                    # Switch to custom tools in the next round
                    if tool_switch:
                        available_tools = system_tools + custom_tools
                        function_call_enabled = True
                        print(f"🔄 Agent {self.agent_id} (Gemini) switching to custom tools in the next round")
                    
                    curr_round += 1
                    self.state.chat_round += 1 
                    
                    # Check if agent voted or failed
                    if self.state.status in ["voted", "failed"]:
                        break

            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {self.state.chat_round}: {e}")                   
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                self.state.chat_round += 1
                # DEBUGGING
                with open("errors.txt", "a") as f:
                    f.write(f"Agent {self.agent_id} error in round {self.state.chat_round}: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n\n")
        
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
    
    def _get_builtin_tools(self) -> List[Dict[str, Any]]:
        """Return the built-in tools that are available to Grok models. 
        Only live_search is supported right now.
        """
        return ["live_search"]
    
    def work_on_task(self, task: TaskInput) -> List[Dict[str, str]]:
        """
        Work on the task using the Grok backend with conversation continuation.
        
        NOTE:
        OpenAI's function call can not have id, otherwise it will require related reasoning items which we did not maintain.
        You must provide function call and function call output in consecutive messages.
        
        Args:
            task: The task to work on
            messages: Starting conversation history
            
        Returns:
            Updated conversation history including agent's work
        """    
        
        # Get available tools (system tools + built-in tools + custom tools)
        all_tools = []
        all_tools.extend(self._get_builtin_tools())
        all_tools.extend(self._get_registered_tools())
        all_tools.extend(self._get_system_tools())
        
        # Initialize working messages
        curr_round = 0
        working_messages = self._get_task_input_messages(task)
        
        # Start the task solving loop
        while curr_round < self.max_rounds and self.state.status == "working":
            try:
                # Call LLM with current conversation
                result = self.process_message(messages=working_messages, 
                                              tools=all_tools)
                
                # Add assistant response
                if result.text:
                    working_messages.append({"role": "assistant", "content": result.text})
                
                # Execute function calls if any
                if result.function_calls:
                    # Deduplicate function calls by their name
                    result.function_calls = self.deduplicate_function_calls(result.function_calls)
                    function_outputs = self._execute_function_calls(result.function_calls)
                    # Add function call results to conversation
                    for function_call, function_output in zip(result.function_calls, function_outputs):
                        # [OpenAI] Remove id to avoid the requirements of related reasoning items
                        working_messages.extend([function_call, function_output])
                else:
                    # No function calls - check if we should continue or stop
                    if self.state.status == "voted":
                        # Agent has voted, exit the work loop
                        break
                    else:
                        # Check if there is any update from other agents that are unseen by this agent
                        has_update = self.check_update()
                        if has_update: 
                            # Renew the conversation within the loop
                            working_messages = self._get_task_input_messages(task)
                        else: # Continue the current conversation and prompting checkin
                            working_messages.append({"role": "user", "content": "Please use either `add_answer` or `vote` tool once your analysis is done."})
                 
                curr_round += 1
                self.state.chat_round += 1        
                
                # Check if agent voted or failed
                if self.state.status in ["voted", "failed"]:
                    break
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error in round {self.state.chat_round}: {e}")                   
                if self.orchestrator:
                    self.orchestrator.mark_agent_failed(self.agent_id, str(e))
                self.state.chat_round += 1
                # DEBUGGING
                with open("errors.txt", "a") as f:
                    f.write(f"Agent {self.agent_id} error in round {self.state.chat_round}: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n\n")
                           
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

