import os
import sys
import time
import json
from typing import Callable, Union, Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

from .types import TaskInput, AgentState, AgentResponse, ModelConfig
from .utils import get_agent_type_from_model, function_to_json
from abc import ABC, abstractmethod
from typing import Any, Callable, Union, Optional, List, Dict
from .backends import oai, gemini, grok

SYSTEM_INSTRUCTION = """
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
Any new answer you add via the `new_answer` tool must be self-contained, complete, and ready to serve as the definitive final response to the user's request.
"""

AGENT_ANSWER_MESSAGE = """
<ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS FROM THE AGENTS>
{agent_answers}
<END OF CURRENT ANSWERS>
"""

AGENT_ANSWER_AND_VOTE_MESSAGE = """
<ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS FROM THE AGENTS>
{agent_answers}
<END OF CURRENT ANSWERS>

<OTHERS' VOTES>
{agent_votes}
<END OF OTHERS' VOTES>

Please use your expertise and tools (if available) to analyze the answers and votes.
Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
Any new answer you add via the `new_answer` tool must be self-contained, complete, and ready to serve as the definitive final response to the user's request.
"""

class MassAgent(ABC):
    """
    Abstract base class for all agents in the MASS system.

    All agent implementations must inherit from this class and implement
    the required methods while following the standardized workflow.
    """

    def __init__(
        self, 
        agent_id: int, 
        orchestrator=None,
        model_config: Optional[ModelConfig] = None,
        stream_callback: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the agent with configuration parameters.

        Args:
            agent_id: Unique identifier for this agent
            orchestrator: Reference to the MassOrchestrator
            model_config: Configuration object containing model parameters (model, tools, 
                         temperature, top_p, max_tokens, processing_timeout, max_retries, stream)
            stream_callback: Optional callback function for streaming chunks
            agent_type: Type of agent ("openai", "gemini", "grok") to determine backend
            **kwargs: Additional parameters specific to the agent implementation
        """
        self.agent_id = agent_id
        self.orchestrator = orchestrator
        self.state = AgentState(agent_id=agent_id)
        
        # Initialize model configuration with defaults if not provided
        if model_config is None:
            model_config = ModelConfig()
        
        # Store configuration parameters
        self.model = model_config.model
        self.agent_type = get_agent_type_from_model(self.model)
        # Map agent types to their backend modules
        process_message_impl_map = {
            "openai": oai.process_message,
            "gemini": gemini.process_message, 
            "grok": grok.process_message
        }
        if self.agent_type not in process_message_impl_map:
            raise ValueError(f"Unknown agent type: {self.agent_type}. Available types: {list(process_message_impl_map.keys())}")
        
        # Get the appropriate process_message implementation based on the agent type
        self.process_message_impl = process_message_impl_map[self.agent_type]
                
        # Other model configuration parameters
        self.tools = model_config.tools
        self.max_retries = model_config.max_retries
        self.max_rounds = model_config.max_rounds
        self.max_tokens = model_config.max_tokens
        self.temperature = model_config.temperature
        self.top_p = model_config.top_p
        self.processing_timeout = model_config.processing_timeout
        self.stream = model_config.stream
        self.stream_callback = stream_callback
        self.kwargs = kwargs

    def process_message(self, messages: List[Dict[str, str]], tools: List[str] = None) -> AgentResponse:
        """
        Core LLM inference function for task processing.

        This method handles the actual LLM interaction using the agent's
        specific backend (OpenAI, Gemini, Grok, etc.) and returns a standardized response.
        All configuration parameters are stored as instance variables and accessed
        via self.model, self.tools, self.temperature, etc.

        Args:
            messages: List of messages in OpenAI format
            tools: List of tools to use

        Returns:
            AgentResponse containing the agent's response text, code, citations, etc.
        """
        
        # Create configuration dictionary using model configuration parameters
        config = {
            "model": self.model,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "api_key": None,  # Let backend use environment variable
            "processing_timeout": self.processing_timeout,
            "stream": self.stream,
            "stream_callback": self.stream_callback
        }
        
        try:
            result = self.process_message_impl(
                messages=messages,
                tools=tools,
                **config
            )

            # Backend implementations now return AgentResponse objects directly
            return result
        except Exception as e:
            # Return error response
            return AgentResponse(
                text=f"Error in {self.model} agent processing: {str(e)}",
                code=[],
                citations=[],
                function_calls=[],
            )

    def new_answer(self, answer: str):
        """
        Record your work on the task: your analysis, approach, solution, and reasoning. Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents.

        Args:
            answer: The new answer, which should be self-contained, complete, and ready to serve as the definitive final response.
        """
        # Use the orchestrator to update the answer and notify other agents to restart
        self.orchestrator.notify_answer_update(self.agent_id, answer)
        return f"The new answer has been added."
    
    def vote(self, agent_id: int, reason: str = ""):
        """
        Vote for the representative agent, who you believe has found the correct solution.

        Args:
            agent_id: ID of the voted agent
            reason: Your full explanation of why you voted for this agent
        """
        self.orchestrator.cast_vote(self.agent_id, agent_id, reason)
        return f"Your vote for Agent {agent_id} has been cast."
    
    def check_update(self) -> bool:
        """
        Check if there are any updates from other agents since this agent last saw them.
        """
        # Get updates from other agents since this agent last saw them
        for other_id, other_state in self.orchestrator.agent_states.items():
            if other_id != self.agent_id and other_state.update_history:
                last_seen = self.state.seen_updates_timestamps.get(other_id, 0)
                for update in other_state.update_history:
                    if update.timestamp > last_seen:
                        return True
        return False
    
    def mark_failed(self, reason: str = ""):
        """
        Mark this agent as failed.

        Args:
            reason: Optional reason for the failure
        """
        # Report the failure to the orchestrator for logging
        self.orchestrator.mark_agent_failed(self.agent_id, reason)

    def deduplicate_function_calls(self, function_calls: List[Dict]):
        """Deduplicate function calls by their name and arguments."""
        deduplicated_function_calls = []
        for func_call in function_calls:
            if func_call not in deduplicated_function_calls:
                deduplicated_function_calls.append(func_call)
        return deduplicated_function_calls
    
    def _execute_function_calls(self, function_calls: List[Dict]):
        """Execute function calls and return function outputs."""
        from .tools import register_tool
        function_outputs = []
        
        # DEBUGGING
        with open("function_calls.txt", "a") as f:
            # Write the function calls to the file
            # Include the agent id, agent model name, time, and function calls
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Agent {self.agent_id} ({self.model}):\n")
            f.write(f"{json.dumps(function_calls, indent=2)}\n")
        
        for func_call in function_calls:
            func_call_id = func_call.get("call_id")
            func_name = func_call.get("name")
            func_args = func_call.get("arguments", {})
            if isinstance(func_args, str):
                func_args = json.loads(func_args)
            
            try:
                if func_name == "new_answer":
                    result = self.new_answer(func_args.get("answer", ""))
                elif func_name == "vote":
                    result = self.vote(func_args.get("agent_id"), func_args.get("reason", ""))
                elif func_name in register_tool:
                    result = register_tool[func_name](**func_args)
                else:
                    result = {
                        "type": "function_call_output",
                        "call_id": func_call_id,
                        "output": f"Error: Function '{func_name}' not found in tool mapping"
                    }
                    
                # Add function call and result to messages
                function_output = {
                    "type": "function_call_output",
                    "call_id": func_call_id,
                    "output": str(result)
                }
                function_outputs.append(function_output)
                
                # DEBUGGING
                with open("function_outputs.txt", "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Agent {self.agent_id} ({self.model}):\n")
                    f.write(f"{json.dumps(function_output, indent=2)}\n")
                
            except Exception as e:
                # Handle execution errors
                error_output = {
                    "type": "function_call_output", 
                    "call_id": func_call_id,
                    "output": f"Error executing function: {str(e)}"
                }
                function_outputs.append(error_output)
                print(f"Error executing function {func_name}: {e}")
                
                # DEBUGGING
                with open("function_outputs.txt", "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Agent {self.agent_id} ({self.model}):\n")
                    f.write(f"{json.dumps(error_output, indent=2)}\n")
                
        return function_outputs
      
    def _get_system_tools(self) -> List[Dict[str, Any]]:
        """
        The system tools available to this agent for orchestration:
        - new_answer: Your added new answer, which should be self-contained, complete, and ready to serve as the definitive final response.
        - vote: Vote for the representative agent, who you believe has found the correct solution.
        """
        return [
            {
                "type": "function",
                "name": "new_answer",
                "description": "Add your new answer if you believe it is better than the current answers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your new answer, which should be self-contained, complete, and ready to serve as the definitive final response."
                        }
                    },
                    "required": ["answer"]
                }
            },
            {
                "type": "function",
                "name": "vote",
                "description": "Vote for the agent whose answer you believe is the best and addresses the original message.",
                "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "integer",
                                "description": "The ID of the agent you believe has found the best answer that addresses the original message."
                            },
                            "reason": {
                                "type": "string",
                                "description": "Your full explanation of why you voted for this agent."
                            }
                        },
                        "required": ["agent_id", "reason"]
                    }
                }
        ]

    def _get_registered_tools(self) -> List[Dict[str, Any]]:
        """Return the tool schema for the tools that are available to this agent."""
        # Register tools from the global registry, JSON schema
        custom_tools = []
        from .tools import register_tool
        for tool_name, tool_func in register_tool.items():
            if tool_name in self.tools:
                tool_schema = function_to_json(tool_func)
                custom_tools.append(tool_schema)
        return custom_tools
    
    def _get_all_answers(self) -> List[str]:
        """Get all answers from all agents.
        Format:
        **Agent 1**: Answer 1
        **Agent 2**: Answer 2
        ...
        """
        # Case 1: Initial round without running answer
        agent_answers = []
        for agent_id, agent_state in self.orchestrator.agent_states.items():
            if agent_state.curr_answer:
                agent_answers.append(f"**Agent {agent_id}**: {agent_state.curr_answer}")
        return agent_answers
    
    def _get_all_votes(self) -> List[str]:
        """Get all votes from all agents.
        Format:
        **Vote for Agent 1**: Reason 1
        **Vote for Agent 2**: Reason 2
        ...
        """
        agent_votes = []
        for agent_id, agent_state in self.orchestrator.agent_states.items():
            if agent_state.curr_vote:
                agent_votes.append(f"**Vote for Agent {agent_state.curr_vote.target_id}**: {agent_state.curr_vote.reason}")
        return agent_votes
    
    def _get_task_input(self, task: TaskInput) -> str:
        """Get the initial task input as the user message."""
        # Case 1: Initial round without running answer
        if not self.state.curr_answer:
            return AGENT_ANSWER_MESSAGE.format(task=task.question, agent_answers="None") + \
                   "There are no current answers right now. Please use your expertise and tools (if available) to provide a new answer and submit it using the `add_answer` tool first."
    
        all_agent_answers = self._get_all_answers()
        all_agent_answers_str = "\n\n".join(all_agent_answers)
        # Check if in debate mode or not
        voted_agents = [agent_id for agent_id, agent_state in self.orchestrator.agent_states.items() if agent_state.curr_vote is not None]
        if len(voted_agents) == len(self.orchestrator.agent_states):
            # Case 2: All agents have voted and are debating. Can not use agent status to check as they have been updated to 'working/debate'
            all_agent_votes = self._get_all_votes()
            all_agent_votes_str = "\n\n".join(all_agent_votes)
            return AGENT_ANSWER_AND_VOTE_MESSAGE.format(task=task.question, agent_answers=all_agent_answers_str, agent_votes=all_agent_votes_str)
        else:
            # Case 3: All agents are working and not in debating
            return AGENT_ANSWER_MESSAGE.format(task=task.question, agent_answers=all_agent_answers_str)
    
    def _get_task_input_messages(self, task: TaskInput) -> List[Dict[str, str]]:
        """Get the task input messages for the agent."""
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": self._get_task_input(task)}
        ]
            
        
    @abstractmethod
    def _get_builtin_tools(self) -> List[Dict[str, Any]]:
        """Return the built-in tools that are available to different agent backends."""
        pass
    
    @abstractmethod
    def work_on_task(self, task: TaskInput, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Work on the task with conversation continuation.
        
        Args:
            task: The task to work on
            messages: Current conversation history
            restart_instruction: Optional instruction for restarting work (e.g., updates from other agents)
            
        Returns:
            Updated conversation history including agent's work
            
        This method should be implemented by concrete agent classes.
        The agent continues the conversation until it votes or reaches max rounds.
        """
        pass