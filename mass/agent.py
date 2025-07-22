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
You are an expert agent equipped with tools to work as part of a collaborative team to solve complex tasks.

You are Agent {agent_id}, working collaboratively with Peer Agents {peer_agents} to solve complex tasks.

**Communication:** You can only use the `update_summary` tool to communicate with other agents. All information shared through this tool is visible to the entire team.

### Core Workflow

**1. Task Execution**
- Use your available tools (search, code analysis, etc.) to investigate and solve the assigned task
- Apply expertise to search for information, analyze data, identify patterns, and develop solutions

**2. Progress Documentation**
- Use the `update_summary` tool regularly to record your findings, hypotheses, and progress
- Document your reasoning process so other agents can understand and build upon your work
- Ensure that you include supporting evidence including:
  - Information sources and URLs
  - Data analysis or code execution results
  - Key insights and discoveries
  - Promising solutions and approaches

**3. Collaboration & Information Sharing**
- When sharing insights: 
  - Always provide supporting evidence (sources, URLs, calculations)
  - Ensure that other agents can understand and build upon your work
- When receiving updates: 
  - Critically evaluate information from other agents
  - Verify their claims with your tools and expertise
  - Identify gaps, inconsistencies, or errors in collective analysis
- If no updates received yet:
  - Continue working on the task
  - Verify your own progress with your tools and expertise
  - Find missing pieces in the analysis or other perspectives
  - Use the `update_summary` tool to update the team on your progress

**4. Solution Validation**
  - Continuously verify information accuracy
  - Cross-check findings against multiple sources
  - Challenge assumptions and test hypotheses
  - Look for missing pieces in the analysis

**5. Consensus Building**
- Use the `vote` tool to nominate an agent as representative to present the solution
- Vote only when confident the solution is accurate and complete
- Continue working until the team reaches consensus on the best answer

**Key Principles**
- Collaborative mindset: This is a team effort - share knowledge generously and build on others' work
- Evidence-based reasoning: Always support your claims with verifiable sources and data
- Quality over speed: Prioritize accuracy and thoroughness over quick answers
- Continuous verification: Question and verify information, even from trusted team members
- You are Agent {agent_id}. That is your identifier in the team. 
"""

PROMPT_UPDATE_NOTIFICATION = """
If you have any valuable information that have not been shared with the team, please use the `update_summary` tool to record your work on the task: your analysis, approach, solution, and reasoning. 
Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents. 
If you have found any agents that have found the correct solution (include yourself), use the `vote` tool to vote for them.
Or you can choose to continue working on the task and then share your progress with the team.
"""

UPDATE_NOTIFICATION = """
[NOTIFICATION] Below are the recent updates from other agents:

{updates}
"""

NO_UPDATE_NOTIFICATION = """
There are no updates from other agents right now. 
You can continue working on the task and share your progress with the team using the `update_summary` tool.
Or you can vote for the representative agent and stop working using the `vote` tool.
"""

PRESENTATION_NOTIFICATION = """
You have been nominated as the representative agent to present the solution.

Below are the vote information of the team:

{votes}

Please incorporate all useful information from the team to present the final answer.
"""

DEBATE_NOTIFICATION = """
The team has different opinions on the representative agent to present the solution.

Below are the vote information of the team:

{votes}

Please share your opinion via `update_summary` tool, or vote again with the `vote` tool.
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

    def update_summary(self, new_content: str):
        """
        Record your work on the task: your analysis, approach, solution, and reasoning. Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents.

        Args:
            new_content: Comprehensive progress report
        """
        # Use the orchestrator to update the summary and notify other agents to restart
        self.orchestrator.notify_summary_update(self.agent_id, new_content)
        return f"Your summary report has been updated and shared with other agents."
    
    def vote(self, target_agent_id: int, response_text: str = ""):
        """
        Vote for the representative agent, who you believe has found the correct solution.

        Args:
            target_agent_id: ID of the voted representative agent
            response_text: Your full explanation of why you voted for this agent
        """
        self.orchestrator.cast_vote(self.agent_id, target_agent_id, response_text)
        return f"Your vote for Agent {target_agent_id} has been cast."
    
    def mark_failed(self, reason: str = ""):
        """
        Mark this agent as failed.

        Args:
            reason: Optional reason for the failure
        """
        # Report the failure to the orchestrator for logging
        self.orchestrator.mark_agent_failed(self.agent_id, reason)

    def _get_system_tools(self) -> List[Dict[str, Any]]:
        """
        The system tools available to this agent for orchestration:
        - update_summary: Record your work on the task: your analysis, approach, solution, and reasoning. Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents.
        - vote: Vote for the representative agent, who you believe has found the correct solution.
        """
        return [
            {
                "type": "function",
                "name": "update_summary",
                "description": "Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents. Your updated content should be fully self-contained, including your analysis, approach, solution, and reasoning on the task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_content": {
                            "type": "string",
                            "description": "Your work on the task: problem analysis, solution approach, final answer, and reasoning. Include insights from other agents if relevant."
                        }
                    },
                    "required": ["new_content"]
                }
            },
            {
                "type": "function",
                "name": "vote",
                "description": "Vote for the representative agent, who you believe has found the correct solution.",
                "parameters": {
                        "type": "object",
                        "properties": {
                            "target_agent_id": {
                                "type": "integer",
                                "description": "The ID of the agent you believe has found the correct solution."
                            },
                            "response_text": {
                                "type": "string",
                                "description": "Your full explanation of why you voted for this agent."
                            }
                        },
                        "required": ["target_agent_id", "response_text"]
                    }
                }
        ]

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Return the tool schema for the tools that are available to this agent."""
        # System tools (always available), JSON schema
        system_tools = self._get_system_tools()
        
        # Built-in tools from model config, string name only
        built_in_tools = []
        if self.tools:
            for tool in self.tools:
                if tool in ["live_search", "code_execution"]:
                    built_in_tools.append(tool)
        
        # Register tools from the global registry, JSON schema
        custom_tools = []
        from .tools import register_tool
        for tool_name, tool_func in register_tool.items():
            if tool_name in self.tools:
                tool_schema = function_to_json(tool_func)
                custom_tools.append(tool_schema)

        return {"system_tools": system_tools, "built_in_tools": built_in_tools, "custom_tools": custom_tools}

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
                if func_name == "update_summary":
                    result = self.update_summary(func_args.get("new_content", ""))
                elif func_name == "vote":
                    result = self.vote(func_args.get("agent_id", func_args.get("target_agent_id")), "")
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
      
    @abstractmethod
    def work_on_task(self, task: TaskInput, messages: List[Dict[str, str]], restart_instruction: Optional[str] = None) -> List[Dict[str, str]]:
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