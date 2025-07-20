import os
import sys
from typing import Callable, Union, Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()

from .types import TaskInput, AgentState, AgentResponse, ModelConfig
from .utils import get_agent_type_from_model, function_to_json
from abc import ABC, abstractmethod
from typing import Any, Callable, Union, Optional, List, Dict
from backends import oai, gemini, grok


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

PRESENTATION_NOTIFICATION = """
You have been nominated as the representative agent to present the solution.

Below are the vote information of the team:
{options}

Please incorporate all useful information from the team to present the final answer.
"""

DEBATE_NOTIFICATION = """
The team has different opinions on the representative agent to present the solution.

Below are the vote information of the team:
{options}

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

            return AgentResponse(
                text=result.get("text", ""),
                code=result.get("code", []),
                citations=result.get("citations", []),
                function_calls=result.get("function_calls", []),
            )
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
    
    def check_updates(self, latest_only: bool = True) -> Dict[str, Any]:
        """
        Check for updates from other agents working on the same task.
        Tracks which updates have been seen to avoid processing duplicates.

        Args:
            latest_only: If True, only return the latest update for each agent. Otherwise, return all updates since the last check.

        Returns:
            Dictionary containing updates from all agents, including their summaries and states
        """
        updates = {}
        
        if self.orchestrator:
            # Get updates (only new ones by default)
            session_info = self.orchestrator.get_session_info()
            agent_states = session_info["agent_states"]

            for agent_id, state in agent_states.items():
                if agent_id != self.agent_id:
                    last_seen = self.state.seen_updates_timestamps.get(agent_id, 0.0)
                    for update in state.update_history:
                        if update.timestamp > last_seen:
                            updates[agent_id].append(update)

                    if latest_only:
                        # keep only the latest update for this agent
                        updates[agent_id] = updates[agent_id][-1] 
                    
                    # record the last seen timestamp for this agent
                    self.state.seen_updates_timestamps[agent_id] = updates[agent_id][-1].timestamp

        return updates

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
                "function": {
                    "name": "update_summary",
                    "description": "Record your work on the task: your analysis, approach, solution, and reasoning. Update when you solve the problem, find better solutions, or incorporate valuable insights from other agents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary_report": {
                                "type": "string",
                                "description": "Your work on the task: problem analysis, solution approach, final answer, and reasoning. Include insights from other agents if relevant."
                            }
                        },
                        "required": ["summary_report"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
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
            }
        ]

    def _build_update_message(self) -> str:
        """
        Build the information message of all recent updates
        
        Example:
        **Agent 1 [1726752000]:**
        I have found the solution to the task.
        **Agent 2 [1726752100]:**
        I have found the solution to the task.
        **Agent 3 [1726752200]:**
        I have found the solution to the task.
        """
        updates = self.check_updates()
        # Build the information message of all recent updates
        update_message = ""
        if updates:
            for agent_id, agent_updates in updates.items():
                for update in agent_updates:
                    update_message += f"**Agent {agent_id} [{update.timestamp}]:**\n{update.summary}\n\n"
        return update_message
    
    def _build_vote_message(self) -> str:
        """
        Build the information message of all votes
        
        Example:
        **Agent 1 vote for Agent 2**
        Reason: I believe Agent 2 has found the correct solution.
        
        **Agent 2 vote for Agent 1**
        Reason: I believe Agent 1 has found the correct solution.
        
        **Agent 3 vote for Agent 1**
        Reason: I believe Agent 1 has found the correct solution.
        """
        votes = self.orchestrator.votes
        vote_message = ""
        if votes:
            for vote in votes:
                vote_message += f"**Agent {vote.voter_id} vote for Agent {vote.target_id}**\n"
                if vote.reason:
                    vote_message += f"Reason: "
        return vote_message
        
    def get_instruction_based_on_status(self, status: str) -> List[Dict[str, str]]:
        """
        Build the messages for the agent based on the status:
        - system: The system instruction for the agent
        - notification: The notification message for the agent
        - debate: The message for the agent to debate the solution
        - presentation: The message for the agent to present the solution, including the options to vote for
        """
        assert status in ["system", "notification", "debate", "presentation"], "Invalid status"
        
        if status == "system":
            peer_agents = [agent.agent_id for agent in self.orchestrator.agents if agent.agent_id != self.agent_id]
            system_instruction = SYSTEM_INSTRUCTION.format(agent_id=self.agent_id, 
                        peer_agents=", ".join([str(agent) for agent in peer_agents]))
            return system_instruction
        
        elif status == "notification":
            # check if there are updates from other agents
            update_message = self._build_update_message()
            if update_message:
                return PROMPT_UPDATE_NOTIFICATION + UPDATE_NOTIFICATION.format(updates=update_message)
            else:
                return PROMPT_UPDATE_NOTIFICATION
            
        elif status == "presentation":
            # Build the information message of all votes
            vote_message = self._build_vote_message()
            return PRESENTATION_NOTIFICATION.format(options=vote_message)
        
        elif status == "debate":
            # Build the information message of all votes
            vote_message = self._build_vote_message()
            return DEBATE_NOTIFICATION.format(options=vote_message)
        
        return messages
        
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
        
                 return system_tools + built_in_tools + custom_tools
         
    def _check_if_final_presentation(self, response_text: str):
        """
        Check if this is the final presentation and capture it.
        Called when the representative agent responds during consensus phase.
        """
        if (self.orchestrator and 
            self.orchestrator.system_state.phase == "consensus" and
            self.orchestrator.system_state.representative_agent_id == self.agent_id):
            
            # This is the representative agent presenting the final answer
            self.orchestrator.capture_final_response(response_text)
         
    def _execute_function_calls(self, function_calls: List[Dict], messages: List[Dict]):
        """Execute function calls and add results to conversation."""
        from .tools import register_tool
        function_outputs = []
        
        for func_call in function_calls:
            func_name = func_call.get("name")
            func_args = func_call.get("arguments", {})
            func_call_id = func_call.get("call_id")
            
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
                
            except Exception as e:
                # Handle execution errors
                error_output = {
                    "type": "function_call_output", 
                    "call_id": func_call_id,
                    "output": f"Error executing function: {str(e)}"
                }
                function_outputs.append(error_output)
                # print(f"Error executing function {function_name}: {e}")
                
                 return function_outputs
      
    @abstractmethod
    def work_on_task(self, task: TaskInput):
        """
        Work on the task.
        
        This method should be implemented by concrete agent classes.
        """
        pass