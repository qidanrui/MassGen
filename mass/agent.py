import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Union, Optional, List, Dict

@dataclass
class TaskInput:
    """Represents a task to be processed by the MASS system."""

    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None


@dataclass
class AgentState:
    """Represents the current state of an agent in the MASS system."""

    agent_id: int
    status: str = "working"  # "working", "voted", or "failed"
    working_summary: str = ""
    final_answer: str = ""  # The final answer extracted from agent response
    execution_start_time: Optional[float] = None
    execution_end_time: Optional[float] = None
    update_history: List[Dict[str, Any]] = field(default_factory=list)
    vote_target: Optional[int] = None  # Which agent's solution this agent voted for
    seen_updates_timestamps: Dict[int, float] = field(default_factory=dict)  # agent_id -> last_seen_timestamp

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time if both start and end times are available."""
        if self.execution_start_time and self.execution_end_time:
            return self.execution_end_time - self.execution_start_time
        return None

    def add_update(self, summary: str, final_answer: str = "", timestamp: Optional[float] = None):
        """Add an update to the agent's history."""
        if timestamp is None:
            timestamp = time.time()

        self.update_history.append(
            {
                "timestamp": timestamp,
                "summary": summary,
                "final_answer": final_answer,
                "status": self.status,
            }
        )
        self.working_summary = summary
        if final_answer:  # Only update if a final answer is provided
            self.final_answer = final_answer

    def mark_updates_seen(self, agent_updates: Dict[int, float]):
        """Mark updates from other agents as seen."""
        for agent_id, timestamp in agent_updates.items():
            if agent_id != self.agent_id:  # Don't track own updates
                self.seen_updates_timestamps[agent_id] = timestamp

    def has_unseen_updates(self, other_agent_updates: Dict[int, float]) -> bool:
        """Check if there are unseen updates from other agents."""
        for agent_id, latest_timestamp in other_agent_updates.items():
            if agent_id != self.agent_id:  # Don't check own updates
                last_seen = self.seen_updates_timestamps.get(agent_id, 0.0)
                if latest_timestamp > last_seen:
                    return True
        return False


@dataclass
class AgentResponse:
    """Response from an agent's process_message function."""

    text: str
    code: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    function_calls: List[Dict[str, Any]] = field(default_factory=list)


class MassAgent(ABC):
    """
    Abstract base class for all agents in the MASS system.

    All agent implementations must inherit from this class and implement
    the required methods while following the standardized workflow.
    """

    def __init__(self, agent_id: int, orchestration_system=None):
        """
        Initialize the agent with an ID and reference to the orchestration system.

        Args:
            agent_id: Unique identifier for this agent
            orchestration_system: Reference to the MassOrchestrationSystem
        """
        self.agent_id = agent_id
        self.orchestration_system = orchestration_system
        self.state = AgentState(agent_id=agent_id)

    @abstractmethod
    def process_message(
        self,
        messages: List[Dict[str, str]],
        tools: List[str] = None,
        temperature: float = None,
        timeout: float = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Core LLM inference function for task processing.

        This method should handle the actual LLM interaction using the agent's
        specific backend (OpenAI, Gemini, Grok, etc.) and return a standardized response.

        Args:
            messages: List of messages in OpenAI format
            tools: List of tools available to the agent
            temperature: Sampling temperature for the LLM
            timeout: Timeout for processing
            stream: Whether to stream the response
            stream_callback: Optional callback function for streaming chunks
            **kwargs: Additional parameters specific to the agent implementation

        Returns:
            AgentResponse containing the agent's response text, code, citations, etc.
        """

    def cleanup(self):
        """
        Clean up agent resources (HTTP clients, sessions, etc.).

        This method should be implemented by agents that maintain persistent
        connections or background resources that need explicit cleanup.
        Default implementation does nothing.
        """

    def update_summary(self, summary_report: str, final_answer: str = ""):
        """
        Record working process and summary report for sharing with other agents.

        Args:
            summary_report: Comprehensive progress report including reasoning and findings
            final_answer: The final answer to the question (optional)
        """
        if self.orchestration_system:
            self.orchestration_system.update_agent_summary(self.agent_id, summary_report, final_answer)
        else:
            # Fallback: Update local state only if no orchestration system
            self.state.add_update(summary_report, final_answer)

    def check_updates(self) -> Dict[str, Any]:
        """
        Check for updates from other agents working on the same task.
        Now tracks which updates have been seen to avoid processing duplicates.

        Returns:
            Dictionary containing updates from all agents, including their summaries and states
        """
        if self.orchestration_system:
            # Get updates (only new ones by default)
            updates = self.orchestration_system.get_all_updates(exclude_agent_id=self.agent_id, check_new_only=True)

            # Mark the other agents' updates we've seen
            if updates.get("agents"):
                seen_updates = {}
                for agent_id, agent_info in updates["agents"].items():
                    if "latest_timestamp" in agent_info:
                        seen_updates[int(agent_id)] = agent_info["latest_timestamp"]

                if seen_updates:
                    self.orchestration_system.mark_updates_seen_by_agent(self.agent_id, seen_updates)

            return updates
        return {}

    def check_all_updates(self) -> Dict[str, Any]:
        """
        Check for all updates from other agents (including previously seen ones).

        Returns:
            Dictionary containing all updates from all agents
        """
        if self.orchestration_system:
            # Get all updates (including previously seen ones)
            return self.orchestration_system.get_all_updates(exclude_agent_id=self.agent_id, check_new_only=False)
        return {}

    def vote(self, target_agent_id: int, response_text: str = ""):
        """
        Vote for the representative agent.

        Args:
            target_agent_id: ID of the voted representative agent
            response_text: The full response text that led to this vote (optional)
        """
        if self.orchestration_system:
            self.orchestration_system.cast_vote(self.agent_id, target_agent_id, response_text)
        else:
            # Update local state
            self.state.status = "voted"
            self.state.vote_target = target_agent_id
            self.state.execution_end_time = time.time()

    def mark_failed(self, reason: str = ""):
        """
        Mark this agent as failed.

        Args:
            reason: Optional reason for the failure
        """
        if self.orchestration_system:
            self.orchestration_system.mark_agent_failed(self.agent_id, reason)
        else:
            # Update local state
            self.state.status = "failed"
            self.state.execution_end_time = time.time()

    def get_workflow_instructions(self, phase: str) -> str:
        """
        Get phase-specific instructions for the agent workflow.

        Args:
            phase: Current workflow phase ("initial", "collaboration", "consensus")

        Returns:
            Detailed instructions for the specified phase
        """
        instructions = {
            "initial": """
Write a clear, concise, and comprehensive report that answers the user's question.
Your report should:
- Directly address the question with a well-supported answer.
- Summarize the key facts, evidence, and reasoning steps that lead to your conclusion.
- Avoid unnecessary details, excessive length, or unrelated information.
Focus on clarity, accuracy, and logical explanation.
""",
            "collaboration": f"""
You are Agent {self.agent_id} working with other agents as a team to solve a task.

You will be shown the your previous solution and other agents' solution.

**Your Goal**:
1. Review your own solution and all peer solutions against the original task
2. Check each step, statement, assumption, and conclusion of all provided solutions
3. Evaluate whether any agent (including yourself) has found the correct solution
4. Decide:
 - If you are sure that your own or other agent's solution is 100% correct, vote for that agent to be the representative.
 - If you are not the team has found the most correct solution, continue working on finding the correct solution.

**Response Format**:
- Your response should be comprehensive, detailed and grounded.
- To clarify your decision, your response must conclude with the following response format:
```
### Voting Decision
{{"voting": agent_id or None}}
```
*Replace agent_id with the actual agent ID you're voting for, or use "None" if no satisfactory solution exists yet and you propose a new solution.*

**Guidelines**:
- Remember: You are Agent {self.agent_id}
- Carefully review all solutions alongside the original task
- Check each step, statement, information sources, assumption, and conclusion of all provided solutions
- If no satisfactory solution exists yet, you should include your updated solution in summary report, and put `None` in voting decision.
- If you find some agent (including yourself) has find the final solution, include why your believe that in summary report, and put its agent_id in voting decision
""",
            "debate": f"""
You are Agent {self.agent_id} working with other agents as a team to solve a task.

You will be shown the summary report of your own and other agents' solution.

The team has different opinions on the correct solution.

**Your Goal**:
1. Review all peer solutions and your own solution
2. Check each step, statement, information sources, assumption, and conclusion of all provided solutions
3. Evaluate whether any agent (including yourself) has found the correct solution
4. Decide: If you are sure that some agent's solution is correct, vote for that agent to be the representative. If not, continue working on finding the correct solution.

**Response Format**:
- Your response should be comprehensive, detailed and grounded.
- To clarify your decision, your response must conclude with the following response format:
```
### Voting Decision
{{"voting": agent_id or None}}
```
*Replace agent_id with the actual agent ID you're voting for, or use "None" if no satisfactory solution exists yet and you propose a new solution.*
""",
            "presentation": f"""
You are Agent {self.agent_id} working with other agents as a team to solve a task.

You will be shown the summary report of your own and other agents' solution.

You have been voted by other agents to present the final answer.

**Your Goal**:
1. Incorporate all peer solutions and opinions to form a final solution
2. Compose a final summary report in response to the original task question
3. State the final answer at the end of your response

**Response Format**:
- Your response should be comprehensive, detailed and grounded on the original task question.
- To clarify your final answer, your response must conclude with the following response format:
```
### Final answer
[final answer to the question]
```
""",
        }
        return instructions.get(phase, "")

    def process_task(
        self,
        task: TaskInput,
        phase: str = "initial",
        timeout: float = None,
        stream: bool = False,
        stream_callback: Optional[Callable] = None,
    ) -> AgentResponse:
        """
        Process a task in a specific workflow phase.
        Now leverages incremental update detection to avoid reprocessing seen updates.

        Args:
            task: The task to process
            phase: The workflow phase ("initial", "collaboration", "debate", "presentation")
            timeout: Maximum time to spend processing (in seconds)
            stream: Whether to stream the response
            stream_callback: Optional callback function for streaming chunks

        Returns:
            AgentResponse containing the agent's response
        """
        # Phase-specific orchestration handled by workflow system
        if phase == "collaboration":
            # Check for new updates to determine if we should process
            new_updates = self.check_updates()
            has_new_updates = new_updates and new_updates.get("has_new_updates", False)

            # Build context with own summary and only NEW updates from other agents
            context_parts = []

            # Include agent's own latest summary
            context_parts.append(f"Your ID: Agent {self.agent_id}")
            if self.state.working_summary:
                context_parts.append(f"Your latest summary:\n{self.state.working_summary}")

            # Include only NEW updates from other agents (if any)
            print(f"   📥 Agent {self.agent_id} has_new_updates: {has_new_updates}")
            # print(f"   📥 Agent {self.agent_id} new_updates: {new_updates}")
            # _ = input("[DEBUG] Press Enter to continue...")

            if has_new_updates and new_updates.get("agents"):
                new_agent_updates = []
                for agent_id, agent_info in new_updates["agents"].items():
                    if agent_info.get("is_new_update", False):
                        summary = agent_info.get("working_summary", "")
                        if summary:
                            new_agent_updates.append(f"Agent {agent_id} (NEW): {summary}")

                if new_agent_updates:
                    context_parts.append("New updates from other agents:")
                    context_parts.extend(new_agent_updates)

            # Create context showing only own summary + new updates from others
            full_peer_context = "\n\n" + "\n\n".join(context_parts)
            # print(f"   📥 Agent {self.agent_id} full_peer_context: {full_peer_context}")

            # Log whether there are new updates
            if has_new_updates:
                new_count = sum(
                    1 for agent_info in new_updates["agents"].values() if agent_info.get("is_new_update", False)
                )
                print(f"   📥 Agent {self.agent_id} processing with {new_count} new updates available")
            else:
                print(f"   📥 Agent {self.agent_id} processing with no new updates")

        elif phase in ["debate", "presentation"]:
            # In consensus phase, check all updates to make final decision
            updates = self.check_all_updates()

            # Include agent's own ID and latest summary for consensus phase
            own_summary_context = f"\n\nYour ID: Agent {self.agent_id}"
            if self.state.working_summary:
                own_summary_context += f"\n\nYour latest summary:\n{self.state.working_summary}"

            if updates and updates.get("agents"):
                # Get only the latest summary for each peer agent
                peer_summaries = []
                for agent_id, agent_info in updates["agents"].items():
                    # Ensure we get the latest summary - working_summary should be the most recent
                    latest_summary = agent_info.get("working_summary")
                    if latest_summary:
                        peer_summaries.append(f"Agent {agent_id}: {latest_summary}")

                if peer_summaries:
                    peer_context = "\n\nLatest solutions from all peer agents:\n" + "\n".join(peer_summaries)
                else:
                    peer_context = ""
            else:
                peer_context = ""

            # Combine own latest summary and peer latest summaries
            full_peer_context = own_summary_context + peer_context

        else:
            # Initial phase - no peer context, but include agent ID
            full_peer_context = f"\n\nYour ID: Agent {self.agent_id}"

        # Get phase-specific instructions
        instructions = self.get_workflow_instructions(phase)

        # Prepare messages for the LLM
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": "Task: " + task.question + "\n\n" + full_peer_context,
            },
        ]

        # Get available tools (only live_search and code_execution)
        tools = self._get_available_tools()

        # Process the message using the agent's specific implementation
        response = self.process_message(
            messages,
            tools=tools,
            timeout=timeout,
            stream=stream,
            stream_callback=stream_callback,
        )

        return response

    def _get_available_tools(self) -> List[str]:
        """
        Get list of tools available to this agent.

        Note: Orchestration tools (update_summary, check_updates, vote) are NOT
        exposed as tools to agents. They are called programmatically by the
        workflow system at appropriate times.
        """
        # Only live_search and code_execution are available as tools to agents for now
        return ["live_search", "code_execution"]
