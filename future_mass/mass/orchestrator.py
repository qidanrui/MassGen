"""Orchestrator for the simplified MassAgent interface."""

import asyncio
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from .agent import (
    AgentStatus,
    MassAgent,
    NotificationUpdate,
    SessionInfo,
    WorkingSummary,
)


@dataclass
class MassOrchestrationResult:
    """Result of MASS orchestration."""

    session_id: str
    task: str
    final_response: str
    representative_agent: str
    total_time: float
    agent_summaries: Dict[str, str]
    vote_results: Dict[str, str]
    total_cost: float
    session_log: List[str]


class MassOrchestrator:
    """Orchestrator for MassAgent-based collaboration."""

    def __init__(
        self,
        max_duration: float = 30.0,
        check_interval: float = 1.0,
        stream_callback: Union[Callable[[str, str], None], None] = None,
    ):
        self.max_duration = max_duration
        self.check_interval = check_interval
        self.stream_callback = stream_callback

        # Session state
        self.session_id = str(uuid.uuid4())
        self.agents: Dict[str, MassAgent] = {}
        self.session_info: Optional[SessionInfo] = None
        self.session_log: List[str] = []

        # Orchestration control
        self.start_time = 0.0
        self.convergence_reached = False
        self.votes: Dict[str, str] = {}  # agent_id -> voted_for

        # Synchronization
        self.update_lock = asyncio.Lock()

        # Multi-region display for agent streams
        self.agent_regions: Dict[str, List[str]] = {}
        self.system_messages: List[str] = []
        self.display_enabled = True

    def add_agent(self, agent: MassAgent):
        """Add an agent to the orchestration."""
        self.agents[agent.agent_id] = agent
        agent.orchestrator = self

    async def orchestrate(self, task: str) -> MassOrchestrationResult:
        """Run the full MASS orchestration process."""
        if not self.agents:
            raise ValueError("No agents added to orchestrator")

        self.start_time = time.time()
        await self._log("Starting MASS orchestration")
        await self._log(f"Task: {task}")
        await self._log(f"Agents: {list(self.agents.keys())}")

        # Initialize session info
        self.session_info = SessionInfo(
            session_id=self.session_id,
            task=task,
            start_time=self.start_time,
            current_time=self.start_time,
            agent_statuses={aid: AgentStatus.INITIALIZING.value for aid in self.agents},
            total_agents=len(self.agents),
            stopped_agents=0,
            votes_cast=0,
            time_elapsed=0.0,
            estimated_cost=0.0,
        )

        # Set session info for all agents
        for agent in self.agents.values():
            agent.session_info = self.session_info

        # Start all agents working
        await self._start_all_agents(task)

        # Monitor until convergence
        await self._monitor_until_convergence()

        # Conduct final voting if needed
        representative = await self._finalize_voting()

        # Get final answer from representative
        final_response = await self._get_final_answer(representative, task)

        # Calculate results
        total_time = time.time() - self.start_time
        total_cost = sum(
            agent.backend.get_token_usage().estimated_cost
            for agent in self.agents.values()
        )

        await self._log(f"Orchestration complete in {total_time:.1f}s")

        return MassOrchestrationResult(
            session_id=self.session_id,
            task=task,
            final_response=final_response,
            representative_agent=representative,
            total_time=total_time,
            agent_summaries={
                aid: agent.working_summary.content for aid, agent in self.agents.items()
            },
            vote_results=self.votes.copy(),
            total_cost=total_cost,
            session_log=self.session_log.copy(),
        )

    async def _start_all_agents(self, task: str):
        """Start all agents working on the task."""
        await self._log("Starting all agents")

        tasks = []
        for agent in self.agents.values():
            tasks.append(self._run_agent_workflow(agent, task))

        # Start all agent workflows concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_agent_workflow(self, agent: MassAgent, task: str):
        """Run the workflow for a single agent."""
        try:
            await self._log(f"{agent.agent_id} starting work")

            # Stream agent's work
            async for chunk in agent.work_on_task(task):
                await self._stream_output(agent.agent_id, chunk)

            await self._log(f"{agent.agent_id} completed initial work")

            # Set agent status to STOPPED after completing work
            from .agent import AgentStatus

            agent.status = AgentStatus.STOPPED

        except Exception as e:
            await self._log(f"{agent.agent_id} error: {e}")

    async def _monitor_until_convergence(self):
        """Monitor agents until convergence or timeout."""
        await self._log("Monitoring for convergence")

        while not self.convergence_reached:
            current_time = time.time()

            # Check timeout
            if current_time - self.start_time > self.max_duration:
                await self._log("Maximum duration reached")
                break

            # Update session info
            await self._update_session_info()

            # Check for agents that need restarting
            await self._check_restart_conditions()

            # Check convergence
            if await self._check_convergence():
                await self._log("Convergence reached")
                self.convergence_reached = True
                break

            await asyncio.sleep(self.check_interval)

    async def _update_session_info(self):
        """Update session information."""
        current_time = time.time()

        async with self.update_lock:
            self.session_info.current_time = current_time
            self.session_info.time_elapsed = current_time - self.start_time
            # Update agent statuses without recreating the dictionary structure
            # This preserves the agent registry so vote validation doesn't fail
            for agent_id, agent in self.agents.items():
                self.session_info.agent_statuses[agent_id] = agent.status.value
            self.session_info.stopped_agents = sum(
                1
                for agent in self.agents.values()
                if agent.status == AgentStatus.STOPPED
            )
            self.session_info.votes_cast = len(self.votes)
            self.session_info.estimated_cost = sum(
                agent.backend.get_token_usage().estimated_cost
                for agent in self.agents.values()
            )

    async def _check_restart_conditions(self):
        """Check if any agents need to be restarted."""
        for agent in self.agents.values():
            should_restart, reasons = agent.should_restart()
            if should_restart:
                await self._restart_agent(agent, reasons)

    async def _restart_agent(self, agent: MassAgent, reasons: List[str]):
        """Restart an agent with instruction addressing all applicable issues."""
        instruction_parts = []

        # Always start with basic context
        instruction_parts.append("You were stopped and need to address the following:")

        # Add specific, concise instructions for each reason (avoid repeating system message)
        if "notifications" in reasons:
            instruction_parts.append(
                f"- Process {len(agent.pending_notifications)} unprocessed notification(s)"
            )

        if "missing_vote" in reasons:
            instruction_parts.append("- Cast your vote for a representative")

        if "missing_tools" in reasons:
            instruction_parts.append("- Use required tools (see system rules)")

        # Add general continuation instruction
        instruction_parts.append("Continue working to complete these requirements.")

        instruction = "\n".join(instruction_parts)

        await self._log(f"Restarting {agent.agent_id} (reasons: {', '.join(reasons)})")

        try:
            # Use the merged work_on_task method with restart instruction
            async for chunk in agent.work_on_task(
                self.session_info.task, restart_instruction=instruction
            ):
                await self._stream_output(agent.agent_id, chunk)

            # Set agent status to STOPPED after completing restart work
            from .agent import AgentStatus

            agent.status = AgentStatus.STOPPED
        except Exception as e:
            await self._log(f"Error restarting {agent.agent_id}: {e}")

    async def _check_convergence(self) -> bool:
        """Check if convergence has been reached."""
        # Convergence is reached when all agents are stopped and not going to restart
        for agent in self.agents.values():
            # Agent must be stopped (not actively working)
            if agent.status != AgentStatus.STOPPED:
                return False

            # Agent must not need to restart
            should_restart, _ = agent.should_restart()
            if should_restart:
                return False

        return True

    async def _finalize_voting(self) -> str:
        """Finalize voting and select representative."""
        await self._log("Finalizing voting")

        # Get agents who voted (exclude those who just declared no_update)
        voting_agents = [aid for aid, agent in self.agents.items() if agent.has_voted]

        if not self.votes:
            await self._log("No votes recorded, selecting by most recent summary")
            # Select agent with most recent summary among all agents
            representative = max(
                self.agents.keys(),
                key=lambda aid: self.agents[aid].working_summary.timestamp,
            )
            await self._log(
                f"Selected {representative} as representative (most recent summary)"
            )
            return representative

        # Count votes
        vote_counts = Counter(self.votes.values())
        max_votes = max(vote_counts.values())

        # Handle ties by selecting agent with most recent summary
        tied_agents = [
            agent_id for agent_id, count in vote_counts.items() if count == max_votes
        ]

        if len(tied_agents) > 1:
            await self._log(
                f"Tie between {tied_agents}, selecting by most recent summary"
            )
            representative = max(
                tied_agents, key=lambda aid: self.agents[aid].working_summary.timestamp
            )
        else:
            representative = tied_agents[0]

        await self._log(
            f"Selected {representative} as representative ({len(voting_agents)} agents voted)"
        )
        return representative

    async def _get_final_answer(self, representative_id: str, task: str) -> str:
        """Get final answer from representative agent."""
        await self._log(f"Getting final answer from {representative_id}")

        representative = self.agents[representative_id]

        try:
            final_parts = []
            async for chunk in representative.present_final_answer(task):
                final_parts.append(chunk)
                await self._stream_output(representative_id, chunk)

            return "".join(final_parts)

        except Exception as e:
            await self._log(f"Error getting final answer: {e}")
            return f"Error: Could not get final answer from {representative_id}"

    # === System Methods (called by agents) ===

    async def notify_summary_update(self, agent_id: str, summary: WorkingSummary):
        """Handle agent summary update notification."""
        await self._log(f"{agent_id} updated summary v{summary.version}")

        # Create notification for other agents
        notification = NotificationUpdate(
            agent_id=agent_id,
            summary=summary.content,
            timestamp=time.time(),
            version=summary.version,
        )

        # Queue notification to all other agents
        for other_agent in self.agents.values():
            if other_agent.agent_id != agent_id:
                await other_agent.queue_notification(notification)

    async def register_vote(self, voter_id: str, vote_target: str):
        """Register a vote from an agent."""
        async with self.update_lock:
            self.votes[voter_id] = vote_target

        await self._log(f"{voter_id} voted for {vote_target}")

    async def get_all_summaries(self) -> Dict[str, str]:
        """Get all agent summaries (for get_session_info tool)."""
        return {
            aid: agent.working_summary.content for aid, agent in self.agents.items()
        }

    # === Logging and Streaming ===

    async def _log(self, message: str):
        """Log system message."""
        timestamp = time.time() - self.start_time if self.start_time else 0
        log_entry = f"[{timestamp:6.1f}s] SYSTEM: {message}"
        self.session_log.append(log_entry)
        self.system_messages.append(log_entry)
        if self.display_enabled:
            self._update_display()

    async def _stream_output(self, agent_id: str, content: str):
        """Stream agent output to dedicated regions."""
        if self.stream_callback:
            self.stream_callback(agent_id, content)
        else:
            # Initialize agent region if needed
            if agent_id not in self.agent_regions:
                self.agent_regions[agent_id] = []

            # Add content to agent's region
            lines = content.strip().split("\n")
            for line in lines:
                if line.strip():
                    timestamp = time.time() - self.start_time if self.start_time else 0
                    self.agent_regions[agent_id].append(f"[{timestamp:6.1f}s] {line}")

            if self.display_enabled:
                self._update_display()

    def _update_display(self):
        """Update the multi-region display."""
        import os

        # Clear screen
        os.system("clear" if os.name == "posix" else "cls")

        # Get terminal size
        try:
            terminal_width = os.get_terminal_size().columns
        except:
            terminal_width = 120

        # Calculate region widths
        agent_ids = list(self.agent_regions.keys())
        if len(agent_ids) == 0:
            # Only system messages
            print("=" * terminal_width)
            print("SYSTEM MESSAGES".center(terminal_width))
            print("=" * terminal_width)
            for msg in self.system_messages[-20:]:  # Show last 20 messages
                print(msg)
            return

        # Multi-agent layout
        region_width = (terminal_width - 5) // len(
            agent_ids
        )  # Leave space for separators

        # Print header
        print("=" * terminal_width)
        header = ""
        for i, agent_id in enumerate(agent_ids):
            if "chatgpt" in agent_id.lower():
                agent_name = "🤖 ChatGPT Agent"
            elif "grok" in agent_id.lower():
                agent_name = "🚀 Grok Agent"
            else:
                agent_name = f"🔹 {agent_id}"

            header += agent_name.center(region_width)
            if i < len(agent_ids) - 1:
                header += " │ "
        print(header)
        print("=" * terminal_width)

        # Print agent content in parallel columns
        max_lines = (
            max(len(region) for region in self.agent_regions.values())
            if self.agent_regions
            else 0
        )

        for line_idx in range(max_lines):
            row = ""
            for i, agent_id in enumerate(agent_ids):
                region = self.agent_regions[agent_id]
                if line_idx < len(region):
                    content = region[line_idx]
                    # Truncate if too long
                    if len(content) > region_width - 2:
                        content = content[: region_width - 5] + "..."
                else:
                    content = ""

                row += content.ljust(region_width)
                if i < len(agent_ids) - 1:
                    row += " │ "
            print(row)

        # Print recent system messages at bottom
        print("-" * terminal_width)
        print("SYSTEM MESSAGES:")
        for msg in self.system_messages[-5:]:  # Show last 5 system messages
            print(msg)
