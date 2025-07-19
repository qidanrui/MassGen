import logging
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mass_agent import AgentState, TaskInput
from mass_logging import get_log_manager

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class VoteRecord:
    """Records a vote cast by an agent."""

    voter_id: int
    target_id: int
    timestamp: float


@dataclass
class SystemMetrics:
    """Tracks system performance and cost metrics."""

    phase_durations: dict[str, float] = field(default_factory=dict)
    agent_execution_times: dict[int, float] = field(default_factory=dict)


@dataclass
class SystemState:
    """Overall state of the MASS orchestration system."""

    task: TaskInput | None = None
    phase: str = "initial"  # "initial", "collaboration", "debate", "presentation", "completed"
    start_time: float | None = None
    end_time: float | None = None
    consensus_reached: bool = False
    final_solution_agent_id: int | None = None
    total_rounds: int = 0
    extracted_answer: Optional[str] = None
    metrics: SystemMetrics = field(default_factory=SystemMetrics)


def extract_answer_from_summary(summary: str) -> str | None:
    """
    Extract the final answer from an agent's summary report.
    Looks for common answer patterns and formats.
    """
    if not summary:
        return None

    # Common answer patterns to look for (ordered by specificity)
    patterns = [
        # Structured answer sections
        r"(?:### Answer|###\s*Answer)\s*\n(.*?)(?=###|\Z)",
        r"(?:## Answer|##\s*Answer)\s*\n(.*?)(?=##|\Z)",
        r"(?:# Answer|#\s*Answer)\s*\n(.*?)(?=#|\Z)",
        r"(?:### Final Answer|###\s*Final Answer)\s*\n(.*?)(?=###|\Z)",
        r"(?:## Final Answer|##\s*Final Answer)\s*\n(.*?)(?=##|\Z)",
        r"(?:# Final Answer|#\s*Final Answer)\s*\n(.*?)(?=#|\Z)",
        # Direct answer statements
        r"(?:final answer|final answer is|answer|conclusion):\s*(.+?)(?:\n|$|\.|;)",
        r"(?:the answer is|answer is):\s*(.+?)(?:\n|$|\.|;)",
        r"(?:therefore|thus|so|hence),?\s+(?:the answer is)?\s*(.+?)(?:\n|$|\.|;)",
    ]

    extracted_answers = []

    for i, pattern in enumerate(patterns):
        try:
            matches = re.findall(pattern, summary, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                if match and len(match.strip()) > 0:
                    cleaned = match.strip().rstrip(".,!?")
                    if len(cleaned) > 0 and len(cleaned) < 500:  # Reasonable answer length
                        extracted_answers.append(
                            {
                                "answer": cleaned,
                                "pattern_index": i,
                                "pattern": pattern[:50],
                            }
                        )
        except Exception:
            continue

    if not extracted_answers:
        return None

    # Prioritize answers by pattern specificity (lower index = more specific)
    extracted_answers.sort(key=lambda x: (x["pattern_index"], len(x["answer"])))

    return extracted_answers[0]["answer"]


class MassOrchestrationSystem:
    """
    Central orchestration system for managing multiple agents in the MASS framework.

    This system handles:
    - Agent state management and synchronization
    - Voting and consensus building
    - Communication between agents
    - Workflow phase management
    - Session state tracking
    """

    def __init__(
        self,
        max_rounds: int = 5,
        consensus_threshold: float = 1.0,
        streaming_orchestrator=None,
    ):
        """
        Initialize the orchestration system.

        Args:
            max_rounds: Maximum number of collaboration rounds before fallback to majority vote
            consensus_threshold: Fraction of agents that must agree for consensus (1.0 = unanimous)
            streaming_orchestrator: Optional streaming orchestrator for real-time display
        """
        self.agents: dict[int, Any] = {}  # agent_id -> MassAgent instance
        self.agent_states: dict[int, AgentState] = {}
        self.votes: list[VoteRecord] = []
        self.system_state = SystemState()
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self._lock = threading.Lock()
        self.streaming_orchestrator = streaming_orchestrator

        # Communication logs
        self.communication_log: list[dict[str, Any]] = []

        # Get reference to logging system
        self.log_manager = get_log_manager()

    def register_agent(self, agent):
        """
        Register an agent with the orchestration system.

        Args:
            agent: MassAgent instance to register
        """
        with self._lock:
            self.agents[agent.agent_id] = agent
            self.agent_states[agent.agent_id] = agent.state
            agent.orchestration_system = self

    def start_task(self, task: TaskInput):
        """
        Initialize the system for a new task.

        Args:
            task: TaskInput containing the problem to solve
        """
        with self._lock:
            logger.info("🎯 ORCHESTRATION SYSTEM: Starting new task")
            logger.info(f"   Task ID: {task.task_id}")
            logger.info(f"   Question preview: {task.question}")
            logger.info(f"   Registered agents: {list(self.agents.keys())}")
            logger.info(f"   Max rounds: {self.max_rounds}")
            logger.info(f"   Consensus threshold: {self.consensus_threshold}")

            self.system_state.task = task
            self.system_state.start_time = time.time()
            self.system_state.phase = "initial"
            self.system_state.total_rounds = 0

            # Reset all agent states
            logger.debug("   Resetting all agent states...")
            for agent_id, agent in self.agents.items():
                agent.state = AgentState(agent_id=agent_id)
                self.agent_states[agent_id] = agent.state
                logger.debug(f"   - Agent {agent_id} state reset")

            # Clear previous session data
            self.votes.clear()
            self.communication_log.clear()
            logger.debug(f"   Cleared {len(self.votes)} previous votes")
            logger.debug(f"   Cleared {len(self.communication_log)} previous communications")

            self._log_event("task_started", {"task_id": task.task_id, "question": task.question})
            logger.info("✅ Task initialization completed successfully")

    def update_agent_summary(self, agent_id: int, summary: str, final_answer: str = ""):
        """
        Update an agent's working summary and final answer.

        Args:
            agent_id: ID of the agent updating their summary
            summary: New summary content
            final_answer: Final answer content (optional)
        """
        with self._lock:
            if agent_id not in self.agent_states:
                raise ValueError(f"Agent {agent_id} not registered")

            old_summary_length = len(self.agent_states[agent_id].working_summary)
            self.agent_states[agent_id].add_update(summary, final_answer)

            preview = summary[:100] + "..." if len(summary) > 100 else summary
            print(f"📝 Agent {agent_id} summary updated ({old_summary_length} → {len(summary)} chars)")
            print(f"   🔍 {preview}")

            # Log to the comprehensive logging system
            if self.log_manager:
                self.log_manager.log_agent_summary_update(
                    agent_id=agent_id,
                    summary=summary,
                    final_answer=final_answer,
                    phase=self.system_state.phase,
                )

            self._log_event(
                "summary_updated",
                {"agent_id": agent_id, "summary": summary, "timestamp": time.time()},
            )

    def get_all_updates(self, exclude_agent_id: int | None = None, check_new_only: bool = True) -> dict[str, Any]:
        """
        Get updates from all agents, optionally excluding one agent.
        Now supports tracking which updates have been seen to avoid duplicates.

        Args:
            exclude_agent_id: Agent ID to exclude from results (typically the requesting agent)
            check_new_only: If True, only return updates that haven't been seen by the requesting agent

        Returns:
            Dictionary containing all agent states and summaries (new updates only if check_new_only=True)
        """
        with self._lock:
            # Calculate max updates by any agent (this represents the effective "round")
            max_updates_by_any_agent = (
                max(len(state.update_history) for state in self.agent_states.values()) if self.agent_states else 0
            )

            updates = {
                "agents": {},
                "system_status": {
                    "phase": self.system_state.phase,
                    "total_agents": len(self.agents),
                    "voted_agents": len([s for s in self.agent_states.values() if s.status == "voted"]),
                    "working_agents": len([s for s in self.agent_states.values() if s.status == "working"]),
                    "failed_agents": len([s for s in self.agent_states.values() if s.status == "failed"]),
                    "current_round": max_updates_by_any_agent,
                },
                "voting_status": self._get_voting_status(),
                "has_new_updates": False,
            }

            requesting_agent_state = None
            if exclude_agent_id is not None and exclude_agent_id in self.agent_states:
                requesting_agent_state = self.agent_states[exclude_agent_id]

            for agent_id, state in self.agent_states.items():
                if exclude_agent_id is None or agent_id != exclude_agent_id:
                    # Get the latest timestamp for this agent's updates
                    latest_timestamp = state.update_history[-1]["timestamp"] if state.update_history else 0.0

                    # Check if this is a new update for the requesting agent
                    is_new_update = True
                    if check_new_only and requesting_agent_state:
                        # The last seen timestamp for this agent's updates
                        last_seen = requesting_agent_state.seen_updates_timestamps.get(agent_id, 0.0)
                        is_new_update = latest_timestamp > last_seen

                    if not check_new_only or is_new_update:
                        updates["agents"][agent_id] = {
                            "status": state.status,
                            "working_summary": state.working_summary,
                            "final_answer": state.final_answer,
                            "vote_target": state.vote_target,
                            "execution_time": state.execution_time,
                            "last_update": state.update_history[-1] if state.update_history else None,
                            "latest_timestamp": latest_timestamp,
                            "is_new_update": is_new_update,
                        }
                        if is_new_update:
                            updates["has_new_updates"] = True

            return updates

    def mark_updates_seen_by_agent(self, agent_id: int, seen_updates: dict[int, float]):
        """
        Mark that an agent has seen specific updates from other agents.

        Args:
            agent_id: ID of the agent marking updates as seen
            seen_updates: Dictionary of agent_id -> timestamp for updates that were seen
        """
        with self._lock:
            if agent_id in self.agent_states:
                self.agent_states[agent_id].mark_updates_seen(seen_updates)

    def check_for_reactivation(self) -> list[int]:
        """
        Check if any voted agents should be reactivated due to new updates.

        Returns:
            List of agent IDs that should be reactivated
        """
        with self._lock:
            reactivation_candidates = []

            # Get all agents that have voted
            voted_agents = [agent_id for agent_id, state in self.agent_states.items() if state.status == "voted"]

            # Check each voted agent for unseen updates
            for agent_id in voted_agents:
                agent_state = self.agent_states[agent_id]

                # Get latest timestamps from all other agents
                other_agent_timestamps = {}
                for other_id, other_state in self.agent_states.items():
                    if other_id != agent_id and other_state.update_history:
                        other_agent_timestamps[other_id] = other_state.update_history[-1]["timestamp"]

                # Check if this agent has unseen updates
                if agent_state.has_unseen_updates(other_agent_timestamps):
                    reactivation_candidates.append(agent_id)
                    logger.info(f"🔄 Agent {agent_id} marked for reactivation due to new updates")

            return reactivation_candidates

    def reactivate_agent(self, agent_id: int):
        """
        Reactivate a voted agent to working status.

        Args:
            agent_id: ID of the agent to reactivate
        """
        with self._lock:
            if agent_id in self.agent_states:
                previous_status = self.agent_states[agent_id].status
                self.agent_states[agent_id].status = "working"
                self.agent_states[agent_id].vote_target = None

                # Remove their previous vote
                self.votes = [v for v in self.votes if v.voter_id != agent_id]

                logger.info(f"🔄 REACTIVATION: Agent {agent_id} changed from '{previous_status}' to 'working'")
                print(f"      🔄 REACTIVATED: Agent {agent_id} (new updates available)")

                self._log_event(
                    "agent_reactivated",
                    {
                        "agent_id": agent_id,
                        "previous_status": previous_status,
                        "reason": "new_updates_available",
                    },
                )

    def cast_vote(self, voter_id: int, target_id: int, response_text: str = ""):
        """
        Record a vote from one agent for another agent's solution.

        Args:
            voter_id: ID of the agent casting the vote
            target_id: ID of the agent being voted for
            response_text: The full response text that led to this vote (optional)
        """
        with self._lock:
            logger.info(f"🗳️ VOTING: Agent {voter_id} casting vote")
            logger.debug(f"   Vote details: {voter_id} → {target_id}")

            print(f"🗳️  VOTE: Agent {voter_id} → Agent {target_id} ({self.system_state.phase})")
            if response_text:
                print(f"   📝 Response: {len(response_text)} chars")

            if voter_id not in self.agent_states:
                logger.error(f"   ❌ Invalid voter: Agent {voter_id} not registered")
                raise ValueError(f"Voter agent {voter_id} not registered")
            if target_id not in self.agent_states:
                logger.error(f"   ❌ Invalid target: Agent {target_id} not registered")
                raise ValueError(f"Target agent {target_id} not registered")

            # Remove any previous vote from this agent
            previous_votes = [v for v in self.votes if v.voter_id == voter_id]
            self.votes = [v for v in self.votes if v.voter_id != voter_id]
            if previous_votes:
                logger.info(f"   🔄 Changed vote from Agent {previous_votes[0].target_id} to Agent {target_id}")
            else:
                logger.info(f"   ✨ New vote for Agent {target_id}")

            # Add new vote
            vote = VoteRecord(voter_id=voter_id, target_id=target_id, timestamp=time.time())
            self.votes.append(vote)

            # Update agent state
            old_status = self.agent_states[voter_id].status
            self.agent_states[voter_id].status = "voted"
            self.agent_states[voter_id].vote_target = target_id
            self.agent_states[voter_id].execution_end_time = time.time()
            logger.debug(f"   Agent {voter_id} status updated to 'voted'")

            # Log to the comprehensive logging system
            if self.log_manager:
                print(f"      📝 Logging voting event via log_manager")
                self.log_manager.log_voting_event(
                    voter_id=voter_id,
                    target_id=target_id,
                    phase=self.system_state.phase,
                    response_text=response_text,
                )
                print(f"      📝 Logging status change via log_manager")
                self.log_manager.log_agent_status_change(
                    agent_id=voter_id,
                    old_status=old_status,
                    new_status="voted",
                    phase=self.system_state.phase,
                )
                print(f"      ✅ Voting events logged successfully")
            else:
                print(f"      ⚠️  No log_manager available - voting events not logged")

            # Show current vote distribution
            vote_counts = Counter(vote.target_id for vote in self.votes)
            logger.info(f"   📊 Vote distribution: {dict(vote_counts)}")
            logger.info(f"   📈 Voting progress: {len(self.votes)}/{len(self.agent_states)} agents voted")

            # Calculate consensus requirements
            total_agents = len(self.agent_states)
            votes_needed = max(1, int(total_agents * self.consensus_threshold))
            if vote_counts:
                leading_agent, leading_votes = vote_counts.most_common(1)[0]
                logger.info(
                    f"   🏆 Leading: Agent {leading_agent} with {leading_votes} votes (need {votes_needed} for consensus)"
                )

            # Show current vote distribution
            self._log_event(
                "vote_cast",
                {
                    "voter_id": voter_id,
                    "target_id": target_id,
                    "timestamp": vote.timestamp,
                    "vote_distribution": dict(vote_counts),
                    "total_votes": len(self.votes),
                },
            )

            # Check if consensus is reached
            consensus_reached = self._check_consensus()
            if consensus_reached:
                logger.info("🎉 CONSENSUS REACHED!")
                print(f"      🎉 CONSENSUS REACHED!")

            return consensus_reached

    def mark_agent_failed(self, agent_id: int, reason: str = ""):
        """
        Mark an agent as failed.

        Args:
            agent_id: ID of the agent to mark as failed
            reason: Optional reason for the failure
        """
        with self._lock:
            logger.info(f"💥 AGENT FAILURE: Agent {agent_id} marked as failed")
            logger.debug(f"   Failure reason: {reason}")

            print(f"      💥 MARK_FAILED: Agent {agent_id}")
            print(f"      📊 Current phase: {self.system_state.phase}")
            print(f"      📝 Reason: {reason}")

            if agent_id not in self.agent_states:
                logger.error(f"   ❌ Invalid agent: Agent {agent_id} not registered")
                raise ValueError(f"Agent {agent_id} not registered")

            # Update agent state
            old_status = self.agent_states[agent_id].status
            self.agent_states[agent_id].status = "failed"
            self.agent_states[agent_id].execution_end_time = time.time()
            logger.debug(f"   Agent {agent_id} status updated to 'failed'")

            # Log to the comprehensive logging system
            if self.log_manager:
                print(f"      📝 Logging failure event via log_manager")
                self.log_manager.log_agent_status_change(
                    agent_id=agent_id,
                    old_status=old_status,
                    new_status="failed",
                    phase=self.system_state.phase,
                )
                print(f"      ✅ Failure event logged successfully")
            else:
                print(f"      ⚠️  No log_manager available - failure event not logged")

            # Log the failure event
            self._log_event(
                "agent_failed",
                {
                    "agent_id": agent_id,
                    "reason": reason,
                    "timestamp": time.time(),
                    "old_status": old_status,
                },
            )

            # Show current agent status distribution
            status_counts = Counter(state.status for state in self.agent_states.values())
            logger.info(f"   📊 Status distribution: {dict(status_counts)}")
            logger.info(f"   📈 Failed agents: {status_counts.get('failed', 0)}/{len(self.agent_states)} total")

    def _get_voting_status(self) -> dict[str, Any]:
        """Get current voting status and distribution."""
        vote_counts = Counter(vote.target_id for vote in self.votes)
        total_agents = len(self.agents)
        failed_agents = len([s for s in self.agent_states.values() if s.status == "failed"])
        votable_agents = total_agents - failed_agents
        voted_agents = len(self.votes)

        return {
            "vote_distribution": dict(vote_counts),
            "total_agents": total_agents,
            "failed_agents": failed_agents,
            "votable_agents": votable_agents,
            "voted_agents": voted_agents,
            "votes_needed_for_consensus": max(1, int(votable_agents * self.consensus_threshold)),
            "leading_agent": vote_counts.most_common(1)[0] if vote_counts else None,
        }

    def _check_consensus(self):
        """Check if consensus has been reached and update system state accordingly."""
        vote_counts = Counter(vote.target_id for vote in self.votes)
        total_agents = len(self.agents)
        failed_agents_count = len([s for s in self.agent_states.values() if s.status == "failed"])
        votable_agents_count = total_agents - failed_agents_count

        # Calculate votes needed based on votable agents, not total agents
        votes_needed = max(1, int(votable_agents_count * self.consensus_threshold))

        # Calculate max updates by any agent (this is the new "round" concept)
        max_updates_by_any_agent = (
            max(len(state.update_history) for state in self.agent_states.values()) if self.agent_states else 0
        )

        print(f"         🔍 Checking consensus:")
        print(
            f"            👥 Total agents: {total_agents}, Failed: {failed_agents_count}, Votable: {votable_agents_count}"
        )
        print(
            f"            🎯 Votes needed: {votes_needed}/{votable_agents_count} (threshold: {self.consensus_threshold})"
        )
        print(f"            📊 Vote counts: {dict(vote_counts)}")
        print(f"            🔄 Max updates by any agent: {max_updates_by_any_agent}/{self.max_rounds}")

        # Check for unanimous consensus first
        if vote_counts and vote_counts.most_common(1)[0][1] >= votes_needed:
            winning_agent_id = vote_counts.most_common(1)[0][0]
            winning_votes = vote_counts.most_common(1)[0][1]
            print(f"            ✅ CONSENSUS: Agent {winning_agent_id} has {winning_votes} votes!")
            self._reach_consensus(winning_agent_id)
            return True

        # Check if we should force a decision (max updates reached by any agent OR all non-failed agents voted)
        should_force_decision = max_updates_by_any_agent >= self.max_rounds or len(self.votes) == votable_agents_count

        print(f"            🗳️  Current votes: {len(self.votes)}/{votable_agents_count}")
        print(f"            🔄 Should force decision: {should_force_decision}")

        if should_force_decision:
            if vote_counts:
                # Handle ties in voting by earliest vote timestamp as per README
                max_votes = vote_counts.most_common(1)[0][1]
                tied_agents = [agent_id for agent_id, votes in vote_counts.items() if votes == max_votes]

                if len(tied_agents) > 1:
                    # Tie-breaking by earliest vote timestamp as per README requirement
                    print(f"            🎲 TIE DETECTED: {len(tied_agents)} agents tied with {max_votes} votes each")

                    earliest_vote_time = float("inf")
                    winning_agent_id = tied_agents[0]

                    for agent_id in tied_agents:
                        # Find the earliest vote for this agent
                        agent_votes = [v for v in self.votes if v.target_id == agent_id]
                        if agent_votes:
                            earliest_agent_vote = min(agent_votes, key=lambda v: v.timestamp)
                            if earliest_agent_vote.timestamp < earliest_vote_time:
                                earliest_vote_time = earliest_agent_vote.timestamp
                                winning_agent_id = agent_id

                    print(
                        f"            ⏰ Tie-breaking by earliest vote: Agent {winning_agent_id} (vote at {earliest_vote_time})"
                    )
                    logger.info(f"Tie broken by earliest vote timestamp: Agent {winning_agent_id}")
                else:
                    winning_agent_id = tied_agents[0]

                reason = (
                    "MAX UPDATES REACHED"
                    if max_updates_by_any_agent >= self.max_rounds
                    else "ALL NON-FAILED AGENTS VOTED"
                )
                print(f"            🔄 {reason}: Selecting winner based on votes")
                print(f"            🏆 Agent {winning_agent_id} wins with {vote_counts[winning_agent_id]} votes")
                self._reach_consensus(winning_agent_id)
                return True
            else:
                # Edge case: No votes cast (e.g., all agents failed)
                # Select the first working agent, or if none, select any agent that has a summary
                working_agents = [aid for aid, state in self.agent_states.items() if state.status == "working"]
                agents_with_summary = [aid for aid, state in self.agent_states.items() if state.working_summary]

                if working_agents:
                    winning_agent_id = working_agents[0]
                    print(f"            🆘 NO VOTES: Selecting first working agent {winning_agent_id}")
                elif agents_with_summary:
                    winning_agent_id = agents_with_summary[0]
                    print(f"            🆘 NO VOTES: Selecting first agent with summary {winning_agent_id}")
                else:
                    # Last resort: select any agent
                    winning_agent_id = list(self.agents.keys())[0]
                    print(f"            🆘 NO VOTES: Selecting first available agent {winning_agent_id}")

                logger.warning(f"No votes cast - fallback selection: Agent {winning_agent_id}")
                self._reach_consensus(winning_agent_id)
                return True

        # Continue waiting for more votes
        if len(self.votes) < total_agents:
            remaining = total_agents - len(self.votes)
            print(f"            ⏳ Waiting for {remaining} more votes")
        elif not vote_counts:
            print(f"            ⚠️  No votes cast yet - waiting for agents to vote")

        return False

    def _reach_consensus(self, winning_agent_id: int):
        """Mark consensus as reached and finalize the system."""
        old_phase = self.system_state.phase
        self.system_state.consensus_reached = True
        self.system_state.final_solution_agent_id = winning_agent_id
        self.system_state.phase = "completed"
        self.system_state.end_time = time.time()

        # Update streaming orchestrator if available
        if self.streaming_orchestrator:
            vote_distribution = dict(Counter(vote.target_id for vote in self.votes))
            self.streaming_orchestrator.update_consensus_status(winning_agent_id, vote_distribution)

        # Log to the comprehensive logging system
        if self.log_manager:
            vote_distribution = dict(Counter(vote.target_id for vote in self.votes))
            self.log_manager.log_consensus_reached(
                winning_agent_id=winning_agent_id,
                vote_distribution=vote_distribution,
                total_rounds=self.system_state.total_rounds,
                is_fallback=False,
                phase=self.system_state.phase,
            )
            self.log_manager.log_phase_transition(
                old_phase=old_phase,
                new_phase="completed",
                additional_data={
                    "consensus_reached": True,
                    "winning_agent_id": winning_agent_id,
                    "is_fallback": False,
                },
            )

        self._log_event(
            "consensus_reached",
            {
                "winning_agent_id": winning_agent_id,
                "fallback_to_majority": False,
                "total_rounds": self.system_state.total_rounds,
                "final_vote_distribution": dict(Counter(vote.target_id for vote in self.votes)),
            },
        )

    def _log_event(self, event_type: str, data: dict[str, Any]):
        """Log a orchestration system event."""
        self.communication_log.append({"timestamp": time.time(), "event_type": event_type, "data": data})

    def get_final_solution(self) -> dict[str, Any] | None:
        """
        Get the final solution from the consensus winner with extracted answer.
        Returns None if no consensus has been reached.

        Returns:
            Dictionary containing the final solution details, or None if no consensus
        """
        # Return None if consensus not reached - no forcing
        if not self.system_state.consensus_reached:
            logger.warning("No consensus reached - returning None")
            return None

        winning_agent_id = self.system_state.final_solution_agent_id
        winning_agent_state = self.agent_states[winning_agent_id]

        # Extract answer from the winning solution
        extracted_answer = extract_answer_from_summary(winning_agent_state.working_summary)

        # Store the extracted answer
        self.system_state.extracted_answer = extracted_answer

        # Also use any pre-parsed final answer from the winning solution
        if not extracted_answer and winning_agent_state.final_answer:
            extracted_answer = winning_agent_state.final_answer

        # Calculate final metrics
        total_runtime = (
            (self.system_state.end_time - self.system_state.start_time)
            if (self.system_state.end_time and self.system_state.start_time)
            else 0
        )

        # Get vote distribution
        vote_counts = Counter(vote.target_id for vote in self.votes)

        # Collect all agent summaries
        all_summaries = {}
        for agent_id, state in self.agent_states.items():
            all_summaries[agent_id] = {
                "working_summary": state.working_summary,
                "final_answer": state.final_answer,
                "status": state.status,
                "execution_time": state.execution_time,
                "vote_target": state.vote_target,
            }

        final_solution = {
            "agent_id": winning_agent_id,
            "solution": winning_agent_state.working_summary,
            "extracted_answer": extracted_answer,
            "execution_time": winning_agent_state.execution_time,
            "total_rounds": self.system_state.total_rounds,
            "vote_distribution": dict(vote_counts),
            "all_agent_summaries": all_summaries,
            "total_runtime": total_runtime,
            "consensus_method": "natural",
        }

        # Log task completion to the comprehensive logging system
        if self.log_manager:
            self.log_manager.log_task_completion(final_solution)

        return final_solution
    
    def start_phase_timing(self, phase: str):
        """Start timing for a specific phase."""
        self.system_state.metrics.phase_durations[f"{phase}_start"] = time.time()

    def end_phase_timing(self, phase: str):
        """End timing for a specific phase."""
        start_key = f"{phase}_start"
        if start_key in self.system_state.metrics.phase_durations:
            duration = time.time() - self.system_state.metrics.phase_durations[start_key]
            self.system_state.metrics.phase_durations[phase] = duration
            del self.system_state.metrics.phase_durations[start_key]

    def record_agent_execution_time(self, agent_id: int, execution_time: float):
        """Record execution time for a specific agent."""
        self.system_state.metrics.agent_execution_times[agent_id] = execution_time

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status information."""
        return {
            "phase": self.system_state.phase,
            "consensus_reached": self.system_state.consensus_reached,
            "total_rounds": self.system_state.total_rounds,
            "agents": {
                agent_id: {
                    "status": state.status,
                    "has_summary": bool(state.working_summary),
                    "vote_target": state.vote_target,
                    "execution_time": state.execution_time,
                }
                for agent_id, state in self.agent_states.items()
            },
            "voting_status": self._get_voting_status(),
            "runtime": (time.time() - self.system_state.start_time) if self.system_state.start_time else 0,
        }

    def export_detailed_session_log(self) -> dict[str, Any]:
        """
        Export complete detailed session information for comprehensive analysis.
        Includes all outputs, metrics, and evaluation results.
        """
        session_log = {
            "session_metadata": {
                "session_id": f"mass_session_{int(self.system_state.start_time)}"
                if self.system_state.start_time
                else None,
                "start_time": self.system_state.start_time,
                "end_time": self.system_state.end_time,
                "total_duration": (self.system_state.end_time - self.system_state.start_time)
                if self.system_state.start_time and self.system_state.end_time
                else None,
                "timestamp": datetime.now().isoformat(),
                "system_version": "MASS v1.0",
            },
            "task_information": {
                "question": self.system_state.task.question if self.system_state.task else None,
                "task_id": self.system_state.task.task_id if self.system_state.task else None,
                "context": self.system_state.task.context if self.system_state.task else None,
            },
            "system_configuration": {
                "max_rounds": self.max_rounds,
                "consensus_threshold": self.consensus_threshold,
                "total_agents": len(self.agents),
                "agent_types": [type(agent).__name__ for agent in self.agents.values()],
            },
            "execution_results": {
                "consensus_reached": self.system_state.consensus_reached,
                "final_solution_agent_id": self.system_state.final_solution_agent_id,
                "total_rounds_completed": self.system_state.total_rounds,
                "extracted_answer": self.system_state.extracted_answer
            },
            "system_metrics": {
                "performance": {
                    "phase_durations": self.system_state.metrics.phase_durations,
                    "agent_execution_times": self.system_state.metrics.agent_execution_times,
                },
                "efficiency": {
                    "avg_time_per_agent": sum(self.system_state.metrics.agent_execution_times.values())
                    / len(self.system_state.metrics.agent_execution_times)
                    if self.system_state.metrics.agent_execution_times
                    else 0
                },
            },
            "agent_details": {
                agent_id: {
                    "final_status": state.status,
                    "working_summary": state.working_summary,
                    "final_answer": state.final_answer,
                    "vote_target": state.vote_target,
                    "execution_time": state.execution_time,
                    "update_history": state.update_history,
                    "seen_updates_timestamps": state.seen_updates_timestamps,
                    "total_updates": len(state.update_history),
                }
                for agent_id, state in self.agent_states.items()
            },
            "voting_analysis": {
                "vote_records": [
                    {
                        "voter_id": vote.voter_id,
                        "target_id": vote.target_id,
                        "timestamp": vote.timestamp,
                    }
                    for vote in self.votes
                ],
                "final_vote_distribution": dict(Counter(vote.target_id for vote in self.votes)),
            },
            "communication_log": self.communication_log,
            "final_solution": self.get_final_solution() if self.system_state.consensus_reached else None,
        }

        return session_log
