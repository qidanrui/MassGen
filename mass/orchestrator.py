import logging
import re
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Any, Optional, Union, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from .types import SystemState, AgentState, TaskInput, VoteRecord
from .logging import get_log_manager

# Set up logging
logger = logging.getLogger(__name__)


class MassOrchestrator:
    """
    Central orchestrator for managing multiple agents in the MASS framework.

    Simplified workflow:
    1. Agents work on task (status: "working")
    2. When agents vote, they become "voted" 
    3. When all votable agents have voted:
       - Check consensus
       - If consensus reached: select representative to present final answer
       - If no consensus: restart all agents for debate
    4. Representative presents final answer and system completes
    """

    def __init__(
        self,
        max_duration: int = 600, # seconds
        consensus_threshold: float = 1.0,
        streaming_orchestrator=None,
    ):
        """
        Initialize the orchestrator.

        Args:
            max_duration: Maximum duration for the entire task in seconds
            consensus_threshold: Fraction of agents that must agree for consensus (1.0 = unanimous)
            streaming_orchestrator: Optional streaming orchestrator for real-time display
        """
        self.agents: Dict[int, Any] = {}  # agent_id -> MassAgent instance
        self.agent_states: Dict[int, AgentState] = {} # agent_id -> AgentState instance
        self.votes: List[VoteRecord] = []
        self.system_state = SystemState()
        self.max_duration = max_duration
        self.consensus_threshold = consensus_threshold
        self.streaming_orchestrator = streaming_orchestrator

        # Simplified coordination
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Communication and logging
        self.communication_log: List[Dict[str, Any]] = []
        self.final_response: Optional[str] = None
        
        # Initialize log manager
        self.log_manager = get_log_manager()

    def register_agent(self, agent):
        """
        Register an agent with the orchestrator.

        Args:
            agent: MassAgent instance to register
        """
        with self._lock:
            self.agents[agent.agent_id] = agent
            self.agent_states[agent.agent_id] = agent.state
            agent.orchestrator = self

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an orchestrator event."""
        self.communication_log.append({"timestamp": time.time(), "event_type": event_type, "data": data})

    def update_agent_summary(self, agent_id: int, summary: str):
        """
        Update an agent's working summary.

        Args:
            agent_id: ID of the agent updating their summary
            summary: New summary content
        """
        with self._lock:
            if agent_id not in self.agent_states:
                raise ValueError(f"Agent {agent_id} not registered")

            old_summary_length = len(self.agent_states[agent_id].working_summary)
            self.agent_states[agent_id].add_update(summary)

            preview = summary[:100] + "..." if len(summary) > 100 else summary
            print(f"📝 Agent {agent_id} summary updated ({old_summary_length} → {len(summary)} chars)")
            print(f"   🔍 {preview}")

            # Log to the comprehensive logging system
            if self.log_manager:
                self.log_manager.log_agent_summary_update(
                    agent_id=agent_id,
                    summary=summary,
                    phase=self.system_state.phase,
                )

            self._log_event(
                "summary_updated",
                {"agent_id": agent_id, "summary": summary, "timestamp": time.time()},
            )

    def _get_voting_status(self) -> Dict[str, Any]:
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
        
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get the current session information, including all agent states, system status, and voting status.

        Returns:
            Dictionary containing agent states, system status, and voting status
        """
        with self._lock:
            session_info = {
                "agent_states": self.agent_states,
                "system_status": {
                    "phase": self.system_state.phase,
                    "total_agents": len(self.agents),
                    "voted_agents": len([s for s in self.agent_states.values() if s.status == "voted"]),
                    "working_agents": len([s for s in self.agent_states.values() if s.status == "working"]),
                    "failed_agents": len([s for s in self.agent_states.values() if s.status == "failed"]),
                },
                "voting_status": self._get_voting_status(),
            }
            return session_info
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        return {
            "phase": self.system_state.phase,
            "consensus_reached": self.system_state.consensus_reached,
            "agents": {
                agent_id: {
                    "status": state.status,
                    "update_times": len(state.update_history),
                    "chat_length": len(state.chat_history),
                    "chat_round": state.chat_round,
                    "vote_target": state.vote_target,
                    "execution_time": state.execution_time,
                }
                for agent_id, state in self.agent_states.items()
            },
            "voting_status": self._get_voting_status(),
            "runtime": (time.time() - self.system_state.start_time) if self.system_state.start_time else 0,
        }

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
                print(f"   📝 Voting reason: {len(response_text)} chars")

            if voter_id not in self.agent_states:
                logger.error(f"   ❌ Invalid voter: Agent {voter_id} not registered")
                raise ValueError(f"Voter agent {voter_id} not registered")
            if target_id not in self.agent_states:
                logger.error(f"   ❌ Invalid target: Agent {target_id} not registered")
                raise ValueError(f"Target agent {target_id} not registered")

            # Remove any previous vote from this agent
            previous_vote = self.agent_states[voter_id].vote_target
            if previous_vote:
                logger.info(f"   🔄 Agent {voter_id} changed vote from Agent {previous_vote} to Agent {target_id}")
            else:
                logger.info(f"   ✨ Agent {voter_id} new vote for Agent {target_id}")

            # Add new vote
            vote = VoteRecord(voter_id=voter_id, 
                              target_id=target_id, 
                              reason=response_text,
                              timestamp=time.time())
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
            
    def _check_consensus(self) -> bool:
        """Check if consensus has been reached based on current votes."""
        with self._lock:
            total_agents = len(self.agents)
            failed_agents_count = len([s for s in self.agent_states.values() if s.status == "failed"])
            votable_agents_count = total_agents - failed_agents_count
            
            if votable_agents_count == 0:
                return False
                
            vote_counts = Counter(vote.target_id for vote in self.votes)
            votes_needed = max(1, int(votable_agents_count * self.consensus_threshold))
            
            if vote_counts and vote_counts.most_common(1)[0][1] >= votes_needed:
                winning_agent_id = vote_counts.most_common(1)[0][0]
                self._reach_consensus(winning_agent_id)
                return True
                
            return False

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

    def _update_system_status(self):
        """Update the system status based on the agent states and votes."""
        with self._lock:
            total_agents = len(self.agents)
            failed_agents_count = len([s for s in self.agent_states.values() if s.status == "failed"])
            voted_agents_count = len([s for s in self.agent_states.values() if s.status == "voted"])
            working_agents_count = len([s for s in self.agent_states.values() if s.status == "working"])
            votable_agents_count = total_agents - failed_agents_count
            vote_counts = Counter(vote.target_id for vote in self.votes)
            votes_needed = max(1, int(votable_agents_count * self.consensus_threshold))
            
            print(f" 🔍 Checking System Status:")
            print(
                f"  👥 Total agents: {total_agents}, Failed: {failed_agents_count}, Votable: {votable_agents_count}, Voted: {voted_agents_count}"
            )
            print(f"  📊 Vote counts: {dict(vote_counts)}")
            print(f"  🎯 Votes needed: {votes_needed}")
            print(f"  🔍 System status: {self.system_state.phase}")
            
            if votable_agents_count and vote_counts.most_common(1)[0][1] >= votes_needed:
                # Voted agents have reached consensus
                winning_agent_id = vote_counts.most_common(1)[0][0]
                winning_votes = vote_counts.most_common(1)[0][1]
                print(f"  ✅ CONSENSUS: Agent {winning_agent_id} has {winning_votes} votes!")
                self._reach_consensus(winning_agent_id)
            elif votable_agents_count and (votable_agents_count == voted_agents_count) and vote_counts.most_common(1)[0][1] < votes_needed:
                # All votable agents have voted, but consensus is not reached
                self.system_state.phase = "debate"
            else:
                self.system_state.phase = "working"

    def _reach_consensus(self, winning_agent_id: int):
        """Mark consensus as reached and finalize the system."""
        old_phase = self.system_state.phase
        self.system_state.consensus_reached = True
        self.system_state.representative_agent_id = winning_agent_id
        self.system_state.phase = "consensus"
        # self.system_state.end_time = time.time()

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
                "final_vote_distribution": dict(Counter(vote.target_id for vote in self.votes)),
            },
        )

    def export_detailed_session_log(self) -> Dict[str, Any]:
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
                "max_duration": self.max_duration,
                "consensus_threshold": self.consensus_threshold,
                "agents": [agent.model for agent in self.agents.values()],
            },
            "execution_results": {
                "consensus_reached": self.system_state.consensus_reached,
                "representative_agent_id": self.system_state.representative_agent_id,
                "final_response": self.final_response,
            },
            "agent_details": {
                agent_id: {
                    "update_times": len(state.update_history),
                    "chat_length": len(state.chat_history),
                    "chat_round": state.chat_round,
                    "vote_target": state.vote_target,
                    "execution_time": state.execution_time,
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
            "final_response": self.final_response,
        }

        return session_log 
    
    def start_task(self, task: TaskInput):
        """
        Initialize the system for a new task and run the main workflow.

        Args:
            task: TaskInput containing the problem to solve
            
        Returns:
            response: Dict[str, Any] containing the final answer to the task's question, and relevant information
        """
        with self._lock:
            logger.info("🎯 ORCHESTRATOR: Starting new task")
            logger.info(f"   Task ID: {task.task_id}")
            logger.info(f"   Question preview: {task.question}")
            logger.info(f"   Registered agents: {list(self.agents.keys())}")
            logger.info(f"   Max duration: {self.max_duration}")
            logger.info(f"   Consensus threshold: {self.consensus_threshold}")

            self.system_state.task = task
            self.system_state.start_time = time.time()
            self.system_state.phase = "collaboration"
            self.final_response = None

            # Reset all agent states
            logger.debug("   Resetting all agent states...")
            for agent_id, agent in self.agents.items():
                agent.state = AgentState(agent_id=agent_id)
                self.agent_states[agent_id] = agent.state
                # Initialize conversation with task question
                agent.state.chat_history = [
                    {"role": "system", "content": self._get_system_instruction(agent_id)},
                    {"role": "user", "content": task.question}
                ]
                logger.debug(f"   - Agent {agent_id} state reset")

            # Clear previous session data
            self.votes.clear()
            self.communication_log.clear()
            logger.debug(f"   Cleared previous session data")

            self._log_event("task_started", {"task_id": task.task_id, "question": task.question})
            logger.info("✅ Task initialization completed successfully")
            
        # Run the simplified workflow
        return self._run_simplified_workflow(task)

    def _get_system_instruction(self, agent_id: int) -> str:
        """Get system instruction for an agent."""
        peer_agents = [str(aid) for aid in self.agents.keys() if aid != agent_id]
        return f"""You are an expert agent equipped with tools to work as part of a collaborative team to solve complex tasks.

You are Agent {agent_id}, working collaboratively with Peer Agents {peer_agents} to solve complex tasks.

**Communication:** You can only use the `update_summary` tool to communicate with other agents. All information shared through this tool is visible to the entire team.

### Core Workflow

**1. Task Execution**
- Use your available tools (search, code analysis, etc.) to investigate and solve the assigned task
- Apply expertise to search for information, analyze data, identify patterns, and develop solutions

**2. Progress Documentation**
- Use the `update_summary` tool regularly to record your findings, hypotheses, and progress
- Document your reasoning process so other agents can understand and build upon your work
- Include supporting evidence and key insights

**3. Collaboration & Information Sharing**
- When sharing insights: Always provide supporting evidence so other agents can verify your findings
- When receiving updates: Critically evaluate information from other agents
- Build upon each other's work rather than duplicating efforts

**4. Solution Validation**
- Continuously verify information accuracy and look for missing pieces
- Cross-check findings against multiple sources when possible

**5. Consensus Building**
- Use the `vote` tool to nominate an agent as representative to present the solution
- Vote only when confident the solution is accurate and complete
- Continue working until the team reaches consensus on the best answer

**Key Principles**
- Collaborative mindset: This is a team effort - share knowledge generously
- Evidence-based reasoning: Always support your claims with verifiable sources
- Quality over speed: Prioritize accuracy and thoroughness over quick answers
- You are Agent {agent_id}. That is your identifier in the team.
"""

    def _run_simplified_workflow(self, task: TaskInput) -> Dict[str, Any]:
        """
        Run the simplified MASS workflow:
        1. All agents work in parallel
        2. When all have voted, check consensus
        3. If no consensus, restart all for debate
        4. If consensus, representative presents final answer
        """
        logger.info("🚀 Starting simplified MASS workflow")
        
        debate_rounds = 0
        max_debate_rounds = 3
        start_time = time.time()
        
        while not self._stop_event.is_set():
            # Check timeout
            if time.time() - start_time > self.max_duration:
                logger.warning("⏰ Maximum duration reached - forcing consensus")
                self._force_consensus_by_timeout()
                break
            
            # Run all agents in parallel
            logger.info(f"📢 Starting collaboration round {debate_rounds + 1}")
            self._run_all_agents_parallel(task)
            
            # Check if all votable agents have voted
            if self._all_agents_voted():
                logger.info("🗳️ All agents have voted - checking consensus")
                
                if self._check_consensus():
                    logger.info("🎉 Consensus reached!")
                    # Representative will present final answer
                    self._run_final_presentation(task)
                    break
                else:
                    # No consensus - start debate round
                    debate_rounds += 1
                    if debate_rounds >= max_debate_rounds:
                        logger.warning(f"⚠️ Maximum debate rounds ({max_debate_rounds}) reached")
                        self._force_consensus_by_timeout()
                        break
                    
                    logger.info(f"🗣️ No consensus - starting debate round {debate_rounds}")
                    self._restart_all_agents_for_debate()
            else:
                # Still waiting for some agents to vote
                time.sleep(1)
                
        return self._finalize_session()

    def _run_all_agents_parallel(self, task: TaskInput):
        """Run all agents in parallel until they vote or fail."""
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all agent tasks
            futures = []
            for agent_id in self.agents.keys():
                if self.agent_states[agent_id].status not in ["failed"]:
                    # Reset status to working
                    self.agent_states[agent_id].status = "working"
                    self.agent_states[agent_id].execution_start_time = time.time()
                    
                    future = executor.submit(self._run_single_agent, agent_id, task)
                    futures.append(future)
            
            # Wait for all agents to complete (vote or fail)
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"❌ Agent task failed: {e}")

    def _run_single_agent(self, agent_id: int, task: TaskInput):
        """Run a single agent's work_on_task method."""
        agent = self.agents[agent_id]
        try:
            logger.info(f"🤖 Agent {agent_id} starting work")
            
            # Get restart instruction if there are updates from other agents
            restart_instruction = self._get_restart_instruction(agent_id)
            
            # Run agent's work_on_task with current conversation state
            current_messages = self.agent_states[agent_id].chat_history.copy()
            updated_messages = agent.work_on_task(task, current_messages, restart_instruction)
            
            # Update conversation state
            self.agent_states[agent_id].chat_history = updated_messages
            self.agent_states[agent_id].chat_round += 1
            
            logger.info(f"✅ Agent {agent_id} completed work with status: {self.agent_states[agent_id].status}")
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_id} failed: {e}")
            self.mark_agent_failed(agent_id, str(e))

    def _get_restart_instruction(self, agent_id: int) -> Optional[str]:
        """Get restart instruction based on updates from other agents."""
        agent_state = self.agent_states[agent_id]
        
        # Get updates from other agents since this agent last saw them
        unseen_updates = []
        for other_id, other_state in self.agent_states.items():
            if other_id != agent_id and other_state.update_history:
                last_seen = agent_state.seen_updates_timestamps.get(other_id, 0)
                for update in other_state.update_history:
                    if update.timestamp > last_seen:
                        unseen_updates.append(f"Agent {other_id}: {update.summary}")
        
        if unseen_updates:
            # Mark updates as seen
            for other_id, other_state in self.agent_states.items():
                if other_id != agent_id and other_state.update_history:
                    latest_timestamp = max(update.timestamp for update in other_state.update_history)
                    agent_state.seen_updates_timestamps[other_id] = latest_timestamp
            
            return f"[NOTIFICATION] Recent updates from other agents:\n" + "\n".join(unseen_updates)
        
        return None

    def _all_agents_voted(self) -> bool:
        """Check if all votable agents have voted."""
        votable_agents = [aid for aid, state in self.agent_states.items() 
                         if state.status not in ["failed"]]
        voted_agents = [aid for aid, state in self.agent_states.items() 
                       if state.status == "voted"]
        
        return len(voted_agents) == len(votable_agents) and len(votable_agents) > 0

    def _restart_all_agents_for_debate(self):
        """Restart all agents for debate by resetting their status and clearing votes."""
        logger.info("🔄 Restarting all agents for debate")
        
        with self._lock:
            # Clear all votes
            self.votes.clear()
            
            # Reset agent statuses (but keep their conversation history)
            for agent_id, state in self.agent_states.items():
                if state.status not in ["failed"]:
                    state.status = "working"
                    state.vote_target = None
                    logger.debug(f"   Agent {agent_id} reset for debate")

    def _run_final_presentation(self, task: TaskInput):
        """Run the final presentation by the representative agent."""
        representative_id = self.system_state.representative_agent_id
        if not representative_id:
            logger.error("No representative agent selected")
            return
            
        logger.info(f"🎯 Agent {representative_id} presenting final answer")
        
        try:
            agent = self.agents[representative_id]
            agent_state = self.agent_states[representative_id]
            
            # Add presentation instruction to conversation
            presentation_msg = self._get_presentation_instruction()
            current_messages = agent_state.chat_history.copy()
            current_messages.append({"role": "user", "content": presentation_msg})
            
            # Agent presents final answer
            updated_messages = agent.work_on_task(task, current_messages, None)
            
            # Extract final response from the last assistant message
            for msg in reversed(updated_messages):
                if msg.get("role") == "assistant":
                    self.final_response = msg.get("content", "")
                    break
            
            self.system_state.phase = "completed"
            self.system_state.end_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Final presentation failed: {e}")
            self.final_response = f"Error in final presentation: {str(e)}"

    def _get_presentation_instruction(self) -> str:
        """Get instruction for final presentation."""
        vote_info = self._get_voting_status()
        return f"""You have been selected as the representative agent to present the final solution.

Vote distribution: {vote_info['vote_distribution']}

Please incorporate all useful information from the team to present the final answer clearly and comprehensively."""

    # Old complex methods removed - replaced by simplified workflow above
             
    def _force_consensus_by_timeout(self):
        """
        Force consensus selection when maximum duration is reached.
        """
        logger.warning("⏰ Forcing consensus due to timeout")
        
        with self._lock:
            # Find agent with most votes, or earliest voter in case of tie
            vote_counts = Counter(vote.target_id for vote in self.votes)
            
            if vote_counts:
                # Select agent with most votes
                winning_agent_id = vote_counts.most_common(1)[0][0]
                logger.info(f"   Selected Agent {winning_agent_id} with {vote_counts[winning_agent_id]} votes")
            else:
                # No votes - select first working agent
                working_agents = [aid for aid, state in self.agent_states.items() 
                                if state.status == "working"]
                winning_agent_id = working_agents[0] if working_agents else list(self.agents.keys())[0]
                logger.info(f"   No votes - selected Agent {winning_agent_id} as fallback")
                
            self._reach_consensus(winning_agent_id)
            
    def _finalize_session(self) -> Dict[str, Any]:
        """
        Finalize the session and return comprehensive results.
        """
        logger.info("🏁 Finalizing session")
        
        with self._lock:
            if not self.system_state.end_time:
                self.system_state.end_time = time.time()
                
            session_duration = (self.system_state.end_time - self.system_state.start_time 
                               if self.system_state.start_time else 0)
            
            # Prepare final results
            result = {
                "answer": self.final_response or "No final answer generated",
                "consensus_reached": self.system_state.consensus_reached,
                "representative_agent_id": self.system_state.representative_agent_id,
                "session_duration": session_duration,
                "agent_summary": {
                    agent_id: {
                        "status": state.status,
                        "updates_count": len(state.update_history),
                        "vote_target": state.vote_target,
                        "execution_time": state.execution_time
                    }
                    for agent_id, state in self.agent_states.items()
                },
                "voting_results": {
                    "votes": [{"voter": v.voter_id, "target": v.target_id} for v in self.votes],
                    "distribution": dict(Counter(v.target_id for v in self.votes))
                },
                "system_logs": self.export_detailed_session_log()
            }
            
            logger.info(f"✅ Session completed in {session_duration:.2f} seconds")
            logger.info(f"   Consensus: {result['consensus_reached']}")
            logger.info(f"   Representative: Agent {result['representative_agent_id']}")
            
            return result
            
    def notify_summary_update(self, agent_id: int, summary: str):
        """
        Called when an agent updates their summary.
        In the simplified system, this just updates the summary - 
        agents will see updates when they restart naturally.
        """
        logger.info(f"📢 Agent {agent_id} updated summary")
        
        # Update the summary in agent state
        self.update_agent_summary(agent_id, summary)
                    
    def cleanup(self):
        """
        Clean up resources and stop all agents.
        """
        logger.info("🧹 Cleaning up orchestrator resources")
        self._stop_event.set()
        
        # Wait for all threads to finish
        for thread in self._agent_threads.values():
            if thread.is_alive():
                thread.join(timeout=5)
                
        logger.info("✅ Orchestrator cleanup completed")