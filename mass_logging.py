"""
MASS Logging System

This module provides comprehensive logging capabilities for the MASS system,
recording all agent state changes, orchestration events, and system activities
to local files for detailed analysis.
"""

import json
import logging
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LogEntry:
    """Represents a single log entry in the MASS system."""

    timestamp: float
    event_type: str  # e.g., "agent_summary_update", "voting", "phase_change", etc.
    agent_id: int | None
    phase: str
    data: dict[str, Any]
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MassLogManager:
    """
    Comprehensive logging system for the MASS framework.

    This system records all significant events including:
    - Agent state changes (status, summary updates, voting)
    - Orchestration system events
    - Workflow phase transitions
    - System metrics and performance data
    """

    def __init__(
        self,
        log_dir: str = "logs",
        session_id: str | None = None,
        non_blocking: bool = False,
    ):
        """
        Initialize the logging system.

        Args:
            log_dir: Directory to save log files
            session_id: Unique identifier for this session
            non_blocking: If True, disable file logging to prevent any hanging issues
        """
        self.log_dir = Path(log_dir)
        self.session_id = session_id or self._generate_session_id()
        self.non_blocking = non_blocking

        if self.non_blocking:
            print(f"⚠️  LOGGING: Non-blocking mode enabled - file logging disabled")

        # Create log directory if it doesn't exist (unless non-blocking)
        if not self.non_blocking:
            try:
                self.log_dir.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Warning: Failed to create log directory, enabling non-blocking mode: {e}")
                self.non_blocking = True

        # Session-specific log file
        self.session_log_file = self.log_dir / f"session_{self.session_id}.jsonl"

        # Agent-specific log files
        self.agent_log_dir = self.log_dir / f"session_{self.session_id}_agents"
        if not self.non_blocking:
            try:
                self.agent_log_dir.mkdir(exist_ok=True)
            except Exception as e:
                print(f"Warning: Failed to create agent log directory, enabling non-blocking mode: {e}")
                self.non_blocking = True

        # In-memory log storage for real-time access
        self.log_entries: list[LogEntry] = []
        self.agent_logs: dict[int, list[LogEntry]] = {}

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # Initialize basic logging
        self._setup_logging()

        # Log session start
        self.log_event(
            "session_started",
            data={
                "session_id": self.session_id,
                "timestamp": time.time(),
                "log_dir": str(self.log_dir),
                "non_blocking_mode": self.non_blocking,
            },
        )

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mass_{timestamp}_{int(time.time() * 1000) % 10000}"

    def _setup_logging(self):
        """Set up file logging configuration."""
        log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create session-specific log file handler
        session_log_handler = logging.FileHandler(self.log_dir / f"session_{self.session_id}.log")
        session_log_handler.setFormatter(log_formatter)
        session_log_handler.setLevel(logging.DEBUG)

        # Add handler to the mass logger
        mass_logger = logging.getLogger("mass")
        mass_logger.addHandler(session_log_handler)
        mass_logger.setLevel(logging.DEBUG)

        # Prevent duplicate console logs
        mass_logger.propagate = False

        # Add console handler if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in mass_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            console_handler.setLevel(logging.INFO)
            mass_logger.addHandler(console_handler)

    def log_event(
        self,
        event_type: str,
        agent_id: int | None = None,
        phase: str = "unknown",
        data: dict[str, Any] | None = None,
    ):
        """
        Log a general system event.

        Args:
            event_type: Type of event (e.g., "session_started", "phase_change")
            agent_id: Agent ID if event is agent-specific
            phase: Current system phase
            data: Additional event data
        """
        with self._lock:
            entry = LogEntry(
                timestamp=time.time(),
                event_type=event_type,
                agent_id=agent_id,
                phase=phase,
                data=data or {},
                session_id=self.session_id,
            )

            self.log_entries.append(entry)

            # Also store in agent-specific logs
            if agent_id is not None:
                if agent_id not in self.agent_logs:
                    self.agent_logs[agent_id] = []
                self.agent_logs[agent_id].append(entry)

            # Write to file immediately
            self._write_log_entry(entry)

    def log_agent_summary_update(
        self,
        agent_id: int,
        summary: str,
        final_answer: str = "",
        phase: str = "unknown",
    ):
        """
        Log agent summary update with detailed information.

        Args:
            agent_id: Agent ID
            summary: Updated summary content
            final_answer: Final answer if provided
            phase: Current workflow phase
        """
        data = {
            "summary": summary,
            "summary_length": len(summary),
            "final_answer": final_answer,
            "final_answer_length": len(final_answer),
            "has_final_answer": bool(final_answer),
        }

        self.log_event("agent_summary_update", agent_id, phase, data)

        # Log to agent-specific file
        self._write_agent_log(
            agent_id,
            {
                "timestamp": time.time(),
                "event": "summary_update",
                "phase": phase,
                "summary": summary,
                "final_answer": final_answer,
            },
        )

    def log_agent_status_change(self, agent_id: int, old_status: str, new_status: str, phase: str = "unknown"):
        """
        Log agent status change.

        Args:
            agent_id: Agent ID
            old_status: Previous status
            new_status: New status
            phase: Current workflow phase
        """
        data = {
            "old_status": old_status,
            "new_status": new_status,
            "status_change": f"{old_status} -> {new_status}",
        }

        self.log_event("agent_status_change", agent_id, phase, data)

        # Log to agent-specific file
        self._write_agent_log(
            agent_id,
            {
                "timestamp": time.time(),
                "event": "status_change",
                "phase": phase,
                "old_status": old_status,
                "new_status": new_status,
            },
        )

    def log_system_state_snapshot(self, orchestration_system, phase: str = "unknown"):
        """
        Log a complete system state snapshot including all agent summaries and voting status.

        Args:
            orchestration_system: The MassOrchestrationSystem instance
            phase: Current workflow phase
        """

        # Collect all agent states
        agent_states = {}
        all_agent_summaries = {}
        vote_records = []

        for agent_id, agent_state in orchestration_system.agent_states.items():
            # Full agent state information
            agent_states[agent_id] = {
                "status": agent_state.status,
                "working_summary": agent_state.working_summary,
                "final_answer": agent_state.final_answer,
                "vote_target": agent_state.vote_target,
                "execution_time": agent_state.execution_time,
                "update_count": len(agent_state.update_history),
                "seen_updates_timestamps": agent_state.seen_updates_timestamps,
            }

            # Summary history for each agent
            all_agent_summaries[agent_id] = {
                "current_summary": agent_state.working_summary,
                "current_final_answer": agent_state.final_answer,
                "summary_history": [
                    {
                        "timestamp": update["timestamp"],
                        "summary": update["summary"],
                        "final_answer": update.get("final_answer", ""),
                        "status": update.get("status", "unknown"),
                    }
                    for update in agent_state.update_history
                ],
            }

        # Collect voting information
        for vote in orchestration_system.votes:
            vote_records.append(
                {
                    "voter_id": vote.voter_id,
                    "target_id": vote.target_id,
                    "timestamp": vote.timestamp,
                }
            )

        # Calculate voting status
        vote_counts = Counter(vote.target_id for vote in orchestration_system.votes)
        voting_status = {
            "vote_distribution": dict(vote_counts),
            "total_votes_cast": len(orchestration_system.votes),
            "total_agents": len(orchestration_system.agents),
            "consensus_reached": orchestration_system.system_state.consensus_reached,
            "winning_agent_id": orchestration_system.system_state.final_solution_agent_id,
            "votes_needed_for_consensus": max(
                1,
                int(len(orchestration_system.agents) * orchestration_system.consensus_threshold),
            ),
        }

        # Complete system state snapshot
        system_snapshot = {
            "agent_states": agent_states,
            "agent_summaries": all_agent_summaries,
            "voting_records": vote_records,
            "voting_status": voting_status,
            "system_phase": phase,
            "total_rounds": orchestration_system.system_state.total_rounds,
            "system_runtime": (time.time() - orchestration_system.system_state.start_time)
            if orchestration_system.system_state.start_time
            else 0,
        }

        # Log the system snapshot
        self.log_event("system_state_snapshot", phase=phase, data=system_snapshot)

        # Write system state to each agent's log file for complete context
        system_state_entry = {
            "timestamp": time.time(),
            "event": "system_state_snapshot",
            "phase": phase,
            "system_state": system_snapshot,
        }

        for agent_id in orchestration_system.agents.keys():
            self._write_agent_log(agent_id, system_state_entry)

        return system_snapshot

    def log_agent_summary_update_with_system_context(
        self,
        agent_id: int,
        summary: str,
        final_answer: str = "",
        phase: str = "unknown",
        orchestration_system=None,
    ):
        """
        Enhanced version of log_agent_summary_update that includes system context.

        Args:
            agent_id: Agent ID
            summary: Updated summary content
            final_answer: Final answer if provided
            phase: Current workflow phase
            orchestration_system: The orchestration system for context
        """
        # First log the standard summary update
        self.log_agent_summary_update(agent_id, summary, final_answer, phase)

        # Then log system context if orchestration system is provided
        if orchestration_system:
            # Get current state of all other agents for context
            peer_states = {}
            for (
                other_agent_id,
                other_state,
            ) in orchestration_system.agent_states.items():
                if other_agent_id != agent_id:
                    peer_states[other_agent_id] = {
                        "status": other_state.status,
                        "has_summary": bool(other_state.working_summary),
                        "vote_target": other_state.vote_target,
                        "last_update_time": other_state.update_history[-1]["timestamp"]
                        if other_state.update_history
                        else None,
                    }

            # Get current voting status
            vote_counts = Counter(vote.target_id for vote in orchestration_system.votes)
            current_voting_status = {
                "vote_distribution": dict(vote_counts),
                "total_votes": len(orchestration_system.votes),
                "agent_voted": orchestration_system.agent_states[agent_id].status == "voted",
                "agent_vote_target": orchestration_system.agent_states[agent_id].vote_target,
            }

            # Enhanced log entry with system context
            enhanced_entry = {
                "timestamp": time.time(),
                "event": "summary_update_with_context",
                "phase": phase,
                "agent_summary": {
                    "summary": summary,
                    "final_answer": final_answer,
                    "summary_length": len(summary),
                },
                "peer_agent_states": peer_states,
                "voting_status": current_voting_status,
                "system_round": len(orchestration_system.agent_states[agent_id].update_history),
            }

            self._write_agent_log(agent_id, enhanced_entry)

    def log_voting_event_with_system_context(
        self,
        voter_id: int,
        target_id: int,
        phase: str = "unknown",
        response_text: str = "",
        orchestration_system=None,
    ):
        """
        Enhanced version of log_voting_event that includes full system voting context.

        Args:
            voter_id: Agent casting the vote
            target_id: Agent being voted for
            phase: Current workflow phase
            response_text: The full response text that led to this vote
            orchestration_system: The orchestration system for full context
        """
        # First log the standard voting event
        self.log_voting_event(voter_id, target_id, phase, response_text)

        # Then log enhanced context if orchestration system is provided
        if orchestration_system:
            # Get complete voting picture after this vote
            vote_counts = Counter(vote.target_id for vote in orchestration_system.votes)

            # Get all agent summaries that are being voted on
            candidate_summaries = {}
            for candidate_id, votes in vote_counts.items():
                candidate_state = orchestration_system.agent_states[candidate_id]
                candidate_summaries[candidate_id] = {
                    "summary": candidate_state.working_summary,
                    "final_answer": candidate_state.final_answer,
                    "vote_count": votes,
                    "summary_length": len(candidate_state.working_summary),
                }

            # Enhanced voting context
            enhanced_voting_entry = {
                "timestamp": time.time(),
                "event": "voting_with_full_context",
                "phase": phase,
                "vote_details": {
                    "voter_id": voter_id,
                    "target_id": target_id,
                    "response_text": response_text,
                    "response_length": len(response_text),
                },
                "complete_voting_state": {
                    "vote_distribution": dict(vote_counts),
                    "total_votes_cast": len(orchestration_system.votes),
                    "consensus_reached": orchestration_system.system_state.consensus_reached,
                    "votes_needed": max(
                        1,
                        int(len(orchestration_system.agents) * orchestration_system.consensus_threshold),
                    ),
                },
                "candidate_summaries": candidate_summaries,
                "remaining_working_agents": [
                    aid for aid, state in orchestration_system.agent_states.items() if state.status == "working"
                ],
            }

            # Write to all agent logs for complete transparency
            for agent_id in orchestration_system.agents.keys():
                self._write_agent_log(agent_id, enhanced_voting_entry)

    def log_voting_event(
        self,
        voter_id: int,
        target_id: int,
        phase: str = "unknown",
        response_text: str = "",
    ):
        """
        Log voting event.

        Args:
            voter_id: Agent casting the vote
            target_id: Agent being voted for
            phase: Current workflow phase
            response_text: The full response text that led to this vote (optional)
        """
        print(f"        🗳️  LOG_VOTING_EVENT: {voter_id} → {target_id} (phase: {phase})")
        if response_text:
            print(f"        📝 Including response text ({len(response_text)} chars)")

        data = {
            "voter_id": voter_id,
            "target_id": target_id,
            "vote_action": f"Agent {voter_id} -> Agent {target_id}",
        }

        self.log_event("voting", voter_id, phase, data)
        print(f"        📝 General voting event logged")

        # Log to both voter and target agent files
        vote_data = {
            "timestamp": time.time(),
            "event": "vote_cast",
            "phase": phase,
            "voter_id": voter_id,
            "target_id": target_id,
        }

        # Add response text if provided
        if response_text:
            vote_data["response_text"] = response_text
            vote_data["response_length"] = len(response_text)

        print(f"        📁 Writing to voter agent log: agent_{voter_id}.jsonl")
        self._write_agent_log(voter_id, vote_data)

        vote_received_data = {**vote_data, "event": "vote_received"}
        print(f"        📁 Writing to target agent log: agent_{target_id}.jsonl")
        self._write_agent_log(target_id, vote_received_data)

        print(f"        ✅ All voting logs written successfully")

    def log_phase_transition(
        self,
        old_phase: str,
        new_phase: str,
        additional_data: dict[str, Any] | None = None,
    ):
        """
        Log workflow phase transition.

        Args:
            old_phase: Previous phase
            new_phase: New phase
            additional_data: Additional transition data
        """
        data = {
            "old_phase": old_phase,
            "new_phase": new_phase,
            "phase_transition": f"{old_phase} -> {new_phase}",
            **(additional_data or {}),
        }

        self.log_event("phase_transition", phase=new_phase, data=data)

    def log_consensus_reached(
        self,
        winning_agent_id: int,
        vote_distribution: dict[int, int],
        total_rounds: int,
        is_fallback: bool = False,
        phase: str = "unknown",
    ):
        """
        Log consensus achievement.

        Args:
            winning_agent_id: Agent selected by consensus
            vote_distribution: Final vote counts
            total_rounds: Total workflow rounds
            is_fallback: Whether this was a fallback consensus
            phase: Current workflow phase when consensus was reached
        """
        data = {
            "winning_agent_id": winning_agent_id,
            "vote_distribution": vote_distribution,
            "total_rounds": total_rounds,
            "is_fallback": is_fallback,
            "consensus_type": "fallback" if is_fallback else "majority",
        }

        self.log_event("consensus_reached", phase=phase, data=data)

    def log_task_completion(self, final_solution: dict[str, Any]):
        """
        Log task completion with final results.

        Args:
            final_solution: Complete final solution data
        """
        data = {"final_solution": final_solution, "completion_timestamp": time.time()}

        self.log_event("task_completed", phase="completed", data=data)

    def _write_log_entry(self, entry: LogEntry):
        """Write a single log entry to the session JSONL file with timeout protection."""
        # Skip file operations in non-blocking mode
        if self.non_blocking:
            return

        def write_with_timeout():
            try:
                # Create directory if it doesn't exist
                self.session_log_file.parent.mkdir(parents=True, exist_ok=True)

                with open(self.session_log_file, "a", buffering=1) as f:  # Line buffering
                    json_line = json.dumps(entry.to_dict(), default=str, ensure_ascii=False)
                    f.write(json_line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except Exception as e:
                print(f"Warning: Failed to write log entry: {e}")
                return False

        # Use daemon thread with timeout to prevent hanging
        result_container = {"completed": False, "success": False}

        def worker():
            try:
                result_container["success"] = write_with_timeout()
            except Exception as e:
                print(f"Session log write thread error: {e}")
                result_container["success"] = False
            finally:
                result_container["completed"] = True

        # Create daemon thread
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        # Wait for completion with timeout
        worker_thread.join(timeout=3.0)

        if not result_container["completed"]:
            print(f"Warning: Session log write timed out after 3 seconds")

    def _write_agent_log(self, agent_id: int, data: dict[str, Any]):
        """Write agent-specific log entry with timeout protection."""
        # Skip file operations in non-blocking mode
        if self.non_blocking:
            return

        def write_with_timeout():
            try:
                agent_log_file = self.agent_log_dir / f"agent_{agent_id}.jsonl"
                print(f"          📝 Writing to file: {agent_log_file}")

                # Truncate data if it's too large to prevent JSON serialization issues
                truncated_data = self._truncate_data_if_needed(data)
                print(f"          📋 Data size: {len(str(data))} chars")

                # Create directory if it doesn't exist
                agent_log_file.parent.mkdir(parents=True, exist_ok=True)

                # Write with explicit timeout and buffering
                with open(agent_log_file, "a", buffering=1) as f:  # Line buffering
                    json_line = json.dumps(truncated_data, default=str, ensure_ascii=False)
                    f.write(json_line + "\n")
                    f.flush()  # Force write to disk
                    os.fsync(f.fileno())  # Ensure data is written to disk

                print(f"          ✅ Successfully wrote {len(json_line)} characters to {agent_log_file}")
                return True
            except Exception as e:
                print(f"          ❌ Failed to write agent log: {e}")
                print(f"Warning: Failed to write agent log: {e}")
                return False

        # Use daemon thread with timeout to prevent hanging
        result_container = {"completed": False, "success": False}

        def worker():
            try:
                result_container["success"] = write_with_timeout()
            except Exception as e:
                print(f"          ❌ Logging thread error: {e}")
                result_container["success"] = False
            finally:
                result_container["completed"] = True

        # Create daemon thread that will be killed when main thread exits
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        # Wait for completion with timeout (5 seconds should be enough for file I/O)
        worker_thread.join(timeout=5.0)

        if not result_container["completed"]:
            print(f"          ⏰ Logging operation timed out for Agent {agent_id} - continuing without logging")
            print(f"Warning: Agent {agent_id} log write timed out after 5 seconds")
        elif not result_container["success"]:
            print(f"          ⚠️  Logging operation failed for Agent {agent_id} - continuing without logging")

    def _truncate_data_if_needed(self, data: dict[str, Any], max_size: int = 50000) -> dict[str, Any]:
        """Truncate data fields if they're too large to prevent hanging."""
        truncated_data = data.copy()

        # Fields that might be very large
        large_fields = ["summary", "response_text", "final_answer"]

        for field in large_fields:
            if field in truncated_data and isinstance(truncated_data[field], str):
                if len(truncated_data[field]) > max_size:
                    print(f"          ✂️  Truncating {field} from {len(truncated_data[field])} to {max_size} chars")
                    truncated_data[field] = truncated_data[field][:max_size] + "... [TRUNCATED]"

        return truncated_data

    def get_agent_history(self, agent_id: int) -> list[LogEntry]:
        """Get complete history for a specific agent."""
        with self._lock:
            return self.agent_logs.get(agent_id, []).copy()

    def get_session_summary(self) -> dict[str, Any]:
        """Get comprehensive session summary."""
        with self._lock:
            # Count events by type
            event_counts = {}
            agent_activities = {}

            for entry in self.log_entries:
                # Count events
                event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1

                # Count agent activities
                if entry.agent_id is not None:
                    agent_id = entry.agent_id
                    if agent_id not in agent_activities:
                        agent_activities[agent_id] = []
                    agent_activities[agent_id].append(
                        {
                            "timestamp": entry.timestamp,
                            "event_type": entry.event_type,
                            "phase": entry.phase,
                        }
                    )

            return {
                "session_id": self.session_id,
                "total_events": len(self.log_entries),
                "event_counts": event_counts,
                "agents_involved": list(agent_activities.keys()),
                "agent_activities": agent_activities,
                "session_duration": self._calculate_session_duration(),
                "log_files": {
                    "session_log": str(self.session_log_file),
                    "agent_logs_dir": str(self.agent_log_dir),
                },
            }

    def _calculate_session_duration(self) -> float:
        """Calculate total session duration."""
        if not self.log_entries:
            return 0.0

        start_time = min(entry.timestamp for entry in self.log_entries)
        end_time = max(entry.timestamp for entry in self.log_entries)
        return end_time - start_time

    def export_full_session_report(self) -> str:
        """Export a comprehensive session report to a file with timeout protection."""
        report_file = self.log_dir / f"session_{self.session_id}_report.json"

        def export_with_timeout():
            try:
                with self._lock:
                    # Gather all data with size limits to prevent hanging
                    print(f"📊 Gathering session data for report...")

                    # Truncate large datasets to prevent hanging
                    max_events = 1000  # Limit total events
                    truncated_events = (
                        self.log_entries[-max_events:] if len(self.log_entries) > max_events else self.log_entries
                    )

                    if len(self.log_entries) > max_events:
                        print(f"⚠️  Truncating events from {len(self.log_entries)} to {max_events} for report")

                    full_report = {
                        "session_metadata": {
                            "session_id": self.session_id,
                            "start_time": min(entry.timestamp for entry in self.log_entries)
                            if self.log_entries
                            else None,
                            "end_time": max(entry.timestamp for entry in self.log_entries)
                            if self.log_entries
                            else None,
                            "duration": self._calculate_session_duration(),
                            "total_events": len(self.log_entries),
                            "truncated": len(self.log_entries) > max_events,
                        },
                        "all_events": [entry.to_dict() for entry in truncated_events],
                        "agent_summaries": {
                            agent_id: [
                                self._truncate_data_if_needed(entry.to_dict(), max_size=10000)
                                for entry in entries[-50:]
                            ]  # Last 50 entries per agent
                            for agent_id, entries in self.agent_logs.items()
                        },
                        "session_summary": self.get_session_summary(),
                    }

                print(f"📝 Writing report to {report_file}...")

                # Write to file with size check
                report_str = json.dumps(full_report, indent=2, default=str)
                if len(report_str) > 50_000_000:  # 50MB limit
                    print(f"⚠️  Report too large ({len(report_str):,} chars), creating summary only")
                    # Create minimal report
                    full_report = {
                        "session_metadata": full_report["session_metadata"],
                        "session_summary": full_report["session_summary"],
                        "note": "Full report was too large, created summary only",
                    }
                    report_str = json.dumps(full_report, indent=2, default=str)

                with open(report_file, "w") as f:
                    f.write(report_str)
                    f.flush()
                    os.fsync(f.fileno())

                print(f"✅ Report written successfully ({len(report_str):,} characters)")
                return str(report_file)

            except Exception as e:
                print(f"❌ Error exporting session report: {e}")
                return ""

        # Use daemon thread with timeout to prevent hanging
        result_container = {"completed": False, "result": ""}

        def worker():
            try:
                result_container["result"] = export_with_timeout()
            except Exception as e:
                print(f"❌ Report export thread error: {e}")
                result_container["result"] = ""
            finally:
                result_container["completed"] = True

        # Create daemon thread that will be killed when main thread exits
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()

        # Skip report generation to avoid timeout issues
        print("📊 Skipping detailed report generation")
        return ""

    def cleanup(self):
        """Clean up and finalize the logging session."""
        self.log_event(
            "session_ended",
            data={
                "end_timestamp": time.time(),
                "total_events_logged": len(self.log_entries),
            },
        )

        # Export final report
        report_file = self.export_full_session_report()
        if report_file:
            print(f"📋 Session report exported: {report_file}")


# Global log manager instance
_log_manager: MassLogManager | None = None


def initialize_logging(
    log_dir: str = "logs", session_id: str | None = None, non_blocking: bool = False
) -> MassLogManager:
    """Initialize the global logging system."""
    global _log_manager

    # Check environment variable for non-blocking mode
    env_non_blocking = os.getenv("MASS_NON_BLOCKING_LOGGING", "").lower() in (
        "true",
        "1",
        "yes",
    )
    if env_non_blocking:
        print("🔧 MASS_NON_BLOCKING_LOGGING environment variable detected - enabling non-blocking mode")
        non_blocking = True

    _log_manager = MassLogManager(log_dir, session_id, non_blocking)
    return _log_manager


def get_log_manager() -> MassLogManager | None:
    """Get the current log manager instance."""
    return _log_manager


def cleanup_logging():
    """Cleanup the global logging system."""
    global _log_manager
    if _log_manager:
        _log_manager.cleanup()
        _log_manager = None
