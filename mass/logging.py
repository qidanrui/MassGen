"""
MASS Logging System

This module provides comprehensive logging capabilities for the MASS system,
recording all agent state changes, orchestration events, and system activities
to local files for detailed analysis.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import Counter

from .types import LogEntry

class MassLogManager:
    """
    Comprehensive logging system for the MASS framework.
    
    Records all significant events including:
    - Agent state changes (working, voted, failed)
    - Summary updates and notifications  
    - Voting events and consensus decisions
    - Phase transitions (collaboration, debate, consensus)
    - System metrics and performance data
    """
    
    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None, non_blocking: bool = False):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory to save log files
            session_id: Unique identifier for this session
            non_blocking: If True, disable file logging to prevent hanging issues
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
        self.log_entries: List[LogEntry] = []
        self.agent_logs: Dict[int, List[LogEntry]] = {}
        
        # MASS-specific event counters
        self.event_counters = {
            "summary_updates": 0,
            "votes_cast": 0,
            "consensus_reached": 0,
            "debates_started": 0,
            "agent_restarts": 0,
            "notifications_sent": 0
        }
        
        # Thread lock for concurrent access
        self._lock = threading.Lock()
        
        # Initialize logging
        self._setup_logging()
        
        # Log session start
        self.log_event("session_started", data={
            "session_id": self.session_id,
            "timestamp": time.time(),
            "log_dir": str(self.log_dir),
            "non_blocking_mode": self.non_blocking
        })
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mass_{timestamp}_{int(time.time() * 1000) % 10000}"
    
    def _setup_logging(self):
        """Set up file logging configuration."""
        # Skip file logging setup in non-blocking mode
        if self.non_blocking:
            return
            
        log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure log directory exists before creating file handler
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create log directory {self.log_dir}, skipping file logging: {e}")
            return
        
        # Create session-specific log file handler
        session_log_handler = logging.FileHandler(
            self.log_dir / f"session_{self.session_id}.log"
        )
        session_log_handler.setFormatter(log_formatter)
        session_log_handler.setLevel(logging.DEBUG)
        
        # Add handler to the mass logger
        mass_logger = logging.getLogger('mass')
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
    
    def log_event(self, event_type: str, agent_id: Optional[int] = None, 
                  phase: str = "unknown", data: Optional[Dict[str, Any]] = None):
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
                session_id=self.session_id
            )
            
            self.log_entries.append(entry)
            
            # Also store in agent-specific logs
            if agent_id is not None:
                if agent_id not in self.agent_logs:
                    self.agent_logs[agent_id] = []
                self.agent_logs[agent_id].append(entry)
            
            # Write to file immediately
            self._write_log_entry(entry)
    
    def log_agent_summary_update(self, agent_id: int, summary: str, 
                                phase: str = "unknown"):
        """
        Log agent summary update with detailed information.
        
        Args:
            agent_id: Agent ID
            summary: Updated summary content
            phase: Current workflow phase
        """
        data = {
            "summary": summary,
            "summary_length": len(summary),
        }
        
        self.log_event("agent_summary_update", agent_id, phase, data)
        
        # Log to agent-specific file
        self._write_agent_log(agent_id, {
            "timestamp": time.time(),
            "event": "summary_update",
            "phase": phase,
            "summary": summary,
        })
    
    def log_agent_status_change(self, agent_id: int, old_status: str, 
                               new_status: str, phase: str = "unknown"):
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
            "status_change": f"{old_status} -> {new_status}"
        }
        
        self.log_event("agent_status_change", agent_id, phase, data)
        
        # Log to agent-specific file
        self._write_agent_log(agent_id, {
            "timestamp": time.time(),
            "event": "status_change",
            "phase": phase,
            "old_status": old_status,
            "new_status": new_status
        })
    
    def log_system_state_snapshot(self, orchestrator, phase: str = "unknown"):
        """
        Log a complete system state snapshot including all agent summaries and voting status.
        
        Args:
            orchestrator: The MassOrchestrator instance
            phase: Current workflow phase
        """
        
        # Collect all agent states
        agent_states = {}
        all_agent_summaries = {}
        vote_records = []
        
        for agent_id, agent_state in orchestrator.agent_states.items():
            # Full agent state information
            agent_states[agent_id] = {
                "status": agent_state.status,
                "working_summary": agent_state.working_summary,
                "vote_target": agent_state.vote_target,
                "execution_time": agent_state.execution_time,
                "update_count": len(agent_state.update_history),
                "seen_updates_timestamps": agent_state.seen_updates_timestamps
            }
            
            # Summary history for each agent
            all_agent_summaries[agent_id] = {
                "current_summary": agent_state.working_summary,
                "summary_history": [
                    {
                        "timestamp": update.timestamp,
                        "summary": update.summary,
                        "status": update.status
                    }
                    for update in agent_state.update_history
                ]
            }
        
        # Collect voting information
        for vote in orchestrator.votes:
            vote_records.append({
                "voter_id": vote.voter_id,
                "target_id": vote.target_id,
                "timestamp": vote.timestamp
            })
        
        # Calculate voting status
        vote_counts = Counter(vote.target_id for vote in orchestrator.votes)
        voting_status = {
            "vote_distribution": dict(vote_counts),
            "total_votes_cast": len(orchestrator.votes),
            "total_agents": len(orchestrator.agents),
            "consensus_reached": orchestrator.system_state.consensus_reached,
            "winning_agent_id": orchestrator.system_state.representative_agent_id,
            "votes_needed_for_consensus": max(1, int(len(orchestrator.agents) * orchestrator.consensus_threshold))
        }
        
        # Complete system state snapshot
        system_snapshot = {
            "agent_states": agent_states,
            "agent_summaries": all_agent_summaries,
            "voting_records": vote_records,
            "voting_status": voting_status,
            "system_phase": phase,
            "system_runtime": (time.time() - orchestrator.system_state.start_time) if orchestrator.system_state.start_time else 0
        }
        
        # Log the system snapshot
        self.log_event("system_state_snapshot", phase=phase, data=system_snapshot)
        
        # Write system state to each agent's log file for complete context
        system_state_entry = {
            "timestamp": time.time(),
            "event": "system_state_snapshot",
            "phase": phase,
            "system_state": system_snapshot
        }
        
        for agent_id in orchestrator.agents.keys():
            self._write_agent_log(agent_id, system_state_entry)
        
        return system_snapshot
    
    def log_voting_event(self, voter_id: int, target_id: int, phase: str = "unknown", response_text: str = ""):
        """
        Log a voting event with detailed information.
        
        Args:
            voter_id: ID of the agent casting the vote
            target_id: ID of the agent being voted for
            phase: Current workflow phase
            response_text: Full response text that led to the vote
        """
        with self._lock:
            self.event_counters["votes_cast"] += 1
            
        data = {
            "voter_id": voter_id,
            "target_id": target_id,
            "response_text_length": len(response_text),
            "total_votes_cast": self.event_counters["votes_cast"]
        }
        
        self.log_event("voting_event", voter_id, phase, data)
        
        # Log to both voter and target agent files
        vote_entry = {
            "timestamp": time.time(),
            "event": "vote_cast",
            "phase": phase,
            "voter_id": voter_id,
            "target_id": target_id,
            "response_text": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }
        self._write_agent_log(voter_id, vote_entry)
        if target_id != voter_id:
            vote_received_entry = {
                "timestamp": time.time(),
                "event": "vote_received",
                "phase": phase,
                "from_agent": voter_id,
                "reason_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
            self._write_agent_log(target_id, vote_received_entry)
    
    def log_consensus_reached(self, winning_agent_id: int, vote_distribution: Dict[int, int], 
                             is_fallback: bool = False, phase: str = "unknown"):
        """
        Log when consensus is reached.
        
        Args:
            winning_agent_id: ID of the winning agent
            vote_distribution: Dictionary of agent_id -> vote_count
            is_fallback: Whether this was a fallback consensus (timeout)
            phase: Current workflow phase
        """
        with self._lock:
            self.event_counters["consensus_reached"] += 1
            
        data = {
            "winning_agent_id": winning_agent_id,
            "vote_distribution": vote_distribution,
            "is_fallback": is_fallback,
            "total_consensus_events": self.event_counters["consensus_reached"]
        }
        
        self.log_event("consensus_reached", winning_agent_id, phase, data)
        
        # Log to all agent files
        consensus_entry = {
            "timestamp": time.time(),
            "event": "consensus_reached",
            "phase": phase,
            "winning_agent_id": winning_agent_id,
            "vote_distribution": vote_distribution,
            "is_fallback": is_fallback
        }
        for agent_id in vote_distribution.keys():
            self._write_agent_log(agent_id, consensus_entry)
    
    def log_phase_transition(self, old_phase: str, new_phase: str, additional_data: Dict[str, Any] = None):
        """
        Log system phase transitions.
        
        Args:
            old_phase: Previous phase
            new_phase: New phase
            additional_data: Additional context data
        """
        data = {
            "old_phase": old_phase,
            "new_phase": new_phase,
            "phase_transition": f"{old_phase} -> {new_phase}",
            **(additional_data or {})
        }
        
        self.log_event("phase_transition", phase=new_phase, data=data)
    
    def log_notification_sent(self, agent_id: int, notification_type: str, content_preview: str, phase: str = "unknown"):
        """
        Log when a notification is sent to an agent.
        
        Args:
            agent_id: Target agent ID
            notification_type: Type of notification (update, debate, presentation, prompt)
            content_preview: Preview of notification content
            phase: Current workflow phase
        """
        with self._lock:
            self.event_counters["notifications_sent"] += 1
            
        data = {
            "notification_type": notification_type,
            "content_preview": content_preview[:200] + "..." if len(content_preview) > 200 else content_preview,
            "content_length": len(content_preview),
            "total_notifications_sent": self.event_counters["notifications_sent"]
        }
        
        self.log_event("notification_sent", agent_id, phase, data)
        
        # Log to agent file
        notification_entry = {
            "timestamp": time.time(),
            "event": "notification_received",
            "phase": phase,
            "notification_type": notification_type,
            "content": content_preview
        }
        self._write_agent_log(agent_id, notification_entry)
    
    def log_agent_restart(self, agent_id: int, reason: str, phase: str = "unknown"):
        """
        Log when an agent is restarted.
        
        Args:
            agent_id: ID of the restarted agent
            reason: Reason for restart
            phase: Current workflow phase
        """
        with self._lock:
            self.event_counters["agent_restarts"] += 1
            
        data = {
            "restart_reason": reason,
            "total_restarts": self.event_counters["agent_restarts"]
        }
        
        self.log_event("agent_restart", agent_id, phase, data)
        
        # Log to agent file
        restart_entry = {
            "timestamp": time.time(),
            "event": "agent_restarted",
            "phase": phase,
            "reason": reason
        }
        self._write_agent_log(agent_id, restart_entry)
    
    def log_debate_started(self, phase: str = "unknown"):
        """
        Log when a debate phase starts.
        
        Args:
            phase: Current workflow phase
        """
        with self._lock:
            self.event_counters["debates_started"] += 1
            
        data = {
            "total_debates": self.event_counters["debates_started"]
        }
        
        self.log_event("debate_started", phase=phase, data=data)
    
    def log_task_completion(self, final_solution: Dict[str, Any]):
        """
        Log task completion with final results.
        
        Args:
            final_solution: Complete final solution data
        """
        data = {
            "final_solution": final_solution,
            "completion_timestamp": time.time()
        }
        
        self.log_event("task_completed", phase="completed", data=data)
    
    def _write_log_entry(self, entry: LogEntry):
        """Write a single log entry to the session JSONL file."""
        # Skip file operations in non-blocking mode
        if self.non_blocking:
            return
        
        try:
            # Create directory if it doesn't exist
            self.session_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.session_log_file, 'a', buffering=1) as f:  # Line buffering
                json_line = json.dumps(entry.to_dict(), default=str, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write log entry: {e}")
    
    def _write_agent_log(self, agent_id: int, data: Dict[str, Any]):
        """Write agent-specific log entry."""
        # Skip file operations in non-blocking mode
        if self.non_blocking:
            return
        
        try:
            agent_log_file = self.agent_log_dir / f"agent_{agent_id}.jsonl"
            
            # Create directory if it doesn't exist
            agent_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with explicit buffering
            with open(agent_log_file, 'a', buffering=1) as f:  # Line buffering
                json_line = json.dumps(data, default=str, ensure_ascii=False)
                f.write(json_line + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Failed to write agent log: {e}")
      
    def get_agent_history(self, agent_id: int) -> List[LogEntry]:
        """Get complete history for a specific agent."""
        with self._lock:
            return self.agent_logs.get(agent_id, []).copy()
    
    def get_session_summary(self) -> Dict[str, Any]:
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
                    agent_activities[agent_id].append({
                        "timestamp": entry.timestamp,
                        "event_type": entry.event_type,
                        "phase": entry.phase
                    })
            
            return {
                "session_id": self.session_id,
                "total_events": len(self.log_entries),
                "event_counts": event_counts,
                "agents_involved": list(agent_activities.keys()),
                "agent_activities": agent_activities,
                "session_duration": self._calculate_session_duration(),
                "log_files": {
                    "session_log": str(self.session_log_file),
                    "agent_logs_dir": str(self.agent_log_dir)
                }
            }
    
    def _calculate_session_duration(self) -> float:
        """Calculate total session duration."""
        if not self.log_entries:
            return 0.0
        
        start_time = min(entry.timestamp for entry in self.log_entries)
        end_time = max(entry.timestamp for entry in self.log_entries)
        return end_time - start_time
     
    def cleanup(self):
        """Clean up and finalize the logging session."""
        self.log_event("session_ended", data={
            "end_timestamp": time.time(),
            "total_events_logged": len(self.log_entries)
        })

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics.
        
        Returns:
            Dictionary containing session metrics and statistics
        """
        with self._lock:
            total_events = len(self.log_entries)
            agent_event_counts = {}
            
            for agent_id, logs in self.agent_logs.items():
                agent_event_counts[agent_id] = len(logs)
                
            return {
                "session_id": self.session_id,
                "total_events": total_events,
                "event_counters": self.event_counters.copy(),
                "agent_event_counts": agent_event_counts,
                "total_agents": len(self.agent_logs),
                "session_duration": time.time() - (self.log_entries[0].timestamp if self.log_entries else time.time())
            }


# Global log manager instance
_log_manager: Optional[MassLogManager] = None

def initialize_logging(log_dir: str = "logs", session_id: Optional[str] = None, 
                      non_blocking: bool = False) -> MassLogManager:
    """Initialize the global logging system."""
    global _log_manager
    
    # Check environment variable for non-blocking mode
    env_non_blocking = os.getenv("MASS_NON_BLOCKING_LOGGING", "").lower() in ("true", "1", "yes")
    if env_non_blocking:
        print("🔧 MASS_NON_BLOCKING_LOGGING environment variable detected - enabling non-blocking mode")
        non_blocking = True
    
    _log_manager = MassLogManager(log_dir, session_id, non_blocking)
    return _log_manager

def get_log_manager() -> Optional[MassLogManager]:
    """Get the current log manager instance."""
    return _log_manager

def cleanup_logging():
    """Cleanup the global logging system."""
    global _log_manager
    if _log_manager:
        _log_manager.cleanup()
        _log_manager = None 