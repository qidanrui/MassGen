"""Multi-Agent Scaling System (MASS)

A framework for coordinating multiple AI agents to solve complex tasks.
"""

__version__ = "0.1.0"

# Core simplified interface
from .agent import MassAgent, AgentStatus, SessionInfo, WorkingSummary
from .orchestrator import MassOrchestrator, MassOrchestrationResult
from .agent_config import AgentConfig


__all__ = [
    "MassAgent",
    "MassOrchestrator", 
    "AgentConfig",
    "MassOrchestrationResult",
    "AgentStatus",
    "SessionInfo",
    "WorkingSummary"
]