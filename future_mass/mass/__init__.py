"""Multi-Agent Scaling System (MASS)

A framework for coordinating multiple AI agents to solve complex tasks.
"""

__version__ = "0.1.0"

# Core simplified interface
from .agent import AgentStatus, MassAgent, SessionInfo, WorkingSummary
from .agent_config import AgentConfig
from .orchestrator import MassOrchestrationResult, MassOrchestrator

__all__ = [
    "MassAgent",
    "MassOrchestrator",
    "AgentConfig",
    "MassOrchestrationResult",
    "AgentStatus",
    "SessionInfo",
    "WorkingSummary",
]
