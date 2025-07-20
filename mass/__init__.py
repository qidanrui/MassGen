"""Mass Agent System - A multi-agent orchestration framework."""

__version__ = "1.0.0"

# Import main components for easy access
from .agent import MassAgent
from .agents import create_agent
from .orchestrator import MassOrchestrator
from .workflow import MassWorkflowManager
from .logging import initialize_logging, cleanup_logging, get_log_manager
from .streaming_display import create_streaming_display
from .main import MassSystem
from .types import (
    AgentState, 
    TaskInput, 
    AgentResponse, 
    VoteRecord, 
    SystemState, 
    LogEntry,
    SummaryRecord
)

__all__ = [
    "MassAgent",
    "AgentState", 
    "TaskInput",
    "AgentResponse",
    "VoteRecord",
    "SystemState",
    "LogEntry",
    "SummaryRecord",
    "create_agent",
    "MassOrchestrator",
    "MassWorkflowManager",
    "initialize_logging",
    "cleanup_logging",
    "get_log_manager",
    "create_streaming_display",
    "MassSystem",
] 