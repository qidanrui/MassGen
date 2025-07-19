"""Mass Agent System - A multi-agent orchestration framework."""

__version__ = "1.0.0"

# Import main components for easy access
from .agent import MassAgent, AgentState, TaskInput, AgentResponse
from .agents import create_agent
from .orchestration import MassOrchestrationSystem
from .workflow import MassWorkflowManager
from .logging import initialize_logging, cleanup_logging, get_log_manager
from .streaming_display import create_streaming_display
from .main import MassSystem

__all__ = [
    "MassAgent",
    "AgentState", 
    "TaskInput",
    "AgentResponse",
    "create_agent",
    "MassOrchestrationSystem",
    "MassWorkflowManager",
    "initialize_logging",
    "cleanup_logging",
    "get_log_manager",
    "create_streaming_display",
    "MassSystem",
] 