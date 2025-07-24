"""
MASS (Multi-Agent Summary System) Framework

A proven binary decision framework for multi-agent coordination that eliminates
perfectionism loops and produces reliable decisions across diverse domains.

Key Components:
- ChatAgent: Common interface for individual agents and orchestrators
- MassOrchestrator: Coordinated multi-agent system with real-time streaming
- MessageTemplates: Proven message patterns for binary decision making
- AgentConfig: Flexible agent configuration with web search and code execution

Usage:
    from mass import OpenAIBackend, create_simple_agent, create_orchestrator
    
    backend = OpenAIBackend()
    agents = [("agent1", create_simple_agent(backend, "Expert system message"))]
    orchestrator = create_orchestrator(agents)
    
    async for chunk in orchestrator.chat_simple("Your question here"):
        print(chunk.content)
"""

# Import main classes for convenience
from .backends.openai_backend import OpenAIBackend
from .chat_agent import (
    ChatAgent, 
    SingleAgent, 
    ConfigurableAgent,
    create_simple_agent,
    create_expert_agent, 
    create_research_agent,
    create_computational_agent
)
from .orchestrator import MassOrchestrator, create_orchestrator
from .message_templates import MessageTemplates, get_templates
from .agent_config import AgentConfig

__version__ = "1.0.0"
__author__ = "MASS Framework Contributors"

__all__ = [
    # Backends
    "OpenAIBackend",
    
    # Agents
    "ChatAgent",
    "SingleAgent", 
    "ConfigurableAgent",
    "create_simple_agent",
    "create_expert_agent",
    "create_research_agent", 
    "create_computational_agent",
    
    # Orchestrator
    "MassOrchestrator",
    "create_orchestrator",
    
    # Configuration
    "AgentConfig",
    "MessageTemplates",
    "get_templates",
    
    # Metadata
    "__version__",
    "__author__"
]