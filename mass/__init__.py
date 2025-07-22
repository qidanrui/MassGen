"""
MASS - Multi-Agent Scaling System

A powerful multi-agent collaboration framework that enables multiple AI agents
to work together on complex tasks, share insights, vote on solutions, and reach
consensus through structured collaboration and debate.

The system also supports single-agent mode for simpler tasks that don't require
multi-agent collaboration.

Key Features:
- Single-agent mode for simple, direct processing
- Multi-agent collaboration with dynamic restart logic
- Comprehensive YAML configuration system
- Real-time streaming display with multi-region layout
- Robust consensus mechanisms with debate phases
- Support for multiple LLM backends (OpenAI, Gemini, Grok)
- Comprehensive logging and monitoring

Command-Line Usage:
    # Use cli.py for all command-line operations
    
    # Single agent mode
    python cli.py "What is 2+2?" --models gpt-4o
    
    # Multi-agent mode
    python cli.py "What is 2+2?" --models gpt-4o gemini-2.5-flash
    python cli.py "Complex question" --config examples/production.yaml

Programmatic Usage:
    # Using YAML configuration
    from mass import run_mass_with_config, load_config_from_yaml
    config = load_config_from_yaml("config.yaml")
    result = run_mass_with_config("Your question here", config)
    
    # Using simple model list (single agent)
    from mass import run_mass_agents
    result = run_mass_agents("What is 2+2?", ["gpt-4o"])
    
    # Using simple model list (multi-agent)
    from mass import run_mass_agents
    result = run_mass_agents("What is 2+2?", ["gpt-4o", "gemini-2.5-flash"])
    
    # Using configuration objects
    from mass import MassSystem, create_config_from_models
    config = create_config_from_models(["gpt-4o", "grok-3"])
    system = MassSystem(config)
    result = system.run("Complex question here")
"""

# Core system components
from .main import (
    MassSystem, 
    run_mass_agents, 
    run_mass_with_config
)

# Configuration system
from .config import (
    load_config_from_yaml,
    create_config_from_models,
    ConfigurationError
)

# Configuration classes
from .types import (
    MassConfig,
    OrchestratorConfig, 
    AgentConfig,
    ModelConfig,
    StreamingDisplayConfig,
    LoggingConfig,
    TaskInput
)

# Advanced components (for custom usage)
from .orchestrator import MassOrchestrator
from .streaming_display import create_streaming_display
from .logging import MassLogManager

__version__ = "1.0.0"

__all__ = [
    # Main interfaces
    "MassSystem",
    "run_mass_agents", 
    "run_mass_with_config",
    
    # Configuration system
    "load_config_from_yaml",
    "create_config_from_models",
    "ConfigurationError",
    
    # Configuration classes
    "MassConfig",
    "OrchestratorConfig",
    "AgentConfig", 
    "ModelConfig",
    "StreamingDisplayConfig",
    "LoggingConfig",
    "TaskInput",
    
    # Advanced components
    "MassOrchestrator",
    "create_streaming_display",
    "MassLogManager",
] 