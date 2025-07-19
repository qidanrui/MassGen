# 🔮 MASS Future Architecture

> **This directory contains the advanced MASS architecture that this project is evolving toward.**

This is a **reference implementation** showing the planned features and sophisticated multi-agent collaboration patterns that will be integrated into the main MassAgent project.

## 🎯 Key Differences from Current Implementation

### Advanced Agent Interface

- **Sophisticated Tool System**: `update_summary()`, `check_updates()`, `vote()`, `no_update()` with proper validation
- **Smart Notifications**: Queued notification system with intelligent reactivation
- **Streaming Integration**: Full streaming support for all agent interactions

### Enhanced Backend Support

- **Multi-Provider Support**: OpenAI, Anthropic, xAI, Google with unified interface
- **Advanced Tool Integration**: Built-in tools, MCP servers, custom functions
- **Cost Tracking**: Comprehensive cost analysis and budget management

### Sophisticated Orchestration

- **Convergence Detection**: Smart detection of when agents reach consensus
- **Restart Logic**: Intelligent agent reactivation based on notifications and state
- **Performance Analytics**: Detailed tracing and performance analysis

## 🚀 Integration Plan

The features in this directory will be gradually integrated into the main project:

1. **Phase 1**: Backend unification and tool integration
2. **Phase 2**: Enhanced agent interface and notification system
3. **Phase 3**: Advanced orchestration and convergence detection
4. **Phase 4**: Analytics, tracing, and performance optimization

## 📖 Key Files to Study

- `mass/agent.py` - Advanced agent interface with tool validation
- `mass/orchestrator.py` - Sophisticated orchestration logic
- `mass/backends/` - Unified backend architecture
- `examples/summary_demo.py` - Complete working example
- `VISUALIZATION_GUIDE.md` - Advanced visualization features

## 🤝 How to Contribute

Study this architecture to understand the target design, then help implement these patterns in the main project. Focus on:

- Maintaining backward compatibility during migration
- Implementing features incrementally with proper testing
- Following the established patterns for extensibility

---

**This represents the future vision of MASS - a production-ready multi-agent collaboration framework.**

---

# Original Advanced MASS Documentation

A powerful framework for combining multiple frontier language model-based agents (e.g., Claude, Gemini, GPT, Grok + tools and MCP servers) to boost performance on challenging benchmarks like Human-Level Evaluation (HLE) style tasks.

## 🌟 Features

- **Multi-Agent Orchestration**: Coordinate multiple frontier model-based agents in parallel and async
- **Advanced Consensus Strategies**: Every agent adjusts its own answer after seeing the work of others until convergence
- **Comprehensive Benchmarking**: HLE-style tasks across reasoning, math, coding, and language
- **Extensible Framework**: Easy to add new agents, benchmarks, and evaluation metrics
- **Rich Analytics**: Detailed performance analysis and comparison tools
- **CLI Interface**: Command-line tools for easy benchmarking and comparison

## 🏗️ Architecture

TODO: visual

Each agent works async. They maintain a working summary each and keep updating their own working summary until convergence. They can read other agents' working summary but can't write them. They get notified when another agent has updates in their working summary. They can decide how to use other agents' working summary. The convergence is reached when all agents are stopped and no agent needs to restart - meaning all agents have finished working, voted, processed all notifications, and used required tools. Note that as long as one agent updates its working summary, it can trigger other agents to receive notifications and potentially restart their work, even if they had previously stopped.

The system presents the working process of each agent during the entire session.
After the convergence, the system will use the voted agent to present the final answer. If there's a tie in the votes, the system will use the "time to stop updating" to break the tie.

## Agent Interface

Each agent uses the `MassAgent` interface with minimal complexity designed for LLM-based agents:

### LLM-Callable Tools (provided by system)

- `update_summary(new_content)` - Update working summary, triggers notifications to others
- `check_updates()` - Get dict of updated summaries from other agents
- `vote(agent_id)` - Vote for representative (can be called multiple times to change vote)
- `no_update()` - Signal that no further updates are needed (requires update_summary AND vote first)
- `get_session_info(include_summaries=False)` - Get session context including agent statuses, time, costs

### Core Methods

- `process_message(message)` - Common async streaming method for all LLM interactions
- `work_on_task(task, restart_instruction=None)` - Work on task (handles both initial work and restarts)
- `present_final_answer(task)` - Present final answer as representative (uses process_message)

### System Behavior

- **Notifications queued by default** until agent calls `check_updates()`
- **Tool usage validation** enforces correct order: `update_summary()` → `vote()` → `no_update()`
- **Multiple voting allowed** - agents can change votes with new `vote()` calls
- **System triggers `work_on_task()` restart** when:
  - Agent stops but has unprocessed notifications or hasn't voted
  - Agent completes work without calling required tools
  - An agent receives notifications while not working (reactivation)
- **All agent interactions stream responses** for real-time visibility

### Agent Configuration

Agents are configured with model settings and specialized instructions for:

1. **Summary updates** - How and when to call `update_summary()`
2. **Notification handling** - When to call `check_updates()`
3. **Stopping and voting** - When to call `vote()` and stop working
4. **Final answer presentation** - How to structure final responses
5. **Collaboration** - How to incorporate others' insights
6. **Reactivation** - How to decide whether to resume after stopping

Each agent can use any backend (frontier models + tools/MCP, custom implementations) as long as they implement this interface. The system provides complete session state including per-agent status, update counts, costs, and convergence progress.

### Running Example

A user creates two agents, one based on the default model of ChatGPT, and another based on Grok-4. Both agents have built-in tools enabled. The task is:

  What is the earliest known date recorded by a pre-Columbian civilization in the Americas in the aboriginal writing system?

The ChatGPT agent finds the correct Long Count date (7.16.3.2.13) but gets the Gregorian conversion wrong (estimates Sep 9, 36 BCE). The Grok-4 agent initially gets the timeframe completely wrong. After they see each other's summaries through the notification system, Grok-4 uses ChatGPT's Long Count date and provides the correct conversion: 7.16.3.2.13 = December 10th, 36 B.C. The system converges on this correct answer, demonstrating how multi-agent collaboration improves accuracy.

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd mass
pip install -e .
```

### Configuration

#### 1. Set up API keys with .env file (recommended)

```bash
# Run the setup script
python3 setup_env.py

# Edit the .env file with your API keys
# .env file will be created from .env.example
```

Your `.env` file should look like:

```bash
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-key-here

# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

#### 2. Create agents with configurations

```python
from mass import AgentConfig, MassAgent, MassOrchestrator
import os

# API keys are automatically loaded from .env file
custom_config = AgentConfig(
    backend_params={
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 4000,
        "temperature": 0.7,
        "timeout_seconds": 300
    }
)

# Create agents
agent = MassAgent("custom_agent", custom_config)

# Create different agent configurations
analytical_config = AgentConfig(
    backend_params={
        "model": "gpt-4o-mini",
        "temperature": 0.2,  # More focused
        "max_tokens": 6000
    }
)
creative_config = AgentConfig(
    backend_params={
        "model": "gpt-4o-mini",
        "temperature": 0.8,  # More creative
        "max_tokens": 4000
    }
)
```

### Basic Usage

#### CLI Commands

*Note: CLI interface is planned for future development. Currently use Python API only.*

#### Python API

```python
import asyncio
from mass import MassAgent, MassOrchestrator, AgentConfig

async def main():
    # Create orchestrator
    orchestrator = MassOrchestrator(max_duration=30.0)

    # Configure different agent types
    analytical_config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 6000
        }
    )
    creative_config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.8,
            "max_tokens": 4000
        }
    )

    # Add agents with different personalities
    agents = [
        MassAgent("analytical_agent", analytical_config),
        MassAgent("creative_agent", creative_config)
    ]

    for agent in agents:
        orchestrator.add_agent(agent)

    # Run orchestration
    result = await orchestrator.orchestrate(
        "What is the derivative of x^2 * sin(x)?"
    )

    print(f"Answer: {result.final_response}")
    print(f"Representative: {result.representative_agent}")
    print(f"Time: {result.total_time:.1f}s")

asyncio.run(main())
```

## 📊 Benchmark Types

*Comprehensive benchmarking framework is planned for future development.*

### Planned Features

- **HLE-Style Tasks**: Human-Level Evaluation benchmark integration
- **Multi-domain evaluation**: Math, reasoning, coding, language tasks
- **Performance metrics**: Accuracy, cost efficiency, response time
- **Comparative analysis**: Agent configuration comparison

## 🔧 Extending the Framework

### Adding New Agent Backends

```python
from mass import MassAgent, AgentConfig

# Create custom agent configuration for different models
openai_config = AgentConfig(
    model_name="gpt-4",
    api_key="your-openai-key",
    temperature=0.7
)

anthropic_config = AgentConfig(
    model_name="claude-3-sonnet-20240229",
    api_key="your-anthropic-key",
    temperature=0.7
)

# Agents use the same interface regardless of backend
gpt_agent = MassAgent("gpt_agent", openai_config)
claude_agent = MassAgent("claude_agent", anthropic_config)
```

### Creating Custom Benchmarks

*Benchmarking framework is planned for future development.*

## 📈 Performance Analysis

### Configuration Comparison

```python
import asyncio
from mass import MassAgent, MassOrchestrator
from mass import AgentConfig

async def compare_configurations():
    # Configure different agent types
    analytical_config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 6000
        }
    )
    creative_config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.8,
            "max_tokens": 4000
        }
    )
    practical_config = AgentConfig(
        backend_params={
            "model": "gpt-4o-mini",
            "temperature": 0.5,
            "max_tokens": 3000
        }
    )

    # Test different agent combinations
    configs = [
        ("analytical_team", [analytical_config, analytical_config]),
        ("diverse_team", [analytical_config, creative_config, practical_config]),
        ("creative_team", [creative_config, creative_config])
    ]

    task = "Design a sustainable urban transportation system"
    results = {}

    for config_name, agent_configs in configs:
        orchestrator = MassOrchestrator()
        for i, config in enumerate(agent_configs):
            agent = MassAgent(f"agent_{i}", config)
            orchestrator.add_agent(agent)

        result = await orchestrator.orchestrate(task)
        results[config_name] = result

    return results
```

### Detailed Analytics

- Agent performance rankings
- Task type difficulty analysis
- Consensus vs accuracy correlation
- Response time vs quality tradeoffs
- Provider-specific strengths/weaknesses

## 🔒 Security & Best Practices

- API keys stored as environment variables
- Rate limiting and timeout protection
- Secure configuration management
- No logging of sensitive prompts (configurable)
- Graceful failure handling

## 📚 Examples

See the `examples/` directory for complete working examples:

- `summary_demo.py` - Complete demonstration of the MASS system with streaming interactions
- `mock_backend_demo.py` - Demo using mock backend for testing
- `flexible_tools_demo.py` - Demonstration of flexible tool usage
- `trace_demo.py` - API tracing and monitoring example
- `user_tools_demo.py` - Custom user tools integration

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
black mass/
isort mass/
flake8 mass/
mypy mass/
```

## 📖 Configuration Reference

### AgentConfig Options

```python
from mass import AgentConfig

config = AgentConfig(
    # Model settings
    model_name="claude-3-sonnet-20240229",  # LLM model to use
    api_key=None,  # API key (or use environment variables)
    max_tokens=4000,  # Maximum tokens per request
    temperature=0.7,  # Response randomness (0.0-1.0)
    timeout_seconds=300,  # Request timeout

    # Framework-specific settings
    message_templates=None,  # Custom message templates

    # Instructions for different behaviors (customizable)
    summary_update_instructions="...",  # How to use update_summary()
    notification_handling_instructions="...",  # How to use check_updates()
    stopping_voting_instructions="...",  # When to vote() and stop
    final_answer_instructions="...",  # How to present final answers
    collaboration_instructions="...",  # How to collaborate
    reactivation_instructions="...",  # How to handle restart_work()

    # Advanced options
    enable_streaming=True,  # Enable streaming responses
    max_restart_attempts=3,  # Max restart attempts
    cost_limits={  # Cost tracking limits
        "max_input_tokens": 50000,
        "max_output_tokens": 20000,
        "max_cost_usd": 10.0
    }
)
```

### Custom Configurations

```python
from mass import AgentConfig

# Create custom configurations for different use cases
analytical_config = AgentConfig(
    backend_params={
        "model": "gpt-4o-mini",
        "temperature": 0.2,      # Focused and systematic
        "max_tokens": 6000,      # Detailed analysis
        "builtin_tools": ["web_search", "code_interpreter"]
    }
)

creative_config = AgentConfig(
    backend_params={
        "model": "gpt-4o-mini",
        "temperature": 0.8,      # Creative and innovative
        "max_tokens": 4000,      # Concise but creative
        "builtin_tools": ["web_search"]
    }
)

practical_config = AgentConfig(
    backend_params={
        "model": "gpt-4o-mini",
        "temperature": 0.5,      # Balanced approach
        "max_tokens": 3000,      # Practical solutions
        "builtin_tools": ["web_search"]
    }
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

- 📖 Documentation: See inline docstrings and examples
- 🐛 Issues: Please report bugs via GitHub issues
- 💡 Feature Requests: Submit enhancement proposals via GitHub

## 🎯 Roadmap

- [ ] Larger scale test (sponsorship is appreciated)
- [ ] Support for more model providers (Cohere, Together, etc.)
- [ ] Web interface for benchmark management
- [ ] Cost optimization and budget management
- TODO

---

**Built with ❤️ for advancing multi-agent AI research and practical applications.**
