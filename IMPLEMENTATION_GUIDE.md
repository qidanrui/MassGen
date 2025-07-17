# MASS Implementation Guide

## Overview

The MASS (Multi-Agent Scaling System) has been successfully implemented based on the design specifications in the README. This guide explains how to use the system to run multi-agent collaborative tasks.

## System Architecture

The implementation consists of several key components:

### 1. Core Interfaces (`mass_agent.py`)

- **`MassAgent`**: Abstract base class that all agents must implement
- **`AgentState`**: Manages individual agent state (status, summaries, execution metrics)  
- **`TaskInput`**: Standardized task representation
- **`AgentResponse`**: Standardized response format
- **Tool Separation**: Agents only have access to `live_search` and `code_execution` tools
- **Coordination**: `update_summary()`, `check_updates()`, `vote()` are called by workflow system, not agents

### 2. Coordination System (`mass_coordination.py`)

- **`MassCoordinationSystem`**: Central coordination hub
- **`VoteRecord`**: Vote tracking
- **`SystemState`**: Overall system state management
- Thread-safe communication and state synchronization
- Consensus detection and fallback mechanisms

### 3. Workflow Management (`mass_workflow.py`)

- **`MassWorkflowManager`**: Orchestrates the four-phase workflow
- Parallel and sequential execution modes
- Progress tracking and error handling
- Phase-specific callback support

### 4. Agent Implementations (`mass_agents.py`)

- **`OpenAIMassAgent`**: Wrapper for OpenAI agents
- **`GeminiMassAgent`**: Wrapper for Gemini agents  
- **`GrokMassAgent`**: Wrapper for Grok agents
- **`create_agent()`**: Factory function for agent creation

### 5. Main Entry Point (`mass_main.py`)

- **`MassSystem`**: High-level system orchestrator
- Command-line interface
- Configuration management
- Results export functionality

## Quick Start

### Basic Usage

```python
from mass_main import MassSystem

# Initialize system
system = MassSystem()

# Run a task with multiple agents
results = system.run_task_from_file(
    task_file_path="hle/test_case.json",
    agent_types=["openai", "gemini", "grok"]
)

# Check results
if results["success"]:
    final_solution = results["final_solution"]
    print(f"Winner: Agent {final_solution['agent_id']}")
    print(f"Solution: {final_solution['solution']}")
```

### Command Line Usage

```bash
# Basic execution
python3 mass_main.py \
    --task-file hle/test_case.json \
    --agents openai,gemini,grok

# With custom configuration
python3 mass_main.py \
    --task-file task.json \
    --agents openai,openai,gemini \
    --max-rounds 5 \
    --consensus-threshold 0.8 \
    --output results.json

# Sequential execution (instead of parallel)
python3 mass_main.py \
    --task-file task.json \
    --agents openai,gemini \
    --sequential
```

## Tool Architecture & Coordination

### Agent Tool Access
**IMPORTANT**: Agents only have access to two tools:
- `live_search` - Web search capabilities
- `code_execution` - Python code execution

### Coordination Functions
The coordination functions are **NOT** exposed as tools to agents:
- `update_summary()` - Called by workflow system after agent responses
- `check_updates()` - Called by workflow system to provide peer context
- `vote()` - Called by workflow system based on agent response analysis

### Workflow System Responsibilities
The workflow orchestrates coordination by:
1. Analyzing agent responses for voting intentions
2. Calling coordination functions at appropriate times
3. Providing peer context to agents during collaboration phases
4. Managing consensus detection and fallback mechanisms

## Four-Phase Workflow

### Phase 1: Initial Processing

All agents work independently on the task:

1. Each agent receives the task
2. Agents process using `live_search` and `code_execution` tools
3. Workflow system calls `update_summary()` with agent responses
4. No voting occurs in this phase

### Phase 2: Collaboration & Refinement

Agents review each other's work with workflow coordination:

1. Workflow system calls `check_updates()` and provides peer solutions as context
2. Agents analyze peer solutions and either:
   - Provide an improved solution (workflow calls `update_summary()`)
   - Indicate preference for another agent's solution (workflow calls `vote()`)
3. Multiple rounds may occur until consensus or voting completes

### Phase 3: Consensus Building & Debate

Structured debate and consensus building:

1. System checks if consensus is reached from Phase 2
2. If no consensus, remaining working agents engage in structured debate
3. Workflow analyzes responses for voting patterns (e.g., "Agent X has the best solution")
4. Iterative review process continues until consensus emerges
5. Fallback to majority vote if maximum rounds reached

### Phase 4: Final Presentation

Selected representative presents the final solution:

1. The winning agent (determined by consensus or majority vote) becomes the representative
2. Representative receives complete context of all peer solutions and voting opinions
3. Representative synthesizes and presents the final answer with full reasoning
4. System finalizes with comprehensive solution

## Configuration Options

### System Configuration

```python
config = {
    "max_rounds":5,                    # Maximum collaboration rounds
    "consensus_threshold": 1.0,          # Unanimous (1.0) vs majority (0.5+)
    "max_phase_duration": 300,           # Max seconds per phase
    "parallel_execution": True           # Parallel vs sequential
}

system = MassSystem(config=config)
```

### Agent Configuration

```python
agent_configs = [
    {
        "type": "openai",
        "kwargs": {
            "model": "o3",
            "temperature": 0.3
        }
    },
    {
        "type": "gemini", 
        "kwargs": {
            "model": "gemini-2.5-pro",
            "temperature": 0.5
        }
    }
]

system.initialize_system(agent_configs)
```

## Task Format

Tasks should follow this JSON structure:

```json
{
    "id": "unique_task_id",
    "question": "The main question or task description",
    "context": {
        "additional": "context information",
        "task_type": "classification",
        "difficulty": "medium"
    }
}
```

## Results Structure

The system returns comprehensive results:

```python
{
    "success": True,
    "total_workflow_time": 45.2,
    "final_solution": {
        "agent_id": 1,
        "solution": "Final answer text...",
        "execution_time": 12.5,
        "total_rounds": 2,
        "vote_distribution": {1: 3},
        "all_agent_summaries": {...}
    },
    "phase_results": {
        "initial": [...],
        "collaboration": [...], 
        "consensus": [...]
    },
    "system_status": {...},
    "session_log": {...}
}
```

## Error Handling

The system includes robust error handling:

- **Agent Failures**: Individual agent errors don't crash the system
- **Import Errors**: Missing dependencies gracefully degrade functionality
- **Timeout Handling**: Phase duration limits prevent infinite execution
- **Fallback Mechanisms**: Majority voting if consensus isn't reached

## Dependencies

### Required for Core System
- Python 3.6+
- Standard library modules (threading, concurrent.futures, etc.)

### Required for Agent Backends
- OpenAI: `openai` package + API key
- Gemini: `google-generativeai` package + API key  
- Grok: `xai-sdk` package + API key

### Optional
- `python-dotenv` for environment variable management

## Testing

Run the validation suite:

```bash
python3 test_mass_system.py
```

This tests all core components without requiring API keys or external dependencies.

## Advanced Usage

### Custom Agents

Create custom agent implementations:

```python
class CustomAgent(MassAgent):
    def process_message(self, messages, tools=None, temperature=0.5, **kwargs):
        # Your custom logic here
        return AgentResponse(
            text="Custom response",
            code=[],
            citations=[],
            function_calls=[]
        )

# Register and use
agent = CustomAgent(agent_id=0, coordination_system=coord_system)
```

### Progress Monitoring

Add progress callbacks:

```python
def progress_callback(phase, current, total):
    print(f"Phase {phase}: {current}/{total} agents completed")

results = system.run_task(task, progress_callback=progress_callback)
```

### Session Analysis

Export detailed session logs:

```python
session_log = results["session_log"]
# Analyze agent interactions, voting patterns, consensus formation
```

## Best Practices

1. **Agent Diversity**: Use different agent types for varied perspectives
2. **Consensus Threshold**: Adjust based on task criticality (1.0 for critical, 0.7+ for most tasks)
3. **Round Limits**: Set appropriate max_rounds (5-10 typically sufficient)
4. **Error Monitoring**: Check individual agent success rates in results
5. **Resource Management**: Consider parallel vs sequential based on system capacity

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Python 3.6+ and required packages installed
2. **API Key Issues**: Set environment variables or pass in agent kwargs
3. **Timeout Errors**: Increase max_phase_duration for complex tasks
4. **Memory Issues**: Use sequential execution for large agent counts

### Debug Mode

Enable verbose logging:

```bash
python3 mass_main.py --task-file task.json --agents openai,gemini --verbose
```

This implementation provides a complete, robust multi-agent system that follows all specifications from the original README design document. 