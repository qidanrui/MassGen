# MASS Implementation Documentation

This document provides a comprehensive mapping between the design features described in `DESIGN_DOC` and their actual implementation in the MASS codebase.

## 🏗️ Architecture Implementation

### Single Agent vs Multi-Agent Processing

**Design Feature**: The MASS system supports both single-agent and multi-agent scenarios with automatic detection.

**Implementation**: 
- **Location**: `mass/main.py`, function `_run_single_agent_simple()` (lines 50-111)
- **Detection Logic**: The system checks `len(config.agents) == 1` to automatically switch to single-agent mode
- **Single Agent Path**: 
  - Bypasses the full `MassOrchestrator` system entirely
  - Creates a single agent with `orchestrator=None`
  - Uses simple conversation format with system/user messages
  - Calls `agent.process_message()` directly
  - Returns formatted response mimicking multi-agent system structure
- **Multi-Agent Path**: Uses the full `MassOrchestrator` system via `run_mass_system()` function

**Key Code**:
```python
# mass/main.py:74-90
def _run_single_agent_simple(question: str, config: MassConfig) -> Dict[str, Any]:
    # Create the single agent without orchestrator (None)
    agent = create_agent(
        agent_type=agent_config.agent_type,
        agent_id=agent_config.agent_id,
        orchestrator=None,  # No orchestrator needed for single agent
        model_config=agent_config.model_config,
        stream_callback=None
    )
```

### Multi-Agent Architecture Implementation

**Design Feature**: Each agent works asynchronously, maintains working summaries, and shares updates.

**Implementation**:
- **Core Class**: `MassOrchestrator` in `mass/orchestrator.py`
- **Agent State Management**: `AgentState` dataclass in `mass/types.py` (lines 76-125)
- **Asynchronous Execution**: Uses `ThreadPoolExecutor` for parallel agent execution
- **Update Tracking**: Each agent maintains `update_history: List[SummaryRecord]` and `seen_updates_timestamps: Dict[int, float]`

**Agent State Structure**:
```python
# mass/types.py:76-125
@dataclass
class AgentState:
    agent_id: int
    status: str = "working"  # "working", "voted", "failed"
    working_summary: str = ""
    update_history: List[SummaryRecord] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    vote_target: Optional[int] = None
    seen_updates_timestamps: Dict[int, float] = field(default_factory=dict)
```

## Agent Workflow and Consensus Logic Implementation

### Voting and Consensus System

**Design Feature**: Agents vote for representatives, system checks consensus, handles debate phases.

**Implementation**:
- **Voting Function**: `cast_vote()` in `mass/orchestrator.py` (lines 151-252)
- **Consensus Checking**: `_check_consensus()` in `mass/orchestrator.py` (lines 254-300)
- **Vote Storage**: `List[VoteRecord]` with timestamp and reason tracking

**Consensus Logic Implementation**:
```python
# mass/orchestrator.py:254-300
def _check_consensus(self) -> bool:
    total_agents = len(self.agents)
    failed_agents_count = len([s for s in self.agent_states.values() if s.status == "failed"])
    votable_agents_count = total_agents - failed_agents_count
    
    vote_counts = Counter(vote.target_id for vote in self.votes)
    votes_needed = max(1, int(votable_agents_count * self.consensus_threshold))
    
    if vote_counts and vote_counts.most_common(1)[0][1] >= votes_needed:
        winning_agent_id = vote_counts.most_common(1)[0][0]
        self._reach_consensus(winning_agent_id)
        return True
```

### Agent Restart Logic

**Design Feature**: Agents restart when they receive new updates from peers, even after voting.

**Implementation**:
- **Update Notification System**: Orchestrator tracks which agents have seen which updates
- **Restart Triggers**: Implemented in `has_unseen_updates()` method in `AgentState`
- **Restart Execution**: Agents call `work_on_task()` with `restart_instruction` parameter

## Agent Backend Implementation

**Design Feature**: Support for OpenAI, Gemini, and Grok backends with unified interface.

**Implementation**:
- **Backend Modules**: `mass/backends/` directory containing `oai.py`, `gemini.py`, `grok.py`
- **Wrapper Classes**: `mass/agents.py` contains `OpenAIMassAgent`, `GeminiMassAgent`, `GrokMassAgent`
- **Factory Function**: `create_agent()` in `mass/agents.py` creates appropriate agent based on type
- **Model Mapping**: `get_agent_type_from_model()` in `mass/utils.py` maps model names to agent types

**Agent Creation**:
```python
# mass/agents.py:340-349
def create_agent(agent_type: str, agent_id: int, orchestrator=None, model_config=None, stream_callback=None):
    if agent_type == "openai":
        return OpenAIMassAgent(agent_id, orchestrator, model_config, stream_callback)
    elif agent_type == "gemini":
        return GeminiMassAgent(agent_id, orchestrator, model_config, stream_callback)
    elif agent_type == "grok":
        return GrokMassAgent(agent_id, orchestrator, model_config, stream_callback)
```

## Agent Interface Implementation

**Design Feature**: Unified `MassAgent` interface with core methods.

**Implementation**:
- **Base Class**: `MassAgent` in `mass/agent.py` (lines 86-369)
- **Core Methods**: All backends implement the same interface

### Core Methods Implementation:

1. **`process_message()`**: 
   - **Location**: Implemented in each backend (`mass/backends/`)
   - **Function**: Handles LLM communication with tools
   - **Returns**: `AgentResponse` object with text, code, citations, function_calls

2. **`work_on_task()`**:
   - **Location**: Each agent wrapper class in `mass/agents.py`
   - **Function**: Main task execution loop with conversation continuation
   - **Handles**: Initial work and restarts with new instructions

3. **`present_final_answer()`**:
   - **Implementation**: Uses `process_message()` with special prompt for final presentation
   - **Triggered**: When agent is selected as representative

## LLM-Callable Tools Implementation

**Design Feature**: `update_summary()` and `vote()` tools available to all agents.

**Implementation**:
- **System Tools**: Registered in each agent's `_get_available_tools()` method
- **Tool Registration**: Dynamic registration system in `mass/tools.py`
- **Orchestrator Integration**: Tools call back to orchestrator methods

**System Tools Registration**:
```python
# In each agent class (e.g., OpenAIMassAgent)
def _get_available_tools(self):
    tools = {}
    
    # Add system tools
    tools["update_summary"] = {
        "function": lambda content: self.orchestrator.update_agent_summary(self.agent_id, content),
        "description": "Record your working process and final summary report"
    }
    
    tools["vote"] = {
        "function": lambda agent_id: self.orchestrator.cast_vote(self.agent_id, agent_id),
        "description": "Vote for the representative agent to solve the task"
    }
```

**Built-in Tools**:
- **Location**: `mass/tools.py`
- **Available Tools**: `python_interpreter`, `calculator`
- **Registration**: Global `register_tool` dictionary

## Orchestrator System Implementation

**Design Feature**: Central coordination system managing agent status, voting, and session state.

**Implementation**:
- **Core Class**: `MassOrchestrator` in `mass/orchestrator.py`
- **Responsibilities**:
  - Agent registration and lifecycle management
  - Vote collection and consensus checking
  - Update distribution and notification
  - Phase management and coordination

**Key Orchestrator Components**:

1. **Agent Management**:
   ```python
   self.agents: Dict[int, Any] = {}  # agent_id -> MassAgent instance
   self.agent_states: Dict[int, AgentState] = {}  # agent_id -> AgentState
   ```

2. **Voting System**:
   ```python
   self.votes: List[VoteRecord] = []  # All votes with timestamps and reasons
   ```

3. **System State**:
   ```python
   self.system_state = SystemState()  # Overall system phase and status
   ```

4. **Synchronization**:
   ```python
   self._lock = threading.RLock()  # Thread-safe operations
   self._stop_event = threading.Event()  # Coordination signal
   ```

## Streaming Display Implementation

**Design Feature**: Multi-region display showing per-agent progress and system status.

**Implementation**:
- **Core Class**: `MultiRegionDisplay` in `mass/streaming_display.py`
- **Features**:
  - Individual agent columns with real-time output
  - System status panel with phase transitions
  - Vote distribution tracking
  - File logging for all conversations

**Display Components**:
```python
# mass/streaming_display.py:15-30
class MultiRegionDisplay:
    def __init__(self):
        self.agent_outputs: Dict[int, str] = {}  # Agent conversations
        self.agent_models: Dict[int, str] = {}   # Model names per agent
        self.agent_statuses: Dict[int, str] = {} # Agent status tracking
        self.system_messages: List[str] = []     # System events
        self.vote_distribution: Dict[int, int] = {}  # Current votes
```

## Configuration System Implementation

**Design Feature**: Support for YAML configuration files and programmatic model-based configs.

**Implementation**:
- **Core Module**: `mass/config.py`
- **Configuration Types**: `MassConfig` dataclass in `mass/types.py` (lines 191-228)

**Configuration Loading**:
1. **YAML Loading**: `load_config_from_yaml()` function
2. **Model-based Creation**: `create_config_from_models()` function
3. **CLI Integration**: Command-line argument parsing in `cli.py`

**Configuration Structure**:
```python
# mass/types.py:191-228
@dataclass
class MassConfig:
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    agents: List[AgentConfig] = field(default_factory=list)
    streaming_display: StreamingDisplayConfig = field(default_factory=StreamingDisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
```

## Entry Points and CLI Implementation

**Design Feature**: Clean command-line interface supporting both config files and direct model specification.

**Implementation**:
- **CLI Module**: `cli.py` (lines 1-129)
- **Main Module**: `mass/main.py` provides programmatic interfaces
- **Key Functions**:
  - `run_mass_with_config()`: Main entry point for YAML configs
  - `run_mass_agents()`: Simple entry point for model lists
  - `MassSystem` class: Object-oriented interface

**CLI Usage Patterns**:
```bash
# YAML configuration
python cli.py "Question" --config examples/production.yaml

# Direct model specification
python cli.py "Question" --models gpt-4o gemini-2.5-flash

# Single agent mode (automatic detection)
python cli.py "Question" --models gpt-4o
```

## Logging and Session Management

**Design Feature**: Comprehensive logging with session tracking and file output.

**Implementation**:
- **Logging Manager**: `MassLogManager` in `mass/logging.py`
- **Session Tracking**: Timestamped log directories
- **Event Types**: Agent updates, voting events, phase changes, status changes
- **File Output**: Separate files for each agent and system messages

**Log Structure**:
```
logs/
├── YYYYMMDD_HHMMSS/
│   ├── agent_1.txt
│   ├── agent_2.txt
│   ├── system.txt
│   └── session_metadata.json
```

## Error Handling and Edge Cases

**Implementation Features**:
1. **Agent Failure Handling**: `mark_agent_failed()` method in orchestrator
2. **Timeout Management**: Processing timeouts for individual agents
3. **Consensus Edge Cases**: Single agent scenarios, failed agent exclusion
4. **Configuration Validation**: Comprehensive validation in `MassConfig.validate()`

## Performance Optimizations

**Implementation Features**:
1. **Thread Pool Management**: Configurable pool size and timeout
2. **Non-blocking Logging**: Optional non-blocking file I/O
3. **Memory Management**: Limited conversation history and display lines
4. **Efficient State Tracking**: Delta-based update notifications

---

This implementation successfully realizes all the design features described in the DESIGN_DOC, providing a robust, scalable, and user-friendly multi-agent system with comprehensive tooling and monitoring capabilities. 