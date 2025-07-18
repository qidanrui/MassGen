# MASS (Multi-Agent Scaling System)

MASS is an advanced multi-agent orchestration system that leverages multiple AI agents to collaboratively solve complex tasks through intelligent parallel processing, peer verification, and consensus-building.

## Overview

The system assigns the same task to multiple AI agents who work independently while observing and learning from each other's progress. Agents collaborate through a structured workflow that ensures high-quality solutions via:

- **Parallel Processing**: Multiple agents tackle problems simultaneously
- **Peer Review**: Agents examine and borrow insights from each other's work and reasoning
- **Consensus Building**: Agents vote for a representative to present the final answer after reviewing all solutions
- **Fallback Mechanism**: If consensus isn't reached, majority voting determines the outcome (if tie, choose the one who finished working earlist)

## System Architecture

### Agent Interface

All agents implement the **`MassAgent`** interface, providing:

- **Core Function**: `process_message()` - Primary LLM inference function with built-in search and code execution tools to solve any task
- **Orchestration Tools**: Built-in functions for summary updates, status checking, and voting, maintained by the system
- **Standardized Communication**: Consistent interaction protocols across all agent types

### Agent State Management

The system maintains real-time information for each agent:

- **Summary Report**: Comprehensive progress report
- **Status**: Current state (`working` or `voted`)
- **Execution Metrics**: Runtime and resource usage tracking
- **Update History**: Record of changes for each agent's summary, status, etc (saved in logs for analysis)

## Agent Workflow

Each agent follows a structured four-phase workflow designed to maximize collaboration and solution quality:

### Phase 1: Initial Processing

**Objective**: Generate independent solution as starting point

**Process**:
1. Agent receives the task and processes it using `process_message()`
2. Develops initial solution with comprehensive summary report and answer
3. System automatically saves the summary and makes it visible to all peer agents

### Phase 2: Collaboration & Refinement

**Objective**: Leverage peer insights to improve solutions or identify the best candidate

**Process**:
1. Agent continuously monitors peer updates every `check_update_frequency` (default: 3 seconds)
2. **Upon discovering new peer summary reports**:
   - Triggers new inference cycle to reassess the situation
   - Analyzes own solution alongside all peer solutions against the original task
   - Evaluates the relative quality and correctness of each solution
3. **Agent chooses one of two actions**:
   - **Continue Working**: Incorporates peer insights to refine own solution, maintains `working` status
   - **Vote**: Identifies the agent with the superior solution and changes status to `voted`

### Phase 3: Consensus Building & Debate

**Objective**: Reach agreement on the best solution through iterative review

**Process**:
1. **Consensus Check**: When all agents have voted, the system evaluates whether the results meet the majority voting threshold
2. **Outcome Determination**:
   - **Consensus Achieved**: The majority-selected agent becomes the representative
   - **No Consensus**: System initiates structured debate, providing all solutions and voting rationales to each agent for reconsideration
3. **Iterative Process**: Phases 2 and 3 alternate until either consensus is reached or maximum rounds are completed

### Phase 4: Final Presentation

**Objective**: Deliver the definitive solution with full context

**Process**:
The selected representative agent receives the complete collection of all peer solutions and voting opinions, then synthesizes and presents the final answer with full reasoning and context.


## Configuration

### Agent Setup
- Agents configured with IDs (0 to N)
- Flexible backend support (any LLM with tool capabilities)
- Customizable instructions for each workflow phase

---

*The system provides complete session state information to enable sophisticated orchestration strategies while maintaining simplicity through the standardized `MassAgent` interface.*


## TODO List
1. update the voting mechanism (if tie, choose the earlist one), choose the representative to present the answer
2. record the inference input and output
3. record the summary update and voting update
4. Find a clear example where multiple agents can work while single-agent not work