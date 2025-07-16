# MASS (Multi-Agent Scaling System)

MASS is an advanced multi-agent coordination system that leverages multiple AI agents to collaboratively solve complex tasks through intelligent parallel processing, peer verification, and consensus-building.

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
- **Coordination Tools**: Built-in functions for summary updates, status checking, and voting, maintained by the system
- **Standardized Communication**: Consistent interaction protocols across all agent types

### Agent State Management

The system maintains real-time information for each agent:

- **Summary Report**: Comprehensive progress report
- **Status**: Current state (`working` or `voted`)
- **Execution Metrics**: Runtime and resource usage tracking
- **Update History**: Record of changes for each agent's summary, status, etc (saved in logs for analysis)

## Agent Workflow

Each agent follows a structured three-phase workflow:

### Phase 1: Initial Processing

**Objective**: Generate initial solution independently

**Process**:
1. Agent receives the task and processes it using `process_message()`
2. Generates initial solution with summary report and answer
3. System saves the summary and makes it available to peer agents

**Response Format**:
- Include agent ID in the response
- Conclude with:
```
### Summary report
[progress and solution summary with all necessary steps, evidence, and information sources]
```

### Phase 2: Collaboration & Refinement

**Objective**: Review peer solutions and refine or vote for the best answer

**Process**:
1. Agent checks for peer updates every `check_update_frequency` (default: 3 seconds)
2. **When new peer summary reports are available**:
   - Reviews own solution and all peer solutions against the original task
   - Evaluates whether any agent (including self) has found the correct solution
3. **Agent decides**:
   - **Continue Working**: Update solution incorporating peer insights, maintain `working` status
   - **Vote**: Select the agent with the best solution, change status to `voted`

**Response Format**:
- Include all agents' IDs with their summary reports
- Conclude with:
```
### Summary report
[updated solution or evaluation of peer solutions]

## Voting Decision
{"voting: {agent_id} or None}
```
*Use "None" if no satisfactory solution exists yet; use "{agent_id}" if voting for the best solution*

### Phase 3: Consensus & Presentation

**Objective**: Determine final answer through consensus

**Process**:
1. Voted agents may reactivate if peers provide new updates
2. Process continues until all agents vote or maximum rounds reached (default: 5)
3. **Majority Voting**: Select agent with most votes (ties broken by earliest vote timestamp)
4. **Final Presentation**: Selected agent presents final answer including all recent peer summaries

## Configuration

### Agent Setup
- Agents configured with IDs (0 to N)
- Flexible backend support (any LLM with tool capabilities)
- Customizable instructions for each workflow phase

---

*The system provides complete session state information to enable sophisticated coordination strategies while maintaining simplicity through the standardized `MassAgent` interface.*


## TODO List
1. update the voting mechanism (if tie, choose the earlist one), choose the representative to present the answer
2. record the inference input and output
3. record the summary update and voting update
4. Find a clear example where multiple agents can work while single-agent not work