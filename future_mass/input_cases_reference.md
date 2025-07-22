# MASS Input Cases Reference

This document provides comprehensive reference for all possible input cases to the MASS (Multi-Agent Summary System) backend using the **proven working approach** that eliminates perfectionism loops.

## 🎯 Executive Summary

The MASS system uses a **binary decision framework** where agents either:
1. **Vote** for the best existing answer (if adequate)
2. **Create new answer** with additional work (if inadequate)

This eliminates endless update loops and produces reliable decisions across diverse domains.

## ⚠️ CRITICAL IMPLEMENTATION INSIGHTS

### 1. Tool Naming Affects Behavior
**CRUCIAL**: Use `new_answer` with clear description:

- ✅ **`new_answer`**: "Provide improved answer to ORIGINAL MESSAGE" → Perfect behavior
- ❌ **`update_summary`**: "Create/update working summary" → Historical issue: caused voting bias

**Key Insight**: Tool names and descriptions significantly influence agent decision-making patterns.

### 2. Backend API Evolution
- **OpenAI Responses API**: Uses `input` parameter (not `messages`)
- **Conversation Context**: Full message history preserved including assistant responses
- **Built-in Tools**: Web search, code execution integrate seamlessly

### 3. Domain Agnostic Success
Proven across research (econometrics), computation (Tower of Hanoi), and analytical tasks.

## New Simplified Approach

Based on extensive testing, the MASS system now uses a simplified message structure that eliminates perfectionism loops and produces reliable tool usage decisions.

**Key Principle**: Each agent starts with a **single user message** containing: task + current answers + decision prompt. Additional enforcement messages may follow if needed.

## Core Components

### System Message Structure
```
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
```

- **Evaluation context**: Establishes the agent's role and decision framework
- **Tool guidance**: Clear instructions on when to use each tool

### User Message Structure  
```
<ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS from the agents>
{agent_summaries_or_none}
<END OF CURRENT ANSWERS>
```

### Tool Definitions
- **new_answer(content)**: Provide improved answer to the ORIGINAL MESSAGE
- **vote(agent_id, reason)**: Vote for best agent with reasoning

## Input Cases

**🔄 Dynamic Case Transitions**: Cases are not static! When any agent calls `new_answer`, all other agents automatically transition to Case 2 for their next inference call, since new summaries become available.

**Key Transition Rules**:
- Case 1 → Case 2: When any agent provides `new_answer`
- Case 3 → Case 2: When enforcement succeeds or other agents have provided `new_answer`  
- Case 4 → Case 2: When error recovery succeeds or other agents have provided `new_answer`
- Case 2 → Case 2: When any agent provides `new_answer` during evaluation

### Case 1: Initial Work (No Summaries Exist)

**When**: Agent has no summary yet, no other agents have summaries

**System Message**:
```
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
```

**User Message**:
```
<ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS from the agents>
(no answers available yet)
<END OF CURRENT ANSWERS>
```

**Expected Behavior**: Agent calls `new_answer` (no adequate answers exist)

---

### Case 2: Decision Making (Summaries Exist)

**When**: Agent has summary, other agents may have summaries  

**System Message**:
```
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
```

**User Message**:
```
<ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS from the agents>
<Agent A> {summary_A} <end of Agent A>
<Agent B> {summary_B} <end of Agent B>
<Agent C> {summary_C} <end of Agent C>
<END OF CURRENT ANSWERS>
```

**Expected Behavior**: 
- Agent calls `vote(agent_id, reason)` if a good answer exists
- Agent calls `new_answer(content)` if no adequate answer exists

---

### Case 3: Non-Workflow Response Handling

**When**: Agent calls other tools or provides text response without using `new_answer`/`vote`

**System Message**:
```
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
```

**Conversation Flow**:
```
User: <ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS from the agents>
<Agent A> {summary_A} <end of Agent A>
<Agent B> {summary_B} <end of Agent B>
<Agent C> {summary_C} <end of Agent C>
<END OF CURRENT ANSWERS>
Assistant: {may provide reasoning/analysis or calls other tools without new_answer/vote}
Tool Response: {result if tool call was made}
User: "Finish your work above by making a tool call of `vote` or `new_answer`. Make sure you actually call the tool."
Assistant: {may continue with non-workflow responses}
User: "Finish your work above by making a tool call of `vote` or `new_answer`. Make sure you actually call the tool."
...
```

**Expected Behavior**: Agent redirected to use proper workflow tools

**🔄 Transition Note**: If other agents provide `new_answer` while this agent is in Case 3 enforcement, this agent will transition to Case 2 for their next inference call.

---

### Case 4: Error Recovery  

**When**: Previous tool call failed (error sent via tool response)

**System Message**:
```
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
```

**Conversation Flow**:
```
User: <ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS from the agents>
<Agent A> {summary_A} <end of Agent A>
<Agent B> {summary_B} <end of Agent B>
<Agent C> {summary_C} <end of Agent C>
<END OF CURRENT ANSWERS>
Assistant: {may provide reasoning/analysis}
User: {may provide clarification or additional context}
Assistant: {may continue analysis}
Assistant: {calls tool - e.g., vote(agent_id: "invalid_agent", reason: "...")}
Tool Response: Error: {specific error message - e.g., "Invalid agent_id 'invalid_agent'. Valid agents: Agent A, Agent B, Agent C"}
```

**Expected Behavior**: Agent retries with corrected approach

**🔄 Transition Note**: If other agents provide `new_answer` while this agent is in Case 4 error recovery, this agent will transition to Case 2 for their next inference call.

## Decision Logic

The agent evaluates available answers and makes a binary decision:

1. **Good answer exists** → `vote(agent_id, reason)` for best agent
2. **No good answer exists** → `new_answer(content)` with better answer

This eliminates the perfectionism loop by:
- ✅ **Clear binary decision**: Either vote or update, not both
- ✅ **Task-focused evaluation**: "Does the best answer address the question?"
- ✅ **No self-reference confusion**: Agent evaluates others' answers objectively
- ✅ **Direct tool guidance**: Explicit instruction on which tool to use when

## 🛑 Stop Condition

**Coordination terminates when all agents have `has_voted = True`.**

**Voting State Tracking**:
- **Initialization**: All agents start with `has_voted = False`
- **After `vote` tool call**: Set agent's `has_voted = True`
- **After `new_answer` tool call**: Set ALL agents' `has_voted = False` (new answers invalidate previous votes)
- **Stop check**: Coordination continues until all agents have `has_voted == True`

This natural stop condition ensures:
- **Consensus reached**: All agents have voted and no new answers are being provided
- **Dynamic invalidation**: New answers reset the voting state, requiring fresh consensus
- **No infinite loops**: Process terminates when no agent wants to improve further
- **Semantic correctness**: Termination based on actual voting consensus, not arbitrary counters
- **Handles failures**: Agents that fail (remain `has_voted = False`) don't prevent termination

## Tool Usage Patterns

### Successful Patterns
- **new_answer**: Agent provides improved/complete answer
- **vote**: Agent selects existing adequate answer

### Problematic Patterns (Avoided)
- **Both tools**: Agent calls vote AND new_answer (perfectionism)
- **No tools**: Agent provides text response without tool usage
- **Wrong tool**: Agent votes when answer is inadequate, or creates new answer when answer is adequate

## Implementation Notes

1. **Message Construction**: Single user message combines all context
2. **System Message**: Combine user custom + evaluation role
3. **Agent Summaries**: Present all available summaries as "Current answers"
4. **Decision Prompt**: Always use exact wording proven to work in tests
5. **Tool Enforcement**: Clear instruction on which tool to use in each scenario

## 🧪 Comprehensive Testing Validation

This approach has been validated across **7 clean test files** covering diverse domains:

### 🎯 Core Input Cases (Documented Examples)
- ✅ **Case 1**: No summaries exist → `new_answer` correctly
- ✅ **Case 2**: Summaries exist → `vote` correctly  
- ✅ **Case 3**: Non-workflow response → enforcement redirection works
- ✅ **Case 4**: Error recovery → natural retry behavior without enforcement

### 📚 Research Domain: Econometrics IV Validity Testing
**Files**: `test_econometrics_case*_clean.py`

- ✅ **Case 1**: Agent uses web search to research Hansen J-test vs Anderson-Rubin
- ✅ **Case 2**: Evaluates 4 conflicting answers (B vs E), chooses best reasoning
- ✅ **Case 3**: Handles non-workflow response, enforces proper tool usage
- ✅ **Case 3 Extended**: Multiple non-workflow responses before final tool usage

**Key Findings**:
- Web search integration works seamlessly
- Complex econometric reasoning handled correctly
- Agents distinguish between Hansen J-test limitations and validity

### 🗼 Computational Domain: Tower of Hanoi Problem  
**Files**: `test_hanoi_case*_clean.py`

- ✅ **Case 1**: Uses code execution to solve optimal move calculation
- ✅ **Case 2**: Evaluates 4 solutions (18/11/21/27 moves), identifies validity issues

**Key Findings**:
- Code execution integration works perfectly
- Agents detect rule violations (Agent 39gjg's invalid 11-move solution)
- Computational verification enhances decision quality

### 🔧 Advanced Capabilities
- ✅ **Code Execution**: Seamless integration with OpenAI's built-in interpreter
- ✅ **Web Search**: Research enhancement for complex topics
- ✅ **Multi-turn Conversations**: Full conversation history preserved
- ✅ **Error Recovery**: Natural retry without additional prompting

### 📊 Edge Cases & Reliability
**Demonstrated across test suite**:
- ✅ Multiple correct answers → votes appropriately
- ✅ All inadequate answers → creates new answer  
- ✅ Mixed quality answers → identifies best option
- ✅ Wrong vs right answers → accurate discrimination

## 📈 System Evolution & Lessons Learned

### From Perfectionism Loops to Binary Decisions
**The Problem**: Early versions suffered from "perfectionism loops" where agents endlessly updated summaries instead of making decisions.

**The Solution**: Binary decision framework with clear tool naming and enforcement.

### Key Breakthroughs

1. **Tool Naming Discovery** (Critical for success)
   - `new_answer` with clear description works perfectly
   - Language shapes behavior more than expected

2. **OpenAI API Evolution** (January 2025)
   - Responses API replaces Chat Completions for agentic use
   - Built-in tools (web search, code execution) integrate seamlessly  
   - `input` parameter handles full conversation history

3. **Case 3 Enforcement Pattern**
   - Non-workflow responses require redirection
   - "Finish your work above by making a tool call" works consistently
   - Multiple enforcement rounds may be needed

4. **Domain Agnostic Framework**
   - Same approach works for research, computation, analysis
   - Web search enhances research quality
   - Code execution enables computational verification

## 🔧 Critical Implementation Details

### Tool Naming Impact Study

**CRUCIAL**: Tool naming significantly affects agent decision-making:

✅ **Working Tool Definition:**
```json
{
  "name": "new_answer", 
  "description": "Provide an improved answer to the ORIGINAL MESSAGE."
}
```

❌ **Problematic Tool Definition:**
```json
{
  "name": "update_summary", 
  "description": "Create/update agent's working summary"
}
```

**Result:** Agents have a systematic bias toward voting when the tool is named `update_summary`, even when answers are clearly inadequate. The term "summary" appears to suggest documentation work rather than problem-solving, making agents prefer to delegate via voting.

**Testing Evidence:**
- `new_answer` approach: 5/5 test cases passed  
- `update_summary` approach: 3/5 test cases passed (failed on inadequate answer detection)

### Code Execution Integration

**EXCELLENT**: The simplified approach works seamlessly with OpenAI's built-in code execution:

✅ **Workflow Compliance**: Agents correctly call `new_answer` when computational work is needed  
✅ **Automatic Code Execution**: When enabled with `enable_code_interpreter=True`, agents naturally use Python execution  
✅ **Complete Solutions**: Agents provide calculated numerical results, not just formulas  
✅ **Transparent Process**: Code execution is visible in the streaming output  

## 🎯 Quick Implementation Guide

### 1. System Message
```
You are evaluating answers from multiple agents for final response to a message. Does the best CURRENT ANSWER address the ORIGINAL MESSAGE?

If YES, use the `vote` tool to record your vote and skip the `new_answer` tool.
IF NO, do additional work first, then use the `new_answer` tool to record a better answer to the ORIGINAL MESSAGE. Make sure you actually call the tool.
```

### 2. Tool Definitions
```json
{
  "name": "new_answer",
  "description": "Provide an improved answer to the ORIGINAL MESSAGE",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {"type": "string", "description": "Your improved answer."}
    },
    "required": ["content"]
  }
}
```

```json
{
  "name": "vote", 
  "description": "Vote for the best agent to present final answer",
  "parameters": {
    "type": "object",
    "properties": {
      "agent_id": {"type": "string", "description": "ID of agent to vote for"},
      "reason": {"type": "string", "description": "Brief reason for choice"}
    },
    "required": ["agent_id", "reason"]
  }
}
```

### 3. User Message Format
```
<ORIGINAL MESSAGE> {task} <END OF ORIGINAL MESSAGE>

<CURRENT ANSWERS from the agents>
{agent_summaries_or_none}
<END OF CURRENT ANSWERS>
```

### 4. Enable Built-in Tools
```python
# Code execution for computational problems
enable_code_interpreter=True

# Web search for research tasks  
enable_web_search=True
```

## 🏆 Success Metrics

- **Input Case Validation**: 4/4 core cases pass consistently
- **Domain Testing**: Research + Computation + Analysis all work
- **Tool Usage**: Binary decisions eliminate perfectionism loops
- **Advanced Features**: Web search + code execution integrate seamlessly
- **Reliability**: 7 focused test files validate robustness across domains

The MASS framework now provides a proven, reliable approach for multi-agent decision-making across diverse domains.

**Example Behavior:**
```
💻 [Provider Tool: Code Interpreter] Starting execution...
💻 [Code Executed] P = 10000; r = 0.045; n = 12; t = 7...
✅ [Provider Tool: Code Interpreter] Execution completed

🔧 Tool called: ['new_answer']
📝 Agent's Answer: Final amount: $13,694.52
```

**Key Insight**: Agents seamlessly combine decision-making framework with computational tools, providing both correct workflow compliance and accurate numerical results.

## Migration from Old Approach

**Old (Complex)**:
- Complex multi-step workflows requiring multiple rounds
- Complex system messages with collaboration context
- Vague guidance: "Use tools to update, check, and vote"
- Agent status tracking in system messages

**New (Simplified)**:
- Binary decision framework (vote OR new_answer)
- Simple evaluation-focused system message
- Clear binary decision: "Does best answer address the question?"
- Enforcement reminders for workflow compliance
- Clean separation of concerns