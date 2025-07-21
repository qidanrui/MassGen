# MASS Input Cases Reference

This document provides a quick reference for all possible input cases to the LLM backend using the **simplified approach** proven to work in testing.

## ⚠️ CRITICAL IMPLEMENTATION WARNING

**Tool naming is crucial for success.** Use `update_answer` NOT `update_summary`:

- ✅ **`update_answer`**: "Do additional work and create an improved answer" → Works perfectly
- ❌ **`update_summary`**: "Create/update working summary" → Causes voting bias

This difference results in 5/5 vs 3/5 test success rates. See [Testing Validation](#critical-discovery-tool-naming-matters) for details.

## New Simplified Approach

Based on extensive testing, the MASS system now uses a simplified message structure that eliminates perfectionism loops and produces reliable tool usage decisions.

**Key Principle**: Each agent starts with a **single user message** containing: task + current answers + decision prompt. Additional enforcement messages may follow if needed.

## Core Components

### System Message Structure
```
[USER_CUSTOM_SYSTEM] + "You are evaluating answers for final presentation."
```

- **USER_CUSTOM_SYSTEM**: Optional custom system message from user configuration
- **Fixed evaluation context**: Always added to establish the agent's role

### User Message Structure  
```
{task}

Current answers:
{agent_summaries}

Does the best available answer address the question?

If YES, tell me the agent_id with the best available answer using the `vote` tool. Don't call the `update_answer` tool.
IF NO, do additional work first. Then give a better answer using the `update_answer` tool.
```

### Tool Definitions
- **update_answer(content)**: Do additional work and create an improved answer
- **vote(agent_id)**: Vote for best agent to present final answer

## Input Cases

### Case 1: Initial Work (No Summaries Exist)

**When**: Agent has no summary yet, no other agents have summaries

**System Message**:
```
[USER_CUSTOM_SYSTEM] + "You are evaluating answers for final presentation."
```

**User Message**:
```
{task}

Current answers:
(no answers available yet)

Does the best available answer address the question?

If YES, tell me the agent_id with the best available answer using the `vote` tool. Don't call the `update_answer` tool.
IF NO, do additional work first. Then give a better answer using the `update_answer` tool.
```

**Expected Behavior**: Agent calls `update_answer` (no adequate answers exist)

---

### Case 2: Decision Making (Summaries Exist)

**When**: Agent has summary, other agents may have summaries  

**System Message**:
```
[USER_CUSTOM_SYSTEM] + "You are evaluating answers for final presentation."
```

**User Message**:
```
{task}

Current answers:
- Agent A: {summary_A}
- Agent B: {summary_B}  
- Agent C: {summary_C}

Does the best available answer address the question?

If YES, tell me the agent_id with the best available answer using the `vote` tool. Don't call the `update_answer` tool.
IF NO, do additional work first. Then give a better answer using the `update_answer` tool.
```

**Expected Behavior**: 
- Agent calls `vote(agent_id)` if a good answer exists
- Agent calls `update_answer(content)` if no adequate answer exists

---

### Case 3: Non-Workflow Response Handling

**When**: Agent calls other tools or provides text response without using `update_answer`/`vote`

**System Message**:
```
[USER_CUSTOM_SYSTEM] + "You are evaluating answers for final presentation."
```

**Conversation Flow**:
```
User: {task with current answers and decision prompt}
Assistant: {may provide reasoning/analysis or calls other tools without update_answer/vote}
Tool Response: {result if tool call was made}
User: "Please use either the `vote` tool to select the best agent, or the `update_answer` tool to provide a better solution."
Assistant: {may continue with non-workflow responses}
User: "Please use either the `vote` tool to select the best agent, or the `update_answer` tool to provide a better solution."
...
```

**Expected Behavior**: Agent redirected to use proper workflow tools

---

### Case 4: Error Recovery  

**When**: Previous tool call failed (error sent via tool response)

**System Message**:
```
[USER_CUSTOM_SYSTEM] + "You are evaluating answers for final presentation."
```

**Conversation Flow**:
```
User: {task with current answers and decision prompt}
Assistant: {may provide reasoning/analysis}
User: {may provide clarification or additional context}
Assistant: {may continue analysis}
Assistant: {calls tool - e.g., vote(agent_id: "invalid_agent")}
Tool Response: Error: {specific error message - e.g., "Invalid agent_id 'invalid_agent'. Valid agents: Agent A, Agent B, Agent C"}
```

**Expected Behavior**: Agent retries with corrected approach

## Decision Logic

The agent evaluates available answers and makes a binary decision:

1. **Good answer exists** → `vote(agent_id)` for best agent
2. **No good answer exists** → `update_answer(content)` with better answer

This eliminates the perfectionism loop by:
- ✅ **Clear binary decision**: Either vote or update, not both
- ✅ **Task-focused evaluation**: "Does the best answer address the question?"
- ✅ **No self-reference confusion**: Agent evaluates others' answers objectively
- ✅ **Direct tool guidance**: Explicit instruction on which tool to use when

## Tool Usage Patterns

### Successful Patterns
- **update_answer**: Agent provides improved/complete answer
- **vote**: Agent selects existing adequate answer

### Problematic Patterns (Avoided)
- **Both tools**: Agent calls vote AND update_answer (perfectionism)
- **No tools**: Agent provides text response without tool usage
- **Wrong tool**: Agent votes when answer is inadequate, or updates when answer is adequate

## Implementation Notes

1. **Message Construction**: Single user message combines all context
2. **System Message**: Combine user custom + evaluation role
3. **Agent Summaries**: Present all available summaries as "Current answers"
4. **Decision Prompt**: Always use exact wording proven to work in tests
5. **Tool Enforcement**: Clear instruction on which tool to use in each scenario

## Testing Validation

This approach has been validated across multiple test scenarios:
- ✅ Multiple correct answers → votes correctly
- ✅ All inadequate answers → updates correctly  
- ✅ Mix of good/bad answers → votes for best
- ✅ Wrong vs right answers → identifies correct answer
- ✅ Non-math questions → works for reasoning tasks
- ✅ Computational problems → uses code execution when needed

### Critical Discovery: Tool Naming Matters

**IMPORTANT**: Tool naming significantly affects agent decision-making:

✅ **Working Tool Definition:**
```json
{
  "name": "update_answer",
  "description": "Do additional work and create an improved answer."
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
- `update_answer` approach: 5/5 test cases passed
- `update_summary` approach: 3/5 test cases passed (failed on inadequate answer detection)

### Code Execution Integration

**EXCELLENT**: The simplified approach works seamlessly with OpenAI's built-in code execution:

✅ **Workflow Compliance**: Agents correctly call `update_answer` when computational work is needed  
✅ **Automatic Code Execution**: When enabled with `enable_code_interpreter=True`, agents naturally use Python execution  
✅ **Complete Solutions**: Agents provide calculated numerical results, not just formulas  
✅ **Transparent Process**: Code execution is visible in the streaming output  

**Example Behavior:**
```
💻 [Provider Tool: Code Interpreter] Starting execution...
💻 [Code Executed] P = 10000; r = 0.045; n = 12; t = 7...
✅ [Provider Tool: Code Interpreter] Execution completed

🔧 Tool called: ['update_answer']
📝 Agent's Answer: Final amount: $13,694.52
```

**Key Insight**: Agents seamlessly combine decision-making framework with computational tools, providing both correct workflow compliance and accurate numerical results.

## Migration from Old Approach

**Old (Complex)**:
- Multi-turn conversations
- Complex system messages with collaboration context
- Vague guidance: "Use tools to update, check, and vote"
- Agent status tracking in system messages

**New (Simplified)**:
- Initial single user message with clear decision prompt
- Simple evaluation-focused system message
- Clear binary decision: "Does best answer address the question?"
- Enforcement reminders for workflow compliance
- Clean separation of concerns