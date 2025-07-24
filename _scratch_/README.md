# MASS Framework Implementation

A robust implementation of the MASS (Multi-Agent Scaling System) framework with sophisticated orchestration and conversation management.

## 📁 Directory Structure

```
mass/
├── mass/
│   ├── backends/                     # LLM backend implementations
│   │   ├── base.py                   # Abstract backend interface with StreamChunk
│   │   ├── openai_backend.py         # OpenAI Response API implementation
│   │   ├── chat_completions_backend.py # Chat Completions API base class
│   │   └── grok_backend.py           # Grok/xAI implementation
│   ├── frontend/                     # Frontend display components
│   │   ├── coordination_ui.py        # Main coordination UI interface
│   │   ├── displays/                 # Display implementations
│   │   │   ├── base_display.py       # Abstract display interface
│   │   │   ├── simple_display.py     # Basic text display
│   │   │   └── terminal_display.py   # Rich terminal display
│   │   └── logging/                  # Logging utilities
│   │       └── realtime_logger.py    # Real-time coordination logging
│   ├── chat_agent.py                 # Agent implementations with conversation handling
│   ├── orchestrator.py               # MASS orchestrator with workflow coordination
│   ├── message_templates.py          # Message building and templates
│   └── agent_config.py               # Agent configuration management
├── examples/                         # Usage examples and test cases
├── tests/                           # Test files and validation
├── input_cases_reference.md          # Complete MASS framework documentation
└── README.md                         # This file
```

## 🎯 Core Philosophy

This implementation focuses on the **proven binary decision framework** that eliminates perfectionism loops:

1. **Vote** for best existing answer (if adequate)
2. **Create new answer** with additional work (if inadequate)

## ✨ Key Features

- **Sophisticated Orchestration**: Full workflow coordination with case handling (Cases 1-4)
- **Enhanced Tool Visibility**: Tool calls display actual content and arguments for full transparency
- **Duplicate Answer Prevention**: Agents cannot provide identical answers, improving efficiency
- **Multi-Agent Coordination**: Seamless coordination between multiple specialized agents
- **Rich Frontend Components**: Terminal UI, simple display, and real-time logging
- **Flexible Conversation Management**: `reset_chat` and `clear_history` parameters for precise control
- **Accurate Message Preservation**: `complete_message` chunks maintain exact backend message structure
- **Enhanced Error Recovery**: Comprehensive Case 4 handling with helpful feedback
- **Real-time Streaming**: Full streaming support with proper tool call processing
- **Multiple Backend Support**: OpenAI Response API, Chat Completions API, and Grok/xAI
- **Demo-Ready Examples**: Mock backends for cost-free demonstration and testing
- **Advanced Restart Logic**: Proper handling of vote invalidation and agent restarts 

## 🏗️ Architecture

### Backend Structure

```
LLMBackend (abstract)
    ↓
├── OpenAIBackend (Response API - standalone)
└── ChatCompletionsBackend (Chat Completions base class)
        ↓
        └── GrokBackend (xAI implementation)
```

### Key Architectural Decisions

1. **OpenAI Response API**: Standalone backend optimized for OpenAI's Response API format (`input` parameter)
2. **Chat Completions Base**: Shared streaming logic for backends using Chat Completions API (`messages` parameter)
3. **Grok-Specific Features**: Only Grok-specific logic (API key, base URL, Live Search, pricing)

## 🔧 Key Features

- **API Format Separation**: Response API vs Chat Completions properly separated
- **Tool Format Conversion**: Automatic conversion between API formats
- **Shared Streaming Logic**: Chat Completions streaming handled in base class
- **Built-in Tools Integration**: Web search and code execution support
- **Clean Live Search**: Proper `search_parameters` implementation (no content detection)

## 📚 Documentation

See `input_cases_reference.md` for:
- Complete input case documentation (Cases 1-4)
- Tool naming impact studies
- Testing validation across domains
- Implementation guidelines
- Migration from complex approaches

## 🎯 Key Insights

1. **Tool Naming Matters**: `new_answer` vs `update_summary` significantly affects behavior
2. **Binary Decisions Work**: Eliminates perfectionism loops effectively  
3. **API Separation**: Response API vs Chat Completions require different approaches
4. **Domain Agnostic**: Works across research, computation, and analysis tasks

## 🚀 Usage

### OpenAI with Response API
```python
backend = OpenAIBackend()
await backend.stream_with_tools(
    model="gpt-4o",
    messages=messages,  # Converted to 'input' parameter internally
    tools=tools,
    enable_web_search=True,
    enable_code_interpreter=True
)
```

### Grok with Chat Completions API
```python
backend = GrokBackend()  
await backend.stream_with_tools(
    model="grok-3-mini",
    messages=messages,  # Passed as 'messages' parameter
    tools=tools,
    enable_web_search=True  # Uses proper search_parameters
)
```

## ✨ Grok Backend Improvements

The rewritten Grok backend now contains **only Grok-specific features**:

- ✅ **XAI_API_KEY** environment variable
- ✅ **xAI base URL** (`https://api.x.ai/v1`)
- ✅ **Live Search** with proper `search_parameters` (like xAI native backend)
- ✅ **Grok pricing** calculation
- ✅ **Minimal footprint** - all generic Chat Completions logic moved to base class
- ❌ **No content-based search detection** (removed)
- ❌ **No message modification fallbacks** (removed)

## ✅ Validation

Proven across **7 clean test files** covering:
- Core input cases (1-4)
- Econometrics research domain
- Tower of Hanoi computational problems
- Error recovery and edge cases

The framework provides reliable multi-agent decision-making with clean API separation and minimal backend-specific code.