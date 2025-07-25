# Contributing to MassGen

Thank you for your interest in contributing to MassGen (Multi-Agent Scaling System)! We welcome contributions from the community and are excited to see what you'll bring to the project.

## 🛠️ Development Guidelines

### Project Structure

```
massgen/
├── __init__.py          # Main package exports
├── agent.py             # Abstract base agent class
├── agents.py            # Concrete agent implementations
├── orchestrator.py      # Multi-agent coordination
├── main.py              # Programmatic interfaces
├── config.py            # Configuration management
├── types.py             # Type definitions
├── tools.py             # Custom tools for agent use
├── utils.py             # Helper functions
├── logging.py           # Logging system
├── streaming_display.py # Real-time display
└── backends/           # Model-specific implementations
    ├── oai.py          # OpenAI backend
    ├── gemini.py       # Google Gemini backend
    └── grok.py         # xAI Grok backend
```

### Adding New Model Backends

To add support for a new model provider:

1. Create a new file in `massgen/backends/` (e.g., `claude.py`)
2. Implement the `process_message` and `parse_completion` function with the required signature
3. Add the model mapping in `massgen/utils.py`
4. Update the agent creation logic in `massgen/agents.py` if it is unique
5. Add tests and documentation

To add more tools for agents:

1. Create or extend tool definitions in `massgen/tools.py`
2. Register your custom tool with the appropriate model backends
3. Ensure compatibility with the tool calling interface of each model
4. Test tool functionality across different agent configurations
5. Consider adding MCP Server integrations for broader tool ecosystems
6. Update documentation with tool capabilities and usage examples

Current built-in tool support by model:
- **Gemini**: Live Search ✅, Code Execution ✅
- **OpenAI**: Live Search ✅, Code Execution ✅  
- **Grok**: Live Search ✅, Code Execution ❌

Current custom tool support (`massgen/tools.py`):
- **calculator**
- **python interpretor**

### Contributing Areas

We welcome contributions in these areas:

- **New Model Backends**: Add support for additional AI models
- **Tools and Integrations**: Extend the tool system with new capabilities
- **Performance Improvements**: Optimize coordination, communication, etc
- **Documentation**: Add guides, examples, use cases, and API documentation
- **Testing**: Add comprehensive test coverage
- **Bug Fixes**: Fix issues and edge cases


## 🤝 Community

Join the discussion on the #massgen channel of AG2 Discord server: https://discord.gg/VVrT2rQaz5


## 📄 License

By contributing, you agree that your contributions will be licensed under the same Apache License 2.0 that covers the project.

---

Thank you for contributing to MassGen! 🚀