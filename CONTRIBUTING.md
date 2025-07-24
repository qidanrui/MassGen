# Contributing to MASS

Thank you for your interest in contributing to MASS (Multi-Agent Scaling System)! We welcome contributions from the community and are excited to see what you'll bring to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- API keys for at least one supported model provider (OpenAI, Google Gemini, or xAI Grok)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/MassAgent.git
   cd MassAgent
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**
   ```bash
   cp mass/backends/.env.example mass/backends/.env
   # Edit .env with your API keys
   ```

5. **Test Installation**
   ```bash
   python cli.py --models gpt-4.1 "Hello, world!"
   ```

## 🛠️ Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Project Structure

```
mass/
├── __init__.py          # Main package exports
├── agent.py             # Abstract base agent class
├── agents.py            # Concrete agent implementations
├── orchestrator.py      # Multi-agent coordination
├── main.py              # Programmatic interfaces
├── config.py            # Configuration management
├── types.py             # Type definitions
├── tools.py             # Built-in tools and utilities
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

1. Create a new file in `mass/backends/` (e.g., `claude.py`)
2. Implement the `process_message` function with the required signature
3. Add the model mapping in `mass/utils.py` in `get_agent_type_from_model`
4. Update the agent creation logic in `mass/agents.py`
5. Add tests and documentation

### Contributing Areas

We welcome contributions in these areas:

- **New Model Backends**: Add support for additional AI models
- **Tools and Integrations**: Extend the tool system with new capabilities
- **Performance Improvements**: Optimize coordination and communication
- **Documentation**: Improve guides, examples, and API documentation
- **Testing**: Add comprehensive test coverage
- **Bug Fixes**: Fix issues and edge cases

## 📝 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Test with different models
   python cli.py --models gpt-4.1 "Test question"
   python cli.py --models gemini-2.5-flash "Test question"
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add descriptive commit message"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commits format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

## 🐛 Reporting Issues

When reporting bugs, please include:

- Python version and operating system
- MASS version or commit hash
- Complete error messages and stack traces
- Steps to reproduce the issue
- Expected vs. actual behavior
- Configuration files (with API keys removed)

## 💡 Feature Requests

We welcome feature requests! Please provide:

- Clear description of the proposed feature
- Use cases and motivation
- Implementation suggestions (if any)
- Willingness to contribute the implementation

## 📚 Development Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 8 Style Guide](https://pep8.org/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Google Gemini API](https://ai.google.dev/)
- [xAI Grok API](https://x.ai/docs/)

## 🤝 Community

- Join discussions in GitHub Issues
- Follow the project for updates
- Share your use cases and examples

## 📄 License

By contributing, you agree that your contributions will be licensed under the same Apache License 2.0 that covers the project.

---

Thank you for contributing to MASS! 🚀