# MASS Usage Examples - Model-Based Interface

The MASS system now accepts specific model names instead of generic agent types, giving you fine-grained control over which models to use.

## Available Models

### OpenAI Models
- **GPT-4 variants**: `gpt-4o`, `gpt-4o-2024-11-20`, `gpt-4o-mini`
- **o1 series**: `o1`
- **o3 series**: `o3`, `o3-low`, `o3-medium`, `o3-high`
- **o3-mini series**: `o3-mini`, `o3-mini-low`, `o3-mini-medium`, `o3-mini-high`
- **o4-mini series**: `o4-mini`, `o4-mini-low`, `o4-mini-medium`, `o4-mini-high`
- **GPT-4.1 series**: `gpt-4.1-mini`, `gpt-4.1`

### Gemini Models
- `gemini-2.5-flash`
- `gemini-2.5-pro`

### Grok Models
- `grok-4`

## Command Line Examples

### Basic Usage
```bash
# Mix different model capabilities
python mass_main.py --task-file test_case2.json --agents o4-mini,gemini-2.5-flash,grok-4
```