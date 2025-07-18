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
# Run with three different high-performance models
python3 mass_main.py --task-file hle/test_case.json --agents gpt-4o,gemini-2.5-flash,grok-4

# Use multiple variants of the same model family
python3 mass_main.py --task-file task.json --agents o3-medium,o3-high,gpt-4o

# Mix different model capabilities
python3 mass_main.py --task-file task.json --agents o1,gemini-2.5-pro,grok-4
```

### Advanced Configuration
```bash
# Use custom configuration file
python3 mass_main.py \
  --task-file task.json \
  --agents gpt-4o,gemini-2.5-flash,o3-mini \
  --config example_config.json \
  --output results.json

# Sequential execution instead of parallel
python3 mass_main.py \
  --task-file task.json \
  --agents o3-high,gemini-2.5-pro \
  --sequential

# Custom consensus threshold (majority vote)
python3 mass_main.py \
  --task-file task.json \
  --agents gpt-4o,o3-medium,gemini-2.5-flash,grok-4 \
  --consensus-threshold 0.6
```

### Performance Optimization Examples
```bash
# High-speed configuration (fast models)
python3 mass_main.py \
  --task-file task.json \
  --agents gpt-4o-mini,gemini-2.5-flash,o3-mini-low

# High-quality configuration (best models)
python3 mass_main.py \
  --task-file task.json \
  --agents o3-high,gemini-2.5-pro,gpt-4o

# Balanced configuration (mix of speed and quality)
python3 mass_main.py \
  --task-file task.json \
  --agents gpt-4o,gemini-2.5-flash,o3-medium
```

## Programmatic Usage

```python
from mass_main import MassSystem

# Create system instance
system = MassSystem()

# Run with specific models
result = system.run_task_from_file(
    "hle/test_case.json", 
    ["gpt-4o", "gemini-2.5-flash", "o3-mini"]
)

# Run with custom configuration
config = {
    "max_rounds": 3,
    "consensus_threshold": 0.8,
    "parallel_execution": True,
    "check_update_frequency": 2.0
}

system = MassSystem(config=config)
result = system.run_task_from_file(
    "task.json",
    ["o3-high", "gemini-2.5-pro", "grok-4"]
)
```

## Configuration File Format

Create a JSON configuration file (like `example_config.json`):

```json
{
  "max_rounds": 5,
  "consensus_threshold": 1.0,
  "parallel_execution": true,
  "check_update_frequency": 3.0
}
```

### Configuration Parameters

- **max_rounds**: Maximum collaboration rounds before fallback to majority vote (default: 5)
- **consensus_threshold**: Required agreement fraction (1.0 = unanimous, 0.5 = majority, default: 1.0)
- **parallel_execution**: Whether agents work simultaneously (default: true)
- **check_update_frequency**: Update check interval in seconds (default: 3.0)

## Model Selection Strategy

### For Speed-Critical Tasks
- Use `gpt-4o-mini`, `gemini-2.5-flash`, `o3-mini-low`
- Lower consensus threshold for faster decisions

### For Quality-Critical Tasks  
- Use `o3-high`, `gemini-2.5-pro`, `gpt-4o`
- Keep consensus threshold at 1.0 for unanimous agreement

### For Balanced Performance
- Mix fast and quality models: `gpt-4o`, `gemini-2.5-flash`, `o3-medium`
- Use moderate consensus threshold: 0.7-0.8

### For Maximum Diversity
- Use one model from each provider: `gpt-4o`, `gemini-2.5-pro`, `grok-4`
- Ensures different approaches and perspectives

## Error Handling

If you specify an invalid model name, the system will show all available options:

```bash
$ python3 mass_main.py --task-file task.json --agents invalid-model
ERROR: Invalid model name: invalid-model
Available models: gpt-4o, gpt-4o-2024-11-20, gpt-4o-mini, o1, o3, o3-low, ...
```

## Migration from Agent Types

**Old way (agent types):**
```bash
python3 mass_main.py --task-file task.json --agents openai,gemini,grok
```

**New way (specific models):**
```bash
python3 mass_main.py --task-file task.json --agents gpt-4o,gemini-2.5-flash,grok-4
```

The new approach gives you precise control over which models are used, allowing for better optimization based on your specific needs. 