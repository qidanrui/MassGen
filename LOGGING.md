# MASS Logging System

## Overview

The MASS system now includes comprehensive logging functionality that saves all agent outputs and system messages to separate text files organized in timestamped session folders within the `logs/` directory. This allows you to preserve all information from the multi-agent collaboration sessions without any truncation, with each session neatly organized in its own folder.

**✨ New Feature**: The console display now shows **clickable file links** for each agent and system log, making it easy to access the full logs directly from the running session.

## Features

- **Session-based Organization**: Each MASS session creates its own timestamped folder
- **Separate Agent Logs**: Each agent gets its own log file (`agent_0.txt`, `agent_1.txt`, etc.)
- **System Message Log**: All system messages are saved to a separate file (`system.txt`)
- **Enhanced Console Display**: 
  - **Model Names in Headers**: Shows which model each agent is using (e.g., "Agent 0 (o4-mini)")
  - **Underlined File Links**: File paths are underlined and clickable for instant access
- **Real-time Logging**: Content is written to log files immediately as it's generated
- **No Truncation**: Unlike the console display which limits lines for readability, log files contain all content without any truncation
- **Automatic Directory Creation**: The `logs/` directory and session subdirectories are created automatically
- **Clean File Organization**: Simple, predictable filenames within timestamped session folders

## Console Display with File Links

The MASS console now displays clickable file links that allow you to quickly access the complete logs:

```
───────────────────────────────────────────────────────────────────────────────────────────────────────────
       🤖 Agent 0 (o4-mini)        │   🤖 Agent 1 (gemini-2.5-flash)   │        🤖 Agent 2 (grok-4)        
 📁 logs/20250718_201750/agent_0.txt │ 📁 logs/20250718_201750/agent_1.txt │ 📁 logs/20250718_201750/agent_2.txt
───────────────────────────────────────────────────────────────────────────────────────────────────────────
 🚀 Starting initial phase...      │ 🚀 Starting initial phase...      │ All peer solutions are in unanim… 
 🔄 Processing task (timeout: 150… │ 🔄 Processing task (timeout: 150… │ ## Detailed Reasoning and Verifi… 
 ...                              │ ...                              │ ...

───────────────────────────────────────────────────────────────────────────────────────────────────────────
 📋 SYSTEM MESSAGES

 📁 logs/20250718_201750/system.txt

───────────────────────────────────────────────────────────────────────────────────────────────────────────
 🗳️  Agent 0 VOTED for Agent 1
 🗳️  Agent 1 VOTED for Agent 2
 ...
```

### Enhanced Display Features

- **Model Names in Headers**: Agent headers now show the specific model being used (e.g., "Agent 0 (o4-mini)")
- **Underlined File Links**: File paths are underlined to clearly indicate they are clickable links
- **Smart Layout**: Model names automatically adjust to column width while maintaining readability

### How to Use File Links

- **VS Code**: `Ctrl+Click` (or `Cmd+Click` on Mac) on any underlined file path to open the log file
- **Most Modern Terminals**: Click on underlined paths to open with your default text editor
- **Terminal Emulators**: Many support clickable links (iTerm2, Windows Terminal, etc.)
- **IDEs**: Most development environments automatically make file paths clickable

The file links provide instant access to:
- **Complete agent reasoning**: Full output without console truncation
- **Model-specific analysis**: See how different models approach the same problem
- **All system messages**: Complete voting and phase transition history
- **Raw text format**: Easy to copy, search, or process further

## File Structure

```
logs/
├── 20250718_195850/              # Session timestamp folder
│   ├── agent_0.txt               # Agent 0 complete output
│   ├── agent_1.txt               # Agent 1 complete output  
│   ├── agent_2.txt               # Agent 2 complete output
│   └── system.txt                # System messages
├── 20250718_201230/              # Another session
│   ├── agent_0.txt
│   ├── agent_1.txt
│   ├── agent_2.txt
│   └── system.txt
└── 20250718_203445/              # Yet another session
    ├── agent_0.txt
    ├── agent_1.txt
    └── system.txt
```

## Log File Format

### Agent Log Files
```
MASS Agent 0 Output Log
Session started: 2025-07-18 19:58:50
================================================================================

🚀 Starting initial phase...
🔄 Processing task (timeout: 150s)...
🧠 Reasoning in progress...
The sum is 1 + 1233493139429394 = 1233493139429395.
```

### System Log Files
```
MASS System Messages Log
Session started: 2025-07-18 19:58:50
================================================================================

[19:58:50] 🗳️  Agent 0 VOTED for Agent 1
[19:58:50] 🗳️  Agent 1 VOTED for Agent 2
[19:58:50] 🎯 Consensus reached! Winner: Agent 2 (Agent 1: 1, Agent 2: 2)
[19:58:50] Phase: completed → debate
```

## Configuration

### Enabling/Disabling Logging

By default, logging is **enabled**. You can control it in several ways:

#### 1. Via MassSystem Configuration
```python
# Disable logging for the entire system
config = {"save_logs": False}
mass_system = MassSystem(config=config)
```

#### 2. Via Command Line (if using mass_main.py directly)
The system respects the `save_logs` configuration parameter.

#### 3. Via StreamingOrchestrator Directly
```python
# Create orchestrator with logging disabled
orchestrator = StreamingOrchestrator(
    display_enabled=True,
    save_logs=False  # Disable logging
)
```

## Technical Implementation

The logging functionality is implemented in the following files:

- **`mass_streaming_display.py`**: Core logging implementation in `MultiRegionDisplay` class
- **`mass_workflow.py`**: Integration with workflow manager
- **`mass_main.py`**: Configuration parameter handling

### Key Components

1. **MultiRegionDisplay Class**: 
   - Added `save_logs` parameter to constructor
   - `_setup_logging()`: Creates logs directory and timestamped session folder
   - `_write_agent_log()`: Writes agent content to log files
   - `_write_system_log()`: Writes system messages to log file

2. **StreamingOrchestrator Class**:
   - Passes `save_logs` parameter to `MultiRegionDisplay`
   - Maintains all existing functionality while adding logging

3. **Integration Points**:
   - `stream_agent_output()`: Logs agent content in real-time
   - `add_system_message()`: Logs system messages with timestamps
   - `finalize_agent_message()`: Ensures proper line breaks in logs

## Benefits

1. **Complete Record**: All agent reasoning, outputs, and system messages are preserved
2. **Session Organization**: Each session is neatly contained in its own timestamped folder
3. **Easy Analysis**: Review detailed agent interactions and decision-making processes
4. **Simple File Management**: Predictable filenames make it easy to find specific agent logs
5. **Debugging**: Troubleshoot issues by examining complete logs
6. **Archival**: Keep permanent records of MASS sessions organized by time
7. **No Information Loss**: Unlike console display with line limits, logs contain everything

## Usage Tips

- Log files are automatically created in timestamped folders within `logs/`
- Each session creates a new folder with format `YYYYMMDD_HHMMSS`
- Files use UTF-8 encoding to support emojis and unicode characters
- System messages include timestamps for easy chronological tracking
- Agent files preserve all formatting and special characters from the original output
- To find the most recent session, look for the latest timestamped folder

## File Cleanup

Log folders accumulate over time. You may want to periodically clean old session folders:

```bash
# Remove session folders older than 7 days
find logs/ -type d -name "????????_??????" -mtime +7 -exec rm -rf {} \;

# Remove all sessions except the last 10
cd logs && ls -dt ????????_?????? | tail -n +11 | xargs rm -rf
```

## Session Analysis

To analyze a specific session:

```bash
# View all files in a session
ls -la logs/20250718_195850/

# View agent 0's complete output
cat logs/20250718_195850/agent_0.txt

# View system messages
cat logs/20250718_195850/system.txt

# Compare outputs from different agents
diff logs/20250718_195850/agent_0.txt logs/20250718_195850/agent_1.txt
```

The logging system is designed to be lightweight and non-intrusive while providing comprehensive coverage of all MASS system activities organized in an intuitive, time-based folder structure. 