# LLM Judge Backend System

This system allows dynamic switching between different LLM backends for judge evaluation without restarting your process.
**Currently only works for single node GPU**

## Available Backends

1. **Claude Code CLI Backend** (`ccode-4-sonnet`) - Uses the Claude Code CLI directly
2. **LiteLLM Backend** - Uses LiteLLM API with any supported model (e.g., `anthropic/claude-3-opus-20240229`)

## Default Configuration

Set environment variables before launching your run:

```bash
export CLAUDE_CLI_PATH="/path/to/claude"  # Path to Claude Code CLI executable
export LLM_JUDGE_NAME="ccode-4-sonnet"    # Default judge model
```

With these settings, the system will use Claude Code CLI by default.

## Dynamic Backend Switching

You can switch backends at runtime using the `.judge_backend` file:

### Switch to LiteLLM (e.g., when hitting Claude Code token limits)
```bash
python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229
```

### Switch back to Claude Code CLI
```bash
python switch_judge_backend.py ccode
```

### Check current backend
```bash
python switch_judge_backend.py show
```

### Remove override (revert to environment variable)
```bash
python switch_judge_backend.py remove
```

## How It Works

1. **Priority**: The `.judge_backend` file takes precedence over the `LLM_JUDGE_NAME` environment variable
2. **File Format**:
   - For Claude Code CLI: `ccode-4-sonnet`
   - For LiteLLM: `litellm:model_name` (e.g., `litellm:anthropic/claude-3-opus-20240229`)
3. **Caching**: The file is checked every 5 seconds to minimize file I/O
4. **Logging**: Backend switches are logged at INFO level

## Example Workflow

```bash
# Start with Claude Code CLI
export CLAUDE_CLI_PATH="/usr/local/bin/claude"
export LLM_JUDGE_NAME="ccode-4-sonnet"
python your_training_script.py &

# Monitor and switch when needed
python switch_judge_backend.py show  # Shows: using environment variable

# Hit token limits? Switch to LiteLLM
python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229

# Later, switch back
python switch_judge_backend.py ccode

# Done with overrides
python switch_judge_backend.py remove
```

The system will seamlessly switch backends without interrupting your running process.