# RLLM Trajectory Termination Reasons

## Overview
During training, trajectories can terminate for various reasons. Each termination reason indicates a different bottleneck or completion state.

## Termination Reasons

### TRUNCATION
- **Meaning**: Response tokens exceeded `max_response_length`
- **Location**: `external/rllm/rllm/engine/agent_execution_engine.py:377`
- **Typical cause**: Too much text generated, token budget exhausted

### MAX_STEPS  
- **Meaning**: Agent reached maximum number of environment steps (`max_steps`)
- **Location**: `external/rllm/rllm/engine/agent_execution_engine.py:402`
- **Typical cause**: Task too complex for step limit or agent inefficient

### ENV_DONE
- **Meaning**: Environment returned `done=True` (task successfully completed)
- **Location**: `external/rllm/rllm/engine/agent_execution_engine.py:395`
- **Typical cause**: Normal completion - this is the desired outcome

### TIMEOUT
- **Meaning**: Total execution time exceeded `trajectory_timeout`
- **Location**: `external/rllm/rllm/engine/agent_execution_engine.py:387`
- **Typical cause**: Slow environment or agent taking too long

### ENV_TIMEOUT
- **Meaning**: Single environment step timed out
- **Location**: `external/rllm/rllm/engine/agent_execution_engine.py:313`
- **Typical cause**: Environment hanging or extremely slow operation

### PROMPT_TRUNCATION
- **Meaning**: Single prompt exceeded `max_prompt_length` (when `enforce_max_prompt_length=True`)
- **Location**: `external/rllm/rllm/engine/agent_execution_engine.py:286`
- **Typical cause**: Context grew too large

## Configuration Parameters
- `max_steps`: Maximum environment steps allowed
- `max_response_length`: Maximum tokens for responses
- `max_prompt_length`: Maximum tokens for prompts
- `trajectory_timeout`: Maximum time for entire trajectory

## Log Output
All termination reasons are logged at `external/rllm/rllm/engine/agent_execution_engine.py:424-427` with color coding:
- Green: Positive reward (task completed)
- Yellow: Zero reward (task not completed)