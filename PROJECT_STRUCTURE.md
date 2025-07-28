# Terminal-Bench-RL Project Structure

## Overview

I built Terminal-Bench-RL to showcase a sophisticated reinforcement learning training system designed to train AI agents for long horizon terminal-based tasks. It integrates the [RLLM](https://github.com/agentica-project/rllm) framework with [Terminal Bench](https://github.com/laude-institute/terminal-bench)'s Docker orchestration infrastructure to create isolated, reproducible training environments.

## Directory Structure

```
terminal-bench-rl/
├── src/                        # Core source code
│   ├── tbench_rllm/           # Terminal Bench + RLLM integration
│   └── agent_core/            # Core agent components
├── external/                   # External dependencies
│   └── rllm/                  # RLLM framework (git submodule)
├── training_scripts/           # Training orchestration scripts
├── evaluation/                 # Evaluation and benchmarking
│   ├── terminal_bench_eval/   # Terminal bench evaluation harness
│   └── llm_as_a_judge_evals/  # LLM judge evaluation system
├── dataset/                    # Dataset management
├── test_tasks/                 # Test task definitions
├── docs/                       # Documentation
├── helpful_agent_instructions/ # Local coding agent prompt helpers
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
├── CLAUDE.md                  # Project instructions
└── README.md                  # Project overview
```

## Core Components

### 1. **Source Code (`src/`)**

#### `src/tbench_rllm/` - Terminal Bench Integration
- **`terminal_agent.py`**: Main agent implementation extending RLLM's `BaseAgent`
  - Manages conversation history and system prompts
  - Tracks action trajectories for RL training
  - Handles state updates from environment and model responses
  
- **`docker_env.py`**: Docker environment implementation extending `BaseEnv`
  - Creates isolated Docker containers for each trajectory
  - Integrates with Terminal Bench's `TrialHandler`
  - Executes tool calls made by the LLM
  - Manages container lifecycle and cleanup
  - Thread-safe parallel execution support

- **`agent_env_dtos.py`**: Data transfer objects between agent & env
  - `StepObservation`: Standardized environment observations
  - `ActionPayload`: Agent responses and conversation history

#### `src/agent_core/` - Agent Core Components
- **`env_components/`**: Modular action handling system
  - `actions.py`: Action type definitions (Bash, File, Todo, etc.)
  - `action_parser.py`: XML/YAML parser for LLM responses
  - `action_handler.py`: Routes actions to appropriate executors
  - `step_executor.py`: Orchestrates execution with error handling
  - `command_executor.py`: Bash command execution in containers
  - `file_manager.py`: File operations (read, write, edit)
  - `search_manager.py`: Search tools (grep, glob, ls)
  - `state_managers.py`: Todo list and scratchpad management

- **`system_prompt.md`**: Comprehensive agent instructions
  - Task execution phases (Planning → Exploration → Execution → Verification)
  - Action syntax and formatting rules
  - Best practices for terminal operations

### 2. **Training Scripts (`training_scripts/`)**

- **`launch_training.py`**: Main training orchestrator
  - Preset configurations for different hardware setups
  - Manages Docker cleanup and model downloading
  - Handles checkpoint uploading to HuggingFace

- **`train_terminal_bench_grpo_rllm.sh`**: Core GRPO training script
  - Configures tensor/sequence parallelism
  - Sets up FSDP with offloading

- **Supporting Scripts**:
  - `patch_rllm_mappings.py`: Patches RLLM environment mappings
  - `download_model.py`: Downloads HuggingFace models
  - `upload_checkpoint.py`: Uploads trained checkpoints
  - `switch_judge_backend.py`: Switches LLM judge backends

### 3. **Evaluation System (`evaluation/`)**

#### `terminal_bench_eval/`
Used to run `agent_core` (system message + tools through terminal bench with any model)
- **`run_eval.sh`**: Launches evaluation harness
- **`terminal_agent.py`**: Agent implementation for evaluation
  - Uses LiteLLM for model inference
  - Implements structured action parsing
  - Manages agent state (todos, scratchpad)

#### `llm_as_a_judge_evals/`
- **`judge_eval.py`**: Framework for testing judge consistency
- **`config.yaml`**: Test cases with expected score ranges
- **`judge_eval_cases/`**: Specific test scenarios
- **Judge Prompt Versions**: v4.0, v4.1, v4.2 with evolution tracking

### 4. **Dataset (`dataset/`)**

- **`latest_verified.csv`**: Main dataset with 331 tasks
- **Task Structure**:
  ```
  - task_id: Unique identifier
  - difficulty: easy/medium/hard/extremely_hard
  - category: Software Engineering, System Admin, etc.
  - prompt: Task instruction
  - dockerfile: Environment setup
  - test_functions: Pytest verification
  - test_weights: Test importance scoring
  ```

- **Categories**: 12 categories including:
  - Software Engineering (97 tasks)
  - System Administration (59 tasks)
  - Security (42 tasks)
  - Data Processing (37 tasks)

### 5. **External Dependencies (`external/`)**

- **`rllm/`**: RLLM framework (git submodule)
  - Core abstractions: BaseAgent, BaseEnv, AgentTrainer
  - Distributed training support via VERL
  - Integration with vLLM for efficient inference
  - Supports PPO, GRPO, and other RL algorithms

### 6. **Documentation (`docs/`)**

- **Training Guides**:
  - Single-node GPU setup instructions
  - Multi-node distributed training configuration
  - Docker daemon tuning for parallel containers
  - Network configuration (NCCL, Gloo)

- **LLM Judge Documentation**:
  - Dynamic backend switching (Claude Code CLI ↔ LiteLLM)
  - Runtime configuration without training interruption
  - Token limit handling and performance optimization

- **RLLM Configuration Guide**:
  - Environment settings (timeouts, container management)
  - Model settings (parallelism, batch sizes)
  - Training settings (rejection sampling, validation)
  - Optimizer settings (gradient checkpointing, offloading)

## Key Architecture Patterns

### 1. **Action-Based Architecture**
- Structured XML/YAML action format
- Typed action objects with dedicated handlers
- Sequential execution with error handling
- Consistent result formatting

### 2. **Dual Reward System**
- **Test Score (65%)**: Actual test execution results
- **Judge Score (35%)**: LLM evaluation of agent behavior
- Combined scoring for comprehensive training signal

### 3. **Isolation and Safety**
- Each trajectory in isolated Docker container
- Automatic cleanup and resource management
- Safety checks (e.g., read before edit)
- Command timeouts and error boundaries

### 4. **Distributed Training Support**
- Tensor parallelism and sequence parallelism
- FSDP with CPU offloading
- Asynchronous trajectory generation
- Multi-node scaling capabilities

## Configuration and Setup

### Package Management
- **Package Manager**: UV (modern Python package manager)
- **Python Version**: 3.12.9
- **Key Dependencies**:
  - terminal-bench (forked for Python 3.12 compatibility)
  - litellm, transformers, pydantic
  - RLLM framework via git submodule

### Training Presets
- **Test Run**: 8B model on 2 GPUs
- **Runway**: 32B model on 4 GPUs  
- **Production**: 32B model on 8 GPUs
- **Multi-node**: Distributed across multiple nodes

## Workflow Integration

1. **Dataset Preparation**: Convert CSV tasks to Terminal Bench format
2. **Environment Setup**: Configure Docker and network settings
3. **Training Launch**: Use preset configurations via launch_training.py
4. **Monitoring**: Track metrics via SQLite database and W&B
5. **Evaluation**: Run terminal_bench_eval for performance assessment
6. **Checkpoint Management**: Auto-upload to HuggingFace Hub