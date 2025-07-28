#!/usr/bin/env python3
"""Launch Terminal Bench RLLM training with preset configurations"""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Optional
import atexit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tbench_rllm.docker_cleanup import start_docker_cleanup, stop_docker_cleanup

@dataclass
class TrainingConfig:
    """Training configuration preset"""
    name: str
    description: str
    env_vars: Dict[str, str]
    
    def apply(self):
        """Apply environment variables"""
        for key, value in self.env_vars.items():
            os.environ[key] = str(value)
        
        # Set Python logging level via env var (simplest approach)
        os.environ['PYTHONLOG'] = os.environ.get('LOG_LEVEL', 'INFO')
    
    def run(self, extra_args: Optional[list] = None):
        """Run training with this configuration"""
        self.apply()
        
        # Patch RLLM mappings
        print("\nPatching RLLM mappings...")
        patch_script = os.path.join(os.path.dirname(__file__), "patch_rllm_mappings.py")
        patch_result = subprocess.run([sys.executable, patch_script])
        if patch_result.returncode != 0:
            print("Error patching RLLM mappings")
            return patch_result
        
        # Download model if needed
        print(f"\nChecking model {self.env_vars.get('HF_MODEL_NAME', 'unknown')}...")
        download_script = os.path.join(os.path.dirname(__file__), "download_model.py")
        download_result = subprocess.run([sys.executable, download_script])
        if download_result.returncode != 0:
            print("Error downloading model")
            return download_result
        
        # Run training
        script_path = os.path.join(os.path.dirname(__file__), "train_terminal_bench_grpo_rllm.sh")
        
        # Ensure script is executable
        import stat
        current_permissions = os.stat(script_path).st_mode
        os.chmod(script_path, current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        
        cmd = [script_path] + (extra_args or [])
        return subprocess.run(cmd)


# Common environment variables across all configurations
COMMON_ENV_VARS = {
    "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "LLM_JUDGE_NAME": os.getenv("LLM_JUDGE_NAME", "anthropic/claude-sonnet-4-20250514"),
}

# Configuration presets
CONFIGS = {
    "test_8b_2_gpus": TrainingConfig(
        name="test_8b_2_gpus",
        description="Test configuration with Qwen3-8B on 2x 80GB GPUS (A100/H100)",
        env_vars={
            **COMMON_ENV_VARS,
            "HF_MODEL_NAME": "Qwen/Qwen3-8B",
            "MODEL_PATH": "./models/Qwen_Qwen3-8B",
            "PROJECT_NAME": "terminal_bench_test",
            "EXPERIMENT_NAME": "qwen3-8b_test_run",
            "MAX_SEQUENCE_LENGTH": "8192",
            "MAX_ROLLOUT_LENGTH": "4096",
            "N_ROLLOUTS": "4",
            "TRAIN_BATCH_SIZE": "1",
            "PPO_MINI_BATCH_SIZE": "1",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "N_GPUS_PER_NODE": "2",
            "NNODES": "1",
            "TP_SIZE": "2",
            "ULYSSES_SEQUENCE_PARALLEL_SIZE": "1",
            "ACTOR_LR": "1e-6",  # Standard learning rate
            "MAX_STEPS": "30",
            "TRAJECTORY_TIMEOUT": "300",
            "LOG_LEVEL": "DEBUG",  # More verbose for testing
            "NUM_EPOCHS": "1",  # Shorter for testing
            "SAVE_FREQ": "10",
            "REJECTION_SAMPLING_MULTIPLIER": "2", 
        }
    ),
    
    "runway_32b_4_gpus": TrainingConfig(
        name="runway_32b",
        description="Runway configuration with Qwen3-32B on 4x 80GB GPUs (A100/H100)",
        env_vars={
            **COMMON_ENV_VARS,
            "HF_MODEL_NAME": "Qwen/Qwen3-32B",
            "MODEL_PATH": "./models/Qwen_Qwen3-32B",
            "PROJECT_NAME": "terminal_bench_grpo_agent",
            "EXPERIMENT_NAME": "qwen3-32b_terminal_agent",
            "MAX_SEQUENCE_LENGTH": "32768",
            "MAX_ROLLOUT_LENGTH": "4000",
            "N_ROLLOUTS": "10",
            "TRAIN_BATCH_SIZE": "1",
            "PPO_MINI_BATCH_SIZE": "1",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "N_GPUS_PER_NODE": "4",
            "NNODES": "1",
            "TP_SIZE": "2",
            "ULYSSES_SEQUENCE_PARALLEL_SIZE": "4",
            "ACTOR_LR": "1e-6",  # Standard learning rate
            "MAX_STEPS": "50",
            "TRAJECTORY_TIMEOUT": "600",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.7",  # Conservative memory utilization
            "LOG_LEVEL": "INFO",  # Less verbose for production
            "NUM_EPOCHS": "10",  # Full training run
            "SAVE_FREQ": "5",
            "REJECTION_SAMPLING_MULTIPLIER": "2", 
        }
    ),
    
    "test_32b_2_gpus": TrainingConfig(
        name="test_32b_2_gpus",
        description="Test configuration with Qwen3-32B on 2x 80GB GPUs (A100/H100)",
        env_vars={
            **COMMON_ENV_VARS,
            "HF_MODEL_NAME": "Qwen/Qwen3-32B",
            "MODEL_PATH": "./models/Qwen_Qwen3-32B",
            "PROJECT_NAME": "terminal_bench_32b_test",
            "EXPERIMENT_NAME": "qwen3-32b_2gpu_test",
            "MAX_SEQUENCE_LENGTH": "32768",  # Reduced for memory constraints
            "MAX_ROLLOUT_LENGTH": "4000",
            "N_ROLLOUTS": "2",  # Fewer rollouts for testing
            "TRAIN_BATCH_SIZE": "1",
            "PPO_MINI_BATCH_SIZE": "1",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "N_GPUS_PER_NODE": "2",
            "NNODES": "1",
            "TP_SIZE": "2",  # Tensor parallel across both GPUs
            "ULYSSES_SEQUENCE_PARALLEL_SIZE": "2",  # No sequence parallelism needed
            "ACTOR_LR": "1e-6",
            "MAX_STEPS": "30",
            "TRAJECTORY_TIMEOUT": "600",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.85",  # Can use more memory with fewer GPUs
            "LOG_LEVEL": "INFO",
            "NUM_EPOCHS": "1",  # Short test run
            "SAVE_FREQ": "10",
            "REJECTION_SAMPLING_MULTIPLIER": "2", 
        }
    ),
    
    "prod_32b_8_gpus": TrainingConfig(
        name="prod_32b_8_gpus",
        description="Production configuration with Qwen3-32B on 8x 80GB GPUs (A100/H100)",
        env_vars={
            **COMMON_ENV_VARS,
            "HF_MODEL_NAME": "Qwen/Qwen3-32B",
            "MODEL_PATH": "./models/Qwen_Qwen3-32B",
            "PROJECT_NAME": "terminal_bench_grpo_agent_prod",
            "EXPERIMENT_NAME": "qwen3-32b_prod_8gpu",
            "MAX_SEQUENCE_LENGTH": "32768",
            "MAX_ROLLOUT_LENGTH": "4000",
            "N_ROLLOUTS": "12",  
            "TRAIN_BATCH_SIZE": "1",
            "PPO_MINI_BATCH_SIZE": "1",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "N_GPUS_PER_NODE": "8",
            "NNODES": "1",
            "TP_SIZE": "4",  # Good tensor parallelism for 8 GPUs
            "ULYSSES_SEQUENCE_PARALLEL_SIZE": "8", 
            "ACTOR_LR": "1e-6",  # Standard learning rate
            "MAX_STEPS": "100",
            "TRAJECTORY_TIMEOUT": "600",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.8",  # Conservative memory utilization
            "LOG_LEVEL": "INFO",
            "NUM_EPOCHS": "10",  # Extended training for production
            "SAVE_FREQ": "5",
            "REJECTION_SAMPLING_MULTIPLIER": "2", 
        }
    ),
    
    "prod_32b_2x8_h100": TrainingConfig(
        name="prod_32b_2x8_h100",
        description="Production configuration with Qwen3-32B on 2 nodes x 8 H100 GPUs (16 total)",
        env_vars={
            **COMMON_ENV_VARS,
            "HF_MODEL_NAME": "Qwen/Qwen3-32B",
            "MODEL_PATH": "./models/Qwen_Qwen3-32B",
            "PROJECT_NAME": "terminal_bench_grpo_agent_multinode",
            "EXPERIMENT_NAME": "qwen3-32b_2node_8gpu",
            "MAX_SEQUENCE_LENGTH": "32768",
            "MAX_ROLLOUT_LENGTH": "4000",
            "N_ROLLOUTS": "16",  # More rollouts with more GPUs
            "TRAIN_BATCH_SIZE": "1",
            "PPO_MINI_BATCH_SIZE": "1",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "N_GPUS_PER_NODE": "8",
            "NNODES": "2",  # Multi-node setup
            "TP_SIZE": "8",  # Tensor parallel within each node
            "ULYSSES_SEQUENCE_PARALLEL_SIZE": "16",  # Sequence parallel across all 16 GPUs
            "ACTOR_LR": "1e-6",
            "MAX_STEPS": "100",  # More steps for production
            "TRAJECTORY_TIMEOUT": "600",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.85",  # H100 has more memory
            "LOG_LEVEL": "INFO",
            "NUM_EPOCHS": "10", 
            "SAVE_FREQ": "5",
            "REJECTION_SAMPLING_MULTIPLIER": "2", 
        }
    ),
    
    "prod_32b_4x8_h100": TrainingConfig(
        name="prod_32b_4x8_h100",
        description="Production configuration with Qwen3-32B on 4 nodes x 8 H100 GPUs (32 total)",
        env_vars={
            **COMMON_ENV_VARS,
            "HF_MODEL_NAME": "Qwen/Qwen3-32B",
            "MODEL_PATH": "./models/Qwen_Qwen3-32B",
            "PROJECT_NAME": "terminal_bench_grpo_agent_4node",
            "EXPERIMENT_NAME": "qwen3-32b_4node_8gpu",
            "MAX_SEQUENCE_LENGTH": "32768",
            "MAX_ROLLOUT_LENGTH": "4000",
            "N_ROLLOUTS": "16",  # 16 rollouts (x2 with rejection sampling = 32 total)
            "TRAIN_BATCH_SIZE": "2",  # Increased batch size for more GPUs
            "PPO_MINI_BATCH_SIZE": "2",
            "PPO_MICRO_BATCH_SIZE_PER_GPU": "1",
            "N_GPUS_PER_NODE": "8",
            "NNODES": "4",
            "TP_SIZE": "8",  # Tensor parallel within each node
            "ULYSSES_SEQUENCE_PARALLEL_SIZE": "32",  # Sequence parallel across all 32 GPUs
            "ACTOR_LR": "1e-6",
            "MAX_STEPS": "100",
            "TRAJECTORY_TIMEOUT": "600",
            "VLLM_GPU_MEMORY_UTILIZATION": "0.8",
            "LOG_LEVEL": "INFO",
            "NUM_EPOCHS": "10",
            "SAVE_FREQ": "5",
            "REJECTION_SAMPLING_MULTIPLIER": "2", 
        }
    ),
}


def _run_initial_docker_cleanup():
    """Run initial cleanup of RLLM Docker resources."""
    print("\nRunning initial Docker cleanup of RLLM resources...")
    cleanup_commands = [
        # Stop all RLLM containers
        ["bash", "-c", "docker ps -a | grep 'rllm_' | awk '{print $1}' | xargs -r docker stop"],
        # Remove all RLLM containers
        ["bash", "-c", "docker ps -a | grep 'rllm_' | awk '{print $1}' | xargs -r docker rm -f"],
        # Remove all RLLM networks
        ["bash", "-c", "docker network ls | grep 'rllm_' | awk '{print $1}' | xargs -r docker network rm"],
        # Final cleanup
        ["docker", "container", "prune", "-f"],
        ["docker", "network", "prune", "-f"]
    ]
    
    for cmd in cleanup_commands:
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as e:
            print(f"Warning: Cleanup command failed: {e}")
    
    print("Initial cleanup completed")


def main():
    """CLI for running training configurations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Terminal Bench RLLM training with preset configurations")
    parser.add_argument("config", choices=list(CONFIGS.keys()), nargs='?',
                       help="Configuration preset to use")
    parser.add_argument("--list", action="store_true",
                       help="List available configurations")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print configuration without running")
    parser.add_argument("--skip-upload", action="store_true",
                       help="Skip uploading checkpoint to HuggingFace after training")
    parser.add_argument("extra_args", nargs="*",
                       help="Additional arguments to pass to training script")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available configurations:")
        for name, config in CONFIGS.items():
            print(f"\n{name}: {config.description}")
            print("  Environment variables:")
            for key, value in sorted(config.env_vars.items()):
                print(f"    {key}={value}")
        return
    
    if not args.config:
        parser.error("config is required unless using --list")
    
    config = CONFIGS[args.config]
    print(f"Using configuration: {config.name} - {config.description}")
    
    if args.dry_run:
        print("\nEnvironment variables that are set by default:")
        for key, value in sorted(os.environ.items()):
                print(f"  {key}={value}")

        print("\nEnvironment variables that would be set:")
        for key, value in sorted(config.env_vars.items()):
            print(f"  {key}={value}")
        if args.extra_args:
            print(f"\nExtra arguments: {' '.join(args.extra_args)}")
        return
    
    if os.getenv("WANDB_API_KEY", "") == "":
        raise EnvironmentError("WANDB_API_KEY is not set. Please set it in your environment.")
    
    print("\nStarting training...")
    
    # Run initial aggressive cleanup of RLLM containers/networks
    if not os.getenv("SKIP_INITIAL_CLEANUP", "").lower() in ["true", "1", "yes"]:
        _run_initial_docker_cleanup()
    
    # Start Docker cleanup service
    cleanup_interval = int(os.getenv("DOCKER_CLEANUP_INTERVAL", "120"))  # Default 2 minutes
    print(f"Starting Docker cleanup service (interval: {cleanup_interval}s)")
    start_docker_cleanup(cleanup_interval)
    
    # Ensure cleanup stops on exit
    atexit.register(stop_docker_cleanup)
    
    result = config.run(args.extra_args)
    
    # Upload checkpoint to HuggingFace if training succeeded
    if result.returncode == 0 and not args.skip_upload:
        print("\n" + "="*60)
        print("Training completed successfully!")
        
        if not os.getenv("HF_USERNAME"):
            print("Warning: HF_USERNAME not set. Skipping automatic upload.")
            print("To enable automatic upload, set HF_USERNAME environment variable.")

        elif not os.getenv("HF_TOKEN"):
            print("Warning: HF_TOKEN not set. Skipping automatic upload.")
            print("To enable automatic upload, set HF_TOKEN environment variable.")
        else:
            print("Uploading latest checkpoint to HuggingFace...")
            
            upload_script = os.path.join(os.path.dirname(__file__), "upload_checkpoint.py")
            upload_cmd = [sys.executable, upload_script]
            
            # The upload script will use PROJECT_NAME and EXPERIMENT_NAME from environment
            upload_result = subprocess.run(upload_cmd, env=os.environ)
            
            if upload_result.returncode == 0:
                print("Checkpoint upload completed successfully!")
            else:
                print("Warning: Checkpoint upload failed. You can manually upload later with:")
                print(f"  python {upload_script}")
        
        print("="*60 + "\n")
    elif result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()