#!/usr/bin/env python3
"""Upload Terminal Bench RLLM checkpoints to HuggingFace Hub"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

def find_latest_checkpoint_from_file(project_name: str, experiment_name: str, base_dir: str = "checkpoints") -> Optional[Tuple[str, int]]:
    """Find the latest checkpoint by reading the latest_checkpointed_iteration.txt file.
    
    Returns:
        Tuple of (checkpoint_path, step_number) or None if not found
    """
    checkpoint_dir = os.path.join(base_dir, project_name, experiment_name)
    latest_file = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")
    
    if not os.path.exists(latest_file):
        print(f"No latest checkpoint file found at {latest_file}")
        return None
    
    try:
        with open(latest_file, 'r') as f:
            step = int(f.read().strip())
        
        checkpoint_path = os.path.join(checkpoint_dir, f"global_step_{step}")
        if os.path.exists(checkpoint_path):
            return checkpoint_path, step
        else:
            print(f"Checkpoint directory not found: {checkpoint_path}")
            return None
    except Exception as e:
        print(f"Error reading latest checkpoint file: {e}")
        return None


def find_all_checkpoints(project_name: str, experiment_name: str, base_dir: str = "checkpoints") -> list:
    """Find all checkpoints for a given project and experiment.
    
    Returns:
        List of (checkpoint_path, step_number) tuples sorted by step number
    """
    checkpoint_dir = os.path.join(base_dir, project_name, experiment_name)
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("global_step_"):
            try:
                step = int(item.split("_")[-1])
                checkpoint_path = os.path.join(checkpoint_dir, item)
                checkpoints.append((checkpoint_path, step))
            except ValueError:
                continue
    
    return sorted(checkpoints, key=lambda x: x[1])


def upload_checkpoint_to_hf(checkpoint_path: str, hf_repo_name: str, hf_token: Optional[str] = None, private: bool = False) -> bool:
    """Upload a checkpoint to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., checkpoints/.../global_step_50)
        hf_repo_name: HuggingFace repository name (e.g., "username/model-name")
        hf_token: Optional HuggingFace token (uses HF_TOKEN env var if not provided)
        private: Whether to make the repository private
    
    Returns:
        True if upload was successful, False otherwise
    """
    # Ensure we have the actor subdirectory
    actor_path = os.path.join(checkpoint_path, "actor")
    
    if not os.path.exists(actor_path):
        print(f"Error: Actor checkpoint not found at {actor_path}")
        return False
    
    # Check if we already have a merged HF model
    hf_checkpoint_path = os.path.join(actor_path, "checkpoint")
    if os.path.exists(hf_checkpoint_path):
        print(f"Found pre-merged HuggingFace model at {hf_checkpoint_path}")
        # Use huggingface_hub to upload directly
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
            
            # Create repo if it doesn't exist
            try:
                api.create_repo(repo_id=hf_repo_name, private=private, exist_ok=True)
            except Exception as e:
                print(f"Note: Could not create repo (may already exist): {e}")
            
            # Upload the folder
            api.upload_folder(
                folder_path=hf_checkpoint_path,
                repo_id=hf_repo_name,
                repo_type="model",
                commit_message=f"Upload checkpoint from Terminal Bench training"
            )
            print(f"Successfully uploaded checkpoint to https://huggingface.co/{hf_repo_name}")
            return True
        except ImportError:
            print("huggingface_hub not installed, falling back to model merger script")
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")
            print("Falling back to model merger script...")
    
    # Use the model merger script
    merger_script = os.path.join(os.path.dirname(__file__), "..", "external", "rllm", "verl", "scripts", "model_merger.py")
    if not os.path.exists(merger_script):
        print(f"Error: Model merger script not found at {merger_script}")
        return False
    
    cmd = [
        sys.executable, merger_script, "merge",
        "--backend", "fsdp",
        "--local_dir", actor_path,
        "--hf_upload_path", hf_repo_name,
    ]
    
    if private:
        cmd.append("--private")
    
    if hf_token:
        env = os.environ.copy()
        env["HF_TOKEN"] = hf_token
    else:
        env = os.environ
    
    print(f"Merging and uploading checkpoint to {hf_repo_name}...")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        print(f"Successfully uploaded checkpoint to https://huggingface.co/{hf_repo_name}")
        return True
    else:
        print(f"Error uploading checkpoint (return code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload Terminal Bench RLLM checkpoints to HuggingFace Hub")
    parser.add_argument("--project", default=os.getenv("PROJECT_NAME"), help="Project name (defaults to PROJECT_NAME env var)")
    parser.add_argument("--experiment", default=os.getenv("EXPERIMENT_NAME"), help="Experiment name (defaults to EXPERIMENT_NAME env var)")
    parser.add_argument("--hf-repo", help="HuggingFace repository name (defaults to HF_USERNAME/experiment_name_ddmm)")
    parser.add_argument("--hf-username", default=os.getenv("HF_USERNAME"), help="HuggingFace username (defaults to HF_USERNAME env var)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Base checkpoint directory (default: checkpoints)")
    parser.add_argument("--step", type=int, help="Specific checkpoint step to upload (default: latest)")
    parser.add_argument("--list", action="store_true", help="List all available checkpoints")
    parser.add_argument("--hf-token", help="HuggingFace API token (defaults to HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", default=True, help="Make the HuggingFace repository private (default: True)")
    
    args = parser.parse_args()
    
    # Validate required environment variables if not provided as arguments
    if not args.project:
        print("Error: No project name provided. Set PROJECT_NAME environment variable or use --project")
        return 1
    
    if not args.experiment:
        print("Error: No experiment name provided. Set EXPERIMENT_NAME environment variable or use --experiment")
        return 1
    
    # Generate default HF repo name if not provided
    if not args.hf_repo:
        if not args.hf_username:
            print("Error: No HuggingFace username provided. Set HF_USERNAME environment variable or use --hf-username")
            return 1
        # Format: username/experiment_name_ddmm
        date_str = datetime.now().strftime("%d%m")
        args.hf_repo = f"{args.hf_username}/{args.experiment}_{date_str}"
        print(f"Using auto-generated repo name: {args.hf_repo}")
    
    # List checkpoints if requested
    if args.list:
        checkpoints = find_all_checkpoints(args.project, args.experiment, args.checkpoint_dir)
        if not checkpoints:
            print(f"No checkpoints found for {args.project}/{args.experiment}")
            return 1
        
        print(f"Available checkpoints for {args.project}/{args.experiment}:")
        for ckpt_path, step in checkpoints:
            print(f"  Step {step}: {ckpt_path}")
        
        # Also show latest from file
        latest = find_latest_checkpoint_from_file(args.project, args.experiment, args.checkpoint_dir)
        if latest:
            _, latest_step = latest
            print(f"\nLatest checkpoint (from file): Step {latest_step}")
        return 0
    
    # Find checkpoint to upload
    if args.step:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.project, args.experiment, f"global_step_{args.step}")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return 1
        checkpoint_info = (checkpoint_path, args.step)
    else:
        # Try to find latest checkpoint from file first
        checkpoint_info = find_latest_checkpoint_from_file(args.project, args.experiment, args.checkpoint_dir)
        if not checkpoint_info:
            # Fall back to finding all checkpoints and taking the latest
            checkpoints = find_all_checkpoints(args.project, args.experiment, args.checkpoint_dir)
            if not checkpoints:
                print(f"No checkpoints found for {args.project}/{args.experiment}")
                return 1
            checkpoint_info = checkpoints[-1]  # Latest by step number
    
    checkpoint_path, step = checkpoint_info
    print(f"Uploading checkpoint from step {step}: {checkpoint_path}")
    
    # Upload to HuggingFace
    success = upload_checkpoint_to_hf(
        checkpoint_path,
        args.hf_repo,
        hf_token=args.hf_token,
        private=args.private
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())