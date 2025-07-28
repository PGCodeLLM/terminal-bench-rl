"""Convert Terminal Bench tasks to RLLM/VERL format."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Add project to path
import sys

from src.tbench_rllm.load_tasks import TBenchTrainingTask, load_terminal_bench_tasks
sys.path.append(str(Path(__file__).parent.parent.parent))



def create_prompt_from_task(task: TBenchTrainingTask, system_prompt: Optional[str] = None) -> str:
    """Create a prompt from a terminal bench task."""
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant helping to complete terminal-based tasks. "
            "Follow the instructions carefully and use appropriate commands to accomplish the goal."
        )
    
    # Format as a chat conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.instruction}
    ]
    
    # Convert to string format (you might want to use a specific chat template)
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{task.instruction}\n<|assistant|>\n"
    
    return prompt


def convert_tasks_to_parquet(
    tasks_dir: Path,
    output_dir: Path,
    train_split: Optional[float] = None,
    system_prompt: Optional[str] = None,
    task_names: Optional[List[str]] = None,
    test_tasks_dir: Optional[Path] = None,
) -> None:
    """Convert terminal bench tasks to parquet format for VERL training.
    
    Args:
        tasks_dir: Directory containing terminal bench tasks (or train tasks if test_tasks_dir is provided)
        output_dir: Output directory for parquet files
        train_split: Fraction of data for training (ignored if test_tasks_dir is provided)
        system_prompt: System prompt to use
        task_names: Specific task names to convert
        test_tasks_dir: Directory containing test tasks for validation set
    """
    
    # Load tasks
    if test_tasks_dir is not None:
        # Load train and test tasks separately
        print(f"Loading training tasks from {tasks_dir}")
        train_tasks = load_terminal_bench_tasks(tasks_dir, task_names)
        print(f"Loaded {len(train_tasks)} training tasks")
        
        print(f"Loading validation tasks from {test_tasks_dir}")
        val_tasks = load_terminal_bench_tasks(test_tasks_dir, task_names)
        print(f"Loaded {len(val_tasks)} validation tasks")
        
        tasks = train_tasks + val_tasks
    else:
        # Load all tasks from single directory
        print(f"Loading tasks from {tasks_dir}")
        tasks = load_terminal_bench_tasks(tasks_dir, task_names)
        print(f"Loaded {len(tasks)} tasks")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for parquet
    data_records = []
    
    for task in tqdm(tasks, desc="Converting tasks"):
        record = {
            "prompt": create_prompt_from_task(task, system_prompt),
            "task_name": task.task_name,
            "task_path": str(task.task_path),
            "instruction": task.instruction,
            "data_source": "terminal_bench",  # For reward_fn_key
            "metadata": {
                "test_weights": task.test_weights,
                "max_test_timeout_sec": task.max_test_timeout_sec,
            }
        }
        
        # Always include extra_info with task configuration
        extra_info_dict = {
            "task_name": task.task_name,
            "task_path": str(task.task_path),
            "instruction": task.instruction,
            "test_weights": task.test_weights,
            "dockerfile_contents": task.dockerfile_contents,
            "py_test_file_contents": task.py_test_file_contents,
            "max_test_timeout_sec": task.max_test_timeout_sec,
        }
        
        # Include additional files if present
        if task.additional_files:
            extra_info_dict["additional_files"] = task.additional_files
            
        record["extra_info"] = json.dumps(extra_info_dict)
        
        data_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    # Split into train and validation
    if test_tasks_dir is not None:
        # Use pre-defined split based on directories
        n_train = len(train_tasks)
        train_df = df[:n_train]
        val_df = df[n_train:]
    else:
        # Use train_split parameter
        if train_split is None:
            train_split = 0.9
        n_train = int(len(df) * train_split)
        train_df = df[:n_train]
        val_df = df[n_train:]
    
    # Save to parquet
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Saved {len(train_df)} training examples to {train_path}")
    print(f"Saved {len(val_df)} validation examples to {val_path}")
    
    # Also save task information for the reward function
    tasks_info = {}
    for task in tasks:
        task_info = {
            "task_path": str(task.task_path),
            "test_weights": task.test_weights,
            "dockerfile_contents": task.dockerfile_contents,
            "py_test_file_contents": task.py_test_file_contents,
            "max_test_timeout_sec": task.max_test_timeout_sec,
        }
        # Include additional files if present
        if task.additional_files:
            task_info["additional_files"] = task.additional_files
        
        tasks_info[task.task_name] = task_info
    
    tasks_info_path = output_dir / "tasks_info.json"
    with open(tasks_info_path, "w") as f:
        json.dump(tasks_info, f, indent=2)
    
    print(f"Saved task information to {tasks_info_path}")



def main():
    # Fixed paths
    tasks_dir = Path("tasks")
    test_tasks_dir = Path("test_tasks")
    output_dir = Path("data")
    
    # Check if directories exist
    if not tasks_dir.exists():
        print(f"Error: {tasks_dir} directory not found")
        return
    
    if not test_tasks_dir.exists():
        print(f"Error: {test_tasks_dir} directory not found")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting tasks/ -> train.parquet and test_tasks/ -> val.parquet")
    
    # Convert to parquet only (with extra_info)
    convert_tasks_to_parquet(
        tasks_dir=tasks_dir,
        output_dir=output_dir,
        train_split=None,
        system_prompt=None,
        task_names=None,
        test_tasks_dir=test_tasks_dir,
    )


if __name__ == "__main__":
    main()