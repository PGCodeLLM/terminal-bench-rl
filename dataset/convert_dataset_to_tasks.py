#!/usr/bin/env python3
"""
Convert dataset to terminal_bench task structure.

This script reads the file and creates task structures using the
terminal_bench CLI tool, then customizes the generated files with content
from the dataset.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
from enum import Enum
import time


class TaskStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


def validate_dataset(csv_path: Path, dry_run: bool = True) -> Optional[List[Dict]]:
    """Validate the dataset and return rows if valid."""
    required_columns = {
        'task_id', 'difficulty', 'category', 'tags', 'prompt',
        'dockerfile', 'test_functions', 'test_weights'
    }
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return None
    
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                print(f"❌ Error: Missing required columns: {missing}")
                return None
            
            # Validate each row
            for idx, row in enumerate(reader, 1):
                # Check for empty required fields
                empty_fields = []
                for col in required_columns:
                    if not row.get(col, '').strip():
                        empty_fields.append(col)
                
                if empty_fields:
                    print(f"⚠️  Row {idx} (task_id: {row.get('task_id', 'UNKNOWN')}): "
                          f"Empty fields: {empty_fields}")
                    if not dry_run:
                        continue
                
                rows.append(row)
    
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None
    
    if dry_run:
        print(f"✅ Validation complete: {len(rows)} rows found")
        print(f"✅ All required columns present: {sorted(required_columns)}")
    
    return rows


def create_task_structure(row: Dict, output_dir: Path, dry_run: bool = False) -> TaskStatus:
    """Create task structure using tb CLI and customize files."""
    task_id = row['task_id']
    task_path = output_dir / task_id
    
    # Check if task already exists
    if task_path.exists():
        print(f"⏭️  Skipping {task_id}: already exists")
        return TaskStatus.SKIPPED
    
    if dry_run:
        print(f"✅ Would create task: {task_id}")
        return TaskStatus.SUCCESS
    
    # Parse tags (pipe-separated)
    tags = row['tags'].split('|') if row['tags'] else []
    
    # Map difficulty values
    difficulty = row['difficulty']
    if difficulty == 'extremely_hard':
        difficulty = 'hard'
    
    # Build tb command
    cmd = [
        'uv', 'run', '--active', 'tb', 'tasks', 'create', task_id,
        '--name', 'Dan Austin',
        '--email', 'dan@aituning.ai',
        '--category', row['category'],
        '--difficulty', difficulty,
        '--instruction', row['prompt'],
        '--no-interactive',
        '--tasks-dir', str(output_dir)
    ]
    
    # Add tags
    for tag in tags:
        if tag.strip():
            cmd.extend(['--tag', tag.strip()])
    
    print(f"🔨 Creating task structure: {task_id}")
    
    try:
        # Run tb command with proper environment and timeout
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            env=env,
            timeout=30,  # 30 second timeout
            stdin=subprocess.DEVNULL  # Prevent hanging on input
        )
        
        if result.returncode != 0:
            print(f"❌ Error creating task {task_id}:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return TaskStatus.FAILED
        
        # Replace Dockerfile
        dockerfile_path = task_path / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(row['dockerfile'])
        
        # Replace test_outputs.py
        test_file_path = task_path / 'tests' / 'test_outputs.py'
        with open(test_file_path, 'w') as f:
            f.write(row['test_functions'])
        
        # Create test_weights.json
        test_weights_path = task_path / 'test_weights.json'
        try:
            # Parse JSON string to ensure it's valid
            test_weights = json.loads(row['test_weights'])
            with open(test_weights_path, 'w') as f:
                json.dump(test_weights, f, indent=2)
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Invalid JSON in test_weights for {task_id}: {e}")
            # Write as-is if not valid JSON
            with open(test_weights_path, 'w') as f:
                f.write(row['test_weights'])
        
        # Handle additional files if present
        if 'additional_files' in row and row['additional_files']:
            try:
                additional_files = json.loads(row['additional_files'])
                if isinstance(additional_files, dict):
                    for file_path_str, content in additional_files.items():
                        # Create the file path relative to task directory
                        file_path = task_path / file_path_str
                        # Create parent directories if needed
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        # Write the file content
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"   📄 Created additional file: {file_path_str}")
                else:
                    print(f"⚠️  Warning: additional_files for {task_id} is not a dictionary")
            except json.JSONDecodeError as e:
                raise e
            except Exception as e:
                raise e
        
        print(f"✅ Successfully created task: {task_id}")
        return TaskStatus.SUCCESS
        
    except Exception as e:
        print(f"❌ Error processing task {task_id}: {e}")
        return TaskStatus.FAILED


def process_task_wrapper(args: Tuple[Dict, Path]) -> Tuple[TaskStatus, str]:
    """Wrapper function for multiprocessing to handle a single task."""
    row, output_dir = args
    task_id = row['task_id']
    try:
        status = create_task_structure(row, output_dir, dry_run=False)
        return (status, task_id)
    except Exception as e:
        print(f"❌ Unexpected error processing {task_id}: {e}")
        return (TaskStatus.FAILED, task_id)


def main():
    parser = argparse.ArgumentParser(
        description='Convert dataset to terminal_bench task structures'
    )
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path('dataset/latest_verified.csv'),
        help='Path to the CSV file'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('tasks'),
        help='Directory where tasks will be created (default: tasks)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate dataset without creating tasks'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of tasks to process (useful for testing)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Make paths absolute
    if not args.csv_path.is_absolute():
        args.csv_path = Path.cwd() / args.csv_path
    if not args.output_dir.is_absolute():
        args.output_dir = Path.cwd() / args.output_dir
    
    print(f"📂 CSV Path: {args.csv_path}")
    print(f"📂 Output Directory: {args.output_dir}")
    print(f"🔍 Mode: {'Dry Run' if args.dry_run else 'Create Tasks'}")
    print()
    
    # Validate dataset
    rows = validate_dataset(args.csv_path, args.dry_run)
    if not rows:
        sys.exit(1)
    
    if args.dry_run:
        print(f"\n📊 Dataset Summary:")
        print(f"   Total rows: {len(rows)}")
        
        # Count categories and difficulties
        categories = {}
        difficulties = {}
        for row in rows:
            cat = row.get('category', 'unknown')
            diff = row.get('difficulty', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        print(f"   Categories: {dict(sorted(categories.items()))}")
        print(f"   Difficulties: {dict(sorted(difficulties.items()))}")
        return
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process tasks
    if args.limit:
        rows = rows[:args.limit]
        print(f"⚠️  Limited to first {args.limit} tasks\n")
    
    print(f"🚀 Processing {len(rows)} tasks with {args.workers} workers\n")
    
    # Prepare arguments for multiprocessing
    task_args = [(row, args.output_dir) for row in rows]
    
    # Process tasks in parallel
    success = 0
    failed = 0
    skipped = 0
    
    if args.workers == 1:
        # Sequential processing
        results = []
        for i, task_arg in enumerate(task_args):
            print(f"\n📋 Processing task {i+1}/{len(task_args)}")
            result = process_task_wrapper(task_arg)
            results.append(result)
            time.sleep(0.1)  # Small delay to avoid overwhelming
    else:
        # Parallel processing
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_task_wrapper, task_args)
    
    # Count results
    for status, task_id in results:
        if status == TaskStatus.SUCCESS:
            success += 1
        elif status == TaskStatus.SKIPPED:
            skipped += 1
        else:
            failed += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully created: {success}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏭️  Skipped (existing): {skipped}")


if __name__ == '__main__':
    main()