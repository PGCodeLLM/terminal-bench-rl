#!/usr/bin/env python3
"""Patch RLLM env_agent_mappings to only include terminal_bench"""

import os
import sys

def patch_mappings():
    # Get the path to env_agent_mappings.py
    rllm_path = os.path.join(os.path.dirname(__file__), "..", "external", "rllm")
    mappings_file = os.path.join(rllm_path, "rllm", "trainer", "env_agent_mappings.py")
    
    if not os.path.exists(mappings_file):
        print(f"Error: Could not find {mappings_file}")
        return False
    
    # Create the new content with only terminal_bench
    new_content = '''def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise e
        return None


# Import environment classes
ENV_CLASSES = {
    "terminal_bench": safe_import("src.tbench_rllm.docker_env", "DockerIsolatedEnv"),
}

# Import agent classes
AGENT_CLASSES = {
    "terminal_bench_agent": safe_import("src.tbench_rllm.terminal_agent", "TerminalBenchAgent"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
'''
    
    # Backup the original file
    backup_file = mappings_file + ".backup"
    if not os.path.exists(backup_file):
        with open(mappings_file, 'r') as f:
            original_content = f.read()
        with open(backup_file, 'w') as f:
            f.write(original_content)
        print(f"Created backup at {backup_file}")
    
    # Write the new content
    with open(mappings_file, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully patched {mappings_file}")
    return True

if __name__ == "__main__":
    if patch_mappings():
        sys.exit(0)
    else:
        sys.exit(1)