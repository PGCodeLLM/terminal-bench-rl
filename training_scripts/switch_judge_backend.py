#!/usr/bin/env python3
"""Helper script to switch LLM judge backends dynamically.
Only works on a single node GPU for now.

Usage:
    # Switch to LiteLLM with a specific model
    python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229
    
    # Switch to LiteLLM with environment variable overrides
    python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229 --env ANTHROPIC_API_KEY=sk-ant-... --env ANTHROPIC_BASE_URL=https://...
    
    # Unset an environment variable by setting it to 'unset'
    python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229 --env SOME_VAR=unset --env ANOTHER_VAR=value
    
    # Switch to Claude Code CLI
    python switch_judge_backend.py ccode
    
    # Show current backend
    python switch_judge_backend.py show
    
    # Remove backend override (use environment variable)
    python switch_judge_backend.py remove
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Switch LLM judge backend dynamically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "command",
        choices=["litellm", "ccode", "show", "remove"],
        help="Backend type or command"
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (required for litellm)"
    )
    
    parser.add_argument(
        "--env",
        action="append",
        help="Environment variables to override (format: KEY=value)"
    )
    
    args = parser.parse_args()
    
    config_file = Path(".judge_backend")
    
    if args.command == "show":
        if config_file.exists():
            content = config_file.read_text().strip()
            lines = content.splitlines()
            
            # Parse main backend info
            main_line = lines[0]
            print(f"Current backend: {main_line}")
            
            # Check for environment overrides
            if len(lines) > 1:
                for line in lines[1:]:
                    if line.startswith("env:"):
                        import json
                        env_data = json.loads(line[4:])
                        print("Environment overrides:")
                        for key, value in env_data.items():
                            if value == "unset":
                                print(f"  {key}=(will be unset)")
                            else:
                                # Mask sensitive values
                                if "KEY" in key or "TOKEN" in key or "SECRET" in key:
                                    masked_value = value[:8] + "..." if len(value) > 8 else "***"
                                else:
                                    masked_value = value
                                print(f"  {key}={masked_value}")
        else:
            print("No backend override set (using environment variable)")
    
    elif args.command == "remove":
        if config_file.exists():
            config_file.unlink()
            print("Backend override removed")
        else:
            print("No backend override to remove")
    
    elif args.command == "litellm":
        if not args.model:
            print("Error: Model name required for litellm backend")
            print("Example: python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229")
            sys.exit(1)
        
        # Only accept anthropic/ models
        if not args.model.startswith("anthropic/"):
            print("Error: Only Anthropic models are supported for litellm backend")
            print("Model must start with 'anthropic/'")
            print("Example: python switch_judge_backend.py litellm anthropic/claude-3-opus-20240229")
            sys.exit(1)
        
        # Check for API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable not found")
            print("Please set your Anthropic API key:")
            print("  export ANTHROPIC_API_KEY='your-api-key-here'")
            sys.exit(1)
        
        content = f"litellm:{args.model}"
        
        # Add environment variables if provided
        if args.env:
            env_vars = {}
            for env_str in args.env:
                if '=' not in env_str:
                    print(f"Warning: Invalid environment variable format: {env_str}")
                    print("Expected format: KEY=value")
                    continue
                key, value = env_str.split('=', 1)
                env_vars[key] = value
            
            if env_vars:
                import json
                content += f"\nenv:{json.dumps(env_vars)}"
        
        config_file.write_text(content)
        print(f"Switched to LiteLLM backend with model: {args.model}")
        if args.env:
            print(f"With environment overrides: {', '.join(args.env)}")
    
    elif args.command == "ccode":
        content = "ccode-4-sonnet"
        
        # Add environment variables if provided
        if args.env:
            env_vars = {}
            for env_str in args.env:
                if '=' not in env_str:
                    print(f"Warning: Invalid environment variable format: {env_str}")
                    print("Expected format: KEY=value")
                    continue
                key, value = env_str.split('=', 1)
                env_vars[key] = value
            
            if env_vars:
                import json
                content += f"\nenv:{json.dumps(env_vars)}"
        
        config_file.write_text(content)
        print("Switched to Claude Code CLI backend")
        if args.env:
            print(f"With environment overrides: {', '.join(args.env)}")


if __name__ == "__main__":
    main()