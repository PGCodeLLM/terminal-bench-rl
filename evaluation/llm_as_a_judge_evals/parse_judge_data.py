#!/usr/bin/env python3
"""Parse judge evaluation data from text format into messages.json and dockerfile."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_judge_data(content: str) -> Tuple[str, List[Dict[str, str]]]:
    """Parse the text format into dockerfile contents and messages."""
    
    # Split into dockerfile and trajectory sections
    dockerfile_match = re.search(
        r'# Dockerfile contents:\s*\n```dockerfile\n(.*?)\n```',
        content,
        re.DOTALL
    )
    
    if not dockerfile_match:
        raise ValueError("Could not find dockerfile section in the expected format")
    
    dockerfile_contents = dockerfile_match.group(1).strip()
    
    # Find the trajectory section - look for the LAST ``` that closes the trajectory
    # First find where trajectory starts
    trajectory_start_match = re.search(
        r'# Agent trajectory:\s*\n```\n',
        content
    )
    
    if not trajectory_start_match:
        raise ValueError("Could not find agent trajectory section start")
    
    trajectory_start = trajectory_start_match.end()
    
    # Find all ``` after the trajectory start
    remaining_content = content[trajectory_start:]
    
    # Find the last standalone ``` (on its own line) that closes the trajectory block
    # We need to find the outermost closing ```
    last_closing_backticks = -1
    for match in re.finditer(r'\n```\s*$', remaining_content, re.MULTILINE):
        last_closing_backticks = match.start()
    
    if last_closing_backticks == -1:
        raise ValueError("Could not find closing ``` for agent trajectory")
    
    trajectory_text = remaining_content[:last_closing_backticks].strip()
    
    # Parse messages from trajectory
    messages = []
    
    # Find all message headers and their positions
    message_pattern = r'\[Message \d+ by (User|Agent|Env Response)\]'
    matches = list(re.finditer(message_pattern, trajectory_text))
    
    # Process each message
    for i, match in enumerate(matches):
        role_text = match.group(1)
        
        # Get content: from end of this header to start of next header (or end of text)
        content_start = match.end()
        if i + 1 < len(matches):
            content_end = matches[i + 1].start()
        else:
            content_end = len(trajectory_text)
        
        content = trajectory_text[content_start:content_end].strip()
        
        # Map roles
        if role_text == "User":
            role = "user"
        elif role_text == "Agent":
            role = "assistant"
        elif role_text == "Env Response":
            role = "user"  # Env responses are treated as user messages
        else:
            role = role_text.lower()
        
        messages.append({
            "role": role,
            "content": content
        })
    
    return dockerfile_contents, messages


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parse judge evaluation data from text format"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input .txt file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for messages.json and dockerfile (default: same as input file)"
    )
    parser.add_argument(
        "--test-id",
        type=str,
        help="Test case ID (default: derived from input filename)"
    )
    
    args = parser.parse_args()
    
    # Read input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the data
    try:
        dockerfile_contents, messages = parse_judge_data(content)
    except ValueError as e:
        print(f"Error parsing file: {e}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to judge_eval_cases/<test_id>
        test_id = args.test_id or input_path.stem
        output_dir = Path(__file__).parent / "judge_eval_cases" / test_id
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write dockerfile
    dockerfile_path = output_dir / "dockerfile"
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.write(dockerfile_contents)
    print(f"✓ Wrote dockerfile to: {dockerfile_path}")
    
    # Write messages.json
    messages_path = output_dir / "messages.json"
    with open(messages_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    print(f"✓ Wrote messages.json to: {messages_path}")
    
    # Summary
    print(f"\nParsed {len(messages)} messages:")
    role_counts = {}
    for msg in messages:
        role = msg['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    for role, count in sorted(role_counts.items()):
        print(f"  - {role}: {count}")
    
    print(f"\nTest case data created in: {output_dir}")
    
    # Reminder about config
    if not args.output_dir:
        print(f"\nDon't forget to add this test case to judge_eval_config.yaml:")
        print(f"""
  - id: "{test_id}"
    description: "Description of what this test validates"
    expected_range: [0.0, 1.0]  # Adjust range as needed
""")
    
    return 0


if __name__ == "__main__":
    exit(main())