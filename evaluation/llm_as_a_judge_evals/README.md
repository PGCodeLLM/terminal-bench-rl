# Judge Evaluation Tool

Simple evaluation framework for testing the judge scoring system with different models.

## Usage

```bash
# Run all test cases
uv run python evaluation/llm_as_a_judge_evals/judge_eval.py --model openrouter/openai/gpt-4.1 --attempts 3


# Run a specific test case
uv run python evaluation/judge_evals/judge_eval.py --model "claude-3-5-sonnet-20241022" --test-id perfect_execution

# Generate performance report from results
uv run python evaluation/llm_as_a_judge_evals/report.py

# Generate report for specific judge version
uv run python evaluation/llm_as_a_judge_evals/report.py --judge-version v4.2
```

## Adding New Test Cases

### Method 1: Using the Parser (Recommended - if taking an log from judge_i_o_logs produced within a training run)

1. Create a `.txt` file with the data in the standard format judge input log format:
```
# Dockerfile contents:
```dockerfile
FROM ubuntu:22.04
...
```

# Agent trajectory:
```
[Message 1 by User]
Your task here...

[Message 2 by Agent]
Agent's response...

[Message 3 by Env Response]
Environment output...
```

2. Run the parser:
```bash
# Automatically creates in judge_eval_cases/<filename>/
uv run python evaluation/judge_evals/parse_judge_data.py your_test.txt

# Or specify test ID explicitly
uv run python evaluation/judge_evals/parse_judge_data.py your_test.txt --test-id my_test_id
```

3. Add the test case to `config.yaml` 

### Method 2: Manual Creation

1. Add an entry to `config.yaml`:
```yaml
- id: "your_test_id"
  description: "What this test case validates"
  expected_range: [0.6, 0.8]  # Min and max expected scores
```

2. Create a directory: `judge_eval_cases/your_test_id/`

3. Add two files:
   - `messages.json` - The conversation messages array
   - `dockerfile` - The dockerfile contents

Example messages.json structure:
```json
[
  {
    "role": "user",
    "content": "Create a Python script that..."
  },
  {
    "role": "assistant", 
    "content": "<todo>\noperations:\n  - action: add\n    content: \"Understand the task\"\n</todo>"
  },
  {
    "role": "user",
    "content": "Todo added: \"Understand the task\" (ID: 1)"
  }
]
```