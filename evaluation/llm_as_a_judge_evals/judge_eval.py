#!/usr/bin/env python3
"""Mini-evaluation script for testing judge reward with different models."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tbench_rllm.rewards.judge_reward import calculate_judge_score
from results_store import ResultsStore


class JudgeEvaluator:
    """Evaluator for testing judge scoring with different models."""
    
    def __init__(self, model_name: str, judge_version: str = "latest", csv_path: Path = None, n_attempts: int = 1):
        self.model_name = model_name
        self.judge_version = judge_version
        self.results = []
        self.store = ResultsStore(csv_path)
        self.n_attempts = n_attempts
        
        # Resolve judge version if "latest"
        if judge_version == "latest":
            latest_version, _ = ResultsStore.get_latest_judge_version()
            self.resolved_judge_version = latest_version
        else:
            self.resolved_judge_version = judge_version
        
    async def run_test_case(self, test_case: Dict) -> Tuple[str, float, bool, List[float]]:
        """Run a single test case multiple times and return aggregated results."""
        test_id = test_case['id']
        expected_range = test_case['expected_range']
        
        # Load test data
        case_dir = Path(__file__).parent / "judge_eval_cases" / test_id
        
        if not case_dir.exists():
            return test_id, 0.0, False, []
            
        messages_path = case_dir / "messages.json"
        dockerfile_path = case_dir / "dockerfile"
        
        if not messages_path.exists() or not dockerfile_path.exists():
            print(f"  ⚠️  Missing data files for {test_id}")
            return test_id, 0.0, False, []
            
        # Load messages and dockerfile
        with open(messages_path, 'r') as f:
            messages = json.load(f)
            
        with open(dockerfile_path, 'r') as f:
            dockerfile_contents = f.read()
            
        # Run judge multiple times
        scores = []
        attempt_results = []  # Store individual attempt results for CSV
        
        for attempt in range(self.n_attempts):
            try:
                score = await calculate_judge_score(
                    messages=messages,
                    dockerfile_contents=dockerfile_contents,
                    enable_io_logging=False,  # No logging needed for eval
                    llm_judge_name=self.model_name,
                    judge_version=self.judge_version
                )
                scores.append(score)
                attempt_results.append({
                    'attempt_num': attempt + 1,
                    'score': score,
                    'passed': expected_range[0] <= score <= expected_range[1]
                })
                
            except Exception as e:
                print(f"  ❌ Error running judge (attempt {attempt + 1}): {e}")
                scores.append(0.0)
                attempt_results.append({
                    'attempt_num': attempt + 1,
                    'score': 0.0,
                    'passed': False
                })
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Check if average score is in expected range
        passed = expected_range[0] <= avg_score <= expected_range[1]
        
        return test_id, avg_score, passed, scores, attempt_results
            
    async def run_all_tests(self, config_path: Path, test_filter: str = None):
        """Run all test cases from config."""
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        test_cases = config['test_cases']
        
        # Apply filter if specified
        if test_filter:
            test_cases = [tc for tc in test_cases if tc['id'] == test_filter]
            if not test_cases:
                print(f"No test case found with id: {test_filter}")
                return
                
        print(f"Running Judge Evaluation with model: {self.model_name}")
        
        # Show current and latest versions
        latest_version, version_info = ResultsStore.get_latest_judge_version()
        print(f"Judge version: {self.judge_version}", end="")
        if self.judge_version == "latest":
            print(f" (resolves to: {latest_version})")
        elif latest_version != "unknown" and self.judge_version != latest_version:
            print(f" (latest available: {latest_version})")
        else:
            print()
            
        print(f"Run ID: {self.store.get_run_id()}")
        print(f"{'='*60}\n")
        
        # Run each test case
        total = len(test_cases)
        passed = 0
        
        for i, test_case in enumerate(test_cases, 1):
            test_id = test_case['id']
            description = test_case['description']
            expected_range = test_case['expected_range']
            
            print(f"[{i}/{total}] {test_id}: {description}")
            
            # Check if test data exists
            case_dir = Path(__file__).parent / "judge_eval_cases" / test_id
            if not case_dir.exists():
                print(f"  ⚠️  Test data directory not found: {case_dir}")
                print(f"  Expected range: {expected_range[0]:.2f}-{expected_range[1]:.2f}")
                print()
                continue
                
            # Run test
            _, avg_score, is_passed, scores, attempt_results = await self.run_test_case(test_case)
            
            # Display result
            status = "✓" if is_passed else "✗"
            if self.n_attempts > 1:
                # Show statistics for multiple attempts
                min_score = min(scores)
                max_score = max(scores)
                std_dev = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5 if len(scores) > 1 else 0
                print(f"  Average Score: {avg_score:.2f} {status} (expected: {expected_range[0]:.2f}-{expected_range[1]:.2f})")
                print(f"  Attempts: {len(scores)} | Min: {min_score:.2f} | Max: {max_score:.2f} | Std Dev: {std_dev:.3f}")
                print(f"  Individual scores: {', '.join(f'{s:.2f}' for s in scores)}")
            else:
                print(f"  Score: {avg_score:.2f} {status} (expected: {expected_range[0]:.2f}-{expected_range[1]:.2f})")
            print()
            
            if is_passed:
                passed += 1
                
            self.results.append({
                'test_id': test_id,
                'score': avg_score,
                'passed': is_passed,
                'expected_range': expected_range,
                'scores': scores
            })
            
            # Store each attempt in CSV with task-id-n-of-n format
            for attempt_result in attempt_results:
                attempt_id = f"{test_id}-{attempt_result['attempt_num']}-of-{self.n_attempts}"
                self.store.add_result(
                    test_id=attempt_id,
                    judge_version=self.resolved_judge_version,
                    model_name=self.model_name,
                    score=attempt_result['score'],
                    expected_range=expected_range,
                    passed=attempt_result['passed']
                )
            
        # Summary
        print(f"{'='*60}")
        print(f"Summary: {passed}/{total} passed")
        
        # Failed tests details
        failed_tests = [r for r in self.results if not r['passed']]
        if failed_tests:
            print("\nFailed tests:")
            for result in failed_tests:
                print(f"  - {result['test_id']}: score {result['score']:.2f} not in range {result['expected_range']}")
                
        # Print CSV results info
        print(f"\nResults saved to: {self.store.csv_path}")
        print(f"Run ID: {self.store.get_run_id()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate judge scoring with different models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for judge evaluation"
    )
    parser.add_argument(
        "--test-id",
        type=str,
        help="Run only a specific test case by ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--judge-version",
        type=str,
        default="latest",
        help="Judge version to use (e.g., 'v4.0', 'latest'). Default: latest"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        help="Path to CSV output file (default: judge_eval_results.csv)"
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of attempts per test case (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
        
    # Set model in environment if not already set
    if "LLM_JUDGE_NAME" not in os.environ:
        os.environ["LLM_JUDGE_NAME"] = args.model
        
    # Run evaluation
    csv_path = Path(args.csv_output) if args.csv_output else None
    evaluator = JudgeEvaluator(args.model, args.judge_version, csv_path, args.attempts)
    asyncio.run(evaluator.run_all_tests(config_path, args.test_id))


if __name__ == "__main__":
    main()