#!/usr/bin/env python3
"""Generate report from judge evaluation CSV results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import statistics


def load_csv_data(csv_path: Path) -> List[Dict]:
    """Load CSV data into list of dictionaries."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['score'] = float(row['score'])
            row['expected_min'] = float(row['expected_min'])
            row['expected_max'] = float(row['expected_max'])
            results.append(row)
    return results


def calculate_model_performance(results: List[Dict]) -> Dict:
    """Calculate performance metrics for each model."""
    model_stats = defaultdict(lambda: {
        'scores': [],
        'passes': 0,
        'fails': 0,
        'test_results': defaultdict(list)
    })
    
    for row in results:
        model = row['model_name']
        test_id_base = row['test_id'].split('-')[0]  # Remove attempt suffix
        
        model_stats[model]['scores'].append(row['score'])
        model_stats[model]['test_results'][test_id_base].append(row['score'])
        
        if row['pass_fail'] == 'PASS':
            model_stats[model]['passes'] += 1
        else:
            model_stats[model]['fails'] += 1
    
    # Calculate aggregate stats
    for model, stats in model_stats.items():
        scores = stats['scores']
        stats['total_tests'] = len(scores)
        stats['avg_score'] = statistics.mean(scores) if scores else 0
        stats['std_dev'] = statistics.stdev(scores) if len(scores) > 1 else 0
        stats['pass_rate'] = stats['passes'] / stats['total_tests'] if stats['total_tests'] > 0 else 0
        
        # Calculate per-test averages
        test_averages = {}
        for test_id, test_scores in stats['test_results'].items():
            test_averages[test_id] = statistics.mean(test_scores)
        stats['test_averages'] = test_averages
    
    return dict(model_stats)


def generate_report(csv_path: Path, judge_version: str = None):
    """Generate performance report from CSV data."""
    print(f"\n{'='*80}")
    print("Judge Evaluation Performance Report")
    print(f"{'='*80}\n")
    
    # Load data
    results = load_csv_data(csv_path)
    
    if not results:
        print("No results found in CSV file.")
        return
    
    # Filter by judge version if specified
    if judge_version:
        results = [r for r in results if r['judge_version'] == judge_version]
        print(f"Filter: Judge Version = {judge_version}")
    
    # Get unique judge versions
    judge_versions = set(r['judge_version'] for r in results)
    print(f"Judge Versions in Data: {', '.join(sorted(judge_versions))}")
    
    # Calculate performance
    model_performance = calculate_model_performance(results)
    
    # Sort models by pass rate (higher is better), then by average score (lower is better)
    sorted_models = sorted(
        model_performance.items(),
        key=lambda x: (x[1]['pass_rate'], -x[1]['avg_score']),
        reverse=True
    )
    
    print(f"\nModels Evaluated: {len(sorted_models)}")
    print(f"{'='*80}\n")
    
    # Model Performance Summary
    print("Model Performance Ranking:")
    print("(Note: Lower average scores indicate stricter/better judge performance)")
    print(f"{'Rank':<6}{'Model':<30}{'Pass Rate':<12}{'Avg Score':<12}{'Tests':<8}")
    print(f"{'-'*6}{'-'*30}{'-'*12}{'-'*12}{'-'*8}")
    
    for rank, (model, stats) in enumerate(sorted_models, 1):
        print(f"{rank:<6}{model:<30}{stats['pass_rate']:<12.2%}{stats['avg_score']:<12.2f}{stats['total_tests']:<8}")
    
    # Detailed Model Performance
    print(f"\n{'='*80}")
    print("Detailed Model Performance:")
    print(f"{'='*80}\n")
    
    for model, stats in sorted_models:
        print(f"\nModel: {model}")
        print(f"  Overall Pass Rate: {stats['pass_rate']:.2%} ({stats['passes']}/{stats['total_tests']})")
        print(f"  Average Score: {stats['avg_score']:.2f} (σ={stats['std_dev']:.3f})")
        
        # Test-level performance
        if stats['test_averages']:
            print(f"  Test Performance:")
            for test_id, avg_score in sorted(stats['test_averages'].items()):
                print(f"    {test_id}: {avg_score:.2f}")
    
    # Best model summary
    if sorted_models:
        best_model, best_stats = sorted_models[0]
        print(f"\n{'='*80}")
        print(f"Best Performing Model: {best_model}")
        print(f"  Pass Rate: {best_stats['pass_rate']:.2%}")
        print(f"  Average Score: {best_stats['avg_score']:.2f}")
        print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate report from judge evaluation results")
    parser.add_argument(
        "--csv",
        type=str,
        default="judge_eval_results.csv",
        help="Path to CSV file (default: judge_eval_results.csv)"
    )
    parser.add_argument(
        "--judge-version",
        type=str,
        help="Filter results by specific judge version"
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Try relative to script location
        csv_path = Path(__file__).parent / args.csv
        
    if not csv_path.exists():
        print(f"CSV file not found: {args.csv}")
        return
    
    generate_report(csv_path, args.judge_version)


if __name__ == "__main__":
    main()