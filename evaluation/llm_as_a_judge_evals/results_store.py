"""Results storage for judge evaluations."""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ResultsStore:
    """Store evaluation results in CSV format."""
    
    def __init__(self, csv_path: Optional[Path] = None):
        """Initialize results store.
        
        Args:
            csv_path: Path to CSV file. Defaults to 'judge_eval_results.csv'
        """
        self.csv_path = csv_path or Path(__file__).parent / "judge_eval_results.csv"
        self.run_id = self._generate_run_id()
        self._ensure_csv_exists()
        
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
        
    def _ensure_csv_exists(self):
        """Ensure CSV file exists with headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'run_id',
                    'test_id', 
                    'judge_version',
                    'model_name',
                    'score',
                    'expected_min',
                    'expected_max',
                    'pass_fail',
                    'timestamp'
                ])
    
    def add_result(self, 
                   test_id: str,
                   judge_version: str,
                   model_name: str,
                   score: float,
                   expected_range: List[float],
                   passed: bool):
        """Add a single result to the CSV."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.run_id,
                test_id,
                judge_version,
                model_name,
                score,
                expected_range[0],
                expected_range[1],
                'PASS' if passed else 'FAIL',
                datetime.now().isoformat()
            ])
    
    def get_run_id(self) -> str:
        """Get the current run ID."""
        return self.run_id
    
    def get_results_summary(self) -> Dict:
        """Get summary of current run results."""
        results = []
        
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['run_id'] == self.run_id:
                    results.append(row)
        
        if not results:
            return {'total': 0, 'passed': 0, 'failed': 0}
            
        passed = sum(1 for r in results if r['pass_fail'] == 'PASS')
        failed = sum(1 for r in results if r['pass_fail'] == 'FAIL')
        
        return {
            'run_id': self.run_id,
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(results) if results else 0
        }
    
    @staticmethod
    def get_latest_judge_version() -> Tuple[str, Dict]:
        """Get the latest judge version from versions.json.
        
        Returns:
            Tuple of (version_string, version_info)
        """
        judge_prompts_dir = Path(__file__).parent.parent.parent / "src" / "tbench_rllm" / "rewards" / "judge_prompts"
        versions_file = judge_prompts_dir / "versions.json"
        
        if not versions_file.exists():
            return "unknown", {}
            
        with open(versions_file, "r") as f:
            versions_data = json.load(f)
            
        latest_version = versions_data.get("latest", "unknown")
        version_info = versions_data.get("versions", {}).get(latest_version, {})
        
        return latest_version, version_info