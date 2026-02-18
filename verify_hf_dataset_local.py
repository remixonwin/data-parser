#!/usr/bin/env python3
"""
Verifier Runner Script.

Runs verification commands for HF dataset and quality checks.
Captures stdout/stderr and reports pass/fail status.

Usage:
    python verify_hf_dataset_local.py [--dataset-path PATH]
    
Environment:
    PYTHONPATH is automatically set to include the project root.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup paths
PROJECT_ROOT = Path(__file__).parent
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


class VerifierRunner:
    """Runs verification commands and captures results."""
    
    def __init__(self, dataset_path: Optional[Path] = None, repo_id: str = "Remixonwin/prepware_study_guide-dataset"):
        self.dataset_path = dataset_path
        self.repo_id = repo_id
        self.results: List[Dict] = []
        
    def run_command(
        self, 
        command: List[str], 
        env: Optional[Dict] = None,
        cwd: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """Run a command and capture stdout/stderr.
        
        Args:
            command: Command as list of strings
            env: Optional environment variables
            cwd: Optional working directory
            
        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        
        # Ensure PYTHONPATH is set
        run_env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=run_env,
                cwd=cwd or PROJECT_ROOT,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out after 5 minutes"
        except Exception as e:
            return -1, "", str(e)
    
    def run_verification(self, name: str, command: List[str], cwd: Optional[Path] = None) -> Dict:
        """Run a single verification command.
        
        Args:
            name: Name of the verification
            command: Command to run
            cwd: Optional working directory
            
        Returns:
            Dict with verification results
        """
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        exit_code, stdout, stderr = self.run_command(command, cwd=cwd)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Determine pass/fail
        passed = exit_code == 0
        
        result = {
            "name": name,
            "command": " ".join(command),
            "exit_code": exit_code,
            "passed": passed,
            "duration_seconds": duration,
            "stdout": stdout,
            "stderr": stderr,
            "timestamp": start_time.isoformat()
        }
        
        print(f"Exit Code: {exit_code}")
        print(f"Duration: {duration:.2f}s")
        print(f"Status: {'PASSED' if passed else 'FAILED'}")
        
        if stdout:
            print(f"\nStdout (last 1000 chars):")
            print(stdout[-1000:])
        
        if stderr:
            print(f"\nStderr (last 1000 chars):")
            print(stderr[-1000:])
        
        return result
    
    def run_all_verifications(self) -> Dict:
        """Run all verification commands.
        
        Returns:
            Dict with all results
        """
        print("="*60)
        print("HF Dataset Verification Runner")
        print("="*60)
        print(f"Dataset Path: {self.dataset_path or 'default'}")
        print(f"Repo ID: {self.repo_id}")
        print(f"Time: {datetime.now().isoformat()}")
        
        # List of verifications to run
        verifications = []
        
        # 1. Verify HF dataset can be loaded from local path
        if self.dataset_path and self.dataset_path.exists():
            verifications.append({
                "name": "HF Dataset Local Load Verification",
                "command": [
                    sys.executable, "-c",
                    f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from datasets import load_from_disk
import os
os.chdir('{self.dataset_path}')
ds = load_from_disk('data')
print(f"Loaded dataset with splits: {{list(ds.keys())}}")
print(f"Total examples: {{sum(len(ds[s]) for s in ds)}}")
"""
                ],
                "cwd": self.dataset_path
            })
        
        # 2. Run tests/verify_hf_dataset.py
        verifications.append({
            "name": "HF Dataset Hub Verification",
            "command": [sys.executable, "tests/verify_hf_dataset.py"],
            "cwd": PROJECT_ROOT
        })
        
        # 3. Run verify_dataset_quality.py (if dataset_path is provided, modify to use it)
        quality_cmd = [sys.executable, "verify_dataset_quality.py"]
        verifications.append({
            "name": "Dataset Quality Verification",
            "command": quality_cmd,
            "cwd": PROJECT_ROOT
        })
        
        # Run all verifications
        for v in verifications:
            result = self.run_verification(
                name=v["name"],
                command=v["command"],
                cwd=v.get("cwd")
            )
            self.results.append(result)
        
        # Generate summary
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        total_duration = sum(r["duration_seconds"] for r in self.results)
        
        summary = {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "total_duration_seconds": total_duration,
            "all_passed": failed == 0,
            "results": self.results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("VERIFICATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"\nOverall: {'ALL PASSED' if summary['all_passed'] else 'SOME FAILED'}")
        
        return summary
    
    def save_results(self, output_path: Path):
        """Save verification results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "total_duration_seconds": sum(r["duration_seconds"] for r in self.results),
            "all_passed": all(r["passed"] for r in self.results),
            "timestamp": datetime.now().isoformat(),
            "repo_id": self.repo_id,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "results": [
                {
                    "name": r["name"],
                    "command": r["command"],
                    "exit_code": r["exit_code"],
                    "passed": r["passed"],
                    "duration_seconds": r["duration_seconds"],
                    "timestamp": r["timestamp"],
                    # Truncate output for JSON
                    "stdout_truncated": r["stdout"][:5000] if r["stdout"] else "",
                    "stderr_truncated": r["stderr"][:5000] if r["stderr"] else "",
                }
                for r in self.results
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run HF dataset verifications")
    parser.add_argument(
        "--dataset-path", 
        type=Path,
        help="Path to local dataset directory"
    )
    parser.add_argument(
        "--repo-id", 
        default="Remixonwin/prepware_study_guide-dataset",
        help="HuggingFace dataset repo ID"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save verification results JSON"
    )
    
    args = parser.parse_args()
    
    runner = VerifierRunner(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id
    )
    
    summary = runner.run_all_verifications()
    
    # Save results if requested
    if args.output:
        runner.save_results(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
