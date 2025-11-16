#!/usr/bin/env python3
"""
Ablation Study Runner

Runs 4 configurations to scientifically test factors contributing to high score margins:
- Config A: Baseline (sims=150, tau=0.0)
- Config B: Temperature fix only (sims=150, tau=0.05 with extended schedule)
- Config C: MCTS fix only (sims=400, tau=0.0)
- Config D: Both fixes (sims=400, tau=0.05 with extended schedule)

Usage:
    python scripts/run_ablation_study.py --iterations 5 --config-id a
    python scripts/run_ablation_study.py --iterations 5 --config-id b
    python scripts/run_ablation_study.py --iterations 5 --config-id c
    python scripts/run_ablation_study.py --iterations 5 --config-id d
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_config(config_id, iterations):
    """Run training for a specific ablation config."""
    config_file = f"config_ablation_{config_id}.yaml"

    if not Path(config_file).exists():
        print(f"Error: {config_file} not found")
        return False

    print(f"\n{'='*80}")
    print(f"Running Ablation Config {config_id.upper()}")
    print(f"{'='*80}\n")

    print(f"Config file: {config_file}")
    print(f"Target iterations: {iterations}")
    print()

    start_time = time.time()

    try:
        # Run training
        cmd = [sys.executable, "scripts/self_play_train.py", "--config", config_file]

        # Set environment
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = '.'

        # Run until iterations complete (user will need to Ctrl+C or modify script for auto-stop)
        result = subprocess.run(cmd, env=env)

        elapsed = time.time() - start_time
        print(f"\nConfig {config_id.upper()} completed in {elapsed/60:.1f} minutes")

        return result.returncode == 0

    except KeyboardInterrupt:
        print(f"\nConfig {config_id.upper()} interrupted by user")
        return False
    except Exception as e:
        print(f"\nError running config {config_id.upper()}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run ablation study configuration")
    parser.add_argument('--config-id', type=str, required=True,
                       choices=['a', 'b', 'c', 'd'],
                       help='Which config to run (a/b/c/d)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Target number of iterations (default: 5)')

    args = parser.parse_args()

    success = run_config(args.config_id, args.iterations)

    if success:
        print("\n✅ Config run completed successfully")
    else:
        print("\n❌ Config run failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
