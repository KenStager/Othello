#!/usr/bin/env python3
"""
Ablation Study Results Analyzer

Analyzes score margins from ablation study logs to determine which factors
contribute to high score margins.

Usage:
    python scripts/analyze_ablation_results.py
"""

import re
import glob
import os
from pathlib import Path
import statistics

def extract_score_margins(log_file):
    """Extract score margins from a training log."""
    margins = []

    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        for line in f:
            # Look for lines like: "Avg score margin: ±27.9"
            match = re.search(r'Avg score margin: ±([\d.]+)', line)
            if match:
                margin = float(match.group(1))
                margins.append(margin)

    return margins


def analyze_config(config_id, label):
    """Analyze results for a single config."""
    log_file = f"data/ablation_{config_id}/train.log"

    print(f"\n{'='*80}")
    print(f"Config {config_id.upper()}: {label}")
    print(f"{'='*80}")

    margins = extract_score_margins(log_file)

    if margins is None or len(margins) == 0:
        print("  ⚠️  No data found (log file missing or no iterations completed)")
        return None

    # Statistics
    mean_margin = statistics.mean(margins)
    median_margin = statistics.median(margins)
    stdev_margin = statistics.stdev(margins) if len(margins) > 1 else 0.0
    min_margin = min(margins)
    max_margin = max(margins)

    print(f"  Iterations analyzed: {len(margins)}")
    print(f"  Mean score margin: ±{mean_margin:.2f} discs")
    print(f"  Median score margin: ±{median_margin:.2f} discs")
    print(f"  Std dev: {stdev_margin:.2f}")
    print(f"  Range: [{min_margin:.2f}, {max_margin:.2f}]")
    print()

    # Show last 5 iterations
    if len(margins) >= 5:
        recent = margins[-5:]
        print(f"  Last 5 iterations: {', '.join(f'±{m:.1f}' for m in recent)}")
    else:
        print(f"  All iterations: {', '.join(f'±{m:.1f}' for m in margins)}")

    return {
        'config_id': config_id,
        'label': label,
        'mean': mean_margin,
        'median': median_margin,
        'stdev': stdev_margin,
        'count': len(margins),
        'margins': margins
    }


def main():
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS ANALYSIS")
    print("="*80)
    print("\nPurpose: Identify which factors contribute to high score margins")
    print("Baseline: ±27-28 discs (iteration 45)")
    print("Target: ±8-15 discs (healthy competitive games)")

    configs = [
        ('a', 'BASELINE (sims=150, tau=0.0)'),
        ('b', 'TEMPERATURE FIX ONLY (sims=150, tau=0.05 extended)'),
        ('c', 'MCTS FIX ONLY (sims=400, tau=0.0)'),
        ('d', 'BOTH FIXES (sims=400, tau=0.05 extended)')
    ]

    results = []
    for config_id, label in configs:
        result = analyze_config(config_id, label)
        if result:
            results.append(result)

    if len(results) < 2:
        print("\n⚠️  Insufficient data for comparison (need at least 2 configs)")
        return

    # Comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    # Sort by mean margin (lower is better)
    results_sorted = sorted(results, key=lambda x: x['mean'])

    print("Ranking (lower score margin = better):")
    print()
    for rank, result in enumerate(results_sorted, 1):
        improvement = ""
        if result['config_id'] == 'a':
            improvement = "(baseline)"
        else:
            # Find baseline
            baseline = next((r for r in results if r['config_id'] == 'a'), None)
            if baseline:
                reduction = ((baseline['mean'] - result['mean']) / baseline['mean']) * 100
                improvement = f"({reduction:+.1f}% vs baseline)"

        print(f"  {rank}. Config {result['config_id'].upper()}: ±{result['mean']:.2f} discs {improvement}")
        print(f"     {result['label']}")
        print()

    # Determine best fix
    print("="*80)
    print("CONCLUSIONS")
    print("="*80)
    print()

    baseline = next((r for r in results if r['config_id'] == 'a'), None)
    temp_only = next((r for r in results if r['config_id'] == 'b'), None)
    mcts_only = next((r for r in results if r['config_id'] == 'c'), None)
    both = next((r for r in results if r['config_id'] == 'd'), None)

    if baseline and temp_only:
        temp_effect = baseline['mean'] - temp_only['mean']
        print(f"Temperature fix contribution: {temp_effect:+.2f} discs ({temp_effect/baseline['mean']*100:.1f}%)")

    if baseline and mcts_only:
        mcts_effect = baseline['mean'] - mcts_only['mean']
        print(f"MCTS fix contribution: {mcts_effect:+.2f} discs ({mcts_effect/baseline['mean']*100:.1f}%)")

    if baseline and both:
        combined_effect = baseline['mean'] - both['mean']
        print(f"Combined fix contribution: {combined_effect:+.2f} discs ({combined_effect/baseline['mean']*100:.1f}%)")

    print()

    if mcts_only and temp_only and baseline:
        if mcts_only['mean'] < temp_only['mean']:
            print("✅ Primary cause: Weak MCTS (increasing simulations has larger impact)")
        else:
            print("✅ Primary cause: Aggressive temperature schedule (extending exploration has larger impact)")

    if both and baseline:
        if both['mean'] <= 15.0:
            print("✅ Target achieved: Score margins in healthy range (±8-15 discs)")
        elif both['mean'] <= 20.0:
            print("⚠️  Partial success: Score margins improved but still high (±15-20 discs)")
        else:
            print("❌ Target missed: Score margins still too high (>±20 discs)")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
