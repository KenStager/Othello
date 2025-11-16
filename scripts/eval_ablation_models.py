#!/usr/bin/env python3
"""
Evaluate Ablation Study Models

Loads the final trained model from each ablation config and plays fresh games
to measure actual performance (score margins).

Usage:
    PYTHONPATH=. python scripts/eval_ablation_models.py
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import glob
import re
from src.utils.config import load_config
from src.net.model import OthelloNet
from src.othello.game import Game
from src.train.selfplay import play_one_game

def find_latest_checkpoint(ckpt_dir):
    """Find the most recent checkpoint in a directory."""
    ckpts = glob.glob(os.path.join(ckpt_dir, 'current_iter*.pt'))
    if not ckpts:
        ckpts = glob.glob(os.path.join(ckpt_dir, 'champion_iter*.pt'))

    if not ckpts:
        return None, None

    # Extract iteration numbers and find max
    checkpoints_with_iters = []
    for path in ckpts:
        basename = os.path.basename(path)
        match = re.search(r'iter(\d+)\.pt', basename)
        if match:
            iteration = int(match.group(1))
            checkpoints_with_iters.append((path, iteration))

    if not checkpoints_with_iters:
        return None, None

    latest_path, latest_iter = max(checkpoints_with_iters, key=lambda x: x[1])
    return latest_path, latest_iter

def evaluate_config(config_name, cfg_file, ckpt_dir, num_games=20):
    """Evaluate a single ablation config by playing fresh games."""
    print(f"\n{'='*80}")
    print(f"Evaluating Config {config_name}")
    print(f"{'='*80}")

    # Load config
    cfg = load_config(cfg_file)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Find latest checkpoint
    ckpt_path, iteration = find_latest_checkpoint(ckpt_dir)
    if not ckpt_path:
        print(f"  ⚠️  No checkpoint found in {ckpt_dir}")
        return None

    print(f"  Loading checkpoint: {os.path.basename(ckpt_path)} (iteration {iteration})")

    # Load model
    net = OthelloNet(
        in_channels=4,
        channels=cfg['model']['channels'],
        residual_blocks=cfg['model']['residual_blocks']
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_state = checkpoint.get('model_state', checkpoint)
    net.load_state_dict(model_state)
    net.eval()

    # Get MCTS config
    mcts_cfg = {
        'cpuct': cfg['mcts']['cpuct'],
        'simulations': cfg['mcts']['simulations'],
        'reuse_tree': cfg['mcts'].get('reuse_tree', True),
        'tt_enabled': False,  # Disable for evaluation
        'batch_size': cfg['mcts'].get('batch_size', 64),
        'use_batching': cfg['mcts'].get('use_batching', True)
    }

    temp_schedule = cfg.get('selfplay', {}).get('temp_schedule', {
        'open_to': 12,
        'mid_to': 20,
        'open_tau': 1.0,
        'mid_tau': 0.25,
        'late_tau': 0.0
    })

    print(f"  MCTS: {mcts_cfg['simulations']} simulations")
    print(f"  Temperature: τ={temp_schedule.get('late_tau', 0.0)} (late game)")
    print(f"  Playing {num_games} games...\n")

    # Play games
    margins = []
    lengths = []

    for game_num in range(num_games):
        traj, meta = play_one_game(
            game_cls=Game,
            net=net,
            device=device,
            mcts_cfg=mcts_cfg,
            temp_schedule=temp_schedule,
            max_moves=120,
            dir_alpha=cfg['game']['dirichlet_alpha'],
            dir_frac=cfg['game']['dirichlet_frac']
        )

        margin = abs(meta['score_diff'])
        margins.append(margin)
        lengths.append(meta['length'])

        print(f"    Game {game_num+1}/{num_games}: {meta['winner']:+d} score, "
              f"margin ±{margin} discs, {meta['length']} moves")

    # Calculate statistics
    mean_margin = sum(margins) / len(margins)
    median_margin = sorted(margins)[len(margins)//2]
    min_margin = min(margins)
    max_margin = max(margins)
    mean_length = sum(lengths) / len(lengths)

    print(f"\n  Results:")
    print(f"    Mean margin: ±{mean_margin:.1f} discs")
    print(f"    Median margin: ±{median_margin:.1f} discs")
    print(f"    Range: [{min_margin}, {max_margin}] discs")
    print(f"    Avg game length: {mean_length:.1f} moves")

    return {
        'config': config_name,
        'iteration': iteration,
        'mean_margin': mean_margin,
        'median_margin': median_margin,
        'min_margin': min_margin,
        'max_margin': max_margin,
        'mean_length': mean_length,
        'margins': margins
    }

def main():
    configs = {
        'A': {
            'name': 'BASELINE (150 sims, aggressive temp)',
            'cfg_file': 'config_ablation_a.yaml',
            'ckpt_dir': 'data/ablation_a/checkpoints'
        },
        'B': {
            'name': 'STANDARD MCTS (200 sims, aggressive temp)',
            'cfg_file': 'config_ablation_b.yaml',
            'ckpt_dir': 'data/ablation_b/checkpoints'
        },
        'C': {
            'name': 'TEMP FIX (200 sims, gentle temp)',
            'cfg_file': 'config_ablation_c.yaml',
            'ckpt_dir': 'data/ablation_c/checkpoints'
        },
        'D': {
            'name': 'PROGRESSIVE MCTS + TEMP (300 sims, gentle temp)',
            'cfg_file': 'config_ablation_d.yaml',
            'ckpt_dir': 'data/ablation_d/checkpoints'
        }
    }

    print("\n" + "="*80)
    print("ABLATION STUDY EVALUATION")
    print("="*80)
    print("\nEvaluating trained models by playing fresh games...")
    print("Target: Reduce score margins from ±27-28 discs to ±8-15 discs\n")

    results = []
    for config_id, config_info in configs.items():
        result = evaluate_config(
            config_name=f"{config_id} - {config_info['name']}",
            cfg_file=config_info['cfg_file'],
            ckpt_dir=config_info['ckpt_dir'],
            num_games=20
        )
        if result:
            results.append(result)

    # Comparative analysis
    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*80}\n")

        # Sort by mean margin (lower is better)
        results_sorted = sorted(results, key=lambda x: x['mean_margin'])

        print("Ranking (lower score margin = better):\n")
        for rank, result in enumerate(results_sorted, 1):
            config_letter = result['config'].split()[0]
            print(f"  {rank}. Config {config_letter}: ±{result['mean_margin']:.1f} discs")
            print(f"     {result['config']}")
            print()

        # Isolate factor contributions
        print("="*80)
        print("FACTOR ANALYSIS")
        print("="*80)
        print()

        # Find specific configs
        config_a = next((r for r in results if r['config'].startswith('A')), None)
        config_b = next((r for r in results if r['config'].startswith('B')), None)
        config_c = next((r for r in results if r['config'].startswith('C')), None)
        config_d = next((r for r in results if r['config'].startswith('D')), None)

        if config_a and config_b:
            mcts_effect = config_a['mean_margin'] - config_b['mean_margin']
            pct_improvement = (mcts_effect / config_a['mean_margin']) * 100
            print(f"MCTS improvement (150→200 sims):")
            print(f"  {mcts_effect:+.1f} discs ({pct_improvement:+.1f}%)")
            print()

        if config_b and config_c:
            temp_effect = config_b['mean_margin'] - config_c['mean_margin']
            pct_improvement = (temp_effect / config_b['mean_margin']) * 100
            print(f"Temperature fix effect (at 200 sims):")
            print(f"  {temp_effect:+.1f} discs ({pct_improvement:+.1f}%)")
            print()

        if config_c and config_d:
            progressive_effect = config_c['mean_margin'] - config_d['mean_margin']
            pct_improvement = (progressive_effect / config_c['mean_margin']) * 100
            print(f"Progressive MCTS (200→300 sims with gentle temp):")
            print(f"  {progressive_effect:+.1f} discs ({pct_improvement:+.1f}%)")
            print()

        if config_a and config_d:
            total_effect = config_a['mean_margin'] - config_d['mean_margin']
            pct_improvement = (total_effect / config_a['mean_margin']) * 100
            print(f"Combined effect (baseline → config D):")
            print(f"  {total_effect:+.1f} discs ({pct_improvement:+.1f}%)")
            print()

        # Recommendation
        print("="*80)
        print("RECOMMENDATION")
        print("="*80)
        print()

        best_config = results_sorted[0]
        if best_config['mean_margin'] <= 15.0:
            print(f"✅ SUCCESS: Config {best_config['config'].split()[0]} achieved target")
            print(f"   Score margins in healthy range (±{best_config['mean_margin']:.1f} discs)")
        elif best_config['mean_margin'] <= 20.0:
            print(f"⚠️  PARTIAL: Config {best_config['config'].split()[0]} improved but not at target")
            print(f"   Score margins: ±{best_config['mean_margin']:.1f} discs (target: ±8-15)")
        else:
            print(f"❌ INSUFFICIENT: Best config only reached ±{best_config['mean_margin']:.1f} discs")
            print(f"   May need further investigation (400+ sims, network capacity, etc.)")

        print()
        print("Recommended configuration for main training:")
        print(f"  {best_config['config']}")
        print()

if __name__ == "__main__":
    main()
