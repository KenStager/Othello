#!/usr/bin/env python3
"""
Auxiliary Weight Tuning Script

Tests different weight configurations for auxiliary loss heads to find
optimal balance between multi-task learning and primary objectives.

Current weights:
- mobility: 0.2
- stability: 0.2
- corner: 0.1
- parity: 0.1

Usage:
    python scripts/tune_aux_weights.py --config config.yaml --iterations 15

The script will:
1. Train models with 4 different weight schemes (10-15 iterations each)
2. Evaluate each model against baseline (100 games)
3. Report results and recommend best configuration
"""

import argparse
import os
import sys
import copy
import torch
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.othello.game import Game
from src.net.model import OthelloNet
from src.train.replay import ReplayBuffer
from src.train.selfplay import generate_selfplay
from src.train.trainer import train_steps
from src.train.evaluator import play_match


# Define weight schemes to test
WEIGHT_SCHEMES = {
    'current': {
        'name': 'Current (Baseline)',
        'description': 'Current weights from config',
        'weights': (0.2, 0.2, 0.1, 0.1),  # mobility, stability, corner, parity
    },
    'balanced': {
        'name': 'Balanced',
        'description': 'Equal weight to all auxiliary heads',
        'weights': (0.15, 0.15, 0.15, 0.15),
    },
    'othello_focused': {
        'name': 'Othello-Focused',
        'description': 'Emphasize corners and stability (key Othello concepts)',
        'weights': (0.1, 0.3, 0.25, 0.05),
    },
    'minimal': {
        'name': 'Minimal',
        'description': 'Reduce auxiliary influence, focus on policy/value',
        'weights': (0.1, 0.1, 0.05, 0.05),
    }
}


def setup_device(cfg):
    """Setup device (MPS/CUDA/CPU)."""
    device_cfg = cfg['device']

    if torch.cuda.is_available() and device_cfg == "cuda":
        return torch.device("cuda")

    if torch.backends.mps.is_available() and device_cfg == "mps":
        device = torch.device("mps")
        torch.mps.set_per_process_memory_fraction(0.75)
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        return device

    return torch.device("cpu")


def train_with_weights(cfg, device, weight_scheme, iterations, verbose=True):
    """
    Train a model with specific auxiliary weights.

    Args:
        cfg: Configuration dict
        device: Torch device
        weight_scheme: Dict with 'weights' tuple (mobility, stability, corner, parity)
        iterations: Number of training iterations
        verbose: Print progress

    Returns:
        Trained network
    """
    weights = weight_scheme['weights']
    name = weight_scheme['name']

    if verbose:
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"Description: {weight_scheme['description']}")
        print(f"Weights: mobility={weights[0]:.2f}, stability={weights[1]:.2f}, "
              f"corner={weights[2]:.2f}, parity={weights[3]:.2f}")
        print(f"{'='*80}\n")

    # Create fresh model
    net = OthelloNet(
        in_channels=4,
        channels=cfg['model']['channels'],
        residual_blocks=cfg['model']['residual_blocks']
    ).to(device)

    # Create fresh replay buffer
    replay = ReplayBuffer(
        capacity=cfg['train']['replay_capacity'],
        save_dir=None  # Don't save during tuning
    )

    game_cls = lambda: Game(cfg['game']['board_size'])
    mcts_cfg = {
        'cpuct': cfg['mcts']['cpuct'],
        'simulations': cfg['mcts']['simulations'],
        'dir_alpha': cfg['game']['dirichlet_alpha'],
        'dir_frac': cfg['game']['dirichlet_frac'],
        'reuse_tree': cfg['mcts']['reuse_tree'],
        'tt_enabled': cfg['mcts'].get('tt_enabled', False),
        'batch_size': cfg['mcts'].get('batch_size', 32),
        'use_batching': cfg['mcts'].get('use_batching', True)
    }

    temp_schedule = cfg.get('selfplay', {}).get('temp_schedule', {
        'open_to': 12,
        'mid_to': 20,
        'open_tau': 1.0,
        'mid_tau': 0.25,
        'late_tau': 0.0
    })

    # Training loop
    for it in range(1, iterations + 1):
        if verbose:
            print(f"  Iteration {it}/{iterations}")

        # Self-play (reduced games for faster tuning)
        games_per_iter = min(20, cfg['selfplay']['games_per_iter'])
        generate_selfplay(
            replay=replay,
            game_cls=game_cls,
            net=net,
            device=device,
            mcts_cfg=mcts_cfg,
            games=games_per_iter,
            temp_schedule=temp_schedule,
            max_moves=cfg['selfplay']['max_moves'],
            dir_alpha=cfg['game']['dirichlet_alpha'],
            dir_frac=cfg['game']['dirichlet_frac'],
            verbose=False  # Suppress detailed output
        )

        # Train (if enough data)
        if replay.size() >= cfg['train']['min_replay_to_train']:
            # Reduced training steps for faster tuning
            train_steps_actual = min(100, cfg['train']['steps_per_iter'])

            # Modified trainer to use custom weights
            avg_loss = train_steps_with_custom_weights(
                net=net,
                replay=replay,
                device=device,
                steps=train_steps_actual,
                batch_size=cfg['train']['batch_size'],
                lr=cfg['train']['lr'],
                lr_min=cfg['train']['lr_min'],
                weight_decay=cfg['train']['weight_decay'],
                grad_clip=cfg['train']['grad_clip'],
                aux_weights=weights,
                verbose=False
            )

            if verbose:
                print(f"    Loss: {avg_loss:.4f}, Buffer: {replay.size()}")

    if verbose:
        print(f"\n  Training complete: {name}\n")

    return net


def train_steps_with_custom_weights(net, replay, device, steps, batch_size, lr, lr_min,
                                    weight_decay, grad_clip, aux_weights, verbose=False):
    """
    Modified train_steps that accepts custom auxiliary weights.

    Args:
        aux_weights: Tuple of (mobility_w, stability_w, corner_w, parity_w)
    """
    if replay.size() == 0:
        return 0.0

    from torch.utils.data import Dataset, DataLoader
    from src.train.trainer import ReplayDataset
    import torch.nn as nn
    import torch.optim as optim

    mobility_w, stability_w, corner_w, parity_w = aux_weights

    ds = ReplayDataset(replay)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    value_loss_fn = nn.MSELoss(reduction='mean')
    bce_loss_fn = nn.BCELoss()

    net.train()
    total_loss = 0.0

    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        (s, pi, z_win, z_score, mobility, stability, corner, parity, empties, phases) = batch

        s = s.to(device=device, dtype=torch.float32)
        pi = pi.to(device=device, dtype=torch.float32)
        z_win = z_win.to(device=device)
        z_score = z_score.to(device=device)
        mobility = mobility.to(device=device, dtype=torch.float32)
        stability = stability.to(device=device, dtype=torch.float32)
        corner = corner.to(device=device, dtype=torch.float32)
        parity = parity.to(device=device, dtype=torch.float32)

        outputs = net(s)

        # Policy loss
        logp = torch.log_softmax(outputs.policy_logits, dim=1)
        pol_loss = -(pi * logp).sum(dim=1).mean()

        # Value losses
        val_loss = value_loss_fn(outputs.value_win, z_win)
        score_loss = 0.3 * value_loss_fn(outputs.value_score, z_score)

        # Auxiliary losses (with custom weights)
        mobility_loss = bce_loss_fn(outputs.mobility, mobility)
        stability_loss = bce_loss_fn(outputs.stability_map, stability)
        corner_loss = bce_loss_fn(outputs.corner, corner)
        parity_loss = bce_loss_fn(outputs.parity, parity)

        # Total loss with custom weights
        loss = (
            pol_loss + val_loss + score_loss
            + mobility_w * mobility_loss
            + stability_w * stability_loss
            + corner_w * corner_loss
            + parity_w * parity_loss
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item()

        # MPS memory management
        if device.type == "mps" and (step + 1) % 100 == 0:
            torch.mps.empty_cache()

    return total_loss / max(1, steps)


def evaluate_model(net, baseline, device, cfg, num_games=100, verbose=True):
    """
    Evaluate trained model against baseline.

    Returns:
        Dict with win rate, loss rate, draw rate
    """
    if verbose:
        print(f"  Evaluating: {num_games} games...")

    game_cls = lambda: Game(cfg['game']['board_size'])

    # Use fewer simulations for faster evaluation
    eval_sims = max(100, cfg['mcts']['simulations'] // 2)

    results = play_match(
        net1=net,
        net2=baseline,
        game_cls=game_cls,
        device=device,
        num_games=num_games,
        simulations=eval_sims,
        verbose=False
    )

    if verbose:
        print(f"    Net1 (tested): {results['net1_wins']} wins")
        print(f"    Net2 (baseline): {results['net2_wins']} wins")
        print(f"    Draws: {results['draws']}")
        print(f"    Win rate: {results['net1_win_rate']:.1%}")

    return results


def main(args):
    """Main tuning workflow."""
    print("\n" + "="*80)
    print("AUXILIARY WEIGHT TUNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Iterations per scheme: {args.iterations}")
    print(f"  Evaluation games: {args.eval_games}")
    print(f"  Schemes to test: {len(WEIGHT_SCHEMES)}")
    print()

    # Load config
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = setup_device(cfg)

    print(f"Device: {device}")
    print()

    # Train baseline model (current weights)
    print("="*80)
    print("STEP 1: Train baseline model")
    print("="*80)

    baseline_scheme = WEIGHT_SCHEMES['current']
    baseline_net = train_with_weights(
        cfg, device, baseline_scheme, args.iterations, verbose=True
    )

    # Train and evaluate each scheme
    results = {}

    for scheme_name, scheme in WEIGHT_SCHEMES.items():
        if scheme_name == 'current':
            # Already trained as baseline
            results[scheme_name] = {
                'scheme': scheme,
                'net': baseline_net,
                'vs_baseline': {'net1_win_rate': 0.5, 'draws': 0}  # Self-play would be 50%
            }
            continue

        print("="*80)
        print(f"STEP 2.{len(results)}: Train and evaluate '{scheme['name']}'")
        print("="*80)

        # Train model
        net = train_with_weights(cfg, device, scheme, args.iterations, verbose=True)

        # Evaluate vs baseline
        print(f"\n  Evaluating vs baseline...")
        eval_results = evaluate_model(
            net, baseline_net, device, cfg,
            num_games=args.eval_games,
            verbose=True
        )

        results[scheme_name] = {
            'scheme': scheme,
            'net': net,
            'vs_baseline': eval_results
        }

        print()

    # Print summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()

    # Sort by win rate
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1]['vs_baseline']['net1_win_rate'],
        reverse=True
    )

    print(f"{'Scheme':<20} {'Win Rate':<12} {'Draws':<8} {'Weights (M/S/C/P)'}")
    print("-" * 80)

    for scheme_name, data in sorted_results:
        scheme = data['scheme']
        win_rate = data['vs_baseline']['net1_win_rate']
        draws = data['vs_baseline'].get('draws', 0)
        weights = scheme['weights']

        marker = "🏆" if scheme_name == sorted_results[0][0] else "  "
        print(f"{marker} {scheme['name']:<18} {win_rate:>6.1%}      {draws:<8} "
              f"{weights[0]:.2f} / {weights[1]:.2f} / {weights[2]:.2f} / {weights[3]:.2f}")

    print()

    # Recommendation
    best_scheme_name = sorted_results[0][0]
    best_scheme = sorted_results[0][1]['scheme']
    best_win_rate = sorted_results[0][1]['vs_baseline']['net1_win_rate']

    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    if best_scheme_name == 'current':
        print("✅ Keep current weights (no improvement found)")
        print()
        print("Current weights are already optimal for this configuration.")
    else:
        print(f"✅ Switch to '{best_scheme['name']}' weights")
        print()
        print(f"Description: {best_scheme['description']}")
        print(f"Win rate vs current: {best_win_rate:.1%}")
        print()
        print("Update config.yaml or trainer.py:")
        print("```python")
        print("loss = (")
        print("    pol_loss + val_loss + score_loss")
        weights = best_scheme['weights']
        print(f"    + {weights[0]:.2f} * mobility_loss")
        print(f"    + {weights[1]:.2f} * stability_loss")
        print(f"    + {weights[2]:.2f} * corner_loss")
        print(f"    + {weights[3]:.2f} * parity_loss")
        print(")")
        print("```")

    print()
    print("="*80)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune auxiliary head weights")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--iterations', type=int, default=15,
                       help='Training iterations per scheme (default: 15)')
    parser.add_argument('--eval-games', type=int, default=100,
                       help='Evaluation games per scheme (default: 100)')

    args = parser.parse_args()
    main(args)
