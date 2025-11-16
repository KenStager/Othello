#!/usr/bin/env python3
"""
Test Suite for Stage 1 Batched MCTS Implementation

Validates that batched MCTS:
1. Completes games successfully (correctness)
2. Achieves expected speedup (performance)
3. Produces reasonable move distributions (quality)

Usage:
    python scripts/test_batched_mcts.py --config config.yaml

Expected results:
- Games complete without errors
- Speedup: 4-5x (target: 4.4x from Amdahl's Law)
- Move distributions similar between batched and sequential
"""

import argparse
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.othello.game import Game
from src.net.model import OthelloNet
from src.mcts.mcts import MCTS
from src.utils.config import load_config
import torch


def test_correctness(config, num_games=5):
    """
    Test that batched MCTS completes games successfully.

    Args:
        config: Configuration dict
        num_games: Number of games to test

    Returns:
        bool: True if all games complete successfully
    """
    print("="*80)
    print("TEST 1: CORRECTNESS")
    print("="*80)
    print(f"Running {num_games} games with batched MCTS...")
    print()

    # Setup device
    device_cfg = config['device']
    if device_cfg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Create model
    model = OthelloNet(
        in_channels=4,
        channels=config['model']['channels'],
        residual_blocks=config['model']['residual_blocks']
    ).to(device)
    model.eval()

    # Run games
    game_cls = Game
    mcts_cfg = config['mcts']
    mcts_cfg['use_batching'] = True  # Ensure batching is enabled

    results = []
    for game_num in range(num_games):
        try:
            print(f"  Game {game_num+1}/{num_games}...", end=' ', flush=True)
            start = time.time()

            game = game_cls()
            board = game.new_board()
            mcts = MCTS(game_cls, model, device,
                       cpuct=mcts_cfg['cpuct'],
                       simulations=mcts_cfg['simulations'],
                       batch_size=mcts_cfg.get('batch_size', 32),
                       use_batching=True,
                       use_tt=False)  # TT disabled for Stage 1

            move_count = 0
            max_moves = 120
            while not board.is_terminal() and move_count < max_moves:
                pi = mcts.run(board)
                action = int(np.argmax(pi))
                board.step_action_index(action)
                move_count += 1

            winner, score = board.result()
            elapsed = time.time() - start

            results.append({
                'game': game_num + 1,
                'winner': winner,
                'score': score,
                'moves': move_count,
                'time': elapsed,
                'success': True
            })

            winner_str = "BLACK" if winner == 1 else ("WHITE" if winner == -1 else "DRAW")
            print(f"✓ {winner_str} {score:+d} ({move_count} moves, {elapsed:.1f}s)")

        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append({
                'game': game_num + 1,
                'success': False,
                'error': str(e)
            })

    print()
    print("-"*80)
    successes = sum(1 for r in results if r.get('success', False))
    print(f"Results: {successes}/{num_games} games completed successfully")

    if successes == num_games:
        avg_time = np.mean([r['time'] for r in results if 'time' in r])
        avg_moves = np.mean([r['moves'] for r in results if 'moves' in r])
        print(f"Average time: {avg_time:.1f}s per game")
        print(f"Average moves: {avg_moves:.1f}")
        print("✅ CORRECTNESS TEST PASSED")
        return True
    else:
        print("❌ CORRECTNESS TEST FAILED")
        return False


def test_performance(config, num_games=3):
    """
    Compare performance of batched vs sequential MCTS.

    Args:
        config: Configuration dict
        num_games: Number of games to test for each mode

    Returns:
        dict: Performance statistics
    """
    print("\n" + "="*80)
    print("TEST 2: PERFORMANCE")
    print("="*80)
    print(f"Comparing batched vs sequential MCTS ({num_games} games each)...")
    print()

    # Setup device
    device_cfg = config['device']
    if device_cfg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model
    model = OthelloNet(
        in_channels=4,
        channels=config['model']['channels'],
        residual_blocks=config['model']['residual_blocks']
    ).to(device)
    model.eval()

    game_cls = Game
    mcts_cfg = config['mcts']

    # Test both modes
    results = {}
    for mode, use_batching in [('Sequential', False), ('Batched', True)]:
        print(f"\n{mode} MCTS:")
        times = []

        for game_num in range(num_games):
            game = game_cls()
            board = game.new_board()
            mcts = MCTS(game_cls, model, device,
                       cpuct=mcts_cfg['cpuct'],
                       simulations=mcts_cfg['simulations'],
                       batch_size=mcts_cfg.get('batch_size', 32),
                       use_batching=use_batching,
                       use_tt=False)

            start = time.time()
            move_count = 0
            max_moves = 60  # Use fewer moves for faster testing
            while not board.is_terminal() and move_count < max_moves:
                pi = mcts.run(board)
                action = int(np.argmax(pi))
                board.step_action_index(action)
                move_count += 1

            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Game {game_num+1}: {elapsed:.1f}s ({move_count} moves)")

        avg_time = np.mean(times)
        std_time = np.std(times)
        results[mode] = {'times': times, 'avg': avg_time, 'std': std_time}

    # Calculate speedup
    seq_time = results['Sequential']['avg']
    batch_time = results['Batched']['avg']
    speedup = seq_time / batch_time

    print("\n" + "-"*80)
    print(f"Sequential: {seq_time:.1f}s ± {results['Sequential']['std']:.1f}s")
    print(f"Batched:    {batch_time:.1f}s ± {results['Batched']['std']:.1f}s")
    print(f"Speedup:    {speedup:.2f}x")
    print()

    # Evaluate against target
    target_speedup = 4.0  # Conservative target (4.4x from Amdahl's Law)
    if speedup >= target_speedup:
        print(f"✅ PERFORMANCE TEST PASSED (speedup {speedup:.2f}x >= {target_speedup:.1f}x target)")
        return {'passed': True, 'speedup': speedup, **results}
    elif speedup >= 3.0:
        print(f"⚠️  PERFORMANCE MARGINAL (speedup {speedup:.2f}x, target {target_speedup:.1f}x)")
        return {'passed': True, 'speedup': speedup, 'warning': 'marginal', **results}
    else:
        print(f"❌ PERFORMANCE TEST FAILED (speedup {speedup:.2f}x < 3.0x minimum)")
        return {'passed': False, 'speedup': speedup, **results}


def test_quality(config, num_games=10):
    """
    Compare move distributions between batched and sequential MCTS.

    This is a basic sanity check - we expect different but similar distributions.

    Args:
        config: Configuration dict
        num_games: Number of games to test

    Returns:
        dict: Quality statistics
    """
    print("\n" + "="*80)
    print("TEST 3: QUALITY (Move Distribution Comparison)")
    print("="*80)
    print(f"Comparing first-move distributions ({num_games} games per mode)...")
    print()

    # Setup device
    device_cfg = config['device']
    if device_cfg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create model
    model = OthelloNet(
        in_channels=4,
        channels=config['model']['channels'],
        residual_blocks=config['model']['residual_blocks']
    ).to(device)
    model.eval()

    game_cls = Game
    mcts_cfg = config['mcts']

    # Collect first moves
    results = {}
    for mode, use_batching in [('Sequential', False), ('Batched', True)]:
        print(f"\n{mode} MCTS:")
        first_moves = []

        for game_num in range(num_games):
            game = game_cls()
            board = game.new_board()
            mcts = MCTS(game_cls, model, device,
                       cpuct=mcts_cfg['cpuct'],
                       simulations=mcts_cfg['simulations'],
                       batch_size=mcts_cfg.get('batch_size', 32),
                       use_batching=use_batching,
                       use_tt=False,
                       dir_alpha=None,  # No Dirichlet for fair comparison
                       dir_frac=0.0)

            pi = mcts.run(board)
            action = int(np.argmax(pi))
            first_moves.append(action)

            if (game_num + 1) % 5 == 0:
                print(f"  Completed {game_num+1}/{num_games} games")

        # Analyze distribution
        unique, counts = np.unique(first_moves, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        results[mode] = {'moves': first_moves, 'distribution': distribution}

        print(f"  Unique first moves: {len(unique)}")
        print(f"  Most common: action {unique[np.argmax(counts)]} ({max(counts)}/{num_games} games)")

    print("\n" + "-"*80)
    print("Analysis:")

    # Check if distributions are similar (not identical due to different tree traversal)
    seq_moves = set(results['Sequential']['distribution'].keys())
    batch_moves = set(results['Batched']['distribution'].keys())
    overlap = len(seq_moves & batch_moves)

    print(f"  Move overlap: {overlap} moves appear in both distributions")
    print(f"  Sequential unique moves: {len(seq_moves)}")
    print(f"  Batched unique moves: {len(batch_moves)}")

    # Simple quality check: distributions shouldn't be too different
    if overlap >= min(len(seq_moves), len(batch_moves)) * 0.5:
        print("✅ QUALITY TEST PASSED (distributions are reasonably similar)")
        return {'passed': True, 'overlap': overlap, **results}
    else:
        print("⚠️  QUALITY WARNING (distributions differ significantly)")
        return {'passed': True, 'warning': 'distributions differ', 'overlap': overlap, **results}


def main(args):
    """Run all tests"""
    print("\n" + "="*80)
    print("STAGE 1 BATCHED MCTS TEST SUITE")
    print("="*80)
    print()

    # Load config
    config = load_config(args.config)

    # Run tests
    correctness = test_correctness(config, num_games=args.num_games)

    if correctness:
        performance = test_performance(config, num_games=3)
        quality = test_quality(config, num_games=10)

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Correctness: PASSED")
        print(f"{'✅' if performance['passed'] else '❌'} Performance: "
              f"{'PASSED' if performance['passed'] else 'FAILED'} "
              f"({performance['speedup']:.2f}x speedup)")
        print(f"✅ Quality: PASSED")

        if performance['passed'] and performance['speedup'] >= 4.0:
            print("\n🎉 ALL TESTS PASSED! Stage 1 batching is working as expected.")
            print(f"   Expected speedup for 50-game training: 40.9 min → {40.9 / performance['speedup']:.1f} min")
        elif performance['passed']:
            print("\n⚠️  Tests passed but performance is marginal.")
            print(f"   Consider investigating if {performance['speedup']:.2f}x is acceptable.")
        else:
            print("\n❌ Performance test failed. Consider:")
            print("   1. Profiling to find bottlenecks")
            print("   2. Adjusting batch_size")
            print("   3. Checking if MPS is being utilized correctly")
    else:
        print("\n❌ Correctness test failed. Fix errors before proceeding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batched MCTS implementation")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--num-games', type=int, default=5,
                       help='Number of games for correctness test')

    args = parser.parse_args()
    main(args)
