"""
Dual-Platform Neural Network Inference Optimization Benchmark

Tests inference optimizations (FP16 + Channels Last + TensorRT/TorchScript)
on both Apple Silicon (MPS) and NVIDIA GPU (CUDA) platforms.

Validates:
1. Correctness: Optimized outputs match unoptimized (within epsilon)
2. Performance: Measure actual speedup achieved
3. MCTS Integration: End-to-end self-play performance

Expected Results:
- Mac M4 Max (MPS): 2.0-3.0× speedup (FP16 + Channels Last + TorchScript)
- Vast.ai RTX 4070 Ti (CUDA): 4.0-10.0× speedup (FP16 + Channels Last + TensorRT)
"""

import torch
import numpy as np
import time
from src.net.model import OthelloNet
from src.mcts.batch_evaluator import SimpleBatchEvaluator
from src.mcts.mcts import MCTS
from src.othello.board import Board


def detect_platform():
    """Detect current platform and available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        platform = "NVIDIA CUDA"
        expected_speedup = "4.0-10.0×"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        platform = "Apple Silicon MPS"
        expected_speedup = "2.0-3.0×"
    else:
        device = torch.device("cpu")
        platform = "CPU"
        expected_speedup = "1.0-1.2×"

    return device, platform, expected_speedup


def create_test_positions(num_positions=20):
    """Create diverse test positions for benchmarking."""
    positions = []

    for i in range(num_positions):
        board = Board()
        np.random.seed(i)

        # Advance to random game state
        num_moves = np.random.randint(5, 30)
        for _ in range(num_moves):
            moves = board.legal_moves()
            if not moves:
                board.apply_move(None)
            else:
                move = moves[np.random.randint(len(moves))]
                board.apply_move(move)

            if board.is_terminal():
                break

        # Store encoded state
        positions.append(board.encode())

    return positions


def test_correctness(device, optimization_config):
    """
    Test that optimized model produces same outputs as unoptimized.

    Returns:
        tuple: (max_policy_diff, max_value_diff, passed)
    """
    print("\n" + "="*70)
    print("CORRECTNESS TEST")
    print("="*70)

    # Create evaluators with separate model instances
    print("\nCreating evaluators...")

    print("  - Unoptimized (baseline)")
    model_unopt = OthelloNet(in_channels=4, channels=64, residual_blocks=4, action_size=65)
    model_unopt.eval()
    eval_unopt = SimpleBatchEvaluator(model_unopt, device, batch_size=32, optimization_config=None)

    print("  - Optimized (FP16 + Channels Last + Compilation)")
    model_opt = OthelloNet(in_channels=4, channels=64, residual_blocks=4, action_size=65)
    model_opt.eval()
    eval_opt = SimpleBatchEvaluator(model_opt, device, batch_size=32, optimization_config=optimization_config)

    # Create test positions
    print("\nGenerating test positions...")
    positions = create_test_positions(num_positions=50)
    print(f"  Created {len(positions)} test positions")

    # Evaluate with both
    print("\nEvaluating positions...")
    results_unopt = eval_unopt.evaluate_batch(positions)
    results_opt = eval_opt.evaluate_batch(positions)

    # Compare results
    max_policy_diff = 0.0
    max_value_diff = 0.0

    for i, (r_unopt, r_opt) in enumerate(zip(results_unopt, results_opt)):
        policy_unopt, value_unopt = r_unopt
        policy_opt, value_opt = r_opt

        # Policy difference (L∞ norm)
        policy_diff = np.max(np.abs(policy_unopt - policy_opt))
        max_policy_diff = max(max_policy_diff, policy_diff)

        # Value difference (absolute)
        value_diff = abs(value_unopt - value_opt)
        max_value_diff = max(max_value_diff, value_diff)

        if policy_diff > 0.01 or value_diff > 0.05:
            print(f"  Position {i}: policy_diff={policy_diff:.6f}, value_diff={value_diff:.6f} ⚠️")

    # Results
    print("\n" + "="*70)
    print("CORRECTNESS RESULTS")
    print("="*70)
    print(f"Max policy difference: {max_policy_diff:.6f}")
    print(f"Max value difference:  {max_value_diff:.6f}")

    # Pass criteria: FP16 introduces small numerical errors
    policy_passed = max_policy_diff < 0.01
    value_passed = max_value_diff < 0.05
    passed = policy_passed and value_passed

    if passed:
        print("✅ CORRECTNESS: PASSED")
    else:
        print("⚠️  CORRECTNESS: DIFFERENCES EXCEED THRESHOLD")
        if not policy_passed:
            print(f"   Policy diff {max_policy_diff:.6f} > 0.01 threshold")
        if not value_passed:
            print(f"   Value diff {max_value_diff:.6f} > 0.05 threshold")

    return max_policy_diff, max_value_diff, passed


def test_throughput(device, optimization_config):
    """
    Benchmark inference throughput with and without optimization.

    Returns:
        tuple: (speedup, unopt_time, opt_time)
    """
    print("\n" + "="*70)
    print("THROUGHPUT BENCHMARK")
    print("="*70)

    # Create evaluators with separate model instances
    print("\nCreating evaluators...")

    model_unopt = OthelloNet(in_channels=4, channels=64, residual_blocks=4, action_size=65)
    model_unopt.eval()
    eval_unopt = SimpleBatchEvaluator(model_unopt, device, batch_size=32, optimization_config=None)

    model_opt = OthelloNet(in_channels=4, channels=64, residual_blocks=4, action_size=65)
    model_opt.eval()
    eval_opt = SimpleBatchEvaluator(model_opt, device, batch_size=32, optimization_config=optimization_config)

    # Create test positions
    positions = create_test_positions(num_positions=200)

    # Warmup (important for torch.compile() compilation)
    print("\nWarming up optimized model (torch.compile compilation)...")
    print("  Note: First compilation with max-autotune mode takes 30-60 seconds")

    # Run multiple warmup iterations to ensure compilation completes
    warmup_batch_size = 32
    warmup_iterations = 10  # Run 10 batches to ensure compilation is done

    warmup_start = time.time()
    for i in range(warmup_iterations):
        batch_start_idx = (i * warmup_batch_size) % len(positions)
        batch = positions[batch_start_idx:batch_start_idx + warmup_batch_size]
        _ = eval_opt.evaluate_batch(batch)

        # Print progress every few iterations
        if i == 0:
            first_iter_time = time.time() - warmup_start
            print(f"  First iteration: {first_iter_time:.1f}s (includes compilation)")
        elif i == warmup_iterations - 1:
            total_warmup_time = time.time() - warmup_start
            avg_iter_time = total_warmup_time / warmup_iterations
            print(f"  Total warmup: {total_warmup_time:.1f}s ({warmup_iterations} iterations)")
            print(f"  Average per iteration: {avg_iter_time:.3f}s (post-compilation)")

    print("  ✅ Warmup complete, model fully compiled")

    # Benchmark unoptimized
    print("\nBenchmarking UNOPTIMIZED inference...")
    start = time.time()
    for i in range(0, len(positions), 32):
        batch = positions[i:i+32]
        _ = eval_unopt.evaluate_batch(batch)
    unopt_time = time.time() - start

    unopt_throughput = len(positions) / unopt_time
    print(f"  Time: {unopt_time:.3f}s")
    print(f"  Throughput: {unopt_throughput:.1f} positions/sec")

    # Benchmark optimized
    print("\nBenchmarking OPTIMIZED inference...")
    start = time.time()
    for i in range(0, len(positions), 32):
        batch = positions[i:i+32]
        _ = eval_opt.evaluate_batch(batch)
    opt_time = time.time() - start

    opt_throughput = len(positions) / opt_time
    print(f"  Time: {opt_time:.3f}s")
    print(f"  Throughput: {opt_throughput:.1f} positions/sec")

    # Speedup
    speedup = unopt_time / opt_time

    print("\n" + "="*70)
    print("THROUGHPUT RESULTS")
    print("="*70)
    print(f"Speedup: {speedup:.2f}×")
    print(f"Throughput increase: {(speedup - 1) * 100:.1f}%")

    return speedup, unopt_time, opt_time


def test_mcts_integration(device, optimization_config):
    """
    Test MCTS integration with optimized inference.

    Returns:
        tuple: (speedup, unopt_time, opt_time)
    """
    print("\n" + "="*70)
    print("MCTS INTEGRATION TEST")
    print("="*70)

    # Create evaluators with separate model instances
    print("\nCreating evaluators...")

    model_unopt = OthelloNet(in_channels=4, channels=64, residual_blocks=4, action_size=65)
    model_unopt.eval()
    eval_unopt = SimpleBatchEvaluator(model_unopt, device, batch_size=32, optimization_config=None)

    model_opt = OthelloNet(in_channels=4, channels=64, residual_blocks=4, action_size=65)
    model_opt.eval()
    eval_opt = SimpleBatchEvaluator(model_opt, device, batch_size=32, optimization_config=optimization_config)

    # Warmup optimized model
    print("\nWarming up optimized model...")
    warmup_positions = create_test_positions(num_positions=10)
    _ = eval_opt.evaluate_batch(warmup_positions)

    # Test positions
    num_positions = 10
    simulations = 50

    print(f"\nRunning MCTS on {num_positions} positions ({simulations} sims each)...")

    # Benchmark unoptimized
    print("\nUNOPTIMIZED MCTS:")
    positions_unopt = []
    for i in range(num_positions):
        board = Board()
        np.random.seed(i)
        for _ in range(np.random.randint(5, 20)):
            moves = board.legal_moves()
            if moves:
                board.apply_move(moves[np.random.randint(len(moves))])
        positions_unopt.append(board)

    start = time.time()
    for board in positions_unopt:
        mcts = MCTS(
            game_cls=Board,
            net=None,
            device=None,
            cpuct=1.5,
            simulations=simulations,
            batch_evaluator=eval_unopt,
            reuse_tree=False
        )
        _ = mcts.run(board)
    unopt_time = time.time() - start

    print(f"  Total: {unopt_time:.2f}s")
    print(f"  Per position: {unopt_time / num_positions:.3f}s")

    # Benchmark optimized
    print("\nOPTIMIZED MCTS:")
    positions_opt = []
    for i in range(num_positions):
        board = Board()
        np.random.seed(i)
        for _ in range(np.random.randint(5, 20)):
            moves = board.legal_moves()
            if moves:
                board.apply_move(moves[np.random.randint(len(moves))])
        positions_opt.append(board)

    start = time.time()
    for board in positions_opt:
        mcts = MCTS(
            game_cls=Board,
            net=None,
            device=None,
            cpuct=1.5,
            simulations=simulations,
            batch_evaluator=eval_opt,
            reuse_tree=False
        )
        _ = mcts.run(board)
    opt_time = time.time() - start

    print(f"  Total: {opt_time:.2f}s")
    print(f"  Per position: {opt_time / num_positions:.3f}s")

    # Speedup
    speedup = unopt_time / opt_time

    print("\n" + "="*70)
    print("MCTS INTEGRATION RESULTS")
    print("="*70)
    print(f"Speedup: {speedup:.2f}×")

    if speedup >= 2.0:
        print(f"✅ Excellent MCTS speedup ({speedup:.2f}×)")
    elif speedup >= 1.5:
        print(f"✅ Good MCTS speedup ({speedup:.2f}×)")
    elif speedup >= 1.2:
        print(f"✅ Acceptable MCTS speedup ({speedup:.2f}×)")
    else:
        print(f"⚠️  Speedup below expectation ({speedup:.2f}×)")

    return speedup, unopt_time, opt_time


def main():
    print("\n" + "="*70)
    print("DUAL-PLATFORM INFERENCE OPTIMIZATION BENCHMARK")
    print("="*70)

    # Detect platform
    device, platform, expected_speedup = detect_platform()

    print(f"\nPlatform: {platform}")
    print(f"Device: {device}")
    print(f"Expected speedup: {expected_speedup}")

    # Optimization config
    optimization_config = {
        'enabled': True,
        'precision': 'fp16',
        'use_channels_last': True,
        'use_compilation': True,
        'tensorrt_workspace_gb': 1
    }

    print("\nOptimization config:")
    for key, value in optimization_config.items():
        print(f"  {key}: {value}")

    # Run tests
    try:
        # Test 1: Correctness
        max_policy_diff, max_value_diff, correctness_passed = test_correctness(device, optimization_config)

        # Test 2: Throughput
        throughput_speedup, throughput_unopt, throughput_opt = test_throughput(device, optimization_config)

        # Test 3: MCTS Integration
        mcts_speedup, mcts_unopt, mcts_opt = test_mcts_integration(device, optimization_config)

        # Final Summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Platform: {platform}")
        print(f"Expected speedup: {expected_speedup}")
        print()
        print(f"Correctness:")
        print(f"  Policy diff: {max_policy_diff:.6f} ({'✅ PASS' if max_policy_diff < 0.01 else '⚠️ WARN'})")
        print(f"  Value diff:  {max_value_diff:.6f} ({'✅ PASS' if max_value_diff < 0.05 else '⚠️ WARN'})")
        print()
        print(f"Throughput speedup: {throughput_speedup:.2f}×")
        print(f"  Unoptimized: {throughput_unopt:.3f}s")
        print(f"  Optimized:   {throughput_opt:.3f}s")
        print()
        print(f"MCTS speedup: {mcts_speedup:.2f}×")
        print(f"  Unoptimized: {mcts_unopt:.3f}s")
        print(f"  Optimized:   {mcts_opt:.3f}s")
        print()

        # Overall assessment
        if correctness_passed and throughput_speedup >= 1.5:
            print("✅ SUCCESS: Optimization ready for deployment")
            print(f"   Achieved {throughput_speedup:.2f}× speedup with correct outputs")
        elif correctness_passed:
            print("⚠️  PARTIAL: Correctness OK but speedup below target")
            print(f"   Only {throughput_speedup:.2f}× speedup (expected {expected_speedup})")
        else:
            print("❌ FAILED: Correctness issues detected")
            print("   Review optimization implementation before deployment")

    except Exception as e:
        print(f"\n❌ ERROR during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
