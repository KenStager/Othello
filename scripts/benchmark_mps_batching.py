#!/usr/bin/env python3
"""
MPS Batching Throughput Benchmark

Tests OthelloNet inference performance at various batch sizes to determine
if batched MCTS implementation is worthwhile.

Critical GO/NO-GO criteria:
- If batch_size=32 achieves <8x throughput vs batch_size=1: NOT WORTH IT
- If batch_size=32 achieves 10-15x throughput: PROCEED with batching
- If batch_size=32 achieves >15x throughput: HIGH CONFIDENCE, full implementation

Expected runtime: ~2-3 minutes
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.net.model import OthelloNet
from src.utils.config import load_config


def warmup_model(model, device, batch_size=32, warmup_iters=10):
    """
    Warmup model to ensure GPU/MPS is fully initialized.
    Critical for MPS which has ~100-200ms first-inference overhead.
    """
    print(f"  Warming up model (batch_size={batch_size}, {warmup_iters} iterations)...")
    dummy_input = torch.randn(batch_size, 4, 8, 8, device=device)

    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(dummy_input)

    # Clear any cached memory
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    print("  Warmup complete")


def benchmark_batch_size(model, device, batch_size, num_iterations=100):
    """
    Benchmark model throughput at specific batch size.

    Returns:
        dict with keys: batch_size, samples_per_sec, latency_ms, throughput_ratio
    """
    # Generate random input data
    dummy_input = torch.randn(batch_size, 4, 8, 8, device=device)

    # Warmup for this specific batch size (5 iterations)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)

    # Synchronize to ensure warmup is complete
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(dummy_input)

        # Synchronize to measure actual completion time
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to milliseconds

    # Calculate statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    median_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)

    samples_per_sec = batch_size / (mean_latency / 1000)

    return {
        'batch_size': batch_size,
        'samples_per_sec': samples_per_sec,
        'mean_latency_ms': mean_latency,
        'median_latency_ms': median_latency,
        'p95_latency_ms': p95_latency,
        'std_latency_ms': std_latency,
    }


def analyze_results(results):
    """
    Analyze benchmark results and provide recommendations.

    Args:
        results: List of benchmark dicts from benchmark_batch_size()

    Returns:
        dict with analysis and recommendations
    """
    baseline = results[0]  # batch_size=1
    baseline_throughput = baseline['samples_per_sec']

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Batch Size':<12} {'Latency (ms)':<15} {'Throughput':<20} {'Speedup':<10} {'Efficiency'}")
    print(f"{'':12} {'(mean ± std)':<15} {'(samples/sec)':<20} {'vs batch=1':<10} {'(% ideal)'}")
    print("-"*80)

    analysis_results = []

    for r in results:
        bs = r['batch_size']
        latency_str = f"{r['mean_latency_ms']:.2f} ± {r['std_latency_ms']:.2f}"
        throughput = r['samples_per_sec']
        speedup = throughput / baseline_throughput
        efficiency = (speedup / bs) * 100  # % of ideal linear scaling

        analysis_results.append({
            'batch_size': bs,
            'speedup': speedup,
            'efficiency': efficiency,
            'throughput': throughput
        })

        print(f"{bs:<12} {latency_str:<15} {throughput:>8.1f} ({speedup:>5.1f}x) "
              f"{speedup:>5.1f}x      {efficiency:>5.1f}%")

    print("="*80)

    # Find optimal batch size (highest speedup with acceptable latency)
    batch_32_result = next((r for r in analysis_results if r['batch_size'] == 32), None)
    batch_64_result = next((r for r in analysis_results if r['batch_size'] == 64), None)

    print("\nANALYSIS:")
    print("-"*80)

    if batch_32_result:
        speedup_32 = batch_32_result['speedup']
        print(f"Batch size 32 speedup: {speedup_32:.1f}x")

        if speedup_32 < 8:
            recommendation = "❌ NOT RECOMMENDED"
            reason = f"Speedup {speedup_32:.1f}x is below threshold (8x minimum)"
            action = "PIVOT to alternatives: reduce simulations 200→100 (instant 2x) or cloud GPUs"
        elif speedup_32 < 10:
            recommendation = "⚠️  MARGINAL"
            reason = f"Speedup {speedup_32:.1f}x is modest (target: 10-15x)"
            action = "Consider simpler alternatives first, batching has marginal ROI"
        elif speedup_32 < 15:
            recommendation = "✅ PROCEED WITH CAUTION"
            reason = f"Speedup {speedup_32:.1f}x is good (target: 10-15x)"
            action = "Proceed with Stage 1 (minimal batching), monitor ROI carefully"
        else:
            recommendation = "✅ STRONGLY RECOMMENDED"
            reason = f"Speedup {speedup_32:.1f}x exceeds target (>15x)"
            action = "High confidence - proceed with full batching implementation"

        print(f"\nRecommendation: {recommendation}")
        print(f"Reason: {reason}")
        print(f"Action: {action}")

    print("\n" + "="*80)
    print("PROJECTED MCTS SPEEDUP (Amdahl's Law)")
    print("="*80)

    if batch_32_result:
        nn_speedup = batch_32_result['speedup']

        # Assume NN inference is 80% of MCTS time, rest is tree traversal
        nn_fraction = 0.80
        other_fraction = 0.20

        total_speedup = 1 / (other_fraction + nn_fraction / nn_speedup)

        print(f"Assumptions:")
        print(f"  - NN inference: {nn_fraction*100:.0f}% of MCTS time")
        print(f"  - Tree traversal + features: {other_fraction*100:.0f}% of MCTS time")
        print(f"  - NN speedup at batch_size=32: {nn_speedup:.1f}x")
        print(f"\nProjected total MCTS speedup: {total_speedup:.1f}x")
        print(f"  (from {49:.1f}s/game → {49/total_speedup:.1f}s/game)")

        # With virtual loss (additional 1.2-1.5x from better exploration)
        vl_multiplier = 1.3
        total_with_vl = total_speedup * vl_multiplier
        print(f"\nWith virtual loss (Stage 2): {total_with_vl:.1f}x")
        print(f"  (from {49:.1f}s/game → {49/total_with_vl:.1f}s/game)")

        # With parallel games (additional 1.5-2x from better batch filling)
        parallel_multiplier = 1.7
        total_with_parallel = total_with_vl * parallel_multiplier
        print(f"\nWith parallel games (Stage 3): {total_with_parallel:.1f}x")
        print(f"  (from {49:.1f}s/game → {49/total_with_parallel:.1f}s/game)")

    print("="*80)

    return analysis_results


def main(args):
    """Main benchmarking function"""

    print("="*80)
    print("MPS BATCHING THROUGHPUT BENCHMARK")
    print("="*80)

    # Load config
    cfg = load_config(args.config)

    # Setup device
    device_cfg = cfg.get('device', 'cpu')
    if device_cfg == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon)")
        try:
            recommended = torch.mps.recommended_max_memory() / 1e9
            print(f"  Recommended max memory: {recommended:.2f} GB")
        except:
            pass
    elif device_cfg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU (WARNING: benchmarking on CPU is not representative)")

    print(f"Model: OthelloNet (channels={cfg['model']['channels']}, "
          f"blocks={cfg['model']['residual_blocks']})")
    print(f"Benchmark iterations: {args.iterations} per batch size")
    print()

    # Create model
    model = OthelloNet(
        in_channels=4,
        channels=cfg['model']['channels'],
        residual_blocks=cfg['model']['residual_blocks']
    ).to(device)
    model.eval()

    # Warmup
    warmup_model(model, device, batch_size=32, warmup_iters=args.warmup)

    # Run benchmarks
    batch_sizes = args.batch_sizes
    results = []

    print("\nRunning benchmarks...")
    print("-"*80)

    for batch_size in batch_sizes:
        print(f"Benchmarking batch_size={batch_size}... ", end='', flush=True)
        result = benchmark_batch_size(model, device, batch_size, args.iterations)
        results.append(result)
        print(f"✓ {result['samples_per_sec']:.1f} samples/sec "
              f"({result['mean_latency_ms']:.2f}ms latency)")

    # Analyze and print results
    analyze_results(results)

    # Save raw data if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'device': str(device),
            'model_config': {
                'channels': cfg['model']['channels'],
                'residual_blocks': cfg['model']['residual_blocks']
            },
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nRaw data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MPS batching throughput")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[1, 8, 16, 32, 64, 128],
                       help='Batch sizes to test')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per batch size')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--output', type=str, default=None,
                       help='Save raw results to JSON file')

    args = parser.parse_args()
    main(args)
