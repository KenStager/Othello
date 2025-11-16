#!/usr/bin/env python3
"""
Training Visualization Dashboard

Creates comprehensive plots of training progress from logs/train.tsv.

Features:
- Training loss over time
- Replay buffer size
- Gate win rate trends (champion promotion tracking)
- Phase distribution evolution
- Multi-panel dashboard view

Usage:
    # Generate all plots
    python scripts/plot_training.py

    # Save to specific directory
    python scripts/plot_training.py --output plots/

    # Plot specific metrics only
    python scripts/plot_training.py --metrics loss,win_rate

Requirements:
    pip install matplotlib pandas
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: matplotlib and pandas are required")
    print("Install with: pip install matplotlib pandas")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_training_log(log_path: str = "logs/train.tsv") -> Optional[pd.DataFrame]:
    """
    Load training log TSV file.

    Returns:
        DataFrame with training metrics, or None if file not found
    """
    if not os.path.exists(log_path):
        print(f"Error: Training log not found: {log_path}")
        return None

    try:
        df = pd.read_csv(log_path, sep='\t')
        print(f"Loaded {len(df)} training entries from {log_path}")
        return df
    except Exception as e:
        print(f"Error loading training log: {e}")
        return None


def plot_training_loss(df: pd.DataFrame, output_dir: str):
    """Plot training loss over iterations."""
    if 'train_loss' not in df.columns:
        print("Warning: 'train_loss' column not found in log")
        return

    plt.figure(figsize=(12, 6))

    # Filter out NaN values
    loss_data = df[['train_loss']].dropna()

    if len(loss_data) == 0:
        print("Warning: No training loss data available")
        return

    plt.plot(loss_data.index, loss_data['train_loss'], linewidth=2, alpha=0.7, label='Training Loss')

    # Add smoothed trend line (moving average)
    if len(loss_data) >= 10:
        window = min(10, len(loss_data) // 5)
        smoothed = loss_data['train_loss'].rolling(window=window, center=True).mean()
        plt.plot(smoothed.index, smoothed, linewidth=3, color='red', alpha=0.8, label=f'Trend (MA-{window})')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_replay_buffer(df: pd.DataFrame, output_dir: str):
    """Plot replay buffer size over iterations."""
    if 'replay_size' not in df.columns:
        print("Warning: 'replay_size' column not found in log")
        return

    plt.figure(figsize=(12, 6))

    buffer_data = df[['replay_size']].dropna()

    if len(buffer_data) == 0:
        print("Warning: No replay buffer data available")
        return

    plt.plot(buffer_data.index, buffer_data['replay_size'], linewidth=2, color='green', alpha=0.7)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Buffer Size (samples)', fontsize=12)
    plt.title('Replay Buffer Growth', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'replay_buffer.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_gate_performance(df: pd.DataFrame, output_dir: str):
    """Plot gating/evaluation performance."""
    # Look for gate-related columns
    gate_cols = [col for col in df.columns if 'gate' in col.lower() or 'eval' in col.lower()]

    if not gate_cols:
        print("Warning: No gating/evaluation columns found in log")
        return

    plt.figure(figsize=(12, 6))

    # Try to plot gate win rate if available
    if 'gate_win_rate' in df.columns:
        gate_data = df[['gate_win_rate']].dropna()
        if len(gate_data) > 0:
            plt.plot(gate_data.index, gate_data['gate_win_rate'], 'o-', linewidth=2,
                    markersize=6, alpha=0.7, label='Gate Win Rate')

            # Add promotion threshold line
            plt.axhline(y=0.55, color='red', linestyle='--', linewidth=2,
                       alpha=0.6, label='Promotion Threshold (55%)')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.title('Gate Evaluation Performance', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'gate_performance.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_dashboard(df: pd.DataFrame, output_dir: str):
    """Create comprehensive multi-panel dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Dashboard', fontsize=16, fontweight='bold')

    # Panel 1: Training Loss
    ax = axes[0, 0]
    if 'train_loss' in df.columns:
        loss_data = df[['train_loss']].dropna()
        if len(loss_data) > 0:
            ax.plot(loss_data.index, loss_data['train_loss'], linewidth=2, alpha=0.7)

            if len(loss_data) >= 10:
                window = min(10, len(loss_data) // 5)
                smoothed = loss_data['train_loss'].rolling(window=window, center=True).mean()
                ax.plot(smoothed.index, smoothed, linewidth=3, color='red', alpha=0.8)

            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)

    # Panel 2: Replay Buffer
    ax = axes[0, 1]
    if 'replay_size' in df.columns:
        buffer_data = df[['replay_size']].dropna()
        if len(buffer_data) > 0:
            ax.plot(buffer_data.index, buffer_data['replay_size'],
                   linewidth=2, color='green', alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Buffer Size')
            ax.set_title('Replay Buffer Growth')
            ax.grid(True, alpha=0.3)

    # Panel 3: Gate Win Rate
    ax = axes[1, 0]
    if 'gate_win_rate' in df.columns:
        gate_data = df[['gate_win_rate']].dropna()
        if len(gate_data) > 0:
            ax.plot(gate_data.index, gate_data['gate_win_rate'], 'o-',
                   linewidth=2, markersize=6, alpha=0.7)
            ax.axhline(y=0.55, color='red', linestyle='--', linewidth=2, alpha=0.6)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Win Rate')
            ax.set_title('Gate Evaluation')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)

    # Panel 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Compute summary stats
    summary_text = "Training Summary\n" + "="*30 + "\n\n"

    if len(df) > 0:
        summary_text += f"Total Iterations: {len(df)}\n\n"

        if 'train_loss' in df.columns:
            loss_data = df['train_loss'].dropna()
            if len(loss_data) > 0:
                summary_text += f"Training Loss:\n"
                summary_text += f"  Final: {loss_data.iloc[-1]:.4f}\n"
                summary_text += f"  Min: {loss_data.min():.4f}\n"
                summary_text += f"  Mean: {loss_data.mean():.4f}\n\n"

        if 'replay_size' in df.columns:
            buffer_data = df['replay_size'].dropna()
            if len(buffer_data) > 0:
                summary_text += f"Replay Buffer:\n"
                summary_text += f"  Final Size: {int(buffer_data.iloc[-1]):,}\n\n"

        if 'gate_win_rate' in df.columns:
            gate_data = df['gate_win_rate'].dropna()
            if len(gate_data) > 0:
                promotions = (gate_data >= 0.55).sum()
                summary_text += f"Gating:\n"
                summary_text += f"  Evaluations: {len(gate_data)}\n"
                summary_text += f"  Promotions: {promotions}\n"
                summary_text += f"  Promotion Rate: {promotions/len(gate_data)*100:.1f}%\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dashboard.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description="Visualize training progress")
    parser.add_argument('--log', default='logs/train.tsv',
                       help='Path to training log TSV file')
    parser.add_argument('--output', default='plots',
                       help='Output directory for plots')
    parser.add_argument('--metrics', type=str,
                       help='Comma-separated list of metrics to plot (loss,buffer,gate,dashboard)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"TRAINING VISUALIZATION")
    print(f"{'='*80}\n")

    # Load training log
    df = load_training_log(args.log)
    if df is None:
        return

    print(f"\nAvailable columns: {', '.join(df.columns)}")
    print()

    # Determine which plots to create
    if args.metrics:
        metrics_to_plot = [m.strip() for m in args.metrics.split(',')]
    else:
        metrics_to_plot = ['loss', 'buffer', 'gate', 'dashboard']

    print("Generating plots...")

    # Create plots
    if 'loss' in metrics_to_plot:
        plot_training_loss(df, args.output)

    if 'buffer' in metrics_to_plot:
        plot_replay_buffer(df, args.output)

    if 'gate' in metrics_to_plot:
        plot_gate_performance(df, args.output)

    if 'dashboard' in metrics_to_plot:
        plot_dashboard(df, args.output)

    print()
    print(f"{'='*80}")
    print(f"Plots saved to: {args.output}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
