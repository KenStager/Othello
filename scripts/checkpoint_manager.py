#!/usr/bin/env python3
"""
Checkpoint Management Tool

Utilities for managing, comparing, and analyzing training checkpoints.

Features:
- List all checkpoints with metadata
- Compare checkpoints (iteration, loss, win rate)
- View detailed checkpoint information
- Prune old checkpoints
- Resume from specific iteration
- Export checkpoint metadata

Usage:
    # List all checkpoints
    python scripts/checkpoint_manager.py list

    # Show detailed info for specific checkpoint
    python scripts/checkpoint_manager.py info --checkpoint data/checkpoints/current_iter5.pt

    # Compare two checkpoints
    python scripts/checkpoint_manager.py compare --ckpt1 current_iter3.pt --ckpt2 current_iter5.pt

    # Prune old checkpoints (keep every Nth)
    python scripts/checkpoint_manager.py prune --keep-every 5

    # Export metadata to JSON
    python scripts/checkpoint_manager.py export --output checkpoints.json
"""

import argparse
import glob
import json
import os
import re
import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_checkpoints(checkpoint_dir: str = "data/checkpoints") -> List[Tuple[str, int]]:
    """
    Find all checkpoint files and extract iteration numbers.

    Returns:
        List of (path, iteration) tuples sorted by iteration
    """
    pattern = os.path.join(checkpoint_dir, "current_iter*.pt")
    checkpoint_files = glob.glob(pattern)

    checkpoints = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        match = re.search(r'current_iter(\d+)\.pt', basename)
        if match:
            iteration = int(match.group(1))
            checkpoints.append((path, iteration))

    # Also check for champion checkpoint
    champion_path = os.path.join(checkpoint_dir, "champion.pt")
    if os.path.exists(champion_path):
        checkpoints.append((champion_path, -1))  # Use -1 for champion

    return sorted(checkpoints, key=lambda x: x[1])


def load_checkpoint_metadata(checkpoint_path: str) -> Dict:
    """
    Load checkpoint and extract metadata.

    Returns:
        Dict with checkpoint metadata
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        metadata = {
            'path': checkpoint_path,
            'iteration': checkpoint.get('iteration', None),
            'champion_loss_rate': checkpoint.get('champion_loss_rate', None),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
            'contains_model': 'model_state_dict' in checkpoint,
            'contains_champion': 'champion_state_dict' in checkpoint,
        }

        # Try to extract model architecture info if available
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            metadata['num_parameters'] = sum(p.numel() for p in state_dict.values())

        return metadata

    except Exception as e:
        return {
            'path': checkpoint_path,
            'error': str(e),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
        }


def list_checkpoints(checkpoint_dir: str = "data/checkpoints", verbose: bool = True):
    """List all checkpoints with metadata."""
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []

    if verbose:
        print(f"\n{'='*80}")
        print(f"CHECKPOINTS ({len(checkpoints)} found)")
        print(f"Directory: {checkpoint_dir}")
        print(f"{'='*80}\n")

        print(f"{'Iter':<6} {'File':<30} {'Size (MB)':<12} {'Champion Loss Rate':<20}")
        print("-" * 80)

    checkpoint_data = []
    for path, iteration in checkpoints:
        metadata = load_checkpoint_metadata(path)
        checkpoint_data.append(metadata)

        if verbose:
            filename = os.path.basename(path)
            iter_str = f"{iteration}" if iteration >= 0 else "CHAMP"
            loss_rate = metadata.get('champion_loss_rate', 'N/A')
            loss_str = f"{loss_rate:.3f}" if isinstance(loss_rate, (int, float)) else str(loss_rate)

            print(f"{iter_str:<6} {filename:<30} {metadata['file_size_mb']:>8.2f}    {loss_str:<20}")

    if verbose:
        print()

    return checkpoint_data


def show_checkpoint_info(checkpoint_path: str):
    """Show detailed information about a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n{'='*80}")
    print(f"CHECKPOINT DETAILS")
    print(f"{'='*80}\n")

    metadata = load_checkpoint_metadata(checkpoint_path)

    if 'error' in metadata:
        print(f"Error loading checkpoint: {metadata['error']}")
        return

    print(f"File: {checkpoint_path}")
    print(f"Size: {metadata['file_size_mb']:.2f} MB")
    print()

    print("Contents:")
    print(f"  Iteration: {metadata.get('iteration', 'N/A')}")
    print(f"  Champion Loss Rate: {metadata.get('champion_loss_rate', 'N/A')}")
    print(f"  Contains Model: {metadata.get('contains_model', False)}")
    print(f"  Contains Champion: {metadata.get('contains_champion', False)}")

    if 'num_parameters' in metadata:
        print(f"  Model Parameters: {metadata['num_parameters']:,}")

    print()

    # Load full checkpoint for detailed inspection
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Available Keys:")
        for key in sorted(checkpoint.keys()):
            value_type = type(checkpoint[key]).__name__
            print(f"  - {key}: {value_type}")

    except Exception as e:
        print(f"Error loading full checkpoint: {e}")

    print()


def compare_checkpoints(ckpt1_path: str, ckpt2_path: str):
    """Compare two checkpoints."""
    print(f"\n{'='*80}")
    print(f"CHECKPOINT COMPARISON")
    print(f"{'='*80}\n")

    meta1 = load_checkpoint_metadata(ckpt1_path)
    meta2 = load_checkpoint_metadata(ckpt2_path)

    if 'error' in meta1 or 'error' in meta2:
        print("Error loading one or both checkpoints")
        return

    print(f"Checkpoint 1: {os.path.basename(ckpt1_path)}")
    print(f"Checkpoint 2: {os.path.basename(ckpt2_path)}")
    print()

    # Compare iterations
    iter1 = meta1.get('iteration', 0)
    iter2 = meta2.get('iteration', 0)
    print(f"Iterations:")
    print(f"  Checkpoint 1: {iter1}")
    print(f"  Checkpoint 2: {iter2}")
    print(f"  Difference: {iter2 - iter1}")
    print()

    # Compare champion loss rates
    loss1 = meta1.get('champion_loss_rate')
    loss2 = meta2.get('champion_loss_rate')
    if loss1 is not None and loss2 is not None:
        print(f"Champion Loss Rate:")
        print(f"  Checkpoint 1: {loss1:.3f}")
        print(f"  Checkpoint 2: {loss2:.3f}")
        diff = loss2 - loss1
        trend = "↓ improved" if diff < 0 else "↑ worsened" if diff > 0 else "→ unchanged"
        print(f"  Change: {diff:+.3f} {trend}")
        print()

    # Compare file sizes
    print(f"File Size:")
    print(f"  Checkpoint 1: {meta1['file_size_mb']:.2f} MB")
    print(f"  Checkpoint 2: {meta2['file_size_mb']:.2f} MB")
    print()


def prune_checkpoints(checkpoint_dir: str = "data/checkpoints",
                     keep_every: int = 5,
                     dry_run: bool = True):
    """
    Prune old checkpoints, keeping every Nth iteration.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_every: Keep every Nth checkpoint
        dry_run: If True, only show what would be deleted
    """
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Filter out champion
    regular_checkpoints = [(p, i) for p, i in checkpoints if i >= 0]

    if not regular_checkpoints:
        print("No regular checkpoints to prune")
        return

    # Determine which to keep
    to_keep = []
    to_delete = []

    # Always keep the latest
    latest_iter = max(i for _, i in regular_checkpoints)

    for path, iteration in regular_checkpoints:
        if iteration == latest_iter or iteration % keep_every == 0:
            to_keep.append((path, iteration))
        else:
            to_delete.append((path, iteration))

    print(f"\n{'='*80}")
    print(f"CHECKPOINT PRUNING {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*80}\n")

    print(f"Strategy: Keep every {keep_every} iterations + latest")
    print(f"Total checkpoints: {len(regular_checkpoints)}")
    print(f"To keep: {len(to_keep)}")
    print(f"To delete: {len(to_delete)}")
    print()

    if to_delete:
        total_size_mb = sum(
            os.path.getsize(p) / (1024 * 1024) for p, _ in to_delete
        )

        print(f"Checkpoints to delete ({total_size_mb:.2f} MB):")
        for path, iteration in sorted(to_delete, key=lambda x: x[1]):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  Iteration {iteration:3}: {os.path.basename(path)} ({size_mb:.2f} MB)")

        print()

        if not dry_run:
            confirm = input("Proceed with deletion? [y/N]: ")
            if confirm.lower() == 'y':
                for path, iteration in to_delete:
                    try:
                        os.remove(path)
                        print(f"  Deleted: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"  Error deleting {os.path.basename(path)}: {e}")
                print(f"\nDeleted {len(to_delete)} checkpoints, freed {total_size_mb:.2f} MB")
            else:
                print("Deletion cancelled")
        else:
            print("DRY RUN: No files deleted. Use --execute to actually delete files.")
    else:
        print("No checkpoints to delete")

    print()


def export_metadata(checkpoint_dir: str = "data/checkpoints",
                   output_file: str = "checkpoints.json"):
    """Export checkpoint metadata to JSON."""
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    metadata_list = []
    for path, iteration in checkpoints:
        metadata = load_checkpoint_metadata(path)
        metadata_list.append(metadata)

    with open(output_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\n{'='*80}")
    print(f"METADATA EXPORT")
    print(f"{'='*80}\n")
    print(f"Exported {len(metadata_list)} checkpoints to {output_file}")
    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Checkpoint management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument('--dir', default='data/checkpoints',
                            help='Checkpoint directory')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show checkpoint details')
    info_parser.add_argument('--checkpoint', required=True,
                            help='Checkpoint file path')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two checkpoints')
    compare_parser.add_argument('--ckpt1', required=True,
                               help='First checkpoint path')
    compare_parser.add_argument('--ckpt2', required=True,
                               help='Second checkpoint path')

    # Prune command
    prune_parser = subparsers.add_parser('prune', help='Prune old checkpoints')
    prune_parser.add_argument('--dir', default='data/checkpoints',
                             help='Checkpoint directory')
    prune_parser.add_argument('--keep-every', type=int, default=5,
                             help='Keep every Nth checkpoint (default: 5)')
    prune_parser.add_argument('--execute', action='store_true',
                             help='Actually delete files (default is dry run)')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export metadata to JSON')
    export_parser.add_argument('--dir', default='data/checkpoints',
                              help='Checkpoint directory')
    export_parser.add_argument('--output', default='checkpoints.json',
                              help='Output JSON file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'list':
        list_checkpoints(args.dir)

    elif args.command == 'info':
        show_checkpoint_info(args.checkpoint)

    elif args.command == 'compare':
        compare_checkpoints(args.ckpt1, args.ckpt2)

    elif args.command == 'prune':
        prune_checkpoints(args.dir, args.keep_every, dry_run=not args.execute)

    elif args.command == 'export':
        export_metadata(args.dir, args.output)


if __name__ == "__main__":
    main()
