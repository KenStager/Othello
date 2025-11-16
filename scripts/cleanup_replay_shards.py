#!/usr/bin/env python3
"""
One-Time Replay Shard Cleanup Script

Cleans up accumulated replay shards according to retention policy:
- Keep 3 most recent shards
- Keep milestone shards (every 50,000 samples)
- Delete all others

Usage:
    python scripts/cleanup_replay_shards.py --dry-run  # Preview what would be deleted
    python scripts/cleanup_replay_shards.py             # Actually delete files
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_replay_shards(replay_dir="data/replay"):
    """Find all replay shard files and extract counts."""
    pattern = os.path.join(replay_dir, "replay_*.pkl")
    shard_files = glob.glob(pattern)

    shards = []
    for path in shard_files:
        basename = os.path.basename(path)
        match = re.search(r'replay_(\d+)\.pkl', basename)
        if match:
            count = int(match.group(1))
            size_mb = os.path.getsize(path) / (1024 * 1024)
            shards.append((path, count, size_mb))

    shards.sort(key=lambda x: x[1], reverse=True)
    return shards


def cleanup_shards(replay_dir="data/replay", keep_recent=3, keep_milestone_every=50000, dry_run=True):
    """
    Clean up old replay shards.

    Args:
        replay_dir: Directory containing replay shards
        keep_recent: Number of most recent shards to keep
        keep_milestone_every: Keep milestone shards at intervals
        dry_run: If True, only show what would be deleted
    """
    shards = find_replay_shards(replay_dir)

    if not shards:
        print(f"No replay shards found in {replay_dir}")
        return

    print(f"\n{'='*80}")
    print(f"REPLAY SHARD CLEANUP {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*80}\n")

    print(f"Total shards found: {len(shards)}")
    total_size_mb = sum(s[2] for s in shards)
    print(f"Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    print()

    # Determine which to keep
    keep_paths = set()

    # Keep N most recent
    print(f"Keep {keep_recent} most recent shards:")
    for path, count, size_mb in shards[:keep_recent]:
        keep_paths.add(path)
        print(f"  ✓ {os.path.basename(path)} ({size_mb:.1f} MB)")

    # Keep milestone shards
    print(f"\nKeep milestone shards (every {keep_milestone_every:,} samples):")
    milestone_count = 0
    for path, count, size_mb in shards:
        if count % keep_milestone_every == 0 and path not in keep_paths:
            keep_paths.add(path)
            print(f"  ✓ {os.path.basename(path)} ({size_mb:.1f} MB) [milestone]")
            milestone_count += 1

    if milestone_count == 0:
        print("  (none)")

    # Files to delete
    to_delete = [(path, count, size_mb) for path, count, size_mb in shards if path not in keep_paths]

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    print(f"Keep: {len(keep_paths)} shards ({sum(s[2] for s in shards if s[0] in keep_paths):.1f} MB)")
    print(f"Delete: {len(to_delete)} shards ({sum(s[2] for s in to_delete):.1f} MB)")
    print()

    if to_delete:
        print("Files to delete:")
        for path, count, size_mb in sorted(to_delete, key=lambda x: x[1]):
            print(f"  ✗ {os.path.basename(path)} ({size_mb:.1f} MB)")

        print()

        if not dry_run:
            confirm = input("Proceed with deletion? [y/N]: ")
            if confirm.lower() == 'y':
                deleted_count = 0
                freed_mb = 0
                print("\nDeleting files...")
                for path, count, size_mb in to_delete:
                    try:
                        os.remove(path)
                        deleted_count += 1
                        freed_mb += size_mb
                        print(f"  Deleted: {os.path.basename(path)}")
                    except Exception as e:
                        print(f"  Error deleting {os.path.basename(path)}: {e}")

                print(f"\n✅ Cleanup complete!")
                print(f"   Deleted: {deleted_count} shards")
                print(f"   Freed: {freed_mb:.1f} MB ({freed_mb/1024:.2f} GB)")
            else:
                print("\nCancelled - no files deleted")
        else:
            print("DRY RUN - no files deleted. Use without --dry-run to actually delete.")
    else:
        print("No files to delete - already at minimum retention")

    print()


def main():
    parser = argparse.ArgumentParser(description="Clean up old replay shards")
    parser.add_argument('--replay-dir', default='data/replay',
                       help='Replay directory (default: data/replay)')
    parser.add_argument('--keep-recent', type=int, default=3,
                       help='Number of most recent shards to keep (default: 3)')
    parser.add_argument('--keep-milestone-every', type=int, default=50000,
                       help='Keep milestone shards at intervals (default: 50000)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without deleting (default: False)')

    args = parser.parse_args()

    cleanup_shards(
        replay_dir=args.replay_dir,
        keep_recent=args.keep_recent,
        keep_milestone_every=args.keep_milestone_every,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
