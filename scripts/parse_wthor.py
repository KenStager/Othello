#!/usr/bin/env python3
"""
WTHOR Database Parser for Imitation Learning Bootstrap

Parses .wtb files from the WTHOR expert game database and generates
training samples in ReplayBuffer format for IL pre-training.

Usage:
    PYTHONPATH=. python scripts/parse_wthor.py --input data/wthor_raw --output data/il_bootstrap --max-samples 50000
"""

import sys
import os
import argparse
import struct
import glob
import pickle
import numpy as np
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.othello.board import Board
from src.othello.game import Game
from src.othello.features import (
    compute_mobility_features,
    compute_stability_map,
    compute_corner_control,
    compute_parity_features,
    augment_state
)


def decode_wthor_move(move_byte):
    """
    Decode WTHOR move byte to action index.

    WTHOR encoding:
        column = byte % 10  (1-8 maps to A-H)
        row = byte // 10    (1-8)
        0 = pass move

    Our encoding:
        action_index = (row - 1) * 8 + (column - 1) for board moves [0-63]
        action_index = 64 for pass

    Args:
        move_byte: Raw move byte from WTHOR file

    Returns:
        action_index: Index in [0-64]
    """
    if move_byte == 0:
        return 64  # Pass action

    column = move_byte % 10  # 1-8
    row = move_byte // 10    # 1-8

    # Validate move format
    if not (1 <= column <= 8 and 1 <= row <= 8):
        raise ValueError(f"Invalid WTHOR move byte: {move_byte} (col={column}, row={row})")

    # Convert to our zero-indexed format
    action_index = (row - 1) * 8 + (column - 1)
    return action_index


def parse_wtb_file(file_path):
    """
    Parse a single .wtb file and return game records.

    File format:
        Header (16 bytes):
            - Bytes 0-3: Date metadata
            - Bytes 4-7: Game count (little-endian uint32)
            - Bytes 8-15: Reserved

        Game records (68 bytes each):
            - Bytes 0-1: Tournament ID (uint16)
            - Bytes 2-3: Black player ID (uint16)
            - Bytes 4-5: White player ID (uint16)
            - Byte 6: Real score (black's disc count)
            - Byte 7: Theoretical score
            - Bytes 8-67: Move sequence (60 bytes)

    Args:
        file_path: Path to .wtb file

    Returns:
        List of game dictionaries with 'score' and 'moves' keys
    """
    with open(file_path, 'rb') as f:
        # Read and parse header
        header = f.read(16)
        if len(header) < 16:
            print(f"  Warning: {file_path} has incomplete header, skipping")
            return []

        # Extract game count from bytes 4-7 (little-endian uint32)
        game_count = struct.unpack('<I', header[4:8])[0]

        games = []
        for game_num in range(game_count):
            # Read game record
            record = f.read(68)
            if len(record) < 68:
                print(f"  Warning: Incomplete game record {game_num} in {file_path}")
                break

            # Parse game metadata
            tournament_id = struct.unpack('<H', record[0:2])[0]
            black_player = struct.unpack('<H', record[2:4])[0]
            white_player = struct.unpack('<H', record[4:6])[0]
            real_score = record[6]  # Black's disc count (0-64)
            theoretical_score = record[7]

            # Parse move sequence (60 bytes)
            moves = []
            for i in range(60):
                move_byte = record[8 + i]
                if move_byte == 0:
                    # End of game (0 means no more moves)
                    break
                try:
                    action = decode_wthor_move(move_byte)
                    moves.append(action)
                except ValueError as e:
                    print(f"  Warning: {e} in game {game_num} of {file_path}, skipping rest of game")
                    break

            if len(moves) > 0:
                games.append({
                    'tournament_id': tournament_id,
                    'black_player': black_player,
                    'white_player': white_player,
                    'score': real_score,
                    'moves': moves
                })

        return games


def determine_phase(empties):
    """Determine game phase based on empty squares."""
    if empties >= 45:
        return "opening"
    elif empties <= 14:
        return "endgame"
    else:
        return "midgame"


def generate_il_samples_from_game(game_record, soft_labels=False, confidence=0.8):
    """
    Replay a game and generate IL training samples.

    Args:
        game_record: Dictionary with 'score' and 'moves' keys
        soft_labels: If True, use soft policy targets instead of one-hot
        confidence: Confidence level for expert move (e.g., 0.8 = 80%)

    Returns:
        List of sample dictionaries in ReplayBuffer format
    """
    # Initialize board
    game = Game()
    board = game.new_board()

    # Determine winner from final score
    black_score = game_record['score']
    white_score = 64 - black_score

    if black_score > white_score:
        winner = 1  # BLACK wins
        score_diff = black_score - white_score
    elif white_score > black_score:
        winner = -1  # WHITE wins
        score_diff = white_score - black_score  # Positive for winner
    else:
        winner = 0  # Draw
        score_diff = 0

    samples = []

    # Replay game and generate samples
    for move_num, action in enumerate(game_record['moves']):
        if board.is_terminal():
            break

        # Get current player perspective
        current_player = board.player

        # Encode state
        state = board.encode()  # Shape: (4, 8, 8)

        # Create policy (soft or one-hot)
        policy = np.zeros(65, dtype=np.float32)

        if soft_labels:
            # Soft labels: expert move gets 'confidence', rest distributed over legal moves
            valid_mask = board.valid_action_mask()
            legal_moves = [a for a in range(65) if valid_mask[a] > 0]

            policy[action] = confidence
            remaining = 1.0 - confidence

            # Distribute remaining probability over other legal moves
            other_legal = [m for m in legal_moves if m != action]
            if len(other_legal) > 0:
                for move in other_legal:
                    policy[move] = remaining / len(other_legal)
        else:
            # One-hot: expert move = 1.0
            policy[action] = 1.0

        # Compute value from current player's perspective
        if winner == 0:
            value_win = 0.0
        elif winner == current_player:
            value_win = 1.0
        else:
            value_win = -1.0

        # Normalize score: positive if current player won, negative if lost
        if winner == 0:
            value_score = 0.0
        elif winner == current_player:
            value_score = score_diff / 64.0
        else:
            value_score = -score_diff / 64.0

        # Compute auxiliary features
        mobility = compute_mobility_features(board)
        stability = compute_stability_map(board)
        corner = compute_corner_control(board)
        parity = compute_parity_features(board)

        # Determine phase
        empties = int((board.board == 0).sum())
        phase = determine_phase(empties)

        # Apply 8x dihedral augmentation
        for aug_planes, aug_policy, aug_stability, aug_corner, aug_parity in augment_state(
            state, policy, stability, corner, parity
        ):
            sample = {
                "state": aug_planes.astype(np.float32),
                "policy": aug_policy.astype(np.float32),
                "value_win": float(value_win),
                "value_score": float(value_score),
                "mobility": mobility.astype(np.float32),
                "stability": aug_stability.astype(np.float32),
                "corner": aug_corner.astype(np.float32),
                "parity": aug_parity.astype(np.float32),
                "phase": phase,
                "empties": empties,
            }
            samples.append(sample)

        # Apply the move
        try:
            board.step_action_index(action)
        except Exception as e:
            print(f"  Warning: Invalid move {action} at position {move_num}: {e}")
            break

    return samples


def save_samples_to_shards(samples, output_dir, samples_per_shard=10000):
    """
    Save IL samples to pickle shard files.

    Args:
        samples: List of sample dictionaries
        output_dir: Output directory path
        samples_per_shard: Number of samples per shard file

    Returns:
        Number of shards created
    """
    os.makedirs(output_dir, exist_ok=True)

    num_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard

    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min((shard_idx + 1) * samples_per_shard, len(samples))
        shard_samples = samples[start_idx:end_idx]

        shard_path = os.path.join(output_dir, f'il_shard_{shard_idx:04d}.pkl')
        with open(shard_path, 'wb') as f:
            pickle.dump(shard_samples, f)

        print(f"  Saved shard {shard_idx+1}/{num_shards}: {shard_path} ({len(shard_samples)} samples)")

    return num_shards


def main():
    parser = argparse.ArgumentParser(description='Parse WTHOR database for IL bootstrap')
    parser.add_argument('--input', type=str, default='data/wthor_raw',
                        help='Input directory containing .wtb files')
    parser.add_argument('--output', type=str, default='data/il_bootstrap',
                        help='Output directory for IL sample shards')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to generate (None = unlimited)')
    parser.add_argument('--samples-per-shard', type=int, default=10000,
                        help='Number of samples per shard file')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Test mode: parse only this single .wtb file')

    # Year filtering
    parser.add_argument('--min-year', type=int, default=None,
                        help='Minimum year to include (e.g., 2001)')
    parser.add_argument('--max-year', type=int, default=None,
                        help='Maximum year to include (e.g., 2015)')

    # Soft labels
    parser.add_argument('--soft-labels', action='store_true',
                        help='Use soft policy targets instead of one-hot')
    parser.add_argument('--confidence', type=float, default=0.8,
                        help='Confidence for expert move in soft labels (default: 0.8)')

    # Quality filtering
    parser.add_argument('--min-score-diff', type=int, default=0,
                        help='Minimum score difference for decisive games (default: 0 = all games)')
    parser.add_argument('--min-moves', type=int, default=0,
                        help='Minimum number of moves per game (default: 0 = all games)')

    args = parser.parse_args()

    print("="*80)
    print("WTHOR Database Parser for IL Bootstrap")
    print("="*80)
    print()

    # Configuration summary
    if args.soft_labels:
        print(f"Policy labels: SOFT (expert={args.confidence:.1%}, distributed over legal moves)")
    else:
        print(f"Policy labels: ONE-HOT (expert=100%)")

    if args.min_year or args.max_year:
        year_range = f"{args.min_year or 'any'}-{args.max_year or 'any'}"
        print(f"Year filter: {year_range}")

    if args.min_score_diff > 0:
        print(f"Quality filter: score_diff >= {args.min_score_diff} (decisive games only)")

    if args.min_moves > 0:
        print(f"Quality filter: moves >= {args.min_moves}")

    print()

    # Find .wtb files
    if args.test_file:
        wtb_files = [args.test_file]
        print(f"Test mode: parsing single file {args.test_file}")
    else:
        wtb_pattern = os.path.join(args.input, '*.wtb')
        all_wtb_files = sorted(glob.glob(wtb_pattern))

        # Apply year filtering
        wtb_files = []
        for wtb_file in all_wtb_files:
            basename = os.path.basename(wtb_file)
            # Extract year from filename: WTH_YYYY.wtb
            import re
            match = re.search(r'WTH_(\d{4})\.wtb', basename)
            if match:
                year = int(match.group(1))
                if args.min_year and year < args.min_year:
                    continue
                if args.max_year and year > args.max_year:
                    continue
                wtb_files.append(wtb_file)
            else:
                # Include files without year in filename
                wtb_files.append(wtb_file)

        print(f"Found {len(wtb_files)} .wtb files in {args.input}")
        if args.min_year or args.max_year:
            print(f"  ({len(all_wtb_files) - len(wtb_files)} files filtered by year)")

    if len(wtb_files) == 0:
        print(f"Error: No .wtb files found in {args.input}")
        return 1

    print()

    # Parse games and generate samples
    all_samples = []
    stats = {
        'total_files': len(wtb_files),
        'total_games': 0,
        'total_positions': 0,
        'total_samples': 0,
        'games_filtered': 0,
        'phase_counts': defaultdict(int),
    }

    for file_idx, wtb_file in enumerate(wtb_files):
        file_name = os.path.basename(wtb_file)
        print(f"[{file_idx+1}/{len(wtb_files)}] Parsing {file_name}...")

        # Parse games from file
        games = parse_wtb_file(wtb_file)
        print(f"  Found {len(games)} games")

        # Generate IL samples from each game
        file_samples = 0
        for game_idx, game_record in enumerate(games):
            # Apply quality filtering
            black_score = game_record['score']
            white_score = 64 - black_score
            score_diff = abs(black_score - white_score)

            if args.min_score_diff > 0 and score_diff < args.min_score_diff:
                stats['games_filtered'] += 1
                continue

            if args.min_moves > 0 and len(game_record['moves']) < args.min_moves:
                stats['games_filtered'] += 1
                continue

            samples = generate_il_samples_from_game(
                game_record,
                soft_labels=args.soft_labels,
                confidence=args.confidence
            )

            # Track statistics
            stats['total_positions'] += len(game_record['moves'])
            stats['total_samples'] += len(samples)
            file_samples += len(samples)

            for sample in samples:
                stats['phase_counts'][sample['phase']] += 1

            all_samples.extend(samples)

            # Check if we've reached max samples
            if args.max_samples and len(all_samples) >= args.max_samples:
                print(f"  Reached max samples limit ({args.max_samples}), stopping")
                all_samples = all_samples[:args.max_samples]
                break

        stats['total_games'] += len(games)
        print(f"  Generated {file_samples} IL samples (8x augmentation)")
        print()

        # Stop if max samples reached
        if args.max_samples and len(all_samples) >= args.max_samples:
            break

    # Save samples to shard files
    print("="*80)
    print("Saving IL samples to shards...")
    print("="*80)
    print()

    num_shards = save_samples_to_shards(all_samples, args.output, args.samples_per_shard)

    # Print summary statistics
    print()
    print("="*80)
    print("PARSING COMPLETE")
    print("="*80)
    print()
    print(f"Files parsed: {stats['total_files']}")
    print(f"Games parsed: {stats['total_games']}")
    if stats['games_filtered'] > 0:
        print(f"Games filtered: {stats['games_filtered']} (quality threshold)")
    print(f"Positions: {stats['total_positions']}")
    print(f"IL samples (with augmentation): {stats['total_samples']}")
    print(f"Shards created: {num_shards}")
    print()
    print("Phase distribution:")
    for phase in ['opening', 'midgame', 'endgame']:
        count = stats['phase_counts'][phase]
        pct = 100.0 * count / max(1, stats['total_samples'])
        print(f"  {phase:>8}: {count:>7} samples ({pct:>5.1f}%)")
    print()
    print(f"Output directory: {args.output}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
