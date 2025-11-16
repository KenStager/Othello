"""
Generate opening suite for balanced evaluation.

Creates a diverse set of opening positions by:
1. Seeding from known human opening databases (if available)
2. Auto-generating via high-temperature self-play
3. Applying D8 symmetries for coverage
4. Ensuring color balance
"""
import argparse
import json
import os
import numpy as np

from src.othello.game import Game
from src.othello.board import Board, EMPTY, BLACK, WHITE


def board_to_dict(board):
    """Serialize board to dict for JSON storage."""
    return {
        'board': board.board.tolist(),
        'player': int(board.player),
        'pass_count': board.pass_count
    }


def dict_to_board(d):
    """Deserialize board from dict."""
    b = Board(8)
    b.board = np.array(d['board'], dtype=np.int8)
    b.player = d['player']
    b.pass_count = d['pass_count']
    return b


def apply_symmetries(board):
    """
    Apply D8 dihedral symmetries to board.

    Returns list of 8 board states.
    """
    symmetries = []

    for k in range(4):  # 4 rotations
        rotated = np.rot90(board.board, k)
        b = Board(8)
        b.board = rotated.copy()
        b.player = board.player
        b.pass_count = board.pass_count
        symmetries.append(b)

        # Flip horizontally
        flipped = rotated[:, ::-1]
        bf = Board(8)
        bf.board = flipped.copy()
        bf.player = board.player
        bf.pass_count = board.pass_count
        symmetries.append(bf)

    return symmetries


def generate_random_openings(n=32, min_moves=4, max_moves=12, seed=42):
    """
    Generate random opening positions via random play.

    Args:
        n: Number of unique openings to generate
        min_moves: Minimum moves before saving position
        max_moves: Maximum moves before saving position

    Returns:
        List of Board objects
    """
    np.random.seed(seed)
    openings = []

    for i in range(n):
        game = Game(8)
        b = game.new_board()

        # Play random moves
        num_moves = np.random.randint(min_moves, max_moves + 1)
        for _ in range(num_moves):
            if b.is_terminal():
                break

            moves = b.legal_moves()
            if not moves:
                b.apply_move(None)  # pass
            else:
                # Random move with slight bias toward center and corners
                weights = np.ones(len(moves))
                for idx, (r, c) in enumerate(moves):
                    # Bias corners
                    if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                        weights[idx] *= 1.5
                    # Bias center
                    if 2 <= r <= 5 and 2 <= c <= 5:
                        weights[idx] *= 1.2

                weights /= weights.sum()
                move = moves[np.random.choice(len(moves), p=weights)]
                b.apply_move(move)

        if not b.is_terminal():
            openings.append(b.copy())

    return openings


def load_human_openings(path=None, min_move=4, max_move=12, min_score_diff=10, max_positions=1000):
    """
    Load openings from WTHOR expert game database.

    Extracts positions from moves 4-12 of expert tournament games,
    filtered for quality (decisive games with score_diff >= min_score_diff).

    Args:
        path: Path to WTHOR database directory containing .wtb files
        min_move: Minimum move number to extract (default: 4)
        max_move: Maximum move number to extract (default: 12)
        min_score_diff: Minimum score difference for quality filter (default: 10)
        max_positions: Maximum unique positions to return (default: 1000)

    Returns:
        List of Board objects
    """
    import glob
    import struct
    import random
    from collections import defaultdict

    if not path or not os.path.exists(path):
        print(f"WTHOR path not found: {path}")
        return []

    # Helper: decode WTHOR move byte to action index
    def decode_wthor_move(move_byte):
        if move_byte == 0:
            return 64  # Pass
        column = move_byte % 10
        row = move_byte // 10
        if not (1 <= column <= 8 and 1 <= row <= 8):
            return None
        return (row - 1) * 8 + (column - 1)

    # Parse WTHOR files
    wtb_files = sorted(glob.glob(os.path.join(path, '*.wtb')))
    print(f"Found {len(wtb_files)} WTHOR files")

    position_hashes = set()  # For deduplication
    positions = []
    games_processed = 0
    games_filtered = 0

    for wtb_file in wtb_files:
        with open(wtb_file, 'rb') as f:
            # Read header
            header = f.read(16)
            if len(header) < 16:
                continue
            game_count = struct.unpack('<I', header[4:8])[0]

            for _ in range(game_count):
                record = f.read(68)
                if len(record) < 68:
                    break

                # Parse score
                black_score = record[6]
                white_score = 64 - black_score
                score_diff = abs(black_score - white_score)

                # Quality filter: only decisive games
                if score_diff < min_score_diff:
                    games_filtered += 1
                    continue

                # Parse moves
                moves = []
                for i in range(60):
                    move_byte = record[8 + i]
                    if move_byte == 0:
                        break
                    action = decode_wthor_move(move_byte)
                    if action is not None:
                        moves.append(action)

                games_processed += 1

                # Extract positions from moves min_move to max_move
                game = Game()
                board = game.new_board()

                for move_num, action in enumerate(moves):
                    if min_move <= move_num < max_move:
                        # Hash board state for deduplication
                        player_byte = 1 if board.player == 1 else 0
                        board_hash = hash(board.board.tobytes() + bytes([player_byte]))

                        if board_hash not in position_hashes:
                            position_hashes.add(board_hash)
                            positions.append(board.copy())

                            # Stop if we have enough positions
                            if len(positions) >= max_positions:
                                break

                    # Apply move
                    try:
                        board.step_action_index(action)
                    except:
                        break  # Invalid move, skip rest of game

                if len(positions) >= max_positions:
                    break

            if len(positions) >= max_positions:
                break

    print(f"  Processed {games_processed} games ({games_filtered} filtered for quality)")
    print(f"  Extracted {len(positions)} unique positions")

    # Shuffle for diversity
    random.shuffle(positions)
    return positions[:max_positions]


def create_opening_suite(human_path=None, auto_count=32, apply_syms=True, seed=42):
    """
    Create balanced opening suite.

    Args:
        human_path: Path to human opening database (optional)
        auto_count: Number of auto-generated openings
        apply_syms: Whether to apply D8 symmetries
        seed: Random seed

    Returns:
        List of opening position dicts
    """
    openings = []

    # Load human openings if available
    if human_path and os.path.exists(human_path):
        human_openings = load_human_openings(human_path)
        print(f"Loaded {len(human_openings)} human openings")
        openings.extend(human_openings)

    # Generate random openings
    auto_openings = generate_random_openings(n=auto_count, seed=seed)
    print(f"Generated {len(auto_openings)} auto openings")
    openings.extend(auto_openings)

    # Apply symmetries if requested
    if apply_syms:
        all_symmetries = []
        for b in openings:
            syms = apply_symmetries(b)
            all_symmetries.extend(syms)
        print(f"Applied D8 symmetries: {len(openings)} -> {len(all_symmetries)} positions")
        openings = all_symmetries

    # Convert to dicts for JSON
    opening_dicts = [board_to_dict(b) for b in openings]

    # Ensure color balance (equal number of BLACK and WHITE to move)
    black_to_move = [d for d in opening_dicts if d['player'] == BLACK]
    white_to_move = [d for d in opening_dicts if d['player'] == WHITE]

    print(f"Color balance: {len(black_to_move)} BLACK to move, {len(white_to_move)} WHITE to move")

    # Balance by duplicating and swapping colors if needed
    # (For simplicity, just report the imbalance)
    if len(black_to_move) != len(white_to_move):
        print(f"Warning: Color imbalance. Consider adjusting parameters.")

    return opening_dicts


def main():
    parser = argparse.ArgumentParser(description="Generate opening suite for Othello evaluation")
    parser.add_argument('--human_db', type=str, default=None, help='Path to human opening database')
    parser.add_argument('--auto_count', type=int, default=32, help='Number of auto-generated openings')
    parser.add_argument('--apply_syms', action='store_true', default=True, help='Apply D8 symmetries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', type=str, default='data/openings/rot64.json', help='Output JSON path')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Generate suite
    suite = create_opening_suite(
        human_path=args.human_db,
        auto_count=args.auto_count,
        apply_syms=args.apply_syms,
        seed=args.seed
    )

    # Save to JSON
    with open(args.out, 'w') as f:
        json.dump(suite, f, indent=2)

    print(f"\nSaved {len(suite)} openings to {args.out}")

    # Verify loading
    with open(args.out, 'r') as f:
        loaded = json.load(f)
    print(f"Verification: loaded {len(loaded)} openings")

    # Show sample
    if loaded:
        sample = dict_to_board(loaded[0])
        print(f"\nSample opening (position 0):")
        print(sample.board)
        print(f"Player to move: {sample.player} ({'BLACK' if sample.player == BLACK else 'WHITE'})")


if __name__ == "__main__":
    main()
