"""
Zobrist hashing for Othello board states.

Used for transposition table to detect repeated positions and reuse MCTS subtrees.
"""
import numpy as np

# Fixed seed for reproducibility
_rng = np.random.RandomState(12345)

# Zobrist keys: random 64-bit integers
Z_SIDE = _rng.randint(0, 2**63, dtype=np.uint64)  # For side-to-move
Z_BLACK = _rng.randint(0, 2**63, size=64, dtype=np.uint64)  # For black pieces
Z_WHITE = _rng.randint(0, 2**63, size=64, dtype=np.uint64)  # For white pieces
Z_PASS = _rng.randint(0, 2**63, dtype=np.uint64)  # For pass state


def zobrist_hash(board):
    """
    Compute Zobrist hash for an Othello board.

    Args:
        board: Board object with .board (8x8 numpy array) and .player (BLACK=1, WHITE=-1)

    Returns:
        uint64 hash value
    """
    h = np.uint64(0)

    # Hash side to move
    if board.player == 1:  # BLACK to move
        h ^= Z_SIDE

    # Hash piece positions
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            cell = board.board[r, c]
            if cell == 1:  # BLACK
                h ^= Z_BLACK[idx]
            elif cell == -1:  # WHITE
                h ^= Z_WHITE[idx]
            # cell == 0 (EMPTY) contributes nothing

    # Optional: hash pass state if tracking consecutive passes
    if hasattr(board, 'pass_count') and board.pass_count > 0:
        h ^= Z_PASS

    return h


def incremental_hash_move(current_hash, board_before, board_after, move_action):
    """
    Update hash incrementally after a move (faster than recomputing).

    This is an optimization for future use. Currently we recompute hashes.

    Args:
        current_hash: Hash before move
        board_before: Board state before move
        board_after: Board state after move
        move_action: Action taken (row, col) or None for pass

    Returns:
        Updated hash
    """
    # Flip side-to-move
    h = current_hash ^ Z_SIDE

    if move_action is None:
        # Pass move: just flip side
        return h

    # XOR out old pieces and XOR in new pieces
    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            old_val = board_before.board[r, c]
            new_val = board_after.board[r, c]

            if old_val != new_val:
                # Remove old piece (if any)
                if old_val == 1:
                    h ^= Z_BLACK[idx]
                elif old_val == -1:
                    h ^= Z_WHITE[idx]

                # Add new piece
                if new_val == 1:
                    h ^= Z_BLACK[idx]
                elif new_val == -1:
                    h ^= Z_WHITE[idx]

    return h


def test_zobrist_consistency():
    """Test that identical boards produce identical hashes."""
    from ..othello.board import Board

    b1 = Board(8)
    b2 = Board(8)

    h1 = zobrist_hash(b1)
    h2 = zobrist_hash(b2)

    assert h1 == h2, "Identical boards should have identical hashes"

    # Make a move on b1
    moves = b1.legal_moves()
    if moves:
        b1.apply_move(moves[0])
        h3 = zobrist_hash(b1)
        assert h3 != h1, "Different boards should have different hashes (with high probability)"

    print("Zobrist hash consistency test passed")


if __name__ == "__main__":
    test_zobrist_consistency()
