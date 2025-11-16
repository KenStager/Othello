"""
Oracle bridge for endgame solving using Edax.

Provides exact endgame evaluations when empties <= threshold (typically 14).
Uses subprocess interface to Edax binary.
"""
import subprocess
import os
import numpy as np
from ..othello.board import BLACK, WHITE, EMPTY


def board_to_edax_string(board):
    """
    Convert board to Edax format string.

    Edax format: 64 characters, one per square (row-major)
    'X' = BLACK, 'O' = WHITE, '-' = EMPTY
    Followed by player to move ('X' or 'O')

    Example: "---XO--OX---" + " X"
    """
    mapping = {BLACK: 'X', WHITE: 'O', EMPTY: '-'}
    board_str = ''.join(mapping[board.board.flat[i]] for i in range(64))
    player_str = 'X' if board.player == BLACK else 'O'
    return board_str + ' ' + player_str


def edax_string_to_board(edax_str):
    """
    Convert Edax format string to board.

    Args:
        edax_str: String like "---XO...--- X"

    Returns:
        Board object
    """
    from ..othello.board import Board

    parts = edax_str.strip().split()
    board_str = parts[0]
    player_str = parts[1] if len(parts) > 1 else 'X'

    b = Board(8)
    mapping = {'X': BLACK, 'O': WHITE, '-': EMPTY}

    for i, char in enumerate(board_str[:64]):
        r, c = divmod(i, 8)
        b.board[r, c] = mapping.get(char, EMPTY)

    b.player = BLACK if player_str == 'X' else WHITE
    return b


class EdaxOracle:
    """
    Oracle for exact endgame evaluation using Edax.

    Uses subprocess to call Edax binary.
    """

    def __init__(self, edax_path="third_party/edax/bin/edax", time_limit_ms=100, empties_threshold=14):
        """
        Initialize Edax oracle.

        Args:
            edax_path: Path to Edax binary
            time_limit_ms: Time limit for Edax search (ms)
            empties_threshold: Only use oracle when empties <= this value
        """
        self.edax_path = edax_path
        self.time_limit_ms = time_limit_ms
        self.empties_threshold = empties_threshold
        self.call_count = 0
        self.cache = {}  # Simple cache for repeated positions

    def is_available(self):
        """Check if Edax binary is available."""
        return os.path.exists(self.edax_path) and os.access(self.edax_path, os.X_OK)

    def should_use(self, board):
        """
        Determine if oracle should be used for this position.

        Args:
            board: Board object

        Returns:
            bool: True if empties <= threshold
        """
        empties = int(np.sum(board.board == EMPTY))
        return empties <= self.empties_threshold

    def evaluate(self, board):
        """
        Get exact evaluation from Edax.

        Args:
            board: Board object

        Returns:
            dict with:
                'value': Exact disc differential from current player's perspective
                'best_move': (r, c) tuple or None for pass
                'is_exact': Always True for oracle
        """
        if not self.is_available():
            raise RuntimeError(f"Edax binary not found at {self.edax_path}")

        # Check cache
        board_str = board_to_edax_string(board)
        if board_str in self.cache:
            return self.cache[board_str]

        self.call_count += 1

        try:
            # Call Edax via subprocess
            # Use stdin to send commands: mode 1 (text mode), setboard, solve
            # Edax must be run from its directory to access eval.dat
            edax_dir = os.path.dirname(os.path.dirname(self.edax_path))  # Get edax root dir

            # Prepare Edax commands
            empties = int((board.board == EMPTY).sum())
            commands = f"mode 1\nsetboard {board_str}\nsolve {empties}\nq\n"

            result = subprocess.run(
                [self.edax_path],
                input=commands,
                capture_output=True,
                text=True,
                timeout=self.time_limit_ms / 1000.0 + 1.0,  # Add 1s buffer
                cwd=edax_dir  # Run from edax directory
            )

            # Parse Edax output
            # This is a placeholder - actual parsing depends on Edax output format
            # Typical Edax output includes: score, best move, etc.
            output = result.stdout

            # Placeholder parsing (adjust based on actual Edax format)
            # For now, return a simple result
            score = self._parse_edax_score(output)
            best_move = self._parse_edax_best_move(output)

            result_dict = {
                'value': score,
                'best_move': best_move,
                'is_exact': True,
                'raw_output': output
            }

            # Cache result
            self.cache[board_str] = result_dict
            return result_dict

        except subprocess.TimeoutExpired:
            # Fallback if Edax times out
            return {
                'value': 0.0,
                'best_move': None,
                'is_exact': False,
                'error': 'timeout'
            }
        except Exception as e:
            # Fallback on error
            return {
                'value': 0.0,
                'best_move': None,
                'is_exact': False,
                'error': str(e)
            }

    def _parse_edax_score(self, output):
        """
        Parse score from Edax output.

        Edax outputs a table where each line has format:
        # | depth|score| time   |  nodes (N)  |   N/s    | principal variation
        1|   14   +18     0:00.002      94022   47011000 g8 H7 a8...

        We extract the score from the first result line (best move).
        Score is in discs (e.g., +18 means +18 disc advantage).
        Convert to normalized value in [-1, 1].
        """
        try:
            for line in output.split('\n'):
                # Skip header and separator lines
                if '---+' in line or '# |' in line:
                    continue
                # Look for result lines (start with number|)
                if '|' in line and not line.strip().startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        # Score is in column 3 (index 2)
                        score_str = parts[2].strip()
                        # Parse score (e.g., "+18" or "-04")
                        if score_str:
                            score_discs = int(score_str)
                            # Convert to normalized value: discs / 64
                            # Positive = current player winning
                            return score_discs / 64.0
        except Exception as e:
            print(f"Warning: Failed to parse Edax score: {e}")
        return 0.0

    def _parse_edax_best_move(self, output):
        """
        Parse best move from Edax output.

        Extract first move from principal variation (last column).
        Edax uses lowercase notation (a1-h8) or "pa" for pass.
        Convert to our action index format [0-64].
        """
        try:
            for line in output.split('\n'):
                # Skip header and separator lines
                if '---+' in line or '# |' in line:
                    continue
                # Look for result lines
                if '|' in line and not line.strip().startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 7:
                        # Principal variation is in last column (index 6)
                        pv_str = parts[6].strip()
                        if pv_str:
                            # First move is first token
                            first_move = pv_str.split()[0].lower()
                            # Convert Edax notation to action index
                            if first_move == "pa" or first_move == "ps":
                                return 64  # Pass action
                            elif len(first_move) == 2:
                                col = ord(first_move[0]) - ord('a')  # 0-7
                                row = int(first_move[1]) - 1  # 0-7
                                if 0 <= row <= 7 and 0 <= col <= 7:
                                    return row * 8 + col  # [0-63]
        except Exception as e:
            print(f"Warning: Failed to parse Edax best move: {e}")
        return None

    def get_stats(self):
        """Return oracle usage statistics."""
        return {
            'call_count': self.call_count,
            'cache_size': len(self.cache)
        }


# Placeholder implementation for when Edax is not available
class DummyOracle:
    """Dummy oracle that returns zero evaluations (for testing without Edax)."""

    def __init__(self, empties_threshold=14):
        self.empties_threshold = empties_threshold

    def is_available(self):
        return True

    def should_use(self, board):
        empties = int(np.sum(board.board == EMPTY))
        return empties <= self.empties_threshold

    def evaluate(self, board):
        return {
            'value': 0.0,
            'best_move': None,
            'is_exact': False,
            'dummy': True
        }

    def get_stats(self):
        return {'dummy': True}


def create_oracle(cfg):
    """
    Factory function to create oracle based on config.

    Args:
        cfg: Config dict with oracle settings

    Returns:
        EdaxOracle or DummyOracle
    """
    oracle_cfg = cfg.get('oracle', {})

    if not oracle_cfg.get('use', False):
        return DummyOracle(empties_threshold=oracle_cfg.get('empties_threshold', 14))

    edax_path = oracle_cfg.get('edax_path', 'third_party/edax/bin/edax')
    time_limit = oracle_cfg.get('time_limit_ms', 100)
    threshold = oracle_cfg.get('empties_threshold', 14)

    oracle = EdaxOracle(edax_path=edax_path, time_limit_ms=time_limit, empties_threshold=threshold)

    if not oracle.is_available():
        print(f"Warning: Edax not found at {edax_path}, using dummy oracle")
        return DummyOracle(empties_threshold=threshold)

    return oracle
