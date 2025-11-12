import numpy as np

from .board import BLACK, WHITE, Board


def _normalize_count(count: int, max_count: float = 20.0) -> float:
    return float(np.clip(count / max_count, 0.0, 1.0))


def compute_mobility_features(board: Board) -> np.ndarray:
    cur_moves = len(board.legal_moves(board.player))
    opp_moves = len(board.legal_moves(-board.player))
    return np.array([
        _normalize_count(cur_moves),
        _normalize_count(opp_moves),
    ], dtype=np.float32)


def _line_stable(arr: np.ndarray, stable_mask: np.ndarray, r: int, c: int, color: int, dr: int, dc: int) -> bool:
    n = arr.shape[0]
    for sign in (1, -1):
        rr, cc = r + sign * dr, c + sign * dc
        while 0 <= rr < n and 0 <= cc < n:
            val = arr[rr, cc]
            if val == color:
                rr += sign * dr
                cc += sign * dc
                continue
            if val == 0:
                return False
            if val == -color and not stable_mask[rr, cc]:
                return False
            break
    return True


def compute_stability_map(board: Board) -> np.ndarray:
    arr = board.board
    n = arr.shape[0]
    stability = np.zeros((2, n, n), dtype=np.float32)
    for idx, color in enumerate((BLACK, WHITE)):
        stable_mask = np.zeros((n, n), dtype=bool)
        changed = True
        while changed:
            changed = False
            for r in range(n):
                for c in range(n):
                    if arr[r, c] != color or stable_mask[r, c]:
                        continue
                    if all(
                        _line_stable(arr, stable_mask, r, c, color, dr, dc)
                        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
                    ):
                        stable_mask[r, c] = True
                        changed = True
        stability[idx] = stable_mask.astype(np.float32)
    return stability


def compute_corner_control(board: Board) -> np.ndarray:
    corners = np.array(
        [
            board.board[0, 0],
            board.board[0, board.n - 1],
            board.board[board.n - 1, 0],
            board.board[board.n - 1, board.n - 1],
        ],
        dtype=np.int8,
    )
    corner_control = (corners == board.player).astype(np.float32)
    return corner_control


def compute_parity_features(board: Board) -> np.ndarray:
    empties = (board.board == 0)
    overall = int(empties.sum() % 2)
    mid = board.n // 2
    quadrants = np.array(
        [
            int(empties[:mid, :mid].sum() % 2),
            int(empties[:mid, mid:].sum() % 2),
            int(empties[mid:, :mid].sum() % 2),
            int(empties[mid:, mid:].sum() % 2),
        ],
        dtype=np.int32,
    )
    return np.concatenate(([overall], quadrants.astype(np.float32))).astype(np.float32)


def augment_state(planes: np.ndarray, policy: np.ndarray, stability: np.ndarray, corner: np.ndarray, parity: np.ndarray):
    board_policy = policy[:-1].reshape(8, 8)
    pass_prob = policy[-1]
    corner_matrix = corner.reshape(2, 2)
    parity_global = parity[0]
    parity_quadrants = parity[1:].reshape(2, 2)

    for k in range(4):
        rot_planes = np.rot90(planes, k, axes=(1, 2))
        rot_policy = np.rot90(board_policy, k)
        rot_stability = np.rot90(stability, k, axes=(1, 2))
        rot_corner = np.rot90(corner_matrix, k)
        rot_parity = np.rot90(parity_quadrants, k)

        yield (
            np.ascontiguousarray(rot_planes),
            np.concatenate([rot_policy.flatten(), [pass_prob]]).astype(np.float32),
            np.ascontiguousarray(rot_stability),
            rot_corner.reshape(-1).astype(np.float32),
            np.concatenate([[parity_global], rot_parity.reshape(-1)]).astype(np.float32),
        )

        flip_planes = rot_planes[:, :, ::-1]
        flip_policy = rot_policy[:, ::-1]
        flip_stability = rot_stability[:, :, ::-1]
        flip_corner = rot_corner[:, ::-1]
        flip_parity = rot_parity[:, ::-1]

        yield (
            np.ascontiguousarray(flip_planes),
            np.concatenate([flip_policy.flatten(), [pass_prob]]).astype(np.float32),
            np.ascontiguousarray(flip_stability),
            flip_corner.reshape(-1).astype(np.float32),
            np.concatenate([[parity_global], flip_parity.reshape(-1)]).astype(np.float32),
        )
