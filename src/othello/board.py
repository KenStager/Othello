import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1

DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class Board:
    def __init__(self, n=8):
        assert n == 8, "Only 8x8 supported in this minimal scaffold."
        self.n = n
        self.reset()

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=np.int8)
        mid = self.n//2
        # Standard Othello initial position
        self.board[mid-1, mid-1] = WHITE
        self.board[mid, mid] = WHITE
        self.board[mid-1, mid] = BLACK
        self.board[mid, mid-1] = BLACK
        self.player = BLACK      # BLACK starts
        self.pass_count = 0

    def copy(self):
        b = Board(self.n)
        b.board = self.board.copy()
        b.player = self.player
        b.pass_count = self.pass_count
        return b

    def in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def legal_moves(self, player=None):
        if player is None:
            player = self.player
        opp = -player
        moves = []
        for r in range(self.n):
            for c in range(self.n):
                if self.board[r, c] != EMPTY:
                    continue
                if self._would_flip(r, c, player, opp):
                    moves.append((r, c))
        return moves

    def has_legal_move(self, player=None):
        return len(self.legal_moves(player)) > 0

    def _would_flip(self, r, c, player, opp):
        # Check each direction for a contiguous line of opp stones ending in a player stone
        flipped_any = False
        for dr, dc in DIRECTIONS:
            rr, cc = r + dr, c + dc
            count_opp = 0
            while self.in_bounds(rr, cc) and self.board[rr, cc] == opp:
                rr += dr; cc += dc
                count_opp += 1
            if count_opp > 0 and self.in_bounds(rr, cc) and self.board[rr, cc] == player:
                flipped_any = True
        return flipped_any

    def apply_move(self, move):
        # move: (r,c) or None for pass
        if move is None:
            self.player = -self.player
            self.pass_count += 1
            return
        (r, c) = move
        player = self.player
        opp = -player
        assert self.board[r, c] == EMPTY
        flipped = []
        for dr, dc in DIRECTIONS:
            path = []
            rr, cc = r + dr, c + dc
            while self.in_bounds(rr, cc) and self.board[rr, cc] == opp:
                path.append((rr, cc))
                rr += dr; cc += dc
            if len(path) > 0 and self.in_bounds(rr, cc) and self.board[rr, cc] == player:
                flipped.extend(path)
        if not flipped:
            raise ValueError("Illegal move")
        self.board[r, c] = player
        for (rr, cc) in flipped:
            self.board[rr, cc] = player
        self.player = -self.player
        self.pass_count = 0

    def is_terminal(self):
        if self.pass_count >= 2:
            return True
        if (self.board == EMPTY).sum() == 0:
            return True
        return False

    def result(self):
        # returns winner (+1 black, -1 white, 0 draw) and score diff
        black = (self.board == BLACK).sum()
        white = (self.board == WHITE).sum()
        if black > white:
            return BLACK, black - white
        elif white > black:
            return WHITE, white - black
        else:
            return 0, 0

    # Encoding for NN: planes [current, opponent, valid_mask, player_plane]
    def encode(self):
        cur = (self.board == self.player).astype(np.float32)
        opp = (self.board == -self.player).astype(np.float32)
        valid = np.zeros_like(cur, dtype=np.float32)
        for (r, c) in self.legal_moves(self.player):
            valid[r, c] = 1.0
        player_plane = np.full_like(cur, 1.0 if self.player == BLACK else 0.0, dtype=np.float32)
        planes = np.stack([cur, opp, valid, player_plane], axis=0)  # (4, 8, 8)
        return planes

    def action_space(self):
        # 64 board cells + 1 pass = 65 actions (index 64 is pass)
        return 65

    def valid_action_mask(self):
        mask = np.zeros(65, dtype=np.float32)
        moves = self.legal_moves(self.player)
        if len(moves) == 0:
            mask[64] = 1.0  # pass
        else:
            for (r, c) in moves:
                mask[r * self.n + c] = 1.0
        return mask

    def step_action_index(self, a):
        # a in [0..64], 64==pass
        if a == 64:
            if self.has_legal_move(self.player):
                raise ValueError("Pass chosen but legal moves exist.")
            self.apply_move(None)
            return
        r, c = divmod(int(a), self.n)
        if not self._would_flip(r, c, self.player, -self.player):
            raise ValueError("Illegal move index")
        self.apply_move((r, c))

    @staticmethod
    def symmetries(planes, pi):
        # 8 dihedral symmetries for 8x8. pi is 65-d (last is pass).
        syms = []
        board = planes
        policy = pi[:-1].reshape(8,8)
        pass_val = pi[-1]
        for k in range(4):
            rot_p = np.rot90(board, k, axes=(1,2))
            rot_pi = np.rot90(policy, k)
            syms.append((rot_p, np.concatenate([rot_pi.flatten(), [pass_val]])))
            # reflection (flip LR) after rotation
            fl_p = rot_p[:, :, ::-1]
            fl_pi = rot_pi[:, ::-1]
            syms.append((fl_p, np.concatenate([fl_pi.flatten(), [pass_val]])))
        return syms
