import math, numpy as np

class MCTSNode:
    __slots__ = ('P','N','W','Q','children','is_expanded','terminal','player_to_move','valid_mask')
    def __init__(self, prior, player_to_move):
        self.P = prior   # prior prob for each action
        self.N = np.zeros_like(prior, dtype=np.int32)
        self.W = np.zeros_like(prior, dtype=np.float32)
        self.Q = np.zeros_like(prior, dtype=np.float32)
        self.children = {}      # a -> node
        self.is_expanded = False
        self.terminal = False
        self.player_to_move = player_to_move  # +1 or -1
        self.valid_mask = None

def softmax_masked(logits, mask, temp=1.0):
    logits = logits - logits.max()
    exp = np.exp(logits / max(1e-8, temp)) * mask
    s = exp.sum()
    if s <= 0:
        # if mask empty (shouldn't happen), make uniform over mask
        exp = mask
        s = exp.sum()
    return exp / s

class MCTS:
    def __init__(self, game_cls, net, device, cpuct=1.5, simulations=200,
                 dir_alpha=0.15, dir_frac=0.25, reuse_tree=True):
        self.game_cls = game_cls
        self.net = net
        self.device = device
        self.cpuct = cpuct
        self.simulations = simulations
        self.dir_alpha = dir_alpha
        self.dir_frac = dir_frac
        self.reuse_tree = reuse_tree
        self.root = None

    def reset_root(self):
        self.root = None

    def run(self, board):
        # Create / update root
        if self.root is None or not self.reuse_tree:
            self.root = None
        if self.root is None:
            self.root = self._expand_root(board)
        else:
            # if board advanced, we cannot easily reuse w/o action mapping; keep it simple
            self.root = self._expand_root(board)

        # Dirichlet noise at root
        if self.dir_alpha is not None and self.dir_frac and self.dir_frac > 0:
            mask = self.root.valid_mask
            valid_idx = np.where(mask > 0.0)[0]
            if len(valid_idx) > 1:
                noise = np.random.dirichlet([self.dir_alpha] * len(valid_idx))
                P = self.root.P.copy()
                P[valid_idx] = (1 - self.dir_frac) * P[valid_idx] + self.dir_frac * noise
                self.root.P = P

        for _ in range(self.simulations):
            self._simulate(board.copy(), self.root)

        # Return visit counts as policy target
        N = self.root.N.astype(np.float32)
        if N.sum() == 0:
            pi = self.root.P.copy()
        else:
            pi = N / N.sum()
        return pi

    def _expand_root(self, board):
        planes = board.encode()[None, ...]  # (1,4,8,8)
        import torch
        with torch.no_grad():
            logits, value = self.net(torch.tensor(planes, dtype=torch.float32, device=self.device))
            logits = logits[0].cpu().numpy()
            value = float(value[0].cpu().item())
        mask = board.valid_action_mask()
        probs = softmax_masked(logits, mask, temp=1.0)
        node = MCTSNode(prior=probs, player_to_move=board.player)
        node.is_expanded = True
        node.valid_mask = mask
        node.terminal = board.is_terminal()
        node.Q[:] = 0.0
        return node

    def _policy_value(self, board):
        planes = board.encode()[None, ...]
        import torch
        with torch.no_grad():
            logits, value = self.net(torch.tensor(planes, dtype=torch.float32, device=self.device))
            logits = logits[0].cpu().numpy()
            value = float(value[0].cpu().item())
        mask = board.valid_action_mask()
        probs = softmax_masked(logits, mask, temp=1.0)
        return probs, value, mask

    def _simulate(self, board, node):
        if node.terminal:
            # terminal value from POV of player_to_move at node is 0 (game ended before moving)
            winner, _ = board.result()
            if winner == 0:
                return 0.0
            return 1.0 if winner == node.player_to_move else -1.0

        # Select
        best_a = None
        best_score = -1e9
        sqrt_sum = math.sqrt(max(1, node.N.sum()))
        for a in range(len(node.P)):
            if node.valid_mask[a] <= 0.0:
                continue
            # PUCT
            Q = node.Q[a]
            U = self.cpuct * node.P[a] * (sqrt_sum / (1 + node.N[a]))
            score = Q + U
            if score > best_score:
                best_score = score
                best_a = a

        # Apply action
        current_player = board.player
        board.step_action_index(best_a)

        # Expand
        child = node.children.get(best_a)
        if child is None:
            probs, value, mask = self._policy_value(board)
            child = MCTSNode(prior=probs, player_to_move=board.player)
            child.is_expanded = True
            child.valid_mask = mask
            child.terminal = board.is_terminal()
            node.children[best_a] = child
            # Back up value from the perspective of node.player_to_move
            v = value
        else:
            v = self._simulate(board, child)

        # If player switched, flip sign for backup
        if board.player != current_player:
            # After stepping, board.player is next to play (opponent).
            # Value v is from child's POV (next player), so negate to current player's POV.
            v = -v

        # Backup
        node.N[best_a] += 1
        node.W[best_a] += v
        node.Q[best_a] = node.W[best_a] / node.N[best_a]
        return v
