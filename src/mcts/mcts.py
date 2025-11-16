import math, numpy as np
from .zobrist import zobrist_hash
from .batch_evaluator import SimpleBatchEvaluator, DirectEvaluator

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
                 dir_alpha=0.15, dir_frac=0.25, reuse_tree=True, use_tt=True,
                 batch_size=32, use_batching=True):
        self.game_cls = game_cls
        self.net = net
        self.device = device
        self.cpuct = cpuct
        self.simulations = simulations
        self.dir_alpha = dir_alpha
        self.dir_frac = dir_frac
        self.reuse_tree = reuse_tree
        self.use_tt = use_tt
        self.root = None
        self.transposition_table = {}  # zobrist_hash -> MCTSNode
        self.tt_hits = 0
        self.tt_misses = 0

        # Stage 1 Batching: Add batch evaluator
        self.batch_size = batch_size
        self.use_batching = use_batching
        if use_batching:
            self.batch_evaluator = SimpleBatchEvaluator(net, device, batch_size)
        else:
            self.batch_evaluator = DirectEvaluator(net, device)
        self.leaf_queue = []  # Queue for collecting leaves before batch evaluation

    def reset_root(self):
        self.root = None

    def clear_tt(self):
        """Clear transposition table (use between games)."""
        self.transposition_table.clear()
        self.tt_hits = 0
        self.tt_misses = 0

    def get_tt_stats(self):
        """Return transposition table statistics."""
        total = self.tt_hits + self.tt_misses
        hit_rate = self.tt_hits / max(1, total)
        return {
            'size': len(self.transposition_table),
            'hits': self.tt_hits,
            'misses': self.tt_misses,
            'hit_rate': hit_rate
        }

    def run(self, board):
        """
        Run MCTS simulations and return policy.

        Uses batched inference if use_batching=True (Stage 1).
        Falls back to sequential if use_batching=False.
        """
        if self.use_batching:
            return self._run_batched(board)
        else:
            return self._run_sequential(board)

    def _run_sequential(self, board):
        """Original sequential MCTS (for comparison and fallback)"""
        # Create / update root using TT if available
        board_hash = zobrist_hash(board) if self.use_tt else None

        if self.root is None or not self.reuse_tree:
            self.root = None

        if self.root is None:
            # Try to find in transposition table
            if self.use_tt and board_hash in self.transposition_table:
                self.root = self.transposition_table[board_hash]
                self.tt_hits += 1
            else:
                self.root = self._expand_root(board)
                if self.use_tt and board_hash is not None:
                    self.transposition_table[board_hash] = self.root
                self.tt_misses += 1
        else:
            # Try to reuse from TT, otherwise re-expand
            if self.use_tt and board_hash in self.transposition_table:
                self.root = self.transposition_table[board_hash]
                self.tt_hits += 1
            else:
                self.root = self._expand_root(board)
                if self.use_tt and board_hash is not None:
                    self.transposition_table[board_hash] = self.root
                self.tt_misses += 1

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

    def _run_batched(self, board):
        """
        Stage 1 batched MCTS: Collect leaves and evaluate in batches.

        Simplified approach without virtual loss - collects leaves sequentially
        but evaluates them in batches for GPU efficiency.
        """
        # Initialize root (always create new root for Stage 1, no TT)
        probs, value, mask = self._policy_value(board)
        self.root = MCTSNode(prior=probs, player_to_move=board.player)
        self.root.is_expanded = True
        self.root.valid_mask = mask
        self.root.terminal = board.is_terminal()

        # Dirichlet noise at root
        if self.dir_alpha is not None and self.dir_frac and self.dir_frac > 0:
            valid_idx = np.where(mask > 0.0)[0]
            if len(valid_idx) > 1:
                noise = np.random.dirichlet([self.dir_alpha] * len(valid_idx))
                P = self.root.P.copy()
                P[valid_idx] = (1 - self.dir_frac) * P[valid_idx] + self.dir_frac * noise
                self.root.P = P

        # Collect leaves for batching
        for _ in range(self.simulations):
            board_copy = board.copy()
            self._select_leaf_for_batch(board_copy, self.root, [])

            # Flush batch when full
            if len(self.leaf_queue) >= self.batch_size:
                self._flush_leaf_batch()

        # Process any remaining leaves
        if self.leaf_queue:
            self._flush_leaf_batch()

        # Return visit counts as policy
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
            output = self.net(torch.tensor(planes, dtype=torch.float32, device=self.device))
            logits = output.policy_logits[0].cpu().numpy()
            value = float(output.value_win[0].cpu().item())
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
            output = self.net(torch.tensor(planes, dtype=torch.float32, device=self.device))
            logits = output.policy_logits[0].cpu().numpy()
            value = float(output.value_win[0].cpu().item())
        mask = board.valid_action_mask()
        probs = softmax_masked(logits, mask, temp=1.0)
        return probs, value, mask

    def _simulate(self, board, node, depth=0, max_depth=100):
        # Safety: prevent infinite recursion from TT cycles
        if depth >= max_depth:
            # Return draw value if max depth reached
            return 0.0

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

        # Expand or reuse from TT
        child = node.children.get(best_a)
        if child is None:
            # Check TT for this child position
            child_hash = zobrist_hash(board) if self.use_tt else None
            if self.use_tt and child_hash in self.transposition_table:
                child = self.transposition_table[child_hash]
                node.children[best_a] = child
                self.tt_hits += 1

                # Terminal check before recursion (prevent unnecessary recursion)
                if child.terminal:
                    winner, _ = board.result()
                    if winner == 0:
                        v = 0.0
                    else:
                        v = 1.0 if winner == child.player_to_move else -1.0
                else:
                    v = self._simulate(board, child, depth + 1, max_depth)
            else:
                # Create new node
                probs, value, mask = self._policy_value(board)
                child = MCTSNode(prior=probs, player_to_move=board.player)
                child.is_expanded = True
                child.valid_mask = mask
                child.terminal = board.is_terminal()
                node.children[best_a] = child
                if self.use_tt and child_hash is not None:
                    self.transposition_table[child_hash] = child
                self.tt_misses += 1
                # Back up value from the perspective of node.player_to_move
                v = value
        else:
            # Reusing existing child - check terminal before recursing
            if child.terminal:
                winner, _ = board.result()
                if winner == 0:
                    v = 0.0
                else:
                    v = 1.0 if winner == child.player_to_move else -1.0
            else:
                v = self._simulate(board, child, depth + 1, max_depth)

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

    def _select_leaf_for_batch(self, board, node, path):
        """
        Stage 1 batching: Select a leaf node and queue it for batch evaluation.

        Traverses tree using PUCT until reaching unexpanded node, then queues
        the (node, board, path) tuple for later batch evaluation.

        Args:
            board: Current board state (will be modified during traversal)
            node: Current MCTS node
            path: List of (node, action) pairs representing path from root
        """
        # Terminal node - immediate backup
        if node.terminal:
            winner, _ = board.result()
            if winner == 0:
                v = 0.0
            else:
                v = 1.0 if winner == node.player_to_move else -1.0
            self._backup_path(path, v)
            return

        # If node not expanded, queue for batch evaluation
        if not node.is_expanded:
            self.leaf_queue.append((node, board.copy(), path[:]))
            return

        # Select best action using PUCT
        best_a = None
        best_score = -1e9
        sqrt_sum = math.sqrt(max(1, node.N.sum()))
        for a in range(len(node.P)):
            if node.valid_mask[a] <= 0.0:
                continue
            Q = node.Q[a]
            U = self.cpuct * node.P[a] * (sqrt_sum / (1 + node.N[a]))
            score = Q + U
            if score > best_score:
                best_score = score
                best_a = a

        # Apply action
        current_player = board.player
        board.step_action_index(best_a)

        # Get or create child
        child = node.children.get(best_a)
        if child is None:
            # Create unexpanded child node
            child = MCTSNode(prior=np.ones(65) / 65, player_to_move=board.player)
            child.is_expanded = False
            child.terminal = board.is_terminal()
            child.valid_mask = board.valid_action_mask()
            node.children[best_a] = child

        # Add to path and continue traversal
        path.append((node, best_a, current_player))
        self._select_leaf_for_batch(board, child, path)

    def _flush_leaf_batch(self):
        """
        Stage 1 batching: Evaluate queued leaf nodes as a batch.

        Collects all queued (node, board, path) tuples, evaluates them
        in a single batched forward pass, then expands nodes and backs
        up values along their paths.
        """
        if not self.leaf_queue:
            return

        # Extract board states for batch evaluation
        states = [board.encode() for node, board, path in self.leaf_queue]

        # Batch evaluate all positions
        results = self.batch_evaluator.evaluate_batch(states)

        # Process each leaf: expand and backup
        for (node, board, path), (policy, value) in zip(self.leaf_queue, results):
            # Mask invalid actions
            mask = board.valid_action_mask()
            masked_policy = policy * mask
            masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

            # Expand leaf node
            node.P = masked_policy
            node.is_expanded = True
            node.valid_mask = mask

            # Backup value along path
            self._backup_path(path, value)

        # Clear queue
        self.leaf_queue.clear()

    def _backup_path(self, path, leaf_value):
        """
        Backup value along a path from root to leaf.

        Args:
            path: List of (node, action, player) tuples from root to leaf
            leaf_value: Value to backup (from leaf's perspective)
        """
        # Start with leaf value (negate immediately - we're backing up to parent/opponent)
        v = -leaf_value  # FIX: Negate before backup loop (parent is opponent of leaf)

        # Backup along path in reverse (leaf to root)
        for node, action, current_player in reversed(path):
            # Update statistics
            node.N[action] += 1
            node.W[action] += v
            node.Q[action] = node.W[action] / node.N[action]

            # Flip value for parent's perspective
            # (if player changed, value is from opponent's POV)
            v = -v
