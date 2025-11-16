"""
Batch Evaluator for MCTS Neural Network Inference

Provides batched inference capabilities for MCTS leaf node evaluation,
dramatically improving GPU/MPS utilization and throughput.

Stage 1: Simple batching without virtual loss or complex queuing
Stage 2+: Can be extended with virtual loss and parallel game coordination
"""

import torch
import numpy as np
from typing import List, Tuple


class SimpleBatchEvaluator:
    """
    Simple batch evaluator for MCTS neural network inference.

    Collects multiple board positions and evaluates them in a single
    batched forward pass, leveraging GPU/MPS parallelism for massive
    speedup over sequential evaluation.

    Stage 1 implementation: No virtual loss, no complex queuing.
    Simply batches positions and returns results synchronously.
    """

    def __init__(self, model, device, batch_size=32):
        """
        Initialize batch evaluator.

        Args:
            model: Neural network model (OthelloNet)
            device: torch.device ('cpu', 'cuda', or 'mps')
            batch_size: Maximum batch size for inference (default: 32)
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()  # Ensure model is in eval mode

        # Statistics for monitoring
        self.total_evaluations = 0
        self.total_batches = 0

    def evaluate_batch(self, board_states: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        Evaluate multiple board positions in a single batched forward pass.

        Args:
            board_states: List of encoded board states (numpy arrays, shape: (4, 8, 8))

        Returns:
            List of (policy, value) tuples:
                - policy: numpy array of action probabilities (shape: 65)
                - value: float value estimate from current player's perspective
        """
        if not board_states:
            return []

        batch_size = len(board_states)

        # Convert list of numpy arrays to batched tensor
        # Each state is (4, 8, 8), stack to (N, 4, 8, 8)
        batch_tensor = torch.stack([
            torch.from_numpy(state) for state in board_states
        ]).to(self.device, dtype=torch.float32)

        # Single batched forward pass (the magic!)
        with torch.no_grad():
            outputs = self.model(batch_tensor)

        # Unpack results for each position
        results = []
        for i in range(batch_size):
            # Extract policy logits and apply softmax
            policy_logits = outputs.policy_logits[i]
            policy = torch.softmax(policy_logits, dim=0).cpu().numpy()

            # Extract value estimate
            value = float(outputs.value_win[i].cpu().item())

            results.append((policy, value))

        # Update statistics
        self.total_evaluations += batch_size
        self.total_batches += 1

        return results

    def get_stats(self) -> dict:
        """
        Get evaluation statistics.

        Returns:
            dict with keys: total_evaluations, total_batches, avg_batch_size
        """
        avg_batch_size = self.total_evaluations / max(1, self.total_batches)
        return {
            'total_evaluations': self.total_evaluations,
            'total_batches': self.total_batches,
            'avg_batch_size': avg_batch_size
        }

    def reset_stats(self):
        """Reset evaluation statistics."""
        self.total_evaluations = 0
        self.total_batches = 0


class DirectEvaluator:
    """
    Direct (non-batched) evaluator for comparison and fallback.

    Evaluates positions one at a time, mimicking the original MCTS behavior.
    Useful for:
    - Testing and debugging
    - Fallback when batching is disabled
    - Baseline comparisons
    """

    def __init__(self, model, device):
        """
        Initialize direct evaluator.

        Args:
            model: Neural network model (OthelloNet)
            device: torch.device ('cpu', 'cuda', or 'mps')
        """
        self.model = model
        self.device = device
        self.model.eval()

    def evaluate_single(self, board_state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate a single board position.

        Args:
            board_state: Encoded board state (numpy array, shape: (4, 8, 8))

        Returns:
            Tuple of (policy, value):
                - policy: numpy array of action probabilities (shape: 65)
                - value: float value estimate from current player's perspective
        """
        # Convert to tensor and add batch dimension
        state_tensor = torch.from_numpy(board_state).unsqueeze(0).to(
            self.device, dtype=torch.float32
        )

        # Forward pass
        with torch.no_grad():
            outputs = self.model(state_tensor)

        # Extract results
        policy = torch.softmax(outputs.policy_logits[0], dim=0).cpu().numpy()
        value = float(outputs.value_win[0].cpu().item())

        return policy, value

    def evaluate_batch(self, board_states: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        Evaluate multiple positions sequentially (for interface compatibility).

        Args:
            board_states: List of encoded board states

        Returns:
            List of (policy, value) tuples
        """
        return [self.evaluate_single(state) for state in board_states]
