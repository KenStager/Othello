"""
Batch Evaluator for MCTS Neural Network Inference

Provides batched inference capabilities for MCTS leaf node evaluation,
dramatically improving GPU/MPS utilization and throughput.

Stage 1: Simple batching without virtual loss or complex queuing
Stage 2+: Can be extended with virtual loss and parallel game coordination
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from ..net.inference_optimizer import create_inference_optimizer


class SimpleBatchEvaluator:
    """
    Simple batch evaluator for MCTS neural network inference.

    Collects multiple board positions and evaluates them in a single
    batched forward pass, leveraging GPU/MPS parallelism for massive
    speedup over sequential evaluation.

    Stage 1 implementation: No virtual loss, no complex queuing.
    Simply batches positions and returns results synchronously.
    """

    def __init__(self, model, device, batch_size=32, optimization_config: Optional[Dict[str, Any]] = None):
        """
        Initialize batch evaluator with optional inference optimization.

        Args:
            model: Neural network model (OthelloNet)
            device: torch.device ('cpu', 'cuda', or 'mps')
            batch_size: Maximum batch size for inference (default: 32)
            optimization_config: Optional dict with optimization settings:
                - enabled: bool (default: True if config provided)
                - precision: "fp16" or "fp32" (default: "fp16")
                - use_channels_last: bool (default: True)
                - use_compilation: bool (default: True)
                - tensorrt_workspace_gb: int (default: 1)
        """
        self.device = device
        self.batch_size = batch_size

        # Apply platform-specific optimizations if enabled
        if optimization_config and optimization_config.get('enabled', True):
            optimizer = create_inference_optimizer(optimization_config)
            # Create example input for compilation (batch_size, 4, 8, 8)
            example_input = torch.randn(batch_size, 4, 8, 8).to(device)
            self.model = optimizer.optimize(model, device, example_input)
        else:
            # No optimization: use model as-is
            self.model = model.to(device)
            self.model.eval()

        # Detect actual model dtype (optimizer may have changed precision)
        first_param = next(self.model.parameters())
        self.use_fp16 = (first_param.dtype == torch.float16)

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
        dtype = torch.float16 if self.use_fp16 else torch.float32
        batch_tensor = torch.stack([
            torch.from_numpy(state) for state in board_states
        ]).to(self.device, dtype=dtype)

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

    def __init__(self, model, device, optimization_config: Optional[Dict[str, Any]] = None):
        """
        Initialize direct evaluator with optional inference optimization.

        Args:
            model: Neural network model (OthelloNet)
            device: torch.device ('cpu', 'cuda', or 'mps')
            optimization_config: Optional dict with optimization settings
                (same format as SimpleBatchEvaluator)
        """
        self.device = device

        # Apply platform-specific optimizations if enabled
        if optimization_config and optimization_config.get('enabled', True):
            optimizer = create_inference_optimizer(optimization_config)
            # Create example input for compilation (1, 4, 8, 8)
            example_input = torch.randn(1, 4, 8, 8).to(device)
            self.model = optimizer.optimize(model, device, example_input)
        else:
            # No optimization: use model as-is
            self.model = model.to(device)
            self.model.eval()

        # Detect actual model dtype (optimizer may have changed precision)
        first_param = next(self.model.parameters())
        self.use_fp16 = (first_param.dtype == torch.float16)

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
        dtype = torch.float16 if self.use_fp16 else torch.float32
        state_tensor = torch.from_numpy(board_state).unsqueeze(0).to(
            self.device, dtype=dtype
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


class QueuedBatchEvaluator:
    """
    Queue-based batch evaluator that submits requests to InferenceServer.

    This evaluator is used by worker threads in parallel self-play. Instead of
    maintaining its own model instance (which causes CUDA threading issues),
    it submits inference requests to a centralized InferenceServer running
    in a background thread.

    Benefits:
    - Avoids CUDA multi-threading issues (single GPU context)
    - Enables true parallelism (CPU workers, GPU-only server)
    - Automatic cross-game batching for maximum GPU efficiency
    - 3.49× speedup demonstrated in Phase 1 validation

    Usage:
        # Main thread creates inference server
        from src.mcts.inference_server import InferenceServer, InferenceClient
        server = InferenceServer(model, device, max_batch_size=64)
        server.start()

        # Each worker thread creates its own evaluator
        client = InferenceClient(server)
        evaluator = QueuedBatchEvaluator(client)

        # MCTS uses evaluator as normal
        results = evaluator.evaluate_batch(board_states)
    """

    def __init__(self, inference_client):
        """
        Initialize queued evaluator.

        Args:
            inference_client: InferenceClient instance wrapping InferenceServer
        """
        self.inference_client = inference_client

    def evaluate_batch(self, board_states: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        Evaluate multiple board positions by submitting to inference server.

        Args:
            board_states: List of encoded board states (numpy arrays, shape: (4, 8, 8))

        Returns:
            List of (policy, value) tuples:
                - policy: numpy array of action probabilities (shape: 65) - PROBABILITIES!
                - value: float value estimate from current player's perspective

        Note: Returns probabilities (softmax applied) to match SimpleBatchEvaluator interface.
        """
        if not board_states:
            return []

        results = []
        for state in board_states:
            # Submit request to inference server (blocks until result ready)
            response = self.inference_client.evaluate(state)

            # Apply softmax to convert logits to probabilities
            # (matching SimpleBatchEvaluator behavior)
            policy_logits = torch.from_numpy(response.policy_logits)
            policy_probs = torch.softmax(policy_logits, dim=0).numpy()

            results.append((policy_probs, response.value_win))

        return results

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
        response = self.inference_client.evaluate(board_state)

        # Apply softmax to convert logits to probabilities
        policy_logits = torch.from_numpy(response.policy_logits)
        policy_probs = torch.softmax(policy_logits, dim=0).numpy()

        return policy_probs, response.value_win
