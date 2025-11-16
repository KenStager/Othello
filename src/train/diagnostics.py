"""
Enhanced Training Diagnostics

Provides detailed analysis of training quality including:
- Loss breakdown by game phase (opening/midgame/endgame)
- Q-value calibration (predicted vs actual outcomes)
- Policy entropy tracking (exploration indicator)
- Auxiliary loss correlation analysis
- Phase-specific learning curves

Usage:
    from src.train.diagnostics import Diagnostics

    diagnostics = Diagnostics()

    # During training loop
    diagnostics.record_batch(batch_data, outputs, losses)

    # Print periodic summary
    if step % 50 == 0:
        diagnostics.print_summary()
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


class Diagnostics:
    """
    Tracks and analyzes training metrics for enhanced visibility into model learning.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize diagnostics tracker.

        Args:
            window_size: Number of recent samples to keep for rolling statistics
        """
        self.window_size = window_size

        # Phase-specific loss tracking
        self.phase_losses = {
            'opening': defaultdict(lambda: deque(maxlen=window_size)),
            'midgame': defaultdict(lambda: deque(maxlen=window_size)),
            'endgame': defaultdict(lambda: deque(maxlen=window_size)),
        }

        # Q-value calibration tracking
        self.q_predictions = deque(maxlen=window_size)
        self.q_targets = deque(maxlen=window_size)
        self.q_errors = deque(maxlen=window_size)

        # Policy entropy tracking
        self.policy_entropy = {
            'opening': deque(maxlen=window_size),
            'midgame': deque(maxlen=window_size),
            'endgame': deque(maxlen=window_size),
        }

        # Auxiliary loss correlation
        self.aux_correlations = {
            'mobility': deque(maxlen=window_size),
            'stability': deque(maxlen=window_size),
            'corner': deque(maxlen=window_size),
            'parity': deque(maxlen=window_size),
        }

        # Overall statistics
        self.total_samples = 0
        self.phase_counts = {'opening': 0, 'midgame': 0, 'endgame': 0}

    def record_batch(self, batch: Dict, outputs: 'NetworkOutput', losses: Dict):
        """
        Record metrics from a training batch.

        Args:
            batch: Training batch containing states, targets, phases, etc.
            outputs: NetworkOutput from model forward pass
            losses: Dictionary of computed losses
        """
        import torch

        batch_size = len(batch['phase'])
        self.total_samples += batch_size

        # Convert tensors to numpy for analysis
        value_pred = outputs.value_win.detach().cpu().numpy()
        value_true = batch['value_win'].cpu().numpy() if isinstance(batch['value_win'], torch.Tensor) else batch['value_win']
        policy_logits = outputs.policy_logits.detach().cpu().numpy()

        # Process each sample in batch
        for i in range(batch_size):
            phase = batch['phase'][i]
            self.phase_counts[phase] += 1

            # Record phase-specific losses
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    self.phase_losses[phase][loss_name].append(float(loss_value.item()))

            # Q-value calibration
            self.q_predictions.append(value_pred[i])
            self.q_targets.append(value_true[i])
            self.q_errors.append(abs(value_pred[i] - value_true[i]))

            # Policy entropy
            policy = self._softmax(policy_logits[i])
            entropy = -np.sum(policy * np.log(policy + 1e-10))
            self.policy_entropy[phase].append(entropy)

            # Auxiliary loss correlations (if available)
            if 'loss_mobility' in losses:
                self.aux_correlations['mobility'].append(float(losses['loss_mobility'].item()))
            if 'loss_stability' in losses:
                self.aux_correlations['stability'].append(float(losses['loss_stability'].item()))
            if 'loss_corner' in losses:
                self.aux_correlations['corner'].append(float(losses['loss_corner'].item()))
            if 'loss_parity' in losses:
                self.aux_correlations['parity'].append(float(losses['loss_parity'].item()))

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax of logits."""
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def get_phase_loss_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for losses by phase.

        Returns:
            Dict mapping phase -> loss_name -> mean_value
        """
        summary = {}
        for phase in ['opening', 'midgame', 'endgame']:
            summary[phase] = {}
            for loss_name, values in self.phase_losses[phase].items():
                if values:
                    summary[phase][loss_name] = float(np.mean(values))
        return summary

    def get_calibration_metrics(self) -> Dict[str, float]:
        """
        Analyze Q-value calibration.

        Returns:
            Dict with calibration metrics:
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - correlation: Pearson correlation coefficient
            - bias: Mean error (positive = overestimation)
        """
        if not self.q_predictions:
            return {}

        predictions = np.array(self.q_predictions)
        targets = np.array(self.q_targets)

        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
        bias = np.mean(predictions - targets)

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'bias': float(bias)
        }

    def get_entropy_summary(self) -> Dict[str, float]:
        """
        Get policy entropy by phase.

        Returns:
            Dict mapping phase -> mean_entropy
        """
        summary = {}
        for phase in ['opening', 'midgame', 'endgame']:
            if self.policy_entropy[phase]:
                summary[phase] = float(np.mean(self.policy_entropy[phase]))
        return summary

    def get_aux_loss_summary(self) -> Dict[str, float]:
        """
        Get auxiliary loss summary.

        Returns:
            Dict mapping aux_head -> mean_loss
        """
        summary = {}
        for aux_name, values in self.aux_correlations.items():
            if values:
                summary[aux_name] = float(np.mean(values))
        return summary

    def print_summary(self, step: Optional[int] = None):
        """
        Print comprehensive diagnostics summary.

        Args:
            step: Optional training step number for logging
        """
        header = f"=== Diagnostics Summary (Step {step}) ===" if step else "=== Diagnostics Summary ==="
        print(header)
        print()

        # Phase distribution
        total = sum(self.phase_counts.values())
        if total > 0:
            print(f"Phase Distribution (n={total}):")
            for phase in ['opening', 'midgame', 'endgame']:
                count = self.phase_counts[phase]
                pct = count / total * 100
                print(f"  {phase:>8}: {count:6} ({pct:5.1f}%)")
            print()

        # Loss by phase
        phase_losses = self.get_phase_loss_summary()
        if phase_losses:
            print("Loss by Phase:")
            # Collect all loss names
            all_loss_names = set()
            for phase_dict in phase_losses.values():
                all_loss_names.update(phase_dict.keys())

            for loss_name in sorted(all_loss_names):
                print(f"  {loss_name}:")
                for phase in ['opening', 'midgame', 'endgame']:
                    value = phase_losses.get(phase, {}).get(loss_name, None)
                    if value is not None:
                        print(f"    {phase:>8}: {value:.6f}")
            print()

        # Q-value calibration
        calibration = self.get_calibration_metrics()
        if calibration:
            print("Q-Value Calibration:")
            print(f"  MAE:         {calibration['mae']:.4f}")
            print(f"  RMSE:        {calibration['rmse']:.4f}")
            print(f"  Correlation: {calibration['correlation']:.4f}")
            print(f"  Bias:        {calibration['bias']:+.4f} {'(overestimate)' if calibration['bias'] > 0 else '(underestimate)'}")
            print()

        # Policy entropy
        entropy_summary = self.get_entropy_summary()
        if entropy_summary:
            print("Policy Entropy by Phase:")
            for phase in ['opening', 'midgame', 'endgame']:
                if phase in entropy_summary:
                    print(f"  {phase:>8}: {entropy_summary[phase]:.3f}")
            print()

        # Auxiliary losses
        aux_summary = self.get_aux_loss_summary()
        if aux_summary:
            print("Auxiliary Losses:")
            for aux_name, value in sorted(aux_summary.items()):
                print(f"  {aux_name:>10}: {value:.6f}")
            print()

        print("=" * len(header))
        print()

    def get_training_health_score(self) -> Tuple[float, List[str]]:
        """
        Compute overall training health score and identify issues.

        Returns:
            Tuple of (score, issues) where:
            - score: 0.0-1.0 health score (1.0 = perfect)
            - issues: List of detected issues
        """
        score = 1.0
        issues = []

        # Check Q-value calibration
        calibration = self.get_calibration_metrics()
        if calibration:
            if calibration['mae'] > 0.3:
                score -= 0.2
                issues.append(f"High Q-value MAE: {calibration['mae']:.3f}")
            if abs(calibration['correlation']) < 0.5:
                score -= 0.2
                issues.append(f"Low Q-value correlation: {calibration['correlation']:.3f}")
            if abs(calibration['bias']) > 0.2:
                score -= 0.1
                issues.append(f"Q-value bias: {calibration['bias']:+.3f}")

        # Check policy entropy
        entropy_summary = self.get_entropy_summary()
        if entropy_summary:
            for phase, entropy in entropy_summary.items():
                if entropy < 1.0:
                    score -= 0.1
                    issues.append(f"Low {phase} entropy: {entropy:.3f} (may be overfit)")
                if entropy > 4.0:
                    score -= 0.1
                    issues.append(f"High {phase} entropy: {entropy:.3f} (not learning)")

        # Check phase balance
        total = sum(self.phase_counts.values())
        if total > 0:
            for phase in ['opening', 'midgame', 'endgame']:
                ratio = self.phase_counts[phase] / total
                if ratio < 0.1:
                    score -= 0.1
                    issues.append(f"Undersampled {phase}: {ratio*100:.1f}%")

        return max(0.0, min(1.0, score)), issues

    def reset(self):
        """Reset all statistics."""
        self.phase_losses = {
            'opening': defaultdict(lambda: deque(maxlen=self.window_size)),
            'midgame': defaultdict(lambda: deque(maxlen=self.window_size)),
            'endgame': defaultdict(lambda: deque(maxlen=self.window_size)),
        }
        self.q_predictions.clear()
        self.q_targets.clear()
        self.q_errors.clear()
        for phase in self.policy_entropy:
            self.policy_entropy[phase].clear()
        for aux in self.aux_correlations:
            self.aux_correlations[aux].clear()
        self.total_samples = 0
        self.phase_counts = {'opening': 0, 'midgame': 0, 'endgame': 0}
