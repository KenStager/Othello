"""
Training Stability Monitor

Monitors training for signs of instability and issues:
- Gradient norm explosions/vanishing
- Loss spikes and divergence
- Value distribution collapse
- Policy diversity degradation

Usage:
    from src.train.stability_monitor import StabilityMonitor

    monitor = StabilityMonitor()

    # During training
    monitor.record_step(loss, gradients, outputs)

    # Check for issues
    if monitor.has_critical_issues():
        print("⚠️ Critical training issues detected!")
        monitor.print_alerts()
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional, Tuple


class StabilityMonitor:
    """
    Monitors training stability and detects issues.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize stability monitor.

        Args:
            window_size: Number of recent steps to track
        """
        self.window_size = window_size

        # Gradient tracking
        self.grad_norms = deque(maxlen=window_size)
        self.grad_norms_per_layer = {}  # layer_name -> deque

        # Loss tracking
        self.losses = deque(maxlen=window_size)
        self.loss_spikes = []  # List of (step, loss_value) tuples

        # Value distribution tracking
        self.value_means = deque(maxlen=window_size)
        self.value_stds = deque(maxlen=window_size)

        # Policy diversity tracking
        self.policy_entropies = deque(maxlen=window_size)
        self.policy_max_probs = deque(maxlen=window_size)

        # Issue tracking
        self.warnings = []
        self.critical_issues = []
        self.current_step = 0

        # Thresholds
        self.grad_explosion_threshold = 100.0
        self.grad_vanishing_threshold = 1e-6
        self.loss_spike_multiplier = 3.0
        self.value_collapse_threshold = 0.05
        self.policy_collapse_threshold = 0.5

    def record_step(self, loss: float, gradients: Optional[Dict[str, torch.Tensor]] = None,
                   outputs: Optional['NetworkOutput'] = None):
        """
        Record metrics for current training step.

        Args:
            loss: Training loss value
            gradients: Optional dict of parameter gradients
            outputs: Optional network outputs for distribution analysis
        """
        self.current_step += 1
        self.losses.append(loss)

        # Gradient norms
        if gradients is not None:
            total_norm = 0.0
            for name, grad in gradients.items():
                if grad is not None:
                    param_norm = grad.norm().item()
                    total_norm += param_norm ** 2

                    # Track per-layer norms
                    if name not in self.grad_norms_per_layer:
                        self.grad_norms_per_layer[name] = deque(maxlen=self.window_size)
                    self.grad_norms_per_layer[name].append(param_norm)

            total_norm = total_norm ** 0.5
            self.grad_norms.append(total_norm)

            # Check for gradient issues
            self._check_gradient_health(total_norm)

        # Loss spike detection
        self._check_loss_spike(loss)

        # Value/policy distribution analysis
        if outputs is not None:
            self._analyze_distributions(outputs)

    def _check_gradient_health(self, grad_norm: float):
        """Check for gradient explosion or vanishing."""
        if grad_norm > self.grad_explosion_threshold:
            self.critical_issues.append({
                'step': self.current_step,
                'type': 'gradient_explosion',
                'value': grad_norm,
                'message': f'Gradient explosion detected (norm={grad_norm:.2f})'
            })
        elif grad_norm < self.grad_vanishing_threshold and grad_norm > 0:
            self.warnings.append({
                'step': self.current_step,
                'type': 'gradient_vanishing',
                'value': grad_norm,
                'message': f'Gradient vanishing detected (norm={grad_norm:.2e})'
            })

    def _check_loss_spike(self, loss: float):
        """Detect sudden loss spikes."""
        if len(self.losses) >= 10:
            recent_losses = list(self.losses)[-10:]
            median_loss = np.median(recent_losses)
            if loss > median_loss * self.loss_spike_multiplier:
                self.loss_spikes.append((self.current_step, loss))
                self.warnings.append({
                    'step': self.current_step,
                    'type': 'loss_spike',
                    'value': loss,
                    'baseline': median_loss,
                    'message': f'Loss spike detected ({loss:.4f} vs median {median_loss:.4f})'
                })

    def _analyze_distributions(self, outputs: 'NetworkOutput'):
        """Analyze value and policy distributions."""
        # Value distribution
        values = outputs.value_win.detach().cpu().numpy()
        self.value_means.append(float(np.mean(values)))
        self.value_stds.append(float(np.std(values)))

        # Check for value collapse
        if len(self.value_stds) >= 10:
            recent_std = np.mean(list(self.value_stds)[-10:])
            if recent_std < self.value_collapse_threshold:
                self.warnings.append({
                    'step': self.current_step,
                    'type': 'value_collapse',
                    'value': recent_std,
                    'message': f'Value collapse detected (std={recent_std:.4f})'
                })

        # Policy diversity
        policy_probs = torch.softmax(outputs.policy_logits, dim=1).detach().cpu().numpy()
        max_probs = np.max(policy_probs, axis=1)
        entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-10), axis=1)

        self.policy_max_probs.append(float(np.mean(max_probs)))
        self.policy_entropies.append(float(np.mean(entropy)))

        # Check for policy collapse
        if len(self.policy_entropies) >= 10:
            recent_entropy = np.mean(list(self.policy_entropies)[-10:])
            if recent_entropy < self.policy_collapse_threshold:
                self.warnings.append({
                    'step': self.current_step,
                    'type': 'policy_collapse',
                    'value': recent_entropy,
                    'message': f'Policy collapse detected (entropy={recent_entropy:.4f})'
                })

    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.grad_norms:
            return {}

        norms = list(self.grad_norms)
        return {
            'mean': float(np.mean(norms)),
            'std': float(np.std(norms)),
            'min': float(np.min(norms)),
            'max': float(np.max(norms)),
            'current': float(norms[-1])
        }

    def get_loss_stats(self) -> Dict[str, float]:
        """Get loss statistics."""
        if not self.losses:
            return {}

        losses = list(self.losses)
        return {
            'mean': float(np.mean(losses)),
            'std': float(np.std(losses)),
            'min': float(np.min(losses)),
            'max': float(np.max(losses)),
            'current': float(losses[-1]),
            'num_spikes': len([s for s in self.loss_spikes if s[0] > self.current_step - self.window_size])
        }

    def get_value_stats(self) -> Dict[str, float]:
        """Get value distribution statistics."""
        if not self.value_means:
            return {}

        return {
            'mean': float(np.mean(list(self.value_means))),
            'std_mean': float(np.mean(list(self.value_stds))),
            'std_current': float(list(self.value_stds)[-1]) if self.value_stds else 0.0
        }

    def get_policy_stats(self) -> Dict[str, float]:
        """Get policy diversity statistics."""
        if not self.policy_entropies:
            return {}

        return {
            'entropy_mean': float(np.mean(list(self.policy_entropies))),
            'entropy_current': float(list(self.policy_entropies)[-1]),
            'max_prob_mean': float(np.mean(list(self.policy_max_probs))),
            'max_prob_current': float(list(self.policy_max_probs)[-1])
        }

    def has_critical_issues(self) -> bool:
        """Check if any critical issues have been detected."""
        return len(self.critical_issues) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings have been issued."""
        recent_warnings = [w for w in self.warnings if w['step'] > self.current_step - 50]
        return len(recent_warnings) > 0

    def print_alerts(self, recent_only: bool = True):
        """
        Print all active alerts.

        Args:
            recent_only: If True, only show alerts from last 50 steps
        """
        cutoff_step = self.current_step - 50 if recent_only else 0

        # Critical issues
        recent_critical = [i for i in self.critical_issues if i['step'] > cutoff_step]
        if recent_critical:
            print("🚨 CRITICAL ISSUES:")
            for issue in recent_critical[-5:]:  # Show last 5
                print(f"  Step {issue['step']:6}: {issue['message']}")

        # Warnings
        recent_warnings = [w for w in self.warnings if w['step'] > cutoff_step]
        if recent_warnings:
            print("⚠️  WARNINGS:")
            for warning in recent_warnings[-5:]:  # Show last 5
                print(f"  Step {warning['step']:6}: {warning['message']}")

    def print_summary(self):
        """Print comprehensive stability summary."""
        print("=== Stability Monitor Summary ===")
        print()

        # Gradient statistics
        grad_stats = self.get_gradient_stats()
        if grad_stats:
            print("Gradient Norms:")
            print(f"  Mean:    {grad_stats['mean']:.4f}")
            print(f"  Std:     {grad_stats['std']:.4f}")
            print(f"  Current: {grad_stats['current']:.4f}")
            print(f"  Range:   [{grad_stats['min']:.4f}, {grad_stats['max']:.4f}]")
            print()

        # Loss statistics
        loss_stats = self.get_loss_stats()
        if loss_stats:
            print("Loss:")
            print(f"  Mean:    {loss_stats['mean']:.4f}")
            print(f"  Std:     {loss_stats['std']:.4f}")
            print(f"  Current: {loss_stats['current']:.4f}")
            print(f"  Spikes:  {loss_stats['num_spikes']}")
            print()

        # Value statistics
        value_stats = self.get_value_stats()
        if value_stats:
            print("Value Distribution:")
            print(f"  Mean:        {value_stats['mean']:.4f}")
            print(f"  Std (avg):   {value_stats['std_mean']:.4f}")
            print(f"  Std (curr):  {value_stats['std_current']:.4f}")
            print()

        # Policy statistics
        policy_stats = self.get_policy_stats()
        if policy_stats:
            print("Policy Diversity:")
            print(f"  Entropy (avg):   {policy_stats['entropy_mean']:.3f}")
            print(f"  Entropy (curr):  {policy_stats['entropy_current']:.3f}")
            print(f"  Max prob (avg):  {policy_stats['max_prob_mean']:.3f}")
            print(f"  Max prob (curr): {policy_stats['max_prob_current']:.3f}")
            print()

        # Alerts
        if self.has_critical_issues() or self.has_warnings():
            print()
            self.print_alerts(recent_only=True)
            print()

        print("=" * 33)
        print()

    def get_recommendations(self) -> List[str]:
        """
        Get actionable recommendations based on detected issues.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Gradient issues
        grad_stats = self.get_gradient_stats()
        if grad_stats:
            if grad_stats['max'] > self.grad_explosion_threshold:
                recommendations.append("Reduce learning rate (gradient explosion)")
                recommendations.append("Increase gradient clipping threshold")
            if grad_stats['mean'] < self.grad_vanishing_threshold:
                recommendations.append("Increase learning rate (gradients vanishing)")
                recommendations.append("Check for dead ReLUs or saturated activations")

        # Loss spikes
        loss_stats = self.get_loss_stats()
        if loss_stats and loss_stats['num_spikes'] > 3:
            recommendations.append("Reduce learning rate (frequent loss spikes)")
            recommendations.append("Consider reducing batch size for stability")

        # Value collapse
        value_stats = self.get_value_stats()
        if value_stats and value_stats['std_current'] < self.value_collapse_threshold:
            recommendations.append("Values collapsing - may need more diverse data")
            recommendations.append("Check replay buffer for stale data")

        # Policy collapse
        policy_stats = self.get_policy_stats()
        if policy_stats and policy_stats['entropy_current'] < self.policy_collapse_threshold:
            recommendations.append("Policy becoming deterministic - increase exploration")
            recommendations.append("Consider increasing Dirichlet noise during self-play")

        return recommendations

    def reset(self):
        """Reset all tracking statistics."""
        self.grad_norms.clear()
        self.grad_norms_per_layer.clear()
        self.losses.clear()
        self.loss_spikes.clear()
        self.value_means.clear()
        self.value_stds.clear()
        self.policy_entropies.clear()
        self.policy_max_probs.clear()
        self.warnings.clear()
        self.critical_issues.clear()
        self.current_step = 0
