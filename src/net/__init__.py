from dataclasses import dataclass

import torch


@dataclass
class NetworkOutput:
    """Structured output returned by neural network forward pass."""

    policy_logits: torch.Tensor
    value_win: torch.Tensor
    value_score: torch.Tensor
    mobility: torch.Tensor
    stability_map: torch.Tensor
    corner: torch.Tensor
    parity: torch.Tensor
