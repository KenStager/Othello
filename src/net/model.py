from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NetworkOutput


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return F.relu(out, inplace=True)


class LineAwareBlock(nn.Module):
    """Captures long-range dependencies along ranks/files via axial convolutions."""

    def __init__(self, ch: int):
        super().__init__()
        self.axial_h = nn.Conv2d(ch, ch, kernel_size=(1, 3), padding=(0, 1), groups=ch, bias=False)
        self.axial_w = nn.Conv2d(ch, ch, kernel_size=(3, 1), padding=(1, 0), groups=ch, bias=False)
        self.mix = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.axial_h(x)
        v = self.axial_w(x)
        out = self.mix(h + v)
        out = self.bn(out)
        return F.relu(out + x, inplace=True)


class OthelloNet(nn.Module):
    def __init__(self, in_channels: int = 4, channels: int = 64, residual_blocks: int = 8, action_size: int = 65):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        trunk_layers: List[nn.Module] = []
        for idx in range(residual_blocks):
            trunk_layers.append(ResidualBlock(channels))
            if (idx + 1) % 2 == 0:
                trunk_layers.append(LineAwareBlock(channels))
        self.trunk = nn.Sequential(*trunk_layers)

        # Shared projection for heads
        self.head_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, action_size),
        )

        # Dual value head (win probability + score differential)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 2),
        )

        # Auxiliary heads
        reduced_channels = max(8, channels // 2)
        self.mobility_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 2),
        )

        self.stability_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.Sigmoid(),
        )

        self.corner_head = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(reduced_channels * 4, 4),
        )

        self.parity_head = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(reduced_channels * 4, 5),
        )

    def forward(self, x: torch.Tensor) -> NetworkOutput:
        # x: (B, 4, 8, 8)
        h = self.stem(x)
        h = self.trunk(h)
        h_shared = self.head_conv(h)

        policy_logits = self.policy(h_shared)  # (B, 65)

        value_raw = self.value_head(h_shared)
        value_win = torch.tanh(value_raw[:, 0])
        value_score = torch.tanh(value_raw[:, 1])

        mobility_logits = self.mobility_head(h_shared)
        mobility = torch.sigmoid(mobility_logits)

        stability_map = self.stability_head(h_shared)

        corner_logits = self.corner_head(h_shared)
        corner = torch.sigmoid(corner_logits)

        parity_logits = self.parity_head(h_shared)
        parity = torch.sigmoid(parity_logits)

        return NetworkOutput(
            policy_logits=policy_logits,
            value_win=value_win,
            value_score=value_score,
            mobility=mobility,
            stability_map=stability_map,
            corner=corner,
            parity=parity,
        )
