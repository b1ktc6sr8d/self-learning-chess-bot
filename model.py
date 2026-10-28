"""
Dual-head Convolutional Neural Network for chess.

Architecture
────────────
    ┌─────────────┐
    │  Input Conv  │  18 → F filters, 3×3, BN, ReLU
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Res Block   │  ×N  (conv-BN-ReLU-conv-BN + skip)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │             │
  Policy        Value
  Head          Head
  (4096 logits) (scalar ∈ [-1,+1])

* **No hardcoded piece values.**
  All evaluation is learned purely through self-play.

* **Policy Head** outputs *raw logits* over the 4 096 action space.
  Legal-move masking and softmax are applied externally.

* **Value Head** outputs a single scalar through tanh, representing the
  estimated win probability from the current player's perspective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import NUM_PLANES, ACTION_SIZE


# ── Building blocks ──────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        return F.relu(out + skip)


# ── Main network ─────────────────────────────────────────────────────────

class ChessNet(nn.Module):
    """
    Lightweight dual-head CNN for chess.

    Parameters
    ----------
    num_planes : int
        Number of input feature planes (default 18).
    num_filters : int
        Width of the residual tower (default 128).
    num_res_blocks : int
        Depth of the residual tower (default 4).
    action_size : int
        Dimensionality of the policy output (default 4 096).
    """

    def __init__(
        self,
        num_planes:    int = NUM_PLANES,
        num_filters:   int = 128,
        num_res_blocks: int = 4,
        action_size:   int = ACTION_SIZE,
    ) -> None:
        super().__init__()
        self.action_size = action_size

        # ── Shared Trunk ──
        self.input_conv = nn.Conv2d(num_planes, num_filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(num_filters)
        self.tower       = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # ── Policy Head ──
        #    Conv 1×1 → 2 filters, flatten → FC → 4 096 logits
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, action_size)

        # ── Value Head ──
        #    Conv 1×1 → 1 filter, flatten → FC 64→128 → FC 128→1 (tanh)
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(1 * 8 * 8, 128)
        self.value_fc2  = nn.Linear(128, 1)

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 18, 8, 8)

        Returns
        -------
        policy_logits : (B, 4096)  — raw, unmasked, un-softmaxed
        value         : (B, 1)     — tanh-bounded scalar
        """
        # Shared trunk
        s = F.relu(self.input_bn(self.input_conv(x)))
        s = self.tower(s)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(p.size(0), -1)       # (B, 128)
        p = self.policy_fc(p)           # (B, 4096)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)       # (B, 64)
        v = F.relu(self.value_fc1(v))   # (B, 128)
        v = torch.tanh(self.value_fc2(v))  # (B, 1)

        return p, v
