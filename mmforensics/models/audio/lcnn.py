"""Light CNN (LCNN) with Max-Feature-Map activations over LFCC features —
the strongest classic ASVspoof anti-spoofing baseline architecture."""
from __future__ import annotations

import torch
import torch.nn as nn


class MFM(nn.Module):
    """Max-Feature-Map: split channels in half and take elementwise max —
    a learned competitive activation that suppresses noisy filters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return torch.max(a, b)


def _conv_mfm(cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout * 2, k, s, p), MFM())


class LCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            _conv_mfm(in_ch, 32, k=5, p=2),
            nn.MaxPool2d(2),
            _conv_mfm(32, 32, k=1, p=0), nn.BatchNorm2d(32),
            _conv_mfm(32, 48),
            nn.MaxPool2d(2), nn.BatchNorm2d(48),
            _conv_mfm(48, 48, k=1, p=0), nn.BatchNorm2d(48),
            _conv_mfm(48, 64),
            nn.MaxPool2d(2),
            _conv_mfm(64, 64, k=1, p=0), nn.BatchNorm2d(64),
            _conv_mfm(64, 32), nn.BatchNorm2d(32),
            _conv_mfm(32, 32, k=1, p=0), nn.BatchNorm2d(32),
            _conv_mfm(32, 32),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(32, 64), MFM1d(), nn.Dropout(0.5), nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_lfcc, T) -> logits (B, num_classes)."""
        f = self.pool(self.features(x)).flatten(1)
        return self.classifier(f)


class MFM1d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.max(a, b)
