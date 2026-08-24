"""RawNet2: end-to-end anti-spoofing CNN on the raw waveform.

Sinc-filter front-end (learned band-pass filters) -> residual blocks with
filter-wise feature-map scaling (FMS) -> GRU -> classifier, following
Tak et al., "End-to-end anti-spoofing with RawNet2" (ICASSP 2021).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """Learnable band-pass sinc filters (SincNet front-end)."""

    def __init__(self, out_channels: int = 20, kernel_size: int = 1024, sr: int = 16000):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size, self.sr, self.out_channels = kernel_size, sr, out_channels
        # initialize cutoff frequencies on a mel-spaced grid
        low_hz, high_hz = 0.0, sr / 2 - 100.0
        mel = torch.linspace(self._to_mel(low_hz + 30), self._to_mel(high_hz), out_channels + 1)
        hz = self._to_hz(mel)
        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz[1:] - hz[:-1]).unsqueeze(1))
        n = (kernel_size - 1) / 2
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).unsqueeze(0) / sr)
        self.register_buffer("window_", torch.hamming_window(kernel_size)[: int(n)])

    @staticmethod
    def _to_mel(hz):
        return 2595 * math.log10(1 + hz / 700) if not torch.is_tensor(hz) else 2595 * torch.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, L)
        low = torch.abs(self.low_hz_) + 30.0
        high = torch.clamp(low + torch.abs(self.band_hz_), 50.0, self.sr / 2)
        f_low = torch.matmul(low, self.n_.to(x.dtype))
        f_high = torch.matmul(high, self.n_.to(x.dtype))
        band_left = ((torch.sin(f_high) - torch.sin(f_low)) / (self.n_ / 2)) * self.window_
        band_center = 2 * (high - low)
        filters = torch.cat([band_left, band_center, band_left.flip(1)], dim=1)
        filters = (filters / (2 * (high - low) + 1e-8)).unsqueeze(1)  # (C,1,K)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)


class ResBlockFMS(nn.Module):
    """Residual block with filter-wise feature-map scaling (sigmoid gating)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.shortcut = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.fms = nn.Linear(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.bn1(x), 0.3))
        h = self.conv2(F.leaky_relu(self.bn2(h), 0.3))
        h = h + self.shortcut(x)
        h = F.max_pool1d(h, 3)
        g = torch.sigmoid(self.fms(h.mean(dim=-1))).unsqueeze(-1)
        return h * g + g


class RawNet2(nn.Module):
    def __init__(self, num_classes: int = 2, sinc_channels: int = 20, gru_hidden: int = 256):
        super().__init__()
        self.sinc = SincConv(out_channels=sinc_channels)
        self.first_bn = nn.BatchNorm1d(sinc_channels)
        self.blocks = nn.Sequential(
            ResBlockFMS(sinc_channels, 128),
            ResBlockFMS(128, 128),
            ResBlockFMS(128, 256),
            ResBlockFMS(256, 256),
            ResBlockFMS(256, 256),
            ResBlockFMS(256, 256),
        )
        self.pre_gru_bn = nn.BatchNorm1d(256)
        self.gru = nn.GRU(256, gru_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(gru_hidden, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, L) raw waveform -> logits (B, num_classes)."""
        x = self.sinc(wav.unsqueeze(1))
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.blocks(x)
        x = F.leaky_relu(self.pre_gru_bn(x), 0.3)
        out, _ = self.gru(x.transpose(1, 2))
        return self.fc(out[:, -1])
