"""Spatio-temporal deepfake detector.

Spatial stream: per-frame CNN (Xception — the FaceForensics++ baseline — or
EfficientNet) over face crops. Temporal stream: a lightweight Transformer
encoder (or Bi-LSTM) over the frame-feature sequence to catch flicker,
blending-boundary and identity-consistency artifacts. Frame scores are
attention-pooled into a video-level verdict.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CLASS_NAMES = ["real", "deepfake"]


class TemporalTransformer(nn.Module):
    def __init__(self, dim: int, depth: int = 2, heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                           dim_feedforward=dim * 2,
                                           dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        t = x.shape[1]
        return self.encoder(x + self.pos[:, :t])


class TemporalBiLSTM(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.proj = nn.Linear(hidden * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.proj(out)


class SpatioTemporalDeepfakeNet(nn.Module):
    def __init__(self, backbone: str = "xception", pretrained: bool = True,
                 temporal: str = "transformer", proj_dim: int = 512):
        super().__init__()
        import timm

        try:
            self.spatial = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        except Exception as e:  # weight download unavailable (offline/blocked host)
            if not pretrained:
                raise
            import warnings

            warnings.warn(f"Could not fetch pretrained weights for {backbone} ({e}); "
                          "falling back to random init.")
            self.spatial = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.spatial.num_features
        self.proj = nn.Linear(feat_dim, proj_dim)
        if temporal == "transformer":
            self.temporal = TemporalTransformer(proj_dim)
        elif temporal == "bilstm":
            self.temporal = TemporalBiLSTM(proj_dim)
        else:
            raise ValueError(f"unknown temporal model: {temporal}")
        # attention pooling over frames
        self.attn = nn.Sequential(nn.Linear(proj_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.frame_head = nn.Linear(proj_dim, 1)   # per-frame fake logit
        self.video_head = nn.Linear(proj_dim, 1)   # pooled video-level logit

    def forward(self, frames: torch.Tensor):
        """frames: (B, T, 3, H, W) face crops.

        Returns (video_logit (B,), frame_logits (B, T), attn_weights (B, T)).
        """
        b, t = frames.shape[:2]
        x = frames.flatten(0, 1)                       # (B*T, 3, H, W)
        feat = self.proj(self.spatial(x)).view(b, t, -1)
        feat = self.temporal(feat)                     # (B, T, D)
        frame_logits = self.frame_head(feat).squeeze(-1)
        w = torch.softmax(self.attn(feat).squeeze(-1), dim=1)  # (B, T)
        pooled = (feat * w.unsqueeze(-1)).sum(dim=1)
        video_logit = self.video_head(pooled).squeeze(-1)
        return video_logit, frame_logits, w
