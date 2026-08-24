"""Dual-branch image forensics network.

RGB-content branch (EfficientNet/ViT via timm) + forensic-artifact branch
(shallow CNN over ELA + SRM residual maps) + frequency branch (blockwise-DCT
statistics), fused into an MLP classifier with a U-Net-style decoder head
for pixel-level tampering localization.

Labels: 0 = real, 1 = tampered (splicing/copy-move/retouching), 2 = AI-generated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 3
CLASS_NAMES = ["real", "tampered", "ai_generated"]


class ArtifactCNN(nn.Module):
    """Shallow CNN over stacked ELA(3) + SRM(3) + PRNU(1) maps. Keeps
    intermediate feature maps for the localization decoder's skip connections."""

    def __init__(self, in_ch: int = 7, width: int = 32):
        super().__init__()
        self.block1 = self._block(in_ch, width)
        self.block2 = self._block(width, width * 2)
        self.block3 = self._block(width * 2, width * 4)
        self.out_dim = width * 4

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor):
        f1 = self.block1(x)   # /2
        f2 = self.block2(f1)  # /4
        f3 = self.block3(f2)  # /8
        return f3, (f1, f2, f3)


class LocalizationDecoder(nn.Module):
    """U-Net-style decoder over artifact-branch features -> tampering mask."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec1 = ArtifactCNN._block(width * 4, width * 2)[:-1]  # drop pooling
        self.up2 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec2 = ArtifactCNN._block(width * 2, width)[:-1]
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, skips, out_size):
        f1, f2, f3 = skips
        x = self.up1(f3)
        if x.shape[-2:] != f2.shape[-2:]:  # odd-size inputs floor-divide in pooling
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, f2], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, f1], dim=1))
        mask = self.head(x)
        return F.interpolate(mask, size=out_size, mode="bilinear", align_corners=False)


class DualBranchImageForensics(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b4", pretrained: bool = True,
                 dct_dim: int = 18, width: int = 32, num_classes: int = NUM_CLASSES):
        super().__init__()
        import timm

        self.rgb = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        rgb_dim = self.rgb.num_features
        self.artifact = ArtifactCNN(in_ch=7, width=width)
        self.decoder = LocalizationDecoder(width=width)
        self.dct_proj = nn.Sequential(nn.Linear(dct_dim, 64), nn.ReLU(inplace=True))
        fused = rgb_dim + self.artifact.out_dim + 64
        self.classifier = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, artifacts: torch.Tensor, dct: torch.Tensor):
        """rgb: (B,3,H,W); artifacts: (B,7,H,W) ELA+SRM+PRNU stack; dct: (B,dct_dim).

        Returns (logits (B,C), mask_logits (B,1,H,W)).
        """
        rgb_feat = self.rgb(rgb)
        art_map, skips = self.artifact(artifacts)
        art_feat = F.adaptive_avg_pool2d(art_map, 1).flatten(1)
        dct_feat = self.dct_proj(dct)
        logits = self.classifier(torch.cat([rgb_feat, art_feat, dct_feat], dim=1))
        mask = self.decoder(skips, out_size=rgb.shape[-2:])
        return logits, mask
