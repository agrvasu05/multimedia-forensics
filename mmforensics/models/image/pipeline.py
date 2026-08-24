"""End-to-end image forensics pipeline: file in -> forensic report out."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...preprocessing.image_ops import load_image, compute_ela, srm_residuals, dct_features
from .dual_branch import DualBranchImageForensics, CLASS_NAMES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImagePipeline:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 backbone: str = "efficientnet_b4", input_size: int = 380):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = input_size
        self.model = DualBranchImageForensics(backbone=backbone,
                                              pretrained=checkpoint is None)
        self.trained = checkpoint is not None
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.model.eval().to(self.device)

    def _prepare(self, path: str | Path):
        rgb = load_image(path, size=self.input_size)
        ela = compute_ela(path)
        srm = srm_residuals(rgb)
        # ELA is computed at native resolution; resize to model input
        import cv2

        ela = cv2.resize(ela, (self.input_size, self.input_size))
        dct = dct_features(load_image(path))  # native-res blocks for real JPEG stats
        artifacts = np.concatenate([ela, srm], axis=-1)  # H,W,6

        rgb_n = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None]
        return (to_t(rgb_n).float(), to_t(artifacts).float(),
                torch.from_numpy(dct)[None].float(), rgb, ela)

    @torch.no_grad()
    def analyze(self, path: str | Path) -> dict:
        rgb_t, art_t, dct_t, rgb, ela = self._prepare(path)
        rgb_t, art_t, dct_t = (t.to(self.device) for t in (rgb_t, art_t, dct_t))
        logits, mask_logits = self.model(rgb_t, art_t, dct_t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()

        idx = int(probs.argmax())
        report = {
            "modality": "image",
            "label": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in zip(CLASS_NAMES, probs)},
            "localization_mask": mask,  # HxW float in [0,1]
            "ela_mean": float(ela.mean()),
            "trained_checkpoint": self.trained,
        }
        if not self.trained:
            report["warning"] = ("Model is running with ImageNet-pretrained backbone but an "
                                 "untrained forensic head — train with training/train_image.py "
                                 "before trusting these scores.")
        return report
