"""Image tampering detection pipeline using ONNX Runtime.

Loads the exported ONNX classification head for binary (real vs tampered) detection.
Uses the same preprocessing as training: ELA + SRM + PRNU artifacts + DCT features.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Repo-root anchored so the checkpoint is found regardless of CWD
CKPT_ROOT = Path(__file__).resolve().parents[3] / "checkpoints"


class TamperDetectorORT:
    """ONNX Runtime image tampering detector."""

    def __init__(self, checkpoint: str | Path | None = None):
        import onnxruntime as ort

        from ...preprocessing.image_ops import (compute_ela, srm_residuals,
                                                dct_features, prnu_residual)

        if checkpoint is None:
            checkpoint = CKPT_ROOT / "image_tampering" / "model_cls.onnx"
        checkpoint = Path(checkpoint)

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(checkpoint), providers=providers)

        self.compute_ela = compute_ela
        self.srm_residuals = srm_residuals
        self.dct_features = dct_features
        self.prnu_residual = prnu_residual

    def predict(self, image) -> dict:
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        img = np.asarray(image.resize((256, 256), Image.BICUBIC), dtype=np.float32) / 255.0

        rgb_n = (img - IMAGENET_MEAN) / IMAGENET_STD
        ela = self.compute_ela(img)
        srm = self.srm_residuals(img)
        dct = self.dct_features(img)
        prnu = self.prnu_residual(img)
        artifacts = np.concatenate([ela, srm, prnu], axis=-1)

        rgb = rgb_n.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        art = artifacts.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        dct_arr = dct[np.newaxis].astype(np.float32)

        logits, = self.session.run(None, {
            "rgb": rgb,
            "artifacts": art,
            "dct": dct_arr,
        })

        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        p_tampered = float(probs[0, 1])
        p_real = float(probs[0, 0])
        label = "tampered" if p_tampered > 0.5 else "real"

        return {
            "label": label,
            "confidence": max(p_tampered, p_real),
            "p_tampered": p_tampered,
            "p_real": p_real,
        }
