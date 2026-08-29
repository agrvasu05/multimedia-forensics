"""Screen detection model — real photo vs screen capture/recaptured image.

MobileNetV2 (384x384) exported to ONNX. Sigmoid output with threshold 0.49.
Labels: 0 = Real Photo, 1 = Screen/Recaptured.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ScreenDetectorORT:
    """ONNX-based screen vs real photo detector."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    THRESHOLD = 0.49

    def __init__(self, model_path: str | Path | None = None):
        if not HAS_ORT:
            raise ImportError("pip install onnxruntime")
        if not HAS_PIL:
            raise ImportError("pip install pillow")

        if model_path is None:
            model_path = Path("checkpoints/image_screen/model.onnx")
        model_path = Path(model_path)

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.warm_up()

    def warm_up(self):
        dummy = np.random.randn(1, 3, 384, 384).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((384, 384), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, 0).astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        blob = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: blob})[0]
        prob_screen = float(1 / (1 + np.exp(-logits[0, 0])))
        prob_real = 1.0 - prob_screen
        label = "screen" if prob_screen >= self.THRESHOLD else "real_photo"
        return {
            "label": label,
            "confidence": max(prob_screen, prob_real),
            "p_screen": prob_screen,
            "p_real": prob_real,
        }
