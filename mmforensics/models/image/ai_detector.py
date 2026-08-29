"""AI-generated image detector using capcheck/ai-image-detection (ONNX).

ViT-Base (86M params) fine-tuned on CIFAKE, exported to ONNX for fast CPU
inference. Good balance of speed and accuracy.

Labels: 0 = REAL, 1 = FAKE (AI-generated).
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


class AIDetectorORT:
    """ONNX-based AI-generated image detector (capcheck ViT-Base)."""

    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # Path constant only — no training/downloads happen here; the trained
    # checkpoint in checkpoints/image_ai/ is reused when present.
    CKPT_ROOT = Path(__file__).resolve().parents[3] / "checkpoints"

    def __init__(self, model_path: str | Path | None = None):
        if not HAS_ORT:
            raise ImportError("pip install onnxruntime")
        if not HAS_PIL:
            raise ImportError("pip install pillow")

        if model_path is None:
            model_path = self._download_model()
        model_path = Path(model_path)

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.warm_up()

    def _download_model(self) -> Path:
        from huggingface_hub import hf_hub_download
        cache_dir = self.CKPT_ROOT / "image_ai"
        cache_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_dir / "capcheck_model.onnx"
        if onnx_path.exists():
            return onnx_path
        print("[AIDetector] Downloading capcheck AI image detector ...")
        path = hf_hub_download(
            repo_id="onnx-community/ai-image-detection-ONNX",
            filename="onnx/model.onnx",
            cache_dir=str(cache_dir),
        )
        import shutil
        shutil.copy2(path, onnx_path)
        return onnx_path

    def warm_up(self):
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        self.session.run(None, {self.input_name: dummy})

    def preprocess(self, image: Image.Image) -> np.ndarray:
        import io
        buf = io.BytesIO()
        image.convert("RGB").save(buf, "JPEG", quality=95)
        buf.seek(0)
        img = Image.open(buf).convert("RGB").resize((224, 224), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)
        return np.expand_dims(arr, 0).astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        blob = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: blob})[0]
        probs = self._softmax(logits[0])
        p_real = float(probs[0])
        p_fake = float(probs[1])
        if p_fake > 0.7:
            label = "ai_generated"
        elif p_real > 0.7:
            label = "real"
        else:
            label = "uncertain"
        return {
            "label": label,
            "confidence": max(p_fake, p_real),
            "p_ai": p_fake,
            "p_real": p_real,
        }

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()
