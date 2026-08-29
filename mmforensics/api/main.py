"""FastAPI inference gateway.

Endpoints: POST /analyze/image (multipart image; mode=ai|tamper|both|screen
routes the ONNX detection heads), POST /analyze/text, GET /models (deployed
checkpoint inventory), GET /health. Responses carry label + confidence +
an explanation string. Run with:

    uvicorn mmforensics.api.main:app --reload
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__

app = FastAPI(
    title="Multimedia Forensics API",
    description="Unified detection of tampered/AI-generated images, deepfake "
                "videos, AI-written text, and synthetic audio/music.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# Anchored to this file so the app works regardless of the working directory
# uvicorn is launched from.
ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT / "static"
CHECKPOINT_DIR = ROOT / "checkpoints"

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_ai_detector = None
_tamper_detector = None
_screen_detector = None
_orchestrator = None


def _get_orchestrator():
    """Singleton orchestrator — the text pipeline (~313 MB) must not be
    reloaded on every request."""
    global _orchestrator
    if _orchestrator is None:
        from ..fusion import ForensicOrchestrator

        _orchestrator = ForensicOrchestrator(checkpoint_dir=CHECKPOINT_DIR)
    return _orchestrator


def _get_ai_detector():
    global _ai_detector
    if _ai_detector is None:
        from ..models.image.ai_detector import AIDetectorORT
        _ai_detector = AIDetectorORT()
    return _ai_detector


def _get_tamper_detector():
    global _tamper_detector
    if _tamper_detector is None:
        from ..models.image.tamper_detector import TamperDetectorORT
        _tamper_detector = TamperDetectorORT()
        logger.info("TamperDetectorORT loaded (ONNX)")
    return _tamper_detector


def _get_screen_detector():
    global _screen_detector
    if _screen_detector is None:
        from ..models.image.screen_detector import ScreenDetectorORT
        _screen_detector = ScreenDetectorORT()
    return _screen_detector


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "mean": float(obj.mean()),
                "max": float(obj.max()),
                "flagged_fraction": float((obj > 0.5).mean())}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


class TextRequest(BaseModel):
    text: str
    localize: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/models")
def models():
    """Deployed checkpoint inventory (A/B hook): which modality heads are
    available on disk right now."""
    def existing(*rel: str) -> list[str]:
        return [r for r in rel if (CHECKPOINT_DIR / r).exists()]

    image_files = existing("image_tampering/model_cls.onnx",
                           "image_screen/model.onnx",
                           "image_ai/capcheck_model.onnx")
    text_ok = bool(existing("text/model.safetensors", "text/pytorch_model.bin"))
    return {
        "image": {"deployed": bool(image_files), "checkpoints": image_files},
        "video": {"deployed": bool(existing("video.pt")),
                  "checkpoints": existing("video.pt")},
        "audio": {"deployed": bool(existing("audio.pt")),
                  "checkpoints": existing("audio.pt")},
        "text": {"deployed": text_ok,
                 "checkpoints": ["text/"] if text_ok else []},
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    from ..explainability import explain_report

    report = _get_orchestrator().analyze_text(req.text)
    report["explanation"] = explain_report(report)
    return _jsonable(report)


VALID_IMAGE_MODES = ("ai", "tamper", "both", "screen")


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...), mode: str = Form("ai")):
    from PIL import Image, UnidentifiedImageError
    import base64, io

    if mode not in VALID_IMAGE_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"mode must be one of {', '.join(VALID_IMAGE_MODES)}; got {mode!r}")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        try:
            image = Image.open(tmp_path).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400,
                                detail=f"not a readable image: {file.filename!r}")
        result = {"file": file.filename, "mode": mode}

        if mode in ("tamper", "both"):
            tamper_result = _get_tamper_detector().predict(image)
            mask_b64 = None
            if "mask" in tamper_result:
                mask = tamper_result.pop("mask")
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_uint8, mode="L")
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode()
            result["tamper_detection"] = tamper_result
            result["mask_base64"] = mask_b64

        if mode in ("ai", "both"):
            ai_result = _get_ai_detector().predict(image)
            result["ai_detection"] = ai_result

        if mode == "screen":
            screen_result = _get_screen_detector().predict(image)
            result["screen_detection"] = screen_result

        parts = []
        if "ai_detection" in result:
            ai = result["ai_detection"]
            parts.append(f"AI: {ai['label']} (P(AI)={ai['p_ai']:.3f})")
        if "tamper_detection" in result:
            t = result["tamper_detection"]
            parts.append(f"Tamper: {t['label']} (P(tampered)={t['p_tampered']:.3f})")
        if "screen_detection" in result:
            s = result["screen_detection"]
            parts.append(f"Screen: {s['label']} (P(screen)={s['p_screen']:.3f})")
        result["explanation"] = ". ".join(parts) + "."

        return result
    finally:
        tmp_path.unlink(missing_ok=True)
