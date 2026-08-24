"""FastAPI inference gateway.

Unified /analyze endpoint: accepts any supported file (image / video / audio
/ text), auto-detects its type, routes it through the right pipeline(s), and
returns label + confidence + localization/spans + a natural-language
explanation. Run with:

    uvicorn mmforensics.api.main:app --reload
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from ..fusion import ForensicOrchestrator
from ..explainability import explain_report
from .. import __version__

app = FastAPI(
    title="Multimedia Forensics API",
    description="Unified detection of tampered/AI-generated images, deepfake "
                "videos, AI-written text, and synthetic audio/music.",
    version=__version__,
)

orchestrator = ForensicOrchestrator(checkpoint_dir=Path("checkpoints"))


def _jsonable(obj):
    """Strip numpy arrays (masks) down to summary stats so responses stay small."""
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


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Analyze any supported media file."""
    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        report = orchestrator.analyze_file(tmp_path)
        report["file"] = file.filename
        report["explanation"] = explain_report(report)
        return _jsonable(report)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    """Analyze raw text directly (no file upload needed)."""
    report = orchestrator.analyze_text(req.text)
    report["explanation"] = explain_report(report)
    return _jsonable(report)
