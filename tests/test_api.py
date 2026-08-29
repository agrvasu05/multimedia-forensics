import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from mmforensics.api.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_text_endpoint(client):
    r = client.post("/analyze/text", json={
        "text": "It is important to note that machine learning plays a significant "
                "role in modern society. Furthermore, machine learning offers "
                "numerous benefits for various industries. In conclusion, machine "
                "learning is essential for future development. Additionally, it "
                "provides many advantages. Moreover, its applications continue to "
                "grow rapidly across all sectors of the global economy today.",
        "localize": False})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "text"
    assert body["label"] in {"human", "ai_generated"}
    assert "explanation" in body and 0.0 <= body["p_ai"] <= 1.0


class _FakeTamperDetector:
    def predict(self, image):
        return {"label": "tampered", "confidence": 0.83,
                "p_tampered": 0.83, "p_real": 0.17}


class _FakeAIDetector:
    def predict(self, image):
        return {"label": "ai_generated", "confidence": 0.91,
                "p_ai": 0.91, "p_real": 0.09}


class _FakeScreenDetector:
    def predict(self, image):
        return {"label": "real_photo", "confidence": 0.88,
                "p_screen": 0.12, "p_real": 0.88}


@pytest.fixture()
def fake_detectors(monkeypatch):
    """Hermetic image-mode tests: no ONNX session / model download."""
    from mmforensics.api import main

    monkeypatch.setattr(main, "_get_tamper_detector", lambda: _FakeTamperDetector())
    monkeypatch.setattr(main, "_get_ai_detector", lambda: _FakeAIDetector())
    monkeypatch.setattr(main, "_get_screen_detector", lambda: _FakeScreenDetector())


def _png_bytes() -> bytes:
    import io
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_image_endpoint(client, fake_detectors):
    r = client.post("/analyze/image",
                    files={"file": ("x.png", _png_bytes(), "image/png")},
                    data={"mode": "both"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "both"
    assert body["tamper_detection"]["label"] in {"real", "tampered"}
    assert body["ai_detection"]["label"] in {"real", "ai_generated", "uncertain"}
    assert "explanation" in body and "Tamper" in body["explanation"]


def test_analyze_image_screen_mode(client, fake_detectors):
    r = client.post("/analyze/image",
                    files={"file": ("x.png", _png_bytes(), "image/png")},
                    data={"mode": "screen"})
    assert r.status_code == 200
    body = r.json()
    assert body["screen_detection"]["label"] in {"real_photo", "screen"}
    assert "Screen" in body["explanation"]


def test_analyze_image_invalid_mode(client, fake_detectors):
    r = client.post("/analyze/image",
                    files={"file": ("x.png", _png_bytes(), "image/png")},
                    data={"mode": "bogus"})
    assert r.status_code == 422


def test_analyze_image_rejects_non_image(client):
    r = client.post("/analyze/image",
                    files={"file": ("notes.txt", b"definitely not an image", "text/plain")},
                    data={"mode": "ai"})
    assert r.status_code == 400
