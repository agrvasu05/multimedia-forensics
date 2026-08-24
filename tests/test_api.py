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


def test_analyze_image_endpoint(client, tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    arr = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    p = tmp_path / "x.jpg"
    Image.fromarray(arr).save(p)
    with p.open("rb") as f:
        r = client.post("/analyze", files={"file": ("x.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "image"
    assert body["label"] in {"real", "tampered", "ai_generated"}
    assert "localization_mask" in body  # summarized dict, not raw array


def test_analyze_audio_endpoint(client, tmp_path):
    import soundfile as sf

    sr = 16000
    t = np.arange(sr * 2) / sr
    wav = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, wav, sr)
    with p.open("rb") as f:
        r = client.post("/analyze", files={"file": ("tone.wav", f, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "audio"
    assert body["label"] in {"bonafide", "ai_generated"}
