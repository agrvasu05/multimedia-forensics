"""Tests for the second-pass features: PRNU, augmentations, lip-sync,
T5 perturbation plumbing, and ONNX export."""
import numpy as np
import pytest
import torch


def test_prnu_residual():
    from mmforensics.preprocessing.image_ops import prnu_residual

    rng = np.random.default_rng(0)
    rgb = rng.random((64, 64, 3)).astype(np.float32)
    p = prnu_residual(rgb)
    assert p.shape == (64, 64, 1)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_image_dataset_yields_7_channel_artifacts(tmp_path):
    from mmforensics.training.datasets import (ImageForensicsDataset,
                                               make_synthetic_image_dataset)

    make_synthetic_image_dataset(tmp_path, n_per_class=2, size=64)
    ds = ImageForensicsDataset(tmp_path, "train", size=64)
    rgb, art, dct, y, mask = ds[0]
    assert art.shape == (7, 64, 64)


def test_audio_augment_preserves_range():
    from mmforensics.training.datasets import _augment_audio

    sr = 16000
    wav = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    for _ in range(5):
        out = _augment_audio(wav, sr)
        assert np.abs(out).max() <= 1.0 + 1e-6
        assert np.isfinite(out).all()


def test_text_augment():
    import random
    from mmforensics.training.datasets import augment_text

    rng = random.Random(0)
    text = ("It is important to note that machine learning provides numerous "
            "benefits. It is essential for modern industry. Various uses exist.")
    out = augment_text(text, rng)
    assert isinstance(out, str) and len(out.split()) >= 10


def test_video_dataset_augment(tmp_path):
    from mmforensics.training.datasets import (VideoFramesDataset,
                                               make_synthetic_video_dataset)

    make_synthetic_video_dataset(tmp_path, n_per_class=1, frames=4, size=64)
    ds = VideoFramesDataset(tmp_path, "train", num_frames=4, size=64, augment=True)
    clip, y = ds[0]
    assert clip.shape == (4, 3, 64, 64) and torch.isfinite(clip).all()


def test_lipsync_score():
    from mmforensics.models.video.lipsync import lipsync_score

    sr, t_frames = 16000, 12
    # synced case: mouth motion amplitude tracks audio loudness frame by frame
    frames = np.full((t_frames, 64, 64, 3), 0.5, dtype=np.float32)
    wav = np.zeros(sr, dtype=np.float32)
    frame_len = sr // t_frames
    amps = [0.0, 0.05, 0.3, 0.02, 0.25, 0.0, 0.35, 0.05, 0.2, 0.0, 0.3, 0.1]
    for i, amp in enumerate(amps):
        frames[i, 40:60, 16:48] += amp
        seg = np.arange(frame_len) / sr
        wav[i * frame_len:(i + 1) * frame_len] = amp * 2 * np.sin(2 * np.pi * 200 * seg)
    synced = lipsync_score(frames, wav, sr)
    assert synced is not None and synced["sync_corr"] > 0.3

    # flat tone: constant envelope carries no sync information -> None
    wav_flat = (0.5 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    assert lipsync_score(frames, wav_flat, sr) is None

    # mismatched case: loudness pattern anti-aligned with the mouth motion
    wav_anti = np.zeros(sr, dtype=np.float32)
    for i, amp in enumerate(amps):
        seg = np.arange(frame_len) / sr
        wav_anti[i * frame_len:(i + 1) * frame_len] = \
            (0.4 - amp) * 2 * np.sin(2 * np.pi * 200 * seg)
    mismatched = lipsync_score(frames, wav_anti, sr)
    assert mismatched is not None and mismatched["sync_corr"] < synced["sync_corr"]

    # degenerate input -> None
    assert lipsync_score(frames[:3], wav, sr) is None


def test_detectgpt_lexical_perturb_only():
    """Plumbing test: lexical perturbation path (no LM download needed)."""
    import random
    from mmforensics.models.text.detector import TextPipeline

    pipe = TextPipeline.__new__(TextPipeline)  # skip __init__ (no model load)
    words = "the quick brown fox jumps over the lazy dog again and again".split()
    out = pipe._lexical_perturb(words, 0.2, random.Random(0))
    assert isinstance(out, str) and len(out.split()) >= len(words) - 3


def test_onnx_export_audio(tmp_path):
    """Round-trip: save an audio checkpoint, export to ONNX, check parity."""
    onnx = pytest.importorskip("onnx")  # noqa: F841
    pytest.importorskip("onnxruntime")
    import subprocess, sys
    from pathlib import Path
    from mmforensics.models.audio.lcnn import LCNN
    from mmforensics.models.audio.rawnet2 import RawNet2

    ckpt = tmp_path / "audio.pt"
    torch.save({"lcnn": LCNN().state_dict(), "rawnet": RawNet2().state_dict()}, ckpt)
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(root / "scripts/export_onnx.py"),
                        "--modality", "audio", "--checkpoint", str(ckpt)],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout and "MISMATCH" not in r.stdout
    assert (tmp_path / "audio_lcnn.onnx").exists()
    assert (tmp_path / "audio_rawnet2.onnx").exists()


def test_models_endpoint():
    from fastapi.testclient import TestClient
    from mmforensics.api.main import app

    r = TestClient(app).get("/models")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"image", "video", "audio", "text"}
    assert "deployed" in body["image"]
