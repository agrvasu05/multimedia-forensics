import numpy as np
import pytest

from mmforensics.preprocessing import (
    MediaType, detect_media_type, compute_ela, srm_residuals, dct_features,
    clean_text, split_sentences, mel_spectrogram, lfcc,
)
from mmforensics.preprocessing.text_ops import sliding_windows


def test_type_detection(tmp_path):
    from PIL import Image

    img = tmp_path / "a.jpg"
    Image.new("RGB", (32, 32), (200, 30, 30)).save(img)
    assert detect_media_type(img) is MediaType.IMAGE

    txt = tmp_path / "b.txt"
    txt.write_text("hello world")
    assert detect_media_type(txt) is MediaType.TEXT

    import soundfile as sf

    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16000)
    assert detect_media_type(wav) is MediaType.AUDIO


def test_image_forensic_features():
    rng = np.random.default_rng(0)
    rgb = rng.random((64, 64, 3)).astype(np.float32)
    ela = compute_ela(rgb)
    assert ela.shape == (64, 64, 3) and 0 <= ela.min() and ela.max() <= 1

    srm = srm_residuals(rgb)
    assert srm.shape == (64, 64, 3)

    dct = dct_features(rgb)
    assert dct.shape == (18,) and np.isfinite(dct).all()


def test_text_ops():
    t = clean_text("Hello​   world.\n\n\n\nSecond sentence! Third?  ")
    sents = split_sentences(t)
    assert len(sents) >= 2
    wins = sliding_windows(["a.", "b.", "c.", "d."], window=2)
    assert wins[0][:2] == (0, 2) and wins[-1][1] == 4


def test_audio_features():
    sr = 16000
    t = np.arange(sr) / sr
    wav = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    m = mel_spectrogram(wav, sr=sr)
    assert m.shape[0] == 80
    f = lfcc(wav, sr=sr)
    assert f.shape[0] == 60 and np.isfinite(f).all()
