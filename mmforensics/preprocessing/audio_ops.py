"""Audio preprocessing: waveform loading and forensic spectral front-ends."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: str | Path, sr: int = 16000, max_seconds: float = 30.0) -> np.ndarray:
    """Load a mono float32 waveform resampled to sr, truncated to max_seconds."""
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_seconds)
    return wav.astype(np.float32)


def mel_spectrogram(wav: np.ndarray, sr: int = 16000, n_mels: int = 80,
                    n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Log-Mel spectrogram (n_mels, T) — CNN front-end representation."""
    import librosa

    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                                       hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


def lfcc(wav: np.ndarray, sr: int = 16000, n_lfcc: int = 60,
         n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Linear-frequency cepstral coefficients (n_lfcc, T) — the classic
    ASVspoof anti-spoofing front-end (linear filterbank keeps the high-band
    vocoder artifacts that Mel compression discards)."""
    import librosa
    from scipy.fftpack import dct

    S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop)) ** 2
    fb = _linear_filterbank(sr, n_fft, n_filters=n_lfcc + 8)
    feat = np.log(fb @ S + 1e-10)
    return dct(feat, axis=0, norm="ortho")[:n_lfcc].astype(np.float32)


def _linear_filterbank(sr: int, n_fft: int, n_filters: int) -> np.ndarray:
    """Triangular linear-spaced filterbank (n_filters, n_fft//2 + 1)."""
    n_bins = n_fft // 2 + 1
    edges = np.linspace(0, n_bins - 1, n_filters + 2)
    fb = np.zeros((n_filters, n_bins), dtype=np.float32)
    for i in range(n_filters):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        for b in range(int(np.ceil(left)), int(right) + 1):
            up = (b - left) / (center - left) if center > left else 1.0
            down = (right - b) / (right - center) if right > center else 1.0
            fb[i, b] = max(0.0, min(up, down))
    return fb


def chroma_spectral_contrast(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Chroma + spectral-contrast features for the AI-music sub-branch,
    stacked as (12 + 7, T)."""
    import librosa

    chroma = librosa.feature.chroma_stft(y=wav, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr)
    t = min(chroma.shape[1], contrast.shape[1])
    return np.vstack([chroma[:, :t], contrast[:, :t]]).astype(np.float32)
