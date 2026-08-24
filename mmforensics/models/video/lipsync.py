"""Audio-visual lip-sync consistency check.

A SyncNet-style learned embedding comparison needs pretrained SyncNet
weights; this module implements the training-free signal underneath it:
the temporal correlation between mouth-region motion energy and the audio
loudness envelope. In genuine speech the two co-vary strongly; in a
face-swap with mismatched audio (or a lip-sync deepfake with imperfect
alignment) the correlation collapses. Swap `lipsync_score` for a real
SyncNet forward pass to upgrade without touching the callers.
"""
from __future__ import annotations

import numpy as np


def mouth_motion_energy(frames: np.ndarray) -> np.ndarray:
    """Per-transition motion energy of the mouth region.

    frames: (T, H, W, 3) float face crops in [0,1] (mouth ≈ lower-middle
    third of an aligned face crop). Returns (T-1,) energies.
    """
    t, h, w = frames.shape[:3]
    mouth = frames[:, int(h * 0.60):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
    gray = mouth @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    diffs = np.abs(np.diff(gray, axis=0))
    return diffs.mean(axis=(1, 2))


def audio_envelope(wav: np.ndarray, sr: int, n_points: int) -> np.ndarray:
    """RMS loudness envelope resampled to n_points values."""
    hop = max(len(wav) // (n_points * 4), 1)
    n_frames = len(wav) // hop
    rms = np.sqrt(np.mean(
        wav[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1) + 1e-12)
    x_old = np.linspace(0, 1, len(rms))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, rms)


def lipsync_score(frames: np.ndarray, wav: np.ndarray, sr: int = 16000,
                  max_lag: int = 2) -> dict | None:
    """Correlate mouth motion with the audio envelope over small lags.

    Returns {"sync_corr": best correlation in [-1,1], "best_lag": frames,
    "mismatch": bool} or None when the signal is too short/flat to judge.
    """
    if frames.shape[0] < 6 or wav.size < sr // 2:
        return None
    motion = mouth_motion_energy(frames)
    env = audio_envelope(wav, sr, n_points=len(motion))
    # a near-constant signal (silence, flat tone, static mouth) makes the
    # correlation spurious — require real relative variability in both
    if (motion.std() < 0.05 * (motion.mean() + 1e-9)
            or env.std() < 0.05 * (env.mean() + 1e-9)):
        return None

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    # on short series a wide lag search cherry-picks spurious alignments;
    # allow at most one lag step per ~8 samples
    max_lag = min(max_lag, len(motion) // 8)
    best, best_lag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            c = corr(motion[lag:], env[: len(env) - lag])
        elif lag < 0:
            c = corr(motion[: lag], env[-lag:])
        else:
            c = corr(motion, env)
        if c > best:
            best, best_lag = c, lag
    return {"sync_corr": round(best, 4), "best_lag": best_lag,
            # sparse frame sampling makes this a soft signal: only a clearly
            # negative/absent correlation is treated as a mismatch flag
            "mismatch": best < 0.1}
