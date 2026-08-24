"""Video preprocessing: frame sampling and audio-track extraction."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def extract_frames(path: str | Path, num_frames: int = 16, size: int = 224) -> np.ndarray:
    """Uniformly sample num_frames RGB frames from a video.

    Returns float32 array (T, H, W, 3) in [0, 1]. Frames are resized to
    (size, size); missing frames (short videos) are padded by repetition.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    if not frames:
        raise ValueError(f"No decodable frames in video: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames[:num_frames])


def extract_audio_track(path: str | Path, sr: int = 16000) -> np.ndarray | None:
    """Extract the mono audio track of a video via ffmpeg. Returns float32
    waveform at sr, or None if ffmpeg is unavailable or the video is silent."""
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(sr), "-f", "wav", tmp.name]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            import soundfile as sf

            wav, _ = sf.read(tmp.name, dtype="float32")
            if wav.size == 0:
                return None
            return wav
        except Exception:
            return None


def detect_faces(frame: np.ndarray, min_size: int = 40) -> list[tuple[int, int, int, int]]:
    """Detect face bounding boxes (x, y, w, h) in an RGB float frame.

    Tries facenet-pytorch MTCNN if installed, then OpenCV's Haar cascade
    (removed in OpenCV >= 5). With neither available, returns [] and the
    caller falls back to the full frame, which still carries global artifacts.
    """
    import cv2

    try:
        from facenet_pytorch import MTCNN  # optional, best quality

        if not hasattr(detect_faces, "_mtcnn"):
            detect_faces._mtcnn = MTCNN(keep_all=True, post_process=False)
        boxes, _ = detect_faces._mtcnn.detect((frame * 255).astype(np.uint8))
        if boxes is None:
            return []
        return [(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
                for x0, y0, x1, y1 in boxes if (x1 - x0) >= min_size]
    except ImportError:
        pass

    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
        return [tuple(int(v) for v in f) for f in faces]
    return []


def crop_largest_face(frame: np.ndarray, size: int = 224, margin: float = 0.3) -> np.ndarray:
    """Crop the largest detected face (with margin) or return the full frame
    resized when no face is found — the frame still carries global artifacts."""
    import cv2

    faces = detect_faces(frame)
    if faces:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mh, mw = int(h * margin), int(w * margin)
        y0, y1 = max(0, y - mh), min(frame.shape[0], y + h + mh)
        x0, x1 = max(0, x - mw), min(frame.shape[1], x + w + mw)
        crop = frame[y0:y1, x0:x1]
    else:
        crop = frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
