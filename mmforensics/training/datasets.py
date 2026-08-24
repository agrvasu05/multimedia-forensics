"""Dataset classes.

All modalities use a simple folder convention so any downloaded benchmark
(CASIA, GenImage, FaceForensics++, ASVspoof, HC3, ...) can be dropped in
after conversion by scripts/download_datasets.py:

    data/<modality>/<split>/<class>/*            e.g. data/image/train/real/x.jpg
    data/image/<split>/masks/<stem>.png          optional tampering masks
    data/text/<split>/<class>.jsonl              one {"text": ...} per line

`make_synthetic_*` builders generate tiny synthetic corpora so the full
training loop can be smoke-tested end to end without any downloads.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..preprocessing.image_ops import compute_ela, srm_residuals, dct_features

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_CLASSES = {"real": 0, "tampered": 1, "ai_generated": 2}


def _to_chw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float()


class ImageForensicsDataset(Dataset):
    """Yields (rgb, artifacts, dct, label, mask) tuples for the dual-branch net."""

    def __init__(self, root: str | Path, split: str = "train", size: int = 256,
                 augment: bool = False):
        self.size, self.augment = size, augment
        self.items: list[tuple[Path, int]] = []
        base = Path(root) / split
        for cls, idx in IMG_CLASSES.items():
            d = base / cls
            if d.is_dir():
                self.items += [(p, idx) for p in sorted(d.iterdir())
                               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        self.mask_dir = base / "masks"
        self.tf = _image_augment(size) if augment else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        from PIL import Image

        img = np.asarray(Image.open(path).convert("RGB").resize((self.size, self.size)),
                         dtype=np.float32) / 255.0
        mask_path = self.mask_dir / f"{path.stem}.png"
        if mask_path.exists():
            m = np.asarray(Image.open(mask_path).convert("L").resize((self.size, self.size)),
                           dtype=np.float32) / 255.0
        else:
            m = np.zeros((self.size, self.size), dtype=np.float32)

        if self.tf is not None:
            out = self.tf(image=(img * 255).astype(np.uint8), mask=m)
            img = out["image"].astype(np.float32) / 255.0
            m = out["mask"].astype(np.float32)

        ela = compute_ela(img)
        srm = srm_residuals(img)
        dct = dct_features(img)
        rgb_n = (img - IMAGENET_MEAN) / IMAGENET_STD
        artifacts = np.concatenate([ela, srm], axis=-1)
        return (_to_chw(rgb_n), _to_chw(artifacts), torch.from_numpy(dct),
                torch.tensor(label), torch.from_numpy(m)[None])


def _image_augment(size: int):
    """Plan §8.2: JPEG recompression, resizing, blur, color jitter, cropping."""
    import albumentations as A

    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(40, 95), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.3),
    ])


class VideoFramesDataset(Dataset):
    """Yields (frames (T,3,H,W), label) from pre-extracted frame folders:
    data/video/<split>/<class>/<clip_id>/*.jpg  (class in {real, deepfake})."""

    def __init__(self, root: str | Path, split: str = "train",
                 num_frames: int = 8, size: int = 160):
        self.num_frames, self.size = num_frames, size
        self.items: list[tuple[Path, int]] = []
        base = Path(root) / split
        for cls, idx in {"real": 0, "deepfake": 1}.items():
            d = base / cls
            if d.is_dir():
                self.items += [(clip, idx) for clip in sorted(d.iterdir()) if clip.is_dir()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        clip_dir, label = self.items[i]
        from PIL import Image

        frames = sorted(clip_dir.glob("*.jpg")) + sorted(clip_dir.glob("*.png"))
        idxs = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
        stack = []
        for j in idxs:
            img = np.asarray(Image.open(frames[j]).convert("RGB")
                             .resize((self.size, self.size)), dtype=np.float32) / 255.0
            stack.append((img - IMAGENET_MEAN) / IMAGENET_STD)
        clip = np.stack(stack).transpose(0, 3, 1, 2)
        return torch.from_numpy(clip).float(), torch.tensor(label)


class TextForensicsDataset(Dataset):
    """Reads data/text/<split>/{human,ai}.jsonl with one {"text": ...} per line."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.samples: list[tuple[str, int]] = []
        base = Path(root) / split
        for cls, idx in {"human": 0, "ai": 1}.items():
            f = base / f"{cls}.jsonl"
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.strip():
                        self.samples.append((json.loads(line)["text"], idx))
        random.Random(0).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class AudioSpoofDataset(Dataset):
    """Yields (lfcc (1,F,T), waveform (L,), label) from
    data/audio/<split>/{bonafide,spoof}/*.{wav,flac}."""

    def __init__(self, root: str | Path, split: str = "train",
                 sr: int = 16000, seconds: float = 4.0):
        from ..preprocessing.audio_ops import lfcc as _lfcc  # noqa: F401

        self.sr, self.samples_len = sr, int(sr * seconds)
        self.items: list[tuple[Path, int]] = []
        base = Path(root) / split
        for cls, idx in {"bonafide": 0, "spoof": 1}.items():
            d = base / cls
            if d.is_dir():
                self.items += [(p, idx) for p in sorted(d.iterdir())
                               if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from ..preprocessing.audio_ops import load_audio, lfcc

        path, label = self.items[i]
        wav = load_audio(path, sr=self.sr, max_seconds=self.samples_len / self.sr)
        if len(wav) < self.samples_len:
            wav = np.pad(wav, (0, self.samples_len - len(wav)))
        wav = wav[: self.samples_len]
        feat = lfcc(wav, sr=self.sr)
        return (torch.from_numpy(feat)[None], torch.from_numpy(wav),
                torch.tensor(label))


# --------------------------------------------------------------------------
# Synthetic smoke-test corpora: real training data stand-ins so every
# training script runs end to end on any machine with zero downloads.
# --------------------------------------------------------------------------

def make_synthetic_image_dataset(root: str | Path, n_per_class: int = 24, size: int = 256):
    """'real' = smooth gradients+noise; 'tampered' = same with a pasted patch
    (mask saved); 'ai_generated' = periodic frequency-artifact textures."""
    from PIL import Image

    rng = np.random.default_rng(0)
    root = Path(root)
    for split, n in [("train", n_per_class), ("val", max(n_per_class // 3, 4))]:
        for cls in IMG_CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        for k in range(n):
            base = _gradient_image(rng, size)
            Image.fromarray(base).save(root / split / "real" / f"r{k}.jpg", quality=90)

            tampered = base.copy()
            x, y, w = rng.integers(20, size - 84), rng.integers(20, size - 84), 64
            patch = _gradient_image(rng, 64)
            tampered[y:y + w, x:x + w] = patch
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[y:y + w, x:x + w] = 255
            Image.fromarray(tampered).save(root / split / "tampered" / f"t{k}.jpg", quality=90)
            Image.fromarray(mask).save(root / split / "masks" / f"t{k}.png")

            xx, yy = np.meshgrid(np.arange(size), np.arange(size))
            freq = rng.uniform(0.2, 0.6, 2)
            ai = (127 + 90 * np.sin(freq[0] * xx) * np.cos(freq[1] * yy))
            ai = np.stack([ai] * 3, -1) + rng.normal(0, 4, (size, size, 3))
            Image.fromarray(np.clip(ai, 0, 255).astype(np.uint8)).save(
                root / split / "ai_generated" / f"g{k}.jpg", quality=90)


def _gradient_image(rng, size: int) -> np.ndarray:
    direction = rng.uniform(0, 2 * np.pi)
    xx, yy = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    g = xx * np.cos(direction) + yy * np.sin(direction)
    img = np.stack([g * rng.uniform(80, 255) for _ in range(3)], -1)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_synthetic_video_dataset(root: str | Path, n_per_class: int = 8,
                                 frames: int = 8, size: int = 160):
    """'real' clips move smoothly; 'deepfake' clips add per-frame flicker in a
    face-sized region — the temporal artifact the model should learn."""
    from PIL import Image

    rng = np.random.default_rng(1)
    root = Path(root)
    for split, n in [("train", n_per_class), ("val", max(n_per_class // 2, 2))]:
        for cls in ["real", "deepfake"]:
            for k in range(n):
                d = root / split / cls / f"clip{k}"
                d.mkdir(parents=True, exist_ok=True)
                base = _gradient_image(rng, size)
                for t in range(frames):
                    frame = np.roll(base, shift=t * 3, axis=1).astype(np.float32)
                    if cls == "deepfake":
                        cx, cy, r = size // 2, size // 2, size // 4
                        flick = rng.normal(0, 25)
                        frame[cy - r:cy + r, cx - r:cx + r] += flick
                    Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8)).save(
                        d / f"{t:03d}.jpg", quality=85)


AI_TEXT_TEMPLATES = [
    "It is important to note that {t} plays a significant role in modern society. "
    "Furthermore, {t} offers numerous benefits. In conclusion, {t} is essential.",
    "There are several key aspects of {t}. Firstly, it is widely used. "
    "Secondly, it is highly efficient. Finally, it is very reliable overall.",
]
HUMAN_TEXT_SNIPPETS = [
    "honestly? i tried {t} once and hated it. my roommate swears by it though — go figure. "
    "we argued about it for like an hour over cold pizza.",
    "The {t} broke again Tuesday. Dad fixed it with duct tape, said that's how his father "
    "did it. Worked, weirdly enough! Until Thursday.",
]
TOPICS = ["machine learning", "gardening", "coffee", "cycling", "photography",
          "cooking", "travel", "music", "chess", "astronomy"]


def make_synthetic_text_dataset(root: str | Path, n_per_class: int = 40):
    root = Path(root)
    rng = random.Random(2)
    for split, n in [("train", n_per_class), ("val", max(n_per_class // 4, 6))]:
        (root / split).mkdir(parents=True, exist_ok=True)
        for cls, templates in [("ai", AI_TEXT_TEMPLATES), ("human", HUMAN_TEXT_SNIPPETS)]:
            with (root / split / f"{cls}.jsonl").open("w") as f:
                for _ in range(n):
                    t = rng.choice(TOPICS)
                    text = rng.choice(templates).format(t=t)
                    f.write(json.dumps({"text": text}) + "\n")


def make_synthetic_audio_dataset(root: str | Path, n_per_class: int = 12,
                                 sr: int = 16000, seconds: float = 2.0):
    """'bonafide' = noisy multi-harmonic tones with vibrato; 'spoof' = clean
    pure tones with hard looping seams (vocoder-artifact stand-ins)."""
    import soundfile as sf

    rng = np.random.default_rng(3)
    root = Path(root)
    n_samp = int(sr * seconds)
    t = np.arange(n_samp) / sr
    for split, n in [("train", n_per_class), ("val", max(n_per_class // 3, 3))]:
        for cls in ["bonafide", "spoof"]:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                f0 = rng.uniform(120, 300)
                if cls == "bonafide":
                    vib = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
                    wav = sum(rng.uniform(0.2, 0.5) / (h + 1) *
                              np.sin(2 * np.pi * f0 * (h + 1) * t * vib) for h in range(5))
                    wav += rng.normal(0, 0.02, n_samp)
                else:
                    seg = np.sin(2 * np.pi * f0 * t[: n_samp // 4])
                    wav = np.tile(seg, 4)[:n_samp] * 0.5  # looping seams
                sf.write(d / f"{cls[0]}{k}.wav", wav.astype(np.float32), sr)
