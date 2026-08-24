"""End-to-end audio forensics pipeline.

Ensemble of three speech branches — LCNN on LFCC, RawNet2 on raw waveform,
and an optional fine-tuned Wav2Vec2 encoder — plus a separate music
sub-branch (CNN on chroma + spectral-contrast) for AI-generated music,
since AI-music artifacts (looping seams, unnatural harmonic consistency)
differ from vocoder artifacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ...preprocessing.audio_ops import load_audio, lfcc, chroma_spectral_contrast
from .lcnn import LCNN
from .rawnet2 import RawNet2

CLASS_NAMES = ["bonafide", "spoof"]


class MusicCNN(nn.Module):
    """Genre-agnostic CNN over chroma + spectral-contrast (19, T)."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(19, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))


class Wav2Vec2Branch(nn.Module):
    """Self-supervised speech encoder + linear head; transfers well to
    unseen TTS/vocoder types. Loaded lazily because of its size."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", num_classes: int = 2):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.encoder(wav).last_hidden_state.mean(dim=1)
        return self.head(h)


class AudioPipeline:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 sr: int = 16000, use_wav2vec: bool = False):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.sr = sr
        self.lcnn = LCNN().eval().to(self.device)
        self.rawnet = RawNet2().eval().to(self.device)
        self.music = MusicCNN().eval().to(self.device)
        self.w2v = None
        if use_wav2vec:
            self.w2v = Wav2Vec2Branch().eval().to(self.device)
        self.trained = checkpoint is not None
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.lcnn.load_state_dict(state["lcnn"])
            self.rawnet.load_state_dict(state["rawnet"])
            if "music" in state:
                self.music.load_state_dict(state["music"])
            if self.w2v is not None and "w2v" in state:
                self.w2v.load_state_dict(state["w2v"])

    def _speech_scores(self, wav: np.ndarray) -> dict:
        scores = {}
        with torch.no_grad():
            feat = lfcc(wav, sr=self.sr)
            x = torch.from_numpy(feat)[None, None].to(self.device)
            scores["lcnn"] = float(torch.softmax(self.lcnn(x), dim=1)[0, 1])

            w = wav[: self.sr * 6]
            if len(w) < self.sr:  # RawNet2 needs a minimum length
                w = np.pad(w, (0, self.sr - len(w)))
            wt = torch.from_numpy(w)[None].to(self.device)
            scores["rawnet2"] = float(torch.softmax(self.rawnet(wt), dim=1)[0, 1])

            if self.w2v is not None:
                scores["wav2vec2"] = float(torch.softmax(self.w2v(wt), dim=1)[0, 1])
        return scores

    def _music_score(self, wav: np.ndarray) -> float:
        with torch.no_grad():
            feat = chroma_spectral_contrast(wav, sr=self.sr)
            x = torch.from_numpy(feat)[None].to(self.device)
            return float(torch.softmax(self.music(x), dim=1)[0, 1])

    def analyze(self, path_or_wav, is_music: bool | None = None) -> dict:
        wav = (load_audio(path_or_wav, sr=self.sr)
               if isinstance(path_or_wav, (str, Path)) else np.asarray(path_or_wav, np.float32))
        if wav.size < self.sr // 4:
            return {"modality": "audio", "label": "unknown", "confidence": 0.0,
                    "error": "audio too short to analyze"}

        branches = self._speech_scores(wav)
        p_spoof = float(np.mean(list(branches.values())))
        music_p = self._music_score(wav)
        branches["music_cnn"] = music_p
        if is_music:
            p_spoof = music_p

        label = CLASS_NAMES[int(p_spoof >= 0.5)]
        report = {
            "modality": "audio",
            "label": "ai_generated" if label == "spoof" else "bonafide",
            "confidence": p_spoof if label == "spoof" else 1.0 - p_spoof,
            "p_spoof": p_spoof,
            "branches": branches,
            "duration_sec": round(len(wav) / self.sr, 2),
            "trained_checkpoint": self.trained,
        }
        if not self.trained:
            report["warning"] = ("Untrained anti-spoofing heads — train with "
                                 "training/train_audio.py on ASVspoof/WaveFake first.")
        return report
