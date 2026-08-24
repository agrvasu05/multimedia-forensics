"""End-to-end video deepfake pipeline: video file -> forensic report.

Routes the visual stream through the spatio-temporal deepfake net and, when
an audio track is present, hands the extracted waveform to the audio
pipeline for cross-modal fusion (done in fusion/).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...preprocessing.video_ops import extract_frames, crop_largest_face, extract_audio_track
from .deepfake_net import SpatioTemporalDeepfakeNet, CLASS_NAMES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoPipeline:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 backbone: str = "xception", num_frames: int = 16, face_size: int = 224):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_frames = num_frames
        self.face_size = face_size
        self.model = SpatioTemporalDeepfakeNet(backbone=backbone,
                                               pretrained=checkpoint is None)
        self.trained = checkpoint is not None
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state.get("model", state))
        self.model.eval().to(self.device)

    def _prepare(self, path: str | Path) -> torch.Tensor:
        frames = extract_frames(path, num_frames=self.num_frames, size=360)
        crops = np.stack([crop_largest_face(f, size=self.face_size) for f in frames])
        crops = (crops - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(crops.transpose(0, 3, 1, 2))[None].float()  # (1,T,3,H,W)

    @torch.no_grad()
    def analyze(self, path: str | Path) -> dict:
        clips = self._prepare(path).to(self.device)
        video_logit, frame_logits, attn = self.model(clips)
        p_fake = float(torch.sigmoid(video_logit)[0])
        frame_scores = torch.sigmoid(frame_logits)[0].cpu().numpy().tolist()

        label = CLASS_NAMES[int(p_fake >= 0.5)]
        report = {
            "modality": "video",
            "label": label,
            "confidence": p_fake if label == "deepfake" else 1.0 - p_fake,
            "p_fake": p_fake,
            "frame_scores": frame_scores,
            "frame_attention": attn[0].cpu().numpy().tolist(),
            "num_frames_analyzed": self.num_frames,
            "trained_checkpoint": self.trained,
        }
        if not self.trained:
            report["warning"] = ("Untrained deepfake head — train with "
                                 "training/train_video.py on FaceForensics++/DFDC first.")
        return report

    def extract_audio(self, path: str | Path):
        """Pull the audio track so the fusion layer can run the audio pipeline."""
        return extract_audio_track(path)
