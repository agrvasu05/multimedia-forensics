"""Hub-and-spoke orchestrator: detects file type, routes to the right expert
pipeline(s), fuses multi-pipeline outputs into one forensic report.

A video is routed to BOTH the video pipeline (visual stream) and the audio
pipeline (extracted track); the fusion rule flags the file if either stream
is fake, weighting by per-stream confidence.
"""
from __future__ import annotations

from pathlib import Path

from ..preprocessing.type_detect import MediaType, detect_media_type


class ForensicOrchestrator:
    """Lazily builds pipelines on first use so a text-only query never loads
    vision weights. Checkpoints are picked up from checkpoint_dir when the
    files exist (image.pt / video.pt / text/ / audio.pt)."""

    def __init__(self, checkpoint_dir: str | Path = "checkpoints", device: str | None = None):
        self.ckpt_dir = Path(checkpoint_dir)
        self.device = device
        self._pipes: dict[str, object] = {}

    def _ckpt(self, name: str) -> Path | None:
        p = self.ckpt_dir / name
        return p if p.exists() else None

    def _get(self, modality: str):
        if modality in self._pipes:
            return self._pipes[modality]
        if modality == "image":
            from ..models.image import ImagePipeline

            pipe = ImagePipeline(checkpoint=self._ckpt("image.pt"), device=self.device)
        elif modality == "video":
            from ..models.video import VideoPipeline

            pipe = VideoPipeline(checkpoint=self._ckpt("video.pt"), device=self.device)
        elif modality == "text":
            from ..models.text import TextPipeline

            ckpt = self.ckpt_dir / "text"
            pipe = TextPipeline(checkpoint=ckpt if ckpt.is_dir() else None, device=self.device)
        elif modality == "audio":
            from ..models.audio import AudioPipeline

            pipe = AudioPipeline(checkpoint=self._ckpt("audio.pt"), device=self.device)
        else:
            raise ValueError(f"unknown modality: {modality}")
        self._pipes[modality] = pipe
        return pipe

    # ---------------- fusion rules ----------------
    @staticmethod
    def fuse_video_audio(video_report: dict, audio_report: dict | None) -> dict:
        """Noisy-OR style fusion: a deepfake is flagged if either the visual
        or the audio stream is synthetic; confidence-weighted average keeps
        the score calibrated when the streams disagree."""
        pv = video_report.get("p_fake", 0.0)
        if audio_report is None or "p_spoof" not in audio_report:
            fused = pv
        else:
            pa = audio_report["p_spoof"]
            noisy_or = 1.0 - (1.0 - pv) * (1.0 - pa)
            fused = 0.5 * noisy_or + 0.5 * max(pv, pa)
        label = "deepfake" if fused >= 0.5 else "real"
        return {
            "label": label,
            "confidence": fused if label == "deepfake" else 1.0 - fused,
            "p_fake_fused": fused,
            "visual_stream": video_report,
            "audio_stream": audio_report,
        }

    # ---------------- main entry ----------------
    def analyze_file(self, path: str | Path) -> dict:
        path = Path(path)
        mtype = detect_media_type(path)

        if mtype is MediaType.IMAGE:
            report = self._get("image").analyze(path)
        elif mtype is MediaType.TEXT:
            report = self._get("text").analyze(path.read_text(errors="replace"))
        elif mtype is MediaType.AUDIO:
            report = self._get("audio").analyze(path)
        elif mtype is MediaType.VIDEO:
            vpipe = self._get("video")
            vreport = vpipe.analyze(path)
            wav = vpipe.extract_audio(path)
            areport = self._get("audio").analyze(wav) if wav is not None else None
            fused = self.fuse_video_audio(vreport, areport)
            fused["modality"] = "video"
            report = fused
        else:
            return {"modality": "unknown", "label": "unsupported",
                    "confidence": 0.0, "error": f"unsupported file type: {path.suffix}"}

        report["file"] = path.name
        report["detected_type"] = mtype.value
        return report

    def analyze_text(self, text: str) -> dict:
        return self._get("text").analyze(text)
