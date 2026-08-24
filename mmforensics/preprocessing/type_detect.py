"""Input-file type detection for the ingestion/routing stage."""
from __future__ import annotations

import mimetypes
from enum import Enum
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
TEXT_EXT = {".txt", ".md", ".rst", ".text"}


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    UNKNOWN = "unknown"


def detect_media_type(path: str | Path) -> MediaType:
    """Detect media type from magic bytes when possible, extension as fallback."""
    path = Path(path)
    try:
        import filetype  # magic-byte sniffing

        kind = filetype.guess(str(path))
        if kind is not None:
            mime = kind.mime
            if mime.startswith("image/"):
                return MediaType.IMAGE
            if mime.startswith("video/"):
                return MediaType.VIDEO
            if mime.startswith("audio/"):
                return MediaType.AUDIO
    except Exception:
        pass

    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return MediaType.IMAGE
    if ext in VIDEO_EXT:
        return MediaType.VIDEO
    if ext in AUDIO_EXT:
        return MediaType.AUDIO
    if ext in TEXT_EXT:
        return MediaType.TEXT

    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        for prefix, mt in [("image/", MediaType.IMAGE), ("video/", MediaType.VIDEO),
                           ("audio/", MediaType.AUDIO), ("text/", MediaType.TEXT)]:
            if mime.startswith(prefix):
                return mt

    # Last resort: try decoding a small chunk as UTF-8 text
    try:
        path.read_bytes()[:4096].decode("utf-8")
        return MediaType.TEXT
    except Exception:
        return MediaType.UNKNOWN
