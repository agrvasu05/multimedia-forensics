"""Text cleaning and sentence segmentation for the text-forensics pipeline."""
from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"[ \t]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'“(])")


def clean_text(text: str) -> str:
    """Normalize unicode, strip control chars, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (regex-based; swap for spaCy if installed)."""
    try:
        import spacy  # optional, better segmentation

        if not hasattr(split_sentences, "_nlp"):
            split_sentences._nlp = spacy.blank("en")
            split_sentences._nlp.add_pipe("sentencizer")
        doc = split_sentences._nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sents:
            return sents
    except Exception:
        pass
    parts = _SENT_RE.split(text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def sliding_windows(sentences: list[str], window: int = 3, stride: int = 1) -> list[tuple[int, int, str]]:
    """Yield (start_idx, end_idx, joined_text) sentence windows for span-level
    AI-text localization."""
    if not sentences:
        return []
    out = []
    for start in range(0, max(len(sentences) - window + 1, 1), stride):
        end = min(start + window, len(sentences))
        out.append((start, end, " ".join(sentences[start:end])))
        if end == len(sentences):
            break
    return out
