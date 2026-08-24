"""Stylometric features that survive paraphrase attacks: burstiness,
sentence-length variance, function-word frequency, punctuation patterns.
Human writing is bursty (uneven perplexity/sentence rhythm); AI text tends
to be uniform even after paraphrasing."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

FUNCTION_WORDS = (
    "the a an and or but if while of to in on at by for with about against "
    "between into through during before after above below from up down out "
    "is are was were be been being have has had do does did will would shall "
    "should may might must can could not no nor so than too very just that "
    "this these those he she it they we you i his her its their our your my"
).split()

PUNCT = list(".,;:!?—-'\"()")

FEATURE_NAMES = (
    ["avg_sentence_len", "sentence_len_std", "burstiness",
     "avg_word_len", "type_token_ratio", "hapax_ratio"]
    + [f"fw_{w}" for w in FUNCTION_WORDS[:20]]
    + [f"punct_{i}" for i in range(len(PUNCT))]
)


def stylometric_features(text: str, sentences: list[str] | None = None) -> np.ndarray:
    """Return a fixed-length float32 stylometric feature vector."""
    from ...preprocessing.text_ops import split_sentences

    if sentences is None:
        sentences = split_sentences(text)
    words = re.findall(r"[A-Za-z']+", text.lower())
    n_words = max(len(words), 1)

    sent_lens = np.array([len(re.findall(r"\S+", s)) for s in sentences] or [0], dtype=np.float32)
    mean_len = float(sent_lens.mean())
    std_len = float(sent_lens.std())
    # burstiness B = (sigma - mu) / (sigma + mu); -1 = perfectly regular (AI-like)
    burstiness = (std_len - mean_len) / (std_len + mean_len) if (std_len + mean_len) > 0 else 0.0

    counts = Counter(words)
    feats = [
        mean_len,
        std_len,
        burstiness,
        float(np.mean([len(w) for w in words])) if words else 0.0,
        len(counts) / n_words,                                  # type-token ratio
        sum(1 for c in counts.values() if c == 1) / n_words,    # hapax legomena
    ]
    feats += [counts.get(w, 0) / n_words for w in FUNCTION_WORDS[:20]]
    n_chars = max(len(text), 1)
    feats += [text.count(p) / n_chars for p in PUNCT]
    return np.asarray(feats, dtype=np.float32)
