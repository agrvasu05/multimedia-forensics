"""AI-generated-text detector with four complementary branches.

1. Supervised: fine-tuned RoBERTa/DeBERTa sequence classifier (needs training).
2. Zero-shot: DetectGPT-style curvature — perturb the text and measure the
   log-likelihood drop under a reference LM; AI text sits near a sharp local
   maximum of the LM's likelihood surface, human text does not.
3. Statistical: GLTR-style token-rank histogram + perplexity under the
   reference LM (GPT-2), which works untrained.
4. Stylometric: burstiness/function-word features (see stylometry.py).

Branches 2–4 are training-free, so the pipeline produces meaningful scores
out of the box; the supervised branch joins the ensemble once a checkpoint
trained by training/train_text.py is supplied.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch

from ...preprocessing.text_ops import clean_text, split_sentences, sliding_windows
from .stylometry import stylometric_features

CLASS_NAMES = ["human", "ai_generated"]
REFERENCE_LM = "gpt2"  # small, fast; swap for a larger LM for better curvature


class TextPipeline:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None,
                 supervised_model: str = "roberta-base", reference_lm: str = REFERENCE_LM):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reference_lm_name = reference_lm
        self._lm = None
        self._lm_tok = None
        self.supervised = None
        self.trained = False
        if checkpoint is not None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.sup_tok = AutoTokenizer.from_pretrained(checkpoint)
            self.supervised = AutoModelForSequenceClassification.from_pretrained(
                checkpoint).eval().to(self.device)
            self.trained = True

    # ---------------- reference LM (lazy) ----------------
    def _load_lm(self):
        if self._lm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._lm_tok = AutoTokenizer.from_pretrained(self.reference_lm_name)
            self._lm = AutoModelForCausalLM.from_pretrained(
                self.reference_lm_name).eval().to(self.device)
        return self._lm, self._lm_tok

    @torch.no_grad()
    def _token_logprobs(self, text: str, max_tokens: int = 512):
        """Per-token log-probs and ranks of the actual tokens under the LM."""
        lm, tok = self._load_lm()
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=max_tokens).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return None, None
        logits = lm(ids).logits[0, :-1]                     # predict token t+1
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        targets = ids[0, 1:]
        tok_lp = logprobs.gather(1, targets[:, None]).squeeze(1)
        ranks = (logprobs > tok_lp[:, None]).sum(dim=1) + 1  # 1 = most likely
        return tok_lp.cpu().numpy(), ranks.cpu().numpy()

    # ---------------- branch 2: DetectGPT curvature ----------------
    def detectgpt_curvature(self, text: str, n_perturb: int = 8, mask_frac: float = 0.15,
                            seed: int = 0) -> float | None:
        """Mean log-likelihood drop of word-dropout perturbations vs. the
        original, normalized by the perturbation std (the DetectGPT statistic).
        Positive & large => likely AI. Uses word deletion/swap as a cheap
        stand-in for the T5 mask-fill used in the paper."""
        lp, _ = self._token_logprobs(text)
        if lp is None:
            return None
        orig = float(lp.mean())
        rng = random.Random(seed)
        words = text.split()
        if len(words) < 10:
            return None
        perturbed_scores = []
        for _ in range(n_perturb):
            w = words[:]
            n_edit = max(1, int(len(w) * mask_frac))
            for _ in range(n_edit):
                op = rng.random()
                i = rng.randrange(len(w))
                if op < 0.5 and len(w) > 5:
                    w.pop(i)                       # deletion
                else:
                    j = rng.randrange(len(w))
                    w[i], w[j] = w[j], w[i]        # swap
            plp, _ = self._token_logprobs(" ".join(w))
            if plp is not None:
                perturbed_scores.append(float(plp.mean()))
        if not perturbed_scores:
            return None
        mu, sigma = float(np.mean(perturbed_scores)), float(np.std(perturbed_scores))
        return (orig - mu) / (sigma + 1e-6)

    # ---------------- branch 3: GLTR statistics ----------------
    def gltr_features(self, text: str) -> dict | None:
        lp, ranks = self._token_logprobs(text)
        if lp is None:
            return None
        return {
            "perplexity": float(np.exp(-lp.mean())),
            "mean_logprob": float(lp.mean()),
            "frac_rank_top10": float((ranks <= 10).mean()),
            "frac_rank_top100": float((ranks <= 100).mean()),
            "frac_rank_1000plus": float((ranks > 1000).mean()),
            "logprob_std": float(lp.std()),  # token-level burstiness
        }

    # ---------------- branch 1: supervised ----------------
    @torch.no_grad()
    def supervised_score(self, text: str) -> float | None:
        if self.supervised is None:
            return None
        enc = self.sup_tok(text, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        probs = torch.softmax(self.supervised(**enc).logits, dim=-1)[0]
        return float(probs[1])  # P(ai_generated)

    # ---------------- zero-shot score combination ----------------
    @staticmethod
    def _zero_shot_p_ai(gltr: dict, curvature: float | None, stylo: np.ndarray) -> float:
        """Map training-free signals to a calibrated-ish P(AI) via a hand-set
        logistic. Thresholds follow published GLTR/DetectGPT observations
        (AI text: high top-10 fraction, low perplexity, low burstiness)."""
        z = 0.0
        z += 6.0 * (gltr["frac_rank_top10"] - 0.62)      # AI ~0.75+, human ~0.45
        z += 1.5 * (2.2 - np.log(max(gltr["perplexity"], 1.01)) / np.log(10) * 2.0)
        z += 1.2 * (1.6 - gltr["logprob_std"])            # uniform likelihood
        if curvature is not None:
            z += 1.0 * np.tanh(curvature - 0.4)
        burstiness = float(stylo[2])
        z += 1.0 * (-0.35 - burstiness)                   # regular rhythm => AI
        return float(1.0 / (1.0 + np.exp(-z)))

    # ---------------- main entry ----------------
    def analyze(self, text: str, localize: bool = True) -> dict:
        text = clean_text(text)
        sentences = split_sentences(text)
        stylo = stylometric_features(text, sentences)
        gltr = self.gltr_features(text)
        curvature = self.detectgpt_curvature(text) if len(text.split()) >= 30 else None
        sup = self.supervised_score(text)

        if gltr is None:
            return {"modality": "text", "label": "unknown", "confidence": 0.0,
                    "error": "text too short to analyze"}

        zs = self._zero_shot_p_ai(gltr, curvature, stylo)
        p_ai = 0.6 * sup + 0.4 * zs if sup is not None else zs

        label = CLASS_NAMES[int(p_ai >= 0.5)]
        report = {
            "modality": "text",
            "label": label,
            "confidence": p_ai if label == "ai_generated" else 1.0 - p_ai,
            "p_ai": p_ai,
            "branches": {
                "supervised": sup,
                "zero_shot": zs,
                "detectgpt_curvature": curvature,
                "gltr": gltr,
                "burstiness": float(stylo[2]),
            },
        }
        if localize and len(sentences) >= 4:
            report["span_scores"] = self._localize_spans(sentences)
        if not self.trained:
            report["note"] = ("Supervised branch inactive (no checkpoint); verdict uses "
                              "zero-shot GLTR/DetectGPT/stylometry signals only.")
        return report

    def _localize_spans(self, sentences: list[str], window: int = 3) -> list[dict]:
        """Sentence-window sliding classifier: flags which spans look AI-written."""
        spans = []
        for start, end, chunk in sliding_windows(sentences, window=window):
            g = self.gltr_features(chunk)
            if g is None:
                continue
            s = stylometric_features(chunk)
            p = self._zero_shot_p_ai(g, None, s)
            sup = self.supervised_score(chunk)
            if sup is not None:
                p = 0.6 * sup + 0.4 * p
            spans.append({"sentence_start": start, "sentence_end": end,
                          "text_preview": re.sub(r"\s+", " ", chunk)[:120],
                          "p_ai": round(float(p), 4)})
        return spans
