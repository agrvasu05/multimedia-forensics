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
                 supervised_model: str = "roberta-base", reference_lm: str = REFERENCE_LM,
                 perturbation: str = "lexical"):
        """perturbation: 'lexical' (fast word deletion/swap) or 't5'
        (the paper's T5 mask-fill; downloads t5-small on first use and is
        ~10x slower but produces more fluent perturbations)."""
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.reference_lm_name = reference_lm
        self.perturbation = perturbation
        self._lm = None
        self._lm_tok = None
        self._t5 = None
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
    def _lexical_perturb(self, words: list[str], mask_frac: float,
                         rng: random.Random) -> str:
        w = words[:]
        n_edit = max(1, int(len(w) * mask_frac))
        for _ in range(n_edit):
            i = rng.randrange(len(w))
            if rng.random() < 0.5 and len(w) > 5:
                w.pop(i)                       # deletion
            else:
                j = rng.randrange(len(w))
                w[i], w[j] = w[j], w[i]        # swap
        return " ".join(w)

    def _t5_perturb(self, words: list[str], mask_frac: float,
                    rng: random.Random) -> str | None:
        """The paper's perturbation: mask random 2-word spans and let
        T5 fill them, yielding fluent semantically-close rewrites."""
        try:
            if self._t5 is None:
                from transformers import T5ForConditionalGeneration, T5TokenizerFast

                self._t5_tok = T5TokenizerFast.from_pretrained("t5-small", legacy=False)
                self._t5 = T5ForConditionalGeneration.from_pretrained(
                    "t5-small").eval().to(self.device)
            n_spans = min(max(1, int(len(words) * mask_frac / 2)), 20)
            starts = sorted(rng.sample(range(len(words) - 1), min(n_spans, len(words) - 1)))
            masked, sid, prev = [], 0, 0
            for s in starts:
                if s < prev:  # overlapping span
                    continue
                masked += words[prev:s] + [f"<extra_id_{sid}>"]
                sid += 1
                prev = s + 2
            masked += words[prev:]
            with torch.no_grad():
                ids = self._t5_tok(" ".join(masked), return_tensors="pt",
                                   truncation=True, max_length=512).input_ids.to(self.device)
                out = self._t5.generate(ids, max_new_tokens=6 * sid + 8,
                                        do_sample=True, top_p=0.95,
                                        num_return_sequences=1)
            fills = self._t5_tok.decode(out[0], skip_special_tokens=False)
            # parse "<extra_id_0> fill0 <extra_id_1> fill1 ..."
            result = " ".join(masked)
            for k in range(sid):
                seg = fills.split(f"<extra_id_{k}>")
                fill = seg[1].split("<extra_id_")[0].strip() if len(seg) > 1 else ""
                result = result.replace(f"<extra_id_{k}>", fill, 1)
            return re.sub(r"\s+", " ", result).strip() or None
        except Exception:
            return None  # caller falls back to lexical

    def detectgpt_curvature(self, text: str, n_perturb: int = 8, mask_frac: float = 0.15,
                            seed: int = 0) -> float | None:
        """Mean log-likelihood drop of perturbed variants vs. the original,
        normalized by the perturbation std (the DetectGPT statistic).
        Positive & large => likely AI. Perturbation mode set at init:
        'lexical' (fast) or 't5' (paper-faithful mask-fill)."""
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
            variant = None
            if self.perturbation == "t5":
                variant = self._t5_perturb(words, mask_frac, rng)
            if variant is None:
                variant = self._lexical_perturb(words, mask_frac, rng)
            plp, _ = self._token_logprobs(variant)
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
