"""Explainability layer: Grad-CAM heatmaps, localization-mask overlays, and
natural-language explanations of why content was flagged."""
from __future__ import annotations

import numpy as np


def gradcam_image(model, rgb_t, artifacts_t, dct_t, target_class: int | None = None) -> np.ndarray:
    """Grad-CAM over the RGB backbone's final feature maps of the
    DualBranchImageForensics model. Returns an HxW heatmap in [0, 1]."""
    import torch

    model.eval()
    feats = {}

    def hook(_m, _i, o):
        feats["act"] = o
        o.register_hook(lambda g: feats.__setitem__("grad", g))

    # last conv-bearing module of the timm backbone
    target_layer = [m for m in model.rgb.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    h = target_layer.register_forward_hook(hook)
    try:
        logits, _ = model(rgb_t, artifacts_t, dct_t)
        cls = int(logits.argmax(1)) if target_class is None else target_class
        model.zero_grad()
        logits[0, cls].backward()
        act, grad = feats["act"][0], feats["grad"][0]
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act).sum(0)).detach().cpu().numpy()
    finally:
        h.remove()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    import cv2

    return cv2.resize(cam, (rgb_t.shape[-1], rgb_t.shape[-2]))


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap/mask onto an RGB [0,1] image as a red overlay.
    Returns uint8 HxWx3."""
    import cv2

    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha * mask[..., None]) * rgb + (alpha * mask[..., None]) * heat
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def explain_report(report: dict) -> str:
    """Short natural-language explanation of the flagged artifacts, built
    from whichever signals the pipeline produced."""
    modality = report.get("modality", "unknown")
    label = report.get("label", "unknown")
    conf = report.get("confidence", 0.0)
    parts = [f"The {modality} was classified as '{label}' with {conf:.0%} confidence."]

    if modality == "image":
        mask = report.get("localization_mask")
        if mask is not None and np.asarray(mask).max() > 0.5:
            frac = float((np.asarray(mask) > 0.5).mean())
            parts.append(f"The localization head highlights ~{frac:.0%} of the pixels "
                         "as likely manipulated (see overlay).")
        if report.get("ela_mean", 0) > 0.08:
            parts.append("Error-Level Analysis shows uneven recompression residuals, "
                         "typical of spliced or locally edited regions.")
    elif modality == "video":
        vs = report.get("visual_stream", report)
        fs = vs.get("frame_scores") or []
        if fs:
            hi = sum(s > 0.5 for s in fs)
            parts.append(f"{hi}/{len(fs)} sampled face frames scored as manipulated; "
                         "temporal attention concentrated on the most inconsistent frames.")
        ls = report.get("lipsync")
        if ls and ls.get("mismatch"):
            parts.append(f"Lip-sync check: mouth motion barely correlates with the "
                         f"audio envelope (r={ls['sync_corr']:.2f}), suggesting the "
                         "voice does not belong to the visible speech.")
        au = report.get("audio_stream")
        if au and au.get("p_spoof", 0) > 0.5:
            parts.append("The audio track independently scored as synthetic speech, "
                         "reinforcing the deepfake verdict.")
    elif modality == "text":
        b = report.get("branches", {})
        g = b.get("gltr") or {}
        flagged = label == "ai_generated"
        if flagged and g.get("frac_rank_top10", 0) > 0.7:
            parts.append(f"{g['frac_rank_top10']:.0%} of tokens fall in the reference "
                         "LM's top-10 predictions — far above typical human writing.")
        if g.get("perplexity") is not None:
            parts.append(f"Perplexity under the reference LM is {g['perplexity']:.1f}.")
        if flagged and (c := b.get("detectgpt_curvature")) is not None and c > 0.5:
            parts.append("DetectGPT curvature is high: perturbing the text sharply "
                         "reduces its likelihood, a signature of machine generation.")
        if flagged and (bu := b.get("burstiness")) is not None and bu < -0.4:
            parts.append("Sentence-length rhythm is unusually uniform (low burstiness).")
        if report.get("span_scores"):
            worst = max(report["span_scores"], key=lambda s: s["p_ai"])
            parts.append(f"The most AI-like span is sentences "
                         f"{worst['sentence_start']}–{worst['sentence_end']} "
                         f"(p_ai={worst['p_ai']:.2f}).")
    elif modality == "audio":
        b = report.get("branches", {})
        strong = {k: v for k, v in b.items() if v is not None and v > 0.5}
        if strong:
            parts.append("Branches flagging synthesis: "
                         + ", ".join(f"{k} ({v:.2f})" for k, v in strong.items()) + ".")

    if w := report.get("warning"):
        parts.append(f"NOTE: {w}")
    return " ".join(parts)
