"""Evaluation metrics per the project plan: AUC-ROC, F1, EER, t-DCF-lite,
TPR@low-FPR, and IoU/pixel-F1 for localization masks."""
from __future__ import annotations

import numpy as np


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def f1(labels: np.ndarray, scores: np.ndarray, thresh: float = 0.5) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(labels, scores >= thresh, zero_division=0))


def eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate: operating point where FPR == FNR (ASVspoof standard)."""
    from sklearn.metrics import roc_curve

    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """TPR at a fixed low FPR — controls false accusations for text detection."""
    from sklearn.metrics import roc_curve

    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def mask_iou(pred: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> float:
    """IoU between predicted and ground-truth tampering masks."""
    p, t = pred >= thresh, target >= 0.5
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)


def pixel_f1(pred: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> float:
    p, t = (pred >= thresh).ravel(), (target >= 0.5).ravel()
    tp = np.logical_and(p, t).sum()
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def summarize(labels, scores, prefix: str = "") -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} NaN scores replaced with 0.5")
        scores = scores.copy()
        scores[nan_mask] = 0.5
    return {
        f"{prefix}auc": auc_roc(labels, scores),
        f"{prefix}f1": f1(labels, scores),
        f"{prefix}eer": eer(labels, scores),
        f"{prefix}tpr@1%fpr": tpr_at_fpr(labels, scores),
        f"{prefix}acc": float(((scores >= 0.5) == labels.astype(bool)).mean()),
    }


def summarize_multiclass(labels, probs, prefix: str = "", class_names=None) -> dict:
    labels, probs = np.asarray(labels), np.asarray(probs)
    if class_names is None:
        class_names = ["real", "tampered", "ai_generated"]
    n_classes = probs.shape[1]

    nan_mask = np.isnan(probs).any(axis=1)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} NaN predictions replaced with uniform")
        probs = probs.copy()
        probs[nan_mask] = 1.0 / n_classes

    preds = probs.argmax(axis=1)
    result = {f"{prefix}acc": float((preds == labels).mean())}

    for i, name in enumerate(class_names[:n_classes]):
        binary_labels = (labels == i).astype(int)
        class_scores = probs[:, i]
        result[f"{prefix}auc_{name}"] = auc_roc(binary_labels, class_scores)
        result[f"{prefix}f1_{name}"] = f1(binary_labels, class_scores)

    result[f"{prefix}auc_macro"] = float(np.nanmean([
        result[f"{prefix}auc_{name}"] for name in class_names[:n_classes]
    ]))
    return result
