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
    return {
        f"{prefix}auc": auc_roc(labels, scores),
        f"{prefix}f1": f1(labels, scores),
        f"{prefix}eer": eer(labels, scores),
        f"{prefix}tpr@1%fpr": tpr_at_fpr(labels, scores),
        f"{prefix}acc": float(((scores >= 0.5) == labels.astype(bool)).mean()),
    }
