"""Generic training engine implementing the plan's optimization setup:
AdamW, cosine decay with warm-up, label smoothing, mixed precision,
progressive unfreezing, early stopping on validation AUC, and optional
Weights & Biases / MLflow logging."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import summarize


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    label_smoothing: float = 0.05
    early_stop_patience: int = 7
    freeze_backbone_epochs: int = 2      # progressive unfreezing
    grad_clip: float = 1.0
    amp: bool = True
    out_dir: str = "checkpoints"
    run_name: str = "run"
    tracker: str | None = None           # "wandb" | "mlflow" | None
    extra: dict = field(default_factory=dict)


class Tracker:
    """Thin wrapper over wandb/mlflow; always mirrors metrics to a JSONL file."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.path = Path(cfg.out_dir) / f"{cfg.run_name}_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = None
        if cfg.tracker == "wandb":
            try:
                import wandb

                wandb.init(project="mmforensics", name=cfg.run_name, config=asdict(cfg))
                self.backend = wandb
            except Exception:
                pass
        elif cfg.tracker == "mlflow":
            try:
                import mlflow

                mlflow.start_run(run_name=cfg.run_name)
                mlflow.log_params({k: str(v) for k, v in asdict(cfg).items()})
                self.backend = mlflow
            except Exception:
                pass

    def log(self, metrics: dict, step: int):
        record = {"step": step, "time": time.time(),
                  **{k: (None if isinstance(v, float) and math.isnan(v) else v)
                     for k, v in metrics.items()}}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        try:
            if self.cfg.tracker == "wandb" and self.backend:
                self.backend.log(metrics, step=step)
            elif self.cfg.tracker == "mlflow" and self.backend:
                self.backend.log_metrics(
                    {k: v for k, v in metrics.items()
                     if isinstance(v, (int, float)) and not math.isnan(v)}, step=step)
        except Exception:
            pass


def cosine_warmup_lr(optimizer, epoch: float, cfg: TrainConfig):
    if epoch < cfg.warmup_epochs:
        scale = (epoch + 1e-8) / cfg.warmup_epochs
    else:
        t = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
        scale = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    for g in optimizer.param_groups:
        g["lr"] = cfg.lr * scale


def set_backbone_frozen(model: torch.nn.Module, frozen: bool, backbone_attrs=("rgb", "spatial", "encoder")):
    for attr in backbone_attrs:
        bb = getattr(model, attr, None)
        if bb is not None:
            for p in bb.parameters():
                p.requires_grad = not frozen


def train_classifier(model: torch.nn.Module, train_loader: DataLoader,
                     val_loader: DataLoader, cfg: TrainConfig,
                     forward_fn=None, device: str | None = None,
                     class_weights: torch.Tensor | None = None) -> dict:
    """Train any binary/multiclass classifier.

    forward_fn(model, batch, device) -> (logits, labels[, aux_loss]);
    defaults to model(x) for (x, y) batches. Saves the best-val-AUC
    checkpoint to {out_dir}/{run_name}.pt and returns the metric history.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing,
                                     weight=class_weights.to(device) if class_weights is not None else None)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    tracker = Tracker(cfg)

    if forward_fn is None:
        def forward_fn(m, batch, dev):
            x, y = batch
            return m(x.to(dev)), y.to(dev)

    best_auc, best_epoch, history = -1.0, -1, []
    out_path = Path(cfg.out_dir) / f"{cfg.run_name}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        set_backbone_frozen(model, frozen=epoch < cfg.freeze_backbone_epochs)
        # rebuild optimizer param groups after (un)freezing
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
        cosine_warmup_lr(opt, epoch, cfg)

        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=cfg.amp and device.type == "cuda"):
                out = forward_fn(model, batch, device)
                logits, labels = out[0], out[1]
                loss = crit(logits, labels)
                if len(out) > 2 and out[2] is not None:
                    loss = loss + out[2]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.detach()) * labels.shape[0]
            n += labels.shape[0]

        val = evaluate_classifier(model, val_loader, forward_fn, device)
        metrics = {"epoch": epoch, "train_loss": total_loss / max(n, 1), **val}
        history.append(metrics)
        tracker.log(metrics, step=epoch)
        print(f"[{cfg.run_name}] epoch {epoch}: loss={metrics['train_loss']:.4f} "
              f"val_auc={val.get('val_auc', float('nan')):.4f}")

        auc = val.get("val_auc", float("nan"))
        if not math.isnan(auc) and auc > best_auc:
            best_auc, best_epoch = auc, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_auc": auc, "config": asdict(cfg)}, out_path)
        if epoch - best_epoch >= cfg.early_stop_patience:
            print(f"[{cfg.run_name}] early stop at epoch {epoch} (best={best_auc:.4f})")
            break

    return {"best_val_auc": best_auc, "best_epoch": best_epoch,
            "checkpoint": str(out_path), "history": history}


@torch.no_grad()
def evaluate_classifier(model, loader, forward_fn, device) -> dict:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        out = forward_fn(model, batch, device)
        logits, y = out[0], out[1]
        if logits.ndim == 1 or logits.shape[-1] == 1:
            p = torch.sigmoid(logits.squeeze(-1))
        else:
            p = torch.softmax(logits, dim=-1)[:, -1]  # P(last class = fake)
        scores.append(p.float().cpu().numpy())
        labels.append(y.cpu().numpy())
    scores = np.concatenate(scores)
    labels = (np.concatenate(labels) > 0).astype(int)
    return summarize(labels, scores, prefix="val_")
