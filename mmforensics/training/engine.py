"""Training engine: AdamW with differential LR, cosine warm-up,
progressive unfreezing, early stopping on val AUC, optional W&B/MLflow."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import summarize, summarize_multiclass


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    label_smoothing: float = 0.05
    early_stop_patience: int = 7
    freeze_backbone_epochs: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    out_dir: str = "checkpoints"
    run_name: str = "run"
    tracker: str | None = None
    extra: dict = field(default_factory=dict)


class Tracker:
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
        base = g.get("base_lr", cfg.lr)
        g["lr"] = base * scale


BACKBONE_ATTRS = ("rgb",)


def set_backbone_frozen(model: torch.nn.Module, frozen: bool):
    for attr in BACKBONE_ATTRS:
        bb = getattr(model, attr, None)
        if bb is not None:
            for p in bb.parameters():
                p.requires_grad = not frozen


def _is_backbone(param, model: torch.nn.Module) -> bool:
    for attr in BACKBONE_ATTRS:
        module = getattr(model, attr, None)
        if module is not None:
            for p in module.parameters():
                if p is param:
                    return True
    return False


def _build_param_groups(model, cfg: TrainConfig):
    backbone_params = []
    task_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if _is_backbone(p, model):
            backbone_params.append(p)
        else:
            task_params.append(p)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "base_lr": cfg.lr * cfg.backbone_lr_scale})
    if task_params:
        groups.append({"params": task_params, "base_lr": cfg.lr})
    return groups


def train_classifier(model: torch.nn.Module, train_loader: DataLoader,
                     val_loader: DataLoader, cfg: TrainConfig,
                     forward_fn=None, device: str | None = None,
                     class_weights: torch.Tensor | None = None) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

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
    opt = None

    for epoch in range(cfg.epochs):
        set_backbone_frozen(model, frozen=epoch < cfg.freeze_backbone_epochs)
        groups = _build_param_groups(model, cfg)
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        cosine_warmup_lr(opt, epoch, cfg)

        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        lrs = [f"{g['lr']:.2e}" for g in opt.param_groups]
        print(f"  frozen={frozen_params}/{total_params} LRs={lrs}")

        model.train()
        total_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]", leave=False)
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,
                                    enabled=cfg.amp and device.type == "cuda"):
                out = forward_fn(model, batch, device)
                logits, labels = out[0], out[1]
                loss = crit(logits, labels)
                if len(out) > 2 and out[2] is not None:
                    loss = loss + out[2]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  SKIP batch: loss={loss.item()}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.detach()) * labels.shape[0]
            n += labels.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[-1]['lr']:.2e}")

        val = evaluate_classifier(model, val_loader, forward_fn, device)
        metrics = {"epoch": epoch, "train_loss": total_loss / max(n, 1), **val}
        history.append(metrics)
        tracker.log(metrics, step=epoch)
        print(f"[{cfg.run_name}] epoch {epoch+1}/{cfg.epochs}: loss={metrics['train_loss']:.4f} "
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
    all_scores, all_labels = [], []
    pbar = tqdm(loader, desc="  [Val]  ", leave=False)
    for batch in pbar:
        out = forward_fn(model, batch, device)
        logits, y = out[0], out[1]
        if logits.ndim == 1 or logits.shape[-1] == 1:
            p = torch.sigmoid(logits.squeeze(-1))
            all_scores.append(p.float().cpu().numpy())
        else:
            probs = torch.softmax(logits, dim=-1)
            all_scores.append(probs.float().cpu().numpy())
        all_labels.append(y.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(int)

    if scores.ndim == 2 and scores.shape[1] > 2:
        return summarize_multiclass(labels, scores, prefix="val_")

    if scores.ndim == 2:
        scores = scores[:, -1]
    labels = (labels > 0).astype(int)
    return summarize(labels, scores, prefix="val_")
