"""Train the dual-branch image forensics model (classification + localization).

Experiment A — classification only (stable baseline):
    python -m mmforensics.training.train_image --data data/image --epochs 20

Experiment B — enable localization:
    python -m mmforensics.training.train_image --data data/image --epochs 20 --dice

Smoke test:
    python -m mmforensics.training.train_image --smoke
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..models.image.dual_branch import DualBranchImageForensics
from .datasets import ImageForensicsDataset, make_synthetic_image_dataset
from .engine import TrainConfig, train_classifier


def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(pred_logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def make_forward_fn(use_dice: bool):
    def forward_fn(model, batch, device):
        rgb, art, dct, y, mask = (b.to(device) for b in batch)
        logits, mask_logits = model(rgb, art, dct)
        aux = 0.5 * dice_loss(mask_logits, mask) if use_dice else None
        return logits, y, aux
    return forward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/image")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--tracker", default=None, choices=[None, "wandb", "mlflow"])
    ap.add_argument("--dice", action="store_true",
                    help="Enable Dice localization loss (Experiment B)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        make_synthetic_image_dataset(args.data, n_per_class=12, size=args.size)
        args.epochs, args.batch = 2, 4

    train_ds = ImageForensicsDataset(args.data, "train", size=args.size, augment=not args.smoke)
    val_ds = ImageForensicsDataset(args.data, "val", size=args.size)
    if len(train_ds) == 0:
        raise SystemExit(f"No training images under {args.data}/train/ — run "
                         "scripts/download_datasets.py or pass --smoke.")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = DualBranchImageForensics(backbone=args.backbone, pretrained=True)
    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, out_dir=args.out,
        run_name="image_cls" if not args.dice else "image_cls+dice",
        tracker=args.tracker,
        freeze_backbone_epochs=0 if args.smoke else 2,
        amp=False,
    )

    forward_fn = make_forward_fn(use_dice=args.dice)

    result = train_classifier(
        model,
        DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=args.batch, num_workers=0),
        cfg, forward_fn=forward_fn)
    print("done:", {k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
