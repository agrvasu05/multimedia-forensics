"""Train the dual-branch image forensics model (classification + localization).

    python -m mmforensics.training.train_image --data data/image --epochs 20
    python -m mmforensics.training.train_image --smoke        # synthetic end-to-end test

Real datasets (CASIA v1/v2, Columbia, COVERAGE, NIST MFC, CoMoFoD, GenImage,
CIFAKE, DFFD) are arranged by scripts/download_datasets.py into
data/image/<split>/<class>/.
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


def forward_fn(model, batch, device):
    rgb, art, dct, y, mask = (b.to(device) for b in batch)
    logits, mask_logits = model(rgb, art, dct)
    aux = 0.5 * dice_loss(mask_logits, mask)  # plan §8.4: BCE + Dice for masks
    return logits, y, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/image")
    ap.add_argument("--backbone", default="efficientnet_b0",
                    help="efficientnet_b4 per plan; b0 default for modest GPUs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--tracker", default=None, choices=[None, "wandb", "mlflow"])
    ap.add_argument("--smoke", action="store_true",
                    help="generate a tiny synthetic dataset and run 2 epochs")
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

    model = DualBranchImageForensics(backbone=args.backbone, pretrained=True)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, out_dir=args.out,
                      run_name="image", tracker=args.tracker,
                      freeze_backbone_epochs=0 if args.smoke else 2)
    result = train_classifier(
        model,
        DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=args.batch, num_workers=0),
        cfg, forward_fn=forward_fn)
    print("done:", {k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
