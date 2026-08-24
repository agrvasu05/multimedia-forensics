"""Train the spatio-temporal deepfake detector.

    python -m mmforensics.training.train_video --data data/video --epochs 15
    python -m mmforensics.training.train_video --smoke

Real datasets (FaceForensics++, DFDC, Celeb-DF v2, DeeperForensics-1.0,
WildDeepfake) are converted to face-crop frame folders by
scripts/download_datasets.py: data/video/<split>/<class>/<clip>/*.jpg.
Curriculum per plan §8.3: pretrain on DFDC, fine-tune on Celeb-DF.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..models.video.deepfake_net import SpatioTemporalDeepfakeNet
from .datasets import VideoFramesDataset, make_synthetic_video_dataset
from .engine import TrainConfig, train_classifier


def forward_fn(model, batch, device):
    frames, y = batch[0].to(device), batch[1].to(device)
    video_logit, frame_logits, _ = model(frames)
    # auxiliary frame-level supervision with the video label
    aux = 0.3 * torch.nn.functional.binary_cross_entropy_with_logits(
        frame_logits, y.float().unsqueeze(1).expand_as(frame_logits))
    # engine expects class logits; expand binary logit to 2-class form
    logits2 = torch.stack([-video_logit, video_logit], dim=1)
    return logits2, y, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/video")
    ap.add_argument("--backbone", default="xception")
    ap.add_argument("--temporal", default="transformer", choices=["transformer", "bilstm"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--tracker", default=None, choices=[None, "wandb", "mlflow"])
    ap.add_argument("--resume", default=None, help="checkpoint to fine-tune from "
                    "(curriculum: DFDC -> Celeb-DF)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        make_synthetic_video_dataset(args.data, n_per_class=6, frames=args.frames)
        args.epochs, args.batch, args.backbone = 2, 2, "resnet18"

    train_ds = VideoFramesDataset(args.data, "train", num_frames=args.frames)
    val_ds = VideoFramesDataset(args.data, "val", num_frames=args.frames)
    if len(train_ds) == 0:
        raise SystemExit(f"No clips under {args.data}/train/ — run "
                         "scripts/download_datasets.py or pass --smoke.")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = SpatioTemporalDeepfakeNet(backbone=args.backbone, pretrained=True,
                                      temporal=args.temporal)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, out_dir=args.out,
                      run_name="video", tracker=args.tracker,
                      freeze_backbone_epochs=0 if args.smoke else 2)
    result = train_classifier(
        model,
        DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=args.batch, num_workers=0),
        cfg, forward_fn=forward_fn)
    print("done:", {k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
