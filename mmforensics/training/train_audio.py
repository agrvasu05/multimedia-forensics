"""Train the audio anti-spoofing ensemble (LCNN on LFCC + RawNet2 on raw
waveform). Both are trained jointly on the same loader and saved into one
checkpoint that AudioPipeline loads.

    python -m mmforensics.training.train_audio --data data/audio --epochs 20
    python -m mmforensics.training.train_audio --smoke

Real datasets (ASVspoof 2019/2021, WaveFake, FakeAVCeleb, ADD, the custom
AI-music corpus) are arranged by scripts/download_datasets.py into
data/audio/<split>/{bonafide,spoof}/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.audio.lcnn import LCNN
from ..models.audio.rawnet2 import RawNet2
from .datasets import AudioSpoofDataset, make_synthetic_audio_dataset
from .metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/audio")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        make_synthetic_audio_dataset(args.data, n_per_class=10)
        args.epochs, args.batch = 3, 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = AudioSpoofDataset(args.data, "train", augment=not args.smoke)
    val_ds = AudioSpoofDataset(args.data, "val")
    if len(train_ds) == 0:
        raise SystemExit(f"No audio under {args.data}/train/ — run "
                         "scripts/download_datasets.py or pass --smoke.")
    print(f"train={len(train_ds)} val={len(val_ds)}")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    lcnn, rawnet = LCNN().to(device), RawNet2().to(device)
    params = list(lcnn.parameters()) + list(rawnet.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best_eer, out_path = 1.0, Path(args.out) / "audio.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        lcnn.train(); rawnet.train()
        total = 0.0
        for feat, wav, y in train_dl:
            feat, wav, y = feat.to(device), wav.to(device), y.to(device)
            loss = crit(lcnn(feat), y) + crit(rawnet(wav), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss)

        lcnn.eval(); rawnet.eval()
        scores, labels = [], []
        with torch.no_grad():
            for feat, wav, y in val_dl:
                p1 = torch.softmax(lcnn(feat.to(device)), dim=1)[:, 1]
                p2 = torch.softmax(rawnet(wav.to(device)), dim=1)[:, 1]
                scores += ((p1 + p2) / 2).cpu().tolist()
                labels += y.tolist()
        m = summarize(np.array(labels), np.array(scores), prefix="val_")
        print(f"[audio] epoch {epoch}: loss={total / max(len(train_dl), 1):.4f} "
              f"val_eer={m['val_eer']:.4f} val_auc={m['val_auc']:.4f}")
        if not np.isnan(m["val_eer"]) and m["val_eer"] <= best_eer:
            best_eer = m["val_eer"]
            torch.save({"lcnn": lcnn.state_dict(), "rawnet": rawnet.state_dict(),
                        "epoch": epoch, "val_eer": best_eer}, out_path)
    print(f"done: best_val_eer={best_eer:.4f} checkpoint={out_path}")


if __name__ == "__main__":
    main()
