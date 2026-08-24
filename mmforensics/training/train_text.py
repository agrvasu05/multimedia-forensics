"""Fine-tune the supervised AI-text classifier (RoBERTa/DeBERTa branch).

    python -m mmforensics.training.train_text --data data/text --epochs 3
    python -m mmforensics.training.train_text --smoke

Real datasets (GPT-2 Output, HC3, M4, TuringBench, RAID) are converted to
data/text/<split>/{human,ai}.jsonl by scripts/download_datasets.py.
Saves a HF-format checkpoint dir to checkpoints/text/ which TextPipeline
loads as its supervised branch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import TextForensicsDataset, make_synthetic_text_dataset
from .metrics import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/text")
    ap.add_argument("--model", default="roberta-base",
                    help="roberta-base / microsoft/deberta-v3-base / distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="checkpoints/text")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        make_synthetic_text_dataset(args.data, n_per_class=24)
        args.epochs, args.batch, args.model = 1, 8, "distilroberta-base"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2).to(device)

    train_ds = TextForensicsDataset(args.data, "train")
    val_ds = TextForensicsDataset(args.data, "val")
    if len(train_ds) == 0:
        raise SystemExit(f"No jsonl data under {args.data}/train/ — run "
                         "scripts/download_datasets.py or pass --smoke.")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=args.max_len,
                  padding=True, return_tensors="pt")
        return enc, torch.tensor(labels)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=max(steps, 1), pct_start=0.1)
    best_auc = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for enc, y in train_dl:
            enc, y = enc.to(device), y.to(device)
            loss = model(**enc, labels=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss)
        # eval
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for enc, y in val_dl:
                p = torch.softmax(model(**enc.to(device)).logits, dim=-1)[:, 1]
                scores += p.cpu().tolist()
                labels += y.tolist()
        m = summarize(np.array(labels), np.array(scores), prefix="val_")
        print(f"[text] epoch {epoch}: loss={total / max(len(train_dl), 1):.4f} "
              f"val_auc={m['val_auc']:.4f} val_f1={m['val_f1']:.4f}")
        if not np.isnan(m["val_auc"]) and m["val_auc"] > best_auc:
            best_auc = m["val_auc"]
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
    print(f"done: best_val_auc={best_auc:.4f} checkpoint={args.out}")


if __name__ == "__main__":
    main()
