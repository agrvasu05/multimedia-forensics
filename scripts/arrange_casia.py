"""Arrange CASIA v2 dataset into training format.

Expected output:
    data/image/train/real/*.jpg
    data/image/train/tampered/*.jpg
    data/image/train/masks/*.png
    data/image/val/real/*.jpg
    data/image/val/tampered/*.jpg
    data/image/val/masks/*.png
"""
import os
import shutil
import random
from pathlib import Path
from PIL import Image

SRC = Path("data/temp/casia/CASIA2")
OUT = Path("data/image")
TRAIN_RATIO = 0.9
SEED = 42

def ensure_dirs():
    for split in ["train", "val"]:
        for cls in ["real", "tampered"]:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)
        (OUT / split / "masks").mkdir(parents=True, exist_ok=True)

def convert_to_jpg(src_path, dst_path):
    """Convert any image format to JPEG."""
    try:
        img = Image.open(src_path).convert("RGB")
        img.save(dst_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  Failed to convert {src_path.name}: {e}")
        return False

def main():
    random.seed(SEED)
    ensure_dirs()

    # --- Real images (Au) ---
    au_files = [f for f in (SRC / "Au").iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")]
    print(f"Real images: {len(au_files)}")
    random.shuffle(au_files)
    n_train = int(len(au_files) * TRAIN_RATIO)
    train_real, val_real = au_files[:n_train], au_files[n_train:]

    for i, f in enumerate(train_real):
        dst = OUT / "train" / "real" / f"{i:05d}.jpg"
        if not dst.exists():
            convert_to_jpg(f, dst)
    print(f"  Train real: {len(train_real)}")

    for i, f in enumerate(val_real):
        dst = OUT / "val" / "real" / f"{i:05d}.jpg"
        if not dst.exists():
            convert_to_jpg(f, dst)
    print(f"  Val real: {len(val_real)}")

    # --- Tampered images (Tp) + masks ---
    tp_files = sorted([f for f in (SRC / "Tp").iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")])
    gt_files = sorted([f for f in (SRC / "CASIA 2 Groundtruth").iterdir()
                       if f.suffix.lower() == ".png"])

    # Build mask lookup: base_name -> mask_path
    mask_map = {}
    for gt in gt_files:
        # Masks are named like: Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png
        base = gt.name.replace("_gt.png", "").replace("_gt.jpg", "")
        mask_map[base] = gt

    print(f"Tampered images: {len(tp_files)}")
    print(f"Ground truth masks: {len(gt_files)}")

    # Match tampered images with masks
    matched = []
    unmatched = []
    for tp in tp_files:
        base = tp.stem  # e.g., Tp_D_CND_M_N_ani00018_sec00096_00138
        if base in mask_map:
            matched.append((tp, mask_map[base]))
        else:
            unmatched.append(tp)

    print(f"Matched pairs: {len(matched)}")
    print(f"Unmatched tampered (no mask): {len(unmatched)}")

    # Split matched pairs into train/val
    random.shuffle(matched)
    n_train = int(len(matched) * TRAIN_RATIO)
    train_pairs, val_pairs = matched[:n_train], matched[n_train:]

    # Also include unmatched tampered images (no mask) for classification only
    random.shuffle(unmatched)
    n_train_um = int(len(unmatched) * TRAIN_RATIO)
    train_unmatched, val_unmatched = unmatched[:n_train_um], unmatched[n_train_um:]

    # Copy train tampered + masks
    for i, (tp, mask) in enumerate(train_pairs):
        tp_dst = OUT / "train" / "tampered" / f"{i:05d}.jpg"
        mask_dst = OUT / "train" / "masks" / f"{i:05d}.png"
        if not tp_dst.exists():
            convert_to_jpg(tp, tp_dst)
        if not mask_dst.exists():
            shutil.copy2(mask, mask_dst)

    # Copy unmatched tampered (no mask)
    for i, tp in enumerate(train_unmatched):
        tp_dst = OUT / "train" / "tampered" / f"um_{i:05d}.jpg"
        if not tp_dst.exists():
            convert_to_jpg(tp, tp_dst)

    print(f"  Train tampered: {len(train_pairs) + len(train_unmatched)} (with mask: {len(train_pairs)})")

    # Copy val tampered + masks
    for i, (tp, mask) in enumerate(val_pairs):
        tp_dst = OUT / "val" / "tampered" / f"{i:05d}.jpg"
        mask_dst = OUT / "val" / "masks" / f"{i:05d}.png"
        if not tp_dst.exists():
            convert_to_jpg(tp, tp_dst)
        if not mask_dst.exists():
            shutil.copy2(mask, mask_dst)

    # Copy unmatched tampered (no mask)
    for i, tp in enumerate(val_unmatched):
        tp_dst = OUT / "val" / "tampered" / f"um_{i:05d}.jpg"
        if not tp_dst.exists():
            convert_to_jpg(tp, tp_dst)

    print(f"  Val tampered: {len(val_pairs) + len(val_unmatched)} (with mask: {len(val_pairs)})")

    # Summary
    print("\n=== Dataset Summary ===")
    for split in ["train", "val"]:
        for cls in ["real", "tampered"]:
            n = len(list((OUT / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls}: {n}")
        n_masks = len(list((OUT / split / "masks").glob("*.png")))
        print(f"  {split}/masks: {n_masks}")

if __name__ == "__main__":
    main()
