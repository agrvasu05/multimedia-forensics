#!/usr/bin/env python3
"""Dataset acquisition helper.

Most forensic benchmarks are gated (license agreement, Kaggle account, or
request form), so they cannot be auto-downloaded. This script:
  1. prints the registry of every dataset in the plan with its access URL,
  2. auto-downloads the ones with direct public links,
  3. arranges anything you've downloaded manually into the folder layout the
     training scripts expect (data/<modality>/<split>/<class>/...).

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --fetch cifake
    python scripts/download_datasets.py --arrange image /path/to/CASIA2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REGISTRY = {
    # modality: [(name, access, url)]
    "image": [
        ("CASIA v1/v2", "public mirrors / Kaggle", "https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset"),
        ("Columbia Splicing", "request form", "https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/AuthSplicedDataSet.htm"),
        ("COVERAGE", "GitHub", "https://github.com/wenbihan/coverage"),
        ("NIST MFC / Nimble16", "NIST agreement", "https://mfc.nist.gov/"),
        ("CoMoFoD", "public", "https://www.vcl.fer.hr/comofod/"),
        ("GenImage", "GitHub release", "https://github.com/GenImage-Dataset/GenImage"),
        ("CIFAKE", "Kaggle (kaggle CLI)", "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images"),
        ("DFFD", "request form", "http://cvlab.cse.msu.edu/dffd-dataset.html"),
    ],
    "video": [
        ("FaceForensics++", "request form (email)", "https://github.com/ondyari/FaceForensics"),
        ("DFDC", "Kaggle competition", "https://www.kaggle.com/c/deepfake-detection-challenge/data"),
        ("Celeb-DF v2", "request form", "https://github.com/yuezunli/celeb-deepfakeforensics"),
        ("DeeperForensics-1.0", "request form", "https://github.com/EndlessSora/DeeperForensics-1.0"),
        ("WildDeepfake", "request form", "https://github.com/OpenTAI/wild-deepfake"),
    ],
    "text": [
        ("GPT-2 Output Dataset", "direct download", "https://github.com/openai/gpt-2-output-dataset"),
        ("HC3", "HuggingFace", "https://huggingface.co/datasets/Hello-SimpleAI/HC3"),
        ("M4", "GitHub", "https://github.com/mbzuai-nlp/M4"),
        ("TuringBench", "HuggingFace", "https://huggingface.co/datasets/turingbench/TuringBench"),
        ("RAID", "site", "https://raid-bench.xyz/"),
    ],
    "audio": [
        ("ASVspoof 2019", "direct (Edinburgh DataShare)", "https://datashare.ed.ac.uk/handle/10283/3336"),
        ("ASVspoof 2021", "zenodo", "https://www.asvspoof.org/index2021.html"),
        ("WaveFake", "zenodo", "https://zenodo.org/record/5642694"),
        ("FakeAVCeleb", "request form", "https://github.com/DASH-Lab/FakeAVCeleb"),
        ("ADD 2022/2023", "challenge signup", "http://addchallenge.cn/"),
        ("AI-music corpus", "curate: Suno/Udio outputs vs royalty-free human tracks", "-"),
    ],
}


def cmd_list():
    for modality, items in REGISTRY.items():
        print(f"\n== {modality.upper()} ==")
        for name, access, url in items:
            print(f"  {name:26s} [{access}]\n      {url}")
    print("\nAfter downloading, run --arrange to fold a dataset into data/.")


def cmd_fetch(name: str, data_root: Path):
    """Auto-fetch the handful of datasets with frictionless access."""
    name = name.lower()
    if name == "hc3":
        from datasets import load_dataset

        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
        out = data_root / "text"
        import json, random

        rows = list(ds)
        random.Random(0).shuffle(rows)
        cut = int(len(rows) * 0.9)
        for split, part in [("train", rows[:cut]), ("val", rows[cut:])]:
            (out / split).mkdir(parents=True, exist_ok=True)
            with (out / split / "human.jsonl").open("w") as fh, \
                 (out / split / "ai.jsonl").open("w") as fa:
                for r in part:
                    for t in r.get("human_answers") or []:
                        if t.strip():
                            fh.write(json.dumps({"text": t}) + "\n")
                    for t in r.get("chatgpt_answers") or []:
                        if t.strip():
                            fa.write(json.dumps({"text": t}) + "\n")
        print(f"HC3 arranged under {out}")
    elif name == "cifake":
        print("CIFAKE needs the kaggle CLI:\n"
              "  kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images\n"
              "then: python scripts/download_datasets.py --arrange image <unzipped_dir>")
    else:
        print(f"No auto-fetch recipe for '{name}'. See --list for access instructions.")


def cmd_arrange(modality: str, src: Path, data_root: Path):
    """Symlink a manually-downloaded dataset into the expected layout.
    Heuristic: any directory whose name suggests real/fake maps to a class."""
    import re

    real_pat = re.compile(r"(real|authentic|bona|original|human|pristine|au)$", re.I)
    fake_pat = re.compile(r"(fake|tamper|spoof|spliced|forg|generated|ai|tp|manipulat)", re.I)
    class_map = {
        "image": ("real", "tampered"),
        "video": ("real", "deepfake"),
        "audio": ("bonafide", "spoof"),
    }
    if modality not in class_map:
        sys.exit(f"--arrange supports {list(class_map)}; for text use --fetch hc3 "
                 "or write jsonl directly.")
    real_cls, fake_cls = class_map[modality]
    n = 0
    for d in Path(src).rglob("*"):
        if not d.is_dir():
            continue
        cls = real_cls if real_pat.search(d.name) else (fake_cls if fake_pat.search(d.name) else None)
        if cls is None:
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        cut = int(len(files) * 0.9)
        for split, part in [("train", files[:cut]), ("val", files[cut:])]:
            dest = data_root / modality / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in part:
                link = dest / f"{d.name}_{f.name}"
                if not link.exists():
                    link.symlink_to(f.resolve())
                    n += 1
    print(f"Linked {n} files into {data_root / modality}/. Review class mapping manually!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", metavar="NAME")
    ap.add_argument("--arrange", nargs=2, metavar=("MODALITY", "SRC_DIR"))
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()
    root = Path(args.data_root)
    if args.list:
        cmd_list()
    elif args.fetch:
        cmd_fetch(args.fetch, root)
    elif args.arrange:
        cmd_arrange(args.arrange[0], Path(args.arrange[1]), root)
    else:
        cmd_list()


if __name__ == "__main__":
    main()
