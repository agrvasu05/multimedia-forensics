#!/usr/bin/env python3
"""End-to-end demo: analyze any file (or raw text) from the command line.

    python scripts/demo.py path/to/file.jpg
    python scripts/demo.py --text "Some passage to check..."
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmforensics.fusion import ForensicOrchestrator
from mmforensics.explainability import explain_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="media file to analyze")
    ap.add_argument("--text", help="analyze raw text instead of a file")
    ap.add_argument("--checkpoints", default="checkpoints")
    args = ap.parse_args()

    orch = ForensicOrchestrator(checkpoint_dir=args.checkpoints)
    if args.text:
        report = orch.analyze_text(args.text)
    elif args.file:
        report = orch.analyze_file(args.file)
    else:
        ap.error("pass a file or --text")

    report["explanation"] = explain_report(report)
    printable = {k: v for k, v in report.items() if k != "localization_mask"}
    print(json.dumps(printable, indent=2, default=str))
    print("\n--- EXPLANATION ---\n" + report["explanation"])


if __name__ == "__main__":
    main()
