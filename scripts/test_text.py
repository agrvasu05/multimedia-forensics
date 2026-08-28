#!/usr/bin/env python3
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmforensics.models.text import TextPipeline

pipe = TextPipeline(checkpoint="checkpoints/text")

if len(sys.argv) > 1:
    text = sys.argv[1]
else:
    text = input("Enter text to analyze: ")

score = pipe.supervised_score(text)
label = "ai_generated" if score >= 0.5 else "human"
print(json.dumps({"label": label, "p_ai": score, "supervised_score": score}, indent=2))