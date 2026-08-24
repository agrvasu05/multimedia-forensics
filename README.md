# Multimedia Forensic Forgery & Tampering Detection

A unified deep-learning system that detects four categories of manipulated or
synthetic content behind a single API:

| Modality | Detects | Core models |
|---|---|---|
| 🖼 Image | Splicing, copy-move, retouching, GAN/diffusion images + pixel-level localization | Dual-branch: EfficientNet/ViT (RGB) + shallow CNN over ELA/SRM residuals + DCT stats + U-Net mask decoder |
| 🎬 Video | Face-swap & reenactment deepfakes, temporal flicker, lip-sync mismatch (audio track cross-check) | Xception per-frame + temporal Transformer/Bi-LSTM + attention pooling |
| 📝 Text | LLM-generated, paraphrase-obfuscated, mixed-authorship text with span localization | RoBERTa/DeBERTa (supervised) + DetectGPT curvature + GLTR statistics + stylometry |
| 🎵 Audio | TTS voice, voice cloning, AI-generated music | LCNN on LFCC + RawNet2 (raw waveform) + optional Wav2Vec2 + music CNN (chroma/spectral-contrast) |

The system is **hub-and-spoke**: an orchestrator type-detects the input,
routes it to the right expert pipeline(s) — a video goes to *both* the video
and audio pipelines — and fuses the scores into one explainable verdict.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                          # sanity-check the install
```

Analyze anything from the CLI:

```bash
python scripts/demo.py path/to/photo.jpg
python scripts/demo.py path/to/clip.mp4
python scripts/demo.py song.mp3
python scripts/demo.py --text "Paste a suspicious passage here..."
```

Run the API:

```bash
uvicorn mmforensics.api.main:app --reload
# POST /analyze        (multipart file upload — any supported type)
# POST /analyze/text   ({"text": "..."})
# GET  /health
```

or with Docker:

```bash
docker compose up --build
```

**Note on untrained heads.** The text pipeline works out of the box (its
GLTR/DetectGPT/stylometry branches are training-free). The image, video, and
audio pipelines load ImageNet/self-supervised pretrained backbones but need
their forensic heads trained (below) before their scores are meaningful —
until then every report carries an explicit `warning` field.

## Training

Every training script has a `--smoke` mode that generates a tiny synthetic
dataset and runs the full loop end to end with zero downloads — use it to
verify your setup before committing GPU time:

```bash
python -m mmforensics.training.train_image --smoke
python -m mmforensics.training.train_video --smoke
python -m mmforensics.training.train_text  --smoke
python -m mmforensics.training.train_audio --smoke
```

For real training, list/fetch/arrange the benchmark datasets (most are
gated behind license forms — the script prints access URLs):

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --fetch hc3          # auto-fetches HC3 text data
python scripts/download_datasets.py --arrange image /path/to/CASIA2
```

then train per modality (checkpoints land in `checkpoints/`, which the API
and orchestrator pick up automatically):

```bash
python -m mmforensics.training.train_image --data data/image --backbone efficientnet_b4 --epochs 30
python -m mmforensics.training.train_video --data data/video --epochs 20            # pretrain (e.g. DFDC)
python -m mmforensics.training.train_video --resume checkpoints/video.pt --data data/celebdf  # curriculum fine-tune
python -m mmforensics.training.train_text  --data data/text --model roberta-base
python -m mmforensics.training.train_audio --data data/audio --epochs 30
```

The engine implements the plan's optimization setup: AdamW, cosine decay
with warm-up, label smoothing, mixed precision, progressive backbone
unfreezing, early stopping on validation AUC, and optional `--tracker
wandb|mlflow` experiment logging (mirrored to a JSONL file either way).

## Evaluation

`mmforensics/training/metrics.py` implements the plan's metric suite:
AUC-ROC, F1, accuracy, **EER** (audio/video standard), **TPR@1%FPR** (to
control false accusations on text), and **IoU / pixel-F1** for tampering
localization masks. Cross-dataset generalization — train on one benchmark,
test on an unseen one — is the headline evaluation; point `--data` at a
different dataset's folder and reuse the same scripts.

## Repository layout

```
mmforensics/
├── preprocessing/    type detection, ELA/SRM/DCT, frame & audio extraction, text cleaning, LFCC/Mel
├── models/
│   ├── image/        dual-branch net + localization decoder + pipeline
│   ├── video/        spatio-temporal deepfake net + pipeline
│   ├── text/         supervised + DetectGPT + GLTR + stylometry detector
│   └── audio/        LCNN, RawNet2, Wav2Vec2 branch, music CNN + pipeline
├── fusion/           orchestrator: routing + video/audio noisy-OR fusion
├── explainability/   Grad-CAM, mask overlays, natural-language explanations
├── api/              FastAPI gateway (/analyze, /analyze/text)
└── training/         engine (AdamW/cosine/AMP/unfreezing), datasets, metrics, 4 train scripts
scripts/              dataset registry/fetch/arrange, CLI demo
configs/              default hyperparameters per modality
tests/                unit + API tests
docs/                 project report
```

## Documentation

See [docs/REPORT.md](docs/REPORT.md) for the full write-up: architecture
decisions, dataset table, training methodology, evaluation protocol,
challenges/mitigations, and the roadmap mapping to the original project plan.
