# Multimedia Forensic Forgery & Tampering Detection — Project Report

*A unified deep-learning system for AI-generated & manipulated media detection.*

## 1. Overview

This project implements the full plan laid out in the *Multimedia Forensics
Project Plan*: four independent expert pipelines — image, video, text, and
audio — each with its own preprocessing, model architecture, and training
script, combined behind a fusion/orchestration layer and a single FastAPI
`/analyze` endpoint. Every verdict returns a label, a confidence score,
localization where applicable (pixel mask for images, frame scores for
video, sentence spans for text), and a natural-language explanation.

Core goals delivered:

- **Modularity** — each pipeline lives in `mmforensics/models/<modality>/`,
  is independently trainable (`mmforensics/training/train_<modality>.py`),
  and is lazily loaded by the orchestrator so a text query never touches
  vision weights.
- **Dual coverage** — classic tampering (ELA/SRM/DCT artifact branches,
  copy-move/splicing localization) *and* generative-AI content (GAN/diffusion
  image class, deepfake video, LLM text, TTS/voice-clone/music audio).
- **Explainability** — Grad-CAM over the RGB backbone, U-Net localization
  masks, per-frame attention weights, GLTR/DetectGPT statistics, and a
  report-to-prose `explain_report()` layer.
- **Extensibility** — new manipulation types slot in as new classes or new
  branches; the fusion layer and API are agnostic to pipeline internals.

## 2. Architecture

Hub-and-spoke flow (`mmforensics/fusion/orchestrator.py`):

```
file ──► type detection (magic bytes → extension fallback)
      ├─ image ──► ImagePipeline ─────────────────────────┐
      ├─ text  ──► TextPipeline ──────────────────────────┤
      ├─ audio ──► AudioPipeline ─────────────────────────┼──► report + explanation
      └─ video ──► VideoPipeline (visual stream) ─┐       │
                   └─ ffmpeg audio track ─► AudioPipeline ┴─► noisy-OR fusion
```

Video fusion rule: `p = 0.5·[1−(1−p_v)(1−p_a)] + 0.5·max(p_v, p_a)` — a
deepfake is flagged if *either* stream is synthetic, with the noisy-OR term
keeping the score calibrated when streams disagree.

### 2.1 Image module (`models/image/`)

Dual-branch network per plan §4.3:

- **RGB branch** — any timm backbone (EfficientNet-B4 default; ViT/Swin
  supported by name), ImageNet-pretrained, fine-tuned on forensic data.
- **Artifact branch** — shallow 3-block CNN over a 6-channel stack of ELA
  residuals (JPEG-recompression differences) and three canonical SRM
  high-pass noise filters.
- **Frequency branch** — 18-dim blockwise-DCT statistics (per-coefficient
  mean/std over 8×8 blocks) projected through an MLP; captures JPEG
  recompression and GAN/diffusion frequency artifacts.
- **Localization head** — U-Net-style decoder over the artifact branch's
  skip connections predicting a pixel-level tampering mask (trained with
  BCE + Dice).
- **Classifier** — concatenated branch features → MLP → {real, tampered,
  ai_generated}.

### 2.2 Video module (`models/video/`)

- Face extraction per sampled frame (Haar cascade stand-in, drop-in
  replaceable with MTCNN/RetinaFace), largest-face crop with margin.
- **Spatial stream**: Xception (FaceForensics++ baseline standard) per-frame.
- **Temporal stream**: 2-layer Transformer encoder (or Bi-LSTM, selectable)
  over frame features — catches flicker, blending-boundary and
  identity-consistency artifacts.
- **Aggregation**: attention-weighted pooling to a video-level logit, plus
  per-frame logits for localization-in-time; frame-level auxiliary loss
  during training.
- Audio-visual check: the extracted track is scored by the audio pipeline
  and fused (lip-sync-specific SyncNet is left as the documented upgrade).

### 2.3 Text module (`models/text/`)

Four complementary branches; the pipeline is useful *untrained* because
three branches are training-free:

1. **Supervised** — fine-tuned RoBERTa/DeBERTa sequence classifier
   (activates when `checkpoints/text/` exists).
2. **Zero-shot (DetectGPT-style)** — perturb the text and measure the
   normalized log-likelihood drop under GPT-2; AI text sits at sharp
   likelihood curvature.
3. **Statistical (GLTR-style)** — fraction of tokens in the reference LM's
   top-10/top-100 ranks, perplexity, token-logprob variance.
4. **Stylometric** — burstiness, sentence-length variance, function-word
   frequencies, punctuation profile; robust to paraphrase attacks.

**Span localization**: sliding sentence windows re-scored individually flag
which spans of a mixed-authorship document look AI-written.

### 2.4 Audio module (`models/audio/`)

- **LCNN** with Max-Feature-Map activations over LFCC features — the classic
  ASVspoof baseline (linear filterbank keeps high-band vocoder artifacts).
- **RawNet2** — SincConv learned band-pass front-end, residual blocks with
  filter-wise feature-map scaling, GRU head — end-to-end on raw waveform.
- **Wav2Vec2 branch** (optional flag) — self-supervised encoder + linear
  head for cross-vocoder generalization.
- **Music sub-branch** — CNN over chroma + spectral-contrast (19×T) tuned to
  AI-music artifacts (looping seams, unnatural harmonic consistency).

## 3. Training methodology (plan §8)

Implemented in `training/engine.py` and the per-modality scripts:

- AdamW, cosine decay with linear warm-up, label smoothing 0.05.
- Mixed precision (autocast + GradScaler on CUDA), gradient clipping.
- **Progressive unfreezing**: backbone frozen for the first N epochs, then
  released (discriminative fine-tuning against catastrophic forgetting).
- Early stopping on validation AUC (patience 7), best-checkpoint saving.
- **Curriculum**: `train_video.py --resume` supports pretrain-on-DFDC →
  fine-tune-on-Celeb-DF.
- Augmentation per plan §8.2: JPEG recompression/blur/color-jitter/crops for
  images (Albumentations); the video/audio/text equivalents are documented
  hooks in the dataset classes.
- Experiment tracking: `--tracker wandb|mlflow`, always mirrored to JSONL.
- Every script has `--smoke`: generates a synthetic corpus (gradient images
  with pasted patches + masks; flickering clips; template-vs-casual text;
  harmonic vs. looped tones) and runs the whole loop in minutes on CPU —
  so the infrastructure is verified before any dataset download.

## 4. Datasets

Gated benchmarks cannot be redistributed; `scripts/download_datasets.py`
maintains the registry with access URLs, auto-fetches what it can (HC3 via
HuggingFace), and `--arrange` folds manually-downloaded sets into the
expected `data/<modality>/<split>/<class>/` layout.

| Modality | Benchmarks (plan §§4.2/5.2/6.2/7.2) |
|---|---|
| Image | CASIA v1/v2, Columbia, COVERAGE, NIST MFC, CoMoFoD, GenImage, CIFAKE, DFFD |
| Video | FaceForensics++, DFDC, Celeb-DF v2, DeeperForensics-1.0, WildDeepfake |
| Text | GPT-2 Output, HC3, M4, TuringBench, RAID |
| Audio | ASVspoof 2019/2021, WaveFake, FakeAVCeleb, ADD 2022/23, custom AI-music corpus |

Split hygiene (plan §8.1) is the operator's responsibility when arranging
real data: deduplicate near-identical samples and hold out entire
identities/generators — the report's headline metric is **cross-dataset
generalization** (train on one benchmark, evaluate on another via the same
scripts pointed at a different `--data`).

## 5. Evaluation metrics (plan §9)

`training/metrics.py`: accuracy, AUC-ROC, F1, **EER** (video/audio),
**TPR@1%FPR** (text — controls false accusations), **IoU** and **pixel-F1**
(image localization). `summarize()` is used by every validation loop.

## 6. Serving & deployment (plan §11)

- FastAPI gateway (`api/main.py`): `POST /analyze` (any file, multipart),
  `POST /analyze/text`, `GET /health`. Responses carry label, confidence,
  per-branch scores, summarized localization, and an `explanation` string.
- Checkpoints are discovered from `checkpoints/` at startup; missing heads
  degrade gracefully with an explicit `warning` in the response contract.
- `Dockerfile` + `docker-compose.yml` containerize the service (ffmpeg +
  OpenCV system deps included); mount `checkpoints/` as a volume to A/B new
  checkpoints without rebuilding.

## 7. Verification status

- 15/15 unit + integration tests pass (`pytest`): preprocessing features,
  all model forward passes, fusion rules, metrics, and live API calls for
  text/image/audio uploads.
- All four `--smoke` training runs complete end to end and write checkpoints.

## 8. Challenges & mitigations (plan §13)

| Challenge | What the code does |
|---|---|
| Rapidly evolving generators | per-pipeline retraining; registry refresh in `download_datasets.py` |
| Overfitting to dataset artifacts | heavy augmentation, cross-dataset eval workflow, disjoint-split guidance |
| Class imbalance | class-weight hook in the engine, oversampling-ready dataset classes |
| Adversarial evasion | stylometry branch for paraphrased text; compression augmentation for AV |
| Compute cost | efficient default backbones (B0/distilroberta in smoke), AMP, pretrained warm starts |

## 9. Future work (plan §14)

Real-time streaming inference, C2PA provenance verification as a
complementary signal, continual-learning retraining loop, SyncNet lip-sync
branch, MTCNN/RetinaFace face detector swap, and a consumer front-end.
