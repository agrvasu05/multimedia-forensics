#!/usr/bin/env python3
"""Export trained models to ONNX for optimized serving (plan §10: ONNX
Runtime / TorchServe).

    python scripts/export_onnx.py --modality audio --checkpoint checkpoints/audio.pt
    python scripts/export_onnx.py --modality image --checkpoint checkpoints/image.pt
    python scripts/export_onnx.py --modality all           # export whatever exists

Outputs land next to the checkpoint as <name>.onnx and are verified with a
parity check against the PyTorch forward pass via onnxruntime.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _verify(onnx_path: Path, inputs: dict, torch_out: torch.Tensor):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {k: v.numpy() for k, v in inputs.items()})[0]
    err = float(np.abs(ort_out - torch_out.detach().numpy()).max())
    status = "OK" if err < 1e-3 else "MISMATCH"
    print(f"  {onnx_path.name}: parity max-err={err:.2e} [{status}]")


def export_image(ckpt: Path, backbone: str = "efficientnet_b0"):
    from mmforensics.models.image.dual_branch import DualBranchImageForensics

    model = DualBranchImageForensics(backbone=backbone, pretrained=False).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))

    class ClsOnly(torch.nn.Module):  # export the classifier head (mask via torch)
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, rgb, art, dct):
            return self.m(rgb, art, dct)[0]

    rgb, art, dct = torch.randn(1, 3, 256, 256), torch.randn(1, 7, 256, 256), torch.randn(1, 18)
    out = ckpt.with_suffix(".onnx")
    torch.onnx.export(ClsOnly(model), (rgb, art, dct), str(out),
                      input_names=["rgb", "artifacts", "dct"], output_names=["logits"],
                      dynamic_axes={"rgb": {0: "b"}, "artifacts": {0: "b"}, "dct": {0: "b"}},
                      opset_version=17, dynamo=False)
    _verify(out, {"rgb": rgb, "artifacts": art, "dct": dct}, ClsOnly(model)(rgb, art, dct))


def export_audio(ckpt: Path):
    from mmforensics.models.audio.lcnn import LCNN
    from mmforensics.models.audio.rawnet2 import RawNet2

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    lcnn = LCNN().eval()
    lcnn.load_state_dict(state["lcnn"])
    x = torch.randn(1, 1, 60, 200)
    out = ckpt.parent / "audio_lcnn.onnx"
    torch.onnx.export(lcnn, (x,), str(out), input_names=["lfcc"], output_names=["logits"],
                      dynamic_axes={"lfcc": {0: "b", 3: "t"}}, opset_version=17, dynamo=False)
    _verify(out, {"lfcc": x}, lcnn(x))

    rawnet = RawNet2().eval()
    rawnet.load_state_dict(state["rawnet"])
    wav = torch.randn(1, 64000)
    out2 = ckpt.parent / "audio_rawnet2.onnx"
    torch.onnx.export(rawnet, (wav,), str(out2), input_names=["waveform"],
                      output_names=["logits"], dynamic_axes={"waveform": {0: "b"}},
                      opset_version=17, dynamo=False)
    _verify(out2, {"waveform": wav}, rawnet(wav))


def export_video(ckpt: Path, backbone: str = "resnet18"):
    from mmforensics.models.video.deepfake_net import SpatioTemporalDeepfakeNet

    model = SpatioTemporalDeepfakeNet(backbone=backbone, pretrained=False).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state))

    class VideoLogit(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, frames):
            return self.m(frames)[0]

    frames = torch.randn(1, 8, 3, 160, 160)
    out = ckpt.with_suffix(".onnx")
    torch.onnx.export(VideoLogit(model), (frames,), str(out), input_names=["frames"],
                      output_names=["video_logit"], dynamic_axes={"frames": {0: "b"}},
                      opset_version=17, dynamo=False)
    _verify(out, {"frames": frames}, VideoLogit(model)(frames))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True,
                    choices=["image", "video", "audio", "all"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--image-backbone", default="efficientnet_b0")
    ap.add_argument("--video-backbone", default="resnet18")
    args = ap.parse_args()

    d = Path(args.ckpt_dir)
    todo = [args.modality] if args.modality != "all" else ["image", "video", "audio"]
    for m in todo:
        ckpt = Path(args.checkpoint) if args.checkpoint else d / f"{m}.pt"
        if not ckpt.exists():
            print(f"skip {m}: no checkpoint at {ckpt}")
            continue
        print(f"exporting {m} from {ckpt}...")
        {"image": lambda: export_image(ckpt, args.image_backbone),
         "video": lambda: export_video(ckpt, args.video_backbone),
         "audio": lambda: export_audio(ckpt)}[m]()


if __name__ == "__main__":
    main()
