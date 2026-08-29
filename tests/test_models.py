import numpy as np
import pytest
import torch


def test_dual_branch_image_forward():
    from mmforensics.models.image.dual_branch import DualBranchImageForensics

    model = DualBranchImageForensics(backbone="resnet18", pretrained=False).eval()
    rgb = torch.randn(2, 3, 128, 128)
    art = torch.randn(2, 7, 128, 128)
    dct = torch.randn(2, 18)
    logits, mask = model(rgb, art, dct)
    assert logits.shape == (2, 2)  # real / tampered (AI detection split into ONNX AIDetector)
    assert mask.shape == (2, 1, 128, 128)


def test_deepfake_net_forward():
    from mmforensics.models.video.deepfake_net import SpatioTemporalDeepfakeNet

    model = SpatioTemporalDeepfakeNet(backbone="resnet18", pretrained=False).eval()
    frames = torch.randn(2, 4, 3, 96, 96)
    v, f, w = model(frames)
    assert v.shape == (2,) and f.shape == (2, 4) and w.shape == (2, 4)
    assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)


def test_lcnn_and_rawnet_forward():
    from mmforensics.models.audio.lcnn import LCNN
    from mmforensics.models.audio.rawnet2 import RawNet2

    lcnn = LCNN().eval()
    x = torch.randn(2, 1, 60, 200)
    assert lcnn(x).shape == (2, 2)

    rawnet = RawNet2().eval()
    wav = torch.randn(2, 16000)
    assert rawnet(wav).shape == (2, 2)


def test_music_cnn_forward():
    from mmforensics.models.audio.pipeline import MusicCNN

    m = MusicCNN().eval()
    assert m(torch.randn(2, 19, 100)).shape == (2, 2)


def test_stylometry_features():
    from mmforensics.models.text.stylometry import stylometric_features, FEATURE_NAMES

    f = stylometric_features("This is a test. It has two sentences, honestly!")
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(f).all()


def test_fusion_rule():
    from mmforensics.fusion.orchestrator import ForensicOrchestrator

    v = {"p_fake": 0.9}
    a = {"p_spoof": 0.1}
    fused = ForensicOrchestrator.fuse_video_audio(v, a)
    assert fused["label"] == "deepfake"
    fused2 = ForensicOrchestrator.fuse_video_audio({"p_fake": 0.1}, {"p_spoof": 0.05})
    assert fused2["label"] == "real"
    # audio missing -> falls back to visual score
    fused3 = ForensicOrchestrator.fuse_video_audio({"p_fake": 0.2}, None)
    assert fused3["p_fake_fused"] == pytest.approx(0.2)


def test_metrics():
    from mmforensics.training.metrics import summarize, mask_iou, pixel_f1

    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    m = summarize(labels, scores)
    assert m["auc"] == 1.0 and m["acc"] == 1.0
    pred = np.zeros((8, 8)); pred[:4] = 1.0
    tgt = np.zeros((8, 8)); tgt[:4] = 1.0
    assert mask_iou(pred, tgt) == 1.0 and pixel_f1(pred, tgt) == 1.0
