"""Image preprocessing + classic forensic-artifact features (ELA, SRM, DCT)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load an image as float32 RGB array in [0, 1], optionally resized to (size, size)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def compute_ela(path_or_array, quality: int = 90, scale: float = 15.0) -> np.ndarray:
    """Error Level Analysis: recompress at a known JPEG quality and take the
    absolute difference. Spliced/edited regions recompress differently and
    light up in the residual. Returns float32 HxWx3 in [0, 1]."""
    if isinstance(path_or_array, (str, Path)):
        img = Image.open(path_or_array).convert("RGB")
    else:
        arr = np.clip(path_or_array * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    a = np.asarray(img, dtype=np.float32)
    b = np.asarray(recompressed, dtype=np.float32)
    ela = np.abs(a - b) * scale / 255.0
    return np.clip(ela, 0.0, 1.0)


# Three canonical SRM high-pass filters (subset of the 30-filter bank) that
# expose local noise residuals disturbed by splicing/inpainting.
_SRM_KERNELS = np.array(
    [
        # KB filter (2nd-order)
        [[-1, 2, -2, 2, -1],
         [2, -6, 8, -6, 2],
         [-2, 8, -12, 8, -2],
         [2, -6, 8, -6, 2],
         [-1, 2, -2, 2, -1]],
        # horizontal 1st-order
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, -2, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        # 3x3 square
        [[0, 0, 0, 0, 0],
         [0, -1, 2, -1, 0],
         [0, 2, -4, 2, 0],
         [0, -1, 2, -1, 0],
         [0, 0, 0, 0, 0]],
    ],
    dtype=np.float32,
)
_SRM_NORMS = np.array([12.0, 2.0, 4.0], dtype=np.float32)


def srm_residuals(rgb: np.ndarray) -> np.ndarray:
    """Apply SRM noise-residual filters to the luminance channel.

    rgb: float32 HxWx3 in [0,1]. Returns float32 HxWx3 (one channel per filter).
    """
    from scipy.signal import convolve2d

    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    out = []
    for k, n in zip(_SRM_KERNELS, _SRM_NORMS):
        r = convolve2d(gray, k / n, mode="same", boundary="symm")
        out.append(np.clip(r, -0.5, 0.5) + 0.5)
    return np.stack(out, axis=-1).astype(np.float32)


def dct_features(rgb: np.ndarray, block: int = 8, n_coeffs: int = 9) -> np.ndarray:
    """Blockwise-DCT statistics that capture JPEG recompression / GAN
    frequency artifacts. Returns a fixed-length 1-D feature vector of
    per-coefficient mean-|value| and std over all 8x8 blocks."""
    from scipy.fftpack import dctn

    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    h, w = gray.shape
    h, w = h - h % block, w - w % block
    if h < block or w < block:
        return np.zeros(2 * n_coeffs, dtype=np.float32)
    g = gray[:h, :w].reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
    blocks = g.reshape(-1, block, block)
    coeffs = dctn(blocks, axes=(1, 2), norm="ortho")
    # zig-zag-ish subset: first n_coeffs low-frequency coefficients (skip DC)
    idx = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)][:n_coeffs]
    feats = []
    for (i, j) in idx:
        c = coeffs[:, i, j]
        feats.extend([np.abs(c).mean(), c.std()])
    return np.asarray(feats, dtype=np.float32)
