"""Occlusion-sensitivity heatmap: model-agnostic, CPU-friendly explanation of where the score comes from.

A grey patch is slid over the film; the drop in pneumonia probability at each
position shows which regions drove the score. Works with any backbone/head.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from features.feature_logic import preprocess


def occlusion_map(img: Image.Image, score_fn, image_size: int = 224, patch: int = 48, stride: int = 24):
    """score_fn: list[PIL grayscale] -> np.ndarray of probabilities. Returns (heatmap[HxW] 0..1, base_prob)."""
    base_img = preprocess(img, image_size)
    arr = np.asarray(base_img).copy()
    fill = int(arr.mean())
    positions, variants = [], []
    for y in range(0, image_size - patch + 1, stride):
        for x in range(0, image_size - patch + 1, stride):
            a = arr.copy()
            a[y:y + patch, x:x + patch] = fill
            variants.append(Image.fromarray(a))
            positions.append((y, x))
    probs = score_fn([base_img] + variants)
    base = float(probs[0])
    heat = np.zeros((image_size, image_size), dtype=np.float32)
    count = np.zeros_like(heat)
    for (y, x), p in zip(positions, probs[1:]):
        heat[y:y + patch, x:x + patch] += max(base - float(p), 0.0)
        count[y:y + patch, x:x + patch] += 1
    heat = heat / np.maximum(count, 1)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat, base
