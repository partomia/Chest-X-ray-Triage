"""
Reusable feature logic for cxr-triage.

This module is the ONLY place where features are defined. The same functions are
imported by:
  - features/build_feature_table.py  (offline: builds the versioned feature table)
  - monitor/batch_score.py           (batch: nightly worklist scoring + drift)
  - serve/predict.py                 (online: CAI model endpoint)
  - app/app.py                       (demo: worklist UI + heatmap)
so training and serving can never compute features differently.

Two feature groups are produced per image:
  1. embedding - frozen vision-transformer embedding (768 floats)
  2. quality   - cheap image statistics used for validation and drift monitoring
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from PIL import Image, ImageOps

# Bump this string whenever the CODE in this file changes behaviour.
# It is folded into the feature hash, so a stale feature table is detected.
FEATURE_LOGIC_VERSION = "1.0.0"

QUALITY_FEATURES = [
    "q_mean",            # mean intensity (0-1)
    "q_std",             # global contrast
    "q_p05",             # dark tail
    "q_p95",             # bright tail
    "q_dynamic_range",   # p95 - p05
    "q_sharpness",       # variance of Laplacian (blur detector)
    "q_aspect_ratio",    # width / height of the original film
    "q_orig_height",
    "q_orig_width",
]

ImageSource = Union[str, Path, bytes, Image.Image]


# --------------------------------------------------------------------------- I/O
def load_image(src: ImageSource) -> Image.Image:
    """Load a film from a path, raw bytes, a base64 string or a PIL image -> grayscale PIL."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        img = Image.open(io.BytesIO(src))
    elif isinstance(src, str) and (len(src) > 1024 or not Path(src).exists()):
        img = Image.open(io.BytesIO(base64.b64decode(src)))   # base64 payload from the API
    else:
        img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("L")


# --------------------------------------------------------------- preprocessing
def preprocess(img: Image.Image, image_size: int) -> Image.Image:
    """Letterbox to a square (no distortion) and resize. Returns grayscale PIL."""
    return ImageOps.pad(img, (image_size, image_size), color=0, method=Image.BILINEAR)


# ------------------------------------------------------------ quality features
def quality_features(img: Image.Image) -> dict:
    """Image-quality statistics computed on the ORIGINAL grayscale film."""
    w, h = img.size
    small = img.copy()
    small.thumbnail((512, 512))
    a = np.asarray(small, dtype=np.float32) / 255.0
    p05, p95 = np.percentile(a, [5, 95])
    lap = (
        4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    )
    return {
        "q_mean": float(a.mean()),
        "q_std": float(a.std()),
        "q_p05": float(p05),
        "q_p95": float(p95),
        "q_dynamic_range": float(p95 - p05),
        "q_sharpness": float(lap.var()),
        "q_aspect_ratio": float(w / h),
        "q_orig_height": float(h),
        "q_orig_width": float(w),
    }


# ------------------------------------------------------------------ embeddings
class Embedder:
    """Frozen Hugging Face vision backbone -> fixed-length embedding.

    torch/transformers are imported lazily so unit tests and the CI runner
    can import this module without installing them.
    """

    def __init__(self, backbone: str, revision: str = "main", pooling: str = "cls",
                 batch_size: int = 32, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(backbone, revision=revision)
        self.model = AutoModel.from_pretrained(backbone, revision=revision).to(self.device).eval()
        self.pooling = pooling
        self.batch_size = batch_size

    def embed(self, images: Iterable[Image.Image]) -> np.ndarray:
        images = [im.convert("RGB") for im in images]
        out = []
        with self._torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = self.processor(images=images[i:i + self.batch_size], return_tensors="pt")
                hidden = self.model(**{k: v.to(self.device) for k, v in batch.items()}).last_hidden_state
                vec = hidden[:, 0] if self.pooling == "cls" else hidden[:, 1:].mean(dim=1)
                out.append(vec.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class StubEmbedder:
    """CI / offline stand-in (backbone "stub:<name>"): 16x16 average-pooled pixels.

    Deterministic and dependency-free, so the whole chain can run on a GitHub
    runner. Its feature hash differs from any real backbone, so a stub-built
    table or model can never be served against real features.
    """

    grid = 16

    def __init__(self, *_, **__):
        self.device = "cpu"

    def embed(self, images: Iterable[Image.Image]) -> np.ndarray:
        out = [np.asarray(im.convert("L").resize((self.grid, self.grid), Image.BOX), dtype=np.float32).ravel() / 255.0
               for im in images]
        return np.stack(out).astype(np.float32)


def make_embedder(feature_cfg: dict):
    """The embedder named by features.backbone (real HF backbone, local folder, or stub:<name>)."""
    cls = StubEmbedder if str(feature_cfg["backbone"]).startswith("stub:") else Embedder
    return cls(feature_cfg["backbone"], feature_cfg["backbone_revision"], feature_cfg["embedding_pooling"],
               feature_cfg["batch_size"])


# --------------------------------------------------------------- feature rows
def compute_features(raw_images: list[Image.Image], embedder, image_size: int):
    """raw grayscale films -> (embeddings[n, d], list[quality dict])."""
    quality = [quality_features(im) for im in raw_images]
    prepped = [preprocess(im, image_size) for im in raw_images]
    return embedder.embed(prepped), quality


def model_matrix(embeddings: np.ndarray, quality: list[dict], model_inputs: list[str]) -> np.ndarray:
    """Assemble the classifier input exactly the same way for training and serving."""
    parts = []
    if "embedding" in model_inputs:
        parts.append(np.asarray(embeddings, dtype=np.float32))
    if "quality" in model_inputs:
        parts.append(np.array([[q[k] for k in QUALITY_FEATURES] for q in quality], dtype=np.float32))
    if not parts:
        raise ValueError("features.model_inputs must include 'embedding' and/or 'quality'")
    return np.hstack(parts)


# --------------------------------------------------------------- versioning
def feature_hash(feature_cfg: dict) -> str:
    """Fingerprint of everything that defines the features. Changes => new table needed."""
    keys = ["backbone", "backbone_revision", "image_size", "embedding_pooling"]
    payload = {k: feature_cfg.get(k) for k in keys}
    payload["feature_logic_version"] = FEATURE_LOGIC_VERSION
    payload["quality_features"] = QUALITY_FEATURES
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
