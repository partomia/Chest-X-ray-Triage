"""
Films unfit for AI triage, made from real ones: the positives of the film-quality model
(models.film_qc) and the planted bad films of the simulated hospital.

Each kind is a failure radiographers reject films for, at a severity a radiologist would
send back rather than read through:
  blur          patient motion: Gaussian blur of 0.8-1.6 % of the film's width
  underexposed  too little dose: dark, compressed toward black
  overexposed   burnt out: bright, compressed toward white
  noise         quantum mottle: strong Gaussian noise
  cropped       collimation / positioning: only 45-65 % of the film, one corner
  rotated       stored 90 or 270 degrees off (wrong orientation tag)
  inverted      MONOCHROME1 shown as MONOCHROME2: bones dark, air white

Deterministic in (film, seed): degrade(img, kind, seed) always returns the same film.
The model only ever learns from these synthetic failures; a real deployment would retrain
on the films its radiographers actually rejected (reject analysis), and says so.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps

KINDS = ("blur", "underexposed", "overexposed", "noise", "cropped", "rotated", "inverted")


def _rng(key: str, seed: int) -> random.Random:
    return random.Random(int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()[:12], 16))


def kind_for(key: str, seed: int) -> str:
    return _rng(f"kind:{key}", seed).choice(KINDS)


def degrade(img: Image.Image, kind: str, seed: int = 0, key: str = "") -> Image.Image:
    """A grayscale film with one failure of the given kind."""
    rng = _rng(f"{kind}:{key}", seed)
    img = img.convert("L")
    w, h = img.size
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(w * rng.uniform(0.008, 0.016)))
    if kind in ("underexposed", "overexposed", "noise"):
        a = np.asarray(img, dtype=np.float32) / 255.0
        if kind == "underexposed":
            a = (a ** rng.uniform(2.6, 3.6)) * rng.uniform(0.35, 0.55)
        elif kind == "overexposed":
            a = 1.0 - ((1.0 - a) ** rng.uniform(2.6, 3.6)) * rng.uniform(0.35, 0.55)
        else:
            a = a + np.random.default_rng(rng.randrange(2**31)).normal(0, rng.uniform(0.14, 0.22), a.shape)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    if kind == "cropped":
        f = rng.uniform(0.45, 0.65)
        cw, ch = int(w * f), int(h * f)
        x0, y0 = rng.choice([(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)])
        return img.crop((x0, y0, x0 + cw, y0 + ch))
    if kind == "rotated":
        return img.rotate(rng.choice([90, 270]), expand=True)
    if kind == "inverted":
        return ImageOps.invert(img)
    raise ValueError(f"unknown degradation {kind!r}; one of {KINDS}")
