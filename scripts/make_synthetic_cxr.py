"""
Synthetic chest films in the Kermany layout, for CI and offline smoke runs only.

    data/synthetic/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
    data/synthetic/incoming/*.jpeg
    data/synthetic/nih_cxr14/{train,val,test}/*.jpg + data/synthetic/nih_cxr14_subset.csv
        other films in the NIH manifest layout, for the pneumothorax model's CI run

A "film" is a dark field with two lung fields, a bright mediastinum, faint ribs
and noise; PNEUMONIA films add one or two soft opacities inside a lung, some of
them faint, so the task is learnable but not trivial. File names follow the
real dataset (person<N>_bacteria_<k>.jpeg, IM-<N>-<k>.jpeg) with several films
per patient, so patient grouping and the leakage check are exercised.
Never use these images to show model quality.

  python scripts/make_synthetic_cxr.py                  # default sizes (~1 minute)
  python scripts/make_synthetic_cxr.py --scale 0.3      # smaller
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
# (patients, films per patient max) per split and class: roughly the real class mix
LAYOUT = {
    "train": {"NORMAL": 110, "PNEUMONIA": 260},
    "val": {"NORMAL": 8, "PNEUMONIA": 8},
    "test": {"NORMAL": 90, "PNEUMONIA": 140},
}
FILMS_PER_PATIENT = 3


def film(rng: np.random.Generator, pneumonia: bool) -> Image.Image:
    h = int(rng.integers(300, 420))
    w = int(h * rng.uniform(1.05, 1.35))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.full((h, w), 0.25 + rng.normal(0, 0.03), np.float32)
    body = ((xx - w / 2) / (0.46 * w)) ** 2 + ((yy - h * 0.55) / (0.52 * h)) ** 2 < 1
    a[body] += 0.30
    lungs = []
    for side in (-1, 1):
        cx, cy = w / 2 + side * 0.2 * w, h * 0.5
        rx, ry = 0.13 * w * rng.uniform(0.9, 1.1), 0.3 * h * rng.uniform(0.9, 1.1)
        lung = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
        a[lung] -= 0.28
        lungs.append((cx, cy, rx, ry))
    a[np.abs(xx - w / 2) < 0.06 * w] += 0.25
    for k in range(8):
        a += 0.04 * np.exp(-((yy - h * (0.25 + 0.07 * k) - 0.05 * np.abs(xx - w / 2)) / 4) ** 2)
    if pneumonia:
        for _ in range(int(rng.integers(1, 3))):
            cx, cy, rx, ry = lungs[int(rng.integers(0, 2))]
            bx, by = cx + rng.uniform(-0.5, 0.5) * rx, cy + rng.uniform(-0.5, 0.6) * ry
            r = rng.uniform(0.25, 0.55) * rx
            strength = rng.choice([rng.uniform(0.08, 0.14), rng.uniform(0.18, 0.35)], p=[0.25, 0.75])
            a += strength * np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * r ** 2)))
    a = a * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)
    a += rng.normal(0, 0.035, a.shape)
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply patient counts")
    ap.add_argument("--incoming", type=int, default=24, help="unlabelled films for the worklist")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    shutil.rmtree(out, ignore_errors=True)
    pid = 1
    total = 0
    for split, classes in LAYOUT.items():
        for label, patients in classes.items():
            d = out / "chest_xray" / split / label
            d.mkdir(parents=True, exist_ok=True)
            n_pat = max(4, int(patients * args.scale)) if split != "val" else patients
            for _ in range(n_pat):
                for k in range(1, int(rng.integers(1, FILMS_PER_PATIENT + 1)) + 1):
                    if label == "PNEUMONIA":
                        name = f"person{pid}_{rng.choice(['bacteria', 'virus'])}_{k}.jpeg"
                    else:
                        name = f"IM-{pid:04d}-{k:04d}.jpeg"
                    film(rng, label == "PNEUMONIA").save(d / name, quality=90)
                    total += 1
                pid += 1
    inc = out / "incoming"
    inc.mkdir(parents=True, exist_ok=True)
    for i in range(args.incoming):
        film(rng, bool(i % 3 == 0)).save(inc / f"incoming_{i:03d}.jpeg", quality=90)
    n_nih = nih_like(out, rng, args.scale)
    print(f"wrote {total} labelled films + {args.incoming} incoming films under {out}; "
          f"{n_nih} more in the NIH manifest layout")


def nih_like(out: Path, rng: np.random.Generator, scale: float) -> int:
    """Other films as a manifest-layout dataset (the pneumothorax model's, config/ci.yaml):
    nih_cxr14/<split>/<patient>_<k>.jpg and nih_cxr14_subset.csv. The opacity stands in for
    a pneumothorax, so the code path runs, not the clinical task."""
    import csv

    rows, patient = [], 50000
    for split, classes in LAYOUT.items():
        (out / "nih_cxr14" / split).mkdir(parents=True, exist_ok=True)
        for label, patients in classes.items():
            for _ in range(max(4, int(patients * scale * 0.5)) if split != "val" else patients):
                patient += 1
                for k in range(int(rng.integers(1, FILMS_PER_PATIENT + 1))):
                    name = f"{patient:08d}_{k:03d}.jpg"
                    film(rng, label == "PNEUMONIA").save(out / "nih_cxr14" / split / name, quality=90)
                    rows.append({"image_id": name[:-4], "image_file": name, "split": split,
                                 "pneumothorax": int(label == "PNEUMONIA"), "patient_id": patient,
                                 "source_split": "test" if split == "test" else "train"})
    with open(out / "nih_cxr14_subset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    main()
