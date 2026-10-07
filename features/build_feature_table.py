"""
Job 1 - build-features

Builds the versioned feature table of every model (models in config/pipeline.yaml), or of
CXR_MODEL only when it is set:
    <features.store_dir>/v<version>/features.parquet
    <features.store_dir>/v<version>/manifest.json

The films come from data.layout:
  kermany    {train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg; val re-split from train by patient
  manifest   the films listed in data.manifest with their split and label (NIH ChestX-ray14,
             chosen by scripts/select_nih_subset.py, fetched by cxr-setup-nih)
  film_qc    a sample of the real films of data.sources, each also as one degraded copy
             (features/degrade.py): label 1 = unfit for AI triage. Same splits and patients
             as the source models, so no film is in train here and test there.

Idempotent: if the version already exists with the same feature hash it exits
successfully without recomputing, so the CI chain can call it on every push.
If the version exists with a DIFFERENT hash, it fails and asks for a version bump.

Data checks run before anything is written and are recorded in the manifest:
critical ones (patient leakage between train and val, an empty split/class)
fail the job; warnings (unreadable files, the same film in two splits) do not.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT, feature_table_dir, finish, git_sha, load_config, model_names, parse_args  # noqa: E402
from features.degrade import degrade, kind_for  # noqa: E402
from features.feature_logic import (  # noqa: E402
    QUALITY_FEATURES, compute_features, feature_hash, file_sha1, load_image, make_embedder,
)

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def patient_id(fname: str, label: str) -> str:
    """Kermany filenames encode the patient: person123_bacteria_4.jpeg, IM-0115-0001.jpeg, NORMAL2-IM-0927-0001.jpeg."""
    m = re.match(r"(person\d+)_", fname)
    if m:
        return f"{label}:{m.group(1)}"
    m = re.match(r"((?:NORMAL2-)?IM-\d+)-", fname)
    if m:
        return f"{label}:{m.group(1)}"
    return f"{label}:{Path(fname).stem}"


def discover(cfg: dict) -> pd.DataFrame:
    raw = ROOT / cfg["data"]["raw_dir"]
    rows = []
    for src_split in ("train", "val", "test"):
        for label_dir in sorted((raw / src_split).glob("*")):
            if not label_dir.is_dir():
                continue
            for p in sorted(label_dir.glob("*")):
                if p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                rows.append({
                    "rel_path": str(p.relative_to(ROOT)),
                    "source_split": src_split,
                    "label_name": label_dir.name,
                    "patient_id": patient_id(p.name, label_dir.name),
                })
    if not rows:
        raise SystemExit(f"No images found under {raw}. See the runbook, step 'Load the data'.")
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep the published TEST split as the hold-out. Re-split train+val BY PATIENT (no leakage)."""
    from sklearn.model_selection import GroupShuffleSplit

    df = df.copy()
    df["split"] = np.where(df["source_split"] == "test", "test", "train")
    pool = df[df["split"] == "train"]
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["data"]["val_fraction"],
                            random_state=cfg["data"]["split_seed"])
    _, val_idx = next(gss.split(pool, groups=pool["patient_id"]))
    df.loc[pool.index[val_idx], "split"] = "val"
    df["label"] = (df["label_name"] == cfg["data"]["positive_label"]).astype(int)
    return df


def kermany_films(cfg: dict) -> pd.DataFrame:
    return assign_splits(discover(cfg), cfg).assign(degradation="none")


def manifest_films(cfg: dict) -> pd.DataFrame:
    """The films of data.manifest in train/val/test (its lakehouse films are for the hospital, not the model)."""
    ref = pd.read_csv(ROOT / cfg["data"]["manifest"])
    ref = ref[ref["split"].isin(["train", "val", "test"])]
    raw = Path(cfg["data"]["raw_dir"])
    pos = cfg["data"]["positive_label"]
    return pd.DataFrame({
        "rel_path": [str(raw / s / f) for s, f in zip(ref["split"], ref["image_file"])],
        "source_split": ref["source_split"], "split": ref["split"],
        "label_name": np.where(ref["pneumothorax"] == 1, pos, f"NO_{pos}"),
        "label": ref["pneumothorax"].astype(int),
        "patient_id": [f"nih:{p}" for p in ref["patient_id"]], "degradation": "none"}).reset_index(drop=True)


def qc_films(cfg: dict) -> pd.DataFrame:
    """films_per_split real films per split, half from each source model, plus a degraded copy of each."""
    seed = cfg["data"]["split_seed"]
    sources = cfg["data"]["sources"]
    parts = []
    for src in sources:
        films = LAYOUTS[load_config(model=src)["data"]["layout"]](load_config(model=src))
        for split, n in cfg["data"]["films_per_split"].items():
            pool = films[films["split"] == split]
            parts.append(pool.sample(min(len(pool), n // len(sources)), random_state=seed)
                         .assign(patient_id=lambda d, s=src: s + ":" + d["patient_id"]))
    real = pd.concat(parts, ignore_index=True).assign(degradation="none", label=0, label_name="SUITABLE")
    bad = real.assign(degradation=[kind_for(p, seed) for p in real["rel_path"]], label=1, label_name="UNSUITABLE")
    return pd.concat([real, bad], ignore_index=True)


LAYOUTS = {"kermany": kermany_films, "manifest": manifest_films, "film_qc": qc_films}


def film_image(rel_path: str, degradation: str, seed: int):
    img = load_image(ROOT / rel_path)
    return img if degradation == "none" else degrade(img, degradation, seed, key=rel_path)


def data_checks(df: pd.DataFrame, unreadable: list[str]) -> list[dict]:
    """Checks on the assembled table. severity critical = stop the build."""
    def check(name, severity, passed, observed):
        return {"check": name, "severity": severity, "passed": bool(passed), "observed": observed}

    tr = set(df.loc[df.split == "train", "patient_id"])
    va = set(df.loc[df.split == "val", "patient_id"])
    counts = df.groupby(["split", "label"]).size()
    empty = [f"{s}/{lab}" for s in ("train", "val", "test") for lab in (0, 1) if counts.get((s, lab), 0) == 0]
    sha_splits = df.groupby("image_sha1")["split"].nunique()
    return [
        check("no patient in both train and val", "critical", not (tr & va), len(tr & va)),
        check("every split has both classes", "critical", not empty, empty),
        check("all image files readable", "warning", not unreadable, len(unreadable)),
        check("no identical film in two splits", "warning", int((sha_splits > 1).sum()) == 0,
              int((sha_splits > 1).sum())),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if this version exists")
    ap.add_argument("--limit", type=int, default=0, help="debug: only use N images per split and class")
    args = parse_args(ap)
    names = [os.environ["CXR_MODEL"]] if os.environ.get("CXR_MODEL") else model_names(load_config())
    for name in names:
        print(f"[build-features] model {name}", flush=True)
        rc = build(load_config(model=name), args)
        if rc:
            return rc
    return 0


def build(cfg: dict, args) -> int:
    fcfg = cfg["features"]
    fhash = feature_hash(fcfg)
    out = feature_table_dir(cfg)
    manifest_path = out / "manifest.json"

    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text())
        if existing.get("limit") and not args.limit:
            print(f"[build-features] v{fcfg['version']} was a --limit {existing['limit']} smoke build - rebuilding.")
        elif existing["feature_hash"] == fhash:
            print(f"[build-features] {out.relative_to(ROOT)} already built (hash {fhash}) - skipping.")
            return 0
        else:
            print(f"[build-features] v{fcfg['version']} exists with hash {existing['feature_hash']} but "
                  f"current feature logic hashes to {fhash}. Bump features.version in config/pipeline.yaml.")
            return 1

    df = LAYOUTS[cfg["data"]["layout"]](cfg)
    if args.limit:
        df = df.groupby(["split", "label"], group_keys=False).head(args.limit).reset_index(drop=True)
    print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))

    seed = cfg["data"]["split_seed"]
    embedder = make_embedder(fcfg)
    embs, quals, keep, unreadable = [], [], [], []
    chunk = 256
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk]
        imgs, idx = [], []
        for j, p, kind in zip(part.index, part["rel_path"], part["degradation"]):
            try:
                imgs.append(film_image(p, kind, seed))
                idx.append(j)
            except Exception as e:  # corrupt / truncated JPEG: recorded, not fatal
                unreadable.append(p)
                print(f"  unreadable {p}: {e}")
        if imgs:
            e, q = compute_features(imgs, embedder, fcfg["image_size"])
            embs.append(e)
            quals.extend(q)
            keep.extend(idx)
        print(f"  features {min(i + chunk, len(df))}/{len(df)}", flush=True)

    df = df.loc[keep].reset_index(drop=True)
    emb = np.concatenate(embs)
    df["image_sha1"] = [file_sha1(ROOT / p) for p in df["rel_path"]]
    df["embedding"] = list(emb)
    df = pd.concat([df, pd.DataFrame(quals)], axis=1)
    df["feature_version"] = fcfg["version"]
    df["feature_hash"] = fhash

    checks = data_checks(df, unreadable)
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'} [{c['severity']}] {c['check']}: {c['observed']}")
    if any(not c["passed"] and c["severity"] == "critical" for c in checks):
        print("[build-features] critical data check failed - nothing written.")
        return 1

    # Reference statistics from TRAIN only -> used online for quality flags and for PSI drift.
    train_q = df.loc[df["split"] == "train", QUALITY_FEATURES]
    ref = {c: {"p01": float(train_q[c].quantile(0.01)), "p99": float(train_q[c].quantile(0.99)),
               "mean": float(train_q[c].mean()), "std": float(train_q[c].std())} for c in QUALITY_FEATURES}

    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "features.parquet", index=False)
    manifest = {
        "model": cfg["model"]["name"],
        "layout": cfg["data"]["layout"],
        "feature_version": fcfg["version"],
        "feature_hash": fhash,
        "backbone": fcfg["backbone"],
        "backbone_revision": fcfg["backbone_revision"],
        "image_size": fcfg["image_size"],
        "embedding_pooling": fcfg["embedding_pooling"],
        "embedding_dim": int(emb.shape[1]),
        "quality_features": QUALITY_FEATURES,
        "quality_reference": ref,
        "rows": int(len(df)),
        "rows_by_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "positives_by_split": {k: int(v) for k, v in df.groupby("split")["label"].sum().items()},
        "patients": int(df["patient_id"].nunique()),
        "data_checks": checks,
        "limit": args.limit,
        "git_sha": git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[build-features] wrote {len(df)} rows -> {out}")
    from lakehouse.publish import publish_training_set

    publish_training_set(manifest)
    return 0


if __name__ == "__main__":
    finish(main())
