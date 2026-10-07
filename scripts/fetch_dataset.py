"""
CAI job cxr-setup-data (one-off): the Kermany et al. 2018 chest X-ray dataset in the
layout the pipeline reads, from the Hugging Face mirror hf-vision/chest-xray-pneumonia
(CC BY 4.0, the Kaggle split 5,216 / 16 / 624, original file names and JPEG bytes).
No Kaggle token is needed.

    data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/<original file name>
    data/incoming/   the first 25 PNEUMONIA and 15 NORMAL test films (the app's worklist)

Installs requirements.txt first (ci/sync_code.py, same hash marker), so on a new
project this job is the one-time setup: run it once, then start cxr-00-sync-code.
Idempotent: a split already complete is skipped. HF_TOKEN (optional) avoids rate limits.

  python scripts/fetch_dataset.py
  python scripts/fetch_dataset.py --splits test      # 79 MB only
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

REPO = "hf-vision/chest-xray-pneumonia"
REVISION = "c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc"   # 2023-12-11
SHARDS = {"train": ("train", 7), "val": ("validation", 1), "test": ("test", 1)}
EXPECTED = {"train": 5216, "val": 16, "test": 624}
LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
INCOMING = {"PNEUMONIA": 25, "NORMAL": 15}


def shard_urls(split: str) -> list[str]:
    name, n = SHARDS[split]
    return [f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}-{i:05d}-of-{n:05d}.parquet"
            for i in range(n)]


def download(url: str, dest: Path) -> None:
    import requests

    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def write_films(parquet: Path, out: Path) -> int:
    import pyarrow.parquet as pq

    n = 0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            d = out / LABELS[int(row["label"])]
            d.mkdir(parents=True, exist_ok=True)
            (d / Path(row["image"]["path"]).name).write_bytes(row["image"]["bytes"])
            n += 1
    return n


def count(split_dir: Path) -> int:
    return sum(1 for p in split_dir.rglob("*.jpeg")) if split_dir.exists() else 0


def fetch(split: str, raw: Path) -> int:
    out = raw / split
    have = count(out)
    if have == EXPECTED[split]:
        print(f"{split}: {have} films, complete - skipping")
        return have
    shutil.rmtree(out, ignore_errors=True)
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        for url in shard_urls(split):
            shard = Path(tmp) / url.rsplit("/", 1)[-1]
            print(f"{split}: downloading {shard.name}", flush=True)
            download(url, shard)
            print(f"{split}: {write_films(shard, out)} films from {shard.name}", flush=True)
            shard.unlink()
    have = count(out)
    if have != EXPECTED[split]:
        raise RuntimeError(f"{split}: {have} films, expected {EXPECTED[split]}")
    return have


def fill_incoming(raw: Path, incoming: Path) -> int:
    incoming.mkdir(parents=True, exist_ok=True)
    if any(incoming.glob("*.jpeg")):
        print(f"incoming: {sum(1 for _ in incoming.glob('*.jpeg'))} films already there - left as is")
        return 0
    n = 0
    for label, k in INCOMING.items():
        for p in sorted((raw / "test" / label).glob("*.jpeg"))[:k]:
            shutil.copy2(p, incoming / p.name)
            n += 1
    print(f"incoming: copied {n} test films")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--skip-install", action="store_true", help="do not pip install requirements.txt first")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"(ignoring arguments: {unknown})")
    if not args.skip_install:
        from ci.sync_code import install_requirements

        install_requirements(ROOT)
    from common import load_config

    cfg = load_config()
    raw = ROOT / cfg["data"]["raw_dir"]
    counts = {s: fetch(s, raw) for s in args.splits.split(",")}
    print(f"films: {counts} (revision {REVISION[:7]} of {REPO})")
    if "test" in counts:
        fill_incoming(raw, ROOT / cfg["data"]["incoming_dir"])
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:   # any SystemExit, even 0, reads as failure under the CAI job kernel
        sys.exit(rc)
