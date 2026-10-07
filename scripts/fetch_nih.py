"""
CAI job cxr-setup-nih (one-off): the NIH ChestX-ray14 films listed in
data_refs/nih_cxr14_subset.csv (scripts/select_nih_subset.py), from the Hugging Face mirror
timm/nih-chest-xray-14 at the same pinned revision, as the mirror's JPEG bytes:

    data/raw/nih_cxr14/<split>/<image_file>    split = train | val | test | lakehouse
                                               image_file = <image_id>.jpg

Only the parquet row groups (100 films each) that hold a listed film are read, over HTTP
range requests; nothing else is downloaded or kept. Idempotent: films already on disk are
skipped, so a re-run after a network failure resumes. HF_TOKEN (optional) avoids rate limits.

NIH Clinical Center, ChestX-ray14: "The usage of the data set is unrestricted"; cite Wang et
al., CVPR 2017, and link https://nihcc.app.box.com/v/ChestXray-NIHCC (README).

  python scripts/fetch_nih.py
  python scripts/fetch_nih.py --splits lakehouse      # the 300 hospital films only
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

THREADS = 8


def fetch_shard(shard: str, rows, raw: Path, repo: str, revision: str) -> int:
    """Write the listed rows of one parquet file; rows = DataFrame of (row, split, image_id, image_file)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    todo = rows[[not (raw / s / f).exists() for s, f in zip(rows["split"], rows["image_file"])]]
    if todo.empty:
        return 0
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or None)
    n = 0
    with fs.open(f"datasets/{repo}@{revision}/{shard}", block_size=8 << 20) as f:
        pf = pq.ParquetFile(f)
        starts, s = [], 0
        for g in range(pf.metadata.num_row_groups):
            starts.append(s)
            s += pf.metadata.row_group(g).num_rows
        want = {int(r): (sp, iid, fn) for r, sp, iid, fn
                in zip(todo["row"], todo["split"], todo["image_id"], todo["image_file"])}
        groups = sorted({max(g for g, st in enumerate(starts) if st <= r) for r in want})
        for g in groups:
            tb = pf.read_row_group(g, columns=["image", "image_id"]).to_pylist()
            for k, rec in enumerate(tb):
                hit = want.get(starts[g] + k)
                if hit is None:
                    continue
                split, iid, fname = hit
                if rec["image_id"] != iid:
                    raise RuntimeError(f"{shard} row {starts[g] + k}: {rec['image_id']} != listed {iid}")
                out = raw / split / fname
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(out.suffix + ".part")
                tmp.write_bytes(rec["image"]["bytes"])
                tmp.replace(out)
                n += 1
    print(f"  {shard}: {n} films ({len(groups)} row groups)", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test,lakehouse")
    ap.add_argument("--skip-install", action="store_true", help="do not pip install requirements.txt first")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"(ignoring arguments: {unknown})")
    if not args.skip_install:
        from ci.sync_code import install_requirements

        install_requirements(ROOT)
    import pandas as pd

    from common import load_config
    from scripts.select_nih_subset import REPO, REVISION

    cfg = load_config(model="pneumothorax")
    raw = ROOT / cfg["data"]["raw_dir"]
    ref = pd.read_csv(ROOT / cfg["data"]["manifest"])
    ref = ref[ref["split"].isin(args.splits.split(","))]
    print(f"{len(ref)} films listed in {cfg['data']['manifest']} ({args.splits}), "
          f"{ref['shard'].nunique()} parquet files of {REPO}@{REVISION[:7]}", flush=True)
    with ThreadPoolExecutor(THREADS) as ex:
        done = sum(ex.map(lambda kv: fetch_shard(kv[0], kv[1], raw, REPO, REVISION), ref.groupby("shard")))
    have = sum((raw / s / f).exists() for s, f in zip(ref["split"], ref["image_file"]))
    print(f"fetched {done} films; {have}/{len(ref)} on disk under {raw.relative_to(ROOT)}")
    if have != len(ref):
        raise RuntimeError(f"{len(ref) - have} listed films missing")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:   # any SystemExit, even 0, reads as failure under the CAI job kernel
        sys.exit(rc)
