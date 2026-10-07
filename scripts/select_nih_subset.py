"""
Chooses the NIH ChestX-ray14 films this project uses, from a laptop, and writes the choice to
data_refs/nih_cxr14_subset.csv, and the lakehouse films to cde/reference/adult_films.csv (the
simulated hospital's adults: label PNEUMOTHORAX, NORMAL = No Finding, or OTHER). Both are
committed: which films a model was trained and tested on is part of its version, and
cxr-setup-nih (scripts/fetch_nih.py) fetches exactly these rows.

Source: the Hugging Face mirror timm/nih-chest-xray-14 at a pinned revision (NIH Clinical
Center; "usage of the data set is unrestricted", with the citation in the README). Only the
metadata columns are read, over HTTP range requests; no image is downloaded here.

Rules (seeded, so a re-run gives the same file):
  adults only          18 <= patient_age <= 95 (the pneumothorax model's intended use; the
                       mirror keeps NIH's handful of impossible ages)
  3 films per patient  at most (NIH has patients with dozens of films, who would otherwise
                       dominate a split): the patient's pneumothorax films first, then the
                       earliest follow-ups. Earliest-only would keep 170 of 2,497 test
                       positives - pneumothorax is mostly seen on follow-up films.
  positive             Pneumothorax among the film's labels; negative = every other film
                       (No Finding and the 13 other findings in their natural mix)
  train                official train_val list, folds 1-9: 1,000 positives, 3 negatives each
  val                  fold 0 (patient-grouped by the mirror): 150 positives, 3 negatives each
  test                 official test list, 85% of its patients: 400 positives and 3 negatives
                       each - the KPI gate's hold-out
  lakehouse            official test list, the other patients: 60 positives and 240 negatives,
                       the adult films of the simulated hospital (20% pneumothorax, enriched
                       so that nine demo days hold enough positives to measure)
  locality             negatives come from the parquet row groups the positives already need
                       (row groups are runs of patient ids, so this is not a clinical bias):
                       cxr-setup-nih reads ~750 of the 1,121 row groups (~13 GB)

  python scripts/select_nih_subset.py              # ~8 min, 16 parallel metadata reads
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "timm/nih-chest-xray-14"
REVISION = "c1bf579641b3256b4d49924436984b82bee5834d"
SHARDS = {"train": 32, "test": 10}
META = ["image_id", "labels", "label_names", "fold", "patient_id", "follow_up", "patient_age",
        "patient_sex", "view_position", "original_width", "original_height"]
OUT = ROOT / "data_refs" / "nih_cxr14_subset.csv"
ADULT_FILMS = ROOT / "cde" / "reference" / "adult_films.csv"    # the simulated hospital's adults
SEED = 20261008
MAX_PER_PATIENT = 3
NEG_PER_POS = 3
TEST_PATIENT_SHARE = 0.85
POSITIVES = {"train": 1000, "val": 150, "test": 400}
LAKEHOUSE = {"positives": 60, "negatives": 240}
ROWS_PER_GROUP = 100


def shard_paths() -> list[tuple[str, str]]:
    return [(split, f"data/{split}-{i:05d}-of-{n:05d}.parquet") for split, n in SHARDS.items() for i in range(n)]


def read_meta(split: str, path: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    # small uncached reads: the metadata columns are KBs between each row group's 18 MB of images
    with fs.open(f"datasets/{REPO}@{REVISION}/{path}", block_size=64 << 10, cache_type="none") as f:
        df = pq.ParquetFile(f, pre_buffer=True).read(columns=META).to_pandas()
    df["source_split"] = split
    df["shard"] = path
    df["row"] = range(len(df))
    print(f"  {path}: {len(df)} rows", flush=True)
    return df


def select(meta, seed: int = SEED):
    import pandas as pd

    df = meta[(meta.patient_age >= 18) & (meta.patient_age <= 95)].copy()
    df["pneumothorax"] = df.label_names.apply(lambda xs: int("Pneumothorax" in list(xs)))
    df = (df.sort_values(["patient_id", "pneumothorax", "follow_up"], ascending=[True, False, True])
          .groupby("patient_id", group_keys=False).head(MAX_PER_PATIENT))
    df["group"] = df.shard + ":" + (df.row // ROWS_PER_GROUP).astype(str)
    rng = random.Random(seed)

    def balanced(pool, n_pos):
        pos = pool[pool.pneumothorax == 1]
        pos = pos.sample(min(n_pos, len(pos)), random_state=seed)
        neg = pool[pool.pneumothorax == 0]
        n_neg = NEG_PER_POS * len(pos)
        near = neg[neg.group.isin(set(pos.group))]
        if len(near) >= n_neg:
            neg = near
        return pd.concat([pos, neg.sample(min(len(neg), n_neg), random_state=seed)])

    tr = df[df.source_split == "train"]
    te = df[df.source_split == "test"]
    test_patients = sorted(te.patient_id.unique())
    rng.shuffle(test_patients)
    cut = int(TEST_PATIENT_SHARE * len(test_patients))
    gate_pat, lake_pat = set(test_patients[:cut]), set(test_patients[cut:])
    lake_pool = te[te.patient_id.isin(lake_pat)]
    lake = pd.concat([
        lake_pool[lake_pool.pneumothorax == 1].sample(LAKEHOUSE["positives"], random_state=seed),
        lake_pool[lake_pool.pneumothorax == 0].sample(LAKEHOUSE["negatives"], random_state=seed)])
    parts = {"train": balanced(tr[tr.fold != 0], POSITIVES["train"]),
             "val": balanced(tr[tr.fold == 0], POSITIVES["val"]),
             "test": balanced(te[te.patient_id.isin(gate_pat)], POSITIVES["test"]), "lakehouse": lake}
    out = pd.concat([p.assign(split=s) for s, p in parts.items()], ignore_index=True)
    out["labels"] = out.label_names.apply(lambda xs: "|".join(xs) if len(xs) else "No Finding")
    out["image_file"] = out.image_id + ".jpg"    # the mirror holds the films as 1024x1024 JPEG
    cols = ["image_id", "image_file", "split", "pneumothorax", "labels", "patient_id", "follow_up", "patient_age", "patient_sex",
            "view_position", "original_width", "original_height", "source_split", "shard", "row"]
    return out[cols].sort_values(["shard", "row"]).reset_index(drop=True)


def check(sel) -> None:
    by = sel.groupby("split").patient_id.apply(set)
    for a in by.index:
        for b in by.index:
            if a < b and by[a] & by[b]:
                raise SystemExit(f"patients in both {a} and {b}: {len(by[a] & by[b])}")
    assert sel.image_id.is_unique


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-cache", default="", help="parquet of the metadata scan (re-use between runs)")
    args = ap.parse_args()
    cache = Path(args.meta_cache) if args.meta_cache else None
    if cache and cache.exists():
        meta = pd.read_parquet(cache)
    else:
        with ThreadPoolExecutor(16) as ex:
            meta = pd.concat(ex.map(lambda sp: read_meta(*sp), shard_paths()), ignore_index=True)
        if cache:
            meta.to_parquet(cache)
    sel = select(meta)
    check(sel)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT, index=False)
    lake = sel[sel.split == "lakehouse"]
    hospital = pd.DataFrame({
        "image_file": lake.image_file,
        "label": ["PNEUMOTHORAX" if p else "NORMAL" if lab == "No Finding" else "OTHER"
                  for p, lab in zip(lake.pneumothorax, lake.labels)],
        "age": lake.patient_age, "sex": lake.patient_sex, "patient_id": lake.patient_id,
        "view_position": lake.view_position, "findings": lake.labels})
    hospital.to_csv(ADULT_FILMS, index=False)
    print(f"hospital adult films: {hospital.label.value_counts().to_dict()} -> {ADULT_FILMS.relative_to(ROOT)}")
    summary = sel.groupby("split").agg(films=("image_id", "size"), pneumothorax=("pneumothorax", "sum"),
                                       patients=("patient_id", "nunique"), median_age=("patient_age", "median"))
    print(summary)
    groups = (sel.shard + ":" + (sel.row // ROWS_PER_GROUP).astype(str)).nunique()
    print(f"row groups to fetch: {groups} of ~{len(meta) // ROWS_PER_GROUP} (~{groups * 18 / 1024:.1f} GB read)")
    print(f"wrote {len(sel)} films -> {OUT.relative_to(ROOT)} (revision {REVISION[:7]} of {REPO})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
