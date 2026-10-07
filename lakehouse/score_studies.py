"""
Job 6 - cxr-06-score-studies (triggered by the CDE Airflow DAG after build_gold, per business date)

  1. REFRESH gold.fact_study in Impala and pin its current snapshot
  2. read the date's studies at that snapshot; find each film in the project's archive
     (data/raw/chest_xray/test/**/<image_file>, fetched by cxr-setup-data)
  3. score them with the champion, through serve.predict.score_images - the code and model
     the endpoint serves
  4. replace the date in gold.triage_score (DELETE + INSERT) and append ref.triage_run:
     the snapshot read, counts per band, films not found, PSI of the quality features vs the
     champion's TRAIN split and the drift level

The business date comes from the job run's environment (CXR_BUSINESS_DATE, set by the DAG)
or --business-date; default yesterday. CXR_TRIGGERED_BY records who asked (airflow run id).
Fails (exit 1) when the date has no studies or a film is missing, after recording the run, so
the DAG stops before build_outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:  # CAI job kernels run the script without __file__; cwd is the project
        return Path(os.getcwd())


sys.path.insert(0, str(_repo_root()))

from common import ROOT, finish, load_config, parse_args  # noqa: E402
from lakehouse.store import model_version  # noqa: E402

BATCH = 32


def film_index(cfg: dict) -> dict:
    test = ROOT / cfg["data"]["raw_dir"] / "test"
    return {p.name: p for p in test.rglob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"}}


def champion_scorer():
    """(score(paths) -> (results, quality features), champion meta)."""
    from features.feature_logic import load_image, quality_features
    from serve.predict import _META, score_images

    def score(paths):
        imgs = [load_image(p) for p in paths]
        return score_images(imgs), [quality_features(im) for im in imgs]

    return score, _META


def champion_drift(cfg: dict, meta: dict, quality: list[dict]) -> tuple[dict, str]:
    """PSI per quality feature vs the champion's TRAIN split; ({}, UNAVAILABLE) without the table."""
    import pandas as pd

    from common import load_feature_table
    from features.feature_logic import QUALITY_FEATURES
    from monitor.batch_score import drift_level, psi

    try:
        ref_df, _ = load_feature_table(cfg, meta["feature_version"])
    except FileNotFoundError:
        return {}, "UNAVAILABLE"
    ref = ref_df[ref_df.split == "train"]
    cur = pd.DataFrame(quality)
    drift = {c: round(psi(ref[c].to_numpy(), cur[c].to_numpy()), 4) for c in QUALITY_FEATURES}
    return drift, drift_level(drift, cfg["monitoring"], len(quality))


def score_date(store, d: date, triggered_by: str, scorer=None, drift_fn=None, films: dict | None = None) -> dict:
    cfg = load_config()
    score, meta = scorer or champion_scorer()
    drift_fn = drift_fn or (lambda q: champion_drift(cfg, meta, q))
    films = film_index(cfg) if films is None else films
    run_id = f"score-{d:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    started = datetime.now()
    store.refresh("gold.fact_study")
    snap = store.snapshot_id("gold.fact_study")
    studies = store.query(f"SELECT accession_no, image_file FROM {store.source('gold.fact_study', snap)} "
                          f"WHERE business_date = DATE '{d.isoformat()}' ORDER BY accession_no")
    found = [s for s in studies if s["image_file"] in films]
    missing = [s["accession_no"] for s in studies if s["image_file"] not in films]

    rows, quality = [], []
    version = model_version(meta)
    for i in range(0, len(found), BATCH):
        batch = found[i:i + BATCH]
        results, qual = score([films[s["image_file"]] for s in batch])
        quality += qual
        now = datetime.now()
        rows += [{"business_date": d, "accession_no": s["accession_no"], "image_file": s["image_file"],
                  "probability": r["probability_pneumonia"], "priority": r["priority"], "threshold": r["threshold"],
                  "quality_flags": ",".join(r["quality_flags"]), "model_version": version,
                  "model_git_sha": meta["git_sha"][:7], "feature_version": meta["feature_version"],
                  "run_id": run_id, "scored_at": now} for s, r in zip(batch, results)]
    drift, level = drift_fn(quality) if quality else ({}, "TOO_FEW")
    store.replace_date("gold.triage_score", rows, d)

    status = "FAILED" if not studies else "PARTIAL" if missing else "SUCCEEDED"
    message = ("no studies in gold.fact_study for the date" if not studies else
               f"films not found: {', '.join(missing[:20])}" if missing else "")
    run = {"run_id": run_id, "business_date": d, "triggered_by": triggered_by, "source_snapshot_id": snap,
           "studies": len(studies), "scored": len(rows), "missing_films": len(missing),
           **{b.lower(): sum(1 for r in rows if r["priority"] == b) for b in ("P1", "P2", "P3")},
           "model_version": version, "threshold": meta["threshold"],
           "psi_max": max(drift.values()) if drift else None, "drift_level": level,
           "psi_json": json.dumps(drift), "status": status, "message": message,
           "started_at": started, "ended_at": datetime.now()}
    store.append("ref.triage_run", [run])
    print(json.dumps({k: (str(v) if isinstance(v, (date, datetime)) else v) for k, v in run.items()}, indent=2))
    return run


def business_date(arg: str | None) -> date:
    value = arg or os.environ.get("CXR_BUSINESS_DATE")
    return date.fromisoformat(value) if value else date.today() - timedelta(days=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="score one business date of gold.fact_study")
    ap.add_argument("--business-date", default=None)
    args = parse_args(ap)
    from lakehouse.store import ImpalaStore

    run = score_date(ImpalaStore(), business_date(args.business_date),
                     os.environ.get("CXR_TRIGGERED_BY", "manual"))
    return 0 if run["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    finish(main())
