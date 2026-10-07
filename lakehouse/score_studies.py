"""
Job 6 - cxr-06-score-studies (triggered by the CDE Airflow DAG after build_gold, per business date)

  1. REFRESH gold.fact_study in Impala and pin its current snapshot
  2. read the date's studies (with the patient's age) at that snapshot; find each film in the
     project's archive: data/raw/chest_xray/test/** (children, cxr-setup-data),
     data/raw/nih_cxr14/lakehouse/ (adults, cxr-setup-nih), and films planted unfit for AI
     triage (QC-<KIND>-<film>), made from the original on first use (features/degrade.py)
  3. score them through serve.predict.score_images - the code and models the endpoint serves -
     with every model's champion AND silent trial
  4. replace the date in
       gold.triage_score   the worklist: the champions' band (NA = no AI triage), the model that
                           set it, the film check, and the shadow band (as if every silent-trial
                           model were live)
       gold.model_score    every head's probability for every study, in scope or not
     and append ref.triage_run: the snapshot read, counts per band, films not found, PSI of the
     quality features of the pneumonia model's in-scope films vs its TRAIN split, drift level

The business date comes from the job run's environment (CXR_BUSINESS_DATE, set by the DAG)
or --business-date; default yesterday. CXR_TRIGGERED_BY records who asked (airflow run id).
Fails (exit 1) when the date has no studies or a film is missing, after recording the run, so
the DAG stops before build_outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
SUFFIXES = {".jpeg", ".jpg", ".png"}
PLANTED = re.compile(r"^QC-([A-Z]+)-(.+)$")


def film_index(cfg: dict) -> dict:
    nih = ROOT / load_config(model="pneumothorax")["data"]["raw_dir"] / "lakehouse"
    dirs = [ROOT / cfg["data"]["raw_dir"] / "test", nih, ROOT / "data" / "raw" / "planted"]
    return {p.name: p for d in dirs if d.exists() for p in d.rglob("*") if p.suffix.lower() in SUFFIXES}


def plant(name: str, films: dict) -> Path | None:
    """QC-<KIND>-<film>: the film degraded by features/degrade.py, written once to data/raw/planted/."""
    m = PLANTED.match(name)
    if not m or m.group(2) not in films:
        return None
    from features.degrade import degrade
    from features.feature_logic import load_image

    out = ROOT / "data" / "raw" / "planted" / name
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        degrade(load_image(films[m.group(2)]), m.group(1).lower(), key=m.group(2)).save(out, quality=92)
    return out


def champion_scorer():
    """(score(paths, ages) -> (results, quality features), pneumonia champion meta)."""
    from features.feature_logic import load_image, quality_features
    from serve.predict import _META, score_images

    def score(paths, ages):
        imgs = [load_image(p) for p in paths]
        return score_images(imgs, ages, include_trial=True), [quality_features(im) for im in imgs]

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


def shadow(r: dict) -> tuple[str, float | None, str | None]:
    """The worklist band if every silent-trial model were live: a trial head stands in for (or
    beside) its model's champion; the film check still applies."""
    from evaluate.metrics import BAND_RANK

    if (r.get("film_qc") or {}).get("unsuitable"):
        return "NA", None, None
    heads = {**r.get("findings", {}), **r.get("silent_trial", {})}
    live = [(BAND_RANK[h["priority"]], -h["probability"], n) for n, h in heads.items() if h.get("in_scope")]
    if not live:
        return "NA", None, None
    _, p, name = min(live)
    return heads[name]["priority"], -p, name


def head_rows(d: date, accession: str, r: dict, run_id: str, now: datetime) -> list[dict]:
    rows = []
    qc = r.get("film_qc")
    groups = [(r.get("findings") or {}), (r.get("silent_trial") or {})]
    if qc:
        groups.append({"film_qc": {"probability": qc["probability"], "positive": qc["unsuitable"],
                                   "stage": "champion", "in_scope": True, "model_version": qc["model_version"],
                                   "threshold": None, "priority": None}})
    for heads in groups:
        for name, h in heads.items():
            rows.append({"business_date": d, "accession_no": accession, "model": name, "stage": h["stage"],
                         "model_version": h["model_version"], "in_scope": h["in_scope"],
                         "probability": h["probability"], "threshold": h.get("threshold"),
                         "positive": h["positive"], "priority": h.get("priority"), "run_id": run_id,
                         "scored_at": now})
    return rows


def score_date(store, d: date, triggered_by: str, scorer=None, drift_fn=None, films: dict | None = None) -> dict:
    cfg = load_config()
    score, meta = scorer or champion_scorer()
    drift_fn = drift_fn or (lambda q: champion_drift(cfg, meta, q))
    films = film_index(cfg) if films is None else films
    run_id = f"score-{d:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    started = datetime.now()
    store.refresh("gold.fact_study")
    snap = store.snapshot_id("gold.fact_study")
    studies = store.query(f"SELECT accession_no, image_file, age_years FROM {store.source('gold.fact_study', snap)} "
                          f"WHERE business_date = DATE '{d.isoformat()}' ORDER BY accession_no")
    for s in studies:
        if s["image_file"] not in films and (p := plant(s["image_file"], films)) is not None:
            films[s["image_file"]] = p
    found = [s for s in studies if s["image_file"] in films]
    missing = [s["accession_no"] for s in studies if s["image_file"] not in films]

    rows, heads, drift_q = [], [], []
    version = model_version(meta)
    for i in range(0, len(found), BATCH):
        batch = found[i:i + BATCH]
        results, qual = score([films[s["image_file"]] for s in batch], [s.get("age_years") for s in batch])
        now = datetime.now()
        for k, (s, r) in enumerate(zip(batch, results)):
            by = r.get("triage_model", "pneumonia" if r["priority"] != "NA" else None)
            h = (r.get("findings") or {}).get(by) or {}
            qc = r.get("film_qc")
            sh_band, sh_p, sh_model = shadow(r) if "findings" in r else (r["priority"], None, None)
            rows.append({"business_date": d, "accession_no": s["accession_no"], "image_file": s["image_file"],
                         "probability": h.get("probability", r["probability_pneumonia"] if by else None),
                         "priority": r["priority"], "threshold": h.get("threshold", r["threshold"] if by else None),
                         "quality_flags": ",".join(r["quality_flags"]),
                         "model_version": h.get("model_version", version), "model_git_sha": meta["git_sha"][:7],
                         "feature_version": meta["feature_version"], "run_id": run_id, "scored_at": now,
                         "triage_model": by, "film_qc": None if qc is None else "UNSUITABLE" if qc["unsuitable"] else "OK",
                         "age_years": s.get("age_years"), "shadow_priority": sh_band, "shadow_probability": sh_p,
                         "shadow_model": sh_model})
            heads += head_rows(d, s["accession_no"], r, run_id, now)
            pneu = (r.get("findings") or {}).get("pneumonia")
            if qual and (pneu is None or pneu["in_scope"]):
                drift_q.append(qual[k])
    drift, level = drift_fn(drift_q) if drift_q else ({}, "TOO_FEW")
    store.replace_date("gold.triage_score", rows, d)
    if heads:
        store.replace_date("gold.model_score", heads, d)

    status = "FAILED" if not studies else "PARTIAL" if missing else "SUCCEEDED"
    message = ("no studies in gold.fact_study for the date" if not studies else
               f"films not found: {', '.join(missing[:20])}" if missing else "")
    versions: dict = {}
    for h in heads:
        versions.setdefault(h["model"], {})[h["stage"]] = h["model_version"]
    run = {"run_id": run_id, "business_date": d, "triggered_by": triggered_by, "source_snapshot_id": snap,
           "studies": len(studies), "scored": len(rows), "missing_films": len(missing),
           **{b.lower(): sum(1 for r in rows if r["priority"] == b) for b in ("P1", "P2", "P3")},
           "not_triaged": sum(1 for r in rows if r["priority"] == "NA"),
           "film_unsuitable": sum(1 for r in rows if r["film_qc"] == "UNSUITABLE"),
           "heads_json": json.dumps(versions, sort_keys=True),
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
