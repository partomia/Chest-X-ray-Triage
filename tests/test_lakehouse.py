"""Lakehouse logic that needs no Spark: the deterministic hospital, the reading queue, contracts,
the CAI scoring job against an in-memory store, and the names shared by DAG, scripts and jobs."""
from __future__ import annotations

import ast
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cde" / "jobs"))

import cxr_common as C  # noqa: E402

CFG = C.config()
D = date(2026, 9, 28)
# Impala reserved words that would need backticks in every consumer's SQL
RESERVED = {"location", "view", "method", "change", "matched", "rows", "columns", "range", "partition",
            "date", "timestamp", "comment", "function", "values", "position", "source", "split"}


def test_films_are_the_kermany_test_split():
    films = C.films(CFG)
    assert len(films) == 624
    assert sum(f["label"] == "PNEUMONIA" for f in films) == 390


def test_adult_films_are_nih_patients_the_pneumothorax_model_never_saw():
    import csv

    adults = C.adult_films(CFG)
    assert len(adults) == 300 and sum(f["label"] == "PNEUMOTHORAX" for f in adults) == 60
    assert all(18 <= int(f["age"]) <= 95 and f["image_file"].endswith(".jpg") for f in adults)
    with open(ROOT / "data_refs" / "nih_cxr14_subset.csv") as f:
        model_patients = {r["patient_id"] for r in csv.DictReader(f) if r["split"] != "lakehouse"}
    assert not {f["patient_id"] for f in adults} & model_patients


def test_the_hospital_day_is_deterministic_and_never_repeats_a_film():
    pools = C.pools(CFG)
    a, b = C.studies_of_day(CFG, D, pools), C.studies_of_day(CFG, D, pools)
    assert a == b and len(a) == CFG["studies_per_day"]
    assert sum(s["population"] == "adult" for s in a) == CFG["adult_per_day"]
    assert all(s["age"] >= 18 for s in a if s["population"] == "adult")
    seen = [s["image_file"] for d in CFG["demo_dates"] for s in C.studies_of_day(CFG, date.fromisoformat(d), pools)]
    assert len(seen) == len(set(seen))


def test_planted_films_are_unfit_copies_of_real_ones():
    d, slots = next(iter(CFG["planted_films"].items()))
    studies = C.studies_of_day(CFG, date.fromisoformat(d), C.pools(CFG))
    planted = [s for s in studies if s["planted"]]
    assert sorted(s["planted"] for s in planted) == sorted(p["kind"] for p in slots)
    assert all(s["image_file"].startswith(f"QC-{s['planted'].upper()}-") for s in planted)


def test_read_queue_fifo_and_triage():
    t0 = datetime(2026, 9, 28, 8)
    items = [{"accession_no": f"A{i}", "study_ts": t0, "priority": p, "probability": q}
             for i, (p, q) in enumerate([("P3", 0.1), ("P1", 0.99), ("P2", 0.6), (None, None)])]
    fifo = C.read_queue(items, C.fifo_rank, 1, t0, 10)
    tri = C.read_queue(items, C.triage_rank, 1, t0, 10)
    assert [a for a in sorted(fifo, key=lambda a: fifo[a][0])] == ["A0", "A1", "A2", "A3"]
    assert [a for a in sorted(tri, key=lambda a: tri[a][0])] == ["A1", "A2", "A0", "A3"]   # unscored last
    assert tri["A1"][1] == t0.replace(minute=10)


def test_triage_rank_reads_na_with_p3_in_arrival_order():
    t0 = datetime(2026, 9, 28, 8)
    items = [{"accession_no": f"A{i}", "study_ts": t0.replace(hour=7, minute=i), "priority": p, "probability": q}
             for i, (p, q) in enumerate([("P3", 0.1), ("NA", None), ("P2", 0.6), ("P3", 0.2), (None, None)])]
    tri = C.read_queue(items, C.triage_rank, 1, t0, 10)   # all waiting when the shift starts
    assert sorted(tri, key=lambda a: tri[a][0]) == ["A2", "A0", "A1", "A3", "A4"]


def test_triage_shortens_pneumonia_waits_with_a_perfect_ranking():
    studies = C.studies_of_day(CFG, D, C.pools(CFG))
    for s in studies:
        s["priority"] = "P1" if s["label"] == "PNEUMONIA" else "P3"
    fifo, tri = C.reading(CFG, D, studies), C.reading(CFG, D, studies, C.triage_rank)
    wait = lambda q: sum((q[s["accession_no"]][1] - s["study_ts"]).seconds for s in studies if s["label"] == "PNEUMONIA")  # noqa: E731
    assert wait(tri) < wait(fifo)


def test_contracts_catch_the_planted_faults():
    ris = CFG["contracts"]["ris_order"]
    ok = {"accession_no": "A", "mrn": "M", "order_ts": "2026-10-01 09:00:00", "ordering_unit": "ED",
          "clinical_priority": "STAT", "procedure_code": "XR"}
    assert C.validate(ok, ris) == []
    assert C.validate({**ok, "order_ts": "2026-10-01 25:61:00"}, ris) == ["order_ts: not a timestamp ('2026-10-01 25:61:00')"]
    assert C.validate({**ok, "ordering_unit": "ER"}, ris)[0].startswith("ordering_unit")
    rows, trailer = C.parse_source("psv", "a|b\n1|2\n3\nT|2\n")
    assert trailer == 2 and rows[1][2] == "1 fields, header has 2"


def test_landed_day_has_the_planted_defects_only_on_defects_on():
    import land_sources as L

    clean = L.build_day(CFG, D)
    assert clean["pacs"][2] == clean["ris"][2] == CFG["studies_per_day"]
    bad = L.build_day(CFG, date.fromisoformat(CFG["defects_on"]))
    assert bad["pacs"][2] == CFG["studies_per_day"] + 2
    assert '"AccessionNumber":""' in bad["pacs"][1]
    assert "25:61:00" in bad["ris"][1]


def test_gold_and_cai_columns_avoid_impala_reserved_words():
    from lakehouse.store import TABLES

    cols = {c for cols, _ in TABLES.values() for c, _ in cols}
    for job in ("build_silver.py", "build_gold.py", "build_outcomes.py"):
        text = (ROOT / "cde" / "jobs" / job).read_text()
        cols |= set(re.findall(r"\bAS (\w+)", text)) - {"date", "int", "double", "string", "timestamp_ntz", "bigint", "boolean"}
        cols |= set(re.findall(r"(\w+) (?:string|int|double|date|boolean|timestamp_ntz|bigint)\b", text))
    assert not cols & RESERVED, cols & RESERVED
    assert not {c for c in cols if c in ("_file", "_pos", "_partition", "_spec_id", "_deleted")}


class MemoryStore:
    """The lakehouse.store interface over dicts."""
    engine = "memory"

    def __init__(self, studies):
        self.prefix, self.tables = "t", {"gold.fact_study": studies}

    def t(self, key):
        return key

    def refresh(self, key):
        pass

    def snapshot_id(self, key):
        return 42

    def source(self, key, snap):
        return key

    def query(self, sql):
        d = re.search(r"DATE '([\d-]+)'", sql).group(1)
        return [s for s in self.tables["gold.fact_study"] if str(s["business_date"]) == d]

    def replace_date(self, key, rows, d, column="business_date"):
        self.tables[key] = [r for r in self.tables.get(key, []) if r[column] != d] + rows

    def append(self, key, rows):
        self.tables.setdefault(key, []).extend(rows)


def test_score_studies_records_missing_films_and_fails():
    from lakehouse.score_studies import score_date

    meta = {"feature_version": "1.0.0", "git_sha": "abcdef0123", "threshold": 0.5}
    studies = [{"business_date": D, "accession_no": f"A{i}", "image_file": f"f{i}.jpeg"} for i in range(3)]
    store = MemoryStore(studies)
    scorer = (lambda paths, ages: ([{"probability_pneumonia": 0.9, "priority": "P2", "threshold": 0.5,
                                     "quality_flags": []} for _ in paths], []), meta)
    films = {"f0.jpeg": Path("f0.jpeg"), "f1.jpeg": Path("f1.jpeg")}
    run = score_date(store, D, "test", scorer=scorer, drift_fn=lambda q: ({}, "UNAVAILABLE"), films=films)
    assert run["status"] == "PARTIAL" and run["missing_films"] == 1 and run["scored"] == 2
    assert run["source_snapshot_id"] == 42 and run["model_version"] == "fv1.0.0-abcdef0"
    assert {r["accession_no"] for r in store.tables["gold.triage_score"]} == {"A0", "A1"}
    films["f2.jpeg"] = Path("f2.jpeg")
    assert score_date(store, D, "test", scorer=scorer, drift_fn=lambda q: ({}, "OK"), films=films)["status"] == "SUCCEEDED"
    assert len(store.tables["gold.triage_score"]) == 3        # the date was replaced, not appended


def _head(p, stage, scope=True, thr=0.5, ver="v1"):
    band = "P1" if p >= 0.9 else "P2" if p >= thr else "P3"
    return {"probability": p, "threshold": thr, "positive": p >= thr, "priority": band if scope else None,
            "stage": stage, "in_scope": scope, "model_version": ver}


def test_score_studies_writes_every_head_and_the_shadow_worklist():
    from lakehouse.score_studies import score_date

    meta = {"feature_version": "1.0.0", "git_sha": "abcdef0123", "threshold": 0.5}
    studies = [{"business_date": D, "accession_no": f"A{i}", "image_file": f"f{i}.jpg", "age_years": a}
               for i, a in enumerate([4, 60, 70])]
    qc = lambda bad: {"probability": 0.99 if bad else 0.01, "unsuitable": bad, "model_version": "qc1"}  # noqa: E731
    results = [   # child: pneumonia live; adult: no live adult model yet, pneumothorax in trial; adult unfit film
        {"priority": "P2", "triage_model": "pneumonia", "probability_pneumonia": 0.6, "threshold": 0.5,
         "film_qc": qc(False), "quality_flags": [],
         "findings": {"pneumonia": _head(0.6, "champion")},
         "silent_trial": {"pneumothorax": _head(0.0, "silent_trial", scope=False)}},
        {"priority": "NA", "triage_model": None, "probability_pneumonia": 0.2, "threshold": 0.5,
         "film_qc": qc(False), "quality_flags": [],
         "findings": {"pneumonia": _head(0.2, "champion", scope=False)},
         "silent_trial": {"pneumothorax": _head(0.95, "silent_trial")}},
        {"priority": "NA", "triage_model": None, "probability_pneumonia": 0.1, "threshold": 0.5,
         "film_qc": qc(True), "quality_flags": ["dark"],
         "findings": {"pneumonia": _head(0.1, "champion", scope=False)},
         "silent_trial": {"pneumothorax": _head(0.99, "silent_trial")}}]
    store = MemoryStore(studies)
    films = {s["image_file"]: Path(s["image_file"]) for s in studies}
    run = score_date(store, D, "test", scorer=(lambda paths, ages: (results, []), meta),
                     drift_fn=lambda q: ({}, "TOO_FEW"), films=films)
    rows = {r["accession_no"]: r for r in store.tables["gold.triage_score"]}
    assert [rows[a]["priority"] for a in ("A0", "A1", "A2")] == ["P2", "NA", "NA"]
    assert [rows[a]["shadow_priority"] for a in ("A0", "A1", "A2")] == ["P2", "P1", "NA"]   # unfit film stays NA
    assert rows["A1"]["shadow_model"] == "pneumothorax" and rows["A2"]["film_qc"] == "UNSUITABLE"
    heads = store.tables["gold.model_score"]
    assert len(heads) == 9 and {h["model"] for h in heads} == {"pneumonia", "pneumothorax", "film_qc"}
    assert run["not_triaged"] == 2 and run["film_unsuitable"] == 1
    assert json.loads(run["heads_json"]) == {"film_qc": {"champion": "qc1"}, "pneumonia": {"champion": "v1"},
                                             "pneumothorax": {"silent_trial": "v1"}}


def test_publish_is_a_no_op_without_impala_credentials(monkeypatch):
    from lakehouse.publish import publish_model_event, training_set_rows

    monkeypatch.delenv("CXR_IMPALA_USER", raising=False)
    meta = {"git_sha": "abcdef0", "feature_version": "1.0.0", "feature_hash": "h", "metrics": {}}
    assert publish_model_event("GATE_PASSED", meta) is False
    rows = training_set_rows({"feature_version": "1", "feature_hash": "h", "backbone": "b", "git_sha": "abc1234",
                              "created_at": "2026-10-07T00:00:00+00:00", "rows": 10, "patients": 7,
                              "rows_by_split": {"train": 8, "val": 2}, "positives_by_split": {"train": 5, "val": 1}})
    assert [(r["data_split"], r["films"], r["pneumonia"]) for r in rows] == [("train", 8, 5), ("val", 2, 1), ("all", 10, 6)]


def _dag_job_names() -> set[str]:
    text = (ROOT / "cde" / "dags" / "cxr_dag.py").read_text()
    prefix = re.search(r'JOB_PREFIX = "([\w-]+)"', text).group(1)
    return {f"{prefix}-{j}" for j in re.findall(r'cde_task\("\w+", "([\w-]+)"', text)}


def test_dag_jobs_are_the_deployed_jobs():
    deploy = (ROOT / "cde" / "scripts" / "deploy_jobs.sh").read_text()
    prefix = re.search(r'JOB_PREFIX="\$\{JOB_PREFIX:-([\w-]+)\}"', deploy).group(1)
    deployed = {f"{prefix}-{j}" for j in re.findall(r'create_job "\$\{JOB_PREFIX\}-([\w-]+)"', deploy)}
    assert _dag_job_names() == deployed
    for f in re.findall(r'"(cde/jobs/\w+\.py)"', deploy):
        assert (ROOT / f).exists()
        assert f in deploy.split("FILES=(")[1].split(")")[0]


def test_dag_parses_and_is_paused_on_creation():
    tree = ast.parse((ROOT / "cde" / "dags" / "cxr_dag.py").read_text())
    kw = {k.arg: k.value for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "DAG"
          for k in n.keywords}
    assert ast.literal_eval(kw["is_paused_upon_creation"]) is True
    assert ast.literal_eval(kw["max_active_runs"]) == 1


def test_score_job_is_defined_for_the_dag():
    from ci.cai_jobs import JOBS, SCORE_JOB

    job = next(j for j in JOBS if j["name"] == SCORE_JOB)
    assert job["script"] == "lakehouse/score_studies.py" and job["parent"] is None
    assert (ROOT / job["script"]).exists()


def test_dashboard_pks_and_columns_are_consistent():
    sys.path.insert(0, str(ROOT / "dataviz"))
    import build_dashboard as B

    views = "".join(p.read_text() for p in (ROOT / "sql" / "semantic").glob("*.sql"))
    for _, view, _ in B.DATASETS.values():
        assert f"rsingh_cxr_semantic.{view}\n" in views
    n_visuals = sum(1 for d in B.DASHBOARDS for _ in B.visuals_of(d))
    assert B.DATASET_PK0 + len(B.DATASETS) <= B.VISUAL_PK0 and B.DASHBOARD_PK0 + len(B.DASHBOARDS) <= B.DATASET_PK0
    assert n_visuals < 100 * len(B.DASHBOARDS)
    json.dumps(B.DASHBOARDS)
