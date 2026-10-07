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


def test_the_hospital_day_is_deterministic_and_never_repeats_a_film():
    pool = C.films(CFG)
    a, b = C.studies_of_day(CFG, D, pool), C.studies_of_day(CFG, D, pool)
    assert a == b and len(a) == CFG["studies_per_day"]
    seen = [s["image_file"] for d in CFG["demo_dates"] for s in C.studies_of_day(CFG, date.fromisoformat(d), pool)]
    assert len(seen) == len(set(seen))


def test_read_queue_fifo_and_triage():
    t0 = datetime(2026, 9, 28, 8)
    items = [{"accession_no": f"A{i}", "study_ts": t0, "priority": p, "probability": q}
             for i, (p, q) in enumerate([("P3", 0.1), ("P1", 0.99), ("P2", 0.6), (None, None)])]
    fifo = C.read_queue(items, C.fifo_rank, 1, t0, 10)
    tri = C.read_queue(items, C.triage_rank, 1, t0, 10)
    assert [a for a in sorted(fifo, key=lambda a: fifo[a][0])] == ["A0", "A1", "A2", "A3"]
    assert [a for a in sorted(tri, key=lambda a: tri[a][0])] == ["A1", "A2", "A0", "A3"]   # unscored last
    assert tri["A1"][1] == t0.replace(minute=10)


def test_triage_shortens_pneumonia_waits_with_a_perfect_ranking():
    pool = C.films(CFG)
    studies = C.studies_of_day(CFG, D, pool)
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
    scorer = (lambda paths: ([{"probability_pneumonia": 0.9, "priority": "P2", "threshold": 0.5,
                               "quality_flags": []} for _ in paths], []), meta)
    films = {"f0.jpeg": Path("f0.jpeg"), "f1.jpeg": Path("f1.jpeg")}
    run = score_date(store, D, "test", scorer=scorer, drift_fn=lambda q: ({}, "UNAVAILABLE"), films=films)
    assert run["status"] == "PARTIAL" and run["missing_films"] == 1 and run["scored"] == 2
    assert run["source_snapshot_id"] == 42 and run["model_version"] == "fv1.0.0-abcdef0"
    assert {r["accession_no"] for r in store.tables["gold.triage_score"]} == {"A0", "A1"}
    films["f2.jpeg"] = Path("f2.jpeg")
    assert score_date(store, D, "test", scorer=scorer, drift_fn=lambda q: ({}, "OK"), films=films)["status"] == "SUCCEEDED"
    assert len(store.tables["gold.triage_score"]) == 3        # the date was replaced, not appended


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
