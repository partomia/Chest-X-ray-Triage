"""The models beside pneumonia: per-model config, film degradations, registry versions,
silent-trial promotion criteria, lakehouse schema migration and per-model outcomes."""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cde" / "jobs"))

from common import in_scope, load_config, model_names  # noqa: E402

D = date(2026, 9, 28)


def test_every_model_resolves_with_its_own_data_and_serving_dirs():
    cfg = load_config()
    assert model_names(cfg) == ["pneumonia", "pneumothorax", "film_qc"]
    dirs = set()
    for name in model_names(cfg):
        m = load_config(model=name)
        assert m["model"]["name"] == name and m["model"]["stage_on_pass"] in ("champion", "silent_trial")
        assert m["features"]["backbone"] == cfg["features"]["backbone"]   # one embedding, many heads
        dirs |= {m["serving"]["champion_dir"], m["features"]["store_dir"]}
    assert len(dirs) == 2 * len(model_names(cfg))                         # nothing shared on disk
    assert load_config(model="pneumothorax")["model"]["go_live"]["min_positives"] > 0
    assert load_config(model="pneumonia")["model"]["registry_name"] == "cxr-pneumonia"


def test_intended_use_by_age():
    assert in_scope("pediatric", 4) and not in_scope("pediatric", 40)
    assert in_scope("adult", 18) and not in_scope("adult", 17)
    assert in_scope("adult", None) and in_scope("all", 0)


def test_degradations_are_deterministic_and_change_the_film():
    from features.degrade import KINDS, degrade, kind_for

    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((256, 256)) * 120 + 60).astype(np.uint8), "L")
    base = np.asarray(img, dtype=float)
    for kind in KINDS:
        a, b = degrade(img, kind, 7, "f.jpg"), degrade(img, kind, 7, "f.jpg")
        assert np.array_equal(np.asarray(a), np.asarray(b)), kind
        changed = np.asarray(a.convert("L").resize(img.size), dtype=float)
        assert np.abs(changed - base).mean() > 3, kind
    assert kind_for("f.jpg", 1) == kind_for("f.jpg", 1) and kind_for("f.jpg", 1) in KINDS
    assert len({kind_for(f"f{i}.jpg", 1) for i in range(200)}) == len(KINDS)


META = {"model": "pneumothorax", "finding": "PNEUMOTHORAX", "population": "adult", "git_sha": "abcdef0123",
        "feature_version": "1.0.0", "feature_hash": "h1", "backbone": "vit", "threshold": 0.31, "train_rows": 4000,
        "mlflow_run_id": "run-1", "metrics": {"test": {"auroc": 0.85, "sensitivity": 0.9, "specificity": 0.55}}}


def test_registry_versions_carry_the_audit_tags():
    from serve.registry import register_version, version_tags

    tags = {t["key"]: t["value"] for t in version_tags(META, "silent_trial", {"approved_by": None})}
    assert tags["stage"] == "silent_trial" and tags["model_version"] == "pneumothorax-fv1.0.0-abcdef0"
    assert tags["population"] == "adult" and tags["test_auroc"] == "0.8500" and "approved_by" not in tags

    class Client:
        def __init__(self):
            self.versions, self.calls = [{"number": 1, "model_version_id": "v1"}], []

        def list_registered_models(self, page_size):
            return {"models": [{"name": "cxr-pneumothorax", "model_id": "m1"}]}

        def get_registered_model(self, mid):
            return {"model_versions": list(self.versions)}

        def create_registered_model(self, body):
            self.calls.append(body)
            self.versions.append({"number": 2, "model_version_id": "v2"})
            return {"model_id": "m1"}

    c = Client()
    out = register_version(c, META, "cxr-pneumothorax", "champion", "exp1", project_id="p1", extra={"approved_by": "Dr A"})
    assert out == {"model_id": "m1", "version": "v2", "number": 2, "previous": 1}
    body = c.calls[0]
    assert body["run_id"] == "run-1" and body["experiment_id"] == "exp1" and body["model_path"] == "model"
    assert {"key": "approved_by", "value": "Dr A"} in body["tags"]


def test_registrar_never_fails_the_job():
    from serve.registry import registrar

    assert registrar({"registry": {"enabled": False}}, META, "champion") is None
    cfg = {"registry": {"enabled": True}, "model": {"registry_name": "x"}}
    assert registrar(cfg, {**META, "mlflow_run_id": "missing"}, "champion") is None   # no cmlapi here


def test_promotion_needs_enough_evidence():
    from serve.promote_champion import criteria, trial_evidence

    class Store:
        def t(self, key):
            return key

        def query(self, sql):
            assert "stage = 'silent_trial'" in sql and "model_version = 'v'" in sql
            return [{"business_date": D + timedelta(days=i), "tp": 3, "fn": 0, "fp": 4, "tn": 9} for i in range(8)]

    ev = trial_evidence(Store(), "pneumothorax", "v")
    assert ev == {"days": 8, "positives": 24, "negatives": 104, "tp": 24, "fn": 0, "fp": 32, "tn": 72,
                  "sensitivity": 1.0, "specificity": 0.6923}
    g = load_config(model="pneumothorax")["model"]["go_live"]
    assert all(c["passed"] for c in criteria(ev, g))
    short = criteria({**ev, "days": 3}, g)
    assert [c["check"] for c in short if not c["passed"]] == ["days in silent trial"]
    none = criteria({**ev, "positives": 0, "sensitivity": None}, g)
    assert {c["check"] for c in none if not c["passed"]} == {"reported positives", "sensitivity (live studies)"}


def test_store_adds_the_columns_a_table_has_gained():
    from lakehouse.store import TABLES, _Store

    class Fake(_Store):
        engine = "fake"

        def __init__(self):
            super().__init__("t")
            self.sql = []

        def query(self, sql):
            return [{"name": "business_date"}, {"name": "accession_no"}, {"name": "# Partitioning"},
                    {"name": "model"}]

        def execute(self, sql):
            self.sql.append(sql)

        @staticmethod
        def create_sql(table, cols, part):
            return f"CREATE {table}"

    s = Fake()
    s.ensure("gold.model_score")
    alter = [q for q in s.sql if q.startswith("ALTER")]
    cols = [c for c, _ in TABLES["gold.model_score"][0] if c not in ("business_date", "accession_no")]
    assert len(alter) == 1 and all(f"`{c}`" in alter[0] for c in cols)   # 'model' was below the partition header
    s.ensure("gold.model_score")
    assert len([q for q in s.sql if q.startswith("ALTER")]) == 1          # once per store


def _study(acc, image="f.jpg", age=None):
    return {"accession_no": acc, "study_ts": datetime(2026, 9, 28, 7, int(acc[-1])), "ordering_unit": "ED",
            "clinical_priority": "STAT", "fifo_seq": int(acc[-1]) + 1, "image_file": image, "age_years": age}


def test_outcomes_judge_the_model_that_triaged_and_count_not_triaged():
    import build_outcomes as O
    import cxr_common as C

    cfg = C.config()
    studies = [_study("A0", age=4), _study("A1", age=60), _study("A2", "QC-BLUR-x.jpg", age=5)]
    scores = {"A0": {"priority": "P1", "probability": 0.95, "triage_model": "pneumonia", "film_qc": "OK"},
              "A1": {"priority": "NA", "probability": None, "triage_model": None, "film_qc": "OK",
                     "shadow_priority": "P1", "shadow_probability": 0.97, "shadow_model": "pneumothorax"},
              "A2": {"priority": "NA", "probability": None, "triage_model": None, "film_qc": "UNSUITABLE"}}
    rep = lambda f: {"truth": int(f == "PNEUMONIA"), "truth_pneumothorax": int(f == "PNEUMOTHORAX"),  # noqa: E731
                     "finding": f, "report_ts": datetime(2026, 9, 28, 12)}
    truth = {"A0": rep("PNEUMONIA"), "A1": rep("PNEUMOTHORAX"), "A2": rep("NORMAL")}
    rows = {r["accession_no"]: r for r in O.outcomes_of_day(cfg, D, studies, scores, truth)}
    assert [rows[a]["outcome"] for a in ("A0", "A1", "A2")] == ["TP", "NOT_TRIAGED", "NOT_TRIAGED"]
    assert rows["A1"]["population"] == "adult" and rows["A1"]["shadow_wait_min"] < rows["A1"]["triage_wait_min"]
    s = O.summary_of_day(D, list(rows.values()), datetime.now())
    assert (s["tp"], s["fn"], s["not_triaged"], s["film_unsuitable"], s["pneumothorax"]) == (1, 0, 2, 1, 1)


def test_daily_model_summary_uses_each_models_own_truth():
    import build_outcomes as O

    head = lambda acc, model, stage, pos, scope=True: {  # noqa: E731
        "accession_no": acc, "model": model, "stage": stage, "model_version": f"{model}-v", "in_scope": scope,
        "positive": pos}
    heads = [head("A0", "pneumonia", "champion", True), head("A1", "pneumonia", "champion", False, scope=False),
             head("A1", "pneumothorax", "silent_trial", True), head("A0", "pneumothorax", "silent_trial", False, False),
             head("A0", "film_qc", "champion", False), head("A2", "film_qc", "champion", True)]
    studies = {"A0": {"image_file": "a.jpg"}, "A1": {"image_file": "b.jpg"}, "A2": {"image_file": "QC-BLUR-c.jpg"}}
    truth = {"A0": {"finding": "PNEUMONIA"}, "A1": {"finding": "PNEUMOTHORAX"}}
    out = {m["model"]: m for m in O.model_summaries(D, heads, studies, truth, datetime.now())}
    assert (out["pneumonia"]["in_scope"], out["pneumonia"]["tp"], out["pneumonia"]["scored"]) == (1, 1, 2)
    assert (out["pneumothorax"]["stage"], out["pneumothorax"]["tp"], out["pneumothorax"]["in_scope"]) == \
        ("silent_trial", 1, 1)
    assert (out["film_qc"]["tp"], out["film_qc"]["tn"], out["film_qc"]["reported"]) == (1, 1, 2)   # no report needed
