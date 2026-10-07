"""
Stage 5 - Outcomes, after CAI job cxr-06-score-studies wrote gold.triage_score and
gold.model_score for the date.

For the business date AND the day before (whose reports may have been signed since), per study:
  fifo_*    when the radiologist would report it reading first in, first out (today's practice)
  triage_*  when it would be reported reading the live worklist (P1, then P2, higher probability
            first; then P3 and NA - no AI triage - in arrival order; unscored studies last)
  shadow_*  the same if every model in silent trial were live: what going live would change
All arms replay the same reading model (cxr_common.reading: readers, shift start, minutes per
film), so the differences are the ordering only. Truth is the signed report (gold.fact_report);
studies not yet reported are PENDING and excluded from sensitivity and specificity.

  gold.fact_triage_outcome     one row per study; partition business_date. outcome compares the
                               worklist flag (P1/P2) with the report for the finding of the model
                               that triaged the study; NOT_TRIAGED when no live model did (out of
                               every live model's intended use, or the film check rejected it)
  gold.daily_triage_summary    one row per business date: volumes, P1/P2/P3/NA, worklist confusion
                               counts, sensitivity/specificity on reported studies, median and p90
                               waits for pneumonia, pneumothorax and normal films in each arm
  gold.daily_model_summary     one row per business date, model, stage and version: in-scope
                               studies, reported positives, confusion counts at the model's own
                               threshold. For a silent trial this is the evidence
                               cxr-07-promote-champion weighs. film_qc truth = the planted films.

Reconciliation (stage outcomes): outcome rows = gold studies; scored studies = gold studies
(unscored studies are a MISMATCH unless --allow-unscored, which the local runner uses).

Usage:
  spark-submit build_outcomes.py --business-date 2026-09-28 [--allow-unscored]
"""
from __future__ import annotations

import statistics
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cxr_common as C  # noqa: E402

STAGE = "outcomes"
OUTCOME_SCHEMA = (
    "business_date date, accession_no string, study_ts timestamp_ntz, ordering_unit string, "
    "clinical_priority string, fifo_seq int, triage_seq int, probability double, priority string, "
    "model_version string, scored boolean, truth int, finding string, report_ts timestamp_ntz, "
    "truth_known boolean, flagged boolean, outcome string, fifo_report_ts timestamp_ntz, "
    "triage_report_ts timestamp_ntz, fifo_wait_min double, triage_wait_min double, wait_saved_min double, "
    "population string, age_years int, triage_model string, film_qc string, truth_pneumothorax int, "
    "shadow_priority string, shadow_model string, shadow_wait_min double")
SUMMARY_SCHEMA = (
    "business_date date, studies int, scored int, reported int, truth_completeness double, pneumonia int, "
    "p1 int, p2 int, p3 int, unscored int, tp int, fn int, fp int, tn int, sensitivity double, "
    "specificity double, ppv double, pneumonia_fifo_median_min double, pneumonia_triage_median_min double, "
    "pneumonia_fifo_p90_min double, pneumonia_triage_p90_min double, normal_fifo_median_min double, "
    "normal_triage_median_min double, pneumonia_minutes_saved double, model_version string, "
    "refreshed_at timestamp_ntz, adults int, not_triaged int, film_unsuitable int, pneumothorax int, "
    "pneumothorax_fifo_median_min double, pneumothorax_triage_median_min double, "
    "pneumothorax_shadow_median_min double, pneumonia_shadow_median_min double")
MODEL_SUMMARY_SCHEMA = (
    "business_date date, model string, stage string, model_version string, scored int, in_scope int, "
    "reported int, positives int, tp int, fn int, fp int, tn int, sensitivity double, specificity double, "
    "ppv double, refreshed_at timestamp_ntz")
# the report finding each model detects; film_qc's truth is a planted film (file name QC-...)
FINDING_OF = {"pneumonia": "PNEUMONIA", "pneumothorax": "PNEUMOTHORAX"}
ADULT_AGE = 18


def _minutes(a, b) -> float:
    return round((a - b).total_seconds() / 60.0, 1)


def _pct(values: list[float], q: float):
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * len(v)))], 1)


def _med(values: list[float]):
    return round(statistics.median(values), 1) if values else None


def _ratio(a: int, b: int):
    return round(a / b, 4) if b else None


def _confusion(pairs) -> dict:
    """pairs of (truth 0/1, flagged bool) -> tp, fn, fp, tn, sensitivity, specificity, ppv."""
    c = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for t, f in pairs:
        c[{(1, True): "tp", (1, False): "fn", (0, True): "fp", (0, False): "tn"}[(int(t), bool(f))]] += 1
    return {**c, "sensitivity": _ratio(c["tp"], c["tp"] + c["fn"]), "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
            "ppv": _ratio(c["tp"], c["tp"] + c["fp"])}


def truth_for(model: str, report: dict | None, image_file: str):
    if model == "film_qc":
        return int(str(image_file).startswith("QC-"))
    if report is None or model not in FINDING_OF:
        return None
    return int(report["finding"] == FINDING_OF[model])


def outcomes_of_day(cfg: dict, d, studies: list[dict], scores: dict, truth: dict) -> list[dict]:
    items = []
    for s in studies:
        sc = scores.get(s["accession_no"]) or {}
        items.append({**s, "priority": sc.get("priority"), "probability": sc.get("probability"),
                      "model_version": sc.get("model_version"), "triage_model": sc.get("triage_model"),
                      "film_qc": sc.get("film_qc"), "_shadow": {"priority": sc.get("shadow_priority"),
                                                                "probability": sc.get("shadow_probability")},
                      "shadow_model": sc.get("shadow_model")})
    fifo = C.reading(cfg, d, items, C.fifo_rank)
    tri = C.reading(cfg, d, items, C.triage_rank)
    shadow_items = [{**s, **({k: v for k, v in s["_shadow"].items()} if s["_shadow"]["priority"] else {})}
                    for s in items]
    sha = C.reading(cfg, d, shadow_items, C.triage_rank)
    order = {a: i + 1 for i, a in enumerate(sorted(tri, key=lambda a: tri[a][0]))}
    out = []
    for s in items:
        a = s["accession_no"]
        t = truth.get(a)
        triaged_by = s["triage_model"] or ("pneumonia" if s["priority"] in ("P1", "P2", "P3") else None)
        flagged = s["priority"] in ("P1", "P2") if s["priority"] else None
        if s["priority"] is None:
            outcome = "UNSCORED"
        elif t is None:
            outcome = "PENDING"
        elif triaged_by is None:
            outcome = "NOT_TRIAGED"
        else:
            hit = truth_for(triaged_by, t, s["image_file"])
            outcome = {(1, True): "TP", (1, False): "FN", (0, True): "FP", (0, False): "TN"}[(hit, flagged)]
        fw, tw = _minutes(fifo[a][1], s["study_ts"]), _minutes(tri[a][1], s["study_ts"])
        age = s.get("age_years")
        out.append({"business_date": d, "accession_no": a, "study_ts": s["study_ts"],
                    "ordering_unit": s["ordering_unit"], "clinical_priority": s["clinical_priority"],
                    "fifo_seq": s["fifo_seq"], "triage_seq": order[a], "probability": s["probability"],
                    "priority": s["priority"], "model_version": s["model_version"],
                    "scored": s["priority"] is not None, "truth": t["truth"] if t else None,
                    "finding": t["finding"] if t else None, "report_ts": t["report_ts"] if t else None,
                    "truth_known": t is not None, "flagged": flagged, "outcome": outcome,
                    "fifo_report_ts": fifo[a][1], "triage_report_ts": tri[a][1],
                    "fifo_wait_min": fw, "triage_wait_min": tw, "wait_saved_min": round(fw - tw, 1),
                    "population": None if age is None else "adult" if age >= ADULT_AGE else "pediatric",
                    "age_years": age, "triage_model": triaged_by, "film_qc": s["film_qc"],
                    "truth_pneumothorax": t.get("truth_pneumothorax") if t else None,
                    "shadow_priority": s["_shadow"]["priority"], "shadow_model": s["shadow_model"],
                    "shadow_wait_min": _minutes(sha[a][1], s["study_ts"]), "image_file": s["image_file"]})
    return out


def summary_of_day(d, rows: list[dict], now) -> dict:
    count = lambda pred: sum(1 for r in rows if pred(r))  # noqa: E731
    pos = [r for r in rows if r["truth"] == 1]
    ptx = [r for r in rows if r.get("truth_pneumothorax") == 1]
    neg = [r for r in rows if r["finding"] == "NORMAL"]
    conf = _confusion((1 if r["outcome"] in ("TP", "FN") else 0, r["outcome"] in ("TP", "FP"))
                      for r in rows if r["outcome"] in ("TP", "FN", "FP", "TN"))
    versions = sorted({r["model_version"] for r in rows if r["model_version"]})
    return {"business_date": d, "studies": len(rows), "scored": count(lambda r: r["scored"]),
            "reported": count(lambda r: r["truth_known"]),
            "truth_completeness": _ratio(count(lambda r: r["truth_known"]), len(rows)),
            "pneumonia": len(pos), "p1": count(lambda r: r["priority"] == "P1"),
            "p2": count(lambda r: r["priority"] == "P2"), "p3": count(lambda r: r["priority"] == "P3"),
            "unscored": count(lambda r: not r["scored"]), **{k: conf[k] for k in ("tp", "fn", "fp", "tn")},
            "sensitivity": conf["sensitivity"], "specificity": conf["specificity"], "ppv": conf["ppv"],
            "pneumonia_fifo_median_min": _med([r["fifo_wait_min"] for r in pos]),
            "pneumonia_triage_median_min": _med([r["triage_wait_min"] for r in pos]),
            "pneumonia_fifo_p90_min": _pct([r["fifo_wait_min"] for r in pos], 0.9),
            "pneumonia_triage_p90_min": _pct([r["triage_wait_min"] for r in pos], 0.9),
            "normal_fifo_median_min": _med([r["fifo_wait_min"] for r in neg]),
            "normal_triage_median_min": _med([r["triage_wait_min"] for r in neg]),
            "pneumonia_minutes_saved": round(sum(r["wait_saved_min"] for r in pos), 1),
            "model_version": ",".join(versions) or None, "refreshed_at": now,
            "adults": count(lambda r: r["population"] == "adult"),
            "not_triaged": count(lambda r: r["priority"] == "NA"),
            "film_unsuitable": count(lambda r: r["film_qc"] == "UNSUITABLE"),
            "pneumothorax": len(ptx),
            "pneumothorax_fifo_median_min": _med([r["fifo_wait_min"] for r in ptx]),
            "pneumothorax_triage_median_min": _med([r["triage_wait_min"] for r in ptx]),
            "pneumothorax_shadow_median_min": _med([r["shadow_wait_min"] for r in ptx]),
            "pneumonia_shadow_median_min": _med([r["shadow_wait_min"] for r in pos])}


def model_summaries(d, heads: list[dict], studies: dict, truth: dict, now) -> list[dict]:
    """Per (model, stage, version) on the date: every head's own threshold vs the report."""
    groups: dict = {}
    for h in heads:
        groups.setdefault((h["model"], h["stage"], h["model_version"]), []).append(h)
    out = []
    for (model, stage, version), hs in sorted(groups.items()):
        scope = [h for h in hs if h["in_scope"]]
        pairs = []
        for h in scope:
            t = truth_for(model, truth.get(h["accession_no"]), studies.get(h["accession_no"], {}).get("image_file", ""))
            if t is not None:
                pairs.append((t, h["positive"]))
        conf = _confusion(pairs)
        out.append({"business_date": d, "model": model, "stage": stage, "model_version": version,
                    "scored": len(hs), "in_scope": len(scope), "reported": len(pairs),
                    "positives": sum(t for t, _ in pairs), **conf, "refreshed_at": now})
    return out


def latest(spark, table: str, in_days: str) -> list:
    return spark.sql(f"""SELECT * FROM (
        SELECT s.*, row_number() OVER (PARTITION BY accession_no ORDER BY scored_at DESC) AS _rn
        FROM {table} s WHERE business_date IN ({in_days})) WHERE _rn = 1""").collect()


def run(spark, argv=None) -> dict:
    p = C.base_parser(__doc__)
    p.add_argument("--allow-unscored", action="store_true")
    args = C.parse(p, argv)
    cfg, names, d = C.config(), C.Names(args.db_prefix), args.business_date
    C.ensure_databases(spark, names)
    audit = C.Audit(spark, names, STAGE, d, args.pipeline_run)
    fact, score_t = names.t("gold", "fact_study"), names.t("gold", "triage_score")
    report_t, heads_t = names.t("gold", "fact_report"), names.t("gold", "model_score")
    first = date.fromisoformat(cfg["first_date"])
    days = [x for x in (d - timedelta(days=1), d) if x >= first]
    in_days = ", ".join(f"DATE '{x.isoformat()}'" for x in days)

    studies = spark.sql(f"SELECT * FROM {fact} WHERE business_date IN ({in_days})").collect()
    scores = {r["accession_no"]: r.asDict() for r in latest(spark, score_t, in_days)} \
        if C.table_exists(spark, score_t) else {}
    heads = [r.asDict() for r in spark.sql(f"SELECT * FROM {heads_t} WHERE business_date IN ({in_days})").collect()] \
        if C.table_exists(spark, heads_t) else []
    truth = {r["accession_no"]: r.asDict() for r in spark.table(report_t).collect()} \
        if C.table_exists(spark, report_t) else {}

    now = datetime.now()
    outcome_rows, summaries, per_model = [], [], []
    for x in days:
        day_studies = [r.asDict() for r in studies if r["business_date"] == x]
        rows = outcomes_of_day(cfg, x, day_studies, scores, truth)
        outcome_rows += rows
        summaries.append(summary_of_day(x, rows, now))
        by_acc = {s["accession_no"]: s for s in day_studies}
        per_model += model_summaries(x, [h for h in heads if h["business_date"] == x], by_acc, truth, now)
        unscored = sum(1 for r in rows if not r["scored"])
        audit.check("fact_triage_outcome", f"scored studies = gold studies ({x})", len(rows), len(rows) - unscored,
                    f"{unscored} studies have no triage score", explained=args.allow_unscored)
        pending = sum(1 for r in rows if not r["truth_known"])
        if pending:
            audit.check("fact_triage_outcome", f"studies awaiting a report ({x})", 0, pending,
                        "truth arrives when the report is signed", explained=True)
        rejected = [r["accession_no"] for r in rows if r["film_qc"] == "UNSUITABLE"]
        if rejected:
            audit.check("fact_triage_outcome", f"films unfit for AI triage ({x})", 0, len(rejected),
                        "read without AI priority: " + ", ".join(rejected), explained=True)

    out = names.t("gold", "fact_triage_outcome")
    summary_t, model_t = names.t("gold", "daily_triage_summary"), names.t("gold", "daily_model_summary")
    if outcome_rows:
        C.write_partitions(C.frame(spark, outcome_rows, OUTCOME_SCHEMA), out)
    C.ensure_table(spark, summary_t, SUMMARY_SCHEMA, ["business_date"])
    C.write_partitions(C.frame(spark, summaries, SUMMARY_SCHEMA), summary_t)
    C.ensure_table(spark, model_t, MODEL_SUMMARY_SCHEMA, ["business_date"])
    if per_model:
        C.write_partitions(C.frame(spark, per_model, MODEL_SUMMARY_SCHEMA), model_t)
    audit.check("fact_triage_outcome", "outcome rows = gold studies", len(studies), len(outcome_rows))
    audit.load("fact_triage_outcome", "COMMITTED", rows_in=len(studies), rows_out=len(outcome_rows),
               table=out if outcome_rows else None)
    bad = audit.flush()
    audit.load("*", "FAILED" if bad else "COMPLETED", rows_out=len(outcome_rows))
    for s in summaries:
        print(f"outcomes {s['business_date']}: studies {s['studies']} (adults {s['adults']}) scored {s['scored']} "
              f"reported {s['reported']} sens {s['sensitivity']} spec {s['specificity']} | pneumonia median wait "
              f"FIFO {s['pneumonia_fifo_median_min']} -> triage {s['pneumonia_triage_median_min']} min | "
              f"pneumothorax FIFO {s['pneumothorax_fifo_median_min']} -> triage "
              f"{s['pneumothorax_triage_median_min']} / shadow {s['pneumothorax_shadow_median_min']} min", flush=True)
    for m in per_model:
        if m["business_date"] == d:
            print(f"  {m['model']} {m['stage']} {m['model_version']}: in scope {m['in_scope']} reported "
                  f"{m['reported']} positives {m['positives']} sens {m['sensitivity']} spec {m['specificity']}")
    if bad:
        raise RuntimeError(f"outcomes {d}: {len(bad)} reconciliation mismatch(es)")
    return {"outcomes": len(outcome_rows), "summaries": summaries, "models": per_model}


def main() -> int:
    spark = C.get_spark("cxr-build-outcomes")
    run(spark, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
