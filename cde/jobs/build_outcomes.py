"""
Stage 5 - Outcomes, after CAI job cxr-06-score-studies wrote gold.triage_score for the date.

For the business date AND the day before (whose reports may have been signed since), per study:
  fifo_*    when the radiologist would report it reading first in, first out (today's practice)
  triage_*  when it would be reported reading the model's worklist (P1, then P2, then P3;
            higher probability first; unscored studies last, in arrival order)
Both arms replay the same reading model (cxr_common.reading: readers, shift start, minutes per
film), so the difference is the ordering only. Truth is the signed report (gold.fact_report);
studies not yet reported are PENDING and excluded from sensitivity and specificity.

  gold.fact_triage_outcome     one row per study; partition business_date
  gold.daily_triage_summary    one row per business date: volumes, P1/P2/P3, confusion counts,
                               sensitivity/specificity on reported studies, median and p90 waits
                               for pneumonia and normal films in both arms, truth completeness

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
    "triage_report_ts timestamp_ntz, fifo_wait_min double, triage_wait_min double, wait_saved_min double")
SUMMARY_SCHEMA = (
    "business_date date, studies int, scored int, reported int, truth_completeness double, pneumonia int, "
    "p1 int, p2 int, p3 int, unscored int, tp int, fn int, fp int, tn int, sensitivity double, "
    "specificity double, ppv double, pneumonia_fifo_median_min double, pneumonia_triage_median_min double, "
    "pneumonia_fifo_p90_min double, pneumonia_triage_p90_min double, normal_fifo_median_min double, "
    "normal_triage_median_min double, pneumonia_minutes_saved double, model_version string, "
    "refreshed_at timestamp_ntz")


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


def outcomes_of_day(cfg: dict, d, studies: list[dict], scores: dict, truth: dict) -> list[dict]:
    items = []
    for s in studies:
        sc = scores.get(s["accession_no"])
        items.append({**s, "priority": sc["priority"] if sc else None,
                      "probability": sc["probability"] if sc else None,
                      "model_version": sc["model_version"] if sc else None})
    fifo = C.reading(cfg, d, items, C.fifo_rank)
    tri = C.reading(cfg, d, items, C.triage_rank)
    order = {a: i + 1 for i, a in enumerate(sorted(tri, key=lambda a: tri[a][0]))}
    out = []
    for s in items:
        a = s["accession_no"]
        t = truth.get(a)
        flagged = s["priority"] in ("P1", "P2") if s["priority"] else None
        if t is None or flagged is None:
            outcome = "PENDING" if t is None else "UNSCORED"
        else:
            outcome = {(1, True): "TP", (1, False): "FN", (0, True): "FP", (0, False): "TN"}[(t["truth"], flagged)]
        fw, tw = _minutes(fifo[a][1], s["study_ts"]), _minutes(tri[a][1], s["study_ts"])
        out.append({"business_date": d, "accession_no": a, "study_ts": s["study_ts"],
                    "ordering_unit": s["ordering_unit"], "clinical_priority": s["clinical_priority"],
                    "fifo_seq": s["fifo_seq"], "triage_seq": order[a], "probability": s["probability"],
                    "priority": s["priority"], "model_version": s["model_version"],
                    "scored": s["priority"] is not None, "truth": t["truth"] if t else None,
                    "finding": t["finding"] if t else None, "report_ts": t["report_ts"] if t else None,
                    "truth_known": t is not None, "flagged": flagged, "outcome": outcome,
                    "fifo_report_ts": fifo[a][1], "triage_report_ts": tri[a][1],
                    "fifo_wait_min": fw, "triage_wait_min": tw, "wait_saved_min": round(fw - tw, 1)})
    return out


def summary_of_day(d, rows: list[dict], now) -> dict:
    count = lambda pred: sum(1 for r in rows if pred(r))  # noqa: E731
    pos = [r for r in rows if r["truth"] == 1]
    neg = [r for r in rows if r["truth"] == 0]
    tp, fn = count(lambda r: r["outcome"] == "TP"), count(lambda r: r["outcome"] == "FN")
    fp, tn = count(lambda r: r["outcome"] == "FP"), count(lambda r: r["outcome"] == "TN")
    versions = sorted({r["model_version"] for r in rows if r["model_version"]})
    return {"business_date": d, "studies": len(rows), "scored": count(lambda r: r["scored"]),
            "reported": count(lambda r: r["truth_known"]),
            "truth_completeness": _ratio(count(lambda r: r["truth_known"]), len(rows)),
            "pneumonia": len(pos), "p1": count(lambda r: r["priority"] == "P1"),
            "p2": count(lambda r: r["priority"] == "P2"), "p3": count(lambda r: r["priority"] == "P3"),
            "unscored": count(lambda r: not r["scored"]), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp), "ppv": _ratio(tp, tp + fp),
            "pneumonia_fifo_median_min": _med([r["fifo_wait_min"] for r in pos]),
            "pneumonia_triage_median_min": _med([r["triage_wait_min"] for r in pos]),
            "pneumonia_fifo_p90_min": _pct([r["fifo_wait_min"] for r in pos], 0.9),
            "pneumonia_triage_p90_min": _pct([r["triage_wait_min"] for r in pos], 0.9),
            "normal_fifo_median_min": _med([r["fifo_wait_min"] for r in neg]),
            "normal_triage_median_min": _med([r["triage_wait_min"] for r in neg]),
            "pneumonia_minutes_saved": round(sum(r["wait_saved_min"] for r in pos), 1),
            "model_version": ",".join(versions) or None, "refreshed_at": now}


def run(spark, argv=None) -> dict:
    p = C.base_parser(__doc__)
    p.add_argument("--allow-unscored", action="store_true")
    args = C.parse(p, argv)
    cfg, names, d = C.config(), C.Names(args.db_prefix), args.business_date
    C.ensure_databases(spark, names)
    audit = C.Audit(spark, names, STAGE, d, args.pipeline_run)
    fact, score_t = names.t("gold", "fact_study"), names.t("gold", "triage_score")
    report_t = names.t("gold", "fact_report")
    first = date.fromisoformat(cfg["first_date"])
    days = [x for x in (d - timedelta(days=1), d) if x >= first]
    in_days = ", ".join(f"DATE '{x.isoformat()}'" for x in days)

    studies = spark.sql(f"SELECT * FROM {fact} WHERE business_date IN ({in_days})").collect()
    scores = {}
    if C.table_exists(spark, score_t):
        for r in spark.sql(f"""SELECT * FROM (
                SELECT s.*, row_number() OVER (PARTITION BY accession_no ORDER BY scored_at DESC) AS _rn
                FROM {score_t} s WHERE business_date IN ({in_days})) WHERE _rn = 1""").collect():
            scores[r["accession_no"]] = r.asDict()
    truth = {r["accession_no"]: r.asDict() for r in spark.table(report_t).collect()} \
        if C.table_exists(spark, report_t) else {}

    now = datetime.now()
    outcome_rows, summaries = [], []
    for x in days:
        day_studies = [r.asDict() for r in studies if r["business_date"] == x]
        rows = outcomes_of_day(cfg, x, day_studies, scores, truth)
        outcome_rows += rows
        summaries.append(summary_of_day(x, rows, now))
        unscored = sum(1 for r in rows if not r["scored"])
        audit.check("fact_triage_outcome", f"scored studies = gold studies ({x})", len(rows), len(rows) - unscored,
                    f"{unscored} studies have no triage score", explained=args.allow_unscored)
        pending = sum(1 for r in rows if not r["truth_known"])
        if pending:
            audit.check("fact_triage_outcome", f"studies awaiting a report ({x})", 0, pending,
                        "truth arrives when the report is signed", explained=True)

    out = names.t("gold", "fact_triage_outcome")
    summary_t = names.t("gold", "daily_triage_summary")
    if outcome_rows:
        C.write_partitions(C.frame(spark, outcome_rows, OUTCOME_SCHEMA), out)
    C.ensure_table(spark, summary_t, SUMMARY_SCHEMA, ["business_date"])
    C.write_partitions(C.frame(spark, summaries, SUMMARY_SCHEMA), summary_t)
    audit.check("fact_triage_outcome", "outcome rows = gold studies", len(studies), len(outcome_rows))
    audit.load("fact_triage_outcome", "COMMITTED", rows_in=len(studies), rows_out=len(outcome_rows),
               table=out if outcome_rows else None)
    bad = audit.flush()
    audit.load("*", "FAILED" if bad else "COMPLETED", rows_out=len(outcome_rows))
    for s in summaries:
        print(f"outcomes {s['business_date']}: studies {s['studies']} scored {s['scored']} reported "
              f"{s['reported']} sens {s['sensitivity']} spec {s['specificity']} pneumonia median wait "
              f"FIFO {s['pneumonia_fifo_median_min']} -> triage {s['pneumonia_triage_median_min']} min", flush=True)
    if bad:
        raise RuntimeError(f"outcomes {d}: {len(bad)} reconciliation mismatch(es)")
    return {"outcomes": len(outcome_rows), "summaries": summaries}


def main() -> int:
    spark = C.get_spark("cxr-build-outcomes")
    run(spark, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
