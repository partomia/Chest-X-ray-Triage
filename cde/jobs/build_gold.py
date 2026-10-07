"""
Stage 3 - Gold: the de-identified facts the triage model and the dashboards use.

  gold.dim_patient   patient_key (sha2 of the MRN), birth_year, sex, first_seen_date; no names
  gold.fact_study    one row per chest film of the business date: the PACS study joined to its
                     RIS order (ordering unit, clinical priority, order time), patient age, and
                     fifo_seq = the position in today's first-in-first-out reading list.
                     A study whose order failed the contract keeps ordering_unit UNKNOWN.
                     This is the table CAI job cxr-06-score-studies scores.
  gold.fact_report   every signed report with its study's business date and the truth label;
                     rebuilt each run, because reports land on the day they are signed

Reconciliation (stage gold): silver studies = gold studies; studies without a valid order and
reports without a study are EXPLAINED (they trace back to quarantined source records).

Usage:
  spark-submit build_gold.py --business-date 2026-09-28
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cxr_common as C  # noqa: E402

STAGE = "gold"
PATIENT_KEY = "substr(sha2(concat('cxr:', {mrn}), 256), 1, 16)"
FACT_STUDY_SCHEMA = (
    "business_date date, accession_no string, study_uid string, patient_key string, image_file string, "
    "study_ts timestamp_ntz, order_ts timestamp_ntz, ordering_unit string, clinical_priority string, "
    "indication string, modality string, view_position string, station string, manufacturer string, "
    "rows_px int, columns_px int, age_years int, has_order boolean, fifo_seq int")


def build_dim_patient(spark, names, audit) -> int:
    src, out = names.t("silver", "patient"), names.t("gold", "dim_patient")
    df = spark.sql(f"""
        SELECT {PATIENT_KEY.format(mrn='mrn')} AS patient_key, year(birth_date) AS birth_year, sex,
               first_seen_date FROM {src}""")
    C.replace_table(df, out)
    n = spark.table(out).count()
    audit.load("dim_patient", "COMMITTED", rows_out=n, table=out)
    return n


def build_fact_study(spark, names, audit, d) -> int:
    day = f"DATE '{d.isoformat()}'"
    study, order = names.t("silver", "study"), names.t("silver", "imaging_order")
    patient, out = names.t("silver", "patient"), names.t("gold", "fact_study")
    if not C.table_exists(spark, study):
        audit.load("fact_study", "MISSING", message=f"{study} does not exist")
        return 0
    orders = (f"(SELECT * FROM {order} WHERE business_date BETWEEN date_sub({day}, 1) AND {day})"
              if C.table_exists(spark, order) else
              "(SELECT CAST(NULL AS string) AS accession_no, CAST(NULL AS timestamp_ntz) AS order_ts, "
              "CAST(NULL AS string) AS ordering_unit, CAST(NULL AS string) AS clinical_priority, "
              "CAST(NULL AS string) AS indication WHERE false)")
    df = spark.sql(f"""
        SELECT s.business_date, s.accession_no, s.study_uid, {PATIENT_KEY.format(mrn='s.mrn')} AS patient_key,
               s.image_file, s.study_ts, o.order_ts, coalesce(o.ordering_unit, 'UNKNOWN') AS ordering_unit,
               coalesce(o.clinical_priority, 'UNKNOWN') AS clinical_priority, o.indication, s.modality,
               s.view_position, s.station, s.manufacturer, s.rows_px, s.columns_px,
               CAST(floor(months_between(to_date(s.study_ts), p.birth_date) / 12) AS int) AS age_years,
               o.accession_no IS NOT NULL AS has_order,
               CAST(row_number() OVER (ORDER BY s.study_ts, s.accession_no) AS int) AS fifo_seq
        FROM {study} s
        LEFT JOIN {orders} o ON o.accession_no = s.accession_no
        LEFT JOIN {patient} p ON p.mrn = s.mrn
        WHERE s.business_date = {day}""")
    rows = df.collect()
    n = len(rows)
    if n:
        C.write_partitions(spark.createDataFrame(rows, FACT_STUDY_SCHEMA), out)
    elif C.table_exists(spark, out):
        spark.sql(f"DELETE FROM {out} WHERE business_date = {day}")
    n_silver = spark.table(study).where(f"business_date = {day}").count()
    audit.check("fact_study", "silver studies = gold studies", n_silver, n)
    no_order = sum(1 for r in rows if not r["has_order"])
    if no_order:
        audit.check("fact_study", "studies without a valid order", 0, no_order,
                    ", ".join(r["accession_no"] for r in rows if not r["has_order"]), explained=True)
    no_age = sum(1 for r in rows if r["age_years"] is None)
    audit.check("fact_study", "studies without patient demographics", 0, no_age)
    audit.load("fact_study", "COMMITTED", rows_in=n_silver, rows_out=n, table=out if n else None)
    return n


def build_fact_report(spark, names, audit, d) -> int:
    report, out = names.t("silver", "report"), names.t("gold", "fact_report")
    fact = names.t("gold", "fact_study")
    if not C.table_exists(spark, report):
        audit.load("fact_report", "MISSING", message=f"{report} does not exist")
        return 0
    df = spark.sql(f"""
        SELECT r.accession_no, f.business_date AS study_date, r.business_date AS report_date, r.report_ts,
               r.radiologist_id, r.finding, r.pattern, CASE WHEN r.finding = 'PNEUMONIA' THEN 1 ELSE 0 END AS truth,
               r.impression, f.study_ts,
               CAST((unix_timestamp(r.report_ts) - unix_timestamp(f.study_ts)) / 60.0 AS double) AS report_wait_min
        FROM (SELECT * FROM (SELECT x.*, row_number() OVER (PARTITION BY accession_no ORDER BY report_ts DESC) AS _rn
                             FROM {report} x WHERE business_date <= DATE '{d.isoformat()}') WHERE _rn = 1) r
        LEFT JOIN {fact} f ON f.accession_no = r.accession_no""")
    C.replace_table(df, out)
    orphans = spark.table(out).where("study_date IS NULL").select("accession_no").collect()
    if orphans:
        audit.check("fact_report", "reports without a study", 0, len(orphans),
                    ", ".join(r[0] for r in orphans), explained=True)
    n = spark.table(out).count()
    audit.load("fact_report", "COMMITTED", rows_out=n, table=out)
    return n


def run(spark, argv=None) -> dict:
    args = C.parse(C.base_parser(__doc__), argv)
    names, d = C.Names(args.db_prefix), args.business_date
    C.ensure_databases(spark, names)
    audit = C.Audit(spark, names, STAGE, d, args.pipeline_run)
    summary = {"dim_patient": build_dim_patient(spark, names, audit),
               "fact_study": build_fact_study(spark, names, audit, d)}
    summary["fact_report"] = build_fact_report(spark, names, audit, d)
    bad = audit.flush()
    audit.load("*", "FAILED" if bad else "COMPLETED", rows_out=summary["fact_study"])
    print(f"gold {d}: {summary}", flush=True)
    if bad:
        raise RuntimeError(f"gold {d}: {len(bad)} reconciliation mismatch(es)")
    return summary


def main() -> int:
    spark = C.get_spark("cxr-build-gold")
    run(spark, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
