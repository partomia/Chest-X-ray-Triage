"""
Stage 2 - Silver: typed, de-duplicated, conformed records of one business date.

  silver.imaging_order   one row per accession (RIS), order_ts as timestamp
  silver.study           one row per StudyInstanceUID (PACS); study_ts from StudyDate + StudyTime;
                         re-sent headers are dropped (the latest line wins)
  silver.report          one row per accession signed that date (the latest signature wins)
  silver.patient         current EMR demographics per MRN, rebuilt from every bronze date (SCD1)

Column names avoid Impala reserved words (rows, columns, view, location ...).
Reconciliation (stage silver): bronze rows = silver rows + duplicates dropped, per entity;
duplicates are EXPLAINED.

Usage:
  spark-submit build_silver.py --business-date 2026-09-28
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cxr_common as C  # noqa: E402

STAGE = "silver"

SELECTS = {
    "imaging_order": ("ris_order", "accession_no", [
        "accession_no", "mrn", "CAST(order_ts AS timestamp_ntz) AS order_ts", "ordering_unit",
        "clinical_priority", "indication", "procedure_code"]),
    "study": ("pacs_study", "StudyInstanceUID", [
        "StudyInstanceUID AS study_uid", "AccessionNumber AS accession_no", "PatientID AS mrn",
        "Modality AS modality", "ViewPosition AS view_position", "BodyPartExamined AS body_part",
        "StationName AS station", "Manufacturer AS manufacturer",
        "to_timestamp_ntz(concat(StudyDate, StudyTime), 'yyyyMMddHHmmss') AS study_ts",
        "ImageFile AS image_file", "CAST(`Rows` AS int) AS rows_px", "CAST(`Columns` AS int) AS columns_px"]),
    "report": ("radiology_report", "accession_no", [
        "accession_no", "CAST(report_ts AS timestamp_ntz) AS report_ts", "radiologist_id", "finding",
        "pattern", "impression"]),
}
PATIENT_SCHEMA = ("mrn string, given_name string, family_name string, birth_date date, sex string, "
                  "first_seen_date date, updated_date date")


def latest(spark, table: str, key: str, where: str):
    return spark.sql(f"""
        SELECT * FROM (
          SELECT b.*, row_number() OVER (PARTITION BY `{key}` ORDER BY _ingested_at DESC, _line DESC) AS _rn,
                 count(*) OVER (PARTITION BY `{key}`) AS _copies
          FROM {table} b WHERE {where}) WHERE _rn = 1""")


def build_entity(spark, names, audit, d, target: str) -> int:
    source, key, cols = SELECTS[target]
    bronze = names.t("bronze", source)
    out = names.t("silver", target)
    where = f"business_date = DATE '{d.isoformat()}'"
    if not C.table_exists(spark, bronze):
        audit.load(target, "MISSING", message=f"{bronze} does not exist")
        return 0
    n_bronze = spark.table(bronze).where(where).count()
    df = latest(spark, bronze, key, where).selectExpr("business_date", *cols)
    n = df.count()
    if n:
        C.write_partitions(df, out)
    elif C.table_exists(spark, out):
        spark.sql(f"DELETE FROM {out} WHERE {where}")
    dupes = n_bronze - n
    audit.check(target, "bronze rows = silver rows + duplicates", n_bronze, n + dupes)
    if dupes:
        audit.check(target, "duplicates dropped", 0, dupes, f"{dupes} re-sent {key}", explained=True)
    audit.load(target, "COMMITTED", rows_in=n_bronze, rows_out=n, rows_rejected=dupes, table=out if n else None)
    return n


def build_patient(spark, names, audit, d) -> int:
    bronze = names.t("bronze", "emr_patient")
    out = names.t("silver", "patient")
    if not C.table_exists(spark, bronze):
        C.ensure_table(spark, out, PATIENT_SCHEMA)
        audit.load("patient", "MISSING", message=f"{bronze} does not exist")
        return 0
    df = spark.sql(f"""
        SELECT mrn, given_name, family_name, CAST(birth_date AS date) AS birth_date, sex,
               first_seen_date, updated_date FROM (
          SELECT b.*, min(business_date) OVER (PARTITION BY mrn) AS first_seen_date,
                 business_date AS updated_date,
                 row_number() OVER (PARTITION BY mrn ORDER BY business_date DESC, _line DESC) AS _rn
          FROM {bronze} b WHERE business_date <= DATE '{d.isoformat()}') WHERE _rn = 1""")
    C.replace_table(df, out)
    n = spark.table(out).count()
    audit.load("patient", "COMMITTED", rows_out=n, table=out)
    return n


def run(spark, argv=None) -> dict:
    args = C.parse(C.base_parser(__doc__), argv)
    names, d = C.Names(args.db_prefix), args.business_date
    C.ensure_databases(spark, names)
    audit = C.Audit(spark, names, STAGE, d, args.pipeline_run)
    summary = {t: build_entity(spark, names, audit, d, t) for t in SELECTS}
    summary["patient"] = build_patient(spark, names, audit, d)
    bad = audit.flush()
    audit.load("*", "FAILED" if bad else "COMPLETED", rows_out=sum(summary.values()))
    print(f"silver {d}: {summary}", flush=True)
    if bad:
        raise RuntimeError(f"silver {d}: {len(bad)} reconciliation mismatch(es)")
    return summary


def main() -> int:
    spark = C.get_spark("cxr-build-silver")
    run(spark, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
