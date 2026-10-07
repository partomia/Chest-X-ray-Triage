"""
Stage 1 - Bronze: every source record of one business date as received (all values strings),
validated against its contract in config/lakehouse.json.

  bronze.ris_order, bronze.pacs_study, bronze.radiology_report, bronze.emr_patient
      valid records, with _source_file, _line, _ingested_at; partition business_date
  bronze.quarantine
      records that failed parsing or the contract, with every reason and the raw record

Reconciliation (ref.recon_results, stage bronze), per source: the manifest's record count
against the records in the file, the RIS trailer against its data lines, and received =
loaded + quarantined. Quarantined records are EXPLAINED differences, not mismatches.
A re-run of a date replaces that date's partitions in one commit.

Usage:
  spark-submit ingest_bronze.py --business-date 2026-09-28 [--landing s3a://...]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cxr_common as C  # noqa: E402

STAGE = "bronze"
QUARANTINE_SCHEMA = ("business_date date, source string, entity string, _source_file string, _line int, "
                     "reasons string, record string, _ingested_at timestamp_ntz")


def bronze_schema(contract: dict) -> str:
    cols = [f"`{c['name']}` string" for c in contract["columns"]]
    return ", ".join(["business_date date", "_source_file string", "_line int", "_ingested_at timestamp_ntz", *cols])


def ingest_source(spark, names, audit, cfg, source: str, d, landing: str, fs) -> dict:
    spec = cfg["sources"][source]
    entity, contract = spec["entity"], cfg["contracts"][spec["entity"]]
    folder = C.source_dir(landing, source, d)
    name = C.source_file(cfg, source, d)
    manifest_uri, file_uri = f"{folder}/_manifest.json", f"{folder}/{name}"
    if not fs.exists(file_uri):
        audit.check(entity, "file received", 1, 0, f"no {name} under {folder}")
        audit.load(entity, "MISSING", message=f"no file {file_uri}")
        return {"received": 0, "loaded": 0, "rejected": 0}
    manifest = json.loads(fs.read_text(manifest_uri)) if fs.exists(manifest_uri) else None
    parsed, trailer = C.parse_source(spec["format"], fs.read_text(file_uri))
    now = datetime.now()
    good, bad = [], []
    for line, rec, error in parsed:
        reasons = [error] if error else C.validate(rec, contract)
        if reasons:
            bad.append({"business_date": d, "source": source, "entity": entity, "_source_file": name, "_line": line,
                        "reasons": "; ".join(reasons), "record": json.dumps(rec) if rec else None,
                        "_ingested_at": now})
        else:
            row = {"business_date": d, "_source_file": name, "_line": line, "_ingested_at": now}
            row.update({c["name"]: (None if rec.get(c["name"]) in (None, "") else str(rec[c["name"]]).strip())
                        for c in contract["columns"]})
            good.append(row)
    table = names.t("bronze", entity)
    if good or C.table_exists(spark, table):
        df = C.frame(spark, good, bronze_schema(contract))
        if good:
            C.write_partitions(df, table)
        else:
            spark.sql(f"DELETE FROM {table} WHERE business_date = DATE '{d.isoformat()}'")
    received = len(parsed)
    if manifest is not None:
        audit.check(entity, "manifest records = records in file", manifest["records"], received,
                    f"{name}: the source's own count")
    if trailer is not None:
        audit.check(entity, "trailer count = data lines", trailer, received, f"{name}: T| trailer")
    audit.check(entity, "received = loaded + quarantined", received, len(good) + len(bad))
    if bad:
        audit.check(entity, "records quarantined", 0, len(bad), "; ".join(sorted({b["reasons"] for b in bad}))[:500],
                    explained=True)
    audit.load(entity, "COMMITTED", rows_in=received, rows_out=len(good), rows_rejected=len(bad),
               table=table if good else None)
    return {"received": received, "loaded": len(good), "rejected": len(bad), "quarantine": bad}


def run(spark, argv=None) -> dict:
    args = C.parse(C.base_parser(__doc__), argv)
    cfg, names, d = C.config(), C.Names(args.db_prefix), args.business_date
    C.ensure_databases(spark, names)
    audit = C.Audit(spark, names, STAGE, d, args.pipeline_run)
    fs = C.filesystem(spark, args.landing)
    summary, quarantine = {}, []
    for source in cfg["sources"]:
        r = ingest_source(spark, names, audit, cfg, source, d, args.landing, fs)
        quarantine += r.pop("quarantine", [])
        summary[cfg["sources"][source]["entity"]] = r
    qt = names.t("bronze", "quarantine")
    C.ensure_table(spark, qt, QUARANTINE_SCHEMA, ["business_date"])
    spark.sql(f"DELETE FROM {qt} WHERE business_date = DATE '{d.isoformat()}'")
    if quarantine:
        C.frame(spark, quarantine, QUARANTINE_SCHEMA).writeTo(qt).append()
    bad = audit.flush()
    audit.load("*", "FAILED" if bad else "COMPLETED",
               rows_in=sum(r["received"] for r in summary.values()),
               rows_out=sum(r["loaded"] for r in summary.values()),
               rows_rejected=sum(r["rejected"] for r in summary.values()))
    print(f"bronze {d}: {summary}", flush=True)
    if bad:
        raise RuntimeError(f"bronze {d}: {len(bad)} reconciliation mismatch(es)")
    return summary


def main() -> int:
    spark = C.get_spark("cxr-ingest-bronze")
    run(spark, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
