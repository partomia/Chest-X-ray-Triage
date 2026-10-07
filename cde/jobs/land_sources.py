"""
Stage 0 - the hospital's source systems drop one business date's extracts on the landing zone.

  ris/<date>/orders_<yyyymmdd>.psv       RIS imaging orders, pipe-delimited, trailer T|<count>
  pacs/<date>/studies_<yyyymmdd>.jsonl   PACS study headers (DICOM tags), one JSON per study;
                                         ImageFile names the film in the CAI project's archive
  reports/<date>/reports_<yyyymmdd>.jsonl radiology reports SIGNED that date: the day's own
                                         studies read before midnight and the previous day's
                                         read after it (late-arriving ground truth)
  emr/<date>/patients_<yyyymmdd>.csv     patients seen for the first time that date
  <source>/<date>/_manifest.json         what the source says it sent: file and record count

Everything is deterministic in (seed, date): 48 films a day, 32 children from the 624 Kermany
TEST films (cde/reference/test_films.csv) and 16 adults from the 300 NIH films kept for the
hospital (cde/reference/adult_films.csv), none ever used for training or threshold choice;
the radiologist reads them first in, first out (cxr_common.read_queue), which is today's
practice the triage is measured against.

On config defects_on (2026-10-01) the sources carry planted faults: two PACS headers sent
twice, one PACS header without AccessionNumber, one RIS order with an impossible order time;
and (planted_films) two films unfit for AI triage, one blurred and one rotated.

Usage:
  spark-submit land_sources.py --business-date 2026-09-28 [--landing s3a://...]
"""
from __future__ import annotations

import csv
import io
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cxr_common as C  # noqa: E402

RIS_COLUMNS = ["accession_no", "mrn", "order_ts", "ordering_unit", "clinical_priority", "indication", "procedure_code"]
EMR_COLUMNS = ["mrn", "given_name", "family_name", "birth_date", "sex"]


def pacs_header(s: dict) -> dict:
    return {"StudyInstanceUID": s["study_uid"], "AccessionNumber": s["accession_no"], "PatientID": s["mrn"],
            "Modality": s["modality"], "ViewPosition": s["view_position"], "BodyPartExamined": "CHEST",
            "StationName": s["station"], "Manufacturer": s["manufacturer"],
            "StudyDate": s["study_ts"].strftime("%Y%m%d"), "StudyTime": s["study_ts"].strftime("%H%M%S"),
            "ImageFile": s["image_file"], "Rows": s["rows"], "Columns": s["columns"]}


def ris_line(s: dict) -> dict:
    return {"accession_no": s["accession_no"], "mrn": s["mrn"], "order_ts": s["order_ts"].strftime(C.TS),
            "ordering_unit": s["ordering_unit"], "clinical_priority": s["clinical_priority"],
            "indication": s["indication"], "procedure_code": "XR-CHEST-1V"}


def first_seen(cfg: dict, d: date, pool: dict) -> dict:
    """patient key -> (first business date with a study, that study), for every date up to d."""
    seen: dict = {}
    day = date.fromisoformat(cfg["first_date"])
    while day <= d:
        for s in C.studies_of_day(cfg, day, pool):
            seen.setdefault(s["patient_key"], (day, s))
        day += timedelta(days=1)
    return seen


def signed_reports(cfg: dict, d: date, pool: dict) -> list[dict]:
    rng = random.Random(f"{cfg['seed']}-reports-{d.isoformat()}")
    out = []
    first = date.fromisoformat(cfg["first_date"])
    for study_day in (d - timedelta(days=2), d - timedelta(days=1), d):
        if study_day < first:
            continue
        studies = C.studies_of_day(cfg, study_day, pool)
        reads = C.reading(cfg, study_day, studies)
        out += [C.report_of(s, reads[s["accession_no"]], rng) for s in studies
                if reads[s["accession_no"]][1].date() == d]
    return sorted(out, key=lambda r: r["report_ts"])


def build_day(cfg: dict, d: date) -> dict:
    """source -> (file name, text, records) for the business date."""
    pool = C.pools(cfg)
    studies = C.studies_of_day(cfg, d, pool)
    ris = [ris_line(s) for s in sorted(studies, key=lambda s: s["order_ts"])]
    pacs = [pacs_header(s) for s in studies]
    if d.isoformat() == cfg.get("defects_on"):
        pacs += [dict(pacs[3]), dict(pacs[17])]            # re-sent by the modality
        pacs[25]["AccessionNumber"] = ""                     # technologist skipped the worklist
        ris[9]["order_ts"] = f"{d.isoformat()} 25:61:00"     # RIS clock fault
    seen = first_seen(cfg, d, pool)
    new_keys = sorted({s["patient_key"] for s in studies if seen[s["patient_key"]][0] == d})
    emr = [C.patient_of(cfg, k, d, seen[k][1]["age"], seen[k][1]["sex"]) for k in new_keys]
    reports = signed_reports(cfg, d, pool)

    def psv(rows):
        body = ["|".join(RIS_COLUMNS)] + ["|".join(str(r[c]) for c in RIS_COLUMNS) for r in rows]
        return "\n".join(body + [f"T|{len(rows)}"]) + "\n"

    def jsonl(rows):
        return "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)

    def as_csv(rows):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=EMR_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows({**r, "birth_date": r["birth_date"].isoformat()} for r in rows)
        return buf.getvalue()

    return {"ris": (C.source_file(cfg, "ris", d), psv(ris), len(ris)),
            "pacs": (C.source_file(cfg, "pacs", d), jsonl(pacs), len(pacs)),
            "reports": (C.source_file(cfg, "reports", d), jsonl(reports), len(reports)),
            "emr": (C.source_file(cfg, "emr", d), as_csv(emr), len(emr))}


def land(spark, d: date, landing: str) -> dict:
    cfg = C.config()
    fs = C.filesystem(spark, landing)
    counts = {}
    for source, (name, text, n) in build_day(cfg, d).items():
        folder = C.source_dir(landing, source, d)
        fs.write_text(f"{folder}/{name}", text)
        fs.write_text(f"{folder}/_manifest.json", json.dumps(
            {"source": source, "business_date": d.isoformat(), "files": [name], "records": n}))
        counts[source] = n
    print(f"landed {d}: {counts} under {landing}", flush=True)
    return counts


def run(spark, argv=None) -> dict:
    args = C.parse(C.base_parser(__doc__), argv)
    return land(spark, args.business_date, args.landing)


def main() -> int:
    spark = C.get_spark("cxr-land-sources")
    run(spark, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
