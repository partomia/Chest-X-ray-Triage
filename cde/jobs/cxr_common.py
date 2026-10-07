"""
Shared helpers for the CXR lakehouse CDE jobs. Standard library and PySpark only,
so the jobs need no python-env resource on CDE.

  names       database and table names: <prefix>_<layer>.<table>
  iceberg     one atomic commit per write (overwritePartitions for a date, createOrReplace
              for a rebuilt table); every table is created through _create (format v2)
  audit       ref.load_audit (stage x entity status per business date) and
              ref.recon_results (one row per reconciliation check)
  fs          the landing zone through Hadoop (s3a:// on CDE) or a local folder
  hospital    the deterministic hospital day: which films arrive (children from the Kermany
              test split, adults from the NIH films kept for the hospital), when, from which
              unit, and when a radiologist reads them. land_sources writes it as source files;
              build_outcomes replays the reading queue in triage order
  contracts   parse and validate one source record against config/lakehouse.json

Volumes are small (about 50 studies a day), so source files are parsed on the driver.
On CDE the repository is mounted at /app/mount, so config/ and cde/reference/ are read
relative to this file there and on a laptop alike. All timestamps are hospital local
time, stored as timestamp_ntz: Impala writes Iceberg timestamps without a time zone only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

LAYERS = ("bronze", "silver", "gold", "ref", "semantic")
TS = "%Y-%m-%d %H:%M:%S"
BAND_RANK = {"P1": 0, "P2": 1, "P3": 2}

LOAD_AUDIT_SCHEMA = ("run_id string, pipeline_run string, business_date date, stage string, entity string, "
                     "status string, rows_in bigint, rows_out bigint, rows_rejected bigint, snapshot_id bigint, "
                     "started_at timestamp_ntz, ended_at timestamp_ntz, message string")
RECON_SCHEMA = ("run_id string, business_date date, stage string, entity string, check_name string, "
                "expected double, actual double, status string, detail string, logged_at timestamp_ntz")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config() -> dict:
    return json.loads((repo_root() / "config" / "lakehouse.json").read_text())


def base_parser(description: str) -> argparse.ArgumentParser:
    cfg = config()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--business-date", type=date.fromisoformat, required=True)
    p.add_argument("--db-prefix", default=cfg["db_prefix"])
    p.add_argument("--landing", default=cfg["landing"])
    p.add_argument("--pipeline-run", default=None, help="groups the stages of one DAG run (Airflow run_id)")
    return p


def parse(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    args, _ = parser.parse_known_args(argv)
    return args


class Names:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def db(self, layer: str) -> str:
        assert layer in LAYERS, layer
        return f"{self.prefix}_{layer}"

    def t(self, layer: str, name: str) -> str:
        return f"{self.db(layer)}.{name}"


# ---------------------------------------------------------------- spark + iceberg


def configure(spark) -> None:
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.conf.set("spark.sql.session.timeZone", "UTC")   # timestamp_ntz values are never shifted
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    try:
        spark.sparkContext.addPyFile(str(Path(__file__).resolve()))
    except Exception:  # already shipped in this session
        pass


def get_spark(app: str):
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName(app).getOrCreate()
    configure(spark)
    return spark


def ensure_databases(spark, names: Names) -> None:
    for layer in LAYERS:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {names.db(layer)}")


def table_exists(spark, table: str) -> bool:
    return spark.catalog.tableExists(table)


def snapshot_id(spark, table: str) -> int | None:
    if not table_exists(spark, table):
        return None
    rows = spark.sql(f"SELECT snapshot_id FROM {table}.snapshots ORDER BY committed_at DESC LIMIT 1").collect()
    return int(rows[0][0]) if rows else None


def _create(df, table: str, partition_cols=()):
    from pyspark.sql import functions as F

    w = df.writeTo(table).using("iceberg").tableProperty("format-version", "2")
    if partition_cols:
        w = w.partitionedBy(*[F.col(c) for c in partition_cols])
    return w


def add_missing_columns(spark, table: str, schema) -> list[str]:
    """Columns of schema (a StructType) the existing table lacks are added (Iceberg schema evolution)."""
    have = {f.name.lower() for f in spark.table(table).schema.fields}
    missing = [f for f in schema.fields if f.name.lower() not in have]
    if missing:
        spark.sql(f"ALTER TABLE {table} ADD COLUMNS ("
                  + ", ".join(f"`{f.name}` {f.dataType.simpleString()}" for f in missing) + ")")
        print(f"{table}: added columns {[f.name for f in missing]}", flush=True)
    return [f.name for f in missing]


def write_partitions(df, table: str, partition_cols=("business_date",)) -> None:
    """Replace exactly the partitions present in df, in one commit (a re-run of a date)."""
    spark = df.sparkSession
    if table_exists(spark, table):
        add_missing_columns(spark, table, df.schema)
        from pyspark.sql import functions as F

        names = {c.lower(): c for c in df.columns}
        df = df.select(*[F.col(names[f.name.lower()]) if f.name.lower() in names
                         else F.lit(None).cast(f.dataType).alias(f.name) for f in spark.table(table).schema.fields])
        df.writeTo(table).overwritePartitions()
    else:
        _create(df, table, partition_cols).create()


def replace_table(df, table: str, partition_cols=()) -> None:
    _create(df, table, partition_cols).createOrReplace()


def ensure_table(spark, table: str, schema: str, partition_cols=()) -> None:
    empty = spark.createDataFrame([], schema)
    if not table_exists(spark, table):
        _create(empty, table, partition_cols).create()
    else:
        add_missing_columns(spark, table, empty.schema)


def frame(spark, rows: list[dict], schema: str):
    """A DataFrame from dicts, columns in schema order (missing keys are NULL)."""
    cols = [c.strip().split(" ")[0].strip("`") for c in schema.split(",")]
    return spark.createDataFrame([tuple(r.get(c) for c in cols) for r in rows], schema)


# ---------------------------------------------------------------- audit + reconciliation


class Audit:
    def __init__(self, spark, names: Names, stage: str, business_date: date, pipeline_run: str | None):
        self.spark, self.names, self.stage, self.d = spark, names, stage, business_date
        self.run_id = f"{stage}-{uuid.uuid4().hex[:12]}"
        self.pipeline_run = pipeline_run or self.run_id
        self.started = datetime.now()
        self.recon: list[dict] = []
        ensure_table(spark, names.t("ref", "load_audit"), LOAD_AUDIT_SCHEMA, ["business_date"])
        ensure_table(spark, names.t("ref", "recon_results"), RECON_SCHEMA, ["business_date"])

    def load(self, entity: str, status: str, rows_in=None, rows_out=None, rows_rejected=None,
             table: str | None = None, message: str = "") -> None:
        row = {"run_id": self.run_id, "pipeline_run": self.pipeline_run, "business_date": self.d,
               "stage": self.stage, "entity": entity, "status": status, "rows_in": rows_in, "rows_out": rows_out,
               "rows_rejected": rows_rejected, "snapshot_id": snapshot_id(self.spark, table) if table else None,
               "started_at": self.started, "ended_at": datetime.now(), "message": message[:2000]}
        frame(self.spark, [row], LOAD_AUDIT_SCHEMA).writeTo(self.names.t("ref", "load_audit")).append()

    def check(self, entity: str, check_name: str, expected, actual, detail: str = "", explained: bool = False,
              tolerance: float = 0.0) -> bool:
        e = None if expected is None else float(expected)
        a = None if actual is None else float(actual)
        ok = e is not None and a is not None and abs(a - e) <= tolerance
        self.recon.append({"run_id": self.run_id, "business_date": self.d, "stage": self.stage, "entity": entity,
                           "check_name": check_name, "expected": e, "actual": a,
                           "status": "MATCHED" if ok else ("EXPLAINED" if explained else "MISMATCH"),
                           "detail": detail[:500], "logged_at": datetime.now()})
        return ok

    def flush(self) -> list[dict]:
        """This stage's checks replace its earlier ones for the date (a re-run), in one commit."""
        t = self.names.t("ref", "recon_results")
        keep = self.spark.table(t).where(f"business_date = DATE '{self.d.isoformat()}' AND stage <> '{self.stage}'")
        new = frame(self.spark, self.recon, RECON_SCHEMA)
        keep.unionByName(new).writeTo(t).overwritePartitions()
        bad = [r for r in self.recon if r["status"] == "MISMATCH"]
        print(f"{self.stage} {self.d}: {len(self.recon) - len(bad)} checks matched or explained, "
              f"{len(bad)} mismatched", flush=True)
        for r in bad:
            print(f"  MISMATCH {r['entity']} {r['check_name']}: expected {r['expected']}, got {r['actual']}")
        return bad


# ---------------------------------------------------------------- landing zone


class HadoopFS:
    """Files on the landing zone through the Hadoop FileSystem of the Spark session."""

    def __init__(self, spark):
        self.jvm = spark._jvm
        self.conf = spark._jsc.hadoopConfiguration()

    def _fs(self, uri: str):
        path = self.jvm.org.apache.hadoop.fs.Path(uri)
        return path.getFileSystem(self.conf), path

    def write_text(self, uri: str, text: str) -> None:
        fs, path = self._fs(uri)
        out = fs.create(path, True)
        try:
            out.write(bytearray(text.encode("utf-8")))
        finally:
            out.close()

    def exists(self, uri: str) -> bool:
        fs, path = self._fs(uri)
        return bool(fs.exists(path))

    def read_text(self, uri: str) -> str:
        fs, path = self._fs(uri)
        stream = fs.open(path)
        try:
            return bytes(self.jvm.org.apache.commons.io.IOUtils.toByteArray(stream)).decode("utf-8")
        finally:
            stream.close()


class LocalFS:
    @staticmethod
    def _path(uri: str) -> Path:
        return Path(uri[len("file://"):] if uri.startswith("file://") else uri)

    def write_text(self, uri: str, text: str) -> None:
        p = self._path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()

    def read_text(self, uri: str) -> str:
        return self._path(uri).read_text()


def filesystem(spark, uri: str):
    return HadoopFS(spark) if spark is not None and "://" in uri and not uri.startswith("file:") else LocalFS()


def source_dir(landing: str, source: str, d: date) -> str:
    return f"{landing.rstrip('/')}/{source}/{d.isoformat()}"


def source_file(cfg: dict, source: str, d: date) -> str:
    return cfg["sources"][source]["file"].format(date=d.strftime("%Y%m%d"))


# ---------------------------------------------------------------- the hospital (deterministic)


def films(cfg: dict) -> list[dict]:
    """The paediatric films: the 624 Kermany TEST films."""
    with open(repo_root() / cfg["film_reference"]) as f:
        return list(csv.DictReader(f))


def adult_films(cfg: dict) -> list[dict]:
    """The adult films: NIH ChestX-ray14 films of patients the pneumothorax model never saw
    (data_refs/nih_cxr14_subset.csv, split lakehouse), with the patient's NIH id, age and sex."""
    with open(repo_root() / cfg["adult_film_reference"]) as f:
        return list(csv.DictReader(f))


def pools(cfg: dict) -> dict:
    return {"pediatric": films(cfg), "adult": adult_films(cfg)}


def patient_key(image_file: str) -> str:
    """Kermany file names encode the patient: person123_bacteria_4.jpeg, IM-0115-0001.jpeg, NORMAL2-IM-0927-0001.jpeg."""
    stem = image_file.rsplit(".", 1)[0]
    if stem.startswith("person"):
        return stem.split("_", 1)[0]
    return stem.rsplit("-", 1)[0]


def planted_name(kind: str, image_file: str) -> str:
    """A film made unfit for AI triage (features/degrade.py kind) from a real one; CAI makes the
    file from the original on first use (lakehouse/score_studies.py)."""
    return f"QC-{kind.upper()}-{image_file}"


def mrn_of(key: str, seed: int) -> str:
    return "MRN" + str(int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16) % 10**7).zfill(7)


GIVEN = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Kabir", "Reyansh", "Arjun", "Sai", "Advik", "Krishna",
         "Ananya", "Diya", "Saanvi", "Aadhya", "Kiara", "Myra", "Pari", "Anika", "Navya", "Ira"]
FAMILY = ["Sharma", "Verma", "Iyer", "Reddy", "Nair", "Gupta", "Patel", "Rao", "Das", "Menon",
          "Singh", "Khan", "Joshi", "Kulkarni", "Banerjee", "Pillai", "Chopra", "Mehta", "Bose", "Shetty"]
INDICATIONS = {"ED": ["fever and cough", "fast breathing", "r/o pneumonia", "chest indrawing", "wheeze, fever"],
               "OPD": ["persistent cough", "follow-up", "fever 3 days", "r/o pneumonia", "noisy breathing"],
               "WARD": ["worsening cough", "fever spike", "post-treatment review", "low SpO2"],
               "ICU": ["desaturation", "line check, fever", "ventilated, fever", "increasing O2 need"]}
ADULT_INDICATIONS = {"ED": ["sudden pleuritic chest pain", "dyspnoea", "chest trauma", "r/o pneumothorax", "cough, fever"],
                     "OPD": ["chronic cough", "pre-operative", "follow-up", "breathlessness on exertion"],
                     "WARD": ["post central line insertion", "post pleural tap", "worsening dyspnoea", "fever"],
                     "ICU": ["ventilated, desaturation", "line and tube check", "post-procedure", "rising O2 need"]}
STATIONS = {"ED": ("DX", "ED-DR-01", "Siemens"), "OPD": ("DX", "OPD-DR-03", "Philips"),
            "WARD": ("CR", "WARD-PORT-02", "Carestream"), "ICU": ("CR", "ICU-PORT-01", "GE")}


def day_index(cfg: dict, d: date) -> int:
    i = (d - date.fromisoformat(cfg["first_date"])).days
    if i < 0:
        raise ValueError(f"{d} is before the first business date {cfg['first_date']}")
    return i


def _slice(cfg: dict, d: date, pool: list[dict], n: int, name: str) -> list[dict]:
    """Consecutive slices of one seeded permutation of the pool, so the first len(pool) // n days
    never repeat a film."""
    perm = list(range(len(pool)))
    random.Random(f"{cfg['seed']}-films" + ("" if name == "pediatric" else f"-{name}")).shuffle(perm)
    i = day_index(cfg, d)
    return [dict(pool[perm[(i * n + j) % len(pool)]], population=name) for j in range(n)]


def films_of_day(cfg: dict, d: date, pools: dict) -> list[dict]:
    """The day's films: adult_per_day adult films and the rest paediatric, in a seeded order."""
    n_adult = cfg["adult_per_day"]
    day = (_slice(cfg, d, pools["pediatric"], cfg["studies_per_day"] - n_adult, "pediatric")
           + _slice(cfg, d, pools["adult"], n_adult, "adult"))
    random.Random(f"{cfg['seed']}-mix-{d.isoformat()}").shuffle(day)
    return day


def _weighted(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def studies_of_day(cfg: dict, d: date, pools: dict) -> list[dict]:
    """Every study of the business date, with its order, PACS header and the truth label.
    On a date in planted_films, the listed slots carry a film made unfit for AI triage."""
    rng = random.Random(f"{cfg['seed']}-day-{d.isoformat()}")
    a = cfg["arrivals"]
    hours = list(range(a["first_hour"], a["last_hour"]))
    planted = {p["slot"]: p["kind"] for p in cfg.get("planted_films", {}).get(d.isoformat(), [])}
    out = []
    for j, film in enumerate(films_of_day(cfg, d, pools)):
        adult = film["population"] == "adult"
        unit = _weighted(rng, cfg["units"])
        hour = rng.choices(hours, weights=a["hour_weights"][:len(hours)])[0]
        order_ts = datetime.combine(d, time(hour)) + timedelta(seconds=rng.randrange(3600))
        study_ts = order_ts + timedelta(minutes=rng.randint(8, 45))
        modality, station, maker = STATIONS[unit]
        view = film["view_position"] if adult else "AP" if unit in ("ICU", "WARD") else "PA"
        priority = ("STAT" if unit == "ICU" or (unit == "ED" and rng.random() < 0.4)
                    else "URGENT" if unit in ("ED", "WARD") else "ROUTINE")
        key = f"NIH{film['patient_id']}" if adult else patient_key(film["image_file"])
        indication = rng.choice((ADULT_INDICATIONS if adult else INDICATIONS)[unit])
        image_file = planted_name(planted[j], film["image_file"]) if j in planted else film["image_file"]
        out.append({
            "accession_no": f"ACC{d:%y%m%d}{j:03d}", "study_uid": f"1.2.826.0.1.3680043.10.543.{d:%Y%m%d}.{j}",
            "patient_key": key, "mrn": mrn_of(key, cfg["seed"]), "order_ts": order_ts, "study_ts": study_ts,
            "ordering_unit": unit, "clinical_priority": priority, "indication": indication,
            "modality": modality, "view_position": view, "station": station, "manufacturer": maker,
            "image_file": image_file, "label": film["label"], "population": film["population"],
            "age": int(film["age"]) if adult else None, "sex": film.get("sex"),
            "findings": film.get("findings") or film["label"], "planted": planted.get(j),
            "pattern": ("viral" if "virus" in film["image_file"] else "bacterial") if film["label"] == "PNEUMONIA" else None,
            "rows": rng.randint(1100, 2600), "columns": rng.randint(1300, 2900)})
    return sorted(out, key=lambda s: (s["study_ts"], s["accession_no"]))


def patient_of(cfg: dict, key: str, first_seen: date, age: int | None = None, sex: str | None = None) -> dict:
    """Children (Kermany) are 1-6; an adult's birth date follows the age on the NIH film."""
    rng = random.Random(f"{cfg['seed']}-patient-{key}")
    days = rng.randint(365, 6 * 365) if age is None else int(age * 365.25) + rng.randint(0, 364)
    return {"mrn": mrn_of(key, cfg["seed"]), "given_name": rng.choice(GIVEN), "family_name": rng.choice(FAMILY),
            "birth_date": first_seen - timedelta(days=days), "sex": sex if sex in ("F", "M") else rng.choice("FM")}


def read_queue(items: list[dict], rank, readers: int, shift_start: datetime, minutes: float) -> dict:
    """Radiologists read one study at a time from the studies that have arrived; when a reader is
    free, the study with the lowest rank(item) goes next. FIFO: rank = arrival. Triage: rank =
    (band, -probability, arrival). Returns accession -> (read start, report time, reader)."""
    pending = sorted(items, key=lambda s: (s["study_ts"], s["accession_no"]))
    free = [shift_start] * readers
    waiting, out = [], {}
    while pending or waiting:
        r = min(range(readers), key=lambda k: free[k])
        while pending and pending[0]["study_ts"] <= free[r]:
            waiting.append(pending.pop(0))
        if not waiting:
            free[r] = pending[0]["study_ts"]
            continue
        nxt = min(waiting, key=rank)
        waiting.remove(nxt)
        start = max(free[r], nxt["study_ts"])
        end = start + timedelta(minutes=minutes)
        out[nxt["accession_no"]] = (start, end, f"RAD{r + 1:02d}")
        free[r] = end
    return out


def fifo_rank(s: dict):
    return (s["study_ts"], s["accession_no"])


def triage_rank(s: dict):
    """AI-flagged studies first (P1, then P2, higher probability first); then P3 and NA (no AI
    triage: out of every live model's intended use, or a film unfit for it) in arrival order,
    as a worklist without AI reads them; unscored studies last."""
    band = s.get("priority")
    if band in ("P1", "P2"):
        return (BAND_RANK[band], -float(s.get("probability") or 0.0), s["study_ts"], s["accession_no"])
    if band in ("P3", "NA"):
        return (2, 0.0, s["study_ts"], s["accession_no"])
    return (3, 0.0, s["study_ts"], s["accession_no"])


def reading(cfg: dict, d: date, items: list[dict], rank=fifo_rank) -> dict:
    r = cfg["reading"]
    return read_queue(items, rank, r["readers"], datetime.combine(d, time(r["shift_start_hour"])),
                      r["minutes_per_film"])


IMPRESSIONS = {"NORMAL": ["No focal consolidation. Lungs clear.", "Normal chest radiograph for age.",
                          "No acute cardiopulmonary abnormality."],
               "bacterial": ["Lobar consolidation consistent with bacterial pneumonia.",
                             "Dense focal opacity, likely bacterial pneumonia."],
               "viral": ["Bilateral perihilar interstitial opacities, suggestive of viral pneumonia.",
                         "Diffuse interstitial pattern consistent with viral pneumonia."],
               "PNEUMOTHORAX": ["Pneumothorax with visible pleural line; no mediastinal shift.",
                                "Apical pneumothorax. Clinical correlation and follow-up film advised.",
                                "Pneumothorax present; chest drain position to be reviewed."],
               "ADULT_NORMAL": ["No pneumothorax. Heart and lungs within normal limits.",
                                "No acute cardiopulmonary abnormality."]}


def report_of(s: dict, signed: tuple, rng: random.Random) -> dict:
    """The signed report. An adult film's finding is PNEUMOTHORAX, NORMAL (NIH No Finding) or
    OTHER (any of NIH's other findings, named in the impression)."""
    if s["population"] == "adult":
        finding = s["label"]
        impression = (rng.choice(IMPRESSIONS["PNEUMOTHORAX"]) if finding == "PNEUMOTHORAX"
                      else rng.choice(IMPRESSIONS["ADULT_NORMAL"]) if finding == "NORMAL"
                      else f"No pneumothorax. {s['findings'].replace('|', ', ').replace('_', ' ')}.")
    else:
        finding, impression = s["label"], rng.choice(IMPRESSIONS[s["pattern"] or "NORMAL"])
    return {"accession_no": s["accession_no"], "report_ts": signed[1].strftime(TS), "radiologist_id": signed[2],
            "finding": finding, "pattern": s["pattern"], "impression": impression}


# ---------------------------------------------------------------- contracts


def _valid(value: str, col: dict) -> str | None:
    """None if value fits the column, else the reason."""
    kind = col["type"]
    try:
        if kind == "timestamp":
            datetime.strptime(value, TS)
        elif kind == "date":
            date.fromisoformat(value)
        elif kind == "date8":
            datetime.strptime(value, "%Y%m%d")
        elif kind == "time6":
            datetime.strptime(value, "%H%M%S")
        elif kind == "int":
            int(value)
    except ValueError:
        return f"{col['name']}: not a {kind} ({value[:40]!r})"
    if col.get("values") and value not in col["values"]:
        return f"{col['name']}: {value[:40]!r} not in {col['values']}"
    return None


def validate(record: dict, contract: dict) -> list[str]:
    reasons = []
    for col in contract["columns"]:
        v = record.get(col["name"])
        v = None if v is None else str(v).strip()
        if not v:
            if col["required"]:
                reasons.append(f"{col['name']}: missing")
            continue
        why = _valid(v, col)
        if why:
            reasons.append(why)
    return reasons


def parse_source(fmt: str, text: str) -> tuple[list[tuple[int, dict | None, str | None]], int | None]:
    """[(line number, record or None, parse error)], trailer count (pipe-delimited files only)."""
    out, trailer = [], None
    if fmt == "jsonl":
        for no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                out.append((no, rec, None) if isinstance(rec, dict) else (no, None, "not a JSON object"))
            except ValueError as e:
                out.append((no, None, f"invalid JSON: {str(e)[:100]}"))
    elif fmt == "psv":
        lines = text.splitlines()
        header = lines[0].split("|") if lines else []
        for no, line in enumerate(lines[1:], 2):
            if not line:
                continue
            if line.startswith("T|"):
                trailer = int(line.split("|")[1])
                continue
            parts = line.split("|")
            out.append((no, dict(zip(header, parts)), None) if len(parts) == len(header)
                       else (no, None, f"{len(parts)} fields, header has {len(header)}"))
    elif fmt == "csv":
        for no, rec in enumerate(csv.DictReader(io.StringIO(text)), 2):
            out.append((no, dict(rec), None))
    else:
        raise ValueError(fmt)
    return out, trailer
