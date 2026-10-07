"""
The lakehouse as CAI sees it: the Iceberg tables CDE builds and the ones CAI writes, through
one small interface on two engines.

  ImpalaStore   CDW Impala virtual warehouse over HTTPS with LDAP (impyla). Used in CAI jobs.
                REFRESH picks up snapshots CDE committed since Impala last loaded a table.
  SparkStore    a Spark session with an Iceberg catalog (the local runner and CI).

Reads of CDE tables are pinned to one snapshot (FOR SYSTEM_VERSION AS OF / VERSION AS OF), so
a run scores exactly the rows it recorded. CAI writes a date with DELETE then INSERT (Iceberg
v2 row-level deletes); every row of a CAI table carries the run that wrote it.

Credentials: CXR_IMPALA_USER / CXR_IMPALA_PASSWORD (CAI project environment, or .env locally).
"""
from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "lakehouse.json"

# layer.table -> ([(column, type)], partition column or None). Types are Impala's; SparkStore
# maps TIMESTAMP to TIMESTAMP_NTZ, as CDE writes them.
TABLES = {
    "gold.triage_score": ([
        ("business_date", "DATE"), ("accession_no", "STRING"), ("image_file", "STRING"),
        ("probability", "DOUBLE"), ("priority", "STRING"), ("threshold", "DOUBLE"),
        ("quality_flags", "STRING"), ("model_version", "STRING"), ("model_git_sha", "STRING"),
        ("feature_version", "STRING"), ("run_id", "STRING"), ("scored_at", "TIMESTAMP"),
        ("triage_model", "STRING"), ("film_qc", "STRING"), ("age_years", "INT"),
        ("shadow_priority", "STRING"), ("shadow_probability", "DOUBLE"), ("shadow_model", "STRING")],
        "business_date"),
    # every head's score for every study: champions, silent trials and the film check
    "gold.model_score": ([
        ("business_date", "DATE"), ("accession_no", "STRING"), ("model", "STRING"), ("stage", "STRING"),
        ("model_version", "STRING"), ("in_scope", "BOOLEAN"), ("probability", "DOUBLE"), ("threshold", "DOUBLE"),
        ("positive", "BOOLEAN"), ("priority", "STRING"), ("run_id", "STRING"), ("scored_at", "TIMESTAMP")],
        "business_date"),
    "ref.triage_run": ([
        ("run_id", "STRING"), ("business_date", "DATE"), ("triggered_by", "STRING"),
        ("source_snapshot_id", "BIGINT"), ("studies", "INT"), ("scored", "INT"), ("missing_films", "INT"),
        ("p1", "INT"), ("p2", "INT"), ("p3", "INT"), ("model_version", "STRING"), ("threshold", "DOUBLE"),
        ("psi_max", "DOUBLE"), ("drift_level", "STRING"), ("psi_json", "STRING"), ("status", "STRING"),
        ("message", "STRING"), ("started_at", "TIMESTAMP"), ("ended_at", "TIMESTAMP"),
        ("not_triaged", "INT"), ("film_unsuitable", "INT"), ("heads_json", "STRING")], None),
    "ref.model_event": ([
        ("event_id", "STRING"), ("event", "STRING"), ("recorded_at", "TIMESTAMP"), ("git_sha", "STRING"),
        ("model_version", "STRING"), ("feature_version", "STRING"), ("feature_hash", "STRING"),
        ("threshold", "DOUBLE"), ("val_auroc", "DOUBLE"), ("test_auroc", "DOUBLE"),
        ("test_sensitivity", "DOUBLE"), ("test_specificity", "DOUBLE"), ("test_brier", "DOUBLE"),
        ("train_rows", "INT"), ("gate_passed", "BOOLEAN"), ("gate_checks", "STRING"),
        ("mlflow_run_id", "STRING"), ("detail", "STRING"),
        ("model", "STRING"), ("stage", "STRING"), ("registry_version", "INT")], None),
    "ref.training_set": ([
        ("feature_version", "STRING"), ("feature_hash", "STRING"), ("data_split", "STRING"), ("films", "INT"),
        ("pneumonia", "INT"), ("patients", "INT"), ("backbone", "STRING"), ("git_sha", "STRING"),
        ("created_at", "TIMESTAMP"), ("model", "STRING"), ("positives", "INT")], None),
}


def lake_config() -> dict:
    return json.loads(CONFIG.read_text())


def model_version(meta: dict) -> str:
    """fv1.0.0-abc1234 for pneumonia (as before there were other models), <model>-fv...-sha otherwise."""
    base = f"fv{meta['feature_version']}-{meta['git_sha'][:7]}"
    model = meta.get("model") or "pneumonia"
    return base if model == "pneumonia" else f"{model}-{base}"


def _literal(value, typ: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    if typ == "DOUBLE":
        return repr(float(value))
    if typ in ("INT", "BIGINT"):
        return str(int(value))
    if typ == "BOOLEAN":
        return "true" if value else "false"
    if typ == "DATE":
        v = value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
        return f"DATE '{v.isoformat()}'"
    if typ == "TIMESTAMP":
        v = value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else str(value)[:19]
        return f"CAST('{v}' AS TIMESTAMP)"
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


class _Store:
    def __init__(self, prefix: str | None = None):
        self.prefix = prefix or lake_config()["db_prefix"]
        self._ensured: set[str] = set()

    def t(self, key: str) -> str:
        layer, name = key.split(".")
        return f"{self.prefix}_{layer}.{name}"

    def ensure(self, key: str) -> None:
        """Create the table if absent; add the columns TABLES has gained since it was created."""
        if key in self._ensured:
            return
        cols, part = TABLES[key]
        self.execute(f"CREATE DATABASE IF NOT EXISTS {self.t(key).split('.')[0]}")
        self.execute(self.create_sql(self.t(key), cols, part))
        have = self.columns(key)
        missing = [(c, t) for c, t in cols if c.lower() not in have]
        if missing:
            self.execute(self.add_columns_sql(self.t(key), missing))
            print(f"[lakehouse] {self.t(key)}: added columns {[c for c, _ in missing]}")
        self._ensured.add(key)

    def columns(self, key: str) -> set[str]:
        rows = self.query(f"DESCRIBE {self.t(key)}")
        names = set()
        for r in rows:
            name = str(r.get("name") or r.get("col_name") or "").strip()
            if not name or name.startswith("#"):   # Spark lists the partitioning after a '# ...' header
                break
            names.add(name.lower())
        return names

    @staticmethod
    def add_columns_sql(table: str, cols) -> str:
        return f"ALTER TABLE {table} ADD COLUMNS ({', '.join(f'`{c}` {t}' for c, t in cols)})"

    def replace_date(self, key: str, rows: list[dict], d: date, column: str = "business_date") -> None:
        self.ensure(key)
        self.execute(f"DELETE FROM {self.t(key)} WHERE {column} = DATE '{d.isoformat()}'")
        self.append(key, rows)

    def exists(self, key: str) -> bool:
        db, name = self.t(key).split(".")
        try:
            return any(name == (r.get("name") or r.get("tableName") or next(iter(r.values())))
                       for r in self.query(f"SHOW TABLES IN {db}"))
        except Exception:
            return False


class ImpalaStore(_Store):
    engine = "impala"

    def __init__(self, cfg: dict | None = None, prefix: str | None = None, user: str | None = None,
                 password: str | None = None):
        super().__init__(prefix)
        self.cfg = cfg or lake_config()["impala"]
        self.user = user or os.environ.get("CXR_IMPALA_USER")
        self.password = password or os.environ.get("CXR_IMPALA_PASSWORD")
        if not self.user or not self.password:
            raise RuntimeError("CXR_IMPALA_USER and CXR_IMPALA_PASSWORD must be set (CAI project environment or .env)")
        self._conn = None

    def _connect(self):
        if self._conn is None:
            from impala.dbapi import connect

            c = self.cfg
            self._conn = connect(host=c["host"], port=int(c["port"]), use_ssl=True, use_http_transport=True,
                                 http_path=c["http_path"], auth_mechanism=c["auth_mechanism"],
                                 user=self.user, password=self.password)
        return self._conn

    def query(self, sql: str) -> list[dict]:
        cur = self._connect().cursor()
        try:
            cur.execute(sql)
            if cur.description is None:
                return []
            cols = [d[0].split(".")[-1] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        finally:
            cur.close()

    def execute(self, sql: str) -> None:
        cur = self._connect().cursor()
        try:
            cur.execute(sql)
        finally:
            cur.close()

    def refresh(self, key: str) -> None:
        self.execute(f"REFRESH {self.t(key)}")

    def snapshot_id(self, key: str) -> int | None:
        hist = self.query(f"DESCRIBE HISTORY {self.t(key)}")
        current = [h for h in hist if str(h.get("is_current_ancestor", "true")).lower() == "true"]
        return int(current[-1]["snapshot_id"]) if current else None

    def source(self, key: str, snapshot: int | None) -> str:
        return f"{self.t(key)} FOR SYSTEM_VERSION AS OF {int(snapshot)}" if snapshot else self.t(key)

    @staticmethod
    def create_sql(table: str, cols, part) -> str:
        body = ", ".join(f"`{c}` {t}" for c, t in cols)
        spec = f" PARTITIONED BY SPEC ({part})" if part else ""
        return (f"CREATE TABLE IF NOT EXISTS {table} ({body}){spec} STORED AS ICEBERG "
                f"TBLPROPERTIES ('format-version'='2')")

    def append(self, key: str, rows: list[dict]) -> None:
        if not rows:
            return
        self.ensure(key)
        cols, _ = TABLES[key]
        chunk = int(self.cfg.get("insert_chunk", 500))
        names = ", ".join(f"`{c}`" for c, _ in cols)
        for i in range(0, len(rows), chunk):
            values = ",\n".join("(" + ", ".join(_literal(r.get(c), t) for c, t in cols) + ")"
                                for r in rows[i:i + chunk])
            self.execute(f"INSERT INTO {self.t(key)} ({names}) VALUES {values}")


class SparkStore(_Store):
    engine = "spark"

    def __init__(self, spark, prefix: str | None = None):
        super().__init__(prefix)
        self.spark = spark

    def query(self, sql: str) -> list[dict]:
        return [r.asDict() for r in self.spark.sql(sql).collect()]

    def execute(self, sql: str) -> None:
        self.spark.sql(sql)

    def refresh(self, key: str) -> None:
        if self.exists(key):
            self.spark.catalog.refreshTable(self.t(key))

    def exists(self, key: str) -> bool:
        return self.spark.catalog.tableExists(self.t(key))

    def snapshot_id(self, key: str) -> int | None:
        rows = self.query(f"SELECT snapshot_id FROM {self.t(key)}.snapshots ORDER BY committed_at DESC LIMIT 1")
        return int(rows[0]["snapshot_id"]) if rows else None

    def source(self, key: str, snapshot: int | None) -> str:
        return f"{self.t(key)} VERSION AS OF {int(snapshot)}" if snapshot else self.t(key)

    @staticmethod
    def add_columns_sql(table: str, cols) -> str:
        body = ", ".join(f"`{c}` {'TIMESTAMP_NTZ' if t == 'TIMESTAMP' else t}" for c, t in cols)
        return f"ALTER TABLE {table} ADD COLUMNS ({body})"

    @staticmethod
    def create_sql(table: str, cols, part) -> str:
        body = ", ".join(f"`{c}` {'TIMESTAMP_NTZ' if t == 'TIMESTAMP' else t}" for c, t in cols)
        spec = f" PARTITIONED BY ({part})" if part else ""
        return (f"CREATE TABLE IF NOT EXISTS {table} ({body}) USING iceberg{spec} "
                f"TBLPROPERTIES ('format-version'='2')")

    def append(self, key: str, rows: list[dict]) -> None:
        if not rows:
            return
        self.ensure(key)
        cols, _ = TABLES[key]
        schema = ", ".join(f"`{c}` {'TIMESTAMP_NTZ' if t == 'TIMESTAMP' else t}" for c, t in cols)
        self.spark.createDataFrame([tuple(r.get(c) for c, _ in cols) for r in rows], schema) \
            .writeTo(self.t(key)).append()


def impala_from_env() -> ImpalaStore:
    return ImpalaStore()
