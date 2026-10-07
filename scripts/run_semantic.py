#!/usr/bin/env python3
"""
Semantic layer: the views in sql/semantic/*.sql, on one of two engines.

  --engine impala   CDW Impala (what Data Visualization and Hue query). Needs CXR_IMPALA_USER
                    and CXR_IMPALA_PASSWORD (.env locally, project environment on CAI).
  --engine spark    local Spark + Iceberg (scripts/run_local.py semantic, and CI)

Steps:
  views   create the CAI-written tables if missing (a view must compile), then (re)create
          every view in file order
  check   the certified daily KPI view must agree with the per-film view it summarises:
          films, P1 films, pneumonia and confusion counts per business date. Exit 1 on a
          difference.

SQL uses the default database names (rsingh_cxr_*) so it pastes into Hue as is; --db-prefix
rewrites them.

Usage:
  python scripts/run_semantic.py --engine impala
  python scripts/run_semantic.py --engine impala --steps check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lakehouse.store import TABLES, ImpalaStore, SparkStore, lake_config  # noqa: E402

SQL = ROOT / "sql" / "semantic"
DEFAULT_PREFIX = "rsingh_cxr"
CHECK_COLUMNS = ("studies", "p1", "pneumonia", "tp", "fn", "fp", "tn")


def statements(text: str) -> list[str]:
    """Split a SQL file: comment lines dropped, ';' at a line end ends a statement."""
    out, cur = [], []
    for line in text.splitlines():
        if line.strip().startswith("--"):
            continue
        cur.append(line)
        if line.rstrip().endswith(";"):
            out.append("\n".join(cur).strip().rstrip(";").strip())
            cur = []
    rest = "\n".join(cur).strip()
    return [s for s in out + ([rest] if rest else []) if s]


def render(sql: str, prefix: str) -> str:
    return sql.replace(f"{DEFAULT_PREFIX}_", f"{prefix}_") if prefix != DEFAULT_PREFIX else sql


class SparkEngine(SparkStore):
    pass


RENAMED = [("ref.training_set", "split", "data_split")]     # (table, old column, new column)


def migrate(store) -> None:
    for key, old, new in RENAMED:
        cols = {str(r.get("name") or r.get("col_name")).lower() for r in store.query(f"DESCRIBE {store.t(key)}")}
        if old in cols and new not in cols:
            typ = dict(TABLES[key][0])[new]
            store.execute(f"ALTER TABLE {store.t(key)} CHANGE COLUMN `{old}` `{new}` {typ}" if store.engine == "impala"
                          else f"ALTER TABLE {store.t(key)} RENAME COLUMN `{old}` TO `{new}`")
            print(f"  {store.t(key)}: column {old} renamed to {new}")


def create_views(store) -> int:
    for key in TABLES:
        store.ensure(key)
    migrate(store)
    store.execute(f"CREATE DATABASE IF NOT EXISTS {store.prefix}_semantic")
    n = 0
    for path in sorted(SQL.glob("*.sql")):
        for stmt in statements(path.read_text()):
            store.execute(render(stmt, store.prefix))
            if stmt.upper().startswith("CREATE VIEW"):
                n += 1
        print(f"  {path.name}: done", flush=True)
    return n


def check(store) -> list[str]:
    s = f"{store.prefix}_semantic"
    kpi = {str(r["business_date"]): r for r in store.query(
        f"SELECT business_date, {', '.join(CHECK_COLUMNS)} FROM {s}.v_daily_kpi")}
    films = {str(r["business_date"]): r for r in store.query(f"""
        SELECT business_date, COUNT(*) AS studies, SUM(is_p1) AS p1, SUM(is_pneumonia) AS pneumonia,
               SUM(CASE WHEN outcome = 'TP' THEN 1 ELSE 0 END) AS tp, SUM(CASE WHEN outcome = 'FN' THEN 1 ELSE 0 END) AS fn,
               SUM(CASE WHEN outcome = 'FP' THEN 1 ELSE 0 END) AS fp, SUM(CASE WHEN outcome = 'TN' THEN 1 ELSE 0 END) AS tn
        FROM {s}.v_triage_outcome GROUP BY business_date""")}
    bad = []
    for d in sorted(set(kpi) | set(films)):
        for c in CHECK_COLUMNS:
            a = int((kpi.get(d) or {}).get(c) or 0)
            b = int((films.get(d) or {}).get(c) or 0)
            if a != b:
                bad.append(f"{d} {c}: v_daily_kpi {a} != v_triage_outcome {b}")
    print(f"  check: {len(kpi)} business dates, {len(bad)} difference(s)")
    for b in bad:
        print(f"  MISMATCH {b}")
    return bad


def run(store, prefix: str | None = None, steps=("views", "check")) -> int:
    if "views" in steps:
        print(f"  {create_views(store)} views created on {store.engine}")
    if "check" in steps and check(store):
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["impala", "spark"], default="impala")
    p.add_argument("--steps", default="views,check")
    p.add_argument("--db-prefix", default=lake_config()["db_prefix"])
    p.add_argument("--warehouse", default=str(ROOT / "data" / "lake" / "warehouse"))
    args = p.parse_args()
    steps = tuple(args.steps.split(","))
    if args.engine == "impala":
        store = ImpalaStore(prefix=args.db_prefix)
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_local import local_spark

        store = SparkEngine(local_spark(Path(args.warehouse)), args.db_prefix)
    return run(store, args.db_prefix, steps)


if __name__ == "__main__":
    sys.exit(main())
